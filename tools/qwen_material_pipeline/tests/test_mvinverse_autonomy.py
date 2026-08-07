from __future__ import annotations

import pytest

import qwen_material_pipeline.mvinverse.autonomy as mvinverse_autonomy
from qwen_material_pipeline.mvinverse.autonomy import (
    GENERIC_STEEL_PAINTED,
    MVInverseAutonomyError,
    build_part_view_evidence,
    parameterize_auto_material_plan,
)


PAINT_VARIANT = "mdl:vMaterials_2/Metal/Steel_Painted.mdl#Steel_Painted_Army_Green"
RUBBER = "mdl:vMaterials_2/Other/Rubber/Caoutchouc.mdl#Rubber_Black_Matte"


@pytest.fixture(autouse=True)
def _isolate_parameterization_from_evidence_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fusion module owns its full schema; these tests exercise the join."""

    monkeypatch.setattr(
        mvinverse_autonomy,
        "validate_mvinverse_evidence",
        lambda document: document,
    )


def _batches() -> list[dict]:
    return [
        {
            "mappings": [
                {"part_id": "P0001", "status": "matched", "group_id": "G01"},
                {"part_id": "P0002", "status": "matched", "group_id": "G02"},
                {"part_id": "P0003", "status": "unknown", "group_id": None},
            ]
        }
    ]


def _palette() -> dict:
    return {
        "groups": [
            {
                "group_id": "G01",
                "family_hint": "metal",
                "finish_hint": "painted",
            },
            {
                "group_id": "G02",
                "family_hint": "rubber",
                "finish_hint": "matte",
            },
        ]
    }


def _evidence(*, metallic: float = 0.1, decision: str = "auto") -> dict:
    eligible = decision == "auto"
    return {
        "schema_version": "qwen-mvinverse-pbr-evidence/v1",
        "groups": [
            {
                "group_id": "G01",
                "surface_class": "dielectric",
                "contributing_view_ids": ["ref_front", "ref_side"],
                "metallic": {"median": metallic},
                "suggestion": {
                    "decision": decision,
                    "auto_parameter_eligible": eligible,
                    "base_color_srgb": [0.25, 0.5, 1.0] if eligible else None,
                    "metallic": (
                        (0.0 if metallic <= 0.35 else metallic)
                        if eligible
                        else None
                    ),
                    "roughness": 0.42 if eligible else None,
                },
            }
        ],
    }


def test_build_part_view_evidence_expands_real_group_votes() -> None:
    report = build_part_view_evidence(
        batches=_batches(),
        group_view_choices={
            "G01": [
                {
                    "view_id": "ref_side",
                    "material_id": PAINT_VARIANT,
                    "confidence": 0.96,
                    "candidate_margin": 0.2,
                },
                {
                    "view_id": "ref_front",
                    "material_id": PAINT_VARIANT,
                    "confidence": 0.94,
                    "candidate_margin": None,
                },
            ]
        },
    )

    assert report["schema_version"] == "qwen-material-view-evidence/v1"
    assert report["predictions"] == [
        {
            "part_id": "P0001",
            "view_id": "ref_front",
            "material_id": PAINT_VARIANT,
            "confidence": 0.94,
            "candidate_margin": None,
        },
        {
            "part_id": "P0001",
            "view_id": "ref_side",
            "material_id": PAINT_VARIANT,
            "confidence": 0.96,
            "candidate_margin": 0.2,
        },
    ]


def test_build_part_view_evidence_rejects_duplicate_source_view() -> None:
    duplicate = {
        "view_id": "ref_side",
        "material_id": PAINT_VARIANT,
        "confidence": 0.95,
    }
    with pytest.raises(MVInverseAutonomyError, match="duplicate source view"):
        build_part_view_evidence(
            batches=_batches(), group_view_choices={"G01": [duplicate, duplicate]}
        )


def test_part_view_evidence_requires_same_view_part_mapping() -> None:
    choices = {
        "G01": [
            {
                "view_id": "ref_front",
                "material_id": PAINT_VARIANT,
                "confidence": 0.96,
                "candidate_margin": 0.2,
            },
            {
                "view_id": "ref_side",
                "material_id": PAINT_VARIANT,
                "confidence": 0.94,
                "candidate_margin": 0.2,
            },
        ]
    }
    report = build_part_view_evidence(
        batches=_batches(),
        group_view_choices=choices,
        mapping_votes=[
            {
                "view_id": "ref_front",
                "part_id": "P0001",
                "canonical_group_id": "G02",
                "status": "matched",
                "confidence": 0.99,
            },
            {
                "view_id": "ref_side",
                "part_id": "P0001",
                "canonical_group_id": "G01",
                "status": "matched",
                "confidence": 0.91,
            },
        ],
    )

    assert report["predictions"] == [
        {
            "part_id": "P0001",
            "view_id": "ref_side",
            "material_id": PAINT_VARIANT,
            "confidence": 0.91,
            "candidate_margin": 0.2,
        }
    ]


def test_part_view_evidence_rejects_duplicate_part_view_votes() -> None:
    vote = {
        "view_id": "ref_front",
        "part_id": "P0001",
        "canonical_group_id": "G01",
        "status": "matched",
        "confidence": 0.95,
    }
    with pytest.raises(MVInverseAutonomyError, match="duplicate part/view"):
        build_part_view_evidence(
            batches=_batches(),
            group_view_choices={
                "G01": [
                    {
                        "view_id": "ref_front",
                        "material_id": PAINT_VARIANT,
                        "confidence": 0.96,
                    }
                ]
            },
            mapping_votes=[vote, vote],
        )


def test_parameterize_auto_painted_steel_uses_linear_color_and_roughness() -> None:
    auto_plan = {
        "schema_version": "1.0",
        "assignments": [
            {
                "part_id": "P0001",
                "material_id": PAINT_VARIANT,
                "confidence": 0.96,
                "status": "auto",
                "evidence_views": ["ref_side"],
            },
            {
                "part_id": "P0002",
                "material_id": RUBBER,
                "confidence": 0.97,
                "status": "auto",
                "evidence_views": ["ref_front", "ref_side"],
            },
        ],
    }
    report = parameterize_auto_material_plan(
        auto_material_plan=auto_plan,
        batches=_batches(),
        palette=_palette(),
        mvinverse_evidence=_evidence(),
        allowed_material_ids={GENERIC_STEEL_PAINTED, PAINT_VARIANT, RUBBER},
    )

    first, second = report["material_plan"]["assignments"]
    assert first["material_id"] == PAINT_VARIANT
    assert first["parameters"]["paint_color"] == pytest.approx(
        [0.05087609, 0.21404114, 1.0]
    )
    assert first["parameters"]["paint_roughness"] == 0.42
    assert first["evidence_views"] == ["ref_front", "ref_side"]
    assert second["material_id"] == RUBBER
    assert "parameters" not in second
    assert report["summary"] == {
        "auto_assignment_count": 2,
        "parameterized_assignment_count": 1,
        "unchanged_auto_assignment_count": 1,
    }


def test_parameterization_uses_explicit_verified_group_override() -> None:
    auto_plan = {
        "schema_version": "1.0",
        "assignments": [
            {
                "part_id": "P0002",
                "material_id": PAINT_VARIANT,
                "confidence": 0.75,
                "status": "auto",
                "evidence_views": ["ref_front", "ref_side"],
            }
        ],
    }
    report = parameterize_auto_material_plan(
        auto_material_plan=auto_plan,
        batches=_batches(),
        palette=_palette(),
        mvinverse_evidence=_evidence(),
        allowed_material_ids={GENERIC_STEEL_PAINTED, PAINT_VARIANT},
        part_group_overrides={"P0002": "G01"},
    )

    assert report["decisions"][0]["group_id"] == "G01"
    assert report["decisions"][0]["parameterized"] is True
    assert (
        report["material_plan"]["assignments"][0]["material_id"] == PAINT_VARIANT
    )

    with pytest.raises(MVInverseAutonomyError, match="outside the auto plan"):
        parameterize_auto_material_plan(
            auto_material_plan=auto_plan,
            batches=_batches(),
            palette=_palette(),
            mvinverse_evidence=_evidence(),
            allowed_material_ids={GENERIC_STEEL_PAINTED, PAINT_VARIANT},
            part_group_overrides={"P9999": "G01"},
        )


def test_immutable_selected_mdl_mode_never_writes_parameters() -> None:
    auto_plan = {
        "schema_version": "1.0",
        "assignments": [
            {
                "part_id": "P0001",
                "material_id": PAINT_VARIANT,
                "confidence": 0.96,
                "status": "auto",
                "evidence_views": ["ref_side"],
            }
        ],
    }

    report = parameterize_auto_material_plan(
        auto_material_plan=auto_plan,
        batches=_batches(),
        palette=_palette(),
        mvinverse_evidence=_evidence(),
        allowed_material_ids={GENERIC_STEEL_PAINTED, PAINT_VARIANT},
        allow_parameter_writes=False,
    )

    assignment = report["material_plan"]["assignments"][0]
    assert assignment == auto_plan["assignments"][0]
    assert "parameters" not in assignment
    assert report["decisions"][0]["reason_code"] == (
        "selected_mdl_library_defaults_locked"
    )
    assert report["summary"]["parameterized_assignment_count"] == 0
    assert report["summary"]["parameter_writes_enabled"] is False
    assert report["summary"]["selected_mdl_library_defaults_locked"] is True


@pytest.mark.parametrize(
    ("evidence", "reason"),
    [
        (_evidence(metallic=0.8), "metallic_contradicts_dielectric_paint"),
        (_evidence(decision="preserve"), "mvinverse_parameter_gate_preserved"),
    ],
)
def test_parameterization_fails_closed_when_pbr_is_not_safe(
    evidence: dict, reason: str
) -> None:
    auto_plan = {
        "schema_version": "1.0",
        "assignments": [
            {
                "part_id": "P0001",
                "material_id": PAINT_VARIANT,
                "confidence": 0.96,
                "status": "auto",
                "evidence_views": ["ref_side"],
            }
        ],
    }
    report = parameterize_auto_material_plan(
        auto_material_plan=auto_plan,
        batches=_batches(),
        palette=_palette(),
        mvinverse_evidence=evidence,
        allowed_material_ids={GENERIC_STEEL_PAINTED, PAINT_VARIANT},
    )
    assignment = report["material_plan"]["assignments"][0]
    assert assignment["material_id"] == PAINT_VARIANT
    assert "parameters" not in assignment
    assert report["decisions"][0]["reason_code"] == reason
