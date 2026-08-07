from __future__ import annotations

import copy

import pytest

import qwen_material_pipeline.materials.complete_plan as complete_plan_module
from qwen_material_pipeline.materials.complete_plan import (
    GENERIC_STEEL_PAINTED,
    CompleteMaterialPlanError,
    build_complete_material_plan,
)
from qwen_material_pipeline.mvinverse.evidence import (
    validate_mvinverse_evidence as strict_validate_mvinverse_evidence,
)


@pytest.fixture(autouse=True)
def _isolate_completion_logic_from_full_evidence_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Most tests below exercise completion after upstream evidence validation."""

    monkeypatch.setattr(
        complete_plan_module,
        "validate_mvinverse_evidence",
        lambda document: copy.deepcopy(document),
    )


def _documents() -> tuple[dict, dict, dict, dict]:
    base = {
        "schema_version": "1.0",
        "assignments": [
            {
                "part_id": "P0001",
                "material_id": GENERIC_STEEL_PAINTED,
                "semantic": "green painted frame",
                "confidence": 1.0,
                "evidence_views": [],
                "status": "approved",
                "parameters": {"paint_roughness": 0.3},
            }
        ],
    }
    registry = {
        "asset_sha256": "a" * 64,
        "parts": [{"part_id": "P0001"}, {"part_id": "P0002"}],
    }
    evidence = {
        "schema_version": "qwen-mvinverse-pbr-evidence/v1",
        "groups": [
            {
                "group_id": "G01",
                "surface_class": "dielectric",
                "contributing_view_ids": ["front", "iso"],
                "metallic": {"median": 0.12},
                "suggestion": {
                    "decision": "auto",
                    "auto_parameter_eligible": True,
                    "base_color_srgb": [0.25, 0.5, 0.125],
                    "metallic": 0.0,
                    "roughness": 0.42,
                },
            }
        ],
    }
    policy = {
        "schema_version": "qwen-complete-material-policy/v1",
        "expected_asset_sha256": "a" * 64,
        "parameterized_groups": [
            {
                "group_id": "G01",
                "material_id": GENERIC_STEEL_PAINTED,
                "part_ids": ["P0001"],
            }
        ],
        "fallback_assignments": [
            {
                "part_id": "P0002",
                "material_id": "mdl:Base/Metals/Steel_Stainless.mdl#Steel_Stainless",
                "semantic": "occluded structural washer",
                "confidence": 1.0,
                "evidence_views": [],
                "status": "approved",
            }
        ],
    }
    return base, registry, evidence, policy


def test_complete_plan_parameterizes_and_exactly_covers_registry() -> None:
    base, registry, evidence, policy = _documents()
    original = copy.deepcopy(base)

    plan, report = build_complete_material_plan(
        base_plan=base,
        registry=registry,
        mvinverse_evidence=evidence,
        policy=policy,
    )

    assert base == original
    assert [item["part_id"] for item in plan["assignments"]] == ["P0001", "P0002"]
    painted = plan["assignments"][0]
    assert painted["parameters"]["paint_roughness"] == pytest.approx(0.42)
    assert painted["parameters"]["paint_roughness_variation"] == 0.0
    assert painted["evidence_views"] == ["front", "iso"]
    assert report["summary"] == {
        "registry_part_count": 2,
        "base_assignment_count": 1,
        "parameterized_part_count": 1,
        "fallback_assignment_count": 1,
        "output_assignment_count": 2,
        "face_subset_count": 0,
        "all_registry_parts_assigned": True,
    }
    assert report["parameterized_groups"][0]["observed_metallic"] == pytest.approx(0.12)
    assert report["parameterized_groups"][0]["authored_metallic"] == 0.0


def test_complete_plan_rejects_incomplete_coverage() -> None:
    base, registry, evidence, policy = _documents()
    policy["fallback_assignments"] = []

    with pytest.raises(CompleteMaterialPlanError, match="does not exactly cover"):
        build_complete_material_plan(
            base_plan=base,
            registry=registry,
            mvinverse_evidence=evidence,
            policy=policy,
        )


def test_complete_plan_rejects_noneligible_mvinverse_group() -> None:
    base, registry, evidence, policy = _documents()
    evidence["groups"][0]["suggestion"]["decision"] = "preserve"

    with pytest.raises(CompleteMaterialPlanError, match="not eligible"):
        build_complete_material_plan(
            base_plan=base,
            registry=registry,
            mvinverse_evidence=evidence,
            policy=policy,
        )


def test_complete_plan_fails_closed_on_forged_eligible_suggestion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base, registry, evidence, policy = _documents()
    monkeypatch.setattr(
        complete_plan_module,
        "validate_mvinverse_evidence",
        strict_validate_mvinverse_evidence,
    )

    plan, report = build_complete_material_plan(
        base_plan=base,
        registry=registry,
        mvinverse_evidence=evidence,
        policy=policy,
    )

    painted = plan["assignments"][0]
    assert painted["parameters"] == {"paint_roughness": 0.3}
    assert report["summary"]["parameterized_part_count"] == 0
    assert report["mvinverse_validation"]["state"] == "rejected_fail_closed"
    assert report["skipped_parameterized_groups"] == [
        {
            "group_id": "G01",
            "material_id": GENERIC_STEEL_PAINTED,
            "reason_code": "MVINVERSE_EVIDENCE_STRICT_VALIDATION_FAILED",
        }
    ]
