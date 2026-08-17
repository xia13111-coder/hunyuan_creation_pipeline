from __future__ import annotations

import copy

import pytest

from qwen_material_pipeline.materials.corresponding_material_color import (
    CorrespondingMaterialColorError,
    build_corresponding_material_color_plan,
)
from qwen_material_pipeline.usd.material_common import canonical_sha256


PAINT = "mdl:Miscellaneous/Paint_Matte.mdl#Paint_Matte"
METAL = "mdl:Metals/Aluminum_Anodized.mdl#Aluminum_Anodized"
EXACT = "mdl:Metals/Steel_Stainless.mdl#Steel_Stainless"


def _seal(value: dict) -> dict:
    output = copy.deepcopy(value)
    output["integrity"] = {"document_sha256": canonical_sha256(output)}
    return output


def _evidence_part(part_id: str, color: list[float], samples: int) -> dict:
    return {
        "part_id": part_id,
        "status": "observed",
        "selected_observation_view_id": "front",
        "observations": [
            {
                "view_id": "front",
                "selected_for_material_inference": True,
                "camera_alignment_evidence_weight": 0.8,
            }
        ],
        "descriptor": {
            "robust_color_evidence": {
                "method": "cielab_medoid_fixed_radius",
                "sample_count": samples,
                "inlier_fraction": 0.9,
                "robust_reference_srgb": color,
            }
        },
    }


def _documents() -> tuple[dict, dict, dict]:
    plan = {
        "schema_version": "1.0",
        "assignment_unit": "part_id",
        "assignments": [
            {
                "part_id": "P1",
                "material_id": EXACT,
                "status": "auto",
                "provenance": {},
            },
            {
                "part_id": "P2",
                "material_id": PAINT,
                "status": "review",
                "provenance": {},
            },
            {
                "part_id": "P3",
                "material_id": METAL,
                "status": "auto",
                "provenance": {},
            },
            {
                "part_id": "P4",
                "material_id": METAL,
                "status": "auto",
                "provenance": {},
            },
            {
                "part_id": "P5",
                "material_id": EXACT,
                "status": "policy_fallback",
                "provenance": {},
            },
        ],
        "provenance": {"mode": "fixture"},
    }
    choices = _seal(
        {
            "schema_version": "qwen-part-id-material-rerank/v1",
            "assignment_unit": "part_id",
            "selections": [
                {
                    "part_id": "P1",
                    "material_id": EXACT,
                    "match_type": "EXACT_LIBRARY_MATCH",
                },
                {
                    "part_id": "P2",
                    "material_id": PAINT,
                    "match_type": "CORRESPONDING_MATERIAL",
                },
                {
                    "part_id": "P3",
                    "material_id": METAL,
                    "match_type": "CORRESPONDING_MATERIAL",
                },
                {
                    "part_id": "P4",
                    "material_id": METAL,
                    "match_type": "CORRESPONDING_MATERIAL",
                },
            ],
            "component_identity_consensus": {
                "components": [
                    {
                        "component_id": "C_GREEN",
                        "member_part_ids": ["P3", "P4"],
                        "selected_material_id": METAL,
                        "match_type": "CORRESPONDING_MATERIAL",
                    }
                ]
            },
        }
    )
    evidence = _seal(
        {
            "schema_version": "qwen-part-id-reference-evidence/v1",
            "assignment_unit": "part_id",
            "parts": [
                _evidence_part("P1", [0.7, 0.7, 0.7], 100),
                _evidence_part("P2", [0.7, 0.1, 0.1], 25),
                _evidence_part("P3", [0.1, 0.5, 0.2], 100),
                _evidence_part("P4", [0.12, 0.52, 0.22], 400),
            ],
        }
    )
    return plan, choices, evidence


def _by_id(plan: dict) -> dict[str, dict]:
    return {row["part_id"]: row for row in plan["assignments"]}


def test_colors_only_corresponding_and_preserves_material_ids() -> None:
    source, choices, evidence = _documents()
    output, audit = build_corresponding_material_color_plan(
        source_plan=source,
        qwen_choices=choices,
        part_id_evidence=evidence,
    )
    before = _by_id(source)
    after = _by_id(output)
    assert {part_id: row["material_id"] for part_id, row in after.items()} == {
        part_id: row["material_id"] for part_id, row in before.items()
    }
    assert "parameters" not in after["P1"]
    assert "parameters" not in after["P5"]
    assert after["P2"]["parameters"]
    assert after["P3"]["parameters"] == after["P4"]["parameters"]
    assert audit["status"] == "PASS"
    assert audit["summary"] == {
        "selection_count": 4,
        "exact_library_match_count": 1,
        "corresponding_material_count": 3,
        "parameterized_part_count": 3,
        "colour_scope_count": 2,
        "shared_component_scope_count": 1,
        "independent_scope_count": 1,
        "material_identity_change_count": 0,
    }


def test_group_uses_one_deterministic_photo_medoid() -> None:
    source, choices, evidence = _documents()
    _, audit = build_corresponding_material_color_plan(
        source_plan=source,
        qwen_choices=choices,
        part_id_evidence=evidence,
    )
    scope = next(
        row for row in audit["scopes"] if row["scope_id"] == "COMPONENT:C_GREEN"
    )
    assert scope["member_part_ids"] == ["P3", "P4"]
    assert scope["target_srgb"] in ([0.1, 0.5, 0.2], [0.12, 0.52, 0.22])
    assert scope["medoid_part_id"] in {"P3", "P4"}
    assert max(scope["parameters"]["diffuse_tint"]) < 1.0
    assert scope["color_parameter_audit"]["linear_intensity_gain"] == 1.0
    assert scope["color_parameter_audit"]["color_parameter_semantics"] == (
        "render_calibrated_absolute_linear_color_gain"
    )


def test_linear_intensity_gain_is_bounded_and_scales_without_changing_hue() -> None:
    source, choices, evidence = _documents()
    _, gain_one = build_corresponding_material_color_plan(
        source_plan=source,
        qwen_choices=choices,
        part_id_evidence=evidence,
        linear_intensity_gain=1.0,
    )
    _, gain_two = build_corresponding_material_color_plan(
        source_plan=source,
        qwen_choices=choices,
        part_id_evidence=evidence,
        linear_intensity_gain=2.0,
    )
    first = next(
        row for row in gain_one["scopes"] if row["scope_id"] == "COMPONENT:C_GREEN"
    )["parameters"]["diffuse_tint"]
    second = next(
        row for row in gain_two["scopes"] if row["scope_id"] == "COMPONENT:C_GREEN"
    )["parameters"]["diffuse_tint"]
    assert second == pytest.approx([2.0 * value for value in first])

    with pytest.raises(CorrespondingMaterialColorError, match="linear_intensity_gain"):
        build_corresponding_material_color_plan(
            source_plan=source,
            qwen_choices=choices,
            part_id_evidence=evidence,
            linear_intensity_gain=0.0,
        )


def test_per_scope_gains_are_exact_bounded_and_independent() -> None:
    source, choices, evidence = _documents()
    output, audit = build_corresponding_material_color_plan(
        source_plan=source,
        qwen_choices=choices,
        part_id_evidence=evidence,
        linear_intensity_gains_by_scope={
            "PART:P2": 0.5,
            "COMPONENT:C_GREEN": 2.0,
        },
    )
    scopes = {row["scope_id"]: row for row in audit["scopes"]}
    assert audit["policy"]["gain_mode"] == "per_scope"
    assert audit["policy"]["linear_intensity_gains_by_scope"] == {
        "COMPONENT:C_GREEN": 2.0,
        "PART:P2": 0.5,
    }
    assert scopes["PART:P2"]["color_parameter_audit"]["linear_intensity_gain"] == 0.5
    assert (
        scopes["COMPONENT:C_GREEN"]["color_parameter_audit"]["linear_intensity_gain"]
        == 2.0
    )
    assignments = _by_id(output)
    assert assignments["P3"]["parameters"] == assignments["P4"]["parameters"]

    with pytest.raises(CorrespondingMaterialColorError, match="exactly cover"):
        build_corresponding_material_color_plan(
            source_plan=source,
            qwen_choices=choices,
            part_id_evidence=evidence,
            linear_intensity_gains_by_scope={"PART:P2": 1.0},
        )
    with pytest.raises(CorrespondingMaterialColorError, match="linear_intensity_gain"):
        build_corresponding_material_color_plan(
            source_plan=source,
            qwen_choices=choices,
            part_id_evidence=evidence,
            linear_intensity_gains_by_scope={
                "PART:P2": 1.0,
                "COMPONENT:C_GREEN": 9.0,
            },
        )


def test_rejects_existing_parameters_or_material_drift() -> None:
    source, choices, evidence = _documents()
    source["assignments"][1]["parameters"] = {"diffuse_tint": [1.0, 0.0, 0.0]}
    with pytest.raises(
        CorrespondingMaterialColorError, match="already contains parameters"
    ):
        build_corresponding_material_color_plan(
            source_plan=source,
            qwen_choices=choices,
            part_id_evidence=evidence,
        )

    source, choices, evidence = _documents()
    choices["selections"][1]["material_id"] = METAL
    choices = _seal(
        {key: value for key, value in choices.items() if key != "integrity"}
    )
    with pytest.raises(CorrespondingMaterialColorError, match="material mismatch"):
        build_corresponding_material_color_plan(
            source_plan=source,
            qwen_choices=choices,
            part_id_evidence=evidence,
        )


def test_rejects_exact_and_corresponding_members_mixed_in_one_component() -> None:
    source, choices, evidence = _documents()
    component = choices["component_identity_consensus"]["components"][0]
    component["member_part_ids"] = ["P1", "P3"]
    component["selected_material_id"] = EXACT
    choices = _seal(
        {key: value for key, value in choices.items() if key != "integrity"}
    )
    with pytest.raises(
        CorrespondingMaterialColorError, match="mixes library match types"
    ):
        build_corresponding_material_color_plan(
            source_plan=source,
            qwen_choices=choices,
            part_id_evidence=evidence,
        )


def test_repeated_role_exact_identity_is_still_color_tuned_as_one_scope() -> None:
    source, choices, evidence = _documents()
    for part_id in ("P3", "P4"):
        next(row for row in choices["selections"] if row["part_id"] == part_id)[
            "match_type"
        ] = "EXACT_LIBRARY_MATCH"
    component = choices["component_identity_consensus"]["components"][0]
    component["match_type"] = "EXACT_LIBRARY_MATCH"
    component["consensus_mode"] = "REPEATED_ROLE_JOINT_CONSENSUS"
    choices = _seal(
        {key: value for key, value in choices.items() if key != "integrity"}
    )

    output, audit = build_corresponding_material_color_plan(
        source_plan=source,
        qwen_choices=choices,
        part_id_evidence=evidence,
    )

    assignments = _by_id(output)
    assert assignments["P3"]["material_id"] == METAL
    assert assignments["P4"]["material_id"] == METAL
    assert assignments["P3"]["parameters"] == assignments["P4"]["parameters"]
    assert audit["summary"]["exact_library_match_count"] == 1
    assert audit["summary"]["corresponding_material_count"] == 3
    assert any(scope["scope_id"] == "COMPONENT:C_GREEN" for scope in audit["scopes"])


def test_protected_exact_component_keeps_authored_preset_immutable() -> None:
    source, choices, evidence = _documents()
    for part_id in ("P3", "P4"):
        next(row for row in choices["selections"] if row["part_id"] == part_id)[
            "match_type"
        ] = "EXACT_LIBRARY_MATCH"
    component = choices["component_identity_consensus"]["components"][0]
    component["match_type"] = "EXACT_LIBRARY_MATCH"
    component["consensus_mode"] = "PROTECTED_EXACT_PRESET_PROPAGATED"
    choices = _seal(
        {key: value for key, value in choices.items() if key != "integrity"}
    )

    output, audit = build_corresponding_material_color_plan(
        source_plan=source,
        qwen_choices=choices,
        part_id_evidence=evidence,
    )

    assignments = _by_id(output)
    assert "parameters" not in assignments["P3"]
    assert "parameters" not in assignments["P4"]
    assert audit["summary"]["exact_library_match_count"] == 3
    assert audit["summary"]["corresponding_material_count"] == 1


def test_rejects_component_member_outside_sealed_plans() -> None:
    source, choices, evidence = _documents()
    choices["component_identity_consensus"]["components"][0]["member_part_ids"].append(
        "P999"
    )
    choices = _seal(
        {key: value for key, value in choices.items() if key != "integrity"}
    )
    with pytest.raises(
        CorrespondingMaterialColorError, match="outside the sealed plans"
    ):
        build_corresponding_material_color_plan(
            source_plan=source,
            qwen_choices=choices,
            part_id_evidence=evidence,
        )


def test_rejects_unsealed_or_unsupported_color_evidence() -> None:
    source, choices, evidence = _documents()
    evidence["parts"][1]["descriptor"]["robust_color_evidence"]["method"] = "mean_rgb"
    evidence = _seal(
        {key: value for key, value in evidence.items() if key != "integrity"}
    )
    with pytest.raises(CorrespondingMaterialColorError, match="sealed robust"):
        build_corresponding_material_color_plan(
            source_plan=source,
            qwen_choices=choices,
            part_id_evidence=evidence,
        )
