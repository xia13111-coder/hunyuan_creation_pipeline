from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from qwen_material_pipeline.materials import policy_exact_cover as policy_module
from qwen_material_pipeline.materials.complete_plan import GENERIC_STEEL_PAINTED
from qwen_material_pipeline.materials.quality_repair import (
    AUTHORITATIVE_CANONICAL_GROUP_LOCK_REASON,
    DARK_FOREGROUND_DIAGNOSTIC_THRESHOLDS,
    DARK_FOREGROUND_RESIDUAL_LANE,
    MULTIVIEW_DARK_IDENTITY_LANE,
    REPEATED_GEOMETRY_DARK_RESIDUAL_LANE,
    REPAIR_MODE,
    REPAIR_PROVENANCE_FIELD,
    QualityRepairError,
    _diagnostic_projection,
    _diagnostic_rejects_semantic_group,
    _bounded_signature_sibling_cohort_expansions,
    _dominant_assembly_cohort_expansions,
    _dominant_assembly_member_veto,
    _dominant_residual_spatial_support_views,
    _qa_confirmed_multiview_semantic_review_support_views,
    build_quality_repair_plan,
    main,
)
from qwen_material_pipeline.usd.material_common import (
    POLICY_EXACT_COVER_MODE,
    POLICY_FALLBACK_CONFIDENCE_BASIS,
    SOURCE_VISUAL_PRESERVE_ACTION,
    SOURCE_VISUAL_PRESERVE_TIER,
    source_visual_binding_sha256,
    validate_policy_fallback_authorization,
)


GREEN = "mdl:vMaterials_2/Metal/Steel_Painted.mdl#Steel_Painted_Army_Green"
STAINLESS = "mdl:vMaterials_2/Metal/Stainless_Steel.mdl#Stainless_Steel_Matte"
GALVANIZED = "mdl:vMaterials_2/Metal/Steel_Galvanized.mdl#Steel_Galvanized"
BLACK = "mdl:vMaterials_2/Metal/Steel_Painted.mdl#Steel_Painted_Black"


def test_three_independent_review_votes_can_localize_only_a_qa_missing_group() -> None:
    reference_evidence = {
        view_id: {
            "raw_sha256": character * 64,
            "normalized_pixel_sha256": character.upper() * 64,
            "content_cluster_id": f"content_{view_id}",
            "pose_cluster_id": f"pose_{view_id}",
        }
        for view_id, character in zip(("front", "side", "iso"), "abc")
    }
    votes = [
        {
            "view_id": view_id,
            "reference_sha256": evidence["raw_sha256"],
            "normalized_pixel_sha256": evidence["normalized_pixel_sha256"],
            "content_cluster_id": evidence["content_cluster_id"],
            "pose_cluster_id": evidence["pose_cluster_id"],
            "alignment_trusted": True,
            "unique_canonical_join": True,
            "pixel_gate_accepted": True,
            "status": "review",
            "canonical_group_id": "G07",
            "effective_confidence": 0.6,
        }
        for view_id, evidence in reference_evidence.items()
    ]
    deficit = {
        "repairable": True,
        "supporting_views": [
            {"reference_view_id": view_id} for view_id in ("front", "side", "iso")
        ],
    }
    spatial_part = {"observations": [], "semantic_votes": votes}

    supports, reasons = _qa_confirmed_multiview_semantic_review_support_views(
        part_id="P0161",
        group_id="G07",
        spatial_part=spatial_part,
        spatial_gate_decision=None,
        mapping_decision={
            "output_status": "review",
            "output_group_id": "G07",
        },
        reference_evidence=reference_evidence,
        spatial_policy={"minimum_semantic_conflict_confidence": 0.6},
        deficit=deficit,
    )

    assert supports == ["front", "iso", "side"]
    assert reasons == []

    conflicting = copy.deepcopy(spatial_part)
    conflicting["semantic_votes"][1]["canonical_group_id"] = "G04"
    supports, reasons = _qa_confirmed_multiview_semantic_review_support_views(
        part_id="P0161",
        group_id="G07",
        spatial_part=conflicting,
        spatial_gate_decision=None,
        mapping_decision={
            "output_status": "review",
            "output_group_id": "G07",
        },
        reference_evidence=reference_evidence,
        spatial_policy={"minimum_semantic_conflict_confidence": 0.6},
        deficit=deficit,
    )
    assert supports == ["front", "iso"]
    assert "SEMANTIC_REVIEW_TRUSTED_GROUP_CONFLICT" in reasons


@pytest.mark.parametrize(
    ("shares", "expected"),
    [
        ([0.04, 0.03, 0.02, 0.04, 0.10, 0.01], True),
        ([0.06, 0.03, 0.02, 0.04, 0.01, 0.01], False),
        ([0.04, 0.03, 0.06, 0.04, 0.10, 0.01], False),
        ([0.04, 0.03, 0.02, 0.04, 0.151, 0.01], False),
    ],
)
def test_canonical_diagnostic_semantic_rejection_is_robust_but_bounded(
    shares: list[float],
    expected: bool,
) -> None:
    samples = [
        {
            "group_scores": [
                {"canonical_group_id": "target", "color_share": 1.0 - share},
                {"canonical_group_id": "rejected", "color_share": share},
            ]
        }
        for share in shares
    ]

    assert (
        _diagnostic_rejects_semantic_group(
            samples=samples,
            rejected_group_id="rejected",
        )
        is expected
    )


def test_authenticated_isolated_evidence_uses_source_pixel_diagnostic_floor() -> None:
    policy = {
        "minimum_visible_pixels": 256,
        "minimum_diagnostic_visible_pixels": 128,
        "minimum_isolated_source_visible_pixels": 12,
        "minimum_isolated_source_view_count": 2,
        "minimum_diagnostic_resolved_samples": 3,
        "minimum_diagnostic_consensus_ratio": 0.75,
    }
    observation = {
        "evidence_mode": "isolated_mask_multiview_diagnostic",
        "isolated_evidence_sha256": "a" * 64,
        "isolated_source_view_count": 2,
        "declared_visible_pixels": 30,
        "projected_part_pixels": 28,
        "perturbation_label_stable": None,
        "small_part_diagnostic": {
            "status": "resolved",
            "reason_codes": [],
            "canonical_group_id": "G01",
            "bbox_canonical_group_id": "G01",
            "registration_label_stable": True,
            "resolved_sample_count": 6,
            "target_sample_count": 6,
            "consensus_ratio": 1.0,
            "alternative_canonical_group_ids": [],
        },
    }

    assert _diagnostic_projection(
        observation=observation,
        group_id="G01",
        policy=policy,
    ) == (True, False)

    tampered = copy.deepcopy(observation)
    tampered["isolated_evidence_sha256"] = None
    assert _diagnostic_projection(
        observation=tampered,
        group_id="G01",
        policy=policy,
    ) == (False, False)


def _slender_direct_box_observation() -> dict:
    return {
        "evidence_mode": "source_projection",
        "classification": "insufficient_visibility",
        "reason_code": "part_visible_pixels_below_floor",
        "canonical_group_id": None,
        "bbox_canonical_group_id": None,
        "registration_label_stable": None,
        "perturbation_label_stable": True,
        "declared_visible_pixels": 130,
        "projected_part_pixels": 200,
        "sampled_reference_pixels": 200,
        "group_scores": [
            {
                "local_group_id": "G04",
                "canonical_group_id": "G08",
                "matching_pixels": 140,
                "color_share": 0.7,
                "evidence_scope": "view_local_palette",
            },
            {
                "local_group_id": "G01",
                "canonical_group_id": "G01",
                "matching_pixels": 0,
                "color_share": 0.0,
                "evidence_scope": "view_local_palette",
            },
        ],
        "color_margin": 0.7,
        "projection_perturbations": [
            {
                "offset_pixels": list(offset),
                "sampled_reference_pixels": 200,
                "canonical_group_id": "G08",
                "diagnostic_canonical_group_id": "G08",
                "best_color_share": 0.68,
                "color_margin": 0.65,
            }
            for offset in ((-2, 0), (2, 0), (0, -2), (0, 2))
        ],
        "small_part_diagnostic": {
            "status": "rejected",
            "reason_codes": ["DIAGNOSTIC_BBOX_SAMPLE_DISAGREES"],
            "local_group_id": "G04",
            "canonical_group_id": "G08",
            "bbox_canonical_group_id": None,
            "registration_label_stable": None,
            "resolved_sample_count": 5,
            "target_sample_count": 5,
            "consensus_ratio": 1.0,
            "alternative_canonical_group_ids": [],
        },
        "canonical_palette_diagnostic": None,
        "accepted_evidence_box_overlaps": [
            {
                "local_group_id": "G04",
                "canonical_group_id": "G08",
                "evidence_pixel_count": 500,
                "projected_overlap_pixels": 150,
                "projected_overlap_share": 0.3,
                "evidence_audit_sha256": "a" * 64,
            }
        ],
    }


def test_slender_direct_box_projection_accepts_only_the_bbox_exception() -> None:
    policy = {
        "minimum_visible_pixels": 256,
        "minimum_diagnostic_visible_pixels": 128,
        "minimum_diagnostic_resolved_samples": 3,
        "minimum_diagnostic_consensus_ratio": 0.75,
    }

    assert _diagnostic_projection(
        observation=_slender_direct_box_observation(),
        group_id="G08",
        policy=policy,
    ) == (True, False)
    assert _diagnostic_projection(
        observation=_slender_direct_box_observation(),
        group_id="G07",
        policy=policy,
    ) == (False, False)


def test_slender_direct_box_projection_is_fail_closed_on_tampering() -> None:
    policy = {
        "minimum_visible_pixels": 256,
        "minimum_diagnostic_visible_pixels": 128,
        "minimum_diagnostic_resolved_samples": 3,
        "minimum_diagnostic_consensus_ratio": 0.75,
    }
    cases: dict[str, dict] = {}

    cases["wrong evidence mode"] = _slender_direct_box_observation()
    cases["wrong evidence mode"]["evidence_mode"] = "isolated_mask_multiview_diagnostic"

    cases["extra diagnostic reason"] = _slender_direct_box_observation()
    cases["extra diagnostic reason"]["small_part_diagnostic"]["reason_codes"].append(
        "DIAGNOSTIC_PERTURBATION_CONSENSUS_BELOW_FLOOR"
    )

    cases["forged direct share"] = _slender_direct_box_observation()
    cases["forged direct share"]["group_scores"][0]["matching_pixels"] = 139

    cases["forged direct margin"] = _slender_direct_box_observation()
    cases["forged direct margin"]["color_margin"] = 0.69

    cases["short perturbation sample"] = _slender_direct_box_observation()
    cases["short perturbation sample"]["projection_perturbations"][0][
        "sampled_reference_pixels"
    ] = 199

    cases["duplicate perturbation offset"] = _slender_direct_box_observation()
    cases["duplicate perturbation offset"]["projection_perturbations"][0][
        "offset_pixels"
    ] = [2, 0]

    cases["overlap below diagnostic floor"] = _slender_direct_box_observation()
    cases["overlap below diagnostic floor"]["accepted_evidence_box_overlaps"][0].update(
        {
            "projected_overlap_pixels": 127,
            "projected_overlap_share": 0.254,
        }
    )

    cases["invalid overlap digest"] = _slender_direct_box_observation()
    cases["invalid overlap digest"]["accepted_evidence_box_overlaps"][0][
        "evidence_audit_sha256"
    ] = "not-a-digest"

    cases["mismatched local group"] = _slender_direct_box_observation()
    cases["mismatched local group"]["accepted_evidence_box_overlaps"][0][
        "local_group_id"
    ] = "G05"

    cases["resolved canonical alternative"] = _slender_direct_box_observation()
    cases["resolved canonical alternative"]["canonical_palette_diagnostic"] = {
        "status": "resolved",
        "reason_codes": [],
        "canonical_group_id": "G07",
    }

    for label, observation in cases.items():
        assert _diagnostic_projection(
            observation=observation,
            group_id="G08",
            policy=policy,
        ) == (False, False), label


def _sha256_document(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _seal_spatial(document: dict) -> dict:
    result = copy.deepcopy(document)
    result.pop("integrity", None)
    result["integrity"] = {"report_sha256": _sha256_document(result)}
    return result


def _dominant_anchor_observation(view_id: str, group_id: str = "G06") -> dict:
    return {
        "reference_view_id": view_id,
        "classification": "resolved",
        "canonical_group_id": group_id,
        "registration_label_stable": True,
        "perturbation_label_stable": True,
        "projected_part_pixels": 2048,
        "group_scores": [
            {
                "canonical_group_id": group_id,
                "color_share": 0.95,
            }
        ],
        "color_margin": 0.9,
        "bbox_canonical_group_id": group_id,
        "bbox_group_scores": [
            {
                "canonical_group_id": group_id,
                "color_share": 0.8,
            }
        ],
        "bbox_color_margin": 0.7,
        "projection_perturbations": [
            {
                "offset_pixels": list(offset),
                "sampled_reference_pixels": 512,
                "canonical_group_id": group_id,
                "diagnostic_canonical_group_id": group_id,
                "best_color_share": 0.9,
                "color_margin": 0.8,
            }
            for offset in ((-2, 0), (2, 0), (0, -2), (0, 2))
        ],
    }


def test_dominant_assembly_cohort_uses_structure_not_qwen_part_votes() -> None:
    reference_evidence = {
        view_id: {
            "raw_sha256": character * 64,
            "normalized_pixel_sha256": character.upper() * 64,
            "content_cluster_id": f"content_{view_id}",
            "pose_cluster_id": f"pose_{view_id}",
        }
        for view_id, character in zip(("front", "iso", "side"), "abc")
    }
    alignment_audits = {
        view_id: {
            "score": 0.9,
            "projection_iou": 0.9,
            "ecc_correlation": 0.9,
            "ecc_status": "success",
        }
        for view_id in reference_evidence
    }
    registry_by_part: dict[str, dict] = {}
    spatial_parts: dict[str, dict] = {}
    baseline_by_part: dict[str, dict] = {}
    anchor_views = {
        "P0001": ("front", "side"),
        "P0002": ("front", "iso"),
        "P0003": ("iso", "side"),
    }
    for index in range(1, 7):
        part_id = f"P{index:04d}"
        branch = ((index - 1) % 3) + 1
        registry_by_part[part_id] = {
            "part_id": part_id,
            "parent_path": f"/Asset/Assembly/Branch{branch}/Part{index}",
            "face_count": 100,
            "existing_visual_material_properties": {
                "shader_path": f"/Looks/{part_id}",
                "shader_id": "UsdPreviewSurface",
                "diffuseColor": [0.6, 0.8, 1.0],
                "metallic": 0.5,
                "roughness": 0.5,
            },
        }
        spatial_parts[part_id] = {
            "observations": [
                _dominant_anchor_observation(view_id)
                for view_id in anchor_views.get(part_id, ())
            ],
            # Deliberately wrong, weak review votes cannot create or veto a
            # member in this hierarchy/source-signature lane.
            "semantic_votes": [
                {
                    "view_id": "front",
                    "status": "review",
                    "canonical_group_id": "G02",
                    "effective_confidence": 0.6,
                }
            ],
            "multiview_dark_consensus": None,
        }
        baseline_by_part[part_id] = {
            "part_id": part_id,
            "material_id": STAINLESS,
            "status": "policy_fallback",
            "provenance": {"tier": "neutral_default"},
        }
    for index in range(7, 31):
        part_id = f"P{index:04d}"
        registry_by_part[part_id] = {
            "part_id": part_id,
            "parent_path": f"/Asset/Other/Part{index}",
            "face_count": 100,
            "existing_visual_material_properties": {
                "shader_path": f"/Looks/{part_id}",
                "shader_id": "UsdPreviewSurface",
                "diffuseColor": [0.2, 0.2, 0.2],
                "metallic": 0.0,
                "roughness": 0.8,
            },
        }
        spatial_parts[part_id] = {
            "observations": [],
            "semantic_votes": [],
            "multiview_dark_consensus": None,
        }
        baseline_by_part[part_id] = {
            "part_id": part_id,
            "material_id": STAINLESS,
            "status": "policy_fallback",
            "provenance": {"tier": "neutral_default"},
        }

    records, cohorts, skips = _dominant_assembly_cohort_expansions(
        canonical_groups={
            "G06": {
                "group_id": "G06",
                "base_color": "green",
                "confidence": 0.95,
                "source_view_ids": ["front", "iso", "side"],
                "distinct_view_count": 3,
            }
        },
        repairable_groups={
            "G06": {
                "repairable": True,
                "supporting_views": [
                    {"reference_view_id": view_id}
                    for view_id in ("front", "iso", "side")
                ],
            }
        },
        dominant_residual_groups={
            "G06": {
                "supporting_views": [
                    {
                        "reference_view_id": "side",
                        "reference_share": 0.8,
                        "reference_share_margin": 0.7,
                        "deficit_share": 0.7,
                        "mass_recall": 0.1,
                        "requires_strict_local_projection": True,
                        "deficit_sources": ["dominant_mass_local_projection"],
                    }
                ]
            }
        },
        registry_by_part=registry_by_part,
        spatial_parts=spatial_parts,
        alignment_audits=alignment_audits,
        reference_evidence=reference_evidence,
        gate_decisions={},
        mapping_decisions={},
        geometry_risks={part_id: False for part_id in registry_by_part},
        baseline_by_part=baseline_by_part,
        occupied_proposals_by_part={},
        confirmed_materials={"G06": GREEN},
        provisional_material_groups={"G06"},
        input_hashes={
            key: character * 64
            for key, character in zip(
                (
                    "baseline_plan_sha256",
                    "quality_report_sha256",
                    "palette_fusion_sha256",
                    "spatial_report_sha256",
                    "spatial_gate_audit_sha256",
                    "mapping_consensus_sha256",
                    "geometry_risk_sha256",
                    "group_materials_sha256",
                    "registry_sha256",
                ),
                "abcdefghi",
            )
        },
    )

    assert skips == []
    assert len(cohorts) == 1
    cohort = cohorts[0]
    assert cohort["assembly_path"] == "/Asset/Assembly"
    assert cohort["anchor_part_ids"] == ["P0001", "P0002", "P0003"]
    assert cohort["anchor_supporting_view_ids"] == ["front", "iso", "side"]
    assert cohort["cohort_part_ids"] == [
        "P0001",
        "P0002",
        "P0003",
        "P0004",
        "P0005",
        "P0006",
    ]
    assert cohort["expanded_member_part_ids"] == ["P0004", "P0005", "P0006"]
    assert len(records) == 6
    assert {
        record["part_id"]
        for record in records
        if record["member_role"] == "strict_spatial_anchor"
    } == {"P0001", "P0002", "P0003"}


def test_dominant_assembly_member_hard_vetoes_strong_alternative() -> None:
    registry_part = {
        "existing_visual_material_properties": {
            "shader_path": "/Looks/Target",
            "shader_id": "UsdPreviewSurface",
            "diffuseColor": [0.6, 0.8, 1.0],
        }
    }
    signature = _sha256_document(
        {
            "shader_id": "UsdPreviewSurface",
            "diffuseColor": [0.6, 0.8, 1.0],
        }
    )
    spatial_part = {
        "observations": [
            {
                "reference_view_id": "iso",
                "classification": "conflict",
                "canonical_group_id": "G01",
                "registration_label_stable": True,
                "perturbation_label_stable": True,
                "projected_part_pixels": 800,
                "group_scores": [{"canonical_group_id": "G01", "color_share": 0.92}],
                "color_margin": 0.7,
                "bbox_canonical_group_id": "G01",
                "bbox_group_scores": [
                    {"canonical_group_id": "G01", "color_share": 0.9}
                ],
            }
        ],
        "semantic_votes": [
            {
                "view_id": "front",
                "status": "review",
                "canonical_group_id": "G06",
                "effective_confidence": 0.95,
            }
        ],
        "multiview_dark_consensus": None,
    }

    reasons, evidence = _dominant_assembly_member_veto(
        part_id="P0130",
        target_group_id="G06",
        target_material_id=GREEN,
        target_source_signature=signature,
        registry_part=registry_part,
        spatial_part=spatial_part,
        spatial_gate_decision=None,
        mapping_decision=None,
        geometry_risk=False,
        baseline_assignment={
            "status": "policy_fallback",
            "provenance": {"tier": "neutral_default"},
        },
        existing_proposals=[],
    )

    assert reasons == ["DOMINANT_ASSEMBLY_STRONG_DIRECT_ALTERNATIVE"]
    assert evidence[0]["canonical_group_id"] == "G01"


def _rare_pair_observation(*, exact: bool) -> dict:
    target_count = 6 if exact else 5
    alternatives = [] if exact else ["G02"]
    return {
        "reference_view_id": "front",
        "classification": "insufficient_visibility",
        "canonical_group_id": None,
        "registration_label_stable": True,
        "perturbation_label_stable": exact,
        "declared_visible_pixels": 176 if exact else 284,
        "projected_part_pixels": 176 if exact else 284,
        "group_scores": [
            {
                "canonical_group_id": "G07",
                "color_share": 0.78 if exact else 0.46,
            }
        ],
        "color_margin": 0.78 if exact else 0.25,
        "bbox_canonical_group_id": "G07",
        "bbox_group_scores": [
            {
                "canonical_group_id": "G07",
                "color_share": 0.73 if exact else 0.46,
            }
        ],
        "bbox_color_margin": 0.73 if exact else 0.36,
        "projection_perturbations": [
            {
                "offset_pixels": list(offset),
                "sampled_reference_pixels": 176 if exact else 284,
                "canonical_group_id": ("G07" if exact or offset != (2, 0) else "G02"),
                "diagnostic_canonical_group_id": (
                    "G07" if exact or offset != (2, 0) else "G02"
                ),
                "best_color_share": 0.6,
                "color_margin": 0.55,
            }
            for offset in ((-2, 0), (2, 0), (0, -2), (0, 2))
        ],
        "small_part_diagnostic": {
            "status": "resolved",
            "reason_codes": [],
            "canonical_group_id": "G07",
            "bbox_canonical_group_id": "G07",
            "registration_label_stable": True,
            "resolved_sample_count": 6,
            "target_sample_count": target_count,
            "consensus_ratio": round(target_count / 6, 8),
            "alternative_canonical_group_ids": alternatives,
        },
    }


def _rare_pair_inputs() -> dict:
    stable_properties = {
        "shader_id": "UsdPreviewSurface",
        "diffuseColor": [1.0, 0.5, 0.0],
        "metallic": 0.5,
        "roughness": 0.5,
    }
    registry_by_part = {
        "P0001": {
            "part_id": "P0001",
            "parent_path": "/Asset/OrangeAssembly/Left",
            "face_count": 200,
            "existing_visual_material_properties": {
                "shader_path": "/Looks/Left",
                **stable_properties,
            },
        },
        "P0002": {
            "part_id": "P0002",
            "parent_path": "/Asset/OrangeAssembly/Right",
            "face_count": 220,
            "existing_visual_material_properties": {
                "shader_path": "/Looks/Right",
                **stable_properties,
            },
        },
        "P0003": {
            "part_id": "P0003",
            "parent_path": "/Asset/Other",
            "face_count": 100,
            "existing_visual_material_properties": {
                "shader_path": "/Looks/Other",
                "shader_id": "UsdPreviewSurface",
                "diffuseColor": [0.2, 0.2, 0.2],
                "metallic": 0.0,
                "roughness": 0.8,
            },
        },
    }
    reference_evidence = {
        view_id: {
            "raw_sha256": character * 64,
            "normalized_pixel_sha256": character.upper() * 64,
            "content_cluster_id": f"content_{view_id}",
            "pose_cluster_id": f"pose_{view_id}",
        }
        for view_id, character in zip(("front", "iso", "side"), "abc")
    }
    return {
        "canonical_groups": {
            "G07": {
                "group_id": "G07",
                "base_color": "orange",
                "confidence": 0.6,
                "source_view_ids": ["front", "iso", "side"],
                "distinct_view_count": 3,
                "singleton": False,
            }
        },
        "repairable_groups": {
            "G07": {
                "repairable": True,
                "supporting_views": [
                    {"reference_view_id": view_id}
                    for view_id in ("front", "iso", "side")
                ],
            }
        },
        "registry_by_part": registry_by_part,
        "spatial_parts": {
            "P0001": {
                "observations": [_rare_pair_observation(exact=True)],
                "semantic_votes": [],
                "multiview_dark_consensus": None,
            },
            "P0002": {
                "observations": [_rare_pair_observation(exact=False)],
                "semantic_votes": [],
                "multiview_dark_consensus": None,
            },
            "P0003": {
                "observations": [],
                "semantic_votes": [],
                "multiview_dark_consensus": None,
            },
        },
        "reference_evidence": reference_evidence,
        "gate_decisions": {},
        "mapping_decisions": {},
        "geometry_risks": {part_id: False for part_id in registry_by_part},
        "baseline_by_part": {
            part_id: {
                "part_id": part_id,
                "material_id": STAINLESS,
                "status": "policy_fallback",
                "provenance": {"tier": "neutral_default"},
            }
            for part_id in registry_by_part
        },
        "occupied_proposals_by_part": {},
        "confirmed_materials": {"G07": GREEN},
        "provisional_material_groups": set(),
        "spatial_policy": {
            "minimum_visible_pixels": 256,
            "minimum_diagnostic_visible_pixels": 128,
            "minimum_isolated_source_visible_pixels": 12,
            "minimum_diagnostic_resolved_samples": 3,
            "minimum_diagnostic_consensus_ratio": 0.75,
            "minimum_semantic_confidence": 0.7,
        },
        "minimum_semantic_confidence": 0.7,
        "input_hashes": {
            key: character * 64
            for key, character in zip(
                (
                    "baseline_plan_sha256",
                    "quality_report_sha256",
                    "palette_fusion_sha256",
                    "spatial_report_sha256",
                    "spatial_gate_audit_sha256",
                    "mapping_consensus_sha256",
                    "geometry_risk_sha256",
                    "group_materials_sha256",
                    "registry_sha256",
                ),
                "abcdefghi",
            )
        },
    }


def test_rare_source_identity_pair_is_only_a_pending_atomic_candidate() -> None:
    records, cohorts = _bounded_signature_sibling_cohort_expansions(
        **_rare_pair_inputs()
    )

    assert len(cohorts) == 1
    cohort = cohorts[0]
    assert cohort["candidate_kind"] == "rare_source_identity_pair"
    assert (
        cohort["proposal_policy"] == "single_strict_anchor_bounded_signature_sibling/v1"
    )
    assert cohort["anchor_part_ids"] == ["P0001"]
    assert cohort["expanded_member_part_ids"] == ["P0002"]
    assert cohort["cohort_part_ids"] == ["P0001", "P0002"]
    assert cohort["anchor_evidence"][0]["target_sample_count"] == 6
    assert cohort["bounded_sibling_evidence"]["target_sample_count"] == 5
    assert cohort["render_membership_requirement"] == {
        "minimum_trusted_reference_view_count": 2,
        "all_available_reference_views_required": True,
        "non_target_group_regression_forces_m0": True,
        "incomplete_or_ambiguous_evidence_forces_m0": True,
    }
    assert {record["part_id"]: record["member_role"] for record in records} == {
        "P0001": "strict_spatial_anchor",
        "P0002": "expanded_member",
    }


def test_rare_source_identity_pair_fails_closed_on_contract_weakening() -> None:
    matched_alternative = _rare_pair_inputs()
    matched_alternative["mapping_decisions"] = {
        "P0002": {
            "output_status": "matched",
            "output_group_id": "G02",
        }
    }
    assert _bounded_signature_sibling_cohort_expansions(**matched_alternative) == (
        [],
        [],
    )

    matched_semantic_alternative = _rare_pair_inputs()
    matched_semantic_alternative["spatial_parts"]["P0002"]["semantic_votes"] = [
        {
            "view_id": "front",
            "alignment_trusted": True,
            "unique_canonical_join": True,
            "status": "matched",
            "canonical_group_id": "G02",
        }
    ]
    assert _bounded_signature_sibling_cohort_expansions(
        **matched_semantic_alternative
    ) == ([], [])

    non_rare_signature = _rare_pair_inputs()
    source_properties = copy.deepcopy(
        non_rare_signature["registry_by_part"]["P0001"][
            "existing_visual_material_properties"
        ]
    )
    source_properties["shader_path"] = "/Looks/Third"
    non_rare_signature["registry_by_part"]["P0003"][
        "existing_visual_material_properties"
    ] = source_properties
    assert _bounded_signature_sibling_cohort_expansions(**non_rare_signature) == (
        [],
        [],
    )


def _documents() -> dict[str, dict]:
    registry = {
        "schema_version": "qwen-material-parts/v1",
        "asset_sha256": "f" * 64,
        "part_count": 2,
        "parts": [
            {"part_id": "P0001", "prim_path": "/Asset/Panel/Mesh"},
            {"part_id": "P0002", "prim_path": "/Asset/Bolt/Mesh"},
        ],
    }
    whitelist = {
        "schema_version": 1,
        "material_ids": [GREEN, STAINLESS, GALVANIZED],
    }
    baseline_plan = {
        "schema_version": "1.0",
        "assignments": [
            {
                "part_id": "P0001",
                "material_id": STAINLESS,
                "semantic": "neutral unresolved component",
                "confidence": 0.0,
                "evidence_views": [],
                "status": "policy_fallback",
                "provenance": {
                    "tier": "neutral_default",
                    "reason_codes": ["POLICY_DECLARED_NEUTRAL_DEFAULT"],
                    "output_confidence_basis": (POLICY_FALLBACK_CONFIDENCE_BASIS),
                    "sources": [],
                },
            },
            {
                "part_id": "P0002",
                "material_id": GALVANIZED,
                "semantic": "standard fastener",
                "confidence": 0.0,
                "evidence_views": [],
                "status": "policy_fallback",
                "provenance": {
                    "tier": "semantic_rule",
                    "reason_codes": ["POLICY_SEMANTIC_RULE"],
                    "output_confidence_basis": (POLICY_FALLBACK_CONFIDENCE_BASIS),
                    "sources": [],
                },
            },
        ],
        "provenance": {
            "mode": "explicit_best_effort_policy_exact_cover",
            "registry_asset_sha256": registry["asset_sha256"],
            "registry_sha256": _sha256_document(registry),
            "whitelist_sha256": _sha256_document(whitelist),
        },
    }
    baseline_policy_audit = {
        "schema_version": "qwen-policy-exact-cover-report/v1",
        "summary": {
            "registry_part_count": 2,
            "output_assignment_count": 2,
            "exact_cover": True,
            "all_materials_in_industrial_whitelist": True,
        },
        "input_hashes": copy.deepcopy(baseline_plan["provenance"]),
        "output_plan_sha256": _sha256_document(baseline_plan),
    }
    palette_fusion = {
        "schema_version": "qwen-multiview-palette-fusion/v1",
        "canonical_palette": {
            "schema_version": "qwen-canonical-material-palette/v1",
            "groups": [
                {
                    "group_id": "G01",
                    "family_hint": "metal",
                    "base_color": "green",
                    "finish_hint": "painted",
                    "visual_description": "green painted machine panel",
                    "distinct_view_count": 2,
                    "singleton": False,
                    "source_view_ids": ["ref_a", "ref_b"],
                }
            ],
        },
        "view_group_id_maps": {
            "ref_a": {"L01": "G01"},
            "ref_b": {"L09": "G01"},
        },
    }
    spatial_report = _seal_spatial(
        {
            "schema_version": "qwen-spatial-mapping-audit/v1",
            "policy": {
                "minimum_semantic_confidence": 0.85,
            },
            "inputs": {"files": [], "document_sha256": {}},
            "reference_evidence": [
                {
                    "view_id": "ref_a",
                    "raw_sha256": "a" * 64,
                    "normalized_pixel_sha256": "1" * 64,
                    "perceptual_hash": "0123456789abcdef",
                    "content_cluster_id": "CONTENT_01",
                    "selected_render_view_id": "front",
                    "pose_cluster_id": "front",
                    "alignment_trusted": True,
                    "alignment_score": 0.9,
                },
                {
                    "view_id": "ref_b",
                    "raw_sha256": "b" * 64,
                    "normalized_pixel_sha256": "2" * 64,
                    "perceptual_hash": "fedcba9876543210",
                    "content_cluster_id": "CONTENT_02",
                    "selected_render_view_id": "side",
                    "pose_cluster_id": "side",
                    "alignment_trusted": True,
                    "alignment_score": 0.91,
                },
            ],
            "view_alignments": [],
            "parts": [
                {
                    "part_id": part_id,
                    "observations": [
                        {
                            "reference_view_id": "ref_a",
                            "classification": "resolved",
                            "canonical_group_id": "G01",
                            "registration_label_stable": True,
                            "perturbation_label_stable": True,
                        },
                        {
                            "reference_view_id": "ref_b",
                            "classification": "resolved",
                            "canonical_group_id": "G01",
                            "registration_label_stable": True,
                            "perturbation_label_stable": True,
                        },
                    ],
                    "resolved_support_counts": {"G01": 2},
                    "conflict_view_ids": [],
                    "semantic_votes": [],
                }
                for part_id in ("P0001", "P0002")
            ],
            "summary": {},
        }
    )
    spatial_gate_audit = {
        "schema_version": "qwen-spatial-mapping-gate/v1",
        "policy": {},
        "decisions": [
            {
                "part_id": part_id,
                "output_group_id": "G01",
                "output_status": "matched",
                "decision": "kept_auto",
                "conflicting_view_ids": [],
                "semantic_conflicting_view_ids": [],
                "semantic_unresolved_view_ids": [],
                "semantic_multi_material_view_ids": [],
                "semantic_nondeterministic_content_cluster_ids": [],
                "validation_lanes": ["spatial_projection"],
            }
            for part_id in ("P0001", "P0002")
        ],
        "summary": {},
    }
    mapping_consensus = {
        "schema_version": "qwen-mapping-consensus-audit/v1",
        "policy": {},
        "decisions": [
            {
                "part_id": part_id,
                "output_group_id": "G01",
                "output_status": "matched",
                "conflicting_view_ids": [],
            }
            for part_id in ("P0001", "P0002")
        ],
        "summary": {},
    }
    geometry_risk = {
        "schema_version": "qwen-geometry-uniform-material-risk/v1",
        "part_count": 2,
        "parts": [
            {
                "part_id": part_id,
                "risk": {
                    "multi_material_risk": False,
                    "basis": "test",
                },
            }
            for part_id in ("P0001", "P0002")
        ],
        "summary": {},
    }
    group_materials = {
        "schema_version": "qwen-palette-material/v1",
        "selections": [
            {
                "group_id": "G01",
                "material_id": GREEN,
                "confidence": 0.97,
                "confirmed": True,
            }
        ],
    }

    def quality_view(
        reference_view_id: str,
        render_view_id: str,
        image_sha256: str,
        local_group_id: str,
    ) -> dict:
        return {
            "reference_view_id": reference_view_id,
            "render_view_id": render_view_id,
            "status": "FAIL",
            "reasons": ["trusted_palette_group_missing_from_render"],
            "mapping": {
                "selected_render_view_id": render_view_id,
                "reasons": [],
            },
            "reference": {
                "image_sha256": image_sha256,
                "trusted_evidence": {"usable": True, "reasons": []},
            },
            "render": {},
            "alignment": {"score": 0.9},
            "material_color": {
                "trusted_evidence_group_recall": {
                    "groups": [
                        {
                            "group_id": local_group_id,
                            "base_colors": ["green"],
                            "recall": 0.0,
                        }
                    ]
                }
            },
        }

    quality_report = {
        "schema_version": "qwen-reference-render-comparison/v1",
        "inputs": {},
        "thresholds": {
            "minimum_evidence_group_recall": 0.2,
            "strong_alignment_score": 0.55,
        },
        "aggregate": {
            "status": "FAIL",
            "comparable_view_count": 2,
        },
        "views": [
            quality_view("ref_a", "front", "a" * 64, "L01"),
            quality_view("ref_b", "side", "b" * 64, "L09"),
        ],
    }
    return {
        "baseline_plan": baseline_plan,
        "baseline_policy_audit": baseline_policy_audit,
        "quality_report": quality_report,
        "palette_fusion": palette_fusion,
        "spatial_report": spatial_report,
        "spatial_gate_audit": spatial_gate_audit,
        "mapping_consensus": mapping_consensus,
        "geometry_risk": geometry_risk,
        "group_materials": group_materials,
        "registry": registry,
        "whitelist": whitelist,
    }


def _build(documents: dict[str, dict]) -> tuple[dict, dict]:
    return build_quality_repair_plan(**documents)


def _configure_visual_source_identity_bridge(documents: dict[str, dict]) -> None:
    shared_visual = {
        "diffuse_color": [0.95, 0.35, 0.05],
        "metallic": 0.0,
        "roughness": 0.45,
    }
    for part in documents["registry"]["parts"]:
        part["parent_path"] = f"/Asset/OrangeAssembly/{part['part_id']}"
        part["existing_visual_material_properties"] = copy.deepcopy(shared_visual)
    second_assignment = documents["baseline_plan"]["assignments"][1]
    second_assignment.update(
        {
            "material_id": STAINLESS,
            "semantic": "neutral unresolved component",
            "confidence": 0.0,
            "evidence_views": [],
            "status": "policy_fallback",
            "provenance": {
                "tier": "neutral_default",
                "reason_codes": ["POLICY_DECLARED_NEUTRAL_DEFAULT"],
                "output_confidence_basis": POLICY_FALLBACK_CONFIDENCE_BASIS,
                "sources": [],
            },
        }
    )
    documents["baseline_plan"]["provenance"]["registry_sha256"] = _sha256_document(
        documents["registry"]
    )
    documents["baseline_policy_audit"]["input_hashes"] = copy.deepcopy(
        documents["baseline_plan"]["provenance"]
    )
    documents["baseline_policy_audit"]["output_plan_sha256"] = _sha256_document(
        documents["baseline_plan"]
    )

    spatial = copy.deepcopy(documents["spatial_report"])
    spatial.pop("integrity", None)
    spatial["policy"].update(
        {
            "minimum_semantic_conflict_confidence": 0.6,
            "minimum_visible_pixels": 256,
            "minimum_diagnostic_visible_pixels": 128,
            "minimum_isolated_source_visible_pixels": 12,
            "minimum_isolated_source_view_count": 2,
            "minimum_diagnostic_resolved_samples": 3,
            "minimum_diagnostic_consensus_ratio": 0.75,
        }
    )
    for index, part in enumerate(spatial["parts"]):
        part["observations"] = [
            {
                "reference_view_id": "ref_a",
                "classification": "unresolved",
                "canonical_group_id": None,
                "evidence_mode": "isolated_mask_multiview_diagnostic",
                "isolated_evidence_sha256": f"{index + 3:x}" * 64,
                "isolated_source_view_count": 2,
                "declared_visible_pixels": 30,
                "projected_part_pixels": 28,
                "perturbation_label_stable": None,
                "small_part_diagnostic": {
                    "status": "resolved",
                    "reason_codes": [],
                    "canonical_group_id": "G01",
                    "bbox_canonical_group_id": "G01",
                    "registration_label_stable": True,
                    "resolved_sample_count": 6,
                    "target_sample_count": 6,
                    "consensus_ratio": 1.0,
                    "alternative_canonical_group_ids": [],
                },
            }
        ]
        part["resolved_support_counts"] = {}
        part["semantic_votes"] = (
            [
                {
                    "view_id": "ref_b",
                    "alignment_trusted": True,
                    "unique_canonical_join": True,
                    "canonical_group_id": "G01",
                    "status": "review",
                    "effective_confidence": 0.6,
                    "reason_code": "direct_visual_match",
                }
            ]
            if index == 0
            else []
        )
    documents["spatial_report"] = _seal_spatial(spatial)


def test_visual_objective_bridges_one_direct_cohort_anchor_with_multiview_qa() -> None:
    documents = _documents()
    _configure_visual_source_identity_bridge(documents)

    _legacy_plan, legacy_audit = _build(copy.deepcopy(documents))
    assert legacy_audit["summary"]["changed_count"] == 0

    _plan, audit = build_quality_repair_plan(
        **documents,
        allow_parameter_writes=False,
        material_selection_objective="visual_similarity",
    )
    assert audit["summary"]["changed_count"] == 2
    assert {
        (item["part_id"], item["lane"]) for item in audit["localization_lanes"]
    } == {
        ("P0001", "source_identity_cohort_multiview_consensus"),
        ("P0002", "source_identity_cohort_multiview_consensus"),
    }
    assert {item["source_identity_consensus_mode"] for item in audit["changes"]} == {
        "direct_visual_anchor_plus_multiview_qa_deficit"
    }
    assert {
        tuple(item["source_identity_consensus_view_ids"]) for item in audit["changes"]
    } == {("ref_b",)}


def _enable_dominant_only_deficits(documents: dict[str, dict]) -> None:
    thresholds = documents["quality_report"]["thresholds"]
    thresholds.update(
        {
            "minimum_dominant_reference_share": 0.25,
            "minimum_dominant_share_margin": 0.10,
            "minimum_dominant_mass_recall": 0.80,
            "minimum_dominant_absolute_deficit": 0.08,
            "minimum_dominant_silhouette_iou": 0.75,
        }
    )
    documents["quality_report"]["aggregate"]["reasons"] = [
        "single_strong_view_confirms_dominant_family_mass_deficit"
    ]
    for view in documents["quality_report"]["views"]:
        local_group_id = view["material_color"]["trusted_evidence_group_recall"][
            "groups"
        ][0]["group_id"]
        view["material_color"]["trusted_evidence_group_recall"]["groups"][0][
            "recall"
        ] = 1.0
        view["reasons"] = ["trusted_dominant_family_mass_deficit"]
        view["alignment"]["silhouette_iou"] = 0.9
        view["material_color"].update(
            {
                "reference_distribution": {
                    "category_distribution": {
                        "green": 0.8,
                        "achromatic_mid": 0.2,
                    }
                },
                "render_distribution": {
                    "category_distribution": {
                        "green": 0.5,
                        "achromatic_mid": 0.5,
                    }
                },
                "trusted_evidence_dominant_mass": {
                    "status": "FAIL",
                    "eligible_family_count": 1,
                    "failed_family_count": 1,
                    "families": [
                        {
                            "family_key": "green",
                            "local_group_ids": [local_group_id],
                            "base_colors": ["green"],
                            "render_color_bins": ["green"],
                            "reference_share": 0.8,
                            "runner_up_reference_share": 0.0,
                            "reference_share_margin": 0.8,
                            "observed_render_share": 0.5,
                            "deficit_share": 0.3,
                            "mass_recall": 0.625,
                            "eligible": True,
                            "status": "FAIL",
                            "reason_codes": ["DOMINANT_FAMILY_MASS_DEFICIT"],
                        }
                    ],
                },
            }
        )


def _reseal_baseline(documents: dict[str, dict]) -> None:
    plan = documents["baseline_plan"]
    plan["provenance"]["whitelist_sha256"] = _sha256_document(documents["whitelist"])
    documents["baseline_policy_audit"]["input_hashes"] = copy.deepcopy(
        plan["provenance"]
    )
    documents["baseline_policy_audit"]["output_plan_sha256"] = _sha256_document(plan)


def _dark_diagnostic(
    *,
    part_id: str,
    projected_pixels: int = 200,
    dark_signal_pixels: int = 90,
) -> dict:
    normalized_pixels = 200
    null_shifts = [
        {
            "offset_pixels": offset,
            "retained_pixels": normalized_pixels,
            "valid_area_ratio": 1.0,
            "valid": True,
            "dark_signal_pixels": 20,
            "dark_signal_share": 0.1,
            "mask_sha256": f"{index + 1:x}" * 64,
        }
        for index, offset in enumerate(
            (
                [-20, 0],
                [20, 0],
                [0, -20],
                [0, 20],
                [-20, -20],
                [-20, 20],
                [20, -20],
                [20, 20],
            )
        )
    ]
    dark_share = dark_signal_pixels / normalized_pixels
    diagnostic = {
        "status": "resolved",
        "reason_codes": [],
        "evidence_scope": "dark_on_black_foreground_repair_only",
        "canonical_group_id": "G01",
        "canonical_source_view_ids": ["ref_a", "ref_b"],
        "alternative_canonical_group_ids": [],
        "projected_part_pixels": projected_pixels,
        "normalized_projected_pixels": normalized_pixels,
        "normalization": {
            "long_edge_pixels": 512,
            "original_size": [400, 500],
            "normalized_size": [410, 512],
            "scale": 1.024,
        },
        "alignment": {
            "trusted": True,
            "reason_codes_empty": True,
            "score": 0.92,
            "projection_score": 0.92,
            "projection_iou": 0.90,
            "ecc_status": "success",
            "ecc_correlation": 0.93,
            "transform_constraints_passed": True,
            "strong": True,
        },
        "background": {
            "median_bgr": [0.0, 0.0, 0.0],
            "border_distance_p95": 0.0,
            "distance_threshold": 12.0,
        },
        "thresholds": copy.deepcopy(DARK_FOREGROUND_DIAGNOSTIC_THRESHOLDS),
        "near_black_pixels": 150,
        "near_black_share": 0.75,
        "non_background_pixels": 100,
        "non_background_share": 0.5,
        "dark_signal_pixels": dark_signal_pixels,
        "dark_signal_share": dark_share,
        "dark_signal_purity": dark_signal_pixels / 100,
        "core_pixels": 100,
        "core_dark_signal_pixels": 60,
        "core_dark_signal_share": 0.6,
        "core_distance_pixels": 2.2,
        "adaptive_edge_pixels": 70,
        "adaptive_edge_density": 0.35,
        "adaptive_edge_threshold": 12.0,
        "border_gradient_p99": 0.0,
        "canny_low_threshold": 6,
        "canny_high_threshold": 12,
        "canny_edge_pixels": 40,
        "canny_edge_density": 0.2,
        "null_shifts": null_shifts,
        "valid_null_shift_count": 8,
        "null_dark_signal_share_q75": 0.1,
        "dark_signal_null_margin": dark_share - 0.1,
        "normalized_reference_pixel_sha256": "a" * 64,
        "normalized_projected_mask_sha256": "b" * 64,
        "normalized_near_black_mask_sha256": "c" * 64,
        "normalized_non_background_mask_sha256": "d" * 64,
        "normalized_dark_signal_mask_sha256": "e" * 64,
        "normalized_adaptive_edge_mask_sha256": "f" * 64,
    }
    diagnostic["diagnostic_sha256"] = _sha256_document(diagnostic)
    return diagnostic


def _configure_dark_residual(
    documents: dict[str, dict],
    *,
    part_ids: tuple[str, ...] = ("P0001",),
    projected_pixels: int = 200,
) -> None:
    documents["whitelist"]["material_ids"].append(BLACK)
    group = documents["palette_fusion"]["canonical_palette"]["groups"][0]
    group.update(
        {
            "base_color": "black",
            "visual_description": "black painted metal",
        }
    )
    documents["palette_fusion"]["view_group_id_maps"] = {
        "ref_a": {},
        "ref_b": {},
    }
    documents["group_materials"]["selections"][0]["material_id"] = BLACK
    documents["quality_report"]["views"] = [documents["quality_report"]["views"][0]]
    documents["quality_report"]["aggregate"]["comparable_view_count"] = 1
    view = documents["quality_report"]["views"][0]
    view["alignment"] = {
        "score": 0.92,
        "silhouette_iou": 0.91,
        "edge_f1_tolerance_3px": 0.92,
        "profile_similarity": 0.96,
        "bbox_aspect_similarity": 0.97,
    }
    view["reference"]["foreground"] = {"pixel_count": 10_000}
    view["render"] = {"foreground": {"pixel_count": 12_000}}
    view["material_color"] = {
        "trusted_evidence_group_recall": {
            "groups": [
                {
                    "group_id": "L_LOCAL_PALETTE_HAS_NO_BLACK",
                    "base_colors": ["green"],
                    "recall": 1.0,
                }
            ]
        },
        "reference_distribution": {
            "sampled_pixels": 10_000,
            "sample_step": 1,
            "category_distribution": {
                "black": 0.08,
                "achromatic_dark": 0.02,
                "achromatic_mid": 0.90,
            },
        },
        "render_distribution": {
            "sampled_pixels": 12_000,
            "sample_step": 1,
            "category_distribution": {
                "black": 0.01,
                "achromatic_dark": 0.01,
                "achromatic_mid": 0.98,
            },
        },
    }
    documents["spatial_report"]["view_alignments"] = [
        {
            "reference_view_id": "ref_a",
            "selected_render_view_id": "front",
            "trusted": True,
            "score": 0.92,
            "projection_iou": 0.90,
            "ecc_correlation": 0.93,
            "ecc_status": "success",
        }
    ]
    for part in documents["spatial_report"]["parts"]:
        part_id = part["part_id"]
        if part_id not in part_ids:
            part["observations"] = []
            continue
        diagnostic = _dark_diagnostic(
            part_id=part_id,
            projected_pixels=projected_pixels,
        )
        part["observations"] = [
            {
                "reference_view_id": "ref_a",
                "classification": "review",
                "canonical_group_id": None,
                "registration_label_stable": False,
                "perturbation_label_stable": False,
                "projected_part_pixels": projected_pixels,
                "dark_foreground_diagnostic": diagnostic,
            }
        ]
    for assignment in documents["baseline_plan"]["assignments"]:
        if assignment["part_id"] in part_ids:
            assignment["provenance"]["tier"] = "neutral_default"
    documents["spatial_report"] = _seal_spatial(documents["spatial_report"])
    _reseal_baseline(documents)


def test_dark_foreground_residual_repairs_when_local_palette_missed_black() -> None:
    documents = _documents()
    _configure_dark_residual(documents)

    plan, audit = _build(documents)

    assert documents["palette_fusion"]["view_group_id_maps"]["ref_a"] == {}
    assert plan["assignments"][0]["material_id"] == BLACK
    assert audit["summary"]["changed_count"] == 1
    assert audit["localization_lanes"] == [
        {
            "part_id": "P0001",
            "canonical_group_id": "G01",
            "lane": DARK_FOREGROUND_RESIDUAL_LANE,
        }
    ]
    change = audit["changes"][0]
    support = change["dark_residual_support"]
    assert support["local_group_id"] == "__canonical_dark__:G01"
    assert support["deficit_sources"] == ["dark_foreground_achromatic_residual"]
    assert support["reference_share"] == pytest.approx(0.10)
    assert support["observed_render_share"] == pytest.approx(0.02)
    assert support["budget_pixels"] == 800
    assert change["estimated_contribution_pixels"] == 75
    assert change["selected_contribution_pixels"] == 75
    assert audit["dark_residual_budgets"][0]["selected_part_ids"] == ["P0001"]
    assert plan["assignments"][0]["provenance"]["reason_codes"] == [
        "QA_DARK_FOREGROUND_ACHROMATIC_RESIDUAL",
        "QA_TRUSTED_PART_GROUP_LOCALIZATION",
        "QA_CONFIRMED_WHITELIST_MATERIAL",
    ]
    assert "parameters" not in plan["assignments"][0]


def test_multiview_dark_identity_can_localize_outside_global_deficit_view() -> None:
    documents = _documents()
    _configure_dark_residual(documents)
    spatial = documents["spatial_report"]
    spatial["policy"].update(
        {
            "minimum_spatial_support_views": 2,
            "minimum_color_share": 0.60,
        }
    )
    consensus = {
        "status": "resolved",
        "canonical_group_id": "G01",
        "supporting_view_ids": ["ref_a", "ref_b"],
        "minimum_independent_support_views": 2,
        "evidence_contract": (
            "stable_projection_and_dark_interior_multiview_consensus"
        ),
    }
    part = spatial["parts"][0]
    observations = []
    for view_id, projected_pixels in (("ref_a", 200), ("ref_b", 240)):
        diagnostic = _dark_diagnostic(
            part_id="P0001",
            projected_pixels=projected_pixels,
        )
        # Simulate a black-on-black single-view degeneracy.  The individual
        # diagnostic is deliberately rejected, but its primitive evidence is
        # strong and two independent projections agree on the same part.
        diagnostic["status"] = "rejected"
        diagnostic["reason_codes"] = ["DARK_ALIGNMENT_NOT_STRONG"]
        diagnostic.pop("diagnostic_sha256")
        diagnostic["diagnostic_sha256"] = _sha256_document(diagnostic)
        observations.append(
            {
                "reference_view_id": view_id,
                "classification": "resolved",
                "reason_code": "multiview_dark_consensus_resolved",
                "canonical_group_id": "G01",
                "bbox_canonical_group_id": "G01",
                "registration_label_stable": True,
                "perturbation_label_stable": True,
                "projected_part_pixels": projected_pixels,
                "group_scores": [
                    {
                        "canonical_group_id": "G01",
                        "base_color": "black",
                        "color_share": 0.80,
                    }
                ],
                "dark_foreground_diagnostic": diagnostic,
                "multiview_dark_consensus": copy.deepcopy(consensus),
            }
        )
    part["observations"] = observations
    part["multiview_dark_consensus"] = copy.deepcopy(consensus)
    spatial["parts"][1]["observations"] = []
    documents["spatial_report"] = _seal_spatial(spatial)

    plan, audit = _build(documents)

    assert plan["assignments"][0]["material_id"] == BLACK
    assert audit["summary"]["changed_count"] == 1
    assert audit["localization_lanes"] == [
        {
            "part_id": "P0001",
            "canonical_group_id": "G01",
            "lane": MULTIVIEW_DARK_IDENTITY_LANE,
        }
    ]
    provenance = plan["assignments"][0]["provenance"]["dark_foreground_residual"]
    assert provenance["lane"] == MULTIVIEW_DARK_IDENTITY_LANE
    assert "parameters" not in plan["assignments"][0]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda documents: documents["spatial_report"]["parts"][0]["observations"][0][
            "dark_foreground_diagnostic"
        ].update(
            {
                "status": "rejected",
                "reason_codes": ["DARK_NON_BACKGROUND_PIXELS_BELOW_FLOOR"],
            }
        ),
        lambda documents: documents["mapping_consensus"]["decisions"][0].update(
            {"output_status": "matched", "output_group_id": "G_OTHER"}
        ),
        lambda documents: documents["baseline_plan"]["assignments"][0][
            "provenance"
        ].update({"tier": "semantic_rule"}),
    ],
)
def test_dark_foreground_residual_fails_closed_for_unsafe_inputs(mutate) -> None:
    documents = _documents()
    _configure_dark_residual(documents)
    mutate(documents)
    diagnostic = documents["spatial_report"]["parts"][0]["observations"][0][
        "dark_foreground_diagnostic"
    ]
    if diagnostic["status"] == "rejected":
        diagnostic.pop("diagnostic_sha256")
        diagnostic["diagnostic_sha256"] = _sha256_document(diagnostic)
        documents["spatial_report"] = _seal_spatial(documents["spatial_report"])
    if (
        documents["baseline_plan"]["assignments"][0]["provenance"]["tier"]
        == "semantic_rule"
    ):
        _reseal_baseline(documents)

    plan, audit = _build(documents)

    assert plan == documents["baseline_plan"]
    assert audit["summary"]["changed_count"] == 0


def test_dark_foreground_diagnostic_numeric_tamper_fails_closed() -> None:
    documents = _documents()
    _configure_dark_residual(documents)
    diagnostic = documents["spatial_report"]["parts"][0]["observations"][0][
        "dark_foreground_diagnostic"
    ]
    diagnostic["dark_signal_share"] = 0.99
    diagnostic.pop("diagnostic_sha256")
    diagnostic["diagnostic_sha256"] = _sha256_document(diagnostic)
    documents["spatial_report"] = _seal_spatial(documents["spatial_report"])

    plan, audit = _build(documents)

    assert plan == documents["baseline_plan"]
    assert audit["summary"]["changed_count"] == 0
    assert (
        "DARK_DIAGNOSTIC_NUMERIC_EVIDENCE_INCONSISTENT" in audit["skip_reason_counts"]
    )


def test_dark_foreground_budget_is_bounded_and_order_deterministic() -> None:
    documents = _documents()
    _configure_dark_residual(
        documents,
        part_ids=("P0001", "P0002"),
        projected_pixels=2_000,
    )
    first_plan, first_audit = _build(documents)

    reordered = copy.deepcopy(documents)
    reordered["spatial_report"]["parts"].reverse()
    reordered["spatial_report"] = _seal_spatial(reordered["spatial_report"])
    second_plan, second_audit = _build(reordered)

    assert first_plan["assignments"] == second_plan["assignments"]
    assert first_audit["changes"] == second_audit["changes"]
    budget = first_audit["dark_residual_budgets"][0]
    assert budget["budget_pixels"] == 800
    assert budget["budget_limit_pixels"] == 1_080
    assert budget["selected_part_ids"] == ["P0001"]
    assert budget["selected_contribution_pixels"] == 750
    assert budget["total_contribution_pixels"] <= budget["budget_limit_pixels"]
    rejected = next(item for item in budget["candidates"] if item["part_id"] == "P0002")
    assert rejected["reason_code"] == "DARK_RESIDUAL_TOTAL_BUDGET_EXCEEDED"


def test_dominant_residual_review_override_requires_pixel_disproof_and_anchor() -> None:
    target_scores = [
        {"canonical_group_id": "G01", "color_share": 0.98},
        {"canonical_group_id": "G99", "color_share": 0.01},
    ]
    canonical_samples = [
        {"group_scores": copy.deepcopy(target_scores)},
        {"group_scores": copy.deepcopy(target_scores)},
        *[
            {
                "offset_pixels": list(offset),
                "group_scores": copy.deepcopy(target_scores),
            }
            for offset in ((-2, 0), (2, 0), (0, -2), (0, 2))
        ],
    ]
    observation = {
        "reference_view_id": "ref_a",
        "classification": "resolved",
        "canonical_group_id": "G01",
        "registration_label_stable": True,
        "perturbation_label_stable": True,
        "declared_visible_pixels": 5_000,
        "projected_part_pixels": 5_000,
        "group_scores": copy.deepcopy(target_scores),
        "color_margin": 0.97,
        "bbox_sampled_reference_pixels": 5_000,
        "bbox_canonical_group_id": "G01",
        "bbox_group_scores": copy.deepcopy(target_scores),
        "bbox_color_margin": 0.97,
        "projection_perturbations": [
            {
                "offset_pixels": list(offset),
                "canonical_group_id": "G01",
                "diagnostic_canonical_group_id": "G01",
                "sampled_reference_pixels": 5_000,
                "best_color_share": 0.98,
                "color_margin": 0.97,
            }
            for offset in ((-2, 0), (2, 0), (0, -2), (0, 2))
        ],
        "canonical_palette_diagnostic": {
            "direct_sample": canonical_samples[0],
            "bbox_sample": canonical_samples[1],
            "projection_perturbations": canonical_samples[2:],
        },
    }
    spatial_part = {
        "part_id": "P0001",
        "observations": [observation],
        "semantic_votes": [
            {
                "view_id": "ref_a",
                "reference_sha256": "a" * 64,
                "alignment_trusted": True,
                "unique_canonical_join": True,
                "pixel_gate_accepted": True,
                "status": "review",
                "canonical_group_id": "G99",
                "effective_confidence": 0.6,
            },
            {
                "view_id": "ref_b",
                "reference_sha256": "b" * 64,
                "alignment_trusted": True,
                "unique_canonical_join": True,
                "pixel_gate_accepted": True,
                "status": "review",
                "canonical_group_id": "G01",
                "effective_confidence": 0.81,
            },
        ],
    }
    deficit = {
        "supporting_views": [
            {
                "reference_view_id": "ref_a",
                "reference_sha256": "a" * 64,
                "requires_strict_local_projection": True,
                "deficit_sources": ["dominant_mass_local_projection"],
                "render_foreground_pixels": 10_000,
            }
        ]
    }
    alignments = {
        "ref_a": {
            "score": 0.9,
            "projection_iou": 0.9,
            "ecc_correlation": 0.92,
        },
        "ref_b": {
            "score": 0.9,
            "projection_iou": 0.9,
            "ecc_correlation": 0.92,
        },
    }
    references = {
        "ref_a": {"raw_sha256": "a" * 64},
        "ref_b": {"raw_sha256": "b" * 64},
    }

    supports, reasons, overrides, anchors = _dominant_residual_spatial_support_views(
        part_id="P0001",
        group_id="G01",
        spatial_part=spatial_part,
        spatial_gate_decision=None,
        mapping_decision={
            "output_status": "review",
            "output_group_id": "G01",
        },
        minimum_semantic_confidence=0.85,
        deficit=deficit,
        alignment_audits=alignments,
        spatial_policy={"minimum_semantic_conflict_confidence": 0.5},
        reference_evidence=references,
    )

    assert supports == ["ref_a"]
    assert reasons == []
    assert overrides == ["ref_a"]
    assert anchors == ["ref_b"]

    spatial_part["semantic_votes"][0]["status"] = "matched"
    _, reasons, overrides, anchors = _dominant_residual_spatial_support_views(
        part_id="P0001",
        group_id="G01",
        spatial_part=spatial_part,
        spatial_gate_decision=None,
        mapping_decision={"output_status": "review", "output_group_id": "G01"},
        minimum_semantic_confidence=0.85,
        deficit=deficit,
        alignment_audits=alignments,
        spatial_policy={"minimum_semantic_conflict_confidence": 0.5},
        reference_evidence=references,
    )
    assert "DOMINANT_RESIDUAL_SEMANTIC_GROUP_CONFLICT" in reasons
    assert overrides == []
    assert anchors == []


def _configure_repeated_dark_cohort(documents: dict[str, dict]) -> None:
    documents["whitelist"]["material_ids"].append(BLACK)
    group = documents["palette_fusion"]["canonical_palette"]["groups"][0]
    group.update({"base_color": "black", "visual_description": "black painted metal"})
    documents["group_materials"]["selections"][0]["material_id"] = BLACK
    documents["quality_report"]["views"] = [documents["quality_report"]["views"][0]]
    documents["quality_report"]["aggregate"]["comparable_view_count"] = 1
    view = documents["quality_report"]["views"][0]
    row = view["material_color"]["trusted_evidence_group_recall"]["groups"][0]
    row.update(
        {
            "base_colors": ["black"],
            "render_color_bins": ["black", "achromatic_dark"],
            "required_render_share": 0.05,
            "observed_render_share": 0.0095,
            "recall": 0.19,
        }
    )
    view["render"] = {"foreground": {"pixel_count": 40_000}}
    documents["spatial_report"]["view_alignments"] = [
        {
            "reference_view_id": "ref_a",
            "selected_render_view_id": "front",
            "trusted": True,
            "score": 0.82,
            "projection_iou": 0.86,
            "ecc_correlation": 0.92,
            "ecc_status": "success",
        }
    ]
    stable_properties = {
        "shader_path": "/Asset/Looks/Test/PreviewSurface",
        "shader_id": "UsdPreviewSurface",
        "diffuseColor": [1.0, 1.0, 0.0],
        "metallic": 0.5,
        "roughness": 0.5,
        "opacity": 1.0,
    }
    for index, part in enumerate(documents["registry"]["parts"]):
        part.update(
            {
                "point_count": 440,
                "face_count": 436,
                "world_bbox": [
                    [index * 100.0, 0.0, 0.0],
                    [index * 100.0 + 60, 160, 910],
                ],
                "existing_visual_material_properties": copy.deepcopy(stable_properties),
            }
        )
    for assignment in documents["baseline_plan"]["assignments"]:
        assignment.update(
            {
                "material_id": STAINLESS,
                "semantic": "neutral unresolved component",
                "provenance": {
                    "tier": "neutral_default",
                    "reason_codes": ["POLICY_DECLARED_NEUTRAL_DEFAULT"],
                    "output_confidence_basis": POLICY_FALLBACK_CONFIDENCE_BASIS,
                    "sources": [],
                },
            }
        )
    for part in documents["spatial_report"]["parts"]:
        diagnostic = _dark_diagnostic(
            part_id=part["part_id"],
            projected_pixels=700,
        )
        diagnostic.update(
            {
                "status": "rejected",
                "reason_codes": [
                    "DARK_CANONICAL_GROUP_CONFLICT",
                    "DARK_ALIGNMENT_NOT_STRONG",
                ],
                "projected_part_pixels": 700,
                "alignment": {
                    "trusted": True,
                    "reason_codes_empty": True,
                    "score": 0.82,
                    "projection_score": 0.86,
                    "projection_iou": 0.86,
                    "ecc_status": "success",
                    "ecc_correlation": 0.92,
                    "transform_constraints_passed": True,
                    "strong": False,
                },
            }
        )
        diagnostic.pop("diagnostic_sha256", None)
        diagnostic["diagnostic_sha256"] = _sha256_document(diagnostic)
        part["observations"] = [
            {
                "reference_view_id": "ref_a",
                "classification": "conflict",
                "canonical_group_id": "G01",
                "registration_label_stable": True,
                "perturbation_label_stable": True,
                "projected_part_pixels": 700,
                "group_scores": [
                    {
                        "canonical_group_id": "G01",
                        "color_share": 0.7,
                        "matching_pixels": 490,
                    },
                    {"canonical_group_id": "G99", "color_share": 0.3},
                ],
                "color_margin": 0.4,
                "bbox_canonical_group_id": "G01",
                "bbox_group_scores": [
                    {"canonical_group_id": "G01", "color_share": 0.93},
                    {"canonical_group_id": "G99", "color_share": 0.07},
                ],
                "bbox_color_margin": 0.86,
                "projection_perturbations": [
                    {
                        "offset_pixels": list(offset),
                        "canonical_group_id": "G01",
                        "diagnostic_canonical_group_id": "G01",
                        "sampled_reference_pixels": 700,
                        "best_color_share": 0.7,
                        "color_margin": 0.4,
                    }
                    for offset in ((-2, 0), (2, 0), (0, -2), (0, 2))
                ],
                "dark_foreground_diagnostic": diagnostic,
            }
        ]
        part["semantic_votes"] = []
    for decision in documents["mapping_consensus"]["decisions"]:
        decision.update(
            {
                "main_group_id": "G01",
                "main_status": "review",
                "main_confidence": 0.6,
                "output_group_id": "G01",
                "output_status": "review",
                "output_confidence": 0.6,
            }
        )

    for index in range(3, 41):
        part_id = f"P{index:04d}"
        documents["registry"]["parts"].append(
            {
                "part_id": part_id,
                "prim_path": f"/Asset/Dummy{index}/Mesh",
                "point_count": 100 + index,
                "face_count": 200 + index,
                "world_bbox": [[0, 0, 0], [index, index + 1, index + 2]],
                "existing_visual_material_properties": {
                    **stable_properties,
                    "diffuseColor": [0.5, 0.5, 0.5],
                },
            }
        )
        documents["baseline_plan"]["assignments"].append(
            {
                "part_id": part_id,
                "material_id": STAINLESS,
                "semantic": "neutral unresolved component",
                "confidence": 0.0,
                "evidence_views": [],
                "status": "policy_fallback",
                "provenance": {
                    "tier": "neutral_default",
                    "reason_codes": ["POLICY_DECLARED_NEUTRAL_DEFAULT"],
                    "output_confidence_basis": POLICY_FALLBACK_CONFIDENCE_BASIS,
                    "sources": [],
                },
            }
        )
        documents["spatial_report"]["parts"].append(
            {
                "part_id": part_id,
                "observations": [],
                "resolved_support_counts": {},
                "conflict_view_ids": [],
                "semantic_votes": [],
            }
        )
        documents["geometry_risk"]["parts"].append(
            {
                "part_id": part_id,
                "risk": {"multi_material_risk": False, "basis": "test"},
            }
        )
    documents["registry"]["part_count"] = 40
    documents["geometry_risk"]["part_count"] = 40
    documents["baseline_policy_audit"]["summary"].update(
        {"registry_part_count": 40, "output_assignment_count": 40}
    )
    documents["spatial_report"] = _seal_spatial(documents["spatial_report"])
    _reseal_baseline(documents)


def test_repeated_geometry_dark_cohort_is_atomic_and_not_parameterized() -> None:
    documents = _documents()
    _configure_repeated_dark_cohort(documents)

    plan, audit = _build(documents)

    assert [change["part_id"] for change in audit["changes"]] == ["P0001", "P0002"]
    assert {assignment["material_id"] for assignment in plan["assignments"][:2]} == {
        BLACK
    }
    assert all("parameters" not in assignment for assignment in plan["assignments"][:2])
    assert audit["repeated_geometry_dark_cohorts"][0]["cohort_part_ids"] == [
        "P0001",
        "P0002",
    ]
    assert {lane["lane"] for lane in audit["localization_lanes"]} == {
        REPEATED_GEOMETRY_DARK_RESIDUAL_LANE
    }
    assert all(
        change["repeated_geometry_dark_cohort_id"]
        == audit["repeated_geometry_dark_cohorts"][0]["cohort_id"]
        for change in audit["changes"]
    )


def test_strict_repeated_projection_overrides_nonlocal_qwen_review_votes() -> None:
    documents = _documents()
    _configure_repeated_dark_cohort(documents)
    for part in documents["spatial_report"]["parts"][:2]:
        part["observations"][0].pop("dark_foreground_diagnostic")
        part["semantic_votes"] = [
            {
                "view_id": "front",
                "alignment_trusted": True,
                "unique_canonical_join": True,
                "pixel_gate_accepted": True,
                "canonical_group_id": "G02",
                "status": "matched",
                "effective_confidence": 0.92,
            }
        ]
    for decision in documents["mapping_consensus"]["decisions"][:2]:
        decision.update(
            {
                "main_group_id": "G02",
                "main_status": "review",
                "main_confidence": 0.84,
                "output_group_id": "G02",
                "output_status": "review",
                "output_confidence": 0.84,
            }
        )
    documents["spatial_report"] = _seal_spatial(documents["spatial_report"])

    plan, audit = _build(documents)

    assert [change["part_id"] for change in audit["changes"]] == [
        "P0001",
        "P0002",
    ]
    assert {assignment["material_id"] for assignment in plan["assignments"][:2]} == {
        BLACK
    }
    members = audit["repeated_geometry_dark_cohorts"][0]["members"]
    assert {member["evidence_contract"] for member in members} == {
        "strict_reference_space_projection"
    }


def test_repeated_geometry_dark_cohort_fails_atomically_for_one_weak_member() -> None:
    documents = _documents()
    _configure_repeated_dark_cohort(documents)
    documents["spatial_report"]["parts"][1]["observations"][0]["bbox_color_margin"] = (
        0.79
    )
    documents["spatial_report"] = _seal_spatial(documents["spatial_report"])

    plan, audit = _build(documents)

    assert plan == documents["baseline_plan"]
    assert audit["repeated_geometry_dark_cohorts"] == []
    assert audit["summary"]["changed_count"] == 0


def test_dominant_mass_deficit_joins_one_canonical_group_without_direct_mutation() -> (
    None
):
    documents = _documents()
    _enable_dominant_only_deficits(documents)

    plan, audit = _build(documents)

    assert plan["assignments"][0]["material_id"] == GREEN
    diagnostic = next(
        item
        for item in audit["group_diagnostics"]
        if item["canonical_group_id"] == "G01"
    )
    assert diagnostic["repairable"] is True
    assert {
        tuple(item["deficit_sources"]) for item in diagnostic["supporting_views"]
    } == {("dominant_mass",)}
    assert {
        item["dominant_mass_family_key"] for item in diagnostic["supporting_views"]
    } == {"green"}
    assert audit["localization_lanes"] == [
        {
            "part_id": "P0001",
            "canonical_group_id": "G01",
            "lane": "stable_spatial_multiview",
        }
    ]


def test_dominant_mass_numeric_tampering_fails_closed() -> None:
    documents = _documents()
    _enable_dominant_only_deficits(documents)
    documents["quality_report"]["views"][0]["material_color"][
        "trusted_evidence_dominant_mass"
    ]["families"][0]["mass_recall"] = 0.7

    with pytest.raises(
        QualityRepairError,
        match="dominant-mass numeric evidence is inconsistent",
    ):
        _build(documents)


def test_compiles_one_hash_bound_neutral_fallback_repair() -> None:
    documents = _documents()

    plan, audit = _build(documents)

    assignments = {item["part_id"]: item for item in plan["assignments"]}
    repaired = assignments["P0001"]
    assert repaired["material_id"] == GREEN
    assert repaired["status"] == "policy_fallback"
    assert repaired["confidence"] == 0.0
    assert repaired["evidence_views"] == []
    assert repaired["provenance"]["tier"] == "qa_repair_candidate"
    assert repaired["provenance"]["canonical_group_id"] == "G01"
    assert repaired["provenance"]["supporting_view_ids"] == ["ref_a", "ref_b"]
    assert assignments["P0002"] == documents["baseline_plan"]["assignments"][1]
    assert plan["provenance"]["mode"] == POLICY_EXACT_COVER_MODE
    repair_provenance = plan["provenance"][REPAIR_PROVENANCE_FIELD]
    assert repair_provenance["mode"] == REPAIR_MODE
    assert repair_provenance["input_hashes"][
        "baseline_plan_sha256"
    ] == _sha256_document(documents["baseline_plan"])
    assert (
        repaired["provenance"]["output_confidence_basis"]
        == POLICY_FALLBACK_CONFIDENCE_BASIS
    )
    assert audit["summary"] == {
        "status": "REPAIRED",
        "baseline_assignment_count": 2,
        "repairable_group_count": 1,
        "candidate_part_count": 1,
        "changed_count": 1,
        "no_op": False,
        "exact_cover": True,
        "all_materials_in_whitelist": True,
        "maximum_orchestrator_retry_count": 1,
    }
    assert audit["output_plan_sha256"] == _sha256_document(plan)
    assert audit["changes"][0]["part_id"] == "P0001"
    assert audit["skip_reason_counts"]["BASELINE_TIER_NOT_NEUTRAL_DEFAULT"] == 1
    assert (
        validate_policy_fallback_authorization(
            plan,
            documents["registry"],
            include_policy_fallback=True,
        )
        == 2
    )


def test_mutable_repair_rejects_high_confidence_unconfirmed_material() -> None:
    documents = _documents()
    documents["group_materials"]["selections"][0].update(
        {"confirmed": False, "confidence": 0.90}
    )

    plan, audit = _build(documents)

    assert plan["assignments"][0] == documents["baseline_plan"]["assignments"][0]
    assert audit["changes"] == []
    assert audit["provisional_material_candidate_group_ids"] == []
    assert (
        audit["skip_reason_counts"][
            "MATERIAL_SELECTION_REQUIRES_EXACT_MDL_VISUAL_TOURNAMENT"
        ]
        == 2
    )


def test_immutable_repair_can_use_unconfirmed_material_as_tournament_seed() -> None:
    documents = _documents()
    documents["group_materials"]["selections"][0].update(
        {"confirmed": False, "confidence": 0.90}
    )
    documents["allow_parameter_writes"] = False

    plan, audit = _build(documents)

    repaired = plan["assignments"][0]
    assert repaired["material_id"] == GREEN
    assert repaired["status"] == "policy_fallback"
    assert repaired["confidence"] == 0.0
    assert repaired["provenance"]["reason_codes"] == [
        "QA_MISSING_CANONICAL_GROUP_MULTI_VIEW",
        "QA_TRUSTED_PART_GROUP_LOCALIZATION",
        "QA_HIGH_CONFIDENCE_WHITELIST_MATERIAL_CANDIDATE",
        "QA_POST_RENDER_VALIDATION_REQUIRED",
    ]
    assert repaired["provenance"]["material_selection_basis"] == (
        "high_confidence_whitelist_candidate_pending_render_qa"
    )
    assert audit["changes"][0]["material_selection_basis"] == (
        "high_confidence_whitelist_candidate_pending_render_qa"
    )
    assert audit["provisional_material_candidate_group_ids"] == ["G01"]


def _make_source_visual_preserve(
    documents: dict[str, dict], *, part_index: int, material_path: str
) -> None:
    part = documents["registry"]["parts"][part_index]
    assignment = documents["baseline_plan"]["assignments"][part_index]
    part["existing_visual_material"] = material_path
    assignment.update(
        {
            "apply_action": SOURCE_VISUAL_PRESERVE_ACTION,
            "source_visual_material_prim_path": material_path,
            "source_visual_material_binding_sha256": (
                source_visual_binding_sha256(
                    part_id=part["part_id"],
                    prim_path=part["prim_path"],
                    material_prim_path=material_path,
                )
            ),
            "provenance": {
                "tier": SOURCE_VISUAL_PRESERVE_TIER,
                "reason_codes": [
                    "SOURCE_VISUAL_MATERIAL_PRESENT",
                    "SOURCE_VISUAL_BINDING_HASH_BOUND",
                    "PRESERVE_SOURCE_VISUAL_NOOP",
                ],
                "output_confidence_basis": POLICY_FALLBACK_CONFIDENCE_BASIS,
                "sources": [],
            },
        }
    )
    registry_hash = _sha256_document(documents["registry"])
    documents["baseline_plan"]["provenance"]["registry_sha256"] = registry_hash
    documents["baseline_policy_audit"]["input_hashes"] = copy.deepcopy(
        documents["baseline_plan"]["provenance"]
    )
    documents["baseline_policy_audit"]["output_plan_sha256"] = _sha256_document(
        documents["baseline_plan"]
    )


def test_strong_qa_repair_overrides_source_visual_preserve_contract() -> None:
    documents = _documents()
    _make_source_visual_preserve(
        documents,
        part_index=0,
        material_path="/Asset/Looks/SourceGray",
    )

    plan, audit = _build(documents)

    repaired = plan["assignments"][0]
    assert repaired["material_id"] == GREEN
    assert repaired["provenance"]["tier"] == "qa_repair_candidate"
    assert repaired["provenance"]["baseline_tier"] == SOURCE_VISUAL_PRESERVE_TIER
    assert "apply_action" not in repaired
    assert "source_visual_material_prim_path" not in repaired
    assert "source_visual_material_binding_sha256" not in repaired
    assert audit["summary"]["changed_count"] == 1


def test_source_visual_preserve_without_strong_qa_override_remains_exact_noop() -> None:
    documents = _documents()
    _make_source_visual_preserve(
        documents,
        part_index=1,
        material_path="/Asset/Looks/SourceBlue",
    )
    documents["spatial_report"]["parts"][1]["observations"] = []
    documents["spatial_report"]["parts"][1]["resolved_support_counts"] = {}
    documents["spatial_report"] = _seal_spatial(documents["spatial_report"])
    baseline = copy.deepcopy(documents["baseline_plan"]["assignments"][1])

    plan, audit = _build(documents)

    assert plan["assignments"][1] == baseline
    assert plan["assignments"][1]["apply_action"] == SOURCE_VISUAL_PRESERVE_ACTION
    assert audit["summary"]["changed_count"] == 1


def test_two_stable_spatial_views_can_repair_the_same_qa_group_deficit() -> None:
    documents = _documents()
    for observation in documents["spatial_report"]["parts"][0]["observations"]:
        observation.update(
            {
                "registration_label_stable": True,
                "perturbation_label_stable": True,
            }
        )
    documents["spatial_report"] = _seal_spatial(documents["spatial_report"])
    documents["spatial_gate_audit"]["decisions"] = [
        documents["spatial_gate_audit"]["decisions"][1]
    ]
    documents["mapping_consensus"]["decisions"][0].update(
        {
            "output_group_id": "G99",
            "output_status": "review",
            "conflicting_view_ids": ["ref_a"],
        }
    )

    plan, audit = _build(documents)

    repaired = plan["assignments"][0]
    assert repaired["material_id"] == GREEN
    assert audit["summary"]["changed_count"] == 1
    assert audit["localization_lanes"] == [
        {
            "part_id": "P0001",
            "canonical_group_id": "G01",
            "lane": "stable_spatial_multiview",
        }
    ]


def test_one_stable_view_plus_boundary_diagnostic_can_repair() -> None:
    documents = _documents()
    documents["spatial_report"]["policy"].update(
        {
            "minimum_visible_pixels": 256,
            "minimum_diagnostic_visible_pixels": 128,
            "minimum_diagnostic_resolved_samples": 3,
            "minimum_diagnostic_consensus_ratio": 0.75,
        }
    )
    stable, diagnostic = documents["spatial_report"]["parts"][0]["observations"]
    stable.update(
        {
            "registration_label_stable": True,
            "perturbation_label_stable": True,
        }
    )
    diagnostic.update(
        {
            "classification": "conflict",
            "reason_code": "projection_perturbation_material_instability",
            "declared_visible_pixels": 300,
            "projected_part_pixels": 280,
            "registration_label_stable": True,
            "perturbation_label_stable": False,
            "small_part_diagnostic": {
                "status": "resolved",
                "reason_codes": [],
                "canonical_group_id": "G01",
                "bbox_canonical_group_id": "G01",
                "registration_label_stable": True,
                "resolved_sample_count": 5,
                "target_sample_count": 5,
                "consensus_ratio": 1.0,
                "alternative_canonical_group_ids": [],
            },
        }
    )
    documents["spatial_report"] = _seal_spatial(documents["spatial_report"])
    documents["spatial_gate_audit"]["decisions"] = [
        documents["spatial_gate_audit"]["decisions"][1]
    ]
    documents["mapping_consensus"]["decisions"][0].update(
        {
            "output_group_id": "G99",
            "output_status": "review",
            "conflicting_view_ids": ["ref_a"],
        }
    )

    plan, audit = _build(documents)

    assert plan["assignments"][0]["material_id"] == GREEN
    assert audit["localization_lanes"][0]["lane"] == "bounded_spatial_multiview"


def test_two_diagnostic_views_require_the_same_qa_deficit_group() -> None:
    documents = _documents()
    documents["spatial_report"]["policy"].update(
        {
            "minimum_visible_pixels": 256,
            "minimum_diagnostic_visible_pixels": 128,
            "minimum_diagnostic_resolved_samples": 3,
            "minimum_diagnostic_consensus_ratio": 0.75,
        }
    )
    for observation in documents["spatial_report"]["parts"][0]["observations"]:
        observation.update(
            {
                "classification": "insufficient_visibility",
                "reason_code": "part_visible_pixels_below_floor",
                "declared_visible_pixels": 220,
                "projected_part_pixels": 180,
                "registration_label_stable": None,
                "perturbation_label_stable": None,
                "small_part_diagnostic": {
                    "status": "resolved",
                    "reason_codes": [],
                    "canonical_group_id": "G01",
                    "bbox_canonical_group_id": "G01",
                    "registration_label_stable": True,
                    "resolved_sample_count": 4,
                    "target_sample_count": 3,
                    "consensus_ratio": 0.75,
                    "alternative_canonical_group_ids": ["G02"],
                },
            }
        )
    documents["spatial_report"] = _seal_spatial(documents["spatial_report"])
    documents["spatial_gate_audit"]["decisions"] = [
        documents["spatial_gate_audit"]["decisions"][1]
    ]
    documents["mapping_consensus"]["decisions"][0].update(
        {
            "output_group_id": "G99",
            "output_status": "review",
            "conflicting_view_ids": ["ref_a"],
        }
    )

    plan, audit = _build(documents)

    assert plan["assignments"][0]["material_id"] == GREEN
    assert audit["localization_lanes"][0]["lane"] == "bounded_spatial_multiview"

    documents = _documents()
    documents["spatial_report"]["policy"].update(
        {
            "minimum_visible_pixels": 256,
            "minimum_diagnostic_visible_pixels": 128,
            "minimum_diagnostic_resolved_samples": 3,
            "minimum_diagnostic_consensus_ratio": 0.75,
        }
    )
    for observation in documents["spatial_report"]["parts"][0]["observations"]:
        observation.update(
            {
                "classification": "insufficient_visibility",
                "declared_visible_pixels": 220,
                "projected_part_pixels": 180,
                "small_part_diagnostic": {
                    "status": "resolved",
                    "reason_codes": [],
                    "canonical_group_id": "G01",
                    "bbox_canonical_group_id": "G01",
                    "registration_label_stable": True,
                    "resolved_sample_count": 4,
                    "target_sample_count": 3,
                    "consensus_ratio": 0.75,
                    "alternative_canonical_group_ids": ["G02"],
                },
            }
        )
    for view in documents["quality_report"]["views"]:
        view["material_color"]["trusted_evidence_group_recall"]["groups"][0][
            "recall"
        ] = 1.0
    documents["spatial_report"] = _seal_spatial(documents["spatial_report"])
    documents["spatial_gate_audit"]["decisions"] = [
        documents["spatial_gate_audit"]["decisions"][1]
    ]
    documents["mapping_consensus"]["decisions"][0]["output_status"] = "review"

    plan, audit = _build(documents)

    assert plan == documents["baseline_plan"]
    assert audit["summary"]["candidate_part_count"] == 0
    assert audit["reason_codes"] == ["NO_REPAIRABLE_CANONICAL_GROUP"]


def test_exact_single_qa_view_without_independent_anchor_is_a_safe_noop() -> None:
    documents = _documents()
    documents["quality_report"]["views"].pop()
    documents["quality_report"]["aggregate"]["comparable_view_count"] = 1
    documents["spatial_report"]["policy"].update(
        {
            "minimum_visible_pixels": 256,
            "minimum_color_margin": 0.2,
        }
    )
    observation = documents["spatial_report"]["parts"][0]["observations"][0]
    observation.update(
        {
            "declared_visible_pixels": 400,
            "projected_part_pixels": 350,
            "color_margin": 0.5,
        }
    )
    documents["spatial_report"] = _seal_spatial(documents["spatial_report"])

    plan, audit = _build(documents)

    assert plan == documents["baseline_plan"]
    assert audit["summary"]["changed_count"] == 0
    assert (
        audit["skip_reason_counts"][
            "SINGLE_VIEW_REQUIRES_DOMINANT_OR_INDEPENDENT_ANCHOR"
        ]
        == 1
    )


def test_single_qa_view_with_independent_spatial_anchor_repairs() -> None:
    documents = _documents()
    documents["quality_report"]["views"].pop()
    documents["quality_report"]["aggregate"]["comparable_view_count"] = 1
    documents["spatial_report"]["policy"].update(
        {
            "minimum_visible_pixels": 256,
            "minimum_color_margin": 0.2,
        }
    )
    part = documents["spatial_report"]["parts"][0]
    for observation in part["observations"]:
        observation.update(
            {
                "declared_visible_pixels": 400,
                "projected_part_pixels": 350,
                "color_margin": 0.5,
            }
        )
    documents["spatial_report"] = _seal_spatial(documents["spatial_report"])

    plan, audit = _build(documents)

    assert plan["assignments"][0]["material_id"] == GREEN
    assert audit["summary"]["changed_count"] == 1
    assert audit["localization_lanes"] == [
        {
            "part_id": "P0001",
            "canonical_group_id": "G01",
            "lane": "exact_spatial_single_qa_view_with_spatial_anchor",
        }
    ]
    assert audit["changes"][0]["supporting_view_ids"] == ["ref_a"]
    assert audit["changes"][0]["spatial_anchor_view_ids"] == ["ref_b"]
    group = next(
        item
        for item in audit["group_diagnostics"]
        if item["canonical_group_id"] == "G01"
    )
    assert group["repairable"] is False
    assert group["single_view_spatial_repairable"] is True


def test_single_qa_view_lane_rejects_a_matched_semantic_conflict() -> None:
    documents = _documents()
    documents["quality_report"]["views"].pop()
    documents["spatial_report"]["policy"].update(
        {
            "minimum_visible_pixels": 256,
            "minimum_color_margin": 0.2,
        }
    )
    part = documents["spatial_report"]["parts"][0]
    part["observations"][0].update(
        {
            "declared_visible_pixels": 400,
            "projected_part_pixels": 350,
            "color_margin": 0.5,
        }
    )
    part["semantic_votes"] = [
        {
            "view_id": "ref_a",
            "alignment_trusted": True,
            "unique_canonical_join": True,
            "status": "matched",
            "canonical_group_id": "G99",
            "effective_confidence": 0.95,
        }
    ]
    documents["spatial_report"] = _seal_spatial(documents["spatial_report"])

    plan, audit = _build(documents)

    assert plan == documents["baseline_plan"]
    assert (
        audit["skip_reason_counts"]["SINGLE_VIEW_MATCHED_SEMANTIC_GROUP_CONFLICT"] == 1
    )


def _configure_single_view_semantic_override(
    documents: dict[str, dict],
) -> None:
    _enable_dominant_only_deficits(documents)
    documents["quality_report"]["views"].pop()
    documents["quality_report"]["aggregate"]["comparable_view_count"] = 1
    documents["spatial_report"]["policy"].update(
        {
            "minimum_visible_pixels": 256,
            "minimum_color_margin": 0.2,
        }
    )
    observation = documents["spatial_report"]["parts"][0]["observations"][0]
    observation.update(
        {
            "declared_visible_pixels": 1400,
            "projected_part_pixels": 1300,
            "color_margin": 0.72,
            "group_scores": [
                {"canonical_group_id": "G01", "color_share": 0.76},
                {"canonical_group_id": "G99", "color_share": 0.01},
            ],
            "bbox_canonical_group_id": "G01",
            "bbox_color_margin": 0.66,
            "bbox_group_scores": [
                {"canonical_group_id": "G01", "color_share": 0.71},
                {"canonical_group_id": "G99", "color_share": 0.02},
            ],
            "projection_perturbations": [
                {
                    "offset_pixels": list(offset),
                    "canonical_group_id": "G01",
                    "diagnostic_canonical_group_id": "G01",
                    "sampled_reference_pixels": 1300,
                    "best_color_share": 0.74,
                    "color_margin": 0.68,
                }
                for offset in ((-2, 0), (2, 0), (0, -2), (0, 2))
            ],
        }
    )
    documents["spatial_report"]["parts"][0]["semantic_votes"] = [
        {
            "view_id": "ref_a",
            "alignment_trusted": True,
            "unique_canonical_join": True,
            "status": "matched",
            "canonical_group_id": "G99",
            "effective_confidence": 0.95,
        },
        {
            "view_id": "ref_b",
            "alignment_trusted": True,
            "unique_canonical_join": True,
            "status": "matched",
            "canonical_group_id": "G01",
            "effective_confidence": 0.96,
        },
    ]
    documents["spatial_report"]["view_alignments"] = [
        {
            "reference_view_id": "ref_a",
            "selected_render_view_id": "front",
            "score": 0.9,
            "projection_iou": 0.88,
            "ecc_status": "success",
            "ecc_correlation": 0.91,
            "trusted": True,
        }
    ]
    documents["spatial_report"] = _seal_spatial(documents["spatial_report"])


def test_exact_single_view_pixels_override_same_view_semantic_localization_error() -> (
    None
):
    documents = _documents()
    _configure_single_view_semantic_override(documents)

    plan, audit = _build(documents)

    assert plan["assignments"][0]["material_id"] == GREEN
    assert audit["changes"][0]["semantic_conflict_override_view_ids"] == ["ref_a"]
    assert audit["localization_lanes"][0] == {
        "part_id": "P0001",
        "canonical_group_id": "G01",
        "lane": "exact_spatial_single_qa_view_with_semantic_anchor",
    }


@pytest.mark.parametrize(
    "mutate",
    [
        lambda documents: documents["spatial_report"]["parts"][0].update(
            {
                "semantic_votes": documents["spatial_report"]["parts"][0][
                    "semantic_votes"
                ][:1]
            }
        ),
        lambda documents: documents["spatial_report"]["parts"][0]["observations"][0][
            "group_scores"
        ][1].update({"color_share": 0.051}),
        lambda documents: documents["spatial_report"]["parts"][0]["observations"][
            0
        ].update({"projected_part_pixels": 1023}),
        lambda documents: documents["spatial_report"]["view_alignments"][0].update(
            {"projection_iou": 0.799}
        ),
        lambda documents: documents["spatial_report"]["parts"][0]["observations"][0][
            "projection_perturbations"
        ][0].update({"best_color_share": 0.699}),
        lambda documents: documents["spatial_report"]["parts"][0][
            "semantic_votes"
        ].append(
            {
                "view_id": "ref_b",
                "alignment_trusted": True,
                "unique_canonical_join": True,
                "status": "matched",
                "canonical_group_id": "G98",
                "effective_confidence": 0.95,
            }
        ),
    ],
)
def test_semantic_override_lane_fails_closed_when_strict_evidence_is_missing(
    mutate,
) -> None:
    documents = _documents()
    _configure_single_view_semantic_override(documents)
    mutate(documents)
    documents["spatial_report"] = _seal_spatial(documents["spatial_report"])

    plan, audit = _build(documents)

    assert plan == documents["baseline_plan"]
    assert (
        audit["skip_reason_counts"]["SINGLE_VIEW_MATCHED_SEMANTIC_GROUP_CONFLICT"] == 1
    )


def test_semantic_override_lane_does_not_override_spatial_gate_conflict() -> None:
    documents = _documents()
    _configure_single_view_semantic_override(documents)
    decision = documents["spatial_gate_audit"]["decisions"][0]
    decision["conflicting_view_ids"] = ["ref_a"]

    plan, audit = _build(documents)

    assert plan == documents["baseline_plan"]
    assert audit["skip_reason_counts"]["SPATIAL_OR_SEMANTIC_CONFLICT"] == 1


def _configure_anchored_single_direct_projection(
    documents: dict[str, dict],
) -> None:
    documents["spatial_report"]["policy"].update(
        {
            "minimum_visible_pixels": 256,
            "minimum_color_margin": 0.2,
        }
    )
    # P0001 remains the independently localized two-view anchor. P0002 has
    # exactly one high-purity direct projection and no visibility in ref_b.
    documents["spatial_report"]["parts"][1]["observations"] = [
        {
            "reference_view_id": "ref_a",
            "classification": "resolved",
            "canonical_group_id": "G01",
            "bbox_canonical_group_id": "G01",
            "registration_label_stable": True,
            "perturbation_label_stable": True,
            "declared_visible_pixels": 480,
            "projected_part_pixels": 420,
            "color_margin": 0.81,
            "group_scores": [
                {
                    "canonical_group_id": "G01",
                    "color_share": 0.91,
                }
            ],
            "projection_perturbations": [
                {
                    "offset_pixels": list(offset),
                    "canonical_group_id": "G01",
                    "diagnostic_canonical_group_id": "G01",
                    "sampled_reference_pixels": 420,
                    "best_color_share": 0.89,
                    "color_margin": 0.78,
                }
                for offset in ((-2, 0), (2, 0), (0, -2), (0, 2))
            ],
        },
        {
            "reference_view_id": "ref_b",
            "classification": "insufficient_visibility",
            "declared_visible_pixels": 0,
        },
    ]
    second_assignment = documents["baseline_plan"]["assignments"][1]
    second_assignment.update(
        {
            "semantic": "neutral unresolved component",
            "provenance": {
                "tier": "neutral_default",
                "reason_codes": ["POLICY_DECLARED_NEUTRAL_DEFAULT"],
                "output_confidence_basis": POLICY_FALLBACK_CONFIDENCE_BASIS,
                "sources": [],
            },
        }
    )
    documents["baseline_policy_audit"]["output_plan_sha256"] = _sha256_document(
        documents["baseline_plan"]
    )
    documents["spatial_report"] = _seal_spatial(documents["spatial_report"])


def test_one_direct_view_uses_an_existing_complete_multiview_anchor() -> None:
    documents = _documents()
    _configure_anchored_single_direct_projection(documents)

    plan, audit = _build(documents)

    assert [item["part_id"] for item in audit["changes"]] == ["P0001", "P0002"]
    assert plan["assignments"][1]["material_id"] == GREEN
    change = audit["changes"][1]
    assert change["supporting_view_ids"] == ["ref_a"]
    assert change["anchor_part_ids"] == ["P0001"]
    assert change["anchor_supporting_view_ids"] == ["ref_a", "ref_b"]
    assert audit["localization_lanes"][1] == {
        "part_id": "P0002",
        "canonical_group_id": "G01",
        "lane": "exact_spatial_single_view_with_multiview_anchor",
    }


def test_anchored_lane_overrides_one_disproven_same_view_semantic_vote() -> None:
    documents = _documents()
    _enable_dominant_only_deficits(documents)
    _configure_anchored_single_direct_projection(documents)
    observation = documents["spatial_report"]["parts"][1]["observations"][0]
    observation.update(
        {
            "color_margin": 0.72,
            "group_scores": [
                {"canonical_group_id": "G01", "color_share": 0.76},
                {"canonical_group_id": "G99", "color_share": 0.01},
            ],
            "bbox_canonical_group_id": "G01",
            "bbox_color_margin": 0.66,
            "bbox_group_scores": [
                {"canonical_group_id": "G01", "color_share": 0.71},
                {"canonical_group_id": "G99", "color_share": 0.02},
            ],
        }
    )
    for perturbation in observation["projection_perturbations"]:
        perturbation.update(
            {
                "sampled_reference_pixels": 1100,
                "best_color_share": 0.74,
                "color_margin": 0.68,
            }
        )
    observation["declared_visible_pixels"] = 1200
    observation["projected_part_pixels"] = 1100
    documents["spatial_report"]["parts"][1]["semantic_votes"] = [
        {
            "view_id": "ref_a",
            "alignment_trusted": True,
            "unique_canonical_join": True,
            "status": "matched",
            "canonical_group_id": "G99",
            "effective_confidence": 0.95,
        },
        {
            "view_id": "ref_b",
            "alignment_trusted": True,
            "unique_canonical_join": True,
            "status": "matched",
            "canonical_group_id": "G01",
            "effective_confidence": 0.96,
        },
    ]
    documents["spatial_report"]["view_alignments"] = [
        {
            "reference_view_id": "ref_a",
            "selected_render_view_id": "front",
            "score": 0.9,
            "projection_iou": 0.88,
            "ecc_status": "success",
            "ecc_correlation": 0.91,
            "trusted": True,
        }
    ]
    documents["spatial_report"] = _seal_spatial(documents["spatial_report"])

    plan, audit = _build(documents)

    assert plan["assignments"][1]["material_id"] == GREEN
    change = next(item for item in audit["changes"] if item["part_id"] == "P0002")
    assert change["semantic_conflict_override_view_ids"] == ["ref_a"]
    lane = next(
        item for item in audit["localization_lanes"] if item["part_id"] == "P0002"
    )
    assert lane["lane"] == "exact_spatial_single_view_with_multiview_anchor"


def test_anchored_single_view_is_noop_without_an_accepted_anchor() -> None:
    documents = _documents()
    _configure_anchored_single_direct_projection(documents)
    documents["geometry_risk"]["parts"][0]["risk"]["multi_material_risk"] = True

    plan, audit = _build(documents)

    assert plan == documents["baseline_plan"]
    assert audit["summary"]["changed_count"] == 0
    assert audit["skip_reason_counts"]["MULTIVIEW_ANCHOR_PROPOSAL_MISSING"] == 1


def test_anchored_single_view_rejects_diagnostic_only_localization() -> None:
    documents = _documents()
    _configure_anchored_single_direct_projection(documents)
    observation = documents["spatial_report"]["parts"][1]["observations"][0]
    observation.update(
        {
            "classification": "insufficient_visibility",
            "small_part_diagnostic": {
                "status": "resolved",
                "reason_codes": [],
                "canonical_group_id": "G01",
                "bbox_canonical_group_id": "G01",
                "registration_label_stable": True,
                "resolved_sample_count": 6,
                "target_sample_count": 6,
                "consensus_ratio": 1.0,
                "alternative_canonical_group_ids": [],
            },
        }
    )
    documents["spatial_report"] = _seal_spatial(documents["spatial_report"])

    _, audit = _build(documents)

    assert [item["part_id"] for item in audit["changes"]] == ["P0001"]
    assert (
        audit["skip_reason_counts"][
            "ANCHORED_SINGLE_REQUIRES_EXACTLY_ONE_DIRECT_SUPPORT"
        ]
        == 1
    )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda observation: observation.update({"projected_part_pixels": 255}),
        lambda observation: observation["group_scores"][0].update(
            {"color_share": 0.849}
        ),
        lambda observation: observation.update({"color_margin": 0.699}),
        lambda observation: observation.update({"bbox_canonical_group_id": "G99"}),
        lambda observation: observation["projection_perturbations"][0].update(
            {"canonical_group_id": "G99"}
        ),
        lambda observation: observation["projection_perturbations"][0].update(
            {"diagnostic_canonical_group_id": "G99"}
        ),
        lambda observation: observation["projection_perturbations"][0].update(
            {"sampled_reference_pixels": 255}
        ),
        lambda observation: observation["projection_perturbations"][0].update(
            {"best_color_share": 0.799}
        ),
        lambda observation: observation["projection_perturbations"][0].update(
            {"color_margin": 0.699}
        ),
        lambda observation: observation["projection_perturbations"][0].update(
            {"sampled_reference_pixels": True}
        ),
        lambda observation: observation["projection_perturbations"][0].update(
            {"best_color_share": True}
        ),
        lambda observation: observation["projection_perturbations"][0].update(
            {"color_margin": True}
        ),
    ],
)
def test_anchored_single_view_enforces_direct_projection_purity(mutate) -> None:
    documents = _documents()
    _configure_anchored_single_direct_projection(documents)
    observation = documents["spatial_report"]["parts"][1]["observations"][0]
    mutate(observation)
    documents["spatial_report"] = _seal_spatial(documents["spatial_report"])

    _, audit = _build(documents)

    assert [item["part_id"] for item in audit["changes"]] == ["P0001"]
    assert (
        audit["skip_reason_counts"][
            "ANCHORED_SINGLE_REQUIRES_EXACTLY_ONE_DIRECT_SUPPORT"
        ]
        == 1
    )


def _configure_two_bounded_diagnostics(documents: dict[str, dict]) -> None:
    documents["spatial_report"]["policy"].update(
        {
            "minimum_visible_pixels": 256,
            "minimum_diagnostic_visible_pixels": 128,
            "minimum_diagnostic_resolved_samples": 3,
            "minimum_diagnostic_consensus_ratio": 0.75,
        }
    )
    for observation in documents["spatial_report"]["parts"][0]["observations"]:
        observation.update(
            {
                "classification": "insufficient_visibility",
                "declared_visible_pixels": 220,
                "projected_part_pixels": 180,
                "registration_label_stable": None,
                "perturbation_label_stable": None,
                "small_part_diagnostic": {
                    "status": "resolved",
                    "reason_codes": [],
                    "canonical_group_id": "G01",
                    "bbox_canonical_group_id": "G01",
                    "registration_label_stable": True,
                    "resolved_sample_count": 4,
                    "target_sample_count": 4,
                    "consensus_ratio": 1.0,
                    "alternative_canonical_group_ids": [],
                },
            }
        )
    documents["spatial_gate_audit"]["decisions"] = [
        documents["spatial_gate_audit"]["decisions"][1]
    ]
    documents["mapping_consensus"]["decisions"][0]["output_status"] = "review"


def test_two_independent_diagnostics_override_one_semantic_disagreement() -> None:
    documents = _documents()
    _configure_two_bounded_diagnostics(documents)
    documents["spatial_report"]["parts"][0]["semantic_votes"] = [
        {
            "view_id": "ref_b",
            "alignment_trusted": True,
            "unique_canonical_join": True,
            "status": "matched",
            "canonical_group_id": "G99",
            "effective_confidence": 0.95,
        }
    ]
    documents["spatial_report"] = _seal_spatial(documents["spatial_report"])

    plan, audit = _build(documents)

    assert plan["assignments"][0]["material_id"] == GREEN
    assert audit["changes"][0]["semantic_conflict_override_view_ids"] == ["ref_b"]
    assert audit["localization_lanes"][0]["lane"] == "bounded_spatial_multiview"


def test_two_view_semantic_alternative_remains_a_hard_conflict() -> None:
    documents = _documents()
    _configure_two_bounded_diagnostics(documents)
    documents["spatial_report"]["parts"][0]["semantic_votes"] = [
        {
            "view_id": view_id,
            "alignment_trusted": True,
            "unique_canonical_join": True,
            "status": "matched",
            "canonical_group_id": "G99",
            "effective_confidence": 0.95,
        }
        for view_id in ("ref_a", "ref_b")
    ]
    documents["spatial_report"] = _seal_spatial(documents["spatial_report"])

    plan, audit = _build(documents)

    assert plan == documents["baseline_plan"]
    assert audit["skip_reason_counts"]["HIGH_CONFIDENCE_SEMANTIC_GROUP_CONFLICT"] == 1


def test_invalid_diagnostic_policy_fails_closed_without_zero_division() -> None:
    documents = _documents()
    _configure_two_bounded_diagnostics(documents)
    documents["spatial_report"]["policy"]["minimum_diagnostic_resolved_samples"] = 0
    for observation in documents["spatial_report"]["parts"][0]["observations"]:
        diagnostic = observation["small_part_diagnostic"]
        diagnostic["resolved_sample_count"] = 0
        diagnostic["target_sample_count"] = 0
        diagnostic["consensus_ratio"] = 0.0
    documents["spatial_report"] = _seal_spatial(documents["spatial_report"])

    plan, audit = _build(documents)

    assert plan == documents["baseline_plan"]
    assert audit["summary"]["changed_count"] == 0
    assert audit["skip_reason_counts"]["NO_BOUNDED_SPATIAL_DIAGNOSTIC"] == 1


def test_repair_parameterizes_by_group_when_groups_share_a_preset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    documents = _documents()
    documents["whitelist"]["material_ids"].append(GENERIC_STEEL_PAINTED)
    documents["palette_fusion"]["canonical_palette"]["groups"].append(
        {
            "group_id": "G02",
            "family_hint": "metal",
            "base_color": "green",
            "finish_hint": "painted",
            "visual_description": "second green painted panel",
            "distinct_view_count": 2,
            "singleton": False,
            "source_view_ids": ["ref_a", "ref_b"],
        }
    )
    documents["group_materials"]["selections"].append(
        {
            "group_id": "G02",
            "material_id": GREEN,
            "confidence": 0.96,
            "confirmed": True,
        }
    )
    evidence = {
        "schema_version": "qwen-mvinverse-pbr-evidence/v1",
        "groups": [
            {
                "group_id": group_id,
                "surface_class": "dielectric",
                "contributing_view_ids": ["ref_a", "ref_b"],
                "metallic": {"median": 0.1},
                "suggestion": {
                    "decision": "auto",
                    "auto_parameter_eligible": True,
                    "base_color_srgb": [0.2, 0.5, 0.1],
                    "metallic": 0.0,
                    "roughness": 0.43,
                },
            }
            for group_id in ("G01", "G02")
        ],
    }
    documents["mvinverse_pbr_evidence"] = evidence
    documents["baseline_plan"]["provenance"]["whitelist_sha256"] = _sha256_document(
        documents["whitelist"]
    )
    documents["baseline_policy_audit"]["input_hashes"] = copy.deepcopy(
        documents["baseline_plan"]["provenance"]
    )
    documents["baseline_policy_audit"]["output_plan_sha256"] = _sha256_document(
        documents["baseline_plan"]
    )
    monkeypatch.setattr(
        policy_module,
        "validate_mvinverse_evidence",
        lambda document: copy.deepcopy(document),
    )

    plan, audit = _build(documents)

    repaired = plan["assignments"][0]
    assert repaired["material_id"] == GREEN
    assert repaired["parameters"]["paint_roughness"] == pytest.approx(0.43)
    assert repaired["provenance"]["mvinverse"]["group_id"] == "G01"
    assert audit["mvinverse"]["parameterized_part_ids"] == ["P0001"]
    assert audit["changes"][0]["confirmed_source_material_id"] == GREEN

    locked_plan, locked_audit = build_quality_repair_plan(
        **documents,
        allow_parameter_writes=False,
    )
    locked_repaired = locked_plan["assignments"][0]
    assert locked_repaired["material_id"] == GREEN
    assert "parameters" not in locked_repaired
    assert "mvinverse" not in locked_repaired["provenance"]
    assert locked_audit["summary"]["selected_mdl_library_defaults_required"] is True
    assert locked_audit["mvinverse"]["parameter_writes_allowed"] is False
    assert locked_audit["mvinverse"]["parameterized_part_ids"] == []


@pytest.mark.parametrize(
    ("mutate", "expected_reason"),
    [
        (
            lambda documents: documents["quality_report"]["views"].pop(),
            "MISSING_IN_FEWER_THAN_TWO_TRUSTED_VIEWS",
        ),
        (
            lambda documents: documents["spatial_report"]["reference_evidence"][
                1
            ].update({"content_cluster_id": "CONTENT_01"}),
            "REFERENCE_CONTENT_NOT_DISTINCT",
        ),
        (
            lambda documents: (
                documents["spatial_report"]["reference_evidence"][1].update(
                    {
                        "pose_cluster_id": "front",
                        "selected_render_view_id": "front",
                    }
                )
                or documents["quality_report"]["views"][1].update(
                    {"render_view_id": "front"}
                )
                or documents["quality_report"]["views"][1]["mapping"].update(
                    {"selected_render_view_id": "front"}
                )
            ),
            "CAD_POSE_NOT_DISTINCT",
        ),
    ],
)
def test_requires_two_content_and_pose_distinct_missing_views(
    mutate, expected_reason: str
) -> None:
    documents = _documents()
    mutate(documents)
    documents["spatial_report"] = _seal_spatial(documents["spatial_report"])

    plan, audit = _build(documents)

    assert plan == documents["baseline_plan"]
    assert audit["summary"]["changed_count"] == 0
    group = next(
        item
        for item in audit["group_diagnostics"]
        if item["canonical_group_id"] == "G01"
    )
    assert expected_reason in group["reason_codes"]


@pytest.mark.parametrize(
    ("mutate", "expected_reason"),
    [
        (
            lambda documents: documents["geometry_risk"]["parts"][0]["risk"].update(
                {"multi_material_risk": True}
            ),
            "GEOMETRY_MULTI_MATERIAL_RISK",
        ),
        (
            lambda documents: (
                documents["baseline_plan"]["assignments"][0].update(
                    {
                        "status": "auto",
                        "confidence": 0.95,
                        "provenance": {"tier": "gate_auto", "sources": []},
                    }
                )
                or documents["baseline_policy_audit"].update(
                    {"output_plan_sha256": _sha256_document(documents["baseline_plan"])}
                )
            ),
            "BASELINE_STATUS_NOT_POLICY_FALLBACK",
        ),
        (
            lambda documents: documents["spatial_gate_audit"]["decisions"][0][
                "semantic_conflicting_view_ids"
            ].append("ref_b"),
            "SPATIAL_OR_SEMANTIC_CONFLICT",
        ),
        (
            lambda documents: documents["mapping_consensus"]["decisions"][0].update(
                {
                    "output_group_id": "G99",
                    "output_status": "matched",
                    "conflicting_view_ids": ["ref_b"],
                }
            ),
            "MATCHED_MAPPING_CONSENSUS_GROUP_CONFLICT",
        ),
        (
            lambda documents: documents["group_materials"]["selections"][0].update(
                {"confirmed": False, "confidence": 0.84}
            ),
            "MATERIAL_SELECTION_NOT_CONFIRMED",
        ),
        (
            lambda documents: documents["group_materials"]["selections"][0].update(
                {"material_id": "mdl:not-whitelisted"}
            ),
            "CONFIRMED_MATERIAL_NOT_WHITELISTED",
        ),
    ],
)
def test_skips_unsafe_or_unconfirmed_part_repairs(mutate, expected_reason: str) -> None:
    documents = _documents()
    mutate(documents)

    plan, audit = _build(documents)

    assert plan == documents["baseline_plan"]
    assert audit["summary"]["changed_count"] == 0
    assert expected_reason in audit["skip_reason_counts"]


@pytest.mark.parametrize(
    ("field", "value", "expected_reason"),
    [
        (
            "face_subsets",
            [
                {
                    "subset_name": "Existing",
                    "material_id": GREEN,
                    "face_indices": [0],
                }
            ],
            "BASELINE_HAS_FACE_SUBSETS",
        ),
        (
            "parameters",
            {"paint_roughness": 0.3},
            "BASELINE_HAS_PARAMETERS",
        ),
    ],
)
def test_neutral_fallback_with_existing_surface_delta_is_safe_noop(
    field: str, value: object, expected_reason: str
) -> None:
    documents = _documents()
    documents["baseline_plan"]["assignments"][0][field] = value
    documents["baseline_policy_audit"]["output_plan_sha256"] = _sha256_document(
        documents["baseline_plan"]
    )

    plan, audit = _build(documents)

    assert plan == documents["baseline_plan"]
    assert audit["summary"]["changed_count"] == 0
    assert expected_reason in audit["skip_reason_counts"]


def test_authoritative_canonical_group_lineage_is_never_repaired() -> None:
    documents = _documents()
    documents["baseline_plan"]["assignments"][0]["provenance"][
        "canonical_group_id"
    ] = "G01"
    documents["baseline_policy_audit"]["output_plan_sha256"] = _sha256_document(
        documents["baseline_plan"]
    )

    plan, audit = _build(documents)

    assert plan == documents["baseline_plan"]
    assert audit["summary"]["changed_count"] == 0
    assert audit["skip_reason_counts"][
        AUTHORITATIVE_CANONICAL_GROUP_LOCK_REASON
    ] == 1


def test_missing_part_localization_is_safe_noop() -> None:
    documents = _documents()
    documents["spatial_report"]["parts"][0]["observations"].pop()
    documents["spatial_report"]["parts"][0]["resolved_support_counts"] = {"G01": 1}
    documents["spatial_report"] = _seal_spatial(documents["spatial_report"])

    plan, audit = _build(documents)

    assert plan == documents["baseline_plan"]
    assert audit["summary"]["changed_count"] == 0
    assert (
        audit["skip_reason_counts"]["PART_LOCALIZED_IN_FEWER_THAN_TWO_REFERENCE_VIEWS"]
        == 1
    )


def test_old_spatial_schema_without_content_pose_is_safe_noop() -> None:
    documents = _documents()
    documents["spatial_report"].pop("reference_evidence")
    documents["spatial_report"] = _seal_spatial(documents["spatial_report"])

    plan, audit = _build(documents)

    assert plan == documents["baseline_plan"]
    assert audit["summary"]["status"] == "SAFE_NOOP"
    assert audit["summary"]["changed_count"] == 0
    assert audit["reason_codes"] == ["NO_REPAIRABLE_CANONICAL_GROUP"]


def test_rejects_stale_baseline_policy_hash() -> None:
    documents = _documents()
    documents["baseline_policy_audit"]["output_plan_sha256"] = "0" * 64

    with pytest.raises(QualityRepairError, match="output hash"):
        _build(documents)


def test_cli_writes_plan_and_audit_without_overwriting(tmp_path: Path) -> None:
    documents = _documents()
    arguments: list[str] = []
    for name, value in documents.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        arguments.extend([f"--{name.replace('_', '-')}", str(path)])
    plan_path = tmp_path / "repair-plan.json"
    audit_path = tmp_path / "repair-audit.json"
    arguments.extend(["--output-plan", str(plan_path), "--audit", str(audit_path)])

    assert main(arguments) == 0
    assert (
        json.loads(audit_path.read_text(encoding="utf-8"))["summary"]["changed_count"]
        == 1
    )
    written_plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert written_plan["provenance"]["mode"] == POLICY_EXACT_COVER_MODE
    assert written_plan["provenance"][REPAIR_PROVENANCE_FIELD]["mode"] == REPAIR_MODE

    with pytest.raises(QualityRepairError, match="refusing to overwrite"):
        main(arguments)
