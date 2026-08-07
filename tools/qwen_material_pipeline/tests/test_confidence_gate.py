from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import qwen_material_pipeline.evidence.confidence as confidence_module
from qwen_material_pipeline.evidence.confidence import (
    ConfidenceGateError,
    GatePolicy,
    VIEW_EVIDENCE_SCHEMA_VERSION,
    evaluate_confidence_gate,
    main,
)


def _registry() -> dict:
    return {
        "schema_version": "qwen-material-parts/v1",
        "part_count": 3,
        "parts": [
            {
                "part_id": "P0001",
                "renders": [
                    {"view_id": "front", "visible_pixels": 500},
                    {"view_id": "iso", "visible_pixels": 420},
                ],
            },
            {"part_id": "P0002", "renders": []},
            {
                "part_id": "P0003",
                "renders": [
                    {"view_id": "front", "visible_pixels": 300},
                    {"view_id": "rear", "visible_pixels": 280},
                ],
            },
        ],
    }


def _geometry_risk(
    *,
    risk_part_id: str | None = None,
    advisory_part_id: str | None = None,
    part_ids: tuple[str, ...] = ("P0001", "P0002", "P0003"),
) -> dict:
    parts = []
    reason_counts = {
        "multiple_welded_topology_components": 0,
        "high_raw_topology_component_count": 0,
        "high_surface_patch_count": 0,
    }
    risk_part_ids = []
    for part_id in part_ids:
        at_risk = part_id == risk_part_id
        advisory = part_id == advisory_part_id
        reasons = (
            ["multiple_welded_topology_components"]
            if at_risk
            else ["high_surface_patch_count"]
            if advisory
            else []
        )
        if at_risk:
            risk_part_ids.append(part_id)
        for reason in reasons:
            reason_counts[reason] += 1
        parts.append(
            {
                "part_id": part_id,
                "prim_path": f"/World/{part_id}",
                "face_count": 10,
                "metrics": {
                    "raw_topology_component_count": 1,
                    "welded_topology_component_count": 2 if at_risk else 1,
                    "surface_patch_count": 128 if advisory else 1,
                },
                "risk": {
                    "multi_material_risk": at_risk,
                    "basis": "conservative_topology_complexity_proxy",
                },
                "reason_codes": reasons,
            }
        )
    return {
        "schema_version": "qwen-geometry-uniform-material-risk/v1",
        "asset_usd": "/tmp/fixture.usd",
        "asset_sha256": "a" * 64,
        "source_usd_unchanged": True,
        "face_region_manifest": None,
        "face_region_manifest_sha256": None,
        "rendered_registry": None,
        "rendered_registry_sha256": None,
        "policy": {
            "maximum_welded_topology_component_count": 1,
            "raw_topology_component_risk_threshold": 512,
            "surface_patch_risk_threshold": 128,
        },
        "part_count": len(parts),
        "parts": parts,
        "summary": {
            "part_count": len(parts),
            "face_count": 10 * len(parts),
            "multi_material_risk_part_count": len(risk_part_ids),
            "no_detected_multi_material_risk_part_count": len(parts)
            - len(risk_part_ids),
            "multi_material_risk_part_ids": risk_part_ids,
            "reason_code_counts": reason_counts,
        },
        "limitations": ["Topology complexity is only a conservative proxy."],
    }


def _assignment(
    part_id: str,
    material_id: str,
    *,
    confidence: float,
    status: str,
    evidence_views: list[str],
) -> dict:
    return {
        "part_id": part_id,
        "material_id": material_id,
        "semantic": f"semantic for {part_id}",
        "confidence": confidence,
        "evidence_views": evidence_views,
        "status": status,
    }


def _staged() -> dict:
    return {
        "schema_version": "qwen-staged-material-result/v1",
        "material_plan": {
            "schema_version": "1.0",
            "assignments": [
                _assignment(
                    "P0001",
                    "MAT_A",
                    confidence=0.96,
                    status="auto",
                    evidence_views=["ref_front", "ref_iso"],
                ),
                _assignment(
                    "P0003",
                    "MAT_B",
                    confidence=0.80,
                    status="review",
                    evidence_views=["ref_front"],
                ),
            ],
        },
        "unknown_parts": [{"part_id": "P0002", "reason_code": "no_cad_render"}],
        "audit": {},
    }


def _batches() -> list[dict]:
    return [
        {
            "schema_version": "qwen-part-palette-map/v1",
            "batch_id": "B01",
            "mappings": [
                {
                    "part_id": "P0001",
                    "group_id": "G01",
                    "mapping_confidence": 0.96,
                    "evidence_view_id": "ref_front",
                    "evidence_box_index": 0,
                    "status": "matched",
                    "reason_code": "direct_visual_match",
                },
                {
                    "part_id": "P0003",
                    "group_id": "G02",
                    "mapping_confidence": 0.75,
                    "evidence_view_id": "ref_front",
                    "evidence_box_index": 0,
                    "status": "review",
                    "reason_code": "partial_visibility",
                },
            ],
        }
    ]


def _retrieval(material_id: str, *, normalized_margin: float = 0.20) -> dict:
    top_score = 5.0
    runner_up_score = top_score - normalized_margin * top_score
    return {
        "strategy": "deterministic_metadata_score_v1",
        "pool_count": 4,
        "eligible_pool_count": 4,
        "family_pool_available": True,
        "family_pool_used": True,
        "limit": 4,
        "top_score": top_score,
        "runner_up_score": runner_up_score,
        "score_margin": top_score - runner_up_score,
        "normalized_margin": normalized_margin,
        "margin_available": True,
        "ranking": [
            {
                "rank": 1,
                "material_id": material_id,
                "score": top_score,
                "matched_fields": ["family", "color"],
            },
            {
                "rank": 2,
                "material_id": material_id + "_RUNNER",
                "score": runner_up_score,
                "matched_fields": ["family"],
            },
        ],
    }


def _choice(material_id: str, *, normalized_margin: float = 0.20) -> dict:
    return {
        "retrieval_audit": _retrieval(material_id, normalized_margin=normalized_margin),
        "chosen_retrieval_rank": 1,
        "model_choice_matches_retrieval_top": True,
        "forward": {"material_id": material_id, "confidence": 0.95},
        "reverse": {"material_id": material_id, "confidence": 0.93},
        "confirmed": True,
    }


def _semantic_choice(
    material_id: str,
    *,
    intrinsic_surface_ambiguity: bool = False,
) -> dict:
    choice = _choice(material_id)
    if intrinsic_surface_ambiguity:
        selection_group = {
            "group_id": "G01",
            "family_hint": "metal",
            "base_color": "orange",
            "finish_hint": "other",
            "visual_description": (
                "orange metal appearance region; finer physical material identity "
                "is unresolved in palette metadata"
            ),
            "boxes": [[0, 0, 1000, 1000]],
            "confidence": 0.60,
        }
        reliability = {
            "policy": {
                "automatic_confidence_threshold": 0.85,
                "review_confidence_threshold": 0.60,
                "minimum_independent_support_views": 2,
                "unresolved_values": ["other", "unknown"],
            },
            "finish_hint": {
                "canonical_value": "painted",
                "selection_value": "other",
                "reliable": False,
                "supporting_view_ids": ["front"],
                "conflicting_view_ids": [],
                "conflicting_values": [],
                "maximum_support_confidence": 0.60,
                "multiview_confirmed": False,
                "high_confidence_confirmed": False,
            },
            "visual_description": {
                "canonical_value": "orange painted guard",
                "selection_value": selection_group["visual_description"],
                "reliable": False,
                "canonical_confidence": 0.60,
                "requires_reliable_finish": True,
            },
            "selection_context_modified": True,
            "canonical_group_preserved": True,
            "reason_codes": [
                "single_review_confidence_finish_source",
                "visual_description_below_auto_confidence",
                "visual_description_depends_on_unreliable_finish",
            ],
        }
    else:
        selection_group = {
            "group_id": "G01",
            "family_hint": "metal",
            "base_color": "black",
            "finish_hint": "painted",
            "visual_description": "black painted housing",
            "boxes": [[0, 0, 1000, 1000]],
            "confidence": 0.90,
            "material_selection_objective": "visual_similarity",
        }
        reliability = {
            "policy": {
                "automatic_confidence_threshold": 0.85,
                "review_confidence_threshold": 0.60,
                "minimum_independent_support_views": 2,
                "unresolved_values": ["other", "unknown"],
            },
            "finish_hint": {
                "canonical_value": "painted",
                "selection_value": "painted",
                "reliable": True,
                "supporting_view_ids": ["front"],
                "conflicting_view_ids": [],
                "conflicting_values": [],
                "maximum_support_confidence": 0.90,
                "multiview_confirmed": False,
                "high_confidence_confirmed": True,
            },
            "visual_description": {
                "canonical_value": "black painted housing",
                "selection_value": "black painted housing",
                "reliable": True,
                "canonical_confidence": 0.90,
                "requires_reliable_finish": True,
            },
            "selection_context_modified": False,
            "canonical_group_preserved": True,
            "reason_codes": ["finish_confirmed_by_high_confidence_source"],
        }
    choice["selection_group"] = selection_group
    choice["semantic_reliability"] = reliability
    choice["retrieval_audit"].update(
        {
            "strategy": "family_gated_semantic_mvinverse_similarity_score/v5",
            "semantic_reliability": copy.deepcopy(reliability),
            "finish_evidence_used": not intrinsic_surface_ambiguity,
            "description_evidence_used": not intrinsic_surface_ambiguity,
            "intrinsic_surface_ambiguity": intrinsic_surface_ambiguity,
        }
    )
    return choice


def _material_audit() -> dict:
    return {"G01": _choice("MAT_A"), "G02": _choice("MAT_B")}


def _consistent_view_evidence() -> dict:
    return {
        "schema_version": VIEW_EVIDENCE_SCHEMA_VERSION,
        "predictions": [
            {
                "part_id": "P0001",
                "view_id": view_id,
                "material_id": "MAT_A",
                "confidence": confidence,
                "candidate_margin": 0.20,
            }
            for view_id, confidence in (("ref_front", 0.96), ("ref_iso", 0.94))
        ],
    }


def _spatial_gate_audit(*, confidence: float = 0.86) -> dict:
    return {
        "schema_version": "qwen-spatial-mapping-gate/v1",
        "policy": {
            "minimum_semantic_confidence": 0.85,
            "minimum_semantic_conflict_confidence": 0.60,
        },
        "decisions": [
            {
                "part_id": "P0001",
                "batch_id": "B01",
                "input_group_id": "G01",
                "input_status": "matched",
                "input_confidence": confidence,
                "output_group_id": "G01",
                "output_status": "matched",
                "output_confidence": confidence,
                "decision": "kept_auto",
                "reason_codes": [
                    "independent_validation_met",
                    "semantic_multiview_validation_met",
                    "no_material_conflicts",
                ],
                "supporting_view_ids": [],
                "conflicting_view_ids": [],
                "semantic_supporting_view_ids": ["ref_front", "ref_iso"],
                "semantic_supporting_reference_sha256s": ["a" * 64, "b" * 64],
                "semantic_supporting_pixel_sha256s": ["c" * 64, "d" * 64],
                "semantic_supporting_content_cluster_ids": ["PH01", "PH02"],
                "semantic_supporting_pose_cluster_ids": ["front", "iso"],
                "semantic_conflicting_view_ids": [],
                "semantic_unresolved_view_ids": [],
                "semantic_multi_material_view_ids": [],
                "semantic_nondeterministic_content_cluster_ids": [],
                "validation_lanes": ["semantic_multiview"],
            }
        ],
        "summary": {},
    }


def _review_plan() -> dict:
    return {
        "schema_version": "1.0",
        "assignments": [
            {"part_id": "P0001", "material_id": "MAT_A", "status": "approved"},
            {"part_id": "P0003", "material_id": "MAT_B", "status": "approved"},
        ],
    }


def _evaluate(**overrides: object) -> dict:
    inputs = {
        "staged_result": _staged(),
        "rendered_registry": _registry(),
        "review_plan": _review_plan(),
        "batches": _batches(),
        "material_choice_audit": _material_audit(),
        "view_evidence": _consistent_view_evidence(),
    }
    inputs.update(overrides)
    return evaluate_confidence_gate(**inputs)


def _by_id(report: dict, part_id: str) -> dict:
    return next(item for item in report["decisions"] if item["part_id"] == part_id)


def test_strict_gate_emits_all_three_decisions_and_auto_plan() -> None:
    report = _evaluate()

    assert report["summary"] == {
        "part_count": 3,
        "auto_count": 1,
        "review_count": 1,
        "preserve_count": 1,
        "fail_closed": True,
        "legacy_flow_modified": False,
    }
    assert _by_id(report, "P0001")["decision"] == "auto"
    assert _by_id(report, "P0002")["decision"] == "preserve"
    assert _by_id(report, "P0003")["decision"] == "review"
    assert [
        item["part_id"] for item in report["auto_material_plan"]["assignments"]
    ] == ["P0001"]


def test_independently_validated_mapping_uses_evidence_bound_threshold() -> None:
    staged = _staged()
    staged["material_plan"]["assignments"][0]["confidence"] = 0.86
    batches = _batches()
    batches[0]["mappings"][0]["mapping_confidence"] = 0.86
    view_evidence = _consistent_view_evidence()
    for prediction, confidence in zip(view_evidence["predictions"], (0.86, 0.87)):
        prediction["confidence"] = confidence

    without_validation = _evaluate(
        staged_result=staged,
        batches=batches,
        view_evidence=view_evidence,
    )
    with_validation = _evaluate(
        staged_result=staged,
        batches=batches,
        view_evidence=view_evidence,
        independent_validation_audit=_spatial_gate_audit(),
    )

    assert _by_id(without_validation, "P0001")["decision"] == "review"
    decision = _by_id(with_validation, "P0001")
    assert decision["decision"] == "auto"
    assert decision["independent_validation"] == {
        "batch_id": "B01",
        "group_id": "G01",
        "confidence": 0.86,
        "threshold_profile": "independently_validated",
        "auto_confidence_threshold": 0.85,
        "validation_lanes": ["semantic_multiview"],
        "supporting_view_ids": ["ref_front", "ref_iso"],
    }
    assert decision["threshold_profile"] == "independently_validated"
    assert decision["active_auto_thresholds"] == {
        "model_confidence": 0.85,
        "mapping_confidence": 0.85,
    }


def test_independent_validation_rejects_duplicate_non_kept_decision() -> None:
    audit = _spatial_gate_audit(confidence=0.96)
    downgraded = copy.deepcopy(audit["decisions"][0])
    downgraded["decision"] = "downgraded_review"
    downgraded["output_status"] = "review"
    audit["decisions"].insert(0, downgraded)

    with pytest.raises(ConfidenceGateError, match="duplicate part_id"):
        _evaluate(independent_validation_audit=audit)


def test_independent_validation_conflict_is_rejected() -> None:
    audit = _spatial_gate_audit(confidence=0.96)
    audit["decisions"][0]["semantic_conflicting_view_ids"] = ["ref_top"]

    with pytest.raises(ConfidenceGateError, match="material conflict"):
        _evaluate(independent_validation_audit=audit)


def test_independent_validation_rejects_weak_conflict_detection_policy() -> None:
    audit = _spatial_gate_audit(confidence=0.96)
    audit["policy"]["minimum_semantic_conflict_confidence"] = 0.61

    with pytest.raises(ConfidenceGateError, match="unsafe semantic thresholds"):
        _evaluate(independent_validation_audit=audit)


def test_missing_candidate_margin_is_never_auto() -> None:
    audit = _material_audit()
    for key in (
        "retrieval_audit",
        "chosen_retrieval_rank",
        "model_choice_matches_retrieval_top",
    ):
        audit["G01"].pop(key)

    decision = _by_id(_evaluate(material_choice_audit=audit), "P0001")

    assert decision["decision"] == "review"
    assert "CANDIDATE_MARGIN_UNAVAILABLE" in decision["reason_codes"]
    assert decision["candidate_margin"] is None


def test_low_normalized_candidate_margin_downgrades_to_review() -> None:
    audit = _material_audit()
    audit["G01"] = _choice("MAT_A", normalized_margin=0.149)

    decision = _by_id(_evaluate(material_choice_audit=audit), "P0001")

    assert decision["decision"] == "review"
    assert decision["candidate_margin"] == pytest.approx(0.149)
    assert "CANDIDATE_MARGIN_BELOW_AUTO" in decision["reason_codes"]


def test_derived_selection_confidence_overrides_qwen_numeric_anchor() -> None:
    audit = _material_audit()
    choice = _choice("MAT_A", normalized_margin=0.20)
    choice["forward"]["confidence"] = 0.0
    choice["reverse"]["confidence"] = 0.0
    choice.update(
        {
            "confirmation_basis": "exact_forward_reverse_agreement",
            "confirmed_material_id": "MAT_A",
            "selection_confidence": 0.90,
            "confidence_derivation": {
                "schema_version": (
                    "qwen-derived-material-selection-confidence/v1"
                ),
                "derived_confidence": 0.90,
                "confirmation_basis": "exact_forward_reverse_agreement",
                "reported_forward_confidence": 0.0,
                "reported_reverse_confidence": 0.0,
                "reported_confidence_is_authoritative": False,
            },
        }
    )
    audit["G01"] = choice

    decision = _by_id(_evaluate(material_choice_audit=audit), "P0001")

    assert decision["decision"] == "auto"
    assert "MATERIAL_CHOICE_BELOW_REVIEW" not in decision["reason_codes"]


def test_model_choice_outside_retrieval_top_can_never_be_auto() -> None:
    audit = _material_audit()
    retrieval = audit["G01"]["retrieval_audit"]
    retrieval["ranking"][0]["material_id"] = "MAT_RETRIEVAL_TOP"
    retrieval["ranking"][1]["material_id"] = "MAT_A"
    audit["G01"]["chosen_retrieval_rank"] = 2
    audit["G01"]["model_choice_matches_retrieval_top"] = False

    decision = _by_id(_evaluate(material_choice_audit=audit), "P0001")

    assert decision["decision"] == "review"
    assert "RETRIEVAL_TOP_DISAGREEMENT" in decision["reason_codes"]
    assert "CANDIDATE_MARGIN_UNAVAILABLE" in decision["reason_codes"]


def _visual_retrieval_choice() -> dict:
    choice = _semantic_choice("MAT_A")
    fallback = choice["retrieval_audit"]
    top_score = round(1.0 / 62.0 + 1.2 / 61.0, 10)
    runner_up_score = round(1.0 / 61.0, 10)
    choice.update(
        {
            "retrieval_audit": {
                "strategy": "siglip2_full_catalog_plus_dinov2_masked_rrf/v1",
                "group_id": "G01",
                "pool_count": 4,
                "eligible_pool_count": 4,
                "full_catalog_indexed": True,
                "final_authority": "exact_mdl_render_tournament",
                "fallback_audit": fallback,
                "limit": 4,
                "top_score": top_score,
                "runner_up_score": runner_up_score,
                "score_margin": top_score - runner_up_score,
                "normalized_margin": (
                    (top_score - runner_up_score) / abs(top_score)
                ),
                "margin_available": True,
                "ranking": [
                    {
                        "rank": 1,
                        "material_id": "MAT_VISUAL_TOP",
                        "score": top_score,
                        "matched_fields": [
                            "siglip2_catalog_wide_visual",
                            "dinov2_masked_dense_texture",
                        ],
                        "siglip2_rank": 2,
                        "siglip2_score": 0.76,
                        "dino_rank": 1,
                        "dino_score": 0.27,
                    },
                    {
                        "rank": 2,
                        "material_id": "MAT_A",
                        "score": runner_up_score,
                        "matched_fields": ["siglip2_catalog_wide_visual"],
                        "siglip2_rank": 1,
                        "siglip2_score": 0.77,
                        "dino_rank": None,
                        "dino_score": None,
                    },
                ],
                "fixed_library_defaults_required": True,
            },
            "chosen_retrieval_rank": 2,
            "model_choice_matches_retrieval_top": False,
        }
    )
    return choice


def _base_bank_visual_retrieval_choice() -> dict:
    choice = _visual_retrieval_choice()
    retrieval = choice["retrieval_audit"]
    retrieval["strategy"] = (
        "base_observation_bank_siglip2_dinov2_color_mvinverse_rrf/v1"
    )
    first, second = retrieval["ranking"]
    first.update(
        {
            "matched_fields": [
                "siglip2_base_bank_rig_visual",
                "dinov2_base_bank_surface_texture",
                "masked_color_appearance",
                "mvinverse_authored_pbr_prior",
            ],
            "color_rank": 1,
            "color_score": 0.91,
            "mvinverse_rank": 1,
            "mvinverse_score": 0.88,
        }
    )
    second.update(
        {
            "matched_fields": [
                "siglip2_base_bank_rig_visual",
                "masked_color_appearance",
            ],
            "color_rank": 2,
            "color_score": 0.82,
            "mvinverse_rank": None,
            "mvinverse_score": None,
        }
    )
    first["score"] = round(
        1.0 / 62 + 1.2 / 61 + 0.8 / 61 + 0.2 / 61,
        10,
    )
    second["score"] = round(1.0 / 61 + 0.8 / 62, 10)
    retrieval["top_score"] = first["score"]
    retrieval["runner_up_score"] = second["score"]
    retrieval["score_margin"] = first["score"] - second["score"]
    retrieval["normalized_margin"] = (
        retrieval["score_margin"] / retrieval["top_score"]
    )
    return choice


def test_visual_retrieval_wrapper_and_semantic_fallback_are_both_validated() -> None:
    audit = _material_audit()
    audit["G01"] = _visual_retrieval_choice()

    decision = _by_id(_evaluate(material_choice_audit=audit), "P0001")

    assert decision["decision"] == "review"
    assert "RETRIEVAL_TOP_DISAGREEMENT" in decision["reason_codes"]
    retrieval = decision["material_choice"]["retrieval_audit"]
    assert retrieval["strategy"] == (
        "siglip2_full_catalog_plus_dinov2_masked_rrf/v1"
    )
    assert retrieval["full_catalog_indexed"] is True
    assert retrieval["fallback_audit"]["strategy"] == (
        "family_gated_semantic_mvinverse_similarity_score/v5"
    )
    assert retrieval["normalized_margin"] == pytest.approx(
        (retrieval["top_score"] - retrieval["runner_up_score"])
        / retrieval["top_score"]
    )


def test_base_bank_visual_retrieval_is_validated_by_confidence_gate() -> None:
    audit = _material_audit()
    audit["G01"] = _base_bank_visual_retrieval_choice()

    decision = _by_id(_evaluate(material_choice_audit=audit), "P0001")

    assert decision["material_choice"]["retrieval_audit"]["strategy"] == (
        "base_observation_bank_siglip2_dinov2_color_mvinverse_rrf/v1"
    )


@pytest.mark.parametrize(
    "mutation, message",
    [
        (
            lambda retrieval: retrieval.update(full_catalog_indexed=False),
            "must index the full catalog",
        ),
        (
            lambda retrieval: retrieval.update(final_authority="qwen"),
            "final authority is unsupported",
        ),
        (
            lambda retrieval: retrieval["ranking"][0].update(dino_score=None),
            "DINO rank and score must be supplied together",
        ),
        (
            lambda retrieval: retrieval["ranking"][0].update(score=0.5),
            "RRF score is inconsistent",
        ),
        (
            lambda retrieval: retrieval.update(
                pool_count=999,
                eligible_pool_count=999,
            ),
            "catalog bounds do not match",
        ),
        (
            lambda retrieval: retrieval["ranking"][1].update(
                siglip2_rank=5,
                score=round(1.0 / 65.0, 10),
            ),
            "within the indexed catalog",
        ),
    ],
)
def test_visual_retrieval_wrapper_fails_closed(mutation, message: str) -> None:
    audit = _material_audit()
    audit["G01"] = _visual_retrieval_choice()
    mutation(audit["G01"]["retrieval_audit"])

    with pytest.raises(ConfidenceGateError, match=message):
        _evaluate(material_choice_audit=audit)


def test_retrieval_raw_scores_may_exceed_one_but_normalized_margin_must_match() -> None:
    audit = _material_audit()
    audit["G01"]["retrieval_audit"]["top_score"] = 50.0
    audit["G01"]["retrieval_audit"]["runner_up_score"] = 40.0
    audit["G01"]["retrieval_audit"]["score_margin"] = 10.0
    audit["G01"]["retrieval_audit"]["normalized_margin"] = 0.2
    audit["G01"]["retrieval_audit"]["ranking"][0]["score"] = 50.0
    audit["G01"]["retrieval_audit"]["ranking"][1]["score"] = 40.0

    assert _by_id(_evaluate(material_choice_audit=audit), "P0001")["decision"] == "auto"

    audit["G01"]["retrieval_audit"]["normalized_margin"] = 0.3
    with pytest.raises(ConfidenceGateError, match="normalized_margin is inconsistent"):
        _evaluate(material_choice_audit=audit)


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda r: r["ranking"][1].update(rank=3), "ordered, and contiguous"),
        (
            lambda r: r["ranking"][1].update(material_id="MAT_A"),
            "duplicate material_id",
        ),
        (
            lambda r: r["ranking"][0].update(matched_fields=["family", "family"]),
            "duplicate",
        ),
        (
            lambda r: r["ranking"][0].update(matched_fields=["typical_use"]),
            "unknown matched_fields",
        ),
    ],
)
def test_retrieval_ranking_is_strict(mutation, message: str) -> None:
    audit = _material_audit()
    mutation(audit["G01"]["retrieval_audit"])
    with pytest.raises(ConfidenceGateError, match=message):
        _evaluate(material_choice_audit=audit)


def test_retrieval_accepts_v7_coating_and_tunable_template_matches() -> None:
    audit = _material_audit()
    matched_fields = audit["G01"]["retrieval_audit"]["ranking"][0]["matched_fields"]
    matched_fields.extend(["coating_surface_family", "mvinverse_tunable_template"])

    decision = _by_id(_evaluate(material_choice_audit=audit), "P0001")

    assert decision["decision"] == "auto"
    parsed = decision["material_choice"]["retrieval_audit"]["ranking"][0]
    assert "coating_surface_family" in parsed["matched_fields"]
    assert "mvinverse_tunable_template" in parsed["matched_fields"]


@pytest.mark.parametrize("ambiguous", [False, True])
def test_semantic_retrieval_context_is_strictly_validated(ambiguous: bool) -> None:
    audit = _material_audit()
    audit["G01"] = _semantic_choice("MAT_A", intrinsic_surface_ambiguity=ambiguous)

    decision = _by_id(_evaluate(material_choice_audit=audit), "P0001")

    assert decision["decision"] == "auto"
    retrieval = decision["material_choice"]["retrieval_audit"]
    assert retrieval["intrinsic_surface_ambiguity"] is ambiguous
    assert retrieval["finish_evidence_used"] is not ambiguous
    assert retrieval["description_evidence_used"] is not ambiguous
    assert retrieval["semantic_reliability"] == audit["G01"]["semantic_reliability"]


def test_v6_coating_audit_and_mvinverse_paint_resolution_are_validated() -> None:
    satin = "mdl:Miscellaneous/Paint_Satin.mdl#Paint_Satin"
    matte = "mdl:Miscellaneous/Paint_Matte.mdl#Paint_Matte"
    choice = _semantic_choice(satin)
    retrieval = choice["retrieval_audit"]
    retrieval.update(
        {
            "strategy": "family_gated_semantic_mvinverse_similarity_score/v6",
            "pre_duplicate_alias_dedup_count": 5,
            "duplicate_alias_dedup_count": 1,
            "paint_pool_available": True,
            "paint_pool_used": True,
            "mvinverse_surface_class": "dielectric",
            "observed_metallic": 0.1,
            "applied_coating_confirmed": True,
            "applied_coating_plausible": False,
        }
    )
    retrieval["eligible_pool_count"] = 4
    retrieval["pool_count"] = 6
    retrieval["ranking"][0]["matched_fields"].append("confirmed_applied_coating")
    choice["reverse"]["material_id"] = matte
    choice["confirmed"] = True
    choice["confirmation_basis"] = "mvinverse_resolved_base_paint_finish"
    choice["confirmed_material_id"] = satin

    parsed = confidence_module._material_choices({"G01": choice})["G01"]

    assert parsed["confirmed"] is True
    assert parsed["resolved_material_id"] == satin
    assert parsed["confirmation_basis"] == "mvinverse_resolved_base_paint_finish"
    assert parsed["retrieval_audit"]["paint_pool_used"] is True


def test_sam3_mask_unavailable_choice_is_strict_unconfirmed_preserve() -> None:
    choice = _choice("MAT_A")
    choice.update(
        {
            "forward": {"material_id": "MAT_A", "confidence": 0.0},
            "reverse": {"material_id": "MAT_A", "confidence": 0.0},
            "confirmed": False,
            "confirmation_basis": "sam3_mask_unavailable_fail_closed",
            "confirmed_material_id": None,
            "selection_confidence": 0.0,
            "confidence_derivation": {
                "schema_version": (
                    "qwen-derived-material-selection-confidence/v1"
                ),
                "derived_confidence": 0.0,
                "confirmation_basis": "sam3_mask_unavailable_fail_closed",
                "reported_forward_confidence": 0.0,
                "reported_reverse_confidence": 0.0,
                "reported_confidence_is_authoritative": False,
            },
            "physics_consistency_resolution": {
                "applied": False,
                "mode": "immutable_selected_mdl_preserved",
                "original_material_id": "MAT_A",
                "resolved_material_id": "MAT_A",
                "selected_mdl_parameters_mutable": False,
            },
            "independent_view_choices": [],
        }
    )

    parsed = confidence_module._material_choices({"G01": choice})["G01"]

    assert parsed["confirmed"] is False
    assert parsed["resolved_material_id"] == "MAT_A"
    assert parsed["selection_confidence"] == 0.0

    unsafe = copy.deepcopy(choice)
    unsafe["forward"]["confidence"] = 0.01
    with pytest.raises(ConfidenceGateError, match="resolution contract"):
        confidence_module._material_choices({"G01": unsafe})


def _v7_balanced_dark_surface_choice() -> dict:
    anodized = "mdl:Metals/Aluminum_Anodized_Black.mdl#Aluminum_Anodized_Black"
    blued = "mdl:Metals/Steel_Blued.mdl#Steel_Blued"
    paint = "mdl:Miscellaneous/Paint_Satin.mdl#Paint_Satin"
    carbon = "mdl:Metals/Steel_Carbon.mdl#Steel_Carbon"
    choice = _semantic_choice(anodized)
    retrieval = choice["retrieval_audit"]
    ranking = [
        {
            "rank": 1,
            "material_id": anodized,
            "score": 10.0,
            "matched_fields": [
                "family",
                "color",
                "plausible_applied_coating",
                "mvinverse_metallicity_class",
                "multiview_albedo_color",
            ],
        },
        {
            "rank": 2,
            "material_id": blued,
            "score": 9.0,
            "matched_fields": [
                "family",
                "plausible_applied_coating",
                "mvinverse_metallicity_class",
                "multiview_albedo_color",
            ],
        },
        {
            "rank": 3,
            "material_id": carbon,
            "score": 8.0,
            "matched_fields": [
                "family",
                "mvinverse_metallicity_class",
                "multiview_albedo_color",
            ],
        },
        {
            "rank": 4,
            "material_id": paint,
            "score": 7.0,
            "matched_fields": [
                "plausible_applied_coating",
                "multiview_albedo_color",
                "mvinverse_roughness_class",
            ],
        },
    ]
    albedo = [0.04, 0.05, 0.04]
    retrieval.update(
        {
            "strategy": "family_gated_semantic_mvinverse_similarity_score/v7",
            "pool_count": 6,
            "eligible_pool_count": 5,
            "pre_duplicate_alias_dedup_count": 5,
            "duplicate_alias_dedup_count": 0,
            "mvinverse_tunable_equivalence_dedup_count": 1,
            "paint_pool_available": True,
            "paint_pool_used": False,
            "mvinverse_surface_class": "dielectric",
            "observed_metallic": 0.68,
            "applied_coating_confirmed": False,
            "applied_coating_plausible": True,
            "surface_interpretation_policy": {
                "mode": "balanced_dark_metal_surface_interpretations",
                "active": True,
                "family_reliable": True,
                "semantic_surface_class": "coating",
                "semantic_numeric_conflict": True,
                "multi_view_albedo_reliable": True,
                "albedo_median": albedo,
                "albedo_luminance": (
                    0.2126 * albedo[0] + 0.7152 * albedo[1] + 0.0722 * albedo[2]
                ),
                "dark_multiview_color": True,
                "metallic_reliable": True,
                "metallicity_class": "conductive",
                "roughness_reliable": True,
                "observed_roughness": 0.36,
                "roughness_class": "satin",
                "required_interpretations": [
                    "conversion_coating",
                    "applied_paint",
                    "bare_metal",
                ],
                "available_interpretation_counts": {
                    "conversion_coating": 2,
                    "applied_paint": 1,
                    "bare_metal": 2,
                },
                "selected_material_ids_by_interpretation": {
                    "conversion_coating": [anodized, blued],
                    "applied_paint": [paint],
                    "bare_metal": [carbon],
                },
                "complete_required_coverage": True,
            },
            "limit": 4,
            "top_score": 10.0,
            "runner_up_score": 9.0,
            "score_margin": 1.0,
            "normalized_margin": 0.1,
            "margin_available": True,
            "ranking": ranking,
        }
    )
    return choice


def test_v7_balanced_dark_surface_audit_is_strictly_validated() -> None:
    choice = _v7_balanced_dark_surface_choice()

    parsed = confidence_module._material_choices({"G01": choice})["G01"]

    policy = parsed["retrieval_audit"]["surface_interpretation_policy"]
    assert policy["active"] is True
    assert policy["semantic_numeric_conflict"] is True
    assert policy["complete_required_coverage"] is True
    assert (
        parsed["retrieval_audit"]["mvinverse_tunable_equivalence_dedup_count"]
        == 1
    )

    choice = _v7_balanced_dark_surface_choice()
    choice["retrieval_audit"]["surface_interpretation_policy"][
        "selected_material_ids_by_interpretation"
    ]["applied_paint"] = []
    with pytest.raises(ConfidenceGateError, match="balanced surface policy"):
        confidence_module._material_choices({"G01": choice})

    choice = _v7_balanced_dark_surface_choice()
    choice["retrieval_audit"]["mvinverse_tunable_equivalence_dedup_count"] = 6
    with pytest.raises(ConfidenceGateError, match="tunable-equivalence"):
        confidence_module._material_choices({"G01": choice})


def test_mutable_v7_accepts_catalog_niche_domain_policy() -> None:
    choice = _v7_balanced_dark_surface_choice()
    choice["retrieval_audit"]["niche_domain_policy"] = {
        "mode": "positive_reference_semantics_required",
        "domains": ["automotive_finish", "electronics_surface"],
    }

    parsed = confidence_module._material_choices({"G01": choice})["G01"]

    assert parsed["retrieval_audit"]["niche_domain_policy"] == {
        "mode": "positive_reference_semantics_required",
        "domains": ["automotive_finish", "electronics_surface"],
    }


def test_semantic_retrieval_fields_are_all_or_none() -> None:
    audit = _material_audit()
    audit["G01"] = _semantic_choice("MAT_A")
    audit["G01"]["retrieval_audit"].pop("finish_evidence_used")

    with pytest.raises(ConfidenceGateError, match="incomplete_semantic"):
        _evaluate(material_choice_audit=audit)


def test_retrieval_and_top_level_semantic_snapshots_must_match() -> None:
    audit = _material_audit()
    audit["G01"] = _semantic_choice("MAT_A")
    audit["G01"]["retrieval_audit"]["semantic_reliability"]["reason_codes"] = []

    with pytest.raises(ConfidenceGateError, match="does not match"):
        _evaluate(material_choice_audit=audit)


@pytest.mark.parametrize(
    "field",
    [
        "finish_evidence_used",
        "description_evidence_used",
        "intrinsic_surface_ambiguity",
    ],
)
def test_semantic_retrieval_flags_reject_non_booleans(field: str) -> None:
    audit = _material_audit()
    audit["G01"] = _semantic_choice("MAT_A")
    audit["G01"]["retrieval_audit"][field] = 1

    with pytest.raises(ConfidenceGateError, match="must be boolean"):
        _evaluate(material_choice_audit=audit)


def test_semantic_retrieval_flags_are_reconstructed_from_evidence() -> None:
    audit = _material_audit()
    audit["G01"] = _semantic_choice("MAT_A", intrinsic_surface_ambiguity=True)
    audit["G01"]["retrieval_audit"]["intrinsic_surface_ambiguity"] = False

    with pytest.raises(
        ConfidenceGateError, match="intrinsic_surface_ambiguity is inconsistent"
    ):
        _evaluate(material_choice_audit=audit)


def test_semantic_reliability_policy_and_reason_codes_are_exact() -> None:
    audit = _material_audit()
    audit["G01"] = _semantic_choice("MAT_A")
    for reliability in (
        audit["G01"]["semantic_reliability"],
        audit["G01"]["retrieval_audit"]["semantic_reliability"],
    ):
        reliability["policy"]["automatic_confidence_threshold"] = 0.84

    with pytest.raises(ConfidenceGateError, match="staged semantic policy"):
        _evaluate(material_choice_audit=audit)

    audit = _material_audit()
    audit["G01"] = _semantic_choice("MAT_A")
    for reliability in (
        audit["G01"]["semantic_reliability"],
        audit["G01"]["retrieval_audit"]["semantic_reliability"],
    ):
        reliability["reason_codes"] = []

    with pytest.raises(ConfidenceGateError, match="reason_codes are inconsistent"):
        _evaluate(material_choice_audit=audit)


def test_cross_view_material_conflict_forces_preserve() -> None:
    evidence = {
        "schema_version": VIEW_EVIDENCE_SCHEMA_VERSION,
        "predictions": [
            {
                "part_id": "P0001",
                "view_id": "ref_front",
                "material_id": "MAT_A",
                "confidence": 0.96,
                "candidate_margin": 0.2,
            },
            {
                "part_id": "P0001",
                "view_id": "ref_iso",
                "material_id": "MAT_CONFLICT",
                "confidence": 0.91,
                "candidate_margin": 0.3,
            },
        ],
    }

    decision = _by_id(_evaluate(view_evidence=evidence), "P0001")

    assert decision["decision"] == "preserve"
    assert "CROSS_VIEW_MATERIAL_CONFLICT" in decision["reason_codes"]


def test_independent_view_votes_can_supply_consistent_evidence() -> None:
    staged = _staged()
    staged["material_plan"]["assignments"][0]["evidence_views"] = []
    evidence = {
        "schema_version": VIEW_EVIDENCE_SCHEMA_VERSION,
        "predictions": [
            {
                "part_id": "P0001",
                "view_id": view,
                "material_id": "MAT_A",
                "confidence": confidence,
                "candidate_margin": 0.2,
            }
            for view, confidence in (("ref_front", 0.96), ("ref_iso", 0.94))
        ],
    }

    decision = _by_id(_evaluate(staged_result=staged, view_evidence=evidence), "P0001")

    assert decision["decision"] == "auto"
    assert decision["model"]["independent_reference_count"] == 2
    assert (
        decision["model"]["reference_evidence_source"] == "independent_view_predictions"
    )


def test_staged_reference_citations_cannot_replace_independent_predictions() -> None:
    decision = _by_id(_evaluate(view_evidence=None), "P0001")

    assert decision["decision"] == "review"
    assert decision["model"]["staged_reference_views"] == ["ref_front", "ref_iso"]
    assert decision["model"]["independent_reference_count"] == 0
    assert "INDEPENDENT_VIEW_PREDICTIONS_UNAVAILABLE" in decision["reason_codes"]


def test_one_reference_downgrades_and_no_reference_preserves() -> None:
    staged = _staged()
    staged["material_plan"]["assignments"][0]["evidence_views"] = ["ref_front"]
    decision = _by_id(_evaluate(staged_result=staged, view_evidence=None), "P0001")
    assert decision["decision"] == "review"
    assert "INSUFFICIENT_INDEPENDENT_REFERENCES" in decision["reason_codes"]

    staged["material_plan"]["assignments"][0]["evidence_views"] = []
    decision = _by_id(_evaluate(staged_result=staged, view_evidence=None), "P0001")
    assert decision["decision"] == "preserve"
    assert "NO_INDEPENDENT_REFERENCE_EVIDENCE" in decision["reason_codes"]


def test_cad_visibility_boundaries_are_fail_closed() -> None:
    registry = _registry()
    registry["parts"][0]["renders"] = [{"view_id": "front", "visible_pixels": 63}]
    decision = _by_id(_evaluate(rendered_registry=registry), "P0001")
    assert decision["decision"] == "preserve"
    assert "CAD_EVIDENCE_BELOW_REVIEW" in decision["reason_codes"]

    registry["parts"][0]["renders"] = [{"view_id": "front", "visible_pixels": 300}]
    decision = _by_id(_evaluate(rendered_registry=registry), "P0001")
    assert decision["decision"] == "review"
    assert "CAD_VIEW_COUNT_BELOW_AUTO" in decision["reason_codes"]


def test_isolated_geometry_evidence_does_not_overwrite_raw_visibility() -> None:
    registry = _registry()
    registry["parts"][0]["renders"] = [
        {"view_id": "front", "visible_pixels": 30},
        {"view_id": "iso", "visible_pixels": 24},
    ]
    registry["parts"][0]["isolated_evidence"] = {
        "schema_version": "qwen-isolated-part-evidence/v1",
        "sha256": "a" * 64,
        "selected_view_ids": ["front", "iso"],
        "source_visible_pixels_by_view": {"front": 30, "iso": 24},
        "normalized_visible_pixels_by_view": {"front": 5016, "iso": 4074},
        "source_max_visible_pixels": 30,
        "normalized_max_visible_pixels": 5016,
        "source_evidence_view_count": 2,
        "source_evidence_view_ids": ["front", "iso"],
        "source_pixel_floor": 12,
        "material_neutralized": True,
        "background_removed": True,
    }

    decision = _by_id(_evaluate(rendered_registry=registry), "P0001")

    assert decision["decision"] == "auto"
    assert decision["cad_visibility"]["evidence_mode"] == "isolated_mask_multiview"
    assert decision["cad_visibility"]["source_max_visible_pixels"] == 30
    assert decision["cad_visibility"]["max_visible_pixels"] == 5016
    assert "CAD_EVIDENCE_BELOW_REVIEW" not in decision["reason_codes"]
    assert "CAD_VIEW_COUNT_BELOW_AUTO" not in decision["reason_codes"]


def test_isolated_evidence_source_pixel_tampering_is_rejected() -> None:
    registry = _registry()
    registry["parts"][0]["isolated_evidence"] = {
        "schema_version": "qwen-isolated-part-evidence/v1",
        "sha256": "a" * 64,
        "selected_view_ids": ["front", "iso"],
        "source_visible_pixels_by_view": {"front": 501, "iso": 420},
        "normalized_visible_pixels_by_view": {"front": 5016, "iso": 4074},
        "source_max_visible_pixels": 501,
        "normalized_max_visible_pixels": 5016,
        "source_evidence_view_count": 2,
        "source_evidence_view_ids": ["front", "iso"],
        "source_pixel_floor": 12,
        "material_neutralized": True,
        "background_removed": True,
    }

    with pytest.raises(ConfidenceGateError, match="projection pixels differ"):
        _evaluate(rendered_registry=registry)


def test_missing_mapping_or_material_audit_blocks_auto() -> None:
    mapping_decision = _by_id(_evaluate(batches=None), "P0001")
    assert mapping_decision["decision"] == "review"
    assert "MAPPING_AUDIT_UNAVAILABLE" in mapping_decision["reason_codes"]

    material_decision = _by_id(_evaluate(material_choice_audit=None), "P0001")
    assert material_decision["decision"] == "review"
    assert "MATERIAL_CHOICE_AUDIT_UNAVAILABLE" in material_decision["reason_codes"]


def test_material_choice_integrity_mismatch_forces_preserve() -> None:
    audit = _material_audit()
    audit["G01"]["forward"]["material_id"] = "MAT_OTHER"
    audit["G01"]["reverse"]["material_id"] = "MAT_OTHER"
    audit["G01"]["retrieval_audit"]["ranking"][0]["material_id"] = "MAT_OTHER"
    audit["G01"]["chosen_retrieval_rank"] = 1

    decision = _by_id(_evaluate(material_choice_audit=audit), "P0001")

    assert decision["decision"] == "preserve"
    assert "MATERIAL_CHOICE_INTEGRITY_MISMATCH" in decision["reason_codes"]


def test_human_review_difference_blocks_auto_and_face_subsets_force_preserve() -> None:
    review = _review_plan()
    review["assignments"][0]["material_id"] = "MAT_HUMAN"
    decision = _by_id(_evaluate(review_plan=review), "P0001")
    assert decision["decision"] == "review"
    assert "HUMAN_REVIEW_DIFFERS_FROM_MODEL" in decision["reason_codes"]

    review = _review_plan()
    review["assignments"][0]["face_subsets"] = [
        {"subset_name": "hardware", "face_indices": [1], "material_id": "MAT_METAL"}
    ]
    decision = _by_id(_evaluate(review_plan=review), "P0001")
    assert decision["decision"] == "preserve"
    assert "MULTI_MATERIAL_RISK" in decision["reason_codes"]


def test_unknown_multi_material_part_is_preserved_even_when_highly_visible() -> None:
    staged = _staged()
    staged["unknown_parts"][0]["reason_code"] = "multi_material_mesh"
    registry = _registry()
    registry["parts"][1]["renders"] = [
        {"view_id": "front", "visible_pixels": 10000},
        {"view_id": "iso", "visible_pixels": 12000},
    ]

    decision = _by_id(
        _evaluate(staged_result=staged, rendered_registry=registry), "P0002"
    )

    assert decision["decision"] == "preserve"
    assert "MULTI_MATERIAL_RISK" in decision["reason_codes"]


def test_geometry_multi_material_risk_forces_preserve_and_is_audited() -> None:
    report = _evaluate(geometry_risk_report=_geometry_risk(risk_part_id="P0001"))
    decision = _by_id(report, "P0001")

    assert decision["decision"] == "preserve"
    assert decision["multi_material_risk"] is True
    assert "GEOMETRY_MULTI_MATERIAL_RISK" in decision["reason_codes"]
    assert decision["geometry_risk"]["risk"] == {
        "multi_material_risk": True,
        "basis": "conservative_topology_complexity_proxy",
    }
    assert decision["geometry_risk"]["reason_codes"] == [
        "multiple_welded_topology_components"
    ]
    assert report["auto_material_plan"]["assignments"] == []


def test_valid_no_risk_geometry_report_does_not_promote_or_downgrade() -> None:
    report = _evaluate(geometry_risk_report=_geometry_risk())

    assert _by_id(report, "P0001")["decision"] == "auto"
    assert (
        _by_id(report, "P0001")["geometry_risk"]["risk"]["multi_material_risk"] is False
    )
    assert _by_id(report, "P0002")["decision"] == "preserve"


def test_surface_patch_advisory_does_not_downgrade_an_auto_assignment() -> None:
    report = _evaluate(geometry_risk_report=_geometry_risk(advisory_part_id="P0001"))
    decision = _by_id(report, "P0001")

    assert decision["decision"] == "auto"
    assert decision["multi_material_risk"] is False
    assert decision["geometry_risk"]["reason_codes"] == ["high_surface_patch_count"]
    assert decision["geometry_risk"]["risk"]["multi_material_risk"] is False
    assert "GEOMETRY_MULTI_MATERIAL_RISK" not in decision["reason_codes"]


def test_geometry_risk_path_is_supported_and_contract_is_strict(tmp_path: Path) -> None:
    path = tmp_path / "geometry-risk.json"
    path.write_text(json.dumps(_geometry_risk(risk_part_id="P0001")), encoding="utf-8")
    assert (
        _by_id(_evaluate(geometry_risk_report=path), "P0001")["decision"] == "preserve"
    )

    malformed = _geometry_risk(risk_part_id="P0001")
    malformed["parts"][0]["risk"]["multi_material_risk"] = False
    with pytest.raises(ConfidenceGateError, match="validation failed"):
        _evaluate(geometry_risk_report=malformed)

    incomplete = _geometry_risk(part_ids=("P0001", "P0002"))
    with pytest.raises(ConfidenceGateError, match="does not exactly cover"):
        _evaluate(geometry_risk_report=incomplete)


def test_staged_result_must_exactly_cover_registry() -> None:
    staged = _staged()
    staged["unknown_parts"] = []
    with pytest.raises(ConfidenceGateError, match="does not exactly cover"):
        _evaluate(staged_result=staged)


def test_registry_rejects_duplicate_views_and_invalid_pixels() -> None:
    registry = _registry()
    registry["parts"][0]["renders"].append({"view_id": "front", "visible_pixels": 5})
    with pytest.raises(ConfidenceGateError, match="duplicate render view_id"):
        _evaluate(rendered_registry=registry)

    registry = _registry()
    registry["parts"][0]["renders"][0]["visible_pixels"] = True
    with pytest.raises(ConfidenceGateError, match="non-negative integer"):
        _evaluate(rendered_registry=registry)


def test_confidence_and_policy_reject_nan_and_invalid_threshold_order() -> None:
    staged = _staged()
    staged["material_plan"]["assignments"][0]["confidence"] = float("nan")
    with pytest.raises(ConfidenceGateError, match="finite number"):
        _evaluate(staged_result=staged)

    with pytest.raises(ConfidenceGateError, match="cannot exceed auto"):
        GatePolicy(review_model_confidence=0.95, auto_model_confidence=0.90).validate()


def test_view_evidence_rejects_geometry_and_duplicate_reference_votes() -> None:
    evidence = {
        "schema_version": VIEW_EVIDENCE_SCHEMA_VERSION,
        "predictions": [
            {
                "part_id": "P0001",
                "view_id": "cad_front",
                "material_id": "MAT_A",
                "confidence": 0.9,
            }
        ],
    }
    with pytest.raises(ConfidenceGateError, match="not an independent reference"):
        _evaluate(view_evidence=evidence)

    evidence["predictions"][0]["view_id"] = "CAD_front"
    with pytest.raises(ConfidenceGateError, match="not an independent reference"):
        _evaluate(view_evidence=evidence)

    evidence["predictions"][0]["view_id"] = "ref_front"
    evidence["predictions"].append(copy.deepcopy(evidence["predictions"][0]))
    with pytest.raises(ConfidenceGateError, match="duplicate view evidence"):
        _evaluate(view_evidence=evidence)


def test_cli_writes_atomic_audit_with_input_hashes(tmp_path: Path) -> None:
    paths = {}
    for name, document in {
        "staged": _staged(),
        "registry": _registry(),
        "review": _review_plan(),
        "material": _material_audit(),
        "geometry": _geometry_risk(),
    }.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        paths[name] = path
    batches_dir = tmp_path / "batches"
    batches_dir.mkdir()
    (batches_dir / "B01.json").write_text(json.dumps(_batches()[0]), encoding="utf-8")
    output = tmp_path / "gate.json"

    exit_code = main(
        [
            "--staged-result",
            str(paths["staged"]),
            "--rendered-registry",
            str(paths["registry"]),
            "--review-plan",
            str(paths["review"]),
            "--batches-dir",
            str(batches_dir),
            "--material-choice-audit",
            str(paths["material"]),
            "--geometry-risk",
            str(paths["geometry"]),
            "--output",
            str(output),
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert report["summary"]["part_count"] == 3
    assert len(report["inputs"]["staged_result"]["sha256"]) == 64
    assert len(report["inputs"]["geometry_risk"]["sha256"]) == 64
    assert report["inputs"]["batches"][0]["path"].endswith("B01.json")
    assert not (tmp_path / ".gate.json.tmp").exists()
