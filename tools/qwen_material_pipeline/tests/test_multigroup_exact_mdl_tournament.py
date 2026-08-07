from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from typing import Any

import pytest

from qwen_material_pipeline.materials.membership_tournament import (
    M0_CANDIDATE,
    SELECTION_SCHEMA_VERSION,
)
from qwen_material_pipeline.materials.multigroup_exact_mdl_tournament import (
    BASELINE_GROUP_PRESENT_WITHOUT_LOCALIZATION,
    MultigroupExactMdlTournamentError,
    ROUND_ACCEPTED,
    ROUND_FALLBACK_AMBIGUOUS_WINNER,
    ROUND_FALLBACK_BASELINE_INELIGIBLE,
    ROUND_FALLBACK_INSUFFICIENT_IMPROVEMENT,
    build_exact_mdl_group_candidate_plans,
    build_multigroup_exact_mdl_queue,
    coordinate_descent_exact_mdl_groups,
    finalize_multigroup_exact_mdl_plan,
    select_exact_mdl_group_step,
)


GREEN_BASE = "mdl:Metal/GreenBase.mdl#GreenBase"
GREEN_PAINT = "mdl:Paint/PaintEggshell.mdl#Leaf"
GREEN_STEEL = "mdl:Metal/SteelPainted.mdl#ArmyGreen"
GREEN_EXTENDED = "mdl:Plastic/Extended.mdl#Green"
ORANGE_BASE = "mdl:Metal/OrangeBase.mdl#OrangeBase"
ORANGE_COPPER = "mdl:Metal/Copper.mdl#Aged"
ORANGE_PAINT = "mdl:Metal/SteelPainted.mdl#Orange"
ORANGE_BRASS = "mdl:Metal/Brass.mdl#Red"


def _sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _plan() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "assignments": [
            {
                "part_id": "P0001",
                "material_id": GREEN_BASE,
                "semantic": "main green body",
                "provenance": {"canonical_group_id": "G01"},
            },
            {
                "part_id": "P0002",
                "material_id": ORANGE_BASE,
                "semantic": "small orange accent",
                "provenance": {"canonical_group_id": "G02"},
            },
        ],
        "provenance": {"asset_sha256": "source-sha"},
    }


def _attach_source_appearance_cohort_contract(
    plan: dict[str, Any],
    *,
    group_id: str,
    expected_part_ids: list[str],
    candidate_kind: str = "dominant_assembly",
    cohort_signature_kind: str = "source_appearance_plus_subset_layout",
) -> dict[str, Any]:
    expected_part_ids = sorted(expected_part_ids)
    identity = {
        "schema_version": "qwen-source-appearance-cohort-contract/v1",
        "method": "trusted_spatial_anchor_source_appearance_cohort/v1",
        "candidate_kind": candidate_kind,
        "cohort_signature_kind": cohort_signature_kind,
        "canonical_group_id": group_id,
        "assembly_path": "/Asset/Assembly",
        "source_appearance_cohort_signature_sha256": "d" * 64,
        "anchor_part_ids": [expected_part_ids[0]],
        "expected_member_part_ids": expected_part_ids,
        "registry_sha256": "e" * 64,
        "spatial_report_sha256": "f" * 64,
        "annotation_input_plan_sha256": "a" * 64,
    }
    cohort_id = _sha(identity)
    contract = {
        **identity,
        "cohort_id": cohort_id,
        "subtree_part_ids": expected_part_ids,
        "propagated_member_part_ids": expected_part_ids[1:],
        "advisory_conflicts": {},
        "signature_dominance": {"part_share": 1.0, "face_share": 1.0},
        "exact_cover": True,
        "material_identity_unchanged": True,
        "parameters_unchanged": True,
    }
    contract["contract_sha256"] = _sha(contract)
    for assignment in plan["assignments"]:
        if assignment["part_id"] not in expected_part_ids:
            continue
        assignment["provenance"]["canonical_group_id"] = group_id
        assignment["provenance"]["source_appearance_cohort"] = {
            "schema_version": "qwen-source-appearance-cohort-contract/v1",
            "method": "trusted_spatial_anchor_source_appearance_cohort/v1",
            "cohort_id": cohort_id,
            "contract_sha256": contract["contract_sha256"],
            "canonical_group_id": group_id,
            "member_role": (
                "anchor"
                if assignment["part_id"] == expected_part_ids[0]
                else "propagated_member"
            ),
            "anchor_part_ids": [expected_part_ids[0]],
            "expected_member_part_ids": expected_part_ids,
            "propagated_member_part_ids": expected_part_ids[1:],
            "registry_sha256": "e" * 64,
            "spatial_report_sha256": "f" * 64,
            "annotation_input_plan_sha256": "a" * 64,
            "exact_cover": True,
            "material_identity_unchanged": True,
            "parameters_unchanged": True,
        }
    cohort_audit = {
        "schema_version": "qwen-source-appearance-cohort-propagation/v1",
        "enabled": True,
        "method": "trusted_spatial_anchor_source_appearance_cohort/v1",
        "registry_sha256": "e" * 64,
        "spatial_report_sha256": "f" * 64,
        "annotation_input_plan_sha256": "a" * 64,
        "cohort_count": 1,
        "expected_member_count": len(expected_part_ids),
        "propagated_member_count": len(expected_part_ids) - 1,
        "contracts": [contract],
        "rejected_candidates": [],
        "coverage_blockers": [],
        "exact_cover": True,
        "material_identity_unchanged": True,
        "parameters_unchanged": True,
    }
    plan["provenance"]["visual_group_annotation"] = {
        "schema_version": "qwen-visual-group-plan-annotation/v1",
        "source_appearance_cohort_propagation": copy.deepcopy(cohort_audit),
    }
    return {
        "schema_version": "qwen-visual-group-plan-annotation/v1",
        "annotated_plan_sha256": _sha(plan),
        "source_appearance_cohort_propagation": cohort_audit,
    }


def _plan_with_face_subset() -> dict[str, Any]:
    plan = _plan()
    plan["assignments"] = [plan["assignments"][0]]
    assignment = plan["assignments"][0]
    assignment["parameters"] = {}
    assignment["face_subsets"] = [
        {
            "subset_name": "Cover",
            "semantic": "orange cover",
            "material_id": ORANGE_BASE,
            "parameters": {},
            "face_indices": [1, 2, 3],
        },
        {
            "subset_name": "Seal",
            "semantic": "green seal",
            "material_id": GREEN_PAINT,
            "parameters": {},
            "face_indices": [4, 5],
        },
    ]
    assignment["provenance"]["face_subset_canonical_group_ids"] = {"Cover": "G02"}
    return plan


def _candidate_plan(
    current_plan: dict[str, Any],
    *,
    part_id: str | None,
    material_id: str | None,
) -> dict[str, Any]:
    output = copy.deepcopy(current_plan)
    if part_id is not None:
        assignment = next(
            item for item in output["assignments"] if item["part_id"] == part_id
        )
        assignment["material_id"] = material_id
    return output


def _bundle(
    *,
    current_plan: dict[str, Any],
    candidate_id: str,
    part_id: str | None,
    material_id: str | None,
    score: float,
    status: str = "PASS",
    target_group_id: str = "G01",
    target_part_ids: tuple[str, ...] = ("P0001",),
    target_entities: tuple[Mapping[str, Any], ...] | None = None,
) -> dict[str, Any]:
    plan = _candidate_plan(
        current_plan,
        part_id=part_id,
        material_id=material_id,
    )
    rendered_registry_file_sha256 = f"{candidate_id}-rendered-registry-file"
    output_sha256 = f"{candidate_id}-look-usd"
    passed = status == "PASS"
    reviewed = status == "REVIEW"
    failed = status == "FAIL"
    comparison_scope: dict[str, Any] = {
        "mode": "canonical_group_local",
        "target_group_id": target_group_id,
        "target_part_ids": sorted(target_part_ids),
        "reference_view_ids": ["front", "side"],
    }
    if target_entities is not None:
        comparison_scope["target_entities"] = [
            copy.deepcopy(dict(entity)) for entity in target_entities
        ]
        has_face_subset = any(
            entity.get("subset_name") is not None for entity in target_entities
        )
        comparison_scope["render_mask_granularity"] = (
            "containing_part_proxy" if has_face_subset else "exact_part_id"
        )
        comparison_scope["face_subset_render_mask_exact"] = not has_face_subset
    bundle = {
        "candidate_id": candidate_id,
        "is_baseline": part_id is None,
        "plan": plan,
        "apply_report": {
            "plan_sha256": _sha(plan),
            "output_sha256": output_sha256,
        },
        "rendered_registry": {"asset_sha256": output_sha256},
        "rendered_registry_file_sha256": rendered_registry_file_sha256,
        "quality_report": {
            "schema_version": "qwen-reference-render-comparison/v1",
            "inputs": {
                "rendered_registry_sha256": rendered_registry_file_sha256,
                "comparison_scope": comparison_scope,
            },
            "aggregate": {
                "status": status,
                "material_match_conclusion": (
                    "PASS" if passed else "FAIL" if failed else "NOT_CONCLUSIVE"
                ),
                "material_color_score": score,
                "material_texture_score": score,
                "material_appearance_score": score,
                "texture_comparable_view_count": 2,
                "texture_unscorable_view_count": 0,
                "reference_view_count": 2,
                "comparable_view_count": 2,
                "passed_view_count": 2 if passed else 0,
                "review_view_count": 2 if reviewed else 0,
                "failed_view_count": 2 if failed else 0,
                "unscorable_view_count": 0,
                "reference_view_coverage_status": "PASS",
            },
            "views": [
                {
                    "reference_view_id": "front",
                    "status": status,
                    "material_color": {"score": score},
                    "material_texture": {"score": score},
                    "material_appearance_score": score,
                },
                {
                    "reference_view_id": "side",
                    "status": status,
                    "material_color": {"score": score},
                    "material_texture": {"score": score},
                    "material_appearance_score": score,
                },
            ],
        },
    }
    _sync_global_quality(bundle)
    return bundle


def _sync_global_quality(bundle: dict[str, Any]) -> None:
    """Mirror test-local scores into an independently scoped global guard."""

    quality = copy.deepcopy(bundle["quality_report"])
    view_mapping = {
        str(view["reference_view_id"]): f"render_{view['reference_view_id']}"
        for view in quality["views"]
    }
    quality["inputs"]["comparison_scope"] = {"mode": "whole_asset"}
    quality["inputs"]["seeded_view_mapping"] = copy.deepcopy(view_mapping)
    quality["inputs"]["selected_view_mapping"] = copy.deepcopy(view_mapping)
    for view in quality["views"]:
        view["render_view_id"] = view_mapping[str(view["reference_view_id"])]
    bundle["global_quality_report"] = quality


def _set_quality(
    bundle: dict[str, Any],
    *,
    report_key: str,
    score: float,
    status: str,
) -> None:
    quality = bundle[report_key]
    view_count = len(quality["views"])
    quality["aggregate"].update(
        {
            "status": status,
            "material_match_conclusion": (
                "PASS"
                if status == "PASS"
                else "FAIL"
                if status == "FAIL"
                else "NOT_CONCLUSIVE"
            ),
            "material_color_score": score,
            "material_texture_score": score,
            "material_appearance_score": score,
            "passed_view_count": view_count if status == "PASS" else 0,
            "review_view_count": view_count if status == "REVIEW" else 0,
            "failed_view_count": view_count if status == "FAIL" else 0,
        }
    )
    for view in quality["views"]:
        view.update(
            {
                "status": status,
                "material_color": {"score": score},
                "material_texture": {"score": score},
                "material_appearance_score": score,
            }
        )


def _three_view_bundle(**kwargs: Any) -> dict[str, Any]:
    bundle = _bundle(**kwargs)
    score = float(kwargs["score"])
    status = str(kwargs.get("status", "PASS"))
    passed = status == "PASS"
    reviewed = status == "REVIEW"
    failed = status == "FAIL"
    scope = bundle["quality_report"]["inputs"]["comparison_scope"]
    scope["reference_view_ids"] = ["front", "side", "top"]
    aggregate = bundle["quality_report"]["aggregate"]
    aggregate.update(
        {
            "texture_comparable_view_count": 3,
            "reference_view_count": 3,
            "comparable_view_count": 3,
            "passed_view_count": 3 if passed else 0,
            "review_view_count": 3 if reviewed else 0,
            "failed_view_count": 3 if failed else 0,
        }
    )
    bundle["quality_report"]["views"].append(
        {
            "reference_view_id": "top",
            "status": status,
            "material_color": {"score": score},
            "material_texture": {"score": score},
            "material_appearance_score": score,
        }
    )
    _sync_global_quality(bundle)
    return bundle


def _green_candidate_document() -> dict[str, Any]:
    return {
        "candidates": [
            {
                "material_id": GREEN_PAINT,
                "family": "paint",
                "retrieval_score": 80.0,
                "retrieval_rank": 2,
            },
            {
                "material_id": GREEN_STEEL,
                "family": "metal",
                "retrieval_score": 90.0,
                "retrieval_rank": 1,
            },
        ],
        "tournament_candidates": [
            {
                "material_id": GREEN_EXTENDED,
                "family": "plastic",
                "retrieval_score": 500.0,
                "retrieval_rank": 1,
            },
            {
                "material_id": GREEN_PAINT,
                "family": "paint",
                "retrieval_score": 85.0,
                "retrieval_rank": 3,
            },
        ],
    }


def _retrieval_summary(
    *,
    strategy: str,
    ranking: list[dict[str, Any]],
    pool_count: int,
) -> dict[str, Any]:
    top = float(ranking[0]["score"])
    runner = float(ranking[1]["score"]) if len(ranking) > 1 else None
    margin = top - runner if runner is not None else None
    normalized = (
        margin
        / (
            max(abs(top), 1e-12)
            if strategy.startswith("siglip2_")
            else max(abs(top), 1.0)
        )
        if margin is not None
        else None
    )
    return {
        "strategy": strategy,
        "pool_count": pool_count,
        "eligible_pool_count": pool_count,
        "limit": len(ranking),
        "top_score": top,
        "runner_up_score": runner,
        "score_margin": margin,
        "normalized_margin": normalized,
        "margin_available": runner is not None,
        "ranking": ranking,
        "fixed_library_defaults_required": True,
    }


def _visual_row(
    *,
    rank: int,
    material_id: str,
    siglip_rank: int,
    dino_rank: int,
) -> dict[str, Any]:
    return {
        "rank": rank,
        "material_id": material_id,
        "score": round(
            1.0 / (60 + siglip_rank) + 1.2 / (60 + dino_rank),
            10,
        ),
        "matched_fields": [
            "siglip2_catalog_wide_visual",
            "dinov2_masked_dense_texture",
        ],
        "siglip2_rank": siglip_rank,
        "siglip2_score": 1.0 - siglip_rank / 100.0,
        "dino_rank": dino_rank,
        "dino_score": 1.0 - dino_rank / 100.0,
    }


def _base_bank_visual_row(
    *,
    rank: int,
    material_id: str,
) -> dict[str, Any]:
    score = round(
        1.0 / (60 + rank)
        + 1.2 / (60 + rank)
        + 0.8 / (60 + rank)
        + 0.2 / (60 + rank),
        10,
    )
    return {
        "rank": rank,
        "material_id": material_id,
        "score": score,
        "matched_fields": [
            "siglip2_base_bank_rig_visual",
            "dinov2_base_bank_surface_texture",
            "masked_color_appearance",
            "mvinverse_authored_pbr_prior",
        ],
        "siglip2_rank": rank,
        "siglip2_score": 1.0 - rank / 100.0,
        "dino_rank": rank,
        "dino_score": 1.0 - rank / 100.0,
        "color_rank": rank,
        "color_score": 1.0 - rank / 100.0,
        "mvinverse_rank": rank,
        "mvinverse_score": 1.0 - rank / 100.0,
    }


def _fallback_row(
    *,
    rank: int,
    material_id: str,
    score: float,
    matched_fields: list[str],
) -> dict[str, Any]:
    return {
        "rank": rank,
        "material_id": material_id,
        "score": score,
        "matched_fields": matched_fields,
    }


def _visual_candidate_document() -> tuple[dict[str, Any], set[str]]:
    allowed = {
        GREEN_BASE,
        GREEN_PAINT,
        GREEN_STEEL,
        GREEN_EXTENDED,
        ORANGE_COPPER,
        ORANGE_PAINT,
        ORANGE_BRASS,
    }
    selection_visual = [
        _visual_row(
            rank=1,
            material_id=GREEN_PAINT,
            siglip_rank=1,
            dino_rank=2,
        ),
        _visual_row(
            rank=2,
            material_id=GREEN_STEEL,
            siglip_rank=2,
            dino_rank=3,
        ),
    ]
    tournament_visual = [
        *selection_visual,
        _visual_row(
            rank=3,
            material_id=GREEN_EXTENDED,
            siglip_rank=3,
            dino_rank=4,
        ),
        _visual_row(
            rank=4,
            material_id=ORANGE_BRASS,
            siglip_rank=4,
            dino_rank=5,
        ),
    ]
    selection_fallback = [
        _fallback_row(
            rank=1,
            material_id=ORANGE_PAINT,
            score=100.0,
            matched_fields=["family", "color", "mvinverse_color"],
        ),
        _fallback_row(
            rank=2,
            material_id=ORANGE_COPPER,
            score=90.0,
            matched_fields=["family", "mvinverse_color"],
        ),
    ]
    tournament_fallback = [
        _fallback_row(
            rank=1,
            material_id=ORANGE_COPPER,
            score=80.0,
            matched_fields=["color", "mvinverse_color"],
        ),
        _fallback_row(
            rank=2,
            material_id=GREEN_EXTENDED,
            score=70.0,
            matched_fields=["mvinverse_color"],
        ),
    ]
    selection_audit = _retrieval_summary(
        strategy="siglip2_full_catalog_plus_dinov2_masked_rrf/v1",
        ranking=selection_visual,
        pool_count=len(allowed),
    )
    selection_audit.update(
        {
            "group_id": "G01",
            "full_catalog_indexed": True,
            "final_authority": "exact_mdl_render_tournament",
            "fallback_audit": _retrieval_summary(
                strategy="family_gated_semantic_mvinverse_similarity_score/v12",
                ranking=selection_fallback,
                pool_count=len(allowed),
            ),
        }
    )
    tournament_audit = _retrieval_summary(
        strategy="siglip2_full_catalog_plus_dinov2_masked_rrf/v1",
        ranking=tournament_visual,
        pool_count=len(allowed),
    )
    tournament_audit.update(
        {
            "group_id": "G01",
            "full_catalog_indexed": True,
            "final_authority": "exact_mdl_render_tournament",
            "fallback_audit": _retrieval_summary(
                strategy="visual_mvinverse_similarity_score/v1",
                ranking=tournament_fallback,
                pool_count=len(allowed),
            ),
        }
    )
    return (
        {
            "group": {"group_id": "G01"},
            "candidates": [
                {
                    "material_id": GREEN_PAINT,
                    "family": "paint",
                    "retrieval_score": selection_visual[0]["score"],
                    "retrieval_rank": 1,
                },
                {
                    "material_id": GREEN_STEEL,
                    "family": "metal",
                    "retrieval_score": selection_visual[1]["score"],
                    "retrieval_rank": 2,
                },
            ],
            "tournament_candidates": [
                {
                    "material_id": row["material_id"],
                    "family": "metal",
                    "retrieval_score": row["score"],
                    "retrieval_rank": row["rank"],
                }
                for row in tournament_visual
            ],
            "retrieval_audit": selection_audit,
            "tournament_retrieval_audit": tournament_audit,
        },
        allowed,
    )


def _whole_asset_presence_quality_report(
    rows_by_view: Mapping[str, Mapping[str, tuple[str, float]]],
) -> dict[str, Any]:
    return {
        "schema_version": "qwen-reference-render-comparison/v1",
        "inputs": {"comparison_scope": {"mode": "whole_asset"}},
        "aggregate": {
            "reference_view_coverage_status": "PASS",
            "reference_view_count": len(rows_by_view),
        },
        "views": [
            {
                "reference_view_id": view_id,
                "mapping": {"locked": True},
                "reference": {
                    "trusted_evidence": {
                        "usable": True,
                        "reasons": [],
                        "samples": [
                            {
                                "group_id": local_group_id,
                                "base_color": "green",
                                "representative_srgb": [35, 115, 48],
                                "weight_pixels": 256,
                                "box_normalized_1000": [100, 100, 300, 300],
                            }
                            for local_group_id in sorted(rows)
                        ],
                        "sample_count": len(rows),
                        "total_weight_pixels": 256 * len(rows),
                    }
                },
                "material_color": {
                    "trusted_evidence_group_recall": {
                        "group_count": len(rows),
                        "groups": [
                            {
                                "group_id": local_group_id,
                                "delivery_presence_status": status,
                                "recall": recall,
                            }
                            for local_group_id, (status, recall) in sorted(rows.items())
                        ],
                    }
                },
            }
            for view_id, rows in sorted(rows_by_view.items())
        ],
    }


def _whole_asset_trusted_sample_quality_report(
    sample_group_ids_by_view: Mapping[str, tuple[str, ...]],
) -> dict[str, Any]:
    """Build the reference-side trust evidence used to scope local rounds."""

    return {
        "schema_version": "qwen-reference-render-comparison/v1",
        "inputs": {"comparison_scope": {"mode": "whole_asset"}},
        "aggregate": {
            "reference_view_coverage_status": "PASS",
            "reference_view_count": len(sample_group_ids_by_view),
            "comparable_view_count": len(sample_group_ids_by_view),
            "unscorable_view_count": 0,
        },
        "views": [
            {
                "reference_view_id": view_id,
                "status": "PASS",
                "reference": {
                    "trusted_evidence": {
                        "usable": True,
                        "reasons": [],
                        "samples": [
                            {
                                "group_id": local_group_id,
                                "base_color": "orange",
                                "representative_srgb": [150, 90, 45],
                                "weight_pixels": 128,
                                "box_normalized_1000": [100, 100, 300, 300],
                            }
                            for local_group_id in local_group_ids
                        ],
                        "sample_count": len(local_group_ids),
                        "total_weight_pixels": 128 * len(local_group_ids),
                    }
                },
            }
            for view_id, local_group_ids in sorted(
                sample_group_ids_by_view.items()
            )
        ],
    }


def _four_view_local_group_palette_fusion() -> dict[str, Any]:
    local_group_ids = {
        "front": "F11",
        "iso": "I11",
        "side": "S11",
        "top": "T11",
    }
    return {
        "canonical_palette": {
            "groups": [
                {
                    "group_id": "G01",
                    "sources": [
                        {
                            "view_id": view_id,
                            "local_group_id": local_group_id,
                            "confidence": 0.8,
                            "boxes": [[100, 100, 300, 300]],
                        }
                        for view_id, local_group_id in sorted(
                            local_group_ids.items()
                        )
                    ],
                }
            ]
        },
        "view_group_id_maps": {
            view_id: {local_group_id: "G01"}
            for view_id, local_group_id in sorted(local_group_ids.items())
        },
    }


def _trusted_queue_inputs(palette_fusion: Mapping[str, Any]) -> dict[str, Any]:
    """Complete concise queue fixtures with the production trust contract."""

    fusion = copy.deepcopy(dict(palette_fusion))
    canonical_palette = fusion["canonical_palette"]
    view_maps = copy.deepcopy(fusion.get("view_group_id_maps", {}))
    samples_by_view: dict[str, list[str]] = {}
    for group in canonical_palette["groups"]:
        group_id = str(group["group_id"])
        for source in group["sources"]:
            view_id = str(source["view_id"])
            view_map = view_maps.setdefault(view_id, {})
            local_group_id = source.get("local_group_id")
            if not isinstance(local_group_id, str) or not local_group_id:
                existing = sorted(
                    local_id
                    for local_id, canonical_id in view_map.items()
                    if canonical_id == group_id
                )
                local_group_id = (
                    existing[0] if len(existing) == 1 else f"{view_id}_{group_id}"
                )
                source["local_group_id"] = local_group_id
            view_map[local_group_id] = group_id
            samples_by_view.setdefault(view_id, []).append(local_group_id)
    fusion["view_group_id_maps"] = view_maps
    return {
        "palette_fusion": fusion,
        "quality_report": _whole_asset_trusted_sample_quality_report(
            {
                view_id: tuple(sorted(set(local_group_ids)))
                for view_id, local_group_ids in sorted(samples_by_view.items())
            }
        ),
    }


def _baseline_presence_palette_fusion() -> dict[str, Any]:
    source_views = {
        "G01": ("front", "side"),
        "G02": ("front", "iso"),
        "G04": ("iso", "side", "top"),
        "G05": ("iso", "top"),
        "G07": ("front", "iso", "side", "top"),
    }
    view_group_id_maps = {
        "front": {
            "F01": "G01",
            "F02": "G02",
            "F07": "G07",
        },
        "iso": {
            "I02": "G02",
            "I04": "G04",
            "I05": "G05",
            "I07": "G07",
        },
        "side": {
            "S01": "G01",
            "S04": "G04",
            "S07": "G07",
        },
        "top": {
            "T04": "G04",
            "T05": "G05",
            "T07": "G07",
        },
    }
    return {
        "canonical_palette": {
            "groups": [
                {
                    "group_id": group_id,
                    "sources": [
                        {
                            "view_id": view_id,
                            "local_group_id": next(
                                local_group_id
                                for local_group_id, canonical_group_id in view_group_id_maps[
                                    view_id
                                ].items()
                                if canonical_group_id == group_id
                            ),
                            "confidence": 0.8,
                            "boxes": [[0, 0, 100, 100]],
                        }
                        for view_id in view_ids
                    ],
                }
                for group_id, view_ids in source_views.items()
            ]
        },
        "view_group_id_maps": view_group_id_maps,
    }


def test_group_planner_merges_primary_and_tournament_without_losing_primary() -> None:
    planned, audit = build_exact_mdl_group_candidate_plans(
        source_plan=_plan(),
        group_id="G01",
        target_part_ids=["P0001"],
        candidate_document=_green_candidate_document(),
        allowed_material_ids={
            GREEN_BASE,
            GREEN_PAINT,
            GREEN_STEEL,
            GREEN_EXTENDED,
            ORANGE_BASE,
        },
        maximum_candidates=4,
    )

    # The high-scoring extended candidate cannot evict Paint_Eggshell_Leaf or
    # Steel_Painted_Army_Green from the bounded render round.
    assert [record["material_id"] for record in planned] == [
        GREEN_BASE,
        GREEN_STEEL,
        GREEN_PAINT,
        GREEN_EXTENDED,
    ]
    merge = audit["primary_and_tournament_merge"]
    assert merge["all_primary_candidates_preserved"] is True
    assert merge["primary_unique_nonbaseline_count"] == 2
    assert merge["extended_unique_nonbaseline_count"] == 1
    assert planned[0]["is_baseline"] is True
    assert all(
        not assignment.get("parameters")
        for record in planned
        for assignment in record["plan"]["assignments"]
    )


def test_group_planner_fails_instead_of_silently_truncating_primary_pool() -> None:
    with pytest.raises(
        MultigroupExactMdlTournamentError,
        match="cannot preserve baseline plus every primary candidate",
    ):
        build_exact_mdl_group_candidate_plans(
            source_plan=_plan(),
            group_id="G01",
            target_part_ids=["P0001"],
            candidate_document=_green_candidate_document(),
            allowed_material_ids={
                GREEN_BASE,
                GREEN_PAINT,
                GREEN_STEEL,
                GREEN_EXTENDED,
                ORANGE_BASE,
            },
            maximum_candidates=2,
        )


def test_visual_retrieval_fuses_semantic_and_mvinverse_candidates_by_rank() -> None:
    document, allowed = _visual_candidate_document()
    planned, audit = build_exact_mdl_group_candidate_plans(
        source_plan=_plan(),
        group_id="G01",
        target_part_ids=["P0001"],
        candidate_document=document,
        allowed_material_ids=allowed,
        maximum_candidates=6,
        selection_objective="visual_similarity",
    )

    material_ids = [record["material_id"] for record in planned]
    assert material_ids[:3] == [GREEN_BASE, GREEN_PAINT, GREEN_STEEL]
    assert ORANGE_PAINT in material_ids
    assert ORANGE_COPPER in material_ids
    merge = audit["primary_and_tournament_merge"]
    assert merge["visual_retrieval_fusion_applied"] is True
    assert merge["all_primary_candidates_preserved"] is True
    assert merge["selection_fallback_input_count"] == 2
    assert merge["tournament_fallback_input_count"] == 2
    assert merge["fused_fallback_unique_count"] == 3
    assert merge["raw_scores_compared_across_lanes"] is False
    copper = next(
        row
        for row in merge["selected_candidates"]
        if row["material_id"] == ORANGE_COPPER
    )
    assert copper["sources"] == [
        "selection_fallback",
        "tournament_fallback",
    ]
    assert set(copper["fallback_source_evidence"]) == {
        "selection_fallback",
        "tournament_fallback",
    }


def test_base_bank_visual_retrieval_reaches_exact_render_tournament() -> None:
    document, allowed = _visual_candidate_document()
    strategy = "base_observation_bank_siglip2_dinov2_color_mvinverse_rrf/v1"
    for audit_name in ("retrieval_audit", "tournament_retrieval_audit"):
        audit = document[audit_name]
        rows = [
            _base_bank_visual_row(
                rank=index,
                material_id=row["material_id"],
            )
            for index, row in enumerate(audit["ranking"], start=1)
        ]
        top = rows[0]["score"]
        runner = rows[1]["score"]
        audit.update(
            {
                "strategy": strategy,
                "ranking": rows,
                "top_score": top,
                "runner_up_score": runner,
                "score_margin": top - runner,
                "normalized_margin": (top - runner) / abs(top),
            }
        )

    planned, audit = build_exact_mdl_group_candidate_plans(
        source_plan=_plan(),
        group_id="G01",
        target_part_ids=["P0001"],
        candidate_document=document,
        allowed_material_ids=allowed,
        maximum_candidates=6,
        selection_objective="visual_similarity",
    )

    assert planned
    assert audit["primary_and_tournament_merge"][
        "visual_retrieval_fusion_applied"
    ] is True


def test_visual_retrieval_reserves_three_diverse_candidate_lanes() -> None:
    visual_ids = [f"mdl:Visual/Candidate_{index:02d}.mdl#Visual" for index in range(1, 13)]
    semantic_ids = [
        f"mdl:Semantic/Coating_{index:02d}.mdl#Coating" for index in range(1, 5)
    ]
    numeric_ids = [
        f"mdl:Numeric/Default_{index:02d}.mdl#Default" for index in range(1, 13)
    ]
    allowed = {
        GREEN_BASE,
        ORANGE_BASE,
        *visual_ids,
        *semantic_ids,
        *numeric_ids,
    }
    tournament_visual = [
        _visual_row(
            rank=index,
            material_id=material_id,
            siglip_rank=index,
            dino_rank=index,
        )
        for index, material_id in enumerate(visual_ids, start=1)
    ]
    selection_visual = tournament_visual[:4]
    selection_fallback = [
        _fallback_row(
            rank=index,
            material_id=material_id,
            score=100.0 - index,
            matched_fields=["confirmed_applied_coating", "mvinverse_color"],
        )
        for index, material_id in enumerate(semantic_ids, start=1)
    ]
    tournament_fallback = [
        _fallback_row(
            rank=index,
            material_id=material_id,
            score=200.0 - index,
            matched_fields=["mvinverse_color", "mvinverse_roughness"],
        )
        for index, material_id in enumerate(numeric_ids, start=1)
    ]
    selection_audit = _retrieval_summary(
        strategy="siglip2_full_catalog_plus_dinov2_masked_rrf/v1",
        ranking=selection_visual,
        pool_count=len(allowed),
    )
    selection_audit.update(
        {
            "group_id": "G01",
            "full_catalog_indexed": True,
            "final_authority": "exact_mdl_render_tournament",
            "fallback_audit": _retrieval_summary(
                strategy="family_gated_semantic_mvinverse_similarity_score/v12",
                ranking=selection_fallback,
                pool_count=len(allowed),
            ),
        }
    )
    tournament_audit = _retrieval_summary(
        strategy="siglip2_full_catalog_plus_dinov2_masked_rrf/v1",
        ranking=tournament_visual,
        pool_count=len(allowed),
    )
    tournament_audit.update(
        {
            "group_id": "G01",
            "full_catalog_indexed": True,
            "final_authority": "exact_mdl_render_tournament",
            "fallback_audit": _retrieval_summary(
                strategy="visual_mvinverse_similarity_score/v1",
                ranking=tournament_fallback,
                pool_count=len(allowed),
            ),
        }
    )
    document = {
        "group": {"group_id": "G01"},
        "candidates": [
            {
                "material_id": row["material_id"],
                "family": "visual",
                "retrieval_score": row["score"],
                "retrieval_rank": row["rank"],
            }
            for row in selection_visual
        ],
        "tournament_candidates": [
            {
                "material_id": row["material_id"],
                "family": "visual",
                "retrieval_score": row["score"],
                "retrieval_rank": row["rank"],
            }
            for row in tournament_visual
        ],
        "retrieval_audit": selection_audit,
        "tournament_retrieval_audit": tournament_audit,
    }

    planned, audit = build_exact_mdl_group_candidate_plans(
        source_plan=_plan(),
        group_id="G01",
        target_part_ids=["P0001"],
        candidate_document=document,
        allowed_material_ids=allowed,
        maximum_candidates=12,
        selection_objective="visual_similarity",
    )

    assert [record["material_id"] for record in planned] == [
        GREEN_BASE,
        *visual_ids[:4],
        *semantic_ids[:2],
        numeric_ids[0],
        numeric_ids[4],
        numeric_ids[9],
        *visual_ids[4:6],
    ]
    merge = audit["primary_and_tournament_merge"]
    assert merge["all_primary_candidates_preserved"] is True
    assert merge["reserved_lane_quotas"] == {
        "selection_fallback": 2,
        "tournament_fallback": 3,
        "visual": 2,
    }
    assert merge["reserved_lane_selected_counts"] == merge["reserved_lane_quotas"]
    assert set(merge["reserved_lane_quota_satisfied"].values()) == {True}
    assert merge["reserved_lane_selected_material_ids"]["selection_fallback"] == (
        semantic_ids[:2]
    )
    assert merge["reserved_lane_selected_material_ids"]["tournament_fallback"] == [
        numeric_ids[0],
        numeric_ids[4],
        numeric_ids[9],
    ]
    assert merge["tournament_fallback_diversity_policy"] == (
        "rank_stratified_head_middle_lower_quartile/v1"
    )


@pytest.mark.parametrize(
    "corruption",
    (
        "visual_rrf_score",
        "selection_prefix",
        "fallback_catalog_identity",
        "duplicate_source_rank",
    ),
)
def test_visual_retrieval_fusion_fails_closed_on_malformed_audit(
    corruption: str,
) -> None:
    document, allowed = _visual_candidate_document()
    if corruption == "visual_rrf_score":
        document["tournament_retrieval_audit"]["ranking"][0]["score"] += 0.001
        document["tournament_retrieval_audit"]["top_score"] += 0.001
    elif corruption == "selection_prefix":
        document["retrieval_audit"]["ranking"][0]["material_id"] = ORANGE_BRASS
    elif corruption == "fallback_catalog_identity":
        document["retrieval_audit"]["fallback_audit"]["ranking"][0]["material_id"] = (
            "mdl:Outside/Catalog.mdl#Outside"
        )
    elif corruption == "duplicate_source_rank":
        document["tournament_retrieval_audit"]["ranking"][1]["siglip2_rank"] = document[
            "tournament_retrieval_audit"
        ]["ranking"][0]["siglip2_rank"]
    else:
        raise AssertionError(corruption)

    with pytest.raises(MultigroupExactMdlTournamentError):
        build_exact_mdl_group_candidate_plans(
            source_plan=_plan(),
            group_id="G01",
            target_part_ids=["P0001"],
            candidate_document=document,
            allowed_material_ids=allowed,
            maximum_candidates=6,
            selection_objective="visual_similarity",
        )


def test_group_planner_can_consolidate_a_mixed_visual_group() -> None:
    source = _plan()
    source["assignments"][1]["provenance"]["canonical_group_id"] = "G01"
    planned, audit = build_exact_mdl_group_candidate_plans(
        source_plan=source,
        group_id="G01",
        target_part_ids=["P0001", "P0002"],
        candidate_document={
            "candidates": [
                {
                    "material_id": GREEN_BASE,
                    "family": "metal",
                    "retrieval_score": 90.0,
                },
                {
                    "material_id": GREEN_PAINT,
                    "family": "paint",
                    "retrieval_score": 80.0,
                },
            ],
            "tournament_candidates": [],
        },
        allowed_material_ids={
            GREEN_BASE,
            GREEN_PAINT,
            ORANGE_BASE,
        },
        maximum_candidates=3,
    )

    assert audit["source_material_id"] is None
    assert audit["source_material_ids"] == [GREEN_BASE, ORANGE_BASE]
    assert planned[0]["material_id"] is None
    consolidated = planned[1]["plan"]["assignments"]
    assert {item["material_id"] for item in consolidated} == {GREEN_BASE}
    assert planned[1]["plan"]["provenance"]["exact_mdl_candidate"][
        "changed_part_ids"
    ] == ["P0002"]


def test_queue_contains_every_candidate_bearing_visual_group() -> None:
    source = _plan()
    queue_inputs = _trusted_queue_inputs(
        {
            "canonical_palette": {
                "groups": [
                    {
                        "group_id": "G01",
                        "sources": [
                            {
                                "view_id": "front",
                                "confidence": 0.9,
                                "boxes": [[0, 0, 900, 900]],
                            },
                            {
                                "view_id": "side",
                                "confidence": 0.9,
                                "boxes": [[0, 0, 900, 900]],
                            },
                        ],
                    },
                    {
                        "group_id": "G02",
                        "sources": [
                            {
                                "view_id": "front",
                                "confidence": 0.8,
                                "boxes": [[0, 0, 100, 100]],
                            },
                            {
                                "view_id": "side",
                                "confidence": 0.8,
                                "boxes": [[0, 0, 100, 100]],
                            },
                        ],
                    },
                ]
            }
        }
    )
    queue, audit = build_multigroup_exact_mdl_queue(
        source_plan=source,
        material_candidates_by_group={
            "G01": _green_candidate_document(),
            "G02": {
                "candidates": [
                    {
                        "material_id": ORANGE_COPPER,
                        "family": "metal",
                        "retrieval_score": 70.0,
                    }
                ],
                "tournament_candidates": [],
            },
        },
        material_choice_audit={},
        **queue_inputs,
        allowed_material_ids={
            GREEN_BASE,
            GREEN_PAINT,
            GREEN_STEEL,
            GREEN_EXTENDED,
            ORANGE_BASE,
            ORANGE_COPPER,
        },
        maximum_candidates=4,
    )

    assert [group["group_id"] for group in queue] == ["G01", "G02"]
    assert queue[0]["reference_view_ids"] == ["front", "side"]
    assert queue[0]["reference_view_count"] == 2
    assert audit["significant_group_ids"] == ["G01", "G02"]
    assert audit["all_candidate_bearing_significant_groups_queued"] is True


def test_queue_intersects_fusion_sources_with_trusted_quality_samples() -> None:
    source = _plan()
    source["assignments"] = [source["assignments"][0]]
    quality_report = _whole_asset_trusted_sample_quality_report(
        {
            "front": ("F11",),
            "iso": ("I11",),
            # The side palette has usable evidence overall, but not for the
            # local group fused into G01.  It must not enter the local round.
            "side": ("S_OTHER",),
            "top": ("T11",),
        }
    )

    queue, _audit = build_multigroup_exact_mdl_queue(
        source_plan=source,
        material_candidates_by_group={"G01": _green_candidate_document()},
        material_choice_audit={},
        palette_fusion=_four_view_local_group_palette_fusion(),
        allowed_material_ids={
            GREEN_BASE,
            GREEN_PAINT,
            GREEN_STEEL,
            GREEN_EXTENDED,
        },
        maximum_candidates=4,
        quality_report=quality_report,
    )

    assert len(queue) == 1
    assert queue[0]["canonical_reference_view_ids"] == [
        "front",
        "iso",
        "side",
        "top",
    ]
    assert queue[0]["reference_view_ids"] == ["front", "iso", "top"]
    assert queue[0]["reference_view_count"] == 3
    scope = queue[0]["trusted_scoring_reference_scope"]
    assert scope["selection_mode"] == "WHOLE_ASSET_TRUSTED_EVIDENCE_INTERSECTION"
    assert scope["canonical_reference_view_ids"] == [
        "front",
        "iso",
        "side",
        "top",
    ]
    assert scope["reference_view_ids"] == ["front", "iso", "top"]
    assert scope["excluded_reference_sources"] == [
        {
            "reference_view_id": "side",
            "local_group_id": "S11",
            "reason": "LOCAL_GROUP_ABSENT_FROM_TRUSTED_REFERENCE_EVIDENCE",
            "trusted_evidence_audit": None,
            "trusted_evidence_audit_sha256": None,
            "matching_trusted_sample_count": 0,
        }
    ]
    assert isinstance(scope["quality_report_sha256"], str)
    assert len(scope["quality_report_sha256"]) == 64


def test_queue_preserves_baseline_when_fewer_than_two_sources_are_trusted() -> None:
    source = _plan()
    source["assignments"] = [source["assignments"][0]]
    quality_report = _whole_asset_trusted_sample_quality_report(
        {
            "front": ("F11",),
            "iso": ("I_OTHER",),
            "side": ("S_OTHER",),
            "top": ("T_OTHER",),
        }
    )

    queue, audit = build_multigroup_exact_mdl_queue(
        source_plan=source,
        material_candidates_by_group={"G01": _green_candidate_document()},
        material_choice_audit={},
        palette_fusion=_four_view_local_group_palette_fusion(),
        allowed_material_ids={
            GREEN_BASE,
            GREEN_PAINT,
            GREEN_STEEL,
            GREEN_EXTENDED,
        },
        maximum_candidates=4,
        quality_report=quality_report,
    )

    assert queue == []
    exclusion = next(
        item for item in audit["excluded_groups"] if item["group_id"] == "G01"
    )
    assert exclusion["reason"] == "INSUFFICIENT_TRUSTED_SCORING_REFERENCE_VIEWS"
    assert exclusion["canonical_reference_view_ids"] == [
        "front",
        "iso",
        "side",
        "top",
    ]
    assert exclusion["reference_view_ids"] == ["front"]
    assert exclusion["reference_view_count"] == 1
    assert exclusion["baseline_preserved"] is True
    assert exclusion["authored_target_entity_count"] == 1
    assert exclusion not in audit["coverage_blockers"]
    scope = exclusion["trusted_scoring_reference_scope"]
    assert scope["reference_view_ids"] == ["front"]
    assert [
        item["reference_view_id"] for item in scope["excluded_reference_sources"]
    ] == ["iso", "side", "top"]


def test_source_corroborated_top_level_group_enters_exact_mdl_queue() -> None:
    source = _plan()
    source["assignments"] = [source["assignments"][1]]
    source["assignments"][0]["provenance"] = {
        "tier": "corroborated_source_visual_nvidia_mdl",
        "canonical_group_id": "G_BLUE",
        "source_visual_corroboration": {"canonical_group_id": "G_BLUE"},
    }
    candidate_document = {
        "candidates": [
            {
                "material_id": ORANGE_COPPER,
                "family": "metal",
                "retrieval_score": 70.0,
            }
        ],
        "tournament_candidates": [],
    }
    queue_inputs = _trusted_queue_inputs(
        {
            "canonical_palette": {
                "groups": [
                    {
                        "group_id": "G_BLUE",
                        "sources": [
                            {
                                "view_id": "iso",
                                "confidence": 0.6,
                                "boxes": [[400, 300, 450, 360]],
                            },
                            {
                                "view_id": "top",
                                "confidence": 0.6,
                                "boxes": [[500, 300, 550, 360]],
                            },
                        ],
                    }
                ]
            }
        }
    )

    queue, audit = build_multigroup_exact_mdl_queue(
        source_plan=source,
        material_candidates_by_group={"G_BLUE": candidate_document},
        material_choice_audit={},
        **queue_inputs,
        allowed_material_ids={ORANGE_BASE, ORANGE_COPPER},
        maximum_candidates=4,
    )

    assert len(queue) == 1
    assert queue[0]["group_id"] == "G_BLUE"
    assert queue[0]["target_part_ids"] == ["P0002"]
    assert audit["significant_group_ids"] == ["G_BLUE"]


def test_queue_blocks_multiview_candidate_group_without_target_entities() -> None:
    source = _plan()
    queue_inputs = _trusted_queue_inputs(
        {
            "canonical_palette": {
                "groups": [
                    {
                        "group_id": "G01",
                        "sources": [
                            {
                                "view_id": "front",
                                "confidence": 0.9,
                                "boxes": [[0, 0, 900, 900]],
                            },
                            {
                                "view_id": "side",
                                "confidence": 0.9,
                                "boxes": [[0, 0, 900, 900]],
                            },
                        ],
                    },
                    {
                        "group_id": "G02",
                        "sources": [
                            {
                                "view_id": "front",
                                "confidence": 0.8,
                                "boxes": [[0, 0, 100, 100]],
                            },
                            {
                                "view_id": "side",
                                "confidence": 0.8,
                                "boxes": [[0, 0, 100, 100]],
                            },
                        ],
                    },
                    {
                        "group_id": "G03",
                        "sources": [
                            {
                                "view_id": "front",
                                "confidence": 0.8,
                                "boxes": [[0, 0, 120, 120]],
                            },
                            {
                                "view_id": "side",
                                "confidence": 0.8,
                                "boxes": [[0, 0, 120, 120]],
                            },
                        ],
                    },
                ]
            }
        }
    )
    queue, audit = build_multigroup_exact_mdl_queue(
        source_plan=source,
        material_candidates_by_group={
            "G01": _green_candidate_document(),
            "G02": {
                "candidates": [
                    {
                        "material_id": ORANGE_COPPER,
                        "family": "metal",
                        "retrieval_score": 70.0,
                    }
                ],
                "tournament_candidates": [],
            },
            "G03": {
                "candidates": [
                    {
                        "material_id": ORANGE_COPPER,
                        "family": "metal",
                        "retrieval_score": 70.0,
                    }
                ],
                "tournament_candidates": [],
            },
        },
        material_choice_audit={},
        **queue_inputs,
        allowed_material_ids={
            GREEN_BASE,
            GREEN_PAINT,
            GREEN_STEEL,
            GREEN_EXTENDED,
            ORANGE_BASE,
            ORANGE_COPPER,
        },
        maximum_candidates=4,
    )

    assert [group["group_id"] for group in queue] == ["G01", "G02"]
    assert audit["coverage_blockers"] == [
        {
            "group_id": "G03",
            "reason": "NO_TARGET_MATERIAL_ENTITIES",
            "reference_footprint_score": pytest.approx(0.02304),
            "reference_view_ids": ["front", "side"],
            "reference_view_count": 2,
        }
    ]
    assert audit["coverage_blocker_count"] == 1
    assert audit["all_discovered_significant_groups_queued"] is False
    assert audit["all_candidate_bearing_significant_groups_queued"] is False


def test_queue_preserves_unlocalized_baseline_only_when_every_source_view_is_present() -> (
    None
):
    source = _plan()
    source["assignments"][1]["provenance"].pop("canonical_group_id")
    candidate_documents = {
        group_id: _green_candidate_document()
        for group_id in ("G01", "G02", "G04", "G05", "G07")
    }
    quality_report = _whole_asset_presence_quality_report(
        {
            "front": {
                "F01": ("PRESENT", 1.0),
                "F02": ("PRESENT", 1.0),
                "F07": ("MISSING", 0.0),
            },
            "iso": {
                "I02": ("PRESENT", 1.0),
                "I04": ("PRESENT", 1.0),
                "I05": ("PRESENT", 1.0),
                "I07": ("MISSING", 0.0),
            },
            "side": {
                "S01": ("PRESENT", 1.0),
                "S04": ("PRESENT", 1.0),
                "S07": ("MISSING", 0.0),
            },
            "top": {
                "T04": ("PRESENT", 1.0),
                "T05": ("PRESENT", 1.0),
                "T07": ("MISSING", 0.0),
            },
        }
    )

    queue, audit = build_multigroup_exact_mdl_queue(
        source_plan=source,
        material_candidates_by_group=candidate_documents,
        material_choice_audit={},
        palette_fusion=_baseline_presence_palette_fusion(),
        allowed_material_ids={
            GREEN_BASE,
            GREEN_PAINT,
            GREEN_STEEL,
            GREEN_EXTENDED,
            ORANGE_BASE,
        },
        maximum_candidates=4,
        quality_report=quality_report,
    )

    assert [group["group_id"] for group in queue] == ["G01"]
    exclusions = {
        exclusion["group_id"]: exclusion for exclusion in audit["excluded_groups"]
    }
    for group_id, expected_views in {
        "G02": ["front", "iso"],
        "G04": ["iso", "side", "top"],
        "G05": ["iso", "top"],
    }.items():
        exclusion = exclusions[group_id]
        assert exclusion["reason"] == BASELINE_GROUP_PRESENT_WITHOUT_LOCALIZATION
        assert exclusion["baseline_preserved"] is True
        evidence = exclusion["baseline_presence_evidence"]
        assert evidence["canonical_group_id"] == group_id
        assert evidence["reference_view_ids"] == expected_views
        assert evidence["all_source_views_present"] is True
        assert all(
            view["delivery_presence_status"] == "PRESENT"
            and view["recall"] == 1.0
            and view["trusted_reference_evidence"] is True
            for view in evidence["views"]
        )
    assert exclusions["G07"]["reason"] == "NO_TARGET_MATERIAL_ENTITIES"
    assert audit["coverage_blocker_count"] == 1
    assert [item["group_id"] for item in audit["coverage_blockers"]] == ["G07"]
    assert audit["baseline_presence_uses_material_or_semantic_tokens"] is False


@pytest.mark.parametrize(
    "corruption",
    (
        "missing_quality_report",
        "missing_source_view",
        "untrusted_source_view",
        "missing_local_to_canonical_mapping",
        "missing_group_row",
        "nonpresent_status",
        "nonexact_recall",
    ),
)
def test_queue_unlocalized_baseline_presence_exemption_fails_closed(
    corruption: str,
) -> None:
    source = _plan()
    source["assignments"][1]["provenance"].pop("canonical_group_id")
    palette_fusion = _baseline_presence_palette_fusion()
    quality_report = _whole_asset_presence_quality_report(
        {
            "front": {
                "F01": ("PRESENT", 1.0),
                "F02": ("PRESENT", 1.0),
                "F07": ("MISSING", 0.0),
            },
            "iso": {
                "I02": ("PRESENT", 1.0),
                "I04": ("PRESENT", 1.0),
                "I05": ("PRESENT", 1.0),
                "I07": ("MISSING", 0.0),
            },
            "side": {
                "S01": ("PRESENT", 1.0),
                "S04": ("PRESENT", 1.0),
                "S07": ("MISSING", 0.0),
            },
            "top": {
                "T04": ("PRESENT", 1.0),
                "T05": ("PRESENT", 1.0),
                "T07": ("MISSING", 0.0),
            },
        }
    )
    quality_input: Mapping[str, Any] | None = quality_report
    if corruption == "missing_quality_report":
        quality_input = None
    elif corruption == "missing_source_view":
        quality_report["views"] = [
            view
            for view in quality_report["views"]
            if view["reference_view_id"] != "iso"
        ]
    elif corruption == "untrusted_source_view":
        quality_report["views"][0]["reference"]["trusted_evidence"]["usable"] = False
    elif corruption == "missing_local_to_canonical_mapping":
        palette_fusion["view_group_id_maps"]["iso"].pop("I02")
    elif corruption == "missing_group_row":
        iso = next(
            view
            for view in quality_report["views"]
            if view["reference_view_id"] == "iso"
        )
        iso["material_color"]["trusted_evidence_group_recall"]["groups"] = []
        iso["material_color"]["trusted_evidence_group_recall"]["group_count"] = 0
    elif corruption == "nonpresent_status":
        quality_report["views"][0]["material_color"]["trusted_evidence_group_recall"][
            "groups"
        ][1]["delivery_presence_status"] = "MISSING"
    elif corruption == "nonexact_recall":
        quality_report["views"][0]["material_color"]["trusted_evidence_group_recall"][
            "groups"
        ][1]["recall"] = 0.999
    else:
        raise AssertionError(f"unhandled corruption fixture: {corruption}")

    build_kwargs = {
        "source_plan": source,
        "material_candidates_by_group": {
            "G01": _green_candidate_document(),
            "G02": _green_candidate_document(),
        },
        "material_choice_audit": {},
        "palette_fusion": palette_fusion,
        "allowed_material_ids": {
            GREEN_BASE,
            GREEN_PAINT,
            GREEN_STEEL,
            GREEN_EXTENDED,
            ORANGE_BASE,
        },
        "maximum_candidates": 4,
        "quality_report": quality_input,
    }
    if corruption in {"missing_source_view", "missing_local_to_canonical_mapping"}:
        with pytest.raises(MultigroupExactMdlTournamentError):
            build_multigroup_exact_mdl_queue(**build_kwargs)
        return

    _queue, audit = build_multigroup_exact_mdl_queue(**build_kwargs)

    g02 = next(item for item in audit["excluded_groups"] if item["group_id"] == "G02")
    assert g02["reason"] != BASELINE_GROUP_PRESENT_WITHOUT_LOCALIZATION
    if corruption == "untrusted_source_view":
        assert g02["reason"] == "INSUFFICIENT_TRUSTED_SCORING_REFERENCE_VIEWS"
        assert g02["baseline_preserved"] is True
        assert g02 not in audit["coverage_blockers"]
    else:
        assert g02["reason"] == "NO_TARGET_MATERIAL_ENTITIES"
        assert g02 in audit["coverage_blockers"]


def test_queue_honors_frozen_m0_membership_exclusions() -> None:
    source = _plan()
    source["provenance"]["dominant_assembly_membership_tournaments"] = [
        {
            "schema_version": SELECTION_SCHEMA_VERSION,
            "cohort_id": "a" * 64,
            "canonical_group_id": "G01",
            "selected_membership_mode": M0_CANDIDATE,
            "excluded_expanded_part_ids": ["P0001"],
            "membership_frozen_before_exact_mdl_tournament": True,
            "parameters_locked_to_library_defaults": True,
        }
    ]
    queue_inputs = _trusted_queue_inputs(
        {
            "canonical_palette": {
                "groups": [
                    {
                        "group_id": group_id,
                        "sources": [
                            {
                                "view_id": "front",
                                "confidence": 0.8,
                                "boxes": [[0, 0, 100, 100]],
                            },
                            {
                                "view_id": "side",
                                "confidence": 0.8,
                                "boxes": [[0, 0, 100, 100]],
                            },
                        ],
                    }
                    for group_id in ("G01", "G02")
                ]
            }
        }
    )
    queue, audit = build_multigroup_exact_mdl_queue(
        source_plan=source,
        material_candidates_by_group={
            "G01": _green_candidate_document(),
            "G02": _green_candidate_document(),
        },
        material_choice_audit={},
        **queue_inputs,
        allowed_material_ids={
            GREEN_BASE,
            GREEN_PAINT,
            GREEN_STEEL,
            GREEN_EXTENDED,
            ORANGE_BASE,
        },
        maximum_candidates=4,
    )

    assert [group["group_id"] for group in queue] == ["G02"]
    assert audit["membership_freeze_applied"] is True
    assert audit["membership_excluded_part_count"] == 1
    assert audit["membership_excluded_part_ids_by_group"] == {"G01": ["P0001"]}
    g01 = next(item for item in audit["coverage_blockers"] if item["group_id"] == "G01")
    assert g01["reason"] == "NO_TARGET_MATERIAL_ENTITIES"


def test_queue_verifies_complete_source_appearance_cohort_targets() -> None:
    source = _plan()
    annotation_audit = _attach_source_appearance_cohort_contract(
        source,
        group_id="G01",
        expected_part_ids=["P0001", "P0002"],
    )
    queue_inputs = _trusted_queue_inputs(
        {
            "canonical_palette": {
                "groups": [
                    {
                        "group_id": "G01",
                        "sources": [
                            {
                                "view_id": "front",
                                "confidence": 0.9,
                                "boxes": [[0, 0, 100, 100]],
                            },
                            {
                                "view_id": "side",
                                "confidence": 0.9,
                                "boxes": [[0, 0, 100, 100]],
                            },
                        ],
                    }
                ]
            }
        }
    )

    queue, audit = build_multigroup_exact_mdl_queue(
        source_plan=source,
        material_candidates_by_group={"G01": _green_candidate_document()},
        material_choice_audit={},
        **queue_inputs,
        allowed_material_ids={
            GREEN_BASE,
            GREEN_PAINT,
            GREEN_STEEL,
            GREEN_EXTENDED,
            ORANGE_BASE,
        },
        maximum_candidates=4,
        visual_group_annotation_audit=annotation_audit,
    )

    assert queue[0]["target_part_ids"] == ["P0001", "P0002"]
    assert queue[0]["source_appearance_cohort_expected_part_ids"] == [
        "P0001",
        "P0002",
    ]
    coverage = audit["source_appearance_cohort_coverage"]
    assert coverage["annotation_audit_verified"] is True
    assert coverage["exact_cover"] is True
    assert coverage["coverage_blocker_count"] == 0
    assert coverage["queued_member_part_ids_by_group"] == {
        "G01": ["P0001", "P0002"]
    }


def test_queue_accepts_rare_source_appearance_layout_pair_contract() -> None:
    source = _plan()
    annotation_audit = _attach_source_appearance_cohort_contract(
        source,
        group_id="G01",
        expected_part_ids=["P0001", "P0002"],
        candidate_kind="rare_source_appearance_layout_pair",
        cohort_signature_kind="source_appearance_plus_subset_layout",
    )
    queue_inputs = _trusted_queue_inputs(
        {
            "canonical_palette": {
                "groups": [
                    {
                        "group_id": "G01",
                        "sources": [
                            {
                                "view_id": "front",
                                "confidence": 0.9,
                                "boxes": [[0, 0, 100, 100]],
                            },
                            {
                                "view_id": "side",
                                "confidence": 0.9,
                                "boxes": [[0, 0, 100, 100]],
                            },
                        ],
                    }
                ]
            }
        }
    )

    queue, audit = build_multigroup_exact_mdl_queue(
        source_plan=source,
        material_candidates_by_group={"G01": _green_candidate_document()},
        material_choice_audit={},
        **queue_inputs,
        allowed_material_ids={
            GREEN_BASE,
            GREEN_PAINT,
            GREEN_STEEL,
            GREEN_EXTENDED,
            ORANGE_BASE,
        },
        maximum_candidates=4,
        visual_group_annotation_audit=annotation_audit,
    )

    assert queue[0]["target_part_ids"] == ["P0001", "P0002"]
    assert audit["source_appearance_cohort_coverage"]["exact_cover"] is True


def test_queue_blocks_when_frozen_membership_drops_a_cohort_target() -> None:
    source = _plan()
    source["provenance"]["dominant_assembly_membership_tournaments"] = [
        {
            "schema_version": SELECTION_SCHEMA_VERSION,
            "cohort_id": "a" * 64,
            "canonical_group_id": "G01",
            "selected_membership_mode": M0_CANDIDATE,
            "excluded_expanded_part_ids": ["P0002"],
            "membership_frozen_before_exact_mdl_tournament": True,
            "parameters_locked_to_library_defaults": True,
        }
    ]
    annotation_audit = _attach_source_appearance_cohort_contract(
        source,
        group_id="G01",
        expected_part_ids=["P0001", "P0002"],
    )
    queue_inputs = _trusted_queue_inputs(
        {
            "canonical_palette": {
                "groups": [
                    {
                        "group_id": "G01",
                        "sources": [
                            {
                                "view_id": "front",
                                "confidence": 0.9,
                                "boxes": [[0, 0, 100, 100]],
                            },
                            {
                                "view_id": "side",
                                "confidence": 0.9,
                                "boxes": [[0, 0, 100, 100]],
                            },
                        ],
                    }
                ]
            }
        }
    )

    _queue, audit = build_multigroup_exact_mdl_queue(
        source_plan=source,
        material_candidates_by_group={"G01": _green_candidate_document()},
        material_choice_audit={},
        **queue_inputs,
        allowed_material_ids={
            GREEN_BASE,
            GREEN_PAINT,
            GREEN_STEEL,
            GREEN_EXTENDED,
            ORANGE_BASE,
        },
        maximum_candidates=4,
        visual_group_annotation_audit=annotation_audit,
    )

    blocker = next(
        item
        for item in audit["coverage_blockers"]
        if item["reason"] == "SOURCE_APPEARANCE_COHORT_TARGET_INCOMPLETE"
    )
    assert blocker["group_id"] == "G01"
    assert blocker["missing_part_ids"] == ["P0002"]
    coverage = audit["source_appearance_cohort_coverage"]
    assert coverage["exact_cover"] is False
    assert coverage["coverage_blocker_count"] == 1


def test_queue_covers_parent_and_face_subset_groups_on_the_same_part() -> None:
    source = _plan_with_face_subset()
    queue_inputs = _trusted_queue_inputs(
        {
            "canonical_palette": {
                "groups": [
                    {
                        "group_id": "G01",
                        "sources": [
                            {
                                "view_id": "front",
                                "confidence": 0.9,
                                "boxes": [[0, 0, 900, 900]],
                            },
                            {
                                "view_id": "side",
                                "confidence": 0.9,
                                "boxes": [[0, 0, 900, 900]],
                            },
                        ],
                    },
                    {
                        "group_id": "G02",
                        "sources": [
                            {
                                "view_id": "front",
                                "confidence": 0.8,
                                "boxes": [[0, 0, 100, 100]],
                            },
                            {
                                "view_id": "side",
                                "confidence": 0.8,
                                "boxes": [[0, 0, 100, 100]],
                            },
                        ],
                    },
                ]
            }
        }
    )
    queue, audit = build_multigroup_exact_mdl_queue(
        source_plan=source,
        material_candidates_by_group={
            "G01": _green_candidate_document(),
            "G02": {
                "candidates": [
                    {
                        "material_id": ORANGE_COPPER,
                        "family": "metal",
                        "retrieval_score": 70.0,
                    }
                ],
                "tournament_candidates": [],
            },
        },
        material_choice_audit={},
        **queue_inputs,
        allowed_material_ids={
            GREEN_BASE,
            GREEN_PAINT,
            GREEN_STEEL,
            GREEN_EXTENDED,
            ORANGE_BASE,
            ORANGE_COPPER,
        },
        maximum_candidates=4,
    )

    groups = {group["group_id"]: group for group in queue}
    assert groups["G01"]["target_entities"] == [
        {"entity_kind": "assignment", "part_id": "P0001"}
    ]
    assert groups["G02"]["target_entities"] == [
        {
            "entity_kind": "face_subset",
            "part_id": "P0001",
            "subset_name": "Cover",
        }
    ]
    assert groups["G01"]["target_part_ids"] == ["P0001"]
    assert groups["G02"]["target_part_ids"] == ["P0001"]
    assert groups["G02"]["target_face_subset_count"] == 1
    assert audit["face_subset_groups_supported"] is True
    assert audit["coverage_blocker_count"] == 0


def test_queue_preserves_baseline_for_a_single_reference_view_group() -> None:
    source = _plan()
    queue_inputs = _trusted_queue_inputs(
        {
            "canonical_palette": {
                "groups": [
                    {
                        "group_id": "G01",
                        "sources": [
                            {
                                "view_id": "front",
                                "confidence": 0.9,
                                "boxes": [[0, 0, 900, 900]],
                            },
                            {
                                "view_id": "side",
                                "confidence": 0.9,
                                "boxes": [[0, 0, 900, 900]],
                            },
                        ],
                    },
                    {
                        "group_id": "G02",
                        "sources": [
                            {
                                "view_id": "front",
                                "confidence": 0.8,
                                "boxes": [[0, 0, 100, 100]],
                            }
                        ],
                    },
                ]
            }
        }
    )
    queue, audit = build_multigroup_exact_mdl_queue(
        source_plan=source,
        material_candidates_by_group={
            "G01": _green_candidate_document(),
            "G02": {
                "candidates": [
                    {
                        "material_id": ORANGE_COPPER,
                        "family": "metal",
                        "retrieval_score": 70.0,
                    }
                ],
                "tournament_candidates": [],
            },
        },
        material_choice_audit={},
        **queue_inputs,
        allowed_material_ids={
            GREEN_BASE,
            GREEN_PAINT,
            GREEN_STEEL,
            GREEN_EXTENDED,
            ORANGE_BASE,
            ORANGE_COPPER,
        },
        maximum_candidates=4,
    )

    assert [group["group_id"] for group in queue] == ["G01"]
    assert audit["coverage_blocker_count"] == 0
    assert audit["excluded_groups"] == [
        {
            "group_id": "G02",
            "reason": "INSUFFICIENT_INDEPENDENT_REFERENCE_VIEWS",
            "reference_view_ids": ["front"],
            "reference_view_count": 1,
            "baseline_preserved": True,
            "authored_target_entity_count": 1,
            "baseline_presence_evidence": None,
        }
    ]
    assert audit["insufficient_reference_groups_preserve_baseline"] is True


def test_face_subset_group_candidate_and_selection_change_only_subset_binding() -> None:
    current = _plan_with_face_subset()
    target_entities = [
        {
            "entity_kind": "face_subset",
            "part_id": "P0001",
            "subset_name": "Cover",
        }
    ]
    planned, planning = build_exact_mdl_group_candidate_plans(
        source_plan=current,
        group_id="G02",
        target_part_ids=["P0001"],
        target_entities=target_entities,
        candidate_document={
            "candidates": [
                {
                    "material_id": ORANGE_COPPER,
                    "family": "metal",
                    "retrieval_score": 70.0,
                }
            ],
            "tournament_candidates": [],
        },
        allowed_material_ids={
            GREEN_BASE,
            ORANGE_BASE,
            ORANGE_COPPER,
        },
        maximum_candidates=3,
    )

    assert planning["target_face_subset_count"] == 1
    challenger = planned[1]["plan"]["assignments"][0]
    assert challenger["material_id"] == GREEN_BASE
    assert challenger["face_subsets"][0]["material_id"] == ORANGE_COPPER
    assert challenger["face_subsets"][0]["face_indices"] == [1, 2, 3]
    assert challenger["face_subsets"][1]["material_id"] == GREEN_PAINT
    assert challenger["face_subsets"][1]["face_indices"] == [4, 5]
    assert challenger["parameters"] == {}
    assert challenger["face_subsets"][0]["parameters"] == {}
    assert "provenance" not in challenger["face_subsets"][0]

    bundles: list[dict[str, Any]] = []
    for index, planned_candidate in enumerate(planned):
        bundle = _bundle(
            current_plan=current,
            candidate_id=str(planned_candidate["candidate_id"]),
            part_id=None,
            material_id=None,
            score=0.70 if index == 0 else 0.90,
            target_group_id="G02",
            target_part_ids=("P0001",),
            target_entities=tuple(target_entities),
        )
        bundle["is_baseline"] = planned_candidate["is_baseline"]
        bundle["plan"] = copy.deepcopy(planned_candidate["plan"])
        bundle["apply_report"]["plan_sha256"] = _sha(bundle["plan"])
        bundles.append(bundle)

    output, audit = select_exact_mdl_group_step(
        current_plan=current,
        group_id="G02",
        target_part_ids=["P0001"],
        target_entities=target_entities,
        candidates=bundles,
        allowed_material_ids={
            GREEN_BASE,
            ORANGE_BASE,
            ORANGE_COPPER,
        },
        minimum_score_improvement=0.015,
        minimum_winner_margin=0.005,
    )

    assignment = output["assignments"][0]
    subset = assignment["face_subsets"][0]
    assert audit["status"] == ROUND_ACCEPTED
    assert audit["target_entities"] == target_entities
    assert assignment["material_id"] == GREEN_BASE
    assert subset["material_id"] == ORANGE_COPPER
    assert subset["face_indices"] == [1, 2, 3]
    assert assignment["face_subsets"][1]["material_id"] == GREEN_PAINT
    assert assignment["face_subsets"][1]["face_indices"] == [4, 5]
    assert assignment["parameters"] == {}
    assert subset["parameters"] == {}
    assert "provenance" not in subset
    assert audit["material_changes"] == [
        {
            "part_id": "P0001",
            "subset_name": "Cover",
            "old_material_id": ORANGE_BASE,
            "new_material_id": ORANGE_COPPER,
        }
    ]


def test_face_subset_planner_rejects_target_from_a_different_group() -> None:
    current = _plan_with_face_subset()
    with pytest.raises(
        MultigroupExactMdlTournamentError,
        match="belongs to 'G02', not 'G01'",
    ):
        build_exact_mdl_group_candidate_plans(
            source_plan=current,
            group_id="G01",
            target_part_ids=["P0001"],
            target_entities=[
                {
                    "entity_kind": "face_subset",
                    "part_id": "P0001",
                    "subset_name": "Cover",
                }
            ],
            candidate_document=_green_candidate_document(),
            allowed_material_ids={
                GREEN_BASE,
                GREEN_PAINT,
                GREEN_STEEL,
                GREEN_EXTENDED,
                ORANGE_BASE,
            },
            maximum_candidates=4,
        )


def test_group_step_accepts_only_clear_improvement() -> None:
    current = _plan()
    output, audit = select_exact_mdl_group_step(
        current_plan=current,
        group_id="G01",
        target_part_ids=["P0001"],
        candidates=[
            _bundle(
                current_plan=current,
                candidate_id="baseline",
                part_id=None,
                material_id=None,
                score=0.70,
            ),
            _bundle(
                current_plan=current,
                candidate_id="runner",
                part_id="P0001",
                material_id=GREEN_PAINT,
                score=0.77,
            ),
            _bundle(
                current_plan=current,
                candidate_id="winner",
                part_id="P0001",
                material_id=GREEN_STEEL,
                score=0.80,
            ),
        ],
        allowed_material_ids={
            GREEN_BASE,
            GREEN_PAINT,
            GREEN_STEEL,
            ORANGE_BASE,
        },
        minimum_score_improvement=0.05,
        minimum_winner_margin=0.02,
    )

    assert audit["status"] == ROUND_ACCEPTED
    assert audit["score_improvement"] == pytest.approx(0.10)
    assert audit["winner_margin"] == pytest.approx(0.03)
    assert output["assignments"][0]["material_id"] == GREEN_STEEL
    assert output["provenance"]["immutable_mdl_after_selection"] is True


def test_group_step_reverts_when_improvement_is_too_small() -> None:
    current = _plan()
    output, audit = select_exact_mdl_group_step(
        current_plan=current,
        group_id="G01",
        target_part_ids=["P0001"],
        candidates=[
            _bundle(
                current_plan=current,
                candidate_id="baseline",
                part_id=None,
                material_id=None,
                score=0.70,
            ),
            _bundle(
                current_plan=current,
                candidate_id="tiny_gain",
                part_id="P0001",
                material_id=GREEN_PAINT,
                score=0.71,
            ),
            _bundle(
                current_plan=current,
                candidate_id="lower",
                part_id="P0001",
                material_id=GREEN_STEEL,
                score=0.67,
            ),
        ],
        allowed_material_ids={
            GREEN_BASE,
            GREEN_PAINT,
            GREEN_STEEL,
            ORANGE_BASE,
        },
        minimum_score_improvement=0.02,
        minimum_winner_margin=0.005,
    )

    assert audit["status"] == ROUND_FALLBACK_INSUFFICIENT_IMPROVEMENT
    assert audit["fallback_to_input_plan"] is True
    assert audit["material_changes"] == []
    assert output == current


def test_group_step_accepts_g06_pass_promotion_below_both_score_margins() -> None:
    """A PASS challenger must not be discarded in favor of a REVIEW baseline."""

    current = _plan()
    output, audit = select_exact_mdl_group_step(
        current_plan=current,
        group_id="G01",
        target_part_ids=["P0001"],
        candidates=[
            _bundle(
                current_plan=current,
                candidate_id="g06_review_baseline",
                part_id=None,
                material_id=None,
                score=0.911463,
                status="REVIEW",
            ),
            _bundle(
                current_plan=current,
                candidate_id="g06_pass_runner_up",
                part_id="P0001",
                material_id=GREEN_PAINT,
                score=0.916000,
            ),
            _bundle(
                current_plan=current,
                candidate_id="g06_pass_winner",
                part_id="P0001",
                material_id=GREEN_STEEL,
                score=0.917984,
            ),
        ],
        allowed_material_ids={
            GREEN_BASE,
            GREEN_PAINT,
            GREEN_STEEL,
            ORANGE_BASE,
        },
        minimum_score_improvement=0.015,
        minimum_winner_margin=0.005,
    )

    assert audit["status"] == ROUND_ACCEPTED
    assert audit["accepted_candidate_id"] == "g06_pass_winner"
    assert audit["baseline_quality_tier"] == "COMPLETE_FAIL"
    assert audit["selected_quality_tier"] == "ALL_VIEW_PASS"
    assert audit["quality_tier_promotion"] is True
    assert audit["score_thresholds_applicable"] is False
    assert audit["score_improvement"] == pytest.approx(0.006521)
    assert audit["winner_margin"] == pytest.approx(0.001984)
    assert audit["score_improvement"] < audit["minimum_score_improvement"]
    assert audit["winner_margin"] < audit["minimum_winner_margin"]
    assert audit["reason_codes"] == [
        "ALL_VIEW_PASS_QUALITY_PROMOTION_OVER_NONPASS_BASELINE"
    ]
    assert output["assignments"][0]["material_id"] == GREEN_STEEL
    assert output["assignments"][0].get("parameters", {}) == {}
    assert output["provenance"]["immutable_mdl_after_selection"] is True


def test_group_step_reverts_when_winner_margin_is_ambiguous() -> None:
    current = _plan()
    output, audit = select_exact_mdl_group_step(
        current_plan=current,
        group_id="G01",
        target_part_ids=["P0001"],
        candidates=[
            _bundle(
                current_plan=current,
                candidate_id="baseline",
                part_id=None,
                material_id=None,
                score=0.70,
            ),
            _bundle(
                current_plan=current,
                candidate_id="near_tie",
                part_id="P0001",
                material_id=GREEN_PAINT,
                score=0.797,
            ),
            _bundle(
                current_plan=current,
                candidate_id="winner",
                part_id="P0001",
                material_id=GREEN_STEEL,
                score=0.80,
            ),
        ],
        allowed_material_ids={
            GREEN_BASE,
            GREEN_PAINT,
            GREEN_STEEL,
            ORANGE_BASE,
        },
        minimum_score_improvement=0.02,
        minimum_winner_margin=0.005,
    )

    assert audit["status"] == ROUND_FALLBACK_AMBIGUOUS_WINNER
    assert output == current


def test_group_step_promotes_near_tied_nonfail_review_over_fail_baseline() -> None:
    """A small positive gain in a better tier must not restore a failed baseline."""

    current = _plan()
    output, audit = select_exact_mdl_group_step(
        current_plan=current,
        group_id="G01",
        target_part_ids=["P0001"],
        candidates=[
            _three_view_bundle(
                current_plan=current,
                candidate_id="baseline_fail",
                part_id=None,
                material_id=None,
                score=0.6300,
                status="FAIL",
            ),
            _three_view_bundle(
                current_plan=current,
                candidate_id="near_tied_review",
                part_id="P0001",
                material_id=GREEN_PAINT,
                score=0.6343,
                status="REVIEW",
            ),
            _three_view_bundle(
                current_plan=current,
                candidate_id="review_winner",
                part_id="P0001",
                material_id=GREEN_STEEL,
                score=0.6376,
                status="REVIEW",
            ),
        ],
        allowed_material_ids={
            GREEN_BASE,
            GREEN_PAINT,
            GREEN_STEEL,
            ORANGE_BASE,
        },
        minimum_score_improvement=0.015,
        minimum_winner_margin=0.005,
    )

    assert audit["status"] == ROUND_ACCEPTED
    assert audit["accepted_candidate_id"] == "review_winner"
    assert audit["selected_quality_tier"] == "COMPLETE_NONFAIL_REVIEW"
    assert audit["complete_nonfail_review_quality_promotion"] is True
    assert (
        audit["positive_score_complete_nonfail_review_quality_promotion"]
        is True
    )
    assert audit["score_thresholds_applicable"] is False
    assert audit["score_improvement"] == pytest.approx(0.0076)
    assert audit["score_improvement"] < audit["minimum_score_improvement"]
    assert audit["winner_margin"] == pytest.approx(0.0033)
    assert audit["winner_margin"] < audit["minimum_winner_margin"]
    assert audit["reason_codes"] == [
        "COMPLETE_NONFAIL_REVIEW_QUALITY_PROMOTION_OVER_FAIL_BASELINE"
    ]
    assert output["assignments"][0]["material_id"] == GREEN_STEEL


def test_group_step_accepts_clear_improvement_over_complete_fail_baseline() -> None:
    current = _plan()
    output, audit = select_exact_mdl_group_step(
        current_plan=current,
        group_id="G01",
        target_part_ids=["P0001"],
        candidates=[
            _bundle(
                current_plan=current,
                candidate_id="baseline_review",
                part_id=None,
                material_id=None,
                score=0.60,
                status="FAIL",
            ),
            _bundle(
                current_plan=current,
                candidate_id="challenger",
                part_id="P0001",
                material_id=GREEN_STEEL,
                score=0.90,
            ),
        ],
        allowed_material_ids={
            GREEN_BASE,
            GREEN_STEEL,
            ORANGE_BASE,
        },
    )

    assert audit["status"] == ROUND_ACCEPTED
    assert audit["baseline_aggregate_status"] == "FAIL"
    assert audit["baseline_score_complete"] is True
    assert audit["baseline_all_view_pass"] is False
    assert audit["score_improvement"] == pytest.approx(0.30)
    assert output["assignments"][0]["material_id"] == GREEN_STEEL


def test_group_step_still_requires_challenger_to_pass_every_view() -> None:
    current = _plan()
    output, audit = select_exact_mdl_group_step(
        current_plan=current,
        group_id="G01",
        target_part_ids=["P0001"],
        candidates=[
            _bundle(
                current_plan=current,
                candidate_id="baseline_fail",
                part_id=None,
                material_id=None,
                score=0.60,
                status="FAIL",
            ),
            _bundle(
                current_plan=current,
                candidate_id="higher_review",
                part_id="P0001",
                material_id=GREEN_PAINT,
                score=0.99,
                status="REVIEW",
            ),
            _bundle(
                current_plan=current,
                candidate_id="lower_all_view_pass",
                part_id="P0001",
                material_id=GREEN_STEEL,
                score=0.85,
            ),
        ],
        allowed_material_ids={
            GREEN_BASE,
            GREEN_PAINT,
            GREEN_STEEL,
            ORANGE_BASE,
        },
    )

    assert audit["status"] == ROUND_ACCEPTED
    assert audit["selected_candidate_id"] == "lower_all_view_pass"
    assert output["assignments"][0]["material_id"] == GREEN_STEEL


def test_group_step_accepts_clear_complete_nonfail_review_winner() -> None:
    current = _plan()
    output, audit = select_exact_mdl_group_step(
        current_plan=current,
        group_id="G01",
        target_part_ids=["P0001"],
        candidates=[
            _three_view_bundle(
                current_plan=current,
                candidate_id="baseline_fail",
                part_id=None,
                material_id=None,
                score=0.60,
                status="FAIL",
            ),
            _three_view_bundle(
                current_plan=current,
                candidate_id="review_runner",
                part_id="P0001",
                material_id=GREEN_PAINT,
                score=0.82,
                status="REVIEW",
            ),
            _three_view_bundle(
                current_plan=current,
                candidate_id="review_winner",
                part_id="P0001",
                material_id=GREEN_STEEL,
                score=0.90,
                status="REVIEW",
            ),
        ],
        allowed_material_ids={
            GREEN_BASE,
            GREEN_PAINT,
            GREEN_STEEL,
            ORANGE_BASE,
        },
        minimum_score_improvement=0.015,
        minimum_winner_margin=0.005,
    )

    assert audit["status"] == ROUND_ACCEPTED
    assert audit["selected_candidate_id"] == "review_winner"
    assert audit["selected_quality_tier"] == "COMPLETE_NONFAIL_REVIEW"
    assert audit["score_improvement"] == pytest.approx(0.30)
    assert audit["winner_margin"] == pytest.approx(0.08)
    assert audit["reason_codes"] == [
        "COMPLETE_NONFAIL_REVIEW_CLEAR_VISUAL_WINNER"
    ]
    assert output["assignments"][0]["material_id"] == GREEN_STEEL
    assert output["assignments"][0].get("parameters", {}) == {}


@pytest.mark.parametrize("aggregate_status", ["REVIEW", "INSUFFICIENT_EVIDENCE"])
def test_group_step_uses_complete_baseline_regardless_of_aggregate_label(
    aggregate_status: str,
) -> None:
    current = _plan()
    baseline = _bundle(
        current_plan=current,
        candidate_id=f"baseline_{aggregate_status}",
        part_id=None,
        material_id=None,
        score=0.60,
        status="REVIEW",
    )
    baseline["quality_report"]["aggregate"]["status"] = aggregate_status
    _sync_global_quality(baseline)

    output, audit = select_exact_mdl_group_step(
        current_plan=current,
        group_id="G01",
        target_part_ids=["P0001"],
        candidates=[
            baseline,
            _bundle(
                current_plan=current,
                candidate_id="challenger",
                part_id="P0001",
                material_id=GREEN_STEEL,
                score=0.90,
            ),
        ],
        allowed_material_ids={
            GREEN_BASE,
            GREEN_STEEL,
            ORANGE_BASE,
        },
    )

    assert audit["status"] == ROUND_ACCEPTED
    assert audit["baseline_aggregate_status"] == aggregate_status
    assert audit["baseline_score_complete"] is True
    assert output["assignments"][0]["material_id"] == GREEN_STEEL


@pytest.mark.parametrize("aggregate_status", ["REVIEW", "INSUFFICIENT_EVIDENCE"])
def test_group_step_fails_closed_when_baseline_score_evidence_is_incomplete(
    aggregate_status: str,
) -> None:
    current = _plan()
    baseline = _bundle(
        current_plan=current,
        candidate_id=f"incomplete_{aggregate_status}",
        part_id=None,
        material_id=None,
        score=0.60,
        status="REVIEW",
    )
    baseline["quality_report"]["aggregate"].update(
        {
            "status": aggregate_status,
            "material_appearance_score": None,
            "texture_comparable_view_count": 1,
            "texture_unscorable_view_count": 1,
        }
    )
    _sync_global_quality(baseline)

    output, audit = select_exact_mdl_group_step(
        current_plan=current,
        group_id="G01",
        target_part_ids=["P0001"],
        candidates=[
            baseline,
            _bundle(
                current_plan=current,
                candidate_id="challenger",
                part_id="P0001",
                material_id=GREEN_STEEL,
                score=0.90,
            ),
        ],
        allowed_material_ids={
            GREEN_BASE,
            GREEN_STEEL,
            ORANGE_BASE,
        },
    )

    assert audit["status"] == ROUND_FALLBACK_BASELINE_INELIGIBLE
    assert audit["baseline_score_complete"] is False
    assert (
        "AGGREGATE_TEXTURE_APPEARANCE_SCORE_INCOMPLETE"
        in audit["baseline_comparability_reason_codes"]
    )
    assert output == current


def test_group_step_excludes_unscorable_challenger_and_selects_valid_candidate() -> (
    None
):
    current = _plan()
    unscorable = _bundle(
        current_plan=current,
        candidate_id="transparent_unscorable",
        part_id="P0001",
        material_id=GREEN_PAINT,
        score=0.99,
        status="REVIEW",
    )
    unscorable["quality_report"]["aggregate"].update(
        {
            "status": "INSUFFICIENT_EVIDENCE",
            "material_match_conclusion": "NOT_CONCLUSIVE",
            "material_color_score": None,
            "material_texture_score": None,
            "material_appearance_score": None,
            "texture_comparable_view_count": 0,
            "texture_unscorable_view_count": 2,
            "comparable_view_count": 0,
            "passed_view_count": 0,
            "review_view_count": 0,
            "failed_view_count": 0,
            "unscorable_view_count": 2,
            "reference_view_coverage_status": "INSUFFICIENT_EVIDENCE",
        }
    )
    for view in unscorable["quality_report"]["views"]:
        view.update(
            {
                "status": "UNSCORABLE",
                "material_color": {"score": None},
                "material_texture": {"score": None},
                "material_appearance_score": None,
                "reasons": [
                    "render_target_foreground_too_small",
                    "render_target_foreground_missing",
                ],
            }
        )

    output, audit = select_exact_mdl_group_step(
        current_plan=current,
        group_id="G01",
        target_part_ids=["P0001"],
        candidates=[
            _bundle(
                current_plan=current,
                candidate_id="baseline",
                part_id=None,
                material_id=None,
                score=0.60,
            ),
            unscorable,
            _bundle(
                current_plan=current,
                candidate_id="valid_winner",
                part_id="P0001",
                material_id=GREEN_STEEL,
                score=0.90,
            ),
        ],
        allowed_material_ids={
            GREEN_BASE,
            GREEN_PAINT,
            GREEN_STEEL,
            ORANGE_BASE,
        },
    )

    assert audit["status"] == ROUND_ACCEPTED
    assert audit["accepted_candidate_id"] == "valid_winner"
    assert audit["incomplete_challenger_count"] == 1
    excluded = audit["incomplete_challengers"][0]
    assert excluded["candidate_id"] == "transparent_unscorable"
    assert excluded["local"]["aggregate_status"] == "INSUFFICIENT_EVIDENCE"
    assert excluded["local"]["complete"] is False
    assert excluded["local"]["reason_codes"] == [
        "AGGREGATE_COLOR_SCORE_INCOMPLETE",
        "AGGREGATE_TEXTURE_APPEARANCE_SCORE_INCOMPLETE",
        "LOCAL_PER_VIEW_SCORES_INCOMPLETE",
        "LOCAL_VIEW_COVERAGE_INCOMPLETE",
    ]
    assert excluded["local"]["view_reason_codes"] == [
        "render_target_foreground_missing",
        "render_target_foreground_too_small",
    ]
    assert excluded["global"]["complete"] is True
    assert (
        next(
            assignment["material_id"]
            for assignment in output["assignments"]
            if assignment["part_id"] == "P0001"
        )
        == GREEN_STEEL
    )


def test_group_step_excludes_visually_eligible_challenger_with_missing_scoped_view() -> (
    None
):
    current = _plan()
    missing_iso = _three_view_bundle(
        current_plan=current,
        candidate_id="missing_iso",
        part_id="P0001",
        material_id=GREEN_PAINT,
        score=0.99,
        status="REVIEW",
    )
    missing_iso["quality_report"]["inputs"]["comparison_scope"][
        "reference_view_ids"
    ] = ["front", "iso", "side", "top"]

    output, audit = select_exact_mdl_group_step(
        current_plan=current,
        group_id="G01",
        target_part_ids=["P0001"],
        candidates=[
            _three_view_bundle(
                current_plan=current,
                candidate_id="baseline",
                part_id=None,
                material_id=None,
                score=0.60,
                status="REVIEW",
            ),
            _three_view_bundle(
                current_plan=current,
                candidate_id="valid_winner",
                part_id="P0001",
                material_id=GREEN_STEEL,
                score=0.80,
                status="REVIEW",
            ),
            missing_iso,
        ],
        allowed_material_ids={
            GREEN_BASE,
            GREEN_PAINT,
            GREEN_STEEL,
            ORANGE_BASE,
        },
    )

    assert audit["status"] == ROUND_ACCEPTED
    assert audit["accepted_candidate_id"] == "valid_winner"
    assert audit["incomplete_challenger_count"] == 1
    assert audit["incomplete_challengers"][0]["candidate_id"] == "missing_iso"
    assert (
        next(
            assignment["material_id"]
            for assignment in output["assignments"]
            if assignment["part_id"] == "P0001"
        )
        == GREEN_STEEL
    )


def test_group_step_fails_closed_when_aggregate_counts_disagree_with_views() -> None:
    current = _plan()
    inconsistent_baseline = _three_view_bundle(
        current_plan=current,
        candidate_id="inconsistent_baseline",
        part_id=None,
        material_id=None,
        score=0.60,
    )
    inconsistent_baseline["quality_report"]["aggregate"].update(
        {
            "status": "FAIL",
            "material_match_conclusion": "FAIL",
            "passed_view_count": 0,
            "failed_view_count": 3,
        }
    )
    _sync_global_quality(inconsistent_baseline)

    output, audit = select_exact_mdl_group_step(
        current_plan=current,
        group_id="G01",
        target_part_ids=["P0001"],
        candidates=[
            inconsistent_baseline,
            _three_view_bundle(
                current_plan=current,
                candidate_id="challenger",
                part_id="P0001",
                material_id=GREEN_STEEL,
                score=0.90,
                status="REVIEW",
            ),
        ],
        allowed_material_ids={
            GREEN_BASE,
            GREEN_STEEL,
            ORANGE_BASE,
        },
    )

    assert audit["status"] == ROUND_FALLBACK_BASELINE_INELIGIBLE
    assert audit["baseline_score_complete"] is False
    assert "GLOBAL_VIEW_COVERAGE_INCOMPLETE" in audit[
        "baseline_comparability_reason_codes"
    ]
    assert output == current


def test_group_step_rejects_whole_asset_scores_for_a_local_group() -> None:
    current = _plan()
    baseline = _bundle(
        current_plan=current,
        candidate_id="whole_asset_baseline",
        part_id=None,
        material_id=None,
        score=0.70,
    )
    baseline["quality_report"]["inputs"]["comparison_scope"] = {"mode": "whole_asset"}

    with pytest.raises(
        MultigroupExactMdlTournamentError,
        match="lacks exact group-local",
    ):
        select_exact_mdl_group_step(
            current_plan=current,
            group_id="G01",
            target_part_ids=["P0001"],
            candidates=[
                baseline,
                _bundle(
                    current_plan=current,
                    candidate_id="local_challenger",
                    part_id="P0001",
                    material_id=GREEN_STEEL,
                    score=0.90,
                ),
            ],
            allowed_material_ids={
                GREEN_BASE,
                GREEN_STEEL,
                ORANGE_BASE,
            },
        )


def test_group_step_global_pass_outranks_higher_local_only_score() -> None:
    current = _plan()
    baseline = _bundle(
        current_plan=current,
        candidate_id="baseline",
        part_id=None,
        material_id=None,
        score=0.91,
    )
    _set_quality(
        baseline,
        report_key="global_quality_report",
        score=0.82,
        status="REVIEW",
    )
    local_only = _bundle(
        current_plan=current,
        candidate_id="local_only",
        part_id="P0001",
        material_id=GREEN_PAINT,
        score=0.96,
    )
    _set_quality(
        local_only,
        report_key="global_quality_report",
        score=0.93,
        status="REVIEW",
    )
    globally_passing = _bundle(
        current_plan=current,
        candidate_id="globally_passing",
        part_id="P0001",
        material_id=GREEN_STEEL,
        score=0.88,
    )
    _set_quality(
        globally_passing,
        report_key="global_quality_report",
        score=0.80,
        status="PASS",
    )

    output, audit = select_exact_mdl_group_step(
        current_plan=current,
        group_id="G01",
        target_part_ids=["P0001"],
        candidates=[baseline, local_only, globally_passing],
        allowed_material_ids={
            GREEN_BASE,
            GREEN_PAINT,
            GREEN_STEEL,
            ORANGE_BASE,
        },
    )

    assert audit["status"] == ROUND_ACCEPTED
    assert audit["accepted_candidate_id"] == "globally_passing"
    assert audit["selection_scope"] == "whole_asset_guard"
    assert audit["all_view_pass_quality_promotion"] is True
    assert audit["selected_local_comparison"]["selection_score"] == pytest.approx(
        0.88
    )
    assert audit["selected_global_comparison"]["selection_score"] == pytest.approx(
        0.80
    )
    assert output["assignments"][0]["material_id"] == GREEN_STEEL


def test_group_step_uses_local_pass_while_all_global_views_still_fail() -> None:
    current = _plan()
    baseline = _bundle(
        current_plan=current,
        candidate_id="baseline",
        part_id=None,
        material_id=None,
        score=0.40,
        status="FAIL",
    )
    _set_quality(
        baseline,
        report_key="global_quality_report",
        score=0.40,
        status="FAIL",
    )
    challenger = _bundle(
        current_plan=current,
        candidate_id="local_pass",
        part_id="P0001",
        material_id=GREEN_STEEL,
        score=0.90,
        status="PASS",
    )
    _set_quality(
        challenger,
        report_key="global_quality_report",
        score=0.70,
        status="FAIL",
    )

    output, audit = select_exact_mdl_group_step(
        current_plan=current,
        group_id="G01",
        target_part_ids=["P0001"],
        candidates=[baseline, challenger],
        allowed_material_ids={
            GREEN_BASE,
            GREEN_STEEL,
            ORANGE_BASE,
        },
    )

    assert audit["status"] == ROUND_ACCEPTED
    assert audit["accepted_candidate_id"] == "local_pass"
    assert audit["selection_scope"] == (
        "canonical_group_local_with_whole_asset_nonregression_guard"
    )
    assert audit["selected_local_comparison"]["all_view_nonfail"] is True
    assert audit["selected_global_comparison"]["all_view_nonfail"] is False
    assert audit["global_regression_exclusion_count"] == 0
    assert output["assignments"][0]["material_id"] == GREEN_STEEL


def test_group_step_rejects_local_fail_even_when_global_guard_passes() -> None:
    current = _plan()
    baseline = _bundle(
        current_plan=current,
        candidate_id="baseline",
        part_id=None,
        material_id=None,
        score=0.60,
        status="REVIEW",
    )
    local_failure = _bundle(
        current_plan=current,
        candidate_id="local_failure",
        part_id="P0001",
        material_id=GREEN_PAINT,
        score=0.99,
        status="FAIL",
    )
    _set_quality(
        local_failure,
        report_key="global_quality_report",
        score=0.99,
        status="PASS",
    )
    valid = _bundle(
        current_plan=current,
        candidate_id="valid",
        part_id="P0001",
        material_id=GREEN_STEEL,
        score=0.80,
    )

    output, audit = select_exact_mdl_group_step(
        current_plan=current,
        group_id="G01",
        target_part_ids=["P0001"],
        candidates=[baseline, local_failure, valid],
        allowed_material_ids={
            GREEN_BASE,
            GREEN_PAINT,
            GREEN_STEEL,
            ORANGE_BASE,
        },
    )

    assert audit["accepted_candidate_id"] == "valid"
    assert audit["incomplete_challenger_count"] == 1
    assert audit["incomplete_challengers"][0]["candidate_id"] == "local_failure"
    assert (
        audit["incomplete_challengers"][0]["local"]["all_view_nonfail"] is False
    )
    assert output["assignments"][0]["material_id"] == GREEN_STEEL


def test_group_step_rejects_incomplete_global_guard() -> None:
    current = _plan()
    incomplete = _bundle(
        current_plan=current,
        candidate_id="incomplete_global",
        part_id="P0001",
        material_id=GREEN_PAINT,
        score=0.99,
    )
    incomplete["global_quality_report"]["views"].pop()

    output, audit = select_exact_mdl_group_step(
        current_plan=current,
        group_id="G01",
        target_part_ids=["P0001"],
        candidates=[
            _bundle(
                current_plan=current,
                candidate_id="baseline",
                part_id=None,
                material_id=None,
                score=0.60,
            ),
            incomplete,
            _bundle(
                current_plan=current,
                candidate_id="valid",
                part_id="P0001",
                material_id=GREEN_STEEL,
                score=0.80,
            ),
        ],
        allowed_material_ids={
            GREEN_BASE,
            GREEN_PAINT,
            GREEN_STEEL,
            ORANGE_BASE,
        },
    )

    assert audit["accepted_candidate_id"] == "valid"
    assert audit["incomplete_challenger_count"] == 1
    excluded = audit["incomplete_challengers"][0]
    assert excluded["candidate_id"] == "incomplete_global"
    assert excluded["global"]["complete"] is False
    assert "GLOBAL_PER_VIEW_SCORES_INCOMPLETE" in excluded["global"]["reason_codes"]
    assert output["assignments"][0]["material_id"] == GREEN_STEEL


def test_group_step_preserves_global_pass_against_review_challenger() -> None:
    current = _plan()
    challenger = _bundle(
        current_plan=current,
        candidate_id="review_challenger",
        part_id="P0001",
        material_id=GREEN_STEEL,
        score=0.99,
    )
    _set_quality(
        challenger,
        report_key="global_quality_report",
        score=0.99,
        status="REVIEW",
    )

    output, audit = select_exact_mdl_group_step(
        current_plan=current,
        group_id="G01",
        target_part_ids=["P0001"],
        candidates=[
            _bundle(
                current_plan=current,
                candidate_id="baseline",
                part_id=None,
                material_id=None,
                score=0.70,
            ),
            challenger,
        ],
        allowed_material_ids={
            GREEN_BASE,
            GREEN_STEEL,
            ORANGE_BASE,
        },
    )

    assert audit["accepted_candidate_id"] is None
    assert audit["selected_candidate_id"] == "baseline"
    assert audit["selected_all_view_pass"] is True
    assert output == current


def test_coordinate_descent_uses_accepted_plan_for_the_next_group() -> None:
    initial = _plan()
    observed_second_round_green: list[str] = []

    def provider(
        current_plan: Mapping[str, Any],
        group: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        current = copy.deepcopy(dict(current_plan))
        if group["group_id"] == "G01":
            return [
                _bundle(
                    current_plan=current,
                    candidate_id="g01_baseline",
                    part_id=None,
                    material_id=None,
                    score=0.70,
                ),
                _bundle(
                    current_plan=current,
                    candidate_id="g01_winner",
                    part_id="P0001",
                    material_id=GREEN_STEEL,
                    score=0.82,
                ),
            ]
        observed_second_round_green.append(
            next(
                item["material_id"]
                for item in current["assignments"]
                if item["part_id"] == "P0001"
            )
        )
        return [
            _bundle(
                current_plan=current,
                candidate_id="g02_baseline",
                part_id=None,
                material_id=None,
                score=0.78,
                target_group_id="G02",
                target_part_ids=("P0002",),
            ),
            _bundle(
                current_plan=current,
                candidate_id="g02_tiny_gain",
                part_id="P0002",
                material_id=ORANGE_COPPER,
                score=0.785,
                target_group_id="G02",
                target_part_ids=("P0002",),
            ),
        ]

    output, audit = coordinate_descent_exact_mdl_groups(
        initial_plan=initial,
        significant_groups=[
            {"group_id": "G01", "target_part_ids": ["P0001"]},
            {"group_id": "G02", "target_part_ids": ["P0002"]},
        ],
        round_provider=provider,
        allowed_material_ids={
            GREEN_BASE,
            GREEN_STEEL,
            ORANGE_BASE,
            ORANGE_COPPER,
        },
        minimum_score_improvement=0.02,
        minimum_winner_margin=0.005,
    )

    assert observed_second_round_green == [GREEN_STEEL]
    assert output["assignments"][0]["material_id"] == GREEN_STEEL
    assert output["assignments"][1]["material_id"] == ORANGE_BASE
    assert audit["all_significant_groups_evaluated"] is True
    assert audit["accepted_group_ids"] == ["G01"]
    assert audit["fallback_group_ids"] == ["G02"]
    assert audit["accepted_group_count"] == 1
    assert audit["fallback_group_count"] == 1
    assert (
        output["provenance"]["multigroup_exact_mdl_coordinate_descent"][
            "parameters_locked_to_library_defaults"
        ]
        is True
    )
    assert all(not assignment.get("parameters") for assignment in output["assignments"])


def test_coordinate_descent_allows_parent_and_subset_rounds_on_same_part() -> None:
    initial = _plan_with_face_subset()

    def provider(
        current_plan: Mapping[str, Any],
        group: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        current = copy.deepcopy(dict(current_plan))
        if group["group_id"] == "G01":
            return [
                _bundle(
                    current_plan=current,
                    candidate_id="g01_baseline",
                    part_id=None,
                    material_id=None,
                    score=0.70,
                ),
                _bundle(
                    current_plan=current,
                    candidate_id="g01_winner",
                    part_id="P0001",
                    material_id=GREEN_STEEL,
                    score=0.85,
                ),
            ]
        baseline = _bundle(
            current_plan=current,
            candidate_id="g02_subset_baseline",
            part_id=None,
            material_id=None,
            score=0.70,
            target_group_id="G02",
            target_part_ids=("P0001",),
            target_entities=(
                {
                    "entity_kind": "face_subset",
                    "part_id": "P0001",
                    "subset_name": "Cover",
                },
            ),
        )
        challenger = _bundle(
            current_plan=current,
            candidate_id="g02_subset_winner",
            part_id=None,
            material_id=None,
            score=0.88,
            target_group_id="G02",
            target_part_ids=("P0001",),
            target_entities=(
                {
                    "entity_kind": "face_subset",
                    "part_id": "P0001",
                    "subset_name": "Cover",
                },
            ),
        )
        challenger_plan = copy.deepcopy(current)
        challenger_plan["assignments"][0]["face_subsets"][0]["material_id"] = (
            ORANGE_COPPER
        )
        challenger["plan"] = challenger_plan
        challenger["apply_report"]["plan_sha256"] = _sha(challenger_plan)
        challenger["is_baseline"] = False
        return [baseline, challenger]

    output, audit = coordinate_descent_exact_mdl_groups(
        initial_plan=initial,
        significant_groups=[
            {
                "group_id": "G01",
                "target_part_ids": ["P0001"],
                "target_entities": [{"entity_kind": "assignment", "part_id": "P0001"}],
            },
            {
                "group_id": "G02",
                "target_part_ids": ["P0001"],
                "target_entities": [
                    {
                        "entity_kind": "face_subset",
                        "part_id": "P0001",
                        "subset_name": "Cover",
                    }
                ],
            },
        ],
        round_provider=provider,
        allowed_material_ids={
            GREEN_BASE,
            GREEN_STEEL,
            ORANGE_BASE,
            ORANGE_COPPER,
        },
    )

    assignment = output["assignments"][0]
    assert assignment["material_id"] == GREEN_STEEL
    assert assignment["face_subsets"][0]["material_id"] == ORANGE_COPPER
    assert assignment["parameters"] == {}
    assert assignment["face_subsets"][0]["parameters"] == {}
    assert audit["accepted_group_ids"] == ["G01", "G02"]
    assert audit["changed_part_ids"] == ["P0001"]
    assert audit["changed_entities"] == [
        {"entity_kind": "assignment", "part_id": "P0001"},
        {
            "entity_kind": "face_subset",
            "part_id": "P0001",
            "subset_name": "Cover",
        },
    ]
    assert "provenance" not in assignment["face_subsets"][0]


def test_finalizer_rejects_missing_significant_group_round() -> None:
    plan = _plan()
    with pytest.raises(
        MultigroupExactMdlTournamentError,
        match="not every significant group",
    ):
        finalize_multigroup_exact_mdl_plan(
            initial_plan=plan,
            current_plan=plan,
            significant_group_ids=["G01", "G02"],
            round_audits=[],
        )
