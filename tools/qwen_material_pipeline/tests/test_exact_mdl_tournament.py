from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

import pytest

from qwen_material_pipeline.materials.exact_mdl_tournament import (
    ExactMdlTournamentError,
    SELECTION_OBJECTIVE_VISUAL,
    build_bounded_exact_mdl_candidate_plans,
    build_part_family_contract,
    material_entity_contract_key,
    select_and_replay_exact_mdl_candidate,
)


METAL = "mdl:Metal/Green.mdl#Green"
PAINT = "mdl:Paint/Green.mdl#Green"
PLASTIC = "mdl:Plastic/Green.mdl#Green"


def _sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _plan(material_id: str) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "assignments": [
            {
                "part_id": "P0001",
                "material_id": material_id,
                "semantic": "painted machine enclosure",
                "status": "policy_fallback",
                "provenance": {"canonical_group_id": "G01"},
            }
        ],
        "provenance": {"asset_sha256": "source-sha"},
    }


def _candidate(
    candidate_id: str,
    material_id: str,
    score: float,
    *,
    texture_score: float | None = None,
    appearance_score: float | None = None,
) -> dict[str, Any]:
    texture_score = score if texture_score is None else texture_score
    appearance_score = score if appearance_score is None else appearance_score
    plan = _plan(material_id)
    registry_file_sha = f"{candidate_id}-registry-file-sha"
    output_sha = f"{candidate_id}-usd-sha"
    return {
        "candidate_id": candidate_id,
        "plan": plan,
        "apply_report": {
            "plan_sha256": _sha(plan),
            "output_sha256": output_sha,
        },
        "rendered_registry": {"asset_sha256": output_sha},
        "rendered_registry_file_sha256": registry_file_sha,
        "quality_report": {
            "schema_version": "qwen-reference-render-comparison/v1",
            "inputs": {"rendered_registry_sha256": registry_file_sha},
            "aggregate": {
                "status": "PASS",
                "material_match_conclusion": "PASS",
                "material_color_score": score,
                "material_texture_score": texture_score,
                "material_appearance_score": appearance_score,
                "texture_comparable_view_count": 2,
                "texture_unscorable_view_count": 0,
                "reference_view_count": 2,
                "comparable_view_count": 2,
                "passed_view_count": 2,
                "review_view_count": 0,
                "failed_view_count": 0,
                "unscorable_view_count": 0,
                "reference_view_coverage_status": "PASS",
            },
            "views": [
                {
                    "reference_view_id": "front",
                    "status": "PASS",
                    "material_color": {"score": score},
                    "material_texture": {"score": texture_score},
                    "material_appearance_score": appearance_score,
                },
                {
                    "reference_view_id": "side",
                    "status": "PASS",
                    "material_color": {"score": score},
                    "material_texture": {"score": texture_score},
                    "material_appearance_score": appearance_score,
                },
            ],
        },
    }


def test_family_contract_allows_paint_but_not_plastic_for_painted_metal() -> None:
    audit = {
        "G01": {
            "selection_group": {
                "family_hint": "metal",
                "finish_hint": "painted",
            },
            "retrieval_audit": {
                "applied_coating_confirmed": True,
                "surface_interpretation_policy": {
                    "family_reliable": True,
                    "semantic_surface_class": "coating",
                },
            },
        }
    }

    contract = build_part_family_contract(
        plan=_plan(METAL),
        material_choice_audit=audit,
    )

    assert contract == {"P0001": {"metal", "paint"}}


def test_family_contract_recovers_three_view_consensus_from_palette_fusion() -> None:
    audit = {
        "G01": {
            "selection_group": {
                "family_hint": "metal",
                "finish_hint": "painted",
            },
            "retrieval_audit": {
                "applied_coating_confirmed": True,
                "surface_interpretation_policy": {
                    "family_reliable": False,
                    "semantic_surface_class": "coating",
                },
            },
        }
    }
    palette_fusion = {
        "canonical_palette": {
            "groups": [
                {
                    "group_id": "G01",
                    "family_hint": "metal",
                    "sources": [
                        {
                            "view_id": "front",
                            "family_hint": "metal",
                            "confidence": 0.90,
                        },
                        {
                            "view_id": "side",
                            "family_hint": "metal",
                            "confidence": 0.91,
                        },
                        {
                            "view_id": "iso",
                            "family_hint": "metal",
                            "confidence": 0.70,
                        },
                        {
                            "view_id": "top",
                            "family_hint": "plastic",
                            "confidence": 0.84,
                        },
                    ],
                }
            ]
        }
    }

    contract = build_part_family_contract(
        plan=_plan(METAL),
        material_choice_audit=audit,
        palette_fusion=palette_fusion,
    )

    assert contract == {"P0001": {"metal", "paint"}}


def test_family_contract_keeps_face_subset_family_separate_from_parent() -> None:
    plan = _plan(METAL)
    plan["assignments"][0]["face_subsets"] = [
        {
            "subset_name": "Cover",
            "semantic": "plastic cover",
            "material_id": PLASTIC,
            "face_indices": [1, 2],
        }
    ]
    plan["assignments"][0]["provenance"]["face_subset_canonical_group_ids"] = {
        "Cover": "G02"
    }
    audit = {
        "G01": {
            "selection_group": {
                "family_hint": "metal",
                "finish_hint": "painted",
            },
            "retrieval_audit": {
                "applied_coating_confirmed": True,
                "surface_interpretation_policy": {
                    "family_reliable": True,
                    "semantic_surface_class": "coating",
                },
            },
        },
        "G02": {
            "selection_group": {
                "family_hint": "plastic",
                "finish_hint": "matte",
            },
            "retrieval_audit": {
                "surface_interpretation_policy": {
                    "family_reliable": True,
                    "semantic_surface_class": "bulk",
                },
            },
        },
    }

    contract = build_part_family_contract(
        plan=plan,
        material_choice_audit=audit,
    )

    assert contract["P0001"] == {"metal", "paint"}
    assert contract[material_entity_contract_key("P0001", "Cover")] == {"plastic"}


def test_tournament_rejects_better_colored_cross_family_candidate() -> None:
    baseline = _plan(METAL)
    output, audit = select_and_replay_exact_mdl_candidate(
        baseline_plan=baseline,
        target_plan=copy.deepcopy(baseline),
        candidates=[
            _candidate("compatible_paint", PAINT, 0.76),
            _candidate("wrong_plastic", PLASTIC, 0.95),
        ],
        allowed_material_ids={METAL, PAINT, PLASTIC},
        material_families_by_id={
            METAL: "metal",
            PAINT: "paint",
            PLASTIC: "plastic",
        },
        allowed_families_by_part={"P0001": {"metal", "paint"}},
    )

    assert output["assignments"][0]["material_id"] == PAINT
    assert audit["selected_candidate_id"] == "compatible_paint"
    candidates = {item["candidate_id"]: item for item in audit["candidates"]}
    assert candidates["wrong_plastic"]["visual_eligible"] is True
    assert candidates["wrong_plastic"]["semantic_eligible"] is False
    assert candidates["wrong_plastic"]["eligible"] is False
    assert candidates["wrong_plastic"]["semantic_reason_codes"] == [
        "MATERIAL_FAMILY_CONFLICTS_WITH_MULTIVIEW_CONSENSUS"
    ]


def test_visual_tournament_selects_better_cross_family_candidate() -> None:
    baseline = _plan(METAL)
    output, audit = select_and_replay_exact_mdl_candidate(
        baseline_plan=baseline,
        target_plan=copy.deepcopy(baseline),
        candidates=[
            _candidate("compatible_paint", PAINT, 0.76),
            _candidate("better_plastic", PLASTIC, 0.95),
        ],
        allowed_material_ids={METAL, PAINT, PLASTIC},
        material_families_by_id={
            METAL: "metal",
            PAINT: "paint",
            PLASTIC: "plastic",
        },
        allowed_families_by_part={"P0001": {"metal", "paint"}},
        selection_objective=SELECTION_OBJECTIVE_VISUAL,
    )

    assert output["assignments"][0]["material_id"] == PLASTIC
    assert audit["selected_candidate_id"] == "better_plastic"
    assert audit["selection_objective"] == SELECTION_OBJECTIVE_VISUAL
    assert audit["semantic_family_gate_applied"] is False
    candidates = {item["candidate_id"]: item for item in audit["candidates"]}
    assert candidates["better_plastic"]["visual_eligible"] is True
    assert candidates["better_plastic"]["semantic_eligible"] is False
    assert candidates["better_plastic"]["eligible"] is True


def test_visual_tournament_replays_face_subset_material_without_schema_mutation() -> (
    None
):
    baseline = _plan(METAL)
    baseline["assignments"][0]["parameters"] = {}
    baseline["assignments"][0]["face_subsets"] = [
        {
            "subset_name": "Cover",
            "semantic": "green cover",
            "material_id": PAINT,
            "parameters": {},
            "face_indices": [1, 2, 3],
        }
    ]

    def subset_candidate(
        candidate_id: str,
        material_id: str,
        score: float,
    ) -> dict[str, Any]:
        bundle = _candidate(candidate_id, METAL, score)
        plan = copy.deepcopy(baseline)
        plan["assignments"][0]["face_subsets"][0]["material_id"] = material_id
        bundle["plan"] = plan
        bundle["apply_report"]["plan_sha256"] = _sha(plan)
        return bundle

    output, audit = select_and_replay_exact_mdl_candidate(
        baseline_plan=baseline,
        target_plan=copy.deepcopy(baseline),
        candidates=[
            subset_candidate("subset_baseline", PAINT, 0.72),
            subset_candidate("subset_visual_winner", PLASTIC, 0.91),
        ],
        allowed_material_ids={METAL, PAINT, PLASTIC},
        selection_objective=SELECTION_OBJECTIVE_VISUAL,
    )

    assignment = output["assignments"][0]
    subset = assignment["face_subsets"][0]
    assert assignment["material_id"] == METAL
    assert subset["material_id"] == PLASTIC
    assert subset["face_indices"] == [1, 2, 3]
    assert assignment["parameters"] == {}
    assert subset["parameters"] == {}
    assert "provenance" not in subset
    assert (
        assignment["provenance"]["exact_mdl_face_subset_tournament"]["Cover"][
            "parameters_locked_to_library_defaults"
        ]
        is True
    )
    assert audit["selected_material_changes"] == [
        {
            "part_id": "P0001",
            "subset_name": "Cover",
            "old_material_id": PAINT,
            "new_material_id": PLASTIC,
        }
    ]


@pytest.mark.parametrize("parameters", [[], "roughness=0.5", 0.5])
def test_tournament_rejects_nonobject_assignment_parameters(parameters: Any) -> None:
    baseline = _plan(METAL)
    baseline["assignments"][0]["parameters"] = parameters

    with pytest.raises(
        ExactMdlTournamentError,
        match="modifies selected MDL parameters",
    ):
        select_and_replay_exact_mdl_candidate(
            baseline_plan=baseline,
            target_plan=copy.deepcopy(baseline),
            candidates=[
                _candidate("baseline", METAL, 0.70),
                _candidate("challenger", PAINT, 0.90),
            ],
            allowed_material_ids={METAL, PAINT},
            selection_objective=SELECTION_OBJECTIVE_VISUAL,
        )


def test_tournament_rejects_face_subset_reordering() -> None:
    baseline = _plan(METAL)
    baseline["assignments"][0]["face_subsets"] = [
        {
            "subset_name": "First",
            "material_id": PAINT,
            "face_indices": [1],
        },
        {
            "subset_name": "Second",
            "material_id": PLASTIC,
            "face_indices": [2],
        },
    ]
    baseline_bundle = _candidate("baseline", METAL, 0.70)
    baseline_bundle["plan"] = copy.deepcopy(baseline)
    baseline_bundle["apply_report"]["plan_sha256"] = _sha(baseline_bundle["plan"])
    reordered_bundle = _candidate("reordered", METAL, 0.90)
    reordered_plan = copy.deepcopy(baseline)
    reordered_plan["assignments"][0]["face_subsets"].reverse()
    reordered_bundle["plan"] = reordered_plan
    reordered_bundle["apply_report"]["plan_sha256"] = _sha(reordered_plan)

    with pytest.raises(ExactMdlTournamentError, match="face-subset order"):
        select_and_replay_exact_mdl_candidate(
            baseline_plan=baseline,
            target_plan=copy.deepcopy(baseline),
            candidates=[baseline_bundle, reordered_bundle],
            allowed_material_ids={METAL, PAINT, PLASTIC},
            selection_objective=SELECTION_OBJECTIVE_VISUAL,
        )


def test_visual_tournament_ranks_complete_appearance_not_color_alone() -> None:
    baseline = _plan(METAL)
    output, audit = select_and_replay_exact_mdl_candidate(
        baseline_plan=baseline,
        target_plan=copy.deepcopy(baseline),
        candidates=[
            _candidate(
                "color_only_leader",
                PAINT,
                0.93,
                texture_score=0.58,
                appearance_score=0.735,
            ),
            _candidate(
                "appearance_leader",
                PLASTIC,
                0.84,
                texture_score=0.86,
                appearance_score=0.85,
            ),
        ],
        allowed_material_ids={METAL, PAINT, PLASTIC},
        selection_objective=SELECTION_OBJECTIVE_VISUAL,
    )

    assert output["assignments"][0]["material_id"] == PLASTIC
    assert audit["selected_candidate_id"] == "appearance_leader"
    assert audit["selected_material_color_score"] == pytest.approx(0.84)
    assert audit["selected_material_texture_score"] == pytest.approx(0.86)
    assert audit["selected_material_appearance_score"] == pytest.approx(0.85)


def test_visual_tournament_requires_complete_texture_evidence() -> None:
    baseline = _plan(METAL)
    missing_texture = _candidate("missing_texture", PAINT, 0.99)
    for key in (
        "material_texture_score",
        "material_appearance_score",
        "texture_comparable_view_count",
        "texture_unscorable_view_count",
    ):
        missing_texture["quality_report"]["aggregate"].pop(key)

    output, audit = select_and_replay_exact_mdl_candidate(
        baseline_plan=baseline,
        target_plan=copy.deepcopy(baseline),
        candidates=[
            missing_texture,
            _candidate("complete_appearance", PLASTIC, 0.80),
        ],
        allowed_material_ids={METAL, PAINT, PLASTIC},
        selection_objective=SELECTION_OBJECTIVE_VISUAL,
    )

    assert output["assignments"][0]["material_id"] == PLASTIC
    records = {item["candidate_id"]: item for item in audit["candidates"]}
    assert records["missing_texture"]["visual_eligible"] is False
    assert records["missing_texture"]["visual_eligibility_reason_codes"] == [
        "COMPLETE_TEXTURE_APPEARANCE_EVIDENCE_REQUIRED"
    ]


def test_visual_tournament_keeps_all_view_pass_as_a_hard_gate() -> None:
    baseline = _plan(METAL)
    brighter_review = _candidate("brighter_review", PLASTIC, 0.95)
    brighter_review["quality_report"]["aggregate"].update(
        {
            "status": "REVIEW",
            "material_match_conclusion": "NOT_CONCLUSIVE",
            "passed_view_count": 0,
            "review_view_count": 2,
        }
    )
    for view in brighter_review["quality_report"]["views"]:
        view["status"] = "REVIEW"

    output, audit = select_and_replay_exact_mdl_candidate(
        baseline_plan=baseline,
        target_plan=copy.deepcopy(baseline),
        candidates=[
            _candidate("all_view_pass", PAINT, 0.76),
            brighter_review,
        ],
        allowed_material_ids={METAL, PAINT, PLASTIC},
        material_families_by_id={
            METAL: "metal",
            PAINT: "paint",
            PLASTIC: "plastic",
        },
        allowed_families_by_part={"P0001": {"metal", "paint"}},
        selection_objective=SELECTION_OBJECTIVE_VISUAL,
    )

    assert output["assignments"][0]["material_id"] == PAINT
    assert audit["selected_candidate_id"] == "all_view_pass"


def test_visual_tournament_can_rank_complete_nonfailing_multiview_reviews() -> None:
    baseline = _plan(METAL)

    def reviewed(candidate_id: str, material_id: str, score: float) -> dict[str, Any]:
        candidate = _candidate(candidate_id, material_id, score)
        candidate["quality_report"]["aggregate"].update(
            {
                "status": "REVIEW",
                "material_match_conclusion": "NOT_CONCLUSIVE",
                "texture_comparable_view_count": 3,
                "reference_view_count": 3,
                "comparable_view_count": 3,
                "passed_view_count": 0,
                "review_view_count": 3,
            }
        )
        for view in candidate["quality_report"]["views"]:
            view["status"] = "REVIEW"
        candidate["quality_report"]["views"].append(
            {
                "reference_view_id": "top",
                "status": "REVIEW",
                "material_color": {"score": score},
                "material_texture": {"score": score},
                "material_appearance_score": score,
            }
        )
        return candidate

    output, audit = select_and_replay_exact_mdl_candidate(
        baseline_plan=baseline,
        target_plan=copy.deepcopy(baseline),
        candidates=[
            reviewed("review_runner", PAINT, 0.80),
            reviewed("review_winner", PLASTIC, 0.86),
        ],
        allowed_material_ids={METAL, PAINT, PLASTIC},
        selection_objective=SELECTION_OBJECTIVE_VISUAL,
    )

    assert output["assignments"][0]["material_id"] == PLASTIC
    assert audit["selected_candidate_id"] == "review_winner"
    assert audit["selected_quality_tier"] == "COMPLETE_NONFAIL_REVIEW"
    assert audit["complete_nonfail_review_candidate_count"] == 2


def test_tournament_returns_auditable_failure_when_only_pass_is_wrong_family() -> None:
    baseline = _plan(METAL)
    visual_review = _candidate("compatible_review", PAINT, 0.80)
    visual_review["quality_report"]["aggregate"].update(
        {
            "status": "REVIEW",
            "material_match_conclusion": "NOT_CONCLUSIVE",
            "passed_view_count": 0,
            "review_view_count": 2,
        }
    )
    for view in visual_review["quality_report"]["views"]:
        view["status"] = "REVIEW"

    with pytest.raises(ExactMdlTournamentError) as captured:
        select_and_replay_exact_mdl_candidate(
            baseline_plan=baseline,
            target_plan=copy.deepcopy(baseline),
            candidates=[
                visual_review,
                _candidate("wrong_plastic", PLASTIC, 0.95),
            ],
            allowed_material_ids={METAL, PAINT, PLASTIC},
            material_families_by_id={
                METAL: "metal",
                PAINT: "paint",
                PLASTIC: "plastic",
            },
            allowed_families_by_part={"P0001": {"metal", "paint"}},
        )

    assert captured.value.audit is not None
    assert captured.value.audit["status"] == "NO_ELIGIBLE_CANDIDATE"
    assert captured.value.audit["eligible_candidate_count"] == 0


def test_candidate_planner_targets_dominant_reliable_group_and_excludes_plastic() -> (
    None
):
    source = _plan(METAL)
    source["assignments"].append(
        {
            "part_id": "P0002",
            "material_id": METAL,
            "semantic": "small painted bracket",
            "status": "policy_fallback",
            "provenance": {"canonical_group_id": "G02"},
        }
    )
    audit = {
        group_id: {
            "selection_group": {
                "family_hint": "metal",
                "finish_hint": "painted",
            },
            "retrieval_audit": {
                "applied_coating_confirmed": True,
                "surface_interpretation_policy": {
                    "family_reliable": True,
                    "semantic_surface_class": "coating",
                },
            },
        }
        for group_id in ("G01", "G02")
    }
    palette_fusion = {
        "canonical_palette": {
            "groups": [
                {
                    "group_id": "G01",
                    "family_hint": "metal",
                    "sources": [
                        {
                            "view_id": "front",
                            "family_hint": "metal",
                            "confidence": 0.9,
                            "boxes": [[0, 0, 900, 900]],
                        },
                        {
                            "view_id": "side",
                            "family_hint": "metal",
                            "confidence": 0.9,
                            "boxes": [[0, 0, 800, 800]],
                        },
                    ],
                },
                {
                    "group_id": "G02",
                    "family_hint": "metal",
                    "sources": [
                        {
                            "view_id": "front",
                            "family_hint": "metal",
                            "confidence": 0.95,
                            "boxes": [[0, 0, 100, 100]],
                        },
                        {
                            "view_id": "side",
                            "family_hint": "metal",
                            "confidence": 0.95,
                            "boxes": [[0, 0, 100, 100]],
                        },
                    ],
                },
            ]
        }
    }
    material_candidates = {
        group_id: {
            "candidates": [
                {"material_id": PAINT, "family": "paint"},
                {"material_id": PLASTIC, "family": "plastic"},
            ]
        }
        for group_id in ("G01", "G02")
    }

    candidates, planning = build_bounded_exact_mdl_candidate_plans(
        source_plan=source,
        material_candidates_by_group=material_candidates,
        material_choice_audit=audit,
        palette_fusion=palette_fusion,
        allowed_material_ids={METAL, PAINT, PLASTIC},
        maximum_candidates=4,
    )

    assert planning["selected_group_id"] == "G01"
    assert planning["candidate_material_ids"] == [METAL, PAINT]
    assert len(candidates) == 2
    assert candidates[1]["plan"]["assignments"][0]["material_id"] == PAINT
    assert candidates[1]["plan"]["assignments"][1]["material_id"] == METAL


def test_visual_candidate_planner_keeps_cross_family_candidates() -> None:
    source = _plan(METAL)
    audit = {
        "G01": {
            "selection_group": {
                "family_hint": "metal",
                "finish_hint": "painted",
            },
            "retrieval_audit": {
                "applied_coating_confirmed": True,
                "surface_interpretation_policy": {
                    "family_reliable": True,
                    "semantic_surface_class": "coating",
                },
            },
        }
    }
    palette_fusion = {
        "canonical_palette": {
            "groups": [
                {
                    "group_id": "G01",
                    "family_hint": "metal",
                    "sources": [
                        {
                            "view_id": "front",
                            "family_hint": "metal",
                            "confidence": 0.9,
                            "boxes": [[0, 0, 900, 900]],
                        },
                        {
                            "view_id": "side",
                            "family_hint": "metal",
                            "confidence": 0.9,
                            "boxes": [[0, 0, 800, 800]],
                        },
                    ],
                }
            ]
        }
    }
    material_candidates = {
        "G01": {
            "tournament_candidates": [
                {"material_id": PAINT, "family": "paint"},
                {"material_id": PLASTIC, "family": "plastic"},
            ],
            "candidates": [{"material_id": PAINT, "family": "paint"}],
        }
    }

    candidates, planning = build_bounded_exact_mdl_candidate_plans(
        source_plan=source,
        material_candidates_by_group=material_candidates,
        material_choice_audit=audit,
        palette_fusion=palette_fusion,
        allowed_material_ids={METAL, PAINT, PLASTIC},
        maximum_candidates=4,
        selection_objective=SELECTION_OBJECTIVE_VISUAL,
    )

    assert planning["candidate_material_ids"] == [METAL, PAINT, PLASTIC]
    assert planning["semantic_family_gate_applied"] is False
    assert [candidate["material_id"] for candidate in candidates] == [
        METAL,
        PAINT,
        PLASTIC,
    ]


def test_candidate_planner_uses_extended_tournament_pool_without_expanding_qwen_pool() -> (
    None
):
    alternative = "mdl:Paint/Alternative.mdl#Alternative"
    source = _plan(METAL)
    audit = {
        "G01": {
            "selection_group": {
                "family_hint": "metal",
                "finish_hint": "painted",
            },
            "retrieval_audit": {
                "applied_coating_confirmed": True,
                "surface_interpretation_policy": {
                    "family_reliable": True,
                    "semantic_surface_class": "coating",
                },
            },
        }
    }
    palette_fusion = {
        "canonical_palette": {
            "groups": [
                {
                    "group_id": "G01",
                    "family_hint": "metal",
                    "sources": [
                        {
                            "view_id": "front",
                            "family_hint": "metal",
                            "confidence": 0.9,
                            "boxes": [[0, 0, 900, 900]],
                        },
                        {
                            "view_id": "side",
                            "family_hint": "metal",
                            "confidence": 0.9,
                            "boxes": [[0, 0, 800, 800]],
                        },
                    ],
                }
            ]
        }
    }
    material_candidates = {
        "G01": {
            "candidates": [{"material_id": PAINT, "family": "paint"}],
            "tournament_candidates": [
                {"material_id": PAINT, "family": "paint"},
                {"material_id": alternative, "family": "paint"},
                {"material_id": PLASTIC, "family": "plastic"},
            ],
        }
    }

    candidates, planning = build_bounded_exact_mdl_candidate_plans(
        source_plan=source,
        material_candidates_by_group=material_candidates,
        material_choice_audit=audit,
        palette_fusion=palette_fusion,
        allowed_material_ids={METAL, PAINT, alternative, PLASTIC},
        maximum_candidates=4,
    )

    assert planning["candidate_material_ids"] == [METAL, PAINT, alternative]
    assert [candidate["material_id"] for candidate in candidates] == [
        METAL,
        PAINT,
        alternative,
    ]
