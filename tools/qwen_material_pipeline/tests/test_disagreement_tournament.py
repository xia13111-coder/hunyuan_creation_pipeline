from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from asset_pipeline.visual_materials.orchestrator import (
    _baseline_preserved_disagreement_exemptions,
    _final_baseline_preserved_disagreement_exemptions,
)
from qwen_material_pipeline.materials.disagreement_tournament import (
    DisagreementTournamentContractError,
    build_disagreement_tournament_contract,
    disagreement_is_render_confirmed,
    validate_disagreement_tournament_contract,
)


FORWARD = "mdl:vMaterials_2/Metal/Aluminum.mdl#Grass_Green"
REVERSE = "mdl:vMaterials_2/Metal/Steel.mdl#Army_Green"
NEIGHBOR = "mdl:vMaterials_2/Plastic/Polypropylene.mdl#Green"
RETRIEVAL_CHALLENGER = "mdl:Base/Masonry/Concrete_Smooth.mdl#Concrete_Smooth"


def _contract() -> dict:
    return build_disagreement_tournament_contract(
        forward_material_id=FORWARD,
        reverse_material_id=REVERSE,
        provisional_seed_material_id=FORWARD,
        mvinverse_exact_default_material_ids=[NEIGHBOR],
        tournament_candidate_material_ids=[FORWARD, REVERSE, NEIGHBOR],
    )


def test_contract_requires_both_qwen_choices_and_mvinverse_default() -> None:
    contract = _contract()

    validated = validate_disagreement_tournament_contract(
        contract,
        forward_material_id=FORWARD,
        reverse_material_id=REVERSE,
        tournament_candidate_material_ids=[FORWARD, REVERSE, NEIGHBOR],
    )

    assert validated["provisional_seed_is_final_selection"] is False
    assert validated["required_candidate_material_ids"] == [
        FORWARD,
        REVERSE,
        NEIGHBOR,
    ]
    assert validated["selected_mdl_parameters_mutable"] is False
    assert validated["library_default_parameters_required"] is True


def test_contract_rejects_queue_that_drops_reverse_choice() -> None:
    with pytest.raises(
        DisagreementTournamentContractError,
        match="omits required disagreement evidence",
    ):
        build_disagreement_tournament_contract(
            forward_material_id=FORWARD,
            reverse_material_id=REVERSE,
            provisional_seed_material_id=FORWARD,
            mvinverse_exact_default_material_ids=[NEIGHBOR],
            tournament_candidate_material_ids=[FORWARD, NEIGHBOR],
        )


def test_contract_accepts_ranked_retrieval_when_mvinverse_is_unavailable() -> None:
    contract = build_disagreement_tournament_contract(
        forward_material_id=FORWARD,
        reverse_material_id=REVERSE,
        provisional_seed_material_id=FORWARD,
        mvinverse_exact_default_material_ids=[],
        retrieval_exact_default_material_ids=[RETRIEVAL_CHALLENGER],
        tournament_candidate_material_ids=[
            FORWARD,
            REVERSE,
            RETRIEVAL_CHALLENGER,
        ],
    )

    validated = validate_disagreement_tournament_contract(
        contract,
        forward_material_id=FORWARD,
        reverse_material_id=REVERSE,
        tournament_candidate_material_ids=[
            FORWARD,
            REVERSE,
            RETRIEVAL_CHALLENGER,
        ],
    )

    assert validated["schema_version"].endswith("/v2")
    assert validated["mvinverse_exact_default_material_ids"] == []
    assert validated["retrieval_exact_default_material_ids"] == [
        RETRIEVAL_CHALLENGER
    ]
    assert validated["independent_exact_default_candidates"] == [
        {
            "material_id": RETRIEVAL_CHALLENGER,
            "evidence_basis": (
                "ranked_retrieval_independent_exact_library_default_fallback"
            ),
        }
    ]


def test_contract_rejects_disputed_choice_as_independent_challenger() -> None:
    with pytest.raises(
        DisagreementTournamentContractError,
        match="repeats a disputed Qwen choice",
    ):
        build_disagreement_tournament_contract(
            forward_material_id=FORWARD,
            reverse_material_id=REVERSE,
            provisional_seed_material_id=FORWARD,
            mvinverse_exact_default_material_ids=[],
            retrieval_exact_default_material_ids=[FORWARD],
            tournament_candidate_material_ids=[FORWARD, REVERSE],
        )


def test_validator_rejects_tampered_challenger_provenance() -> None:
    contract = build_disagreement_tournament_contract(
        forward_material_id=FORWARD,
        reverse_material_id=REVERSE,
        provisional_seed_material_id=FORWARD,
        mvinverse_exact_default_material_ids=[],
        retrieval_exact_default_material_ids=[RETRIEVAL_CHALLENGER],
        tournament_candidate_material_ids=[
            FORWARD,
            REVERSE,
            RETRIEVAL_CHALLENGER,
        ],
    )
    contract["independent_exact_default_candidates"][0][
        "evidence_basis"
    ] = "mvinverse_nearest_exact_library_default"

    with pytest.raises(
        DisagreementTournamentContractError,
        match="provenance is inconsistent",
    ):
        validate_disagreement_tournament_contract(
            contract,
            forward_material_id=FORWARD,
            reverse_material_id=REVERSE,
            tournament_candidate_material_ids=[
                FORWARD,
                REVERSE,
                RETRIEVAL_CHALLENGER,
            ],
        )


def test_validator_remains_compatible_with_v1_contracts() -> None:
    legacy = {
        "schema_version": "qwen-forward-reverse-exact-mdl-tournament/v1",
        "required": True,
        "reason_code": "FORWARD_REVERSE_EXACT_MDL_DISAGREEMENT",
        "resolution_policy": (
            "render_confirmation_required_before_material_lock/v1"
        ),
        "forward_material_id": FORWARD,
        "reverse_material_id": REVERSE,
        "provisional_seed_material_id": FORWARD,
        "provisional_seed_is_final_selection": False,
        "mvinverse_exact_default_material_ids": [NEIGHBOR],
        "required_candidate_material_ids": [FORWARD, REVERSE, NEIGHBOR],
        "selected_mdl_parameters_mutable": False,
        "library_default_parameters_required": True,
    }

    assert validate_disagreement_tournament_contract(
        legacy,
        forward_material_id=FORWARD,
        reverse_material_id=REVERSE,
        tournament_candidate_material_ids=[FORWARD, REVERSE, NEIGHBOR],
    ) == legacy


def test_nonpass_seed_threshold_fallback_is_not_render_confirmation() -> None:
    assert (
        disagreement_is_render_confirmed(
            _contract(),
            round_audit={
                "status": "FALLBACK_INSUFFICIENT_IMPROVEMENT",
                "baseline_candidate_id": "g06_01",
                "selected_candidate_id": "g06_09",
                "accepted_candidate_id": None,
                "baseline_all_view_pass": False,
            },
        )
        is False
    )


def test_all_view_pass_challenger_resolves_disagreement() -> None:
    assert (
        disagreement_is_render_confirmed(
            _contract(),
            round_audit={
                "status": "ACCEPTED",
                "baseline_candidate_id": "g06_01",
                "selected_candidate_id": "g06_09",
                "accepted_candidate_id": "g06_09",
                "baseline_all_view_pass": False,
            },
        )
        is True
    )


def test_all_view_pass_seed_may_survive_only_as_best_rendered_candidate() -> None:
    assert (
        disagreement_is_render_confirmed(
            _contract(),
            round_audit={
                "status": "FALLBACK_BASELINE_BEST",
                "baseline_candidate_id": "g06_01",
                "selected_candidate_id": "g06_01",
                "accepted_candidate_id": None,
                "baseline_all_view_pass": True,
            },
        )
        is True
    )


def _baseline_preserved_queue_audit() -> dict:
    return {
        "groups": [{"group_id": "G01", "target_entity_count": 1}],
        "coverage_blockers": [],
        "excluded_groups": [
            {
                "group_id": "G02",
                "reason": "BASELINE_GROUP_PRESENT_WITHOUT_LOCALIZATION",
                "baseline_preserved": True,
                "authored_target_entity_count": 0,
                "baseline_presence_evidence": {
                    "canonical_group_id": "G02",
                    "all_source_views_present": True,
                    "reference_view_ids": ["front", "side"],
                    "views": [
                        {
                            "reference_view_id": "front",
                            "trusted_reference_evidence": True,
                            "delivery_presence_status": "PRESENT",
                            "recall": 1.0,
                        },
                        {
                            "reference_view_id": "side",
                            "trusted_reference_evidence": True,
                            "delivery_presence_status": "PRESENT",
                            "recall": 1.0,
                        },
                    ],
                },
            }
        ],
    }


def _disagreement_choices() -> dict:
    return {
        "G02": {
            "confirmation_basis": "forward_reverse_disagreement",
            "disagreement_tournament": {
                "required": True,
                "schema_version": (
                    "qwen-forward-reverse-exact-mdl-tournament/v1"
                ),
            },
        }
    }


def test_unused_baseline_present_disagreement_is_audited_as_exempt() -> None:
    exemptions = _baseline_preserved_disagreement_exemptions(
        queue_audit=_baseline_preserved_queue_audit(),
        material_choice_audit=_disagreement_choices(),
    )

    assert [record["group_id"] for record in exemptions] == ["G02"]
    assert exemptions[0]["render_confirmation_required"] is False
    assert exemptions[0]["material_selection_was_authored"] is False
    assert exemptions[0]["authored_target_entity_count"] == 0
    assert exemptions[0]["reason_code"] == (
        "FORWARD_REVERSE_DISAGREEMENT_UNUSED_BASELINE_"
        "PRESENT_WITHOUT_AUTHORED_TARGET"
    )


def test_unused_single_view_disagreement_is_presence_only_exempt() -> None:
    queue_audit = _baseline_preserved_queue_audit()
    exclusion = queue_audit["excluded_groups"][0]
    exclusion.update(
        {
            "reason": "INSUFFICIENT_INDEPENDENT_REFERENCE_VIEWS",
            "reference_view_ids": ["front"],
            "reference_view_count": 1,
        }
    )
    presence = exclusion["baseline_presence_evidence"]
    presence.update(
        {
            "reference_view_ids": ["front"],
            "reference_view_count": 1,
            "minimum_source_view_count": 1,
            "views": [presence["views"][0]],
        }
    )

    exemptions = _baseline_preserved_disagreement_exemptions(
        queue_audit=queue_audit,
        material_choice_audit=_disagreement_choices(),
    )

    assert [record["group_id"] for record in exemptions] == ["G02"]
    exemption = exemptions[0]
    assert exemption["render_confirmation_required"] is False
    assert exemption["material_selection_was_authored"] is False
    assert exemption["material_choice_resolved"] is False
    assert exemption["presence_only_not_material_identity_confirmation"] is True
    assert exemption["queue_exclusion_reason"] == (
        "INSUFFICIENT_INDEPENDENT_REFERENCE_VIEWS"
    )


def test_authored_single_view_disagreement_is_never_exempt() -> None:
    queue_audit = _baseline_preserved_queue_audit()
    exclusion = queue_audit["excluded_groups"][0]
    exclusion.update(
        {
            "reason": "INSUFFICIENT_INDEPENDENT_REFERENCE_VIEWS",
            "reference_view_ids": ["front"],
            "reference_view_count": 1,
            "authored_target_entity_count": 1,
        }
    )
    presence = exclusion["baseline_presence_evidence"]
    presence.update(
        {
            "reference_view_ids": ["front"],
            "reference_view_count": 1,
            "minimum_source_view_count": 1,
            "views": [presence["views"][0]],
        }
    )

    assert (
        _baseline_preserved_disagreement_exemptions(
            queue_audit=queue_audit,
            material_choice_audit=_disagreement_choices(),
        )
        == []
    )


def _final_single_view_exemption_inputs(
    tmp_path: Path,
) -> tuple[dict, dict, dict, dict, Path, Path]:
    queue_audit = _baseline_preserved_queue_audit()
    exclusion = queue_audit["excluded_groups"][0]
    exclusion.update(
        {
            "reason": "INSUFFICIENT_INDEPENDENT_REFERENCE_VIEWS",
            "reference_view_ids": ["top"],
            "reference_view_count": 1,
            "authored_target_entity_count": 0,
        }
    )
    final_plan = {
        "schema_version": "1.0",
        "assignments": [
            {
                "part_id": "P0001",
                # Reusing a disputed library identity under another canonical
                # group must not make the unused G02 choice look authored.
                "material_id": FORWARD,
                "provenance": {"canonical_group_id": "G01"},
            }
        ],
    }
    palette_fusion = {
        "view_group_id_maps": {
            "top": {
                "L02": "G02",
            }
        }
    }
    rendered_registry_path = tmp_path / "part_registry.rendered.json"
    rendered_registry_path.write_text(
        json.dumps(
            {
                "schema_version": "qwen-material-parts/v1",
                "asset_sha256": "rendered-asset",
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    rendered_registry_sha256 = hashlib.sha256(
        rendered_registry_path.read_bytes()
    ).hexdigest()
    quality_report = {
        "schema_version": "qwen-reference-render-comparison/v1",
        "inputs": {
            "comparison_scope": {"mode": "whole_asset"},
            "rendered_registry": str(rendered_registry_path),
            "rendered_registry_sha256": rendered_registry_sha256,
        },
        "aggregate": {
            "reference_view_coverage_status": "PASS",
            "reference_view_count": 1,
        },
        "views": [
            {
                "reference_view_id": "top",
                "reference": {
                    "trusted_evidence": {
                        "usable": True,
                        "reasons": [],
                    }
                },
                "material_color": {
                    "trusted_evidence_group_recall": {
                        "group_count": 1,
                        "groups": [
                            {
                                "group_id": "L02",
                                "delivery_presence_status": "PRESENT",
                                "recall": 1.0,
                            }
                        ],
                    }
                },
            }
        ],
    }
    quality_report_path = tmp_path / "final_reference_render_comparison.json"
    quality_report_path.write_text(
        json.dumps(quality_report, sort_keys=True),
        encoding="utf-8",
    )
    return (
        queue_audit,
        _disagreement_choices(),
        final_plan,
        palette_fusion,
        quality_report_path,
        rendered_registry_path,
    )


def test_final_single_view_unused_exemption_is_hash_bound_and_unresolved(
    tmp_path: Path,
) -> None:
    (
        queue_audit,
        choices,
        final_plan,
        palette_fusion,
        quality_report_path,
        rendered_registry_path,
    ) = _final_single_view_exemption_inputs(tmp_path)

    exemptions = _final_baseline_preserved_disagreement_exemptions(
        queue_audit=queue_audit,
        material_choice_audit=choices,
        final_plan=final_plan,
        palette_fusion=palette_fusion,
        final_quality_report_path=quality_report_path,
        final_rendered_registry_path=rendered_registry_path,
    )

    assert [record["group_id"] for record in exemptions] == ["G02"]
    exemption = exemptions[0]
    assert exemption["final_state_revalidated"] is True
    assert exemption["final_authored_target_entity_count"] == 0
    assert exemption["material_choice_resolved"] is False
    assert exemption["material_selection_was_authored"] is False
    assert exemption["render_confirmation_required"] is False
    assert exemption["presence_only_not_material_identity_confirmation"] is True
    assert exemption["evidence_phase"] == "final_whole_asset_post_tournament"
    assert exemption["final_quality_report_file_sha256"] == hashlib.sha256(
        quality_report_path.read_bytes()
    ).hexdigest()
    assert exemption["final_rendered_registry_file_sha256"] == hashlib.sha256(
        rendered_registry_path.read_bytes()
    ).hexdigest()


def test_final_below_significance_unused_disagreement_is_exempt_only_when_unauthored(
    tmp_path: Path,
) -> None:
    (
        queue_audit,
        choices,
        final_plan,
        palette_fusion,
        quality_report_path,
        rendered_registry_path,
    ) = _final_single_view_exemption_inputs(tmp_path)
    exclusion = queue_audit["excluded_groups"][0]
    exclusion.clear()
    exclusion.update(
        {
            "group_id": "G02",
            "reason": "BELOW_REFERENCE_FOOTPRINT_THRESHOLD",
            "reference_footprint_score": 0.004,
        }
    )
    queue_audit["minimum_reference_footprint_score"] = 0.01

    exemptions = _final_baseline_preserved_disagreement_exemptions(
        queue_audit=queue_audit,
        material_choice_audit=choices,
        final_plan=final_plan,
        palette_fusion=palette_fusion,
        final_quality_report_path=quality_report_path,
        final_rendered_registry_path=rendered_registry_path,
    )

    assert [record["group_id"] for record in exemptions] == ["G02"]
    exemption = exemptions[0]
    assert exemption["reason_code"] == (
        "FORWARD_REVERSE_DISAGREEMENT_UNUSED_BELOW_"
        "SIGNIFICANCE_WITHOUT_AUTHORED_TARGET"
    )
    assert exemption["reference_footprint_score"] == 0.004
    assert exemption["minimum_reference_footprint_score"] == 0.01
    assert exemption["final_state_revalidated"] is True
    assert exemption["final_authored_target_entity_count"] == 0
    assert exemption["material_selection_was_authored"] is False
    assert exemption["render_confirmation_required"] is False
    assert exemption["presence_only_not_material_identity_confirmation"] is True

    final_plan["assignments"][0]["provenance"]["canonical_group_id"] = "G02"
    assert (
        _final_baseline_preserved_disagreement_exemptions(
            queue_audit=queue_audit,
            material_choice_audit=choices,
            final_plan=final_plan,
            palette_fusion=palette_fusion,
            final_quality_report_path=quality_report_path,
            final_rendered_registry_path=rendered_registry_path,
        )
        == []
    )


@pytest.mark.parametrize(
    "corruption",
    (
        "final_assignment_group_reappears",
        "final_face_subset_group_reappears",
        "registry_hash_mismatch",
        "registry_path_mismatch",
        "nonexact_recall",
        "untrusted_presence",
        "missing_local_group_row",
        "missing_local_to_canonical_mapping",
        "reference_view_count_mismatch",
    ),
)
def test_final_single_view_unused_exemption_revalidation_fails_closed(
    tmp_path: Path,
    corruption: str,
) -> None:
    (
        queue_audit,
        choices,
        final_plan,
        palette_fusion,
        quality_report_path,
        rendered_registry_path,
    ) = _final_single_view_exemption_inputs(tmp_path)
    quality_report = json.loads(quality_report_path.read_text(encoding="utf-8"))
    if corruption == "final_assignment_group_reappears":
        final_plan["assignments"][0]["provenance"]["canonical_group_id"] = "G02"
    elif corruption == "final_face_subset_group_reappears":
        final_plan["assignments"][0]["provenance"][
            "face_subset_canonical_group_ids"
        ] = {"Cover": "G02"}
    elif corruption == "registry_hash_mismatch":
        quality_report["inputs"]["rendered_registry_sha256"] = "0" * 64
    elif corruption == "registry_path_mismatch":
        quality_report["inputs"]["rendered_registry"] = str(
            tmp_path / "another_registry.json"
        )
    elif corruption == "nonexact_recall":
        quality_report["views"][0]["material_color"][
            "trusted_evidence_group_recall"
        ]["groups"][0]["recall"] = 0.999
    elif corruption == "untrusted_presence":
        quality_report["views"][0]["reference"]["trusted_evidence"][
            "usable"
        ] = False
    elif corruption == "missing_local_group_row":
        group_recall = quality_report["views"][0]["material_color"][
            "trusted_evidence_group_recall"
        ]
        group_recall["groups"] = []
        group_recall["group_count"] = 0
    elif corruption == "missing_local_to_canonical_mapping":
        palette_fusion["view_group_id_maps"]["top"] = {}
    elif corruption == "reference_view_count_mismatch":
        quality_report["aggregate"]["reference_view_count"] = 2
    else:
        raise AssertionError(corruption)
    quality_report_path.write_text(
        json.dumps(quality_report, sort_keys=True),
        encoding="utf-8",
    )

    assert (
        _final_baseline_preserved_disagreement_exemptions(
            queue_audit=queue_audit,
            material_choice_audit=choices,
            final_plan=final_plan,
            palette_fusion=palette_fusion,
            final_quality_report_path=quality_report_path,
            final_rendered_registry_path=rendered_registry_path,
        )
        == []
    )


@pytest.mark.parametrize(
    "corruption",
    (
        "authored_target",
        "coverage_blocker",
        "queued_target",
        "nonexact_recall",
        "untrusted_view",
        "missing_disagreement_contract",
    ),
)
def test_baseline_disagreement_exemption_never_hides_required_work(
    corruption: str,
) -> None:
    queue_audit = _baseline_preserved_queue_audit()
    choices = _disagreement_choices()
    exclusion = queue_audit["excluded_groups"][0]
    if corruption == "authored_target":
        exclusion["authored_target_entity_count"] = 1
    elif corruption == "coverage_blocker":
        queue_audit["coverage_blockers"] = [dict(exclusion)]
    elif corruption == "queued_target":
        queue_audit["groups"].append(
            {"group_id": "G02", "target_entity_count": 1}
        )
    elif corruption == "nonexact_recall":
        exclusion["baseline_presence_evidence"]["views"][0]["recall"] = 0.999
    elif corruption == "untrusted_view":
        exclusion["baseline_presence_evidence"]["views"][0][
            "trusted_reference_evidence"
        ] = False
    elif corruption == "missing_disagreement_contract":
        choices["G02"].pop("disagreement_tournament")
    else:
        raise AssertionError(corruption)

    assert (
        _baseline_preserved_disagreement_exemptions(
            queue_audit=queue_audit,
            material_choice_audit=choices,
        )
        == []
    )
