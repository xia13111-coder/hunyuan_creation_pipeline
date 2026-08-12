from __future__ import annotations

import copy
import hashlib
import json

import pytest

from qwen_material_pipeline.materials.publish_quality_gate import (
    PublishQualityPolicy,
    PublishQualityGateError,
    build_publish_quality_gate,
    require_publish_quality_gate_passed,
    _verified_part_id_policy_replacements,
)


def test_default_neutral_fallback_limit_targets_catastrophic_coverage() -> None:
    assert PublishQualityPolicy().maximum_neutral_fallback_fraction == 0.80


def _confidence(*, part_count: int, auto_count: int) -> dict:
    return {
        "schema_version": "qwen-material-confidence-gate/v1",
        "summary": {
            "part_count": part_count,
            "auto_count": auto_count,
            "review_count": 0,
            "preserve_count": part_count - auto_count,
        },
    }


def _plan(part_count: int, *, fallback_count: int | None = None) -> dict:
    fallback_count = part_count if fallback_count is None else fallback_count
    return {
        "schema_version": "1.0",
        "assignments": [
            {
                "part_id": f"P{index:04d}",
                "status": (
                    "policy_fallback" if index < fallback_count else "auto"
                ),
                "material_id": "mdl:Base/Metals/Steel_Stainless.mdl#Steel_Stainless",
                "provenance": {"tier": "semantic_rule"},
            }
            for index in range(part_count)
        ],
        "provenance": {},
    }


def _sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _spatial_record(part_id: str, group_id: str = "G01") -> dict:
    return {
        "entity_kind": "assignment",
        "part_id": part_id,
        "subset_name": None,
        "outcome": "ANNOTATED",
        "selected_group_id": group_id,
        "spatial_annotation": {
            "supporting_view_ids": ["front", "side"],
            "supporting_view_count": 2,
            "minimum_supporting_view_count": 2,
            "conflicting_view_ids": [],
            "unique_canonical_group": True,
            "whole_part_without_face_subsets": True,
            "material_identity_unchanged": True,
            "parameters_unchanged": True,
        },
    }


def _preserved_record(part_id: str, *, subset_name: str | None = None) -> dict:
    return {
        "entity_kind": "face_subset" if subset_name else "assignment",
        "part_id": part_id,
        "subset_name": subset_name,
        "outcome": "PRESERVED_EXISTING",
        "selected_group_id": "G01",
    }


def _annotation() -> dict:
    records = [_spatial_record(f"P{index:04d}") for index in range(6)]
    records.extend(_preserved_record(f"P{index:04d}") for index in range(6, 10))
    records.extend(
        [
            _preserved_record("P0009", subset_name="paint"),
            {
                "entity_kind": "face_subset",
                "part_id": "P0009",
                "subset_name": "label",
                "outcome": "UNRESOLVED",
                "selected_group_id": None,
            },
        ]
    )
    return {
        "schema_version": "qwen-visual-group-plan-annotation/v1",
        "status": "COMPLETED_WITH_UNRESOLVED",
        "summary": {
            "assignment_count": 10,
            "face_subset_count": 2,
            "annotated_count": 6,
            "preserved_existing_count": 5,
            "ambiguous_count": 0,
            "unresolved_count": 1,
        },
        "records": records,
    }


def _policy() -> dict:
    return {
        "schema_version": "qwen-policy-exact-cover-report/v1",
        "summary": {
            "registry_part_count": 10,
            "output_assignment_count": 10,
            "policy_fallback_count": 5,
            "confidence_gate_auto_count": 2,
            "neutral_default_count": 1,
            "source_preserve_unavailable_neutral_fallback_count": 1,
        },
    }


def test_hidden_p0181_cannot_be_resealed_as_independently_selected() -> None:
    evidence_unsigned = {
        "schema_version": "qwen-part-id-reference-evidence/v1",
        "assignment_unit": "part_id",
        "parts": [
            {
                "part_id": "P0181",
                "status": "unobserved",
                "observations": [],
            }
        ],
    }
    evidence_digest = _sha256(evidence_unsigned)
    evidence = {
        **evidence_unsigned,
        "integrity": {"document_sha256": evidence_digest},
    }
    baseline_assignment = {
        "part_id": "P0181",
        "material_id": "mdl:Base/Metals/Steel_Stainless.mdl#Steel_Stainless",
        "status": "policy_fallback",
        "confidence": 0.0,
        "evidence_views": [],
        "provenance": {"tier": "neutral_default"},
    }
    policy_plan = {
        "schema_version": "1.0",
        "assignments": [baseline_assignment],
        "provenance": {},
    }
    policy_audit = {"output_plan_sha256": _sha256(policy_plan)}
    final_plan = {
        "schema_version": "1.0",
        "assignment_unit": "part_id",
        "palette_fusion_used": False,
        "part_material_groups_used": False,
        "assignments": [copy.deepcopy(baseline_assignment)],
        "provenance": {"part_id_evidence_sha256": evidence_digest},
    }
    final_plan["assignments"][0].update(
        {
            "material_id": "mdl:Base/Metals/Aluminum_Anodized_Blue.mdl#Blue",
            "status": "review",
            "provenance": {"assignment_unit": "part_id"},
        }
    )
    audit_unsigned = {
        "schema_version": "qwen-part-id-material-plan-audit/v1",
        "assignment_unit": "part_id",
        "palette_fusion_used": False,
        "base_plan_sha256": _sha256(policy_plan),
        "part_id_evidence_sha256": evidence_digest,
        "output_plan_sha256": _sha256(final_plan),
        "parts": [
            {
                "part_id": "P0181",
                "status": "independently_selected",
                "material_id": final_plan["assignments"][0]["material_id"],
            }
        ],
        "summary": {
            "part_count": 1,
            "independently_selected_count": 1,
            "unobserved_preserved_count": 0,
            "exact_cover": True,
        },
    }
    part_id_audit = {
        **audit_unsigned,
        "integrity": {"document_sha256": _sha256(audit_unsigned)},
    }

    with pytest.raises(PublishQualityGateError, match="contradicts final evidence"):
        _verified_part_id_policy_replacements(
            final_plan=final_plan,
            final_assignments={"P0181": final_plan["assignments"][0]},
            policy_plan=policy_plan,
            policy_audit=policy_audit,
            part_id_material_audit=part_id_audit,
            part_id_evidence=evidence,
        )


def _queue() -> dict:
    return {
        "schema_version": "qwen-multigroup-exact-mdl-queue/v1",
        "groups": [
            {
                "group_id": "G01",
                "target_entities": [
                    {
                        "entity_kind": "assignment",
                        "part_id": "P0000",
                    }
                ],
            }
        ],
        "coverage_blocker_count": 0,
        "all_discovered_significant_groups_queued": True,
        "all_candidate_bearing_significant_groups_queued": True,
    }


def test_balanced_hash_bound_coverage_passes() -> None:
    report = build_publish_quality_gate(
        confidence_gate=_confidence(part_count=10, auto_count=2),
        final_plan=_plan(10, fallback_count=5),
        annotation_audit=_annotation(),
        policy_audit=_policy(),
        queue_audit=_queue(),
    )

    assert report["status"] == "PASS"
    assert report["publication_authorized"] is True
    assert report["reason_codes"] == []
    require_publish_quality_gate_passed(report)


def test_catastrophic_fallback_and_unresolved_coverage_fails_closed() -> None:
    records = [_spatial_record(f"P{index:04d}") for index in range(6)]
    records.extend(
        {
            "entity_kind": "assignment",
            "part_id": f"P{index:04d}",
            "subset_name": None,
            "outcome": "UNRESOLVED",
            "selected_group_id": None,
        }
        for index in range(6, 596)
    )
    records.extend(
        {
            "entity_kind": "face_subset",
            "part_id": f"P{index:04d}",
            "subset_name": "surface",
            "outcome": "UNRESOLVED",
            "selected_group_id": None,
        }
        for index in range(28)
    )
    annotation = {
        "schema_version": "qwen-visual-group-plan-annotation/v1",
        "summary": {
            "assignment_count": 596,
            "face_subset_count": 28,
            "annotated_count": 6,
            "preserved_existing_count": 0,
            "ambiguous_count": 0,
            "unresolved_count": 618,
        },
        "records": records,
    }
    policy = {
        "schema_version": "qwen-policy-exact-cover-report/v1",
        "summary": {
            "registry_part_count": 596,
            "output_assignment_count": 596,
            "policy_fallback_count": 596,
            "confidence_gate_auto_count": 0,
            "neutral_default_count": 11,
            "source_preserve_unavailable_neutral_fallback_count": 330,
        },
    }

    report = build_publish_quality_gate(
        confidence_gate=_confidence(part_count=596, auto_count=0),
        final_plan=_plan(596),
        annotation_audit=annotation,
        policy_audit=policy,
        queue_audit=_queue(),
    )

    assert report["status"] == "FAIL"
    assert set(report["reason_codes"]) == {
        "POLICY_FALLBACK_FRACTION_ABOVE_MAXIMUM",
        "UNRESOLVED_ENTITY_FRACTION_ABOVE_MAXIMUM",
        "UNRESOLVED_FACE_SUBSET_FRACTION_ABOVE_MAXIMUM",
    }
    with pytest.raises(PublishQualityGateError, match="not publishable"):
        require_publish_quality_gate_passed(report)


def test_visible_fallback_pixels_fail_even_when_entity_ratio_is_small() -> None:
    annotation = copy.deepcopy(_annotation())
    assignment = next(
        record
        for record in annotation["records"]
        if record["entity_kind"] == "assignment"
        and record["part_id"] == "P0009"
    )
    assignment["outcome"] = "UNRESOLVED"
    assignment["selected_group_id"] = None
    annotation["summary"]["preserved_existing_count"] = 4
    annotation["summary"]["unresolved_count"] = 2
    registry = {
        "schema_version": "qwen-material-parts/v1",
        "parts": [{"part_id": f"P{index:04d}"} for index in range(10)],
        "render_set": {
            "views": [
                {
                    "view_id": view_id,
                    "visible_parts": [
                        {"part_id": "P0000", "pixels": 100},
                        {"part_id": "P0009", "pixels": 100},
                    ],
                }
                for view_id in ("right", "front")
            ]
        },
    }
    spatial_unsigned = {
        "schema_version": "qwen-spatial-mapping-audit/v1",
        "view_alignments": [
            {
                "reference_view_id": "front_ref",
                "selected_render_view_id": "right",
                "trusted": True,
            },
            {
                "reference_view_id": "side_ref",
                "selected_render_view_id": "front",
                "trusted": True,
            },
        ],
    }
    spatial = {
        **spatial_unsigned,
        "integrity": {"report_sha256": _sha256(spatial_unsigned)},
    }

    report = build_publish_quality_gate(
        confidence_gate=_confidence(part_count=10, auto_count=2),
        final_plan=_plan(10, fallback_count=0),
        annotation_audit=annotation,
        rendered_registry=registry,
        spatial_mapping_report=spatial,
    )

    assert report["status"] == "FAIL"
    assert report["reason_codes"] == [
        "VISIBLE_FALLBACK_FRACTION_ABOVE_MAXIMUM"
    ]
    assert (
        report["metrics"]["visible_fallback"][
            "maximum_view_visible_fallback_fraction"
        ]
        == 0.5
    )


def test_render_verified_replacements_reduce_effective_fallback_without_auto() -> None:
    part_ids = [f"P{index:04d}" for index in range(10)]
    candidate_id = "g01_verified"
    new_material_id = "mdl:Base/Plastics/Rubber_Smooth.mdl#Rubber_Smooth"
    initial_plan_sha256 = "1" * 64
    preseal_plan_sha256 = "2" * 64
    changes = [
        {
            "part_id": part_id,
            "old_material_id": "mdl:Base/Metals/Steel_Stainless.mdl#Steel_Stainless",
            "new_material_id": new_material_id,
        }
        for part_id in part_ids
    ]
    target_entities = [
        {"entity_kind": "assignment", "part_id": part_id}
        for part_id in part_ids
    ]
    round_audit = {
        "schema_version": "qwen-exact-mdl-group-round/v1",
        "status": "ACCEPTED",
        "group_id": "G01",
        "target_part_ids": part_ids,
        "target_entities": target_entities,
        "input_plan_sha256": initial_plan_sha256,
        "output_plan_sha256": preseal_plan_sha256,
        "accepted_candidate_id": candidate_id,
        "fallback_to_input_plan": False,
        "local_nonfail_evidence_required": True,
        "selected_local_comparison": {
            "complete": True,
            "all_view_nonfail": True,
        },
        "material_changes": changes,
    }
    plan = _plan(10)
    for assignment in plan["assignments"]:
        assignment["material_id"] = new_material_id
        assignment["provenance"]["exact_mdl_tournament"] = {
            "candidate_id": candidate_id,
            "old_material_id": (
                "mdl:Base/Metals/Steel_Stainless.mdl#Steel_Stainless"
            ),
            "new_material_id": new_material_id,
            "quality_report_sha256": "8" * 64,
            "parameters_locked_to_library_defaults": True,
        }
    plan_summary = {
        "schema_version": "qwen-multigroup-exact-mdl-coordinate-descent/v1",
        "initial_plan_sha256": initial_plan_sha256,
        "preseal_final_plan_sha256": preseal_plan_sha256,
        "significant_group_ids": ["G01"],
        "accepted_group_ids": ["G01"],
        "fallback_group_ids": [],
        "round_audits_sha256": _sha256([round_audit]),
        "coordinate_descent": True,
        "all_significant_groups_evaluated": True,
        "parameters_locked_to_library_defaults": True,
    }
    plan["provenance"]["multigroup_exact_mdl_coordinate_descent"] = plan_summary
    tournament = {
        **plan_summary,
        "status": "COMPLETED",
        "final_plan_sha256": _sha256(plan),
        "significant_group_count": 1,
        "evaluated_group_count": 1,
        "accepted_group_count": 1,
        "fallback_group_count": 0,
        "changed_part_ids": part_ids,
        "changed_entities": target_entities,
        "rounds": [round_audit],
    }
    annotation = {
        "schema_version": "qwen-visual-group-plan-annotation/v1",
        "summary": {
            "assignment_count": 10,
            "face_subset_count": 0,
            "annotated_count": 10,
            "preserved_existing_count": 0,
            "ambiguous_count": 0,
            "unresolved_count": 0,
        },
        "records": [_spatial_record(part_id) for part_id in part_ids],
    }
    policy = {
        "schema_version": "qwen-policy-exact-cover-report/v1",
        "summary": {
            "registry_part_count": 10,
            "output_assignment_count": 10,
            "policy_fallback_count": 10,
            "confidence_gate_auto_count": 0,
            "neutral_default_count": 0,
            "source_preserve_unavailable_neutral_fallback_count": 0,
        },
    }
    queue = {
        **_queue(),
        "source_plan_sha256": initial_plan_sha256,
        "significant_group_ids": ["G01"],
    }
    queue["groups"][0]["target_part_ids"] = part_ids
    queue["groups"][0]["target_entities"] = target_entities

    report = build_publish_quality_gate(
        confidence_gate=_confidence(part_count=10, auto_count=0),
        final_plan=plan,
        annotation_audit=annotation,
        policy_audit=policy,
        queue_audit=queue,
        tournament_audit=tournament,
    )

    assert report["status"] == "PASS"
    assert report["metrics"]["confidence"]["auto_count"] == 0
    fallback = report["metrics"]["policy_fallback"]
    assert fallback["pre_tournament_policy_fallback_count"] == 10
    assert fallback["verified_tournament_replacement_count"] == 10
    assert fallback["effective_policy_fallback_count"] == 0


@pytest.mark.parametrize(
    ("candidate_kind", "cohort_signature_kind"),
    [
        ("dominant_assembly", "source_appearance_plus_subset_layout"),
        (
            "rare_source_appearance_layout_pair",
            "source_appearance_plus_subset_layout",
        ),
    ],
)
def test_exact_conflict_free_cohort_members_count_as_anchor_bound_owners(
    candidate_kind: str,
    cohort_signature_kind: str,
) -> None:
    part_ids = [f"P{index:04d}" for index in range(4)]
    anchor_ids = [part_ids[0]]
    propagated_ids = part_ids[1:]
    contract = {
        "schema_version": "qwen-source-appearance-cohort-contract/v1",
        "method": "trusted_spatial_anchor_source_appearance_cohort/v1",
        "candidate_kind": candidate_kind,
        "cohort_signature_kind": cohort_signature_kind,
        "canonical_group_id": "G01",
        "assembly_path": "/Assembly",
        "source_appearance_cohort_signature_sha256": "3" * 64,
        "anchor_part_ids": anchor_ids,
        "expected_member_part_ids": part_ids,
        "registry_sha256": "4" * 64,
        "spatial_report_sha256": "5" * 64,
        "annotation_input_plan_sha256": "6" * 64,
        "cohort_id": "7" * 64,
        "subtree_part_ids": part_ids,
        "propagated_member_part_ids": propagated_ids,
        "advisory_conflicts": {},
        "signature_dominance": {"part_share": 1.0, "face_share": 1.0},
        "exact_cover": True,
        "material_identity_unchanged": True,
        "parameters_unchanged": True,
    }
    contract["contract_sha256"] = _sha256(contract)
    cohort = {
        "schema_version": "qwen-source-appearance-cohort-propagation/v1",
        "enabled": True,
        "method": "trusted_spatial_anchor_source_appearance_cohort/v1",
        "registry_sha256": contract["registry_sha256"],
        "spatial_report_sha256": contract["spatial_report_sha256"],
        "annotation_input_plan_sha256": contract[
            "annotation_input_plan_sha256"
        ],
        "exact_cover": True,
        "cohort_count": 1,
        "expected_member_count": 4,
        "propagated_member_count": 3,
        "contracts": [contract],
        "coverage_blockers": [],
    }
    plan = _plan(4, fallback_count=2)
    for assignment in plan["assignments"]:
        part_id = assignment["part_id"]
        lineage = {
            "schema_version": "qwen-source-appearance-cohort-contract/v1",
            "method": "trusted_spatial_anchor_source_appearance_cohort/v1",
            "cohort_id": contract["cohort_id"],
            "contract_sha256": contract["contract_sha256"],
            "canonical_group_id": "G01",
            "member_role": (
                "anchor" if part_id in anchor_ids else "propagated_member"
            ),
            "anchor_part_ids": anchor_ids,
            "expected_member_part_ids": part_ids,
            "propagated_member_part_ids": propagated_ids,
            "registry_sha256": contract["registry_sha256"],
            "spatial_report_sha256": contract["spatial_report_sha256"],
            "annotation_input_plan_sha256": contract[
                "annotation_input_plan_sha256"
            ],
            "exact_cover": True,
            "material_identity_unchanged": True,
            "parameters_unchanged": True,
        }
        assignment["provenance"].update(
            {
                "canonical_group_id": "G01",
                "source_appearance_cohort": lineage,
            }
        )
    plan["provenance"]["visual_group_annotation"] = {
        "source_appearance_cohort_propagation": cohort
    }
    plan_sha256 = _sha256(plan)
    records = [_spatial_record(part_ids[0])]
    records.extend(
        {
            "entity_kind": "assignment",
            "part_id": part_id,
            "subset_name": None,
            "outcome": "ANNOTATED",
            "selected_group_id": "G01",
            "source_appearance_cohort": plan["assignments"][index][
                "provenance"
            ]["source_appearance_cohort"],
        }
        for index, part_id in enumerate(part_ids[1:], start=1)
    )
    annotation = {
        "schema_version": "qwen-visual-group-plan-annotation/v1",
        "annotated_plan_sha256": plan_sha256,
        "summary": {
            "assignment_count": 4,
            "face_subset_count": 0,
            "annotated_count": 4,
            "preserved_existing_count": 0,
            "ambiguous_count": 0,
            "unresolved_count": 0,
        },
        "records": records,
        "source_appearance_cohort_propagation": cohort,
    }
    queue = {
        **_queue(),
        "source_plan_sha256": plan_sha256,
        "source_appearance_cohort_coverage": {
            "annotation_audit_verified": True,
            "exact_cover": True,
            "coverage_blocker_count": 0,
            "coverage_blockers": [],
            "queued_member_part_ids_by_group": {"G01": part_ids},
            "contract_ids_by_group": {"G01": [contract["cohort_id"]]},
        },
    }
    queue["groups"][0]["target_part_ids"] = part_ids
    queue["groups"][0]["target_entities"] = [
        {"entity_kind": "assignment", "part_id": part_id}
        for part_id in part_ids
    ]
    policy = {
        "schema_version": "qwen-policy-exact-cover-report/v1",
        "summary": {
            "registry_part_count": 4,
            "output_assignment_count": 4,
            "policy_fallback_count": 2,
            "confidence_gate_auto_count": 0,
            "neutral_default_count": 0,
            "source_preserve_unavailable_neutral_fallback_count": 0,
        },
    }

    report = build_publish_quality_gate(
        confidence_gate=_confidence(part_count=4, auto_count=0),
        final_plan=plan,
        annotation_audit=annotation,
        policy_audit=policy,
        queue_audit=queue,
    )

    assert report["status"] == "PASS"
    annotation_metrics = report["metrics"]["annotation"]
    assert annotation_metrics["owner_local_annotated_count"] == 4
    assert annotation_metrics["anchor_bound_cohort_entity_count"] == 4
    assert annotation_metrics["owner_local_resolved_fraction"] == 1.0


def test_queued_group_requires_owner_local_or_trusted_existing_entity() -> None:
    queue = _queue()
    queue["groups"][0]["target_entities"][0]["part_id"] = "P0009"
    queue["groups"][0]["target_entities"][0]["subset_name"] = "label"

    report = build_publish_quality_gate(
        confidence_gate=_confidence(part_count=10, auto_count=2),
        final_plan=_plan(10, fallback_count=5),
        annotation_audit=_annotation(),
        policy_audit=_policy(),
        queue_audit=queue,
    )

    assert "QUEUED_GROUP_LACKS_OWNER_LOCAL_ENTITY" in report["reason_codes"]
    assert report["metrics"]["queue"]["unowned_group_ids"] == ["G01"]


def test_topology_complete_repeated_subset_anchor_is_owner_local() -> None:
    annotation = _annotation()
    record = annotation["records"][0]
    spatial = record["spatial_annotation"]
    spatial.pop("whole_part_without_face_subsets")
    spatial.update(
        {
            "method": "trusted_multiview_repeated_subset_visual_cohort/v1",
            "contract_sha256": "a" * 64,
            "candidate_kind": "exact_repeated_geometry_subset_layout",
            "anchor_part_ids": ["P0000"],
            "topology_complete_face_subsets": True,
        }
    )

    report = build_publish_quality_gate(
        confidence_gate=_confidence(part_count=10, auto_count=2),
        final_plan=_plan(10, fallback_count=5),
        annotation_audit=annotation,
        policy_audit=_policy(),
        queue_audit=_queue(),
    )

    assert report["status"] == "PASS"
    assert report["metrics"]["queue"]["unowned_group_ids"] == []


def test_report_integrity_is_required() -> None:
    report = build_publish_quality_gate(
        confidence_gate=_confidence(part_count=10, auto_count=2),
        final_plan=_plan(10, fallback_count=5),
        annotation_audit=_annotation(),
        policy_audit=_policy(),
        queue_audit=_queue(),
    )
    tampered = copy.deepcopy(report)
    tampered["metrics"]["confidence"]["auto_count"] = 0

    with pytest.raises(PublishQualityGateError, match="integrity"):
        require_publish_quality_gate_passed(tampered)
