"""Fail-closed coverage gate for publishing automatic material selections.

Render similarity alone cannot prove that a material plan is useful.  A large
surface can dominate whole-image metrics while most material entities remain
unresolved or are filled by a neutral policy fallback.  This module validates
the independent inference, localization, and queue audits before an immutable
MDL selection may be published.

The gate is deliberately asset-agnostic.  It consumes versioned audit
documents and ratios, never part names, material names, or project-specific
identities.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


SCHEMA_VERSION = "qwen-material-publish-quality-gate/v1"
CONFIDENCE_GATE_SCHEMA_VERSION = "qwen-material-confidence-gate/v1"
ANNOTATION_SCHEMA_VERSION = "qwen-visual-group-plan-annotation/v1"
POLICY_AUDIT_SCHEMA_VERSION = "qwen-policy-exact-cover-report/v1"
PART_ID_AUDIT_SCHEMA_VERSION = "qwen-part-id-material-plan-audit/v1"
EXACT_CAD_INSTANCE_PROPAGATION_SCHEMA_VERSION = (
    "qwen-exact-cad-instance-material-propagation/v1"
)
QUEUE_SCHEMA_VERSION = "qwen-multigroup-exact-mdl-queue/v1"
TOURNAMENT_SCHEMA_VERSION = "qwen-multigroup-exact-mdl-coordinate-descent/v1"
RENDERED_REGISTRY_SCHEMA_VERSION = "qwen-material-parts/v1"
SPATIAL_MAPPING_SCHEMA_VERSION = "qwen-spatial-mapping-audit/v1"
COHORT_AUDIT_SCHEMA_VERSION = "qwen-source-appearance-cohort-propagation/v1"
COHORT_CONTRACT_SCHEMA_VERSION = "qwen-source-appearance-cohort-contract/v1"
COHORT_METHOD = "trusted_spatial_anchor_source_appearance_cohort/v1"
REPEATED_SUBSET_COHORT_METHOD = (
    "trusted_multiview_repeated_subset_visual_cohort/v1"
)
REPEATED_SUBSET_CANDIDATE_KINDS = {
    "direct_multiview_subset_owner",
    "exact_repeated_geometry_subset_layout",
}
COHORT_SIGNATURE_KIND_BY_CANDIDATE_KIND = {
    "dominant_assembly": "source_appearance_plus_subset_layout",
    "rare_source_appearance_pair": "geometry_plus_appearance_plus_subset_layout",
    "rare_source_appearance_layout_pair": (
        "source_appearance_plus_subset_layout"
    ),
}


class PublishQualityGateError(ValueError):
    """Raised when coverage evidence is invalid or cannot authorize publish."""


@dataclass(frozen=True)
class PublishQualityPolicy:
    """Bounded defaults for catastrophic automatic-coverage failures.

    The defaults intentionally reject only severe failures.  They do not
    demand that every hidden CAD entity be visible in the photographs, but a
    plan cannot be published when nearly all entities are unresolved or were
    synthesized by a generic fallback.
    """

    maximum_policy_fallback_fraction: float = 0.90
    maximum_neutral_fallback_fraction: float = 0.80
    maximum_unresolved_entity_fraction: float = 0.90
    maximum_unresolved_face_subset_fraction: float = 0.50
    minimum_owner_local_resolved_fraction: float = 0.50
    maximum_visible_fallback_fraction: float = 0.20

    def __post_init__(self) -> None:
        for field_name, value in self.__dict__.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not 0.0 <= float(value) <= 1.0
            ):
                raise PublishQualityGateError(
                    f"publish quality policy {field_name} must be in [0, 1]"
                )


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _part_identity_registry(
    rendered_registry: Mapping[str, Any],
    *,
    expected_part_ids: set[str],
) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]], str]:
    raw_parts = _array(rendered_registry.get("parts"), "rendered_registry.parts")
    by_part: dict[str, dict[str, Any]] = {}
    groups: dict[str, list[str]] = {}
    identity_rows: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_parts):
        row = _object(raw, f"rendered_registry.parts[{index}]")
        part_id = row.get("part_id")
        signature = {
            "geometry_content_sha256": row.get("geometry_content_sha256"),
            "source_appearance_sha256": row.get("source_appearance_sha256"),
            "source_subset_layout_sha256": row.get("source_subset_layout_sha256"),
            "point_count": row.get("point_count"),
            "face_count": row.get("face_count"),
        }
        if (
            not isinstance(part_id, str)
            or not part_id
            or part_id in by_part
            or not _is_sha256(signature["geometry_content_sha256"])
            or not _is_sha256(signature["source_appearance_sha256"])
            or not _is_sha256(signature["source_subset_layout_sha256"])
            or isinstance(signature["point_count"], bool)
            or not isinstance(signature["point_count"], int)
            or signature["point_count"] < 0
            or isinstance(signature["face_count"], bool)
            or not isinstance(signature["face_count"], int)
            or signature["face_count"] < 0
        ):
            raise PublishQualityGateError(
                "rendered registry has an invalid exact CAD instance identity"
            )
        signature_sha256 = _canonical_sha256(signature)
        by_part[part_id] = signature
        groups.setdefault(signature_sha256, []).append(part_id)
        identity_rows.append({"part_id": part_id, **signature})
    if set(by_part) != expected_part_ids:
        raise PublishQualityGateError(
            "rendered registry does not exactly cover Part-ID publication"
        )
    return (
        by_part,
        {key: sorted(value) for key, value in groups.items()},
        _canonical_sha256(sorted(identity_rows, key=lambda row: str(row["part_id"]))),
    )


def _object(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PublishQualityGateError(f"{label} must be an object")
    return value


def _array(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise PublishQualityGateError(f"{label} must be an array")
    return value


def _count(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PublishQualityGateError(f"{label} must be a non-negative integer")
    return value


def _ratio(numerator: int, denominator: int) -> float | None:
    return float(numerator) / float(denominator) if denominator else None


def _entity_key(record: Mapping[str, Any], *, label: str) -> tuple[str, str]:
    part_id = record.get("part_id")
    if not isinstance(part_id, str) or not part_id:
        raise PublishQualityGateError(f"{label}.part_id must be a non-empty string")
    subset_name = record.get("subset_name")
    if subset_name is None:
        return part_id, ""
    if not isinstance(subset_name, str) or not subset_name:
        raise PublishQualityGateError(
            f"{label}.subset_name must be null or a non-empty string"
        )
    return part_id, subset_name


def _owner_local(record: Mapping[str, Any]) -> bool:
    """Return whether an annotation is bound to one stable rendered owner."""

    spatial = record.get("spatial_annotation")
    if not isinstance(spatial, Mapping):
        return False
    supporting_view_ids = spatial.get("supporting_view_ids")
    supporting_view_count = spatial.get("supporting_view_count")
    minimum_supporting_view_count = spatial.get("minimum_supporting_view_count")
    conflicting_view_ids = spatial.get("conflicting_view_ids")
    if (
        not isinstance(supporting_view_ids, Sequence)
        or isinstance(supporting_view_ids, (str, bytes))
        or not all(isinstance(view_id, str) and view_id for view_id in supporting_view_ids)
        or len(set(supporting_view_ids)) != len(supporting_view_ids)
        or isinstance(supporting_view_count, bool)
        or not isinstance(supporting_view_count, int)
        or supporting_view_count != len(supporting_view_ids)
        or isinstance(minimum_supporting_view_count, bool)
        or not isinstance(minimum_supporting_view_count, int)
        or minimum_supporting_view_count < 1
        or supporting_view_count < minimum_supporting_view_count
        or not isinstance(conflicting_view_ids, Sequence)
        or isinstance(conflicting_view_ids, (str, bytes))
        or len(conflicting_view_ids) != 0
    ):
        return False
    common_contract = (
        spatial.get("unique_canonical_group") is True
        and spatial.get("material_identity_unchanged") is True
        and spatial.get("parameters_unchanged") is True
    )
    if not common_contract:
        return False
    if spatial.get("whole_part_without_face_subsets") is True:
        return True

    # A topology-complete materialBind subset cohort is deliberately not a
    # whole-part-without-subsets observation.  Its stable multiview anchors
    # are nevertheless owner-local: exact geometry/subset layout, the parent,
    # and every subset travel together through the immutable MDL tournament.
    # Accept only the hash-linked producer contract, never a generic subset
    # annotation or a single-view hint.
    anchor_part_ids = spatial.get("anchor_part_ids")
    part_id = record.get("part_id")
    contract_sha256 = spatial.get("contract_sha256")
    return (
        spatial.get("method") == REPEATED_SUBSET_COHORT_METHOD
        and spatial.get("candidate_kind") in REPEATED_SUBSET_CANDIDATE_KINDS
        and spatial.get("topology_complete_face_subsets") is True
        and isinstance(contract_sha256, str)
        and len(contract_sha256) == 64
        and isinstance(anchor_part_ids, Sequence)
        and not isinstance(anchor_part_ids, (str, bytes))
        and bool(anchor_part_ids)
        and all(
            isinstance(anchor_part_id, str) and anchor_part_id
            for anchor_part_id in anchor_part_ids
        )
        and len(set(anchor_part_ids)) == len(anchor_part_ids)
        and isinstance(part_id, str)
        and part_id in anchor_part_ids
    )


def _final_assignments(
    final_plan: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    if final_plan.get("schema_version") != "1.0":
        raise PublishQualityGateError("final_plan schema_version is invalid")
    raw_assignments = _array(final_plan.get("assignments"), "final_plan.assignments")
    assignments: dict[str, Mapping[str, Any]] = {}
    for index, raw_assignment in enumerate(raw_assignments):
        assignment = _object(raw_assignment, f"final_plan.assignments[{index}]")
        part_id = assignment.get("part_id")
        if (
            not isinstance(part_id, str)
            or not part_id
            or part_id in assignments
        ):
            raise PublishQualityGateError(
                "final_plan assignments need unique non-empty part IDs"
            )
        assignments[part_id] = assignment
    if not assignments:
        raise PublishQualityGateError("final_plan must contain assignments")
    return assignments


def _visible_fallback_metrics(
    *,
    rendered_registry: Mapping[str, Any],
    spatial_mapping_report: Mapping[str, Any],
    annotation_records_by_key: Mapping[
        tuple[str, str],
        Mapping[str, Any],
    ],
) -> dict[str, Any]:
    """Measure unresolved lineage by visible pixels in trusted photo views."""

    registry = _object(rendered_registry, "rendered_registry")
    if registry.get("schema_version") != RENDERED_REGISTRY_SCHEMA_VERSION:
        raise PublishQualityGateError("rendered_registry schema_version is invalid")
    raw_parts = _array(registry.get("parts"), "rendered_registry.parts")
    registered_part_ids: set[str] = set()
    for index, raw_part in enumerate(raw_parts):
        part = _object(raw_part, f"rendered_registry.parts[{index}]")
        part_id = part.get("part_id")
        if (
            not isinstance(part_id, str)
            or not part_id
            or part_id in registered_part_ids
        ):
            raise PublishQualityGateError(
                "rendered_registry needs unique non-empty part IDs"
            )
        registered_part_ids.add(part_id)

    render_set = _object(
        registry.get("render_set"),
        "rendered_registry.render_set",
    )
    raw_views = _array(
        render_set.get("views"),
        "rendered_registry.render_set.views",
    )
    views: dict[str, Mapping[str, Any]] = {}
    for index, raw_view in enumerate(raw_views):
        view = _object(raw_view, f"rendered_registry.render_set.views[{index}]")
        view_id = view.get("view_id")
        if not isinstance(view_id, str) or not view_id or view_id in views:
            raise PublishQualityGateError(
                "rendered_registry render views need unique non-empty IDs"
            )
        visible_parts = _array(
            view.get("visible_parts"),
            f"rendered_registry view {view_id}.visible_parts",
        )
        seen_visible: set[str] = set()
        for visible_index, raw_visible in enumerate(visible_parts):
            visible = _object(
                raw_visible,
                f"rendered_registry view {view_id}.visible_parts[{visible_index}]",
            )
            part_id = visible.get("part_id")
            pixels = visible.get("pixels")
            if (
                not isinstance(part_id, str)
                or part_id not in registered_part_ids
                or part_id in seen_visible
                or isinstance(pixels, bool)
                or not isinstance(pixels, int)
                or pixels < 1
            ):
                raise PublishQualityGateError(
                    f"rendered_registry view {view_id} has invalid visible parts"
                )
            seen_visible.add(part_id)
        views[view_id] = view

    spatial = _object(spatial_mapping_report, "spatial_mapping_report")
    if spatial.get("schema_version") != SPATIAL_MAPPING_SCHEMA_VERSION:
        raise PublishQualityGateError(
            "spatial_mapping_report schema_version is invalid"
        )
    spatial_integrity = _object(
        spatial.get("integrity"),
        "spatial_mapping_report.integrity",
    )
    unsigned_spatial = {
        key: value for key, value in spatial.items() if key != "integrity"
    }
    if spatial_integrity.get("report_sha256") != _canonical_sha256(
        unsigned_spatial
    ):
        raise PublishQualityGateError(
            "spatial_mapping_report integrity check failed"
        )

    unresolved_part_ids = {
        part_id
        for (part_id, subset_name), record in annotation_records_by_key.items()
        if not subset_name
        and record.get("entity_kind") == "assignment"
        and record.get("outcome") in {"AMBIGUOUS", "UNRESOLVED"}
    }
    unknown_unresolved = sorted(unresolved_part_ids - registered_part_ids)
    if unknown_unresolved:
        raise PublishQualityGateError(
            "annotation unresolved parts are absent from rendered_registry"
        )

    raw_alignments = _array(
        spatial.get("view_alignments"),
        "spatial_mapping_report.view_alignments",
    )
    records: list[dict[str, Any]] = []
    seen_reference_views: set[str] = set()
    aggregate_visible_pixels = 0
    aggregate_fallback_pixels = 0
    for index, raw_alignment in enumerate(raw_alignments):
        alignment = _object(
            raw_alignment,
            f"spatial_mapping_report.view_alignments[{index}]",
        )
        reference_view_id = alignment.get("reference_view_id")
        if (
            not isinstance(reference_view_id, str)
            or not reference_view_id
            or reference_view_id in seen_reference_views
        ):
            raise PublishQualityGateError(
                "spatial_mapping_report has duplicate/invalid reference views"
            )
        seen_reference_views.add(reference_view_id)
        if alignment.get("trusted") is not True:
            continue
        render_view_id = alignment.get("selected_render_view_id")
        view = views.get(render_view_id) if isinstance(render_view_id, str) else None
        if view is None:
            raise PublishQualityGateError(
                f"trusted spatial view {reference_view_id} references an "
                "unknown rendered view"
            )
        visible_pixels = 0
        fallback_pixels = 0
        visible_fallback_part_ids: list[str] = []
        for raw_visible in view["visible_parts"]:
            visible = _object(raw_visible, "rendered_registry visible part")
            part_id = str(visible["part_id"])
            pixels = int(visible["pixels"])
            visible_pixels += pixels
            if part_id in unresolved_part_ids:
                fallback_pixels += pixels
                visible_fallback_part_ids.append(part_id)
        if visible_pixels < 1:
            raise PublishQualityGateError(
                f"trusted spatial view {reference_view_id} has no visible part pixels"
            )
        fraction = fallback_pixels / visible_pixels
        aggregate_visible_pixels += visible_pixels
        aggregate_fallback_pixels += fallback_pixels
        records.append(
            {
                "reference_view_id": reference_view_id,
                "render_view_id": render_view_id,
                "visible_part_pixels": visible_pixels,
                "visible_fallback_pixels": fallback_pixels,
                "visible_fallback_fraction": fraction,
                "visible_fallback_part_ids": sorted(visible_fallback_part_ids),
            }
        )
    if len(records) < 2:
        raise PublishQualityGateError(
            "visible fallback coverage needs at least two trusted photo views"
        )
    return {
        "method": "trusted_view_visible_part_pixel_coverage/v1",
        "trusted_view_count": len(records),
        "unresolved_assignment_count": len(unresolved_part_ids),
        "maximum_view_visible_fallback_fraction": max(
            record["visible_fallback_fraction"] for record in records
        ),
        "aggregate_visible_part_pixels": aggregate_visible_pixels,
        "aggregate_visible_fallback_pixels": aggregate_fallback_pixels,
        "aggregate_visible_fallback_fraction": (
            aggregate_fallback_pixels / aggregate_visible_pixels
        ),
        "views": sorted(records, key=lambda record: record["reference_view_id"]),
    }


def _verified_cohort_members(
    *,
    final_plan: Mapping[str, Any],
    assignments: Mapping[str, Mapping[str, Any]],
    annotation_audit: Mapping[str, Any] | None,
    annotation_records_by_key: Mapping[tuple[str, str], Mapping[str, Any]],
    queue_audit: Mapping[str, Any] | None,
) -> set[tuple[str, str]]:
    """Validate anchor-bound cohort propagation and return covered assignments."""

    if annotation_audit is None:
        return set()
    raw_cohort = annotation_audit.get("source_appearance_cohort_propagation")
    if raw_cohort is None:
        return set()
    cohort = _object(raw_cohort, "source appearance cohort audit")
    if cohort.get("schema_version") != COHORT_AUDIT_SCHEMA_VERSION:
        raise PublishQualityGateError("source appearance cohort schema is invalid")
    contracts = _array(cohort.get("contracts"), "source appearance cohort contracts")
    blockers = _array(
        cohort.get("coverage_blockers"), "source appearance cohort blockers"
    )
    if cohort.get("exact_cover") is not True or blockers:
        raise PublishQualityGateError(
            "source appearance cohort audit lacks exact conflict-free coverage"
        )
    enabled = cohort.get("enabled")
    if (
        not isinstance(enabled, bool)
        or cohort.get("method") != COHORT_METHOD
        or (not enabled and contracts)
    ):
        raise PublishQualityGateError("source appearance cohort enabled state is invalid")
    for field in (
        "registry_sha256",
        "spatial_report_sha256",
        "annotation_input_plan_sha256",
    ):
        value = cohort.get(field)
        if enabled and (not isinstance(value, str) or len(value) != 64):
            raise PublishQualityGateError(
                f"source appearance cohort audit has an invalid {field}"
            )

    plan_provenance = _object(final_plan.get("provenance"), "final_plan.provenance")
    plan_annotation = _object(
        plan_provenance.get("visual_group_annotation"),
        "final_plan.provenance.visual_group_annotation",
    )
    plan_cohort = _object(
        plan_annotation.get("source_appearance_cohort_propagation"),
        "final plan source appearance cohort audit",
    )
    if _canonical_sha256(plan_cohort) != _canonical_sha256(cohort):
        raise PublishQualityGateError(
            "final plan and annotation audit disagree on cohort propagation"
        )

    verified: set[tuple[str, str]] = set()
    cohort_ids: set[str] = set()
    contract_ids_by_group: dict[str, list[str]] = {}
    members_by_group: dict[str, set[str]] = {}
    for index, raw_contract in enumerate(contracts):
        contract = _object(raw_contract, f"source appearance contract[{index}]")
        if contract.get("schema_version") != COHORT_CONTRACT_SCHEMA_VERSION:
            raise PublishQualityGateError(
                f"source appearance contract[{index}] schema is invalid"
            )
        contract_sha256 = contract.get("contract_sha256")
        unsigned_contract = dict(contract)
        unsigned_contract.pop("contract_sha256", None)
        if (
            not isinstance(contract_sha256, str)
            or len(contract_sha256) != 64
            or _canonical_sha256(unsigned_contract) != contract_sha256
        ):
            raise PublishQualityGateError(
                f"source appearance contract[{index}] failed integrity validation"
            )
        cohort_id = contract.get("cohort_id")
        group_id = contract.get("canonical_group_id")
        candidate_kind = contract.get("candidate_kind")
        expected_signature_kind = (
            COHORT_SIGNATURE_KIND_BY_CANDIDATE_KIND.get(candidate_kind)
            if isinstance(candidate_kind, str)
            else None
        )
        if (
            not isinstance(cohort_id, str)
            or len(cohort_id) != 64
            or cohort_id in cohort_ids
            or not isinstance(group_id, str)
            or not group_id
            or contract.get("method") != COHORT_METHOD
            or expected_signature_kind is None
            or contract.get("cohort_signature_kind") != expected_signature_kind
        ):
            raise PublishQualityGateError(
                f"source appearance contract[{index}] identity is invalid"
            )
        cohort_ids.add(cohort_id)
        expected = list(
            _array(
                contract.get("expected_member_part_ids"),
                f"source appearance contract[{index}] expected members",
            )
        )
        anchors = list(
            _array(
                contract.get("anchor_part_ids"),
                f"source appearance contract[{index}] anchors",
            )
        )
        propagated = list(
            _array(
                contract.get("propagated_member_part_ids"),
                f"source appearance contract[{index}] propagated members",
            )
        )
        if (
            expected != sorted(set(expected))
            or anchors != sorted(set(anchors))
            or propagated != sorted(set(propagated))
            or not expected
            or not anchors
            or not set(anchors) <= set(expected)
            or not set(propagated) <= set(expected)
            or any(not isinstance(part_id, str) or not part_id for part_id in expected)
            or any(
                not isinstance(contract.get(field), str)
                or len(str(contract.get(field))) != 64
                for field in (
                    "registry_sha256",
                    "spatial_report_sha256",
                    "annotation_input_plan_sha256",
                    "source_appearance_cohort_signature_sha256",
                )
            )
            or any(
                contract.get(field) != cohort.get(field)
                for field in (
                    "registry_sha256",
                    "spatial_report_sha256",
                    "annotation_input_plan_sha256",
                )
            )
            or contract.get("advisory_conflicts") not in ({}, None)
            or contract.get("exact_cover") is not True
            or contract.get("material_identity_unchanged") is not True
            or contract.get("parameters_unchanged") is not True
        ):
            raise PublishQualityGateError(
                f"source appearance contract[{index}] is not conflict-free exact evidence"
            )
        overlap = {part_id for part_id in expected if (part_id, "") in verified}
        if overlap:
            raise PublishQualityGateError(
                "source appearance cohort contracts overlap parts: "
                + ", ".join(sorted(overlap))
            )
        for part_id in expected:
            assignment = assignments.get(part_id)
            record = annotation_records_by_key.get((part_id, ""))
            provenance = (
                assignment.get("provenance")
                if isinstance(assignment, Mapping)
                else None
            )
            lineage = (
                provenance.get("source_appearance_cohort")
                if isinstance(provenance, Mapping)
                else None
            )
            if (
                not isinstance(assignment, Mapping)
                or not isinstance(record, Mapping)
                or record.get("outcome") not in {"ANNOTATED", "PRESERVED_EXISTING"}
                or record.get("selected_group_id") != group_id
                or not isinstance(provenance, Mapping)
                or provenance.get("canonical_group_id") != group_id
                or not isinstance(lineage, Mapping)
                or lineage.get("schema_version") != COHORT_CONTRACT_SCHEMA_VERSION
                or lineage.get("method") != COHORT_METHOD
                or lineage.get("cohort_id") != cohort_id
                or lineage.get("contract_sha256") != contract_sha256
                or lineage.get("canonical_group_id") != group_id
                or lineage.get("anchor_part_ids") != anchors
                or lineage.get("expected_member_part_ids") != expected
                or lineage.get("propagated_member_part_ids") != propagated
                or lineage.get("registry_sha256")
                != contract.get("registry_sha256")
                or lineage.get("spatial_report_sha256")
                != contract.get("spatial_report_sha256")
                or lineage.get("annotation_input_plan_sha256")
                != contract.get("annotation_input_plan_sha256")
                or lineage.get("exact_cover") is not True
                or lineage.get("material_identity_unchanged") is not True
                or lineage.get("parameters_unchanged") is not True
                or lineage.get("member_role")
                != (
                    "anchor"
                    if part_id in anchors
                    else "propagated_member"
                    if part_id in propagated
                    else "existing_member"
                )
            ):
                raise PublishQualityGateError(
                    f"source appearance cohort member {part_id} lacks exact lineage"
                )
            verified.add((part_id, ""))
        contract_ids_by_group.setdefault(group_id, []).append(cohort_id)
        members_by_group.setdefault(group_id, set()).update(expected)
    if (
        _count(cohort.get("cohort_count"), "source appearance cohort_count")
        != len(contracts)
        or _count(
            cohort.get("expected_member_count"),
            "source appearance expected_member_count",
        )
        != len(verified)
        or _count(
            cohort.get("propagated_member_count"),
            "source appearance propagated_member_count",
        )
        != sum(
            len(
                _array(
                    _object(contract, "source appearance contract").get(
                        "propagated_member_part_ids"
                    ),
                    "source appearance propagated members",
                )
            )
            for contract in contracts
        )
    ):
        raise PublishQualityGateError("source appearance cohort summary is inconsistent")

    if queue_audit is not None and contracts:
        annotated_plan_sha256 = annotation_audit.get("annotated_plan_sha256")
        queue_source_plan_sha256 = queue_audit.get("source_plan_sha256")
        if (
            not isinstance(annotated_plan_sha256, str)
            or annotated_plan_sha256 != queue_source_plan_sha256
        ):
            raise PublishQualityGateError(
                "cohort annotation audit is not hash-bound to the queue source plan"
            )
        queue_cohort = _object(
            queue_audit.get("source_appearance_cohort_coverage"),
            "queue source appearance cohort coverage",
        )
        queue_blockers = _array(
            queue_cohort.get("coverage_blockers"),
            "queue source appearance cohort blockers",
        )
        if (
            queue_cohort.get("annotation_audit_verified") is not True
            or queue_cohort.get("exact_cover") is not True
            or _count(
                queue_cohort.get("coverage_blocker_count"),
                "queue cohort coverage_blocker_count",
            )
            != 0
            or queue_blockers
        ):
            raise PublishQualityGateError(
                "queue did not verify exact source appearance cohort coverage"
            )
        queued_members = _object(
            queue_cohort.get("queued_member_part_ids_by_group"),
            "queue cohort members by group",
        )
        contract_ids = _object(
            queue_cohort.get("contract_ids_by_group"),
            "queue cohort contract IDs by group",
        )
        if {
            str(group_id): sorted(str(part_id) for part_id in part_ids)
            for group_id, part_ids in queued_members.items()
        } != {
            group_id: sorted(part_ids) for group_id, part_ids in members_by_group.items()
        } or {
            str(group_id): sorted(str(raw_id) for raw_id in raw_ids)
            for group_id, raw_ids in contract_ids.items()
        } != {
            group_id: sorted(ids)
            for group_id, ids in sorted(contract_ids_by_group.items())
        }:
            raise PublishQualityGateError(
                "queue cohort coverage disagrees with annotation contracts"
            )
    return verified


def _verified_tournament_replacements(
    *,
    final_plan: Mapping[str, Any],
    assignments: Mapping[str, Mapping[str, Any]],
    tournament_audit: Mapping[str, Any] | None,
    queue_audit: Mapping[str, Any] | None,
) -> set[tuple[str, str]]:
    """Return assignment entities replaced by a hash-bound render tournament."""

    if tournament_audit is None:
        return set()
    audit = _object(tournament_audit, "tournament_audit")
    if audit.get("schema_version") != TOURNAMENT_SCHEMA_VERSION:
        # Legacy single-group tournaments remain valid selection evidence but
        # cannot reduce exact-cover fallback counts because they do not carry
        # the complete multi-group hash chain validated below.
        return set()
    if (
        audit.get("status") != "COMPLETED"
        or audit.get("all_significant_groups_evaluated") is not True
        or audit.get("coordinate_descent") is not True
        or audit.get("parameters_locked_to_library_defaults") is not True
        or audit.get("final_plan_sha256") != _canonical_sha256(final_plan)
    ):
        raise PublishQualityGateError(
            "tournament audit is not hash-bound completed render evidence"
        )
    if queue_audit is not None and audit.get("initial_plan_sha256") != queue_audit.get(
        "source_plan_sha256"
    ):
        raise PublishQualityGateError(
            "tournament initial plan does not match the exact queue source plan"
        )
    rounds = list(_array(audit.get("rounds"), "tournament_audit.rounds"))
    significant_group_ids = list(
        _array(
            audit.get("significant_group_ids"),
            "tournament_audit.significant_group_ids",
        )
    )
    if (
        _count(audit.get("significant_group_count"), "significant_group_count")
        != len(significant_group_ids)
        or _count(audit.get("evaluated_group_count"), "evaluated_group_count")
        != len(rounds)
        or len(rounds) != len(significant_group_ids)
    ):
        raise PublishQualityGateError("tournament group coverage is inconsistent")
    queued_groups: dict[str, Mapping[str, Any]] = {}
    if queue_audit is not None:
        raw_queued_groups = _array(queue_audit.get("groups"), "queue_audit.groups")
        for index, raw_group in enumerate(raw_queued_groups):
            group = _object(raw_group, f"queue_audit.groups[{index}]")
            group_id = group.get("group_id")
            if (
                not isinstance(group_id, str)
                or not group_id
                or group_id in queued_groups
            ):
                raise PublishQualityGateError("queue group identities are invalid")
            queued_groups[group_id] = group
        if list(queued_groups) != significant_group_ids:
            raise PublishQualityGateError(
                "tournament significant groups do not match the exact queue"
            )
    expected_input_hash = audit.get("initial_plan_sha256")
    accepted_group_ids: list[str] = []
    fallback_group_ids: list[str] = []
    verified: set[tuple[str, str]] = set()
    seen_changed_keys: set[tuple[str, str]] = set()
    computed_changed_entities: list[dict[str, str]] = []
    for index, raw_round in enumerate(rounds):
        round_audit = _object(raw_round, f"tournament round[{index}]")
        group_id = round_audit.get("group_id")
        if (
            round_audit.get("schema_version") != "qwen-exact-mdl-group-round/v1"
            or group_id != significant_group_ids[index]
            or round_audit.get("input_plan_sha256") != expected_input_hash
            or not isinstance(round_audit.get("output_plan_sha256"), str)
        ):
            raise PublishQualityGateError(
                f"tournament round[{index}] breaks the accepted plan hash chain"
            )
        queued_group = queued_groups.get(str(group_id))
        if queued_group is not None and (
            round_audit.get("target_part_ids")
            != queued_group.get("target_part_ids")
            or round_audit.get("target_entities")
            != queued_group.get("target_entities")
        ):
            raise PublishQualityGateError(
                f"tournament round[{index}] targets differ from the exact queue"
            )
        expected_input_hash = round_audit["output_plan_sha256"]
        if round_audit.get("status") != "ACCEPTED":
            fallback_group_ids.append(str(group_id))
            continue
        accepted_group_ids.append(str(group_id))
        local = _object(
            round_audit.get("selected_local_comparison"),
            f"tournament round[{index}] local comparison",
        )
        accepted_candidate_id = round_audit.get("accepted_candidate_id")
        changes = list(
            _array(
                round_audit.get("material_changes"),
                f"tournament round[{index}] material_changes",
            )
        )
        if (
            not isinstance(accepted_candidate_id, str)
            or not accepted_candidate_id
            or round_audit.get("fallback_to_input_plan") is not False
            or round_audit.get("local_nonfail_evidence_required") is not True
            or local.get("complete") is not True
            or local.get("all_view_nonfail") is not True
            or not changes
        ):
            raise PublishQualityGateError(
                f"accepted tournament round[{index}] lacks non-failing render evidence"
            )
        for change_index, raw_change in enumerate(changes):
            change = _object(
                raw_change,
                f"tournament round[{index}] material_changes[{change_index}]",
            )
            key = _entity_key(
                change,
                label=f"tournament round[{index}] material_changes[{change_index}]",
            )
            if key in seen_changed_keys:
                raise PublishQualityGateError(
                    f"tournament changed material entity {key!r} more than once"
                )
            seen_changed_keys.add(key)
            part_id, subset_name = key
            new_material_id = change.get("new_material_id")
            assignment = assignments.get(part_id)
            if subset_name:
                # Face-subset changes cannot resolve a whole-part policy
                # fallback count.  They remain in lineage/coverage metrics but
                # are intentionally not credited in the fallback numerator.
                computed_changed_entities.append(
                    {
                        "entity_kind": "face_subset",
                        "part_id": part_id,
                        "subset_name": subset_name,
                    }
                )
                continue
            provenance = (
                assignment.get("provenance")
                if isinstance(assignment, Mapping)
                else None
            )
            selection = (
                provenance.get("exact_mdl_tournament")
                if isinstance(provenance, Mapping)
                else None
            )
            if (
                not isinstance(assignment, Mapping)
                or not isinstance(new_material_id, str)
                or assignment.get("material_id") != new_material_id
                or not isinstance(selection, Mapping)
                or selection.get("candidate_id") != accepted_candidate_id
                or selection.get("old_material_id")
                != change.get("old_material_id")
                or selection.get("new_material_id") != new_material_id
                or not isinstance(selection.get("quality_report_sha256"), str)
                or len(str(selection.get("quality_report_sha256"))) != 64
                or selection.get("parameters_locked_to_library_defaults") is not True
            ):
                raise PublishQualityGateError(
                    f"final plan lacks tournament selection lineage for {part_id}"
                )
            verified.add(key)
            computed_changed_entities.append(
                {"entity_kind": "assignment", "part_id": part_id}
            )
    if expected_input_hash != audit.get("preseal_final_plan_sha256"):
        raise PublishQualityGateError("tournament final round hash is inconsistent")
    if (
        audit.get("accepted_group_ids") != accepted_group_ids
        or audit.get("fallback_group_ids") != fallback_group_ids
        or _count(audit.get("accepted_group_count"), "accepted_group_count")
        != len(accepted_group_ids)
        or _count(audit.get("fallback_group_count"), "fallback_group_count")
        != len(fallback_group_ids)
        or audit.get("changed_entities")
        != sorted(
            computed_changed_entities,
            key=lambda entity: (entity["part_id"], entity.get("subset_name", "")),
        )
        or audit.get("changed_part_ids")
        != sorted({entity["part_id"] for entity in computed_changed_entities})
    ):
        raise PublishQualityGateError("tournament selection summary is inconsistent")
    plan_provenance = _object(final_plan.get("provenance"), "final_plan.provenance")
    plan_summary = _object(
        plan_provenance.get("multigroup_exact_mdl_coordinate_descent"),
        "final plan tournament provenance",
    )
    for key in (
        "schema_version",
        "initial_plan_sha256",
        "preseal_final_plan_sha256",
        "significant_group_ids",
        "accepted_group_ids",
        "fallback_group_ids",
        "coordinate_descent",
        "all_significant_groups_evaluated",
        "parameters_locked_to_library_defaults",
    ):
        if plan_summary.get(key) != audit.get(key):
            raise PublishQualityGateError(
                f"final plan tournament provenance disagrees on {key}"
            )
    if plan_summary.get("round_audits_sha256") != _canonical_sha256(rounds):
        raise PublishQualityGateError(
            "final plan tournament provenance does not bind the round audits"
        )
    return verified


def _check(
    *,
    check_id: str,
    observed: float | int | bool | None,
    operator: str,
    threshold: float | int | bool,
    passed: bool,
    reason_code: str,
) -> dict[str, Any]:
    return {
        "check_id": check_id,
        "status": "PASS" if passed else "FAIL",
        "observed": observed,
        "operator": operator,
        "threshold": threshold,
        "reason_code": None if passed else reason_code,
    }


def _verified_part_id_policy_replacements(
    *,
    final_plan: Mapping[str, Any],
    final_assignments: Mapping[str, Mapping[str, Any]],
    policy_plan: Mapping[str, Any],
    policy_audit: Mapping[str, Any],
    part_id_material_audit: Mapping[str, Any],
    rendered_registry: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Verify that photographed Part IDs replaced an exact-cover baseline.

    Hidden CAD entities intentionally keep their policy fallback.  They must
    not be counted as failed *visible* coverage when the Part-ID audit proves
    that every photographed entity was selected independently and the
    unobserved assignments were preserved byte-for-byte.
    """

    audit = _object(part_id_material_audit, "part_id_material_audit")
    if audit.get("schema_version") != PART_ID_AUDIT_SCHEMA_VERSION:
        raise PublishQualityGateError(
            "part_id_material_audit schema_version is invalid"
        )
    if (
        final_plan.get("assignment_unit") != "part_id"
        or final_plan.get("palette_fusion_used") is not False
        or final_plan.get("part_material_groups_used") is not False
        or audit.get("assignment_unit") != "part_id"
        or audit.get("palette_fusion_used") is not False
    ):
        raise PublishQualityGateError(
            "Part-ID publication requires independent non-palette assignments"
        )
    integrity = _object(audit.get("integrity"), "part_id_material_audit.integrity")
    unsigned_audit = {key: value for key, value in audit.items() if key != "integrity"}
    if integrity.get("document_sha256") != _canonical_sha256(unsigned_audit):
        raise PublishQualityGateError("part_id_material_audit integrity check failed")

    base = _object(policy_plan, "policy_plan")
    base_assignments = _final_assignments(base)
    if set(base_assignments) != set(final_assignments):
        raise PublishQualityGateError(
            "Part-ID final plan does not exactly cover its policy baseline"
        )
    policy_output_sha256 = policy_audit.get("output_plan_sha256")
    base_sha256 = _canonical_sha256(base)
    final_sha256 = _canonical_sha256(final_plan)
    if (
        not isinstance(policy_output_sha256, str)
        or policy_output_sha256 != base_sha256
        or audit.get("base_plan_sha256") != base_sha256
        or audit.get("output_plan_sha256") != final_sha256
    ):
        raise PublishQualityGateError(
            "Part-ID publication audit is not hash-bound to its policy and final plans"
        )

    raw_rows = _array(audit.get("parts"), "part_id_material_audit.parts")
    rows: dict[str, Mapping[str, Any]] = {}
    for index, raw_row in enumerate(raw_rows):
        row = _object(raw_row, f"part_id_material_audit.parts[{index}]")
        part_id = row.get("part_id")
        if (
            not isinstance(part_id, str)
            or not part_id
            or part_id in rows
        ):
            raise PublishQualityGateError(
                "part_id_material_audit needs unique non-empty Part IDs"
            )
        rows[part_id] = row
    if set(rows) != set(final_assignments):
        raise PublishQualityGateError(
            "part_id_material_audit does not exactly cover the final plan"
        )

    independently_selected_part_ids: list[str] = []
    unobserved_preserved_part_ids: list[str] = []
    exact_instance_propagated_part_ids: list[str] = []
    neutral_tiers = {
        "neutral_default",
        "source_preserve_unavailable_neutral_fallback",
    }
    independently_replaced_neutral_count = 0
    for part_id in sorted(rows):
        row = rows[part_id]
        base_assignment = base_assignments[part_id]
        final_assignment = final_assignments[part_id]
        row_status = row.get("status")
        if row_status == "independently_selected":
            if (
                base_assignment.get("status") != "policy_fallback"
                or final_assignment.get("status") not in {"auto", "review"}
                or row.get("material_id") != final_assignment.get("material_id")
                or not isinstance(final_assignment.get("provenance"), Mapping)
                or final_assignment["provenance"].get("assignment_unit") != "part_id"
            ):
                raise PublishQualityGateError(
                    f"Part-ID replacement lineage is invalid for {part_id}"
                )
            independently_selected_part_ids.append(part_id)
            base_provenance = base_assignment.get("provenance")
            if (
                isinstance(base_provenance, Mapping)
                and base_provenance.get("tier") in neutral_tiers
            ):
                independently_replaced_neutral_count += 1
        elif row_status == "unobserved_preserved":
            if final_assignment != base_assignment:
                raise PublishQualityGateError(
                    f"unobserved Part ID {part_id} changed from its policy baseline"
                )
            unobserved_preserved_part_ids.append(part_id)
        elif row_status == "unobserved_exact_instance_propagated":
            provenance = final_assignment.get("provenance")
            if (
                base_assignment.get("status") != "policy_fallback"
                or final_assignment.get("status") != "review"
                or final_assignment == base_assignment
                or row.get("material_id") != final_assignment.get("material_id")
                or not isinstance(provenance, Mapping)
                or provenance.get("assignment_unit") != "part_id"
                or provenance.get("exact_cad_instance_material_propagation")
                is not True
                or provenance.get("photo_observed") is not False
                or provenance.get("color_parameters_authored") is not False
                or provenance.get("source_policy_assignment_sha256")
                != _canonical_sha256(base_assignment)
                or row.get("source_policy_assignment_sha256")
                != _canonical_sha256(base_assignment)
                or row.get("source_policy_material_id")
                != base_assignment.get("material_id")
                or provenance.get("source_policy_material_id")
                != base_assignment.get("material_id")
            ):
                raise PublishQualityGateError(
                    f"exact CAD instance propagation lineage is invalid for {part_id}"
                )
            exact_instance_propagated_part_ids.append(part_id)
            base_provenance = base_assignment.get("provenance")
            if (
                isinstance(base_provenance, Mapping)
                and base_provenance.get("tier") in neutral_tiers
            ):
                independently_replaced_neutral_count += 1
        else:
            raise PublishQualityGateError(
                f"part_id_material_audit has unsupported status for {part_id}"
            )

    summary = _object(audit.get("summary"), "part_id_material_audit.summary")
    if (
        _count(summary.get("part_count"), "Part-ID part_count") != len(rows)
        or _count(
            summary.get("independently_selected_count"),
            "Part-ID independently_selected_count",
        )
        != len(independently_selected_part_ids)
        or _count(
            summary.get("unobserved_preserved_count"),
            "Part-ID unobserved_preserved_count",
        )
        != len(unobserved_preserved_part_ids)
        or _count(
            summary.get("unobserved_exact_instance_propagated_count", 0),
            "Part-ID unobserved_exact_instance_propagated_count",
        )
        != len(exact_instance_propagated_part_ids)
        or summary.get("exact_cover") is not True
        or not independently_selected_part_ids
    ):
        raise PublishQualityGateError(
            "part_id_material_audit summary does not match its records"
        )
    if exact_instance_propagated_part_ids:
        if rendered_registry is None:
            raise PublishQualityGateError(
                "exact CAD instance propagation requires the rendered registry"
            )
        propagation = _object(
            audit.get("exact_cad_instance_material_propagation"),
            "part_id_material_audit.exact_cad_instance_material_propagation",
        )
        if (
            propagation.get("schema_version")
            != EXACT_CAD_INSTANCE_PROPAGATION_SCHEMA_VERSION
            or propagation.get("status") != "COMPLETED"
        ):
            raise PublishQualityGateError(
                "exact CAD instance propagation audit is incomplete"
            )
        identities, registry_groups, registry_sha256 = _part_identity_registry(
            rendered_registry,
            expected_part_ids=set(rows),
        )
        if propagation.get("part_identity_registry_sha256") != registry_sha256:
            raise PublishQualityGateError(
                "exact CAD instance propagation is not bound to the registry"
            )
        raw_groups = _array(
            propagation.get("propagated_groups"),
            "exact_cad_instance_material_propagation.propagated_groups",
        )
        group_by_signature: dict[str, Mapping[str, Any]] = {}
        declared_targets: set[str] = set()
        for index, raw_group in enumerate(raw_groups):
            group = _object(raw_group, f"propagated_groups[{index}]")
            signature_sha256 = group.get("identity_signature_sha256")
            members = group.get("member_part_ids")
            anchors = group.get("observed_anchor_part_ids")
            targets = group.get("propagated_part_ids")
            material_id = group.get("material_id")
            observed_material_ids = group.get("observed_material_ids")
            observed_audit_statuses = group.get("observed_audit_statuses")
            if (
                not _is_sha256(signature_sha256)
                or signature_sha256 in group_by_signature
                or group.get("status") != "PROPAGATED"
                or not isinstance(members, list)
                or not isinstance(anchors, list)
                or not anchors
                or not isinstance(targets, list)
                or not targets
                or not isinstance(material_id, str)
                or not material_id.startswith("mdl:")
                or not isinstance(observed_material_ids, Mapping)
                or not isinstance(observed_audit_statuses, Mapping)
                or members != registry_groups.get(str(signature_sha256))
                or group.get("identity_signature")
                != identities[str(members[0])]
                or set(anchors) & set(targets)
                or set(anchors) | set(targets) != set(members)
                or group.get("unobserved_member_part_ids") != targets
                or set(observed_material_ids) != set(anchors)
                or observed_audit_statuses
                != {part_id: "independently_selected" for part_id in anchors}
                or any(rows.get(part_id, {}).get("status") != "independently_selected" for part_id in anchors)
                or any(rows.get(part_id, {}).get("status") != "unobserved_exact_instance_propagated" for part_id in targets)
                or any(final_assignments[part_id].get("material_id") != material_id for part_id in members)
                or any(
                    rows[part_id].get("identity_signature_sha256")
                    != signature_sha256
                    or rows[part_id].get("observed_anchor_part_ids") != anchors
                    or final_assignments[part_id].get("provenance", {}).get(
                        "identity_signature_sha256"
                    )
                    != signature_sha256
                    or final_assignments[part_id].get("provenance", {}).get(
                        "observed_anchor_part_ids"
                    )
                    != anchors
                    for part_id in targets
                )
                or any(
                    row_material != material_id
                    for row_material in observed_material_ids.values()
                )
            ):
                raise PublishQualityGateError(
                    "exact CAD instance propagation group is not reproducible"
                )
            group_by_signature[str(signature_sha256)] = group
            declared_targets.update(str(part_id) for part_id in targets)
        if declared_targets != set(exact_instance_propagated_part_ids):
            raise PublishQualityGateError(
                "exact CAD instance propagation target cover is inconsistent"
            )
        propagation_summary = _object(
            propagation.get("summary"),
            "exact_cad_instance_material_propagation.summary",
        )
        if (
            _count(
                propagation_summary.get("propagated_group_count"),
                "exact instance propagated_group_count",
            )
            != len(group_by_signature)
            or _count(
                propagation_summary.get("propagated_part_count"),
                "exact instance propagated_part_count",
            )
            != len(exact_instance_propagated_part_ids)
        ):
            raise PublishQualityGateError(
                "exact CAD instance propagation summary is inconsistent"
            )
    return {
        "independently_selected_part_ids": independently_selected_part_ids,
        "unobserved_preserved_part_ids": unobserved_preserved_part_ids,
        "exact_instance_propagated_part_ids": exact_instance_propagated_part_ids,
        "independently_replaced_neutral_count": (
            independently_replaced_neutral_count
        ),
    }


def build_publish_quality_gate(
    *,
    confidence_gate: Mapping[str, Any],
    final_plan: Mapping[str, Any],
    annotation_audit: Mapping[str, Any] | None = None,
    policy_audit: Mapping[str, Any] | None = None,
    policy_plan: Mapping[str, Any] | None = None,
    part_id_material_audit: Mapping[str, Any] | None = None,
    queue_audit: Mapping[str, Any] | None = None,
    tournament_audit: Mapping[str, Any] | None = None,
    rendered_registry: Mapping[str, Any] | None = None,
    spatial_mapping_report: Mapping[str, Any] | None = None,
    policy: PublishQualityPolicy | None = None,
) -> dict[str, Any]:
    """Build a hash-bound publish decision from independent pipeline audits."""

    effective_policy = policy or PublishQualityPolicy()
    final = _object(final_plan, "final_plan")
    assignments = _final_assignments(final)
    confidence = _object(confidence_gate, "confidence_gate")
    if confidence.get("schema_version") != CONFIDENCE_GATE_SCHEMA_VERSION:
        raise PublishQualityGateError("confidence_gate schema_version is invalid")
    confidence_summary = _object(
        confidence.get("summary"), "confidence_gate.summary"
    )
    part_count = _count(confidence_summary.get("part_count"), "part_count")
    auto_count = _count(confidence_summary.get("auto_count"), "auto_count")
    review_count = _count(confidence_summary.get("review_count"), "review_count")
    preserve_count = _count(
        confidence_summary.get("preserve_count"), "preserve_count"
    )
    if part_count < 1 or auto_count + review_count + preserve_count != part_count:
        raise PublishQualityGateError(
            "confidence gate counts must exactly cover at least one registry part"
        )
    if len(assignments) != part_count:
        raise PublishQualityGateError(
            "final plan assignment count does not match the confidence gate"
        )
    auto_fraction = _ratio(auto_count, part_count)
    assert auto_fraction is not None

    policy_fallback_count = 0
    policy_output_count = part_count
    neutral_fallback_count = 0
    part_id_lineage: dict[str, Any] | None = None
    if policy_audit is not None:
        policy_document = _object(policy_audit, "policy_audit")
        if policy_document.get("schema_version") != POLICY_AUDIT_SCHEMA_VERSION:
            raise PublishQualityGateError("policy_audit schema_version is invalid")
        policy_summary = _object(policy_document.get("summary"), "policy_audit.summary")
        registry_part_count = _count(
            policy_summary.get("registry_part_count"), "policy registry_part_count"
        )
        policy_output_count = _count(
            policy_summary.get("output_assignment_count"),
            "policy output_assignment_count",
        )
        policy_fallback_count = _count(
            policy_summary.get("policy_fallback_count"),
            "policy policy_fallback_count",
        )
        if (
            registry_part_count != part_count
            or policy_output_count != registry_part_count
            or policy_fallback_count > policy_output_count
        ):
            raise PublishQualityGateError(
                "policy audit counts are inconsistent with the confidence gate"
            )
        recorded_auto_count = _count(
            policy_summary.get("confidence_gate_auto_count"),
            "policy confidence_gate_auto_count",
        )
        if recorded_auto_count != auto_count:
            raise PublishQualityGateError(
                "policy audit confidence_gate_auto_count does not match its input"
            )
        neutral_default_count = _count(
            policy_summary.get("neutral_default_count", 0),
            "policy neutral_default_count",
        )
        unavailable_neutral_count = _count(
            policy_summary.get(
                "source_preserve_unavailable_neutral_fallback_count", 0
            ),
            "policy source_preserve_unavailable_neutral_fallback_count",
        )
        neutral_fallback_count = neutral_default_count + unavailable_neutral_count
        if neutral_fallback_count > policy_output_count:
            raise PublishQualityGateError(
                "policy neutral fallback counts exceed output assignments"
            )
        final_policy_fallback_count = sum(
            assignment.get("status") == "policy_fallback"
            for assignment in assignments.values()
        )
        if final_policy_fallback_count != policy_fallback_count and (
            policy_plan is None or part_id_material_audit is None
        ):
            raise PublishQualityGateError(
                "final plan policy-fallback lineage does not match the policy audit"
            )
        if policy_plan is not None or part_id_material_audit is not None:
            if policy_plan is None or part_id_material_audit is None:
                raise PublishQualityGateError(
                    "Part-ID publication requires both policy_plan and "
                    "part_id_material_audit"
                )
            part_id_lineage = _verified_part_id_policy_replacements(
                final_plan=final,
                final_assignments=assignments,
                policy_plan=policy_plan,
                policy_audit=policy_document,
                part_id_material_audit=part_id_material_audit,
                rendered_registry=rendered_registry,
            )
            expected_final_fallback_count = policy_fallback_count - len(
                part_id_lineage["independently_selected_part_ids"]
            ) - len(part_id_lineage["exact_instance_propagated_part_ids"])
            if final_policy_fallback_count != expected_final_fallback_count:
                raise PublishQualityGateError(
                    "final Part-ID policy-fallback count does not match its "
                    "verified independent replacements"
                )
    annotation_metrics: dict[str, Any] | None = None
    annotation_records_by_key: dict[tuple[str, str], Mapping[str, Any]] = {}
    owner_local_keys: set[tuple[str, str]] = set()
    unresolved_fraction: float | None = None
    unresolved_face_subset_fraction: float | None = None
    owner_local_resolved_fraction: float | None = None
    ambiguous_count = 0
    if annotation_audit is not None:
        annotation = _object(annotation_audit, "annotation_audit")
        if annotation.get("schema_version") != ANNOTATION_SCHEMA_VERSION:
            raise PublishQualityGateError("annotation_audit schema_version is invalid")
        summary = _object(annotation.get("summary"), "annotation_audit.summary")
        records = _array(annotation.get("records"), "annotation_audit.records")
        assignment_count = _count(summary.get("assignment_count"), "assignment_count")
        face_subset_count = _count(summary.get("face_subset_count"), "face_subset_count")
        annotated_count = _count(summary.get("annotated_count"), "annotated_count")
        preserved_count = _count(
            summary.get("preserved_existing_count"), "preserved_existing_count"
        )
        ambiguous_count = _count(summary.get("ambiguous_count"), "ambiguous_count")
        unresolved_count = _count(summary.get("unresolved_count"), "unresolved_count")
        entity_count = assignment_count + face_subset_count
        if entity_count != len(records) or (
            annotated_count + preserved_count + ambiguous_count + unresolved_count
            != entity_count
        ):
            raise PublishQualityGateError(
                "annotation summary does not exactly cover its entity records"
            )
        computed_outcomes: dict[str, int] = {}
        unresolved_face_subset_count = 0
        owner_local_count = 0
        for index, raw_record in enumerate(records):
            record = _object(raw_record, f"annotation_audit.records[{index}]")
            key = _entity_key(record, label=f"annotation_audit.records[{index}]")
            if key in annotation_records_by_key:
                raise PublishQualityGateError(
                    f"annotation audit repeats material entity {key!r}"
                )
            annotation_records_by_key[key] = record
            outcome = record.get("outcome")
            if outcome not in {
                "ANNOTATED",
                "PRESERVED_EXISTING",
                "AMBIGUOUS",
                "UNRESOLVED",
            }:
                raise PublishQualityGateError(
                    f"annotation record {key!r} has an invalid outcome"
                )
            computed_outcomes[str(outcome)] = computed_outcomes.get(str(outcome), 0) + 1
            if (
                record.get("entity_kind") == "face_subset"
                and outcome in {"AMBIGUOUS", "UNRESOLVED"}
            ):
                unresolved_face_subset_count += 1
            if outcome == "ANNOTATED" and _owner_local(record):
                owner_local_count += 1
                owner_local_keys.add(key)
        expected_outcomes = {
            "ANNOTATED": annotated_count,
            "PRESERVED_EXISTING": preserved_count,
            "AMBIGUOUS": ambiguous_count,
            "UNRESOLVED": unresolved_count,
        }
        if any(computed_outcomes.get(key, 0) != value for key, value in expected_outcomes.items()):
            raise PublishQualityGateError(
                "annotation summary outcome counts do not match its records"
            )
        unresolved_fraction = _ratio(unresolved_count + ambiguous_count, entity_count)
        unresolved_face_subset_fraction = _ratio(
            unresolved_face_subset_count, face_subset_count
        )
        owner_local_resolved_fraction = _ratio(owner_local_count, annotated_count)
        annotation_metrics = {
            "entity_count": entity_count,
            "assignment_count": assignment_count,
            "face_subset_count": face_subset_count,
            "annotated_count": annotated_count,
            "preserved_existing_count": preserved_count,
            "ambiguous_count": ambiguous_count,
            "unresolved_count": unresolved_count,
            "unresolved_fraction": unresolved_fraction,
            "unresolved_face_subset_count": unresolved_face_subset_count,
            "unresolved_face_subset_fraction": unresolved_face_subset_fraction,
            "owner_local_annotated_count": owner_local_count,
            "owner_local_resolved_fraction": owner_local_resolved_fraction,
        }

    cohort_owner_keys = _verified_cohort_members(
        final_plan=final,
        assignments=assignments,
        annotation_audit=annotation_audit,
        annotation_records_by_key=annotation_records_by_key,
        queue_audit=queue_audit,
    )
    if annotation_metrics is not None and cohort_owner_keys:
        owner_local_keys.update(cohort_owner_keys)
        owner_local_count = sum(
            key in owner_local_keys and record.get("outcome") == "ANNOTATED"
            for key, record in annotation_records_by_key.items()
        )
        annotated_count = int(annotation_metrics["annotated_count"])
        owner_local_resolved_fraction = _ratio(owner_local_count, annotated_count)
        annotation_metrics["owner_local_annotated_count"] = owner_local_count
        annotation_metrics["owner_local_resolved_fraction"] = (
            owner_local_resolved_fraction
        )
        annotation_metrics["anchor_bound_cohort_entity_count"] = len(
            cohort_owner_keys
        )
    elif annotation_metrics is not None:
        annotation_metrics["anchor_bound_cohort_entity_count"] = 0

    if (rendered_registry is None) != (spatial_mapping_report is None):
        raise PublishQualityGateError(
            "visible fallback coverage requires both rendered_registry and "
            "spatial_mapping_report"
        )
    visible_fallback_metrics = (
        _visible_fallback_metrics(
            rendered_registry=rendered_registry,
            spatial_mapping_report=spatial_mapping_report,
            annotation_records_by_key=annotation_records_by_key,
        )
        if rendered_registry is not None
        and spatial_mapping_report is not None
        and annotation_audit is not None
        else None
    )

    verified_tournament_entities = _verified_tournament_replacements(
        final_plan=final,
        assignments=assignments,
        tournament_audit=tournament_audit,
        queue_audit=queue_audit,
    )
    verified_policy_replacement_part_ids = sorted(
        part_id
        for part_id, subset_name in verified_tournament_entities
        if not subset_name and assignments[part_id].get("status") == "policy_fallback"
    )
    if (
        policy_audit is not None
        and len(verified_policy_replacement_part_ids) > policy_fallback_count
    ):
        raise PublishQualityGateError(
            "verified tournament replacements exceed policy fallback inputs"
        )
    effective_policy_fallback_count = max(
        0,
        policy_fallback_count
        - (
            len(verified_policy_replacement_part_ids)
            if policy_audit is not None
            else 0
        ),
    )
    if part_id_lineage is not None:
        effective_policy_fallback_count -= len(
            part_id_lineage["independently_selected_part_ids"]
        )
        effective_policy_fallback_count -= len(
            part_id_lineage["exact_instance_propagated_part_ids"]
        )
    neutral_tiers = {
        "neutral_default",
        "source_preserve_unavailable_neutral_fallback",
    }
    verified_neutral_replacement_count = sum(
        isinstance(assignments[part_id].get("provenance"), Mapping)
        and assignments[part_id]["provenance"].get("tier") in neutral_tiers
        for part_id in verified_policy_replacement_part_ids
    )
    if verified_neutral_replacement_count > neutral_fallback_count:
        raise PublishQualityGateError(
            "verified neutral replacements exceed policy neutral fallback inputs"
        )
    effective_neutral_fallback_count = (
        neutral_fallback_count - verified_neutral_replacement_count
    )
    if part_id_lineage is not None:
        effective_neutral_fallback_count -= int(
            part_id_lineage["independently_replaced_neutral_count"]
        )
    coverage_scope = (
        "photo_observed_and_exact_cad_instance_part_ids"
        if part_id_lineage is not None
        else "all_plan_assignments"
    )
    coverage_denominator = (
        len(part_id_lineage["independently_selected_part_ids"])
        + len(part_id_lineage["exact_instance_propagated_part_ids"])
        if part_id_lineage is not None
        else policy_output_count
    )
    scoped_policy_fallback_count = (
        sum(
            assignments[part_id].get("status") == "policy_fallback"
            for part_id in (
                part_id_lineage["independently_selected_part_ids"]
                + part_id_lineage["exact_instance_propagated_part_ids"]
            )
        )
        if part_id_lineage is not None
        else effective_policy_fallback_count
    )
    scoped_neutral_fallback_count = (
        0
        if part_id_lineage is not None
        else effective_neutral_fallback_count
    )
    policy_fallback_fraction = _ratio(
        scoped_policy_fallback_count, coverage_denominator
    )
    assert policy_fallback_fraction is not None
    neutral_fallback_fraction = _ratio(
        scoped_neutral_fallback_count, coverage_denominator
    )
    assert neutral_fallback_fraction is not None

    queue_metrics: dict[str, Any] | None = None
    queue_complete = True
    unowned_group_ids: list[str] = []
    if queue_audit is not None:
        queue = _object(queue_audit, "queue_audit")
        if queue.get("schema_version") != QUEUE_SCHEMA_VERSION:
            raise PublishQualityGateError("queue_audit schema_version is invalid")
        raw_groups = _array(queue.get("groups"), "queue_audit.groups")
        coverage_blocker_count = _count(
            queue.get("coverage_blocker_count"), "queue coverage_blocker_count"
        )
        queue_complete = (
            coverage_blocker_count == 0
            and queue.get("all_discovered_significant_groups_queued") is True
            and queue.get("all_candidate_bearing_significant_groups_queued") is True
        )
        if annotation_audit is None:
            raise PublishQualityGateError(
                "a queue audit requires its canonical-group annotation audit"
            )
        for group_index, raw_group in enumerate(raw_groups):
            group = _object(raw_group, f"queue_audit.groups[{group_index}]")
            group_id = group.get("group_id")
            if not isinstance(group_id, str) or not group_id:
                raise PublishQualityGateError("queued group_id must be non-empty")
            entities = _array(
                group.get("target_entities"),
                f"queue group {group_id}.target_entities",
            )
            has_bound_owner = False
            for entity_index, raw_entity in enumerate(entities):
                entity = _object(
                    raw_entity,
                    f"queue group {group_id}.target_entities[{entity_index}]",
                )
                key = _entity_key(
                    entity,
                    label=f"queue group {group_id}.target_entities[{entity_index}]",
                )
                annotation_record = annotation_records_by_key.get(key)
                if annotation_record is None:
                    continue
                if annotation_record.get("selected_group_id") != group_id:
                    continue
                if key in owner_local_keys or annotation_record.get(
                    "outcome"
                ) == "PRESERVED_EXISTING":
                    has_bound_owner = True
                    break
            if not has_bound_owner:
                unowned_group_ids.append(group_id)
        queue_metrics = {
            "significant_group_count": len(raw_groups),
            "coverage_blocker_count": coverage_blocker_count,
            "complete": queue_complete,
            "owner_local_group_count": len(raw_groups) - len(unowned_group_ids),
            "unowned_group_ids": sorted(unowned_group_ids),
        }

    checks = [
        _check(
            check_id="policy_fallback_coverage",
            observed=policy_fallback_fraction,
            operator="<=",
            threshold=effective_policy.maximum_policy_fallback_fraction,
            passed=(
                policy_fallback_fraction
                <= effective_policy.maximum_policy_fallback_fraction
            ),
            reason_code="POLICY_FALLBACK_FRACTION_ABOVE_MAXIMUM",
        ),
        _check(
            check_id="neutral_fallback_coverage",
            observed=neutral_fallback_fraction,
            operator="<=",
            threshold=effective_policy.maximum_neutral_fallback_fraction,
            passed=(
                neutral_fallback_fraction
                <= effective_policy.maximum_neutral_fallback_fraction
            ),
            reason_code="NEUTRAL_FALLBACK_FRACTION_ABOVE_MAXIMUM",
        ),
    ]
    if annotation_metrics is not None:
        assert unresolved_fraction is not None
        checks.extend(
            [
                _check(
                    check_id="unresolved_entity_coverage",
                    observed=unresolved_fraction,
                    operator="<=",
                    threshold=effective_policy.maximum_unresolved_entity_fraction,
                    passed=(
                        unresolved_fraction
                        <= effective_policy.maximum_unresolved_entity_fraction
                    ),
                    reason_code="UNRESOLVED_ENTITY_FRACTION_ABOVE_MAXIMUM",
                ),
                _check(
                    check_id="annotation_ambiguity",
                    observed=ambiguous_count,
                    operator="==",
                    threshold=0,
                    passed=ambiguous_count == 0,
                    reason_code="AMBIGUOUS_MATERIAL_ENTITY_PRESENT",
                ),
            ]
        )
        if unresolved_face_subset_fraction is not None:
            checks.append(
                _check(
                    check_id="unresolved_face_subset_coverage",
                    observed=unresolved_face_subset_fraction,
                    operator="<=",
                    threshold=(
                        effective_policy.maximum_unresolved_face_subset_fraction
                    ),
                    passed=(
                        unresolved_face_subset_fraction
                        <= effective_policy.maximum_unresolved_face_subset_fraction
                    ),
                    reason_code=(
                        "UNRESOLVED_FACE_SUBSET_FRACTION_ABOVE_MAXIMUM"
                    ),
                )
            )
    if visible_fallback_metrics is not None:
        maximum_visible_fallback = float(
            visible_fallback_metrics[
                "maximum_view_visible_fallback_fraction"
            ]
        )
        checks.append(
            _check(
                check_id="visible_fallback_coverage",
                observed=maximum_visible_fallback,
                operator="<=",
                threshold=effective_policy.maximum_visible_fallback_fraction,
                passed=(
                    maximum_visible_fallback
                    <= effective_policy.maximum_visible_fallback_fraction
                ),
                reason_code="VISIBLE_FALLBACK_FRACTION_ABOVE_MAXIMUM",
            )
        )
        if owner_local_resolved_fraction is not None:
            checks.append(
                _check(
                    check_id="owner_local_resolved_coverage",
                    observed=owner_local_resolved_fraction,
                    operator=">=",
                    threshold=(
                        effective_policy.minimum_owner_local_resolved_fraction
                    ),
                    passed=(
                        owner_local_resolved_fraction
                        >= effective_policy.minimum_owner_local_resolved_fraction
                    ),
                    reason_code="OWNER_LOCAL_RESOLVED_FRACTION_BELOW_MINIMUM",
                )
            )
    if queue_metrics is not None:
        checks.extend(
            [
                _check(
                    check_id="complete_significant_group_queue",
                    observed=queue_complete,
                    operator="is",
                    threshold=True,
                    passed=queue_complete,
                    reason_code="SIGNIFICANT_GROUP_QUEUE_COVERAGE_INCOMPLETE",
                ),
                _check(
                    check_id="queued_group_owner_localization",
                    observed=len(unowned_group_ids),
                    operator="==",
                    threshold=0,
                    passed=not unowned_group_ids,
                    reason_code="QUEUED_GROUP_LACKS_OWNER_LOCAL_ENTITY",
                ),
            ]
        )

    reason_codes = sorted(
        str(check["reason_code"])
        for check in checks
        if check["status"] == "FAIL"
    )
    unsigned: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS" if not reason_codes else "FAIL",
        "policy": {
            key: float(value) for key, value in effective_policy.__dict__.items()
        },
        "metrics": {
            "confidence": {
                "part_count": part_count,
                "auto_count": auto_count,
                "review_count": review_count,
                "preserve_count": preserve_count,
                "auto_assignment_fraction": auto_fraction,
            },
            "policy_fallback": {
                "coverage_scope": coverage_scope,
                "coverage_denominator": coverage_denominator,
                "scoped_policy_fallback_count": scoped_policy_fallback_count,
                "scoped_neutral_fallback_count": scoped_neutral_fallback_count,
                "output_assignment_count": policy_output_count,
                "pre_tournament_policy_fallback_count": policy_fallback_count,
                "verified_tournament_replacement_count": len(
                    verified_policy_replacement_part_ids
                ),
                "verified_tournament_replacement_part_ids": (
                    verified_policy_replacement_part_ids
                ),
                "effective_policy_fallback_count": (
                    effective_policy_fallback_count
                ),
                "policy_fallback_fraction": policy_fallback_fraction,
                "pre_tournament_neutral_fallback_count": neutral_fallback_count,
                "verified_neutral_replacement_count": (
                    verified_neutral_replacement_count
                ),
                "effective_neutral_fallback_count": (
                    effective_neutral_fallback_count
                ),
                "neutral_fallback_fraction": neutral_fallback_fraction,
                "part_id_lineage": part_id_lineage,
            },
            "selection_lineage": {
                "final_plan_assignment_count": len(assignments),
                "verified_tournament_entity_count": len(
                    verified_tournament_entities
                ),
                "verified_tournament_assignment_count": sum(
                    not subset_name
                    for _part_id, subset_name in verified_tournament_entities
                ),
                "anchor_bound_cohort_entity_count": len(cohort_owner_keys),
            },
            "annotation": annotation_metrics,
            "visible_fallback": visible_fallback_metrics,
            "queue": queue_metrics,
        },
        "checks": checks,
        "reason_codes": reason_codes,
        "publication_authorized": not reason_codes,
        "scope": "automatic_non_bundled_visual_material_selection",
        "input_hashes": {
            "confidence_gate_sha256": _canonical_sha256(confidence_gate),
            "final_plan_sha256": _canonical_sha256(final_plan),
            "annotation_audit_sha256": (
                _canonical_sha256(annotation_audit)
                if annotation_audit is not None
                else None
            ),
            "policy_audit_sha256": (
                _canonical_sha256(policy_audit) if policy_audit is not None else None
            ),
            "policy_plan_sha256": (
                _canonical_sha256(policy_plan) if policy_plan is not None else None
            ),
            "part_id_material_audit_sha256": (
                _canonical_sha256(part_id_material_audit)
                if part_id_material_audit is not None
                else None
            ),
            "queue_audit_sha256": (
                _canonical_sha256(queue_audit) if queue_audit is not None else None
            ),
            "tournament_audit_sha256": (
                _canonical_sha256(tournament_audit)
                if tournament_audit is not None
                else None
            ),
            "rendered_registry_sha256": (
                _canonical_sha256(rendered_registry)
                if rendered_registry is not None
                else None
            ),
            "spatial_mapping_report_sha256": (
                _canonical_sha256(spatial_mapping_report)
                if spatial_mapping_report is not None
                else None
            ),
        },
    }
    return {
        **unsigned,
        "integrity": {"report_sha256": _canonical_sha256(unsigned)},
    }


def require_publish_quality_gate_passed(report: Mapping[str, Any]) -> None:
    """Validate one report and reject publication unless every check passed."""

    document = _object(report, "publish quality gate")
    if document.get("schema_version") != SCHEMA_VERSION:
        raise PublishQualityGateError("publish quality gate schema_version is invalid")
    integrity = _object(document.get("integrity"), "publish quality gate integrity")
    unsigned = {key: value for key, value in document.items() if key != "integrity"}
    if integrity.get("report_sha256") != _canonical_sha256(unsigned):
        raise PublishQualityGateError("publish quality gate integrity check failed")
    checks = _array(document.get("checks"), "publish quality gate checks")
    failed_checks = [
        str(check.get("check_id", "UNKNOWN"))
        for check in checks
        if not isinstance(check, Mapping) or check.get("status") != "PASS"
    ]
    if (
        document.get("status") != "PASS"
        or document.get("publication_authorized") is not True
        or failed_checks
        or document.get("reason_codes") != []
    ):
        reason_codes = document.get("reason_codes")
        raise PublishQualityGateError(
            "automatic material selection is not publishable: "
            f"reason_codes={reason_codes!r}, failed_checks={failed_checks!r}"
        )


__all__ = [
    "ANNOTATION_SCHEMA_VERSION",
    "CONFIDENCE_GATE_SCHEMA_VERSION",
    "POLICY_AUDIT_SCHEMA_VERSION",
    "PublishQualityGateError",
    "PublishQualityPolicy",
    "QUEUE_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "build_publish_quality_gate",
    "require_publish_quality_gate_passed",
]
