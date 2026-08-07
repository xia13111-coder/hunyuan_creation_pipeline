"""Pure validation rules for visual-material quality evidence.

This module owns decisions made from already-produced JSON documents.  It
does not start subprocesses and does not write pipeline artifacts, so its
contracts can be tested without Isaac Sim or any model runtime.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


QUALITY_STATUSES = {"PASS", "REVIEW", "FAIL", "INSUFFICIENT_EVIDENCE"}
PART_ID_QUALITY_GATE_SCHEMA_VERSION = "asset-pipeline-part-id-quality-gate/v1"
PART_ID_QUALITY_VIEW_SCOPE_SCHEMA_VERSION = (
    "asset-pipeline-part-id-quality-view-scope/v1"
)
PART_ID_INAPPLICABLE_PALETTE_REASONS = frozenset(
    {"trusted_palette_group_missing_from_render"}
)


def part_id_quality_scope_from_camera_alignment(
    camera_alignment: Mapping[str, Any],
) -> dict[str, Any]:
    """Convert two-layer camera acceptance into the final-QA view scope.

    Whole-asset anchor views may veto a delivered Look. Views explicitly
    classified for local box refinement remain useful Part-ID evidence, but
    their imperfect whole-asset silhouette must not become a global QA vote.
    """

    raw_anchor_ids = camera_alignment.get("anchor_view_ids")
    raw_views = camera_alignment.get("views")
    if (
        not isinstance(raw_anchor_ids, list)
        or not raw_anchor_ids
        or any(
            not isinstance(view_id, str) or not view_id
            for view_id in raw_anchor_ids
        )
        or len(set(raw_anchor_ids)) != len(raw_anchor_ids)
        or not isinstance(raw_views, Mapping)
        or not raw_views
    ):
        raise ValueError("camera alignment has an invalid anchor-view contract")
    anchor_ids = sorted(raw_anchor_ids)
    if not set(anchor_ids).issubset(raw_views):
        raise ValueError("camera alignment anchor views are missing view records")

    local_only_ids: list[str] = []
    rejected_ids: list[str] = []
    for view_id, raw_view in raw_views.items():
        if not isinstance(view_id, str) or not view_id or not isinstance(
            raw_view, Mapping
        ):
            raise ValueError("camera alignment view records are invalid")
        tier = raw_view.get("tier")
        observation_eligible = raw_view.get("observation_eligible")
        if view_id in anchor_ids:
            if (
                observation_eligible is not True
                or tier
                in {"local_box_refinement_only", "rejected_for_part_id_evidence"}
            ):
                raise ValueError(
                    f"camera alignment anchor view {view_id!r} is not globally usable"
                )
        elif tier == "local_box_refinement_only" and observation_eligible is True:
            local_only_ids.append(view_id)
        else:
            rejected_ids.append(view_id)

    return {
        "schema_version": PART_ID_QUALITY_VIEW_SCOPE_SCHEMA_VERSION,
        "mode": "camera_anchor_views",
        "source_camera_policy": camera_alignment.get("policy"),
        "enforced_reference_view_ids": anchor_ids,
        "local_evidence_only_reference_view_ids": sorted(local_only_ids),
        "rejected_reference_view_ids": sorted(rejected_ids),
    }


def _part_id_quality_view_scope(
    quality_report: Mapping[str, Any],
) -> tuple[dict[str, Any], set[str] | None, list[str]]:
    raw_scope = quality_report.get("part_id_quality_scope")
    if raw_scope is None:
        return (
            {
                "schema_version": PART_ID_QUALITY_VIEW_SCOPE_SCHEMA_VERSION,
                "mode": "all_comparable_views",
                "enforced_reference_view_ids": None,
                "local_evidence_only_reference_view_ids": [],
                "rejected_reference_view_ids": [],
            },
            None,
            [],
        )
    if not isinstance(raw_scope, Mapping):
        return ({}, set(), ["PART_ID_QUALITY_VIEW_SCOPE_INVALID"])
    enforced = raw_scope.get("enforced_reference_view_ids")
    local_only = raw_scope.get("local_evidence_only_reference_view_ids", [])
    rejected = raw_scope.get("rejected_reference_view_ids", [])
    if (
        raw_scope.get("schema_version")
        != PART_ID_QUALITY_VIEW_SCOPE_SCHEMA_VERSION
        or raw_scope.get("mode") != "camera_anchor_views"
        or not isinstance(enforced, list)
        or not enforced
        or any(not isinstance(view_id, str) or not view_id for view_id in enforced)
        or len(set(enforced)) != len(enforced)
        or not isinstance(local_only, list)
        or any(not isinstance(view_id, str) or not view_id for view_id in local_only)
        or not isinstance(rejected, list)
        or any(not isinstance(view_id, str) or not view_id for view_id in rejected)
        or bool(set(enforced) & set(local_only))
        or bool(set(enforced) & set(rejected))
    ):
        return ({}, set(), ["PART_ID_QUALITY_VIEW_SCOPE_INVALID"])
    normalized = {
        "schema_version": PART_ID_QUALITY_VIEW_SCOPE_SCHEMA_VERSION,
        "mode": "camera_anchor_views",
        "source_camera_policy": raw_scope.get("source_camera_policy"),
        "enforced_reference_view_ids": sorted(enforced),
        "local_evidence_only_reference_view_ids": sorted(set(local_only)),
        "rejected_reference_view_ids": sorted(set(rejected)),
    }
    return normalized, set(enforced), []


def evaluate_part_id_quality_gate(
    quality_report: Mapping[str, Any],
    *,
    minimum_aggregate_appearance_score: float,
    minimum_view_appearance_score: float,
    minimum_comparable_views: int = 2,
    coating_consistency_audit: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate whole-asset QA without palette-group completeness rules."""

    aggregate = quality_report.get("aggregate")
    raw_views = quality_report.get("views")
    reason_codes: list[str] = []
    limitations: list[dict[str, Any]] = []
    scored_views: list[dict[str, Any]] = []
    enforced_scored_views: list[dict[str, Any]] = []
    view_scope, enforced_view_ids, scope_reasons = _part_id_quality_view_scope(
        quality_report
    )
    reason_codes.extend(scope_reasons)
    raw_status = aggregate.get("status") if isinstance(aggregate, Mapping) else None
    comparable_view_count = (
        aggregate.get("comparable_view_count")
        if isinstance(aggregate, Mapping)
        else None
    )
    aggregate_score = (
        aggregate.get("material_appearance_score")
        if isinstance(aggregate, Mapping)
        else None
    )

    if (
        isinstance(minimum_comparable_views, bool)
        or not isinstance(minimum_comparable_views, int)
        or minimum_comparable_views <= 0
    ):
        raise ValueError("minimum_comparable_views must be a positive integer")
    for label, value in (
        ("minimum_aggregate_appearance_score", minimum_aggregate_appearance_score),
        ("minimum_view_appearance_score", minimum_view_appearance_score),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
        ):
            raise ValueError(f"{label} must be a finite number from 0 to 1")

    if not isinstance(aggregate, Mapping):
        reason_codes.append("AGGREGATE_MISSING_OR_INVALID")
    if not isinstance(raw_views, list) or not raw_views:
        reason_codes.append("VIEWS_MISSING_OR_INVALID")
        raw_views = []

    unsupported_view_failures: dict[str, list[str]] = {}
    unscorable_view_ids: list[str] = []
    for index, raw_view in enumerate(raw_views):
        if not isinstance(raw_view, Mapping):
            reason_codes.append("VIEW_RECORD_INVALID")
            continue
        reference_view_id = raw_view.get("reference_view_id")
        view_id = (
            reference_view_id
            if isinstance(reference_view_id, str) and reference_view_id
            else f"view_{index + 1}"
        )
        view_status = raw_view.get("status")
        score = raw_view.get("material_appearance_score")
        quality_gate_enforced = (
            enforced_view_ids is None or view_id in enforced_view_ids
        )
        if view_status == "UNSCORABLE":
            unscorable_view_ids.append(view_id)
            continue
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
        ):
            reason_codes.append("COMPARABLE_VIEW_SCORE_MISSING_OR_INVALID")
            continue
        raw_reasons = raw_view.get("reasons", [])
        if not isinstance(raw_reasons, list) or any(
            not isinstance(reason, str) for reason in raw_reasons
        ):
            reason_codes.append("VIEW_REASONS_INVALID")
            raw_reasons = []
        unexpected_reasons = sorted(
            set(raw_reasons) - PART_ID_INAPPLICABLE_PALETTE_REASONS
        )
        if quality_gate_enforced and view_status == "FAIL" and unexpected_reasons:
            unsupported_view_failures[view_id] = unexpected_reasons
        scored_view = {
            "reference_view_id": view_id,
            "render_view_id": raw_view.get("render_view_id"),
            "raw_status": view_status,
            "material_appearance_score": float(score),
            "ignored_palette_group_reasons": sorted(
                set(raw_reasons) & PART_ID_INAPPLICABLE_PALETTE_REASONS
            ),
            "retained_reasons": unexpected_reasons,
            "quality_gate_enforced": quality_gate_enforced,
            "passes_appearance_floor": (
                float(score) >= float(minimum_view_appearance_score)
            ),
        }
        scored_views.append(scored_view)
        if quality_gate_enforced:
            enforced_scored_views.append(scored_view)

    if enforced_view_ids is None:
        gate_comparable_view_count = comparable_view_count
        gate_aggregate_score = aggregate_score
        if (
            isinstance(comparable_view_count, bool)
            or not isinstance(comparable_view_count, int)
            or comparable_view_count < minimum_comparable_views
        ):
            reason_codes.append("INSUFFICIENT_COMPARABLE_VIEWS")
        elif len(scored_views) != comparable_view_count:
            reason_codes.append("COMPARABLE_VIEW_COUNT_MISMATCH")
    else:
        gate_comparable_view_count = len(enforced_scored_views)
        scored_enforced_ids = {
            view["reference_view_id"] for view in enforced_scored_views
        }
        if scored_enforced_ids != enforced_view_ids:
            reason_codes.append("REQUIRED_CAMERA_ANCHOR_VIEW_MISSING_OR_UNSCORABLE")
        if gate_comparable_view_count < minimum_comparable_views:
            reason_codes.append("INSUFFICIENT_COMPARABLE_VIEWS")
        gate_aggregate_score = (
            sum(
                float(view["material_appearance_score"])
                for view in enforced_scored_views
            )
            / len(enforced_scored_views)
            if enforced_scored_views
            else None
        )
    if (
        isinstance(gate_aggregate_score, bool)
        or not isinstance(gate_aggregate_score, (int, float))
        or not math.isfinite(float(gate_aggregate_score))
    ):
        reason_codes.append("AGGREGATE_APPEARANCE_SCORE_MISSING_OR_INVALID")
    elif float(gate_aggregate_score) < float(minimum_aggregate_appearance_score):
        reason_codes.append("AGGREGATE_APPEARANCE_BELOW_FLOOR")
    if any(not view["passes_appearance_floor"] for view in enforced_scored_views):
        reason_codes.append("VIEW_APPEARANCE_BELOW_FLOOR")
    if unsupported_view_failures:
        reason_codes.append("NON_PALETTE_VIEW_FAILURE_REASONS_PRESENT")
    if raw_status not in QUALITY_STATUSES:
        reason_codes.append("RAW_QUALITY_STATUS_INVALID")

    coating_consistency_status: str | None = None
    coating_component_count: int | None = None
    coating_constrained_part_count: int | None = None
    if coating_consistency_audit is not None:
        raw_coating_gate = coating_consistency_audit.get("coating_consistency_gate")
        if not isinstance(raw_coating_gate, Mapping):
            reason_codes.append("COATING_CONSISTENCY_GATE_MISSING_OR_INVALID")
        else:
            raw_gate_status = raw_coating_gate.get("status")
            coating_consistency_status = (
                str(raw_gate_status) if isinstance(raw_gate_status, str) else None
            )
            raw_summary = raw_coating_gate.get("summary")
            if isinstance(raw_summary, Mapping):
                raw_component_count = raw_summary.get("component_count")
                raw_constrained_count = raw_summary.get("constrained_part_count")
                if isinstance(raw_component_count, int) and not isinstance(
                    raw_component_count, bool
                ):
                    coating_component_count = raw_component_count
                if isinstance(raw_constrained_count, int) and not isinstance(
                    raw_constrained_count, bool
                ):
                    coating_constrained_part_count = raw_constrained_count
            if coating_consistency_status != "PASS":
                reason_codes.append("COATING_CONSISTENCY_GATE_NOT_PASSED")
    if unscorable_view_ids:
        limitations.append(
            {
                "code": "UNSCORABLE_REFERENCE_VIEWS",
                "reference_view_ids": sorted(unscorable_view_ids),
            }
        )
    local_evidence_only_ids = view_scope.get(
        "local_evidence_only_reference_view_ids", []
    )
    if local_evidence_only_ids:
        limitations.append(
            {
                "code": "LOCAL_EVIDENCE_ONLY_VIEWS_EXCLUDED_FROM_GLOBAL_GATE",
                "reference_view_ids": list(local_evidence_only_ids),
            }
        )

    accepted = not reason_codes
    return {
        "schema_version": PART_ID_QUALITY_GATE_SCHEMA_VERSION,
        "status": "PASS" if accepted else "FAIL_CLOSED",
        "acceptance_allowed": accepted,
        "assignment_unit": "part_id",
        "raw_quality_status": raw_status,
        "effective_quality_status": "PASS" if accepted else raw_status,
        "view_scope": view_scope,
        "thresholds": {
            "minimum_comparable_views": minimum_comparable_views,
            "minimum_aggregate_appearance_score": float(
                minimum_aggregate_appearance_score
            ),
            "minimum_view_appearance_score": float(minimum_view_appearance_score),
        },
        "measurements": {
            "comparable_view_count": comparable_view_count,
            "quality_gate_comparable_view_count": gate_comparable_view_count,
            "scored_view_count": len(scored_views),
            "aggregate_appearance_score": (
                float(gate_aggregate_score)
                if isinstance(gate_aggregate_score, (int, float))
                and not isinstance(gate_aggregate_score, bool)
                and math.isfinite(float(gate_aggregate_score))
                else None
            ),
            "raw_aggregate_appearance_score": (
                float(aggregate_score)
                if isinstance(aggregate_score, (int, float))
                and not isinstance(aggregate_score, bool)
                and math.isfinite(float(aggregate_score))
                else None
            ),
            "coating_consistency_status": coating_consistency_status,
            "coating_component_count": coating_component_count,
            "coating_constrained_part_count": coating_constrained_part_count,
            "views": scored_views,
        },
        "ignored_reason_codes": sorted(PART_ID_INAPPLICABLE_PALETTE_REASONS),
        "unsupported_view_failures": unsupported_view_failures,
        "limitations": limitations,
        "reason_codes": sorted(set(reason_codes)),
    }


def validated_exact_mdl_tournament_mapping(
    *,
    quality_report: dict[str, Any],
    reference_manifest: dict[str, Any],
    trusted_mapping: dict[str, str],
    rendered_registry: dict[str, Any],
) -> dict[str, str]:
    """Seal the complete one-to-one view mapping already proven by Look QA."""

    inputs = quality_report.get("inputs")
    aggregate = quality_report.get("aggregate")
    thresholds = quality_report.get("thresholds")
    quality_views = quality_report.get("views")
    source_views = reference_manifest.get("source_views")
    render_set = rendered_registry.get("render_set")
    rendered_views = render_set.get("views") if isinstance(render_set, dict) else None
    if (
        quality_report.get("schema_version") != "qwen-reference-render-comparison/v1"
        or not isinstance(inputs, dict)
        or not isinstance(aggregate, dict)
        or not isinstance(thresholds, dict)
        or not isinstance(quality_views, list)
        or not isinstance(source_views, list)
        or not isinstance(rendered_views, list)
    ):
        raise RuntimeError(
            "Immutable MDL tournament lacks a complete baseline QA mapping"
        )
    selected_mapping = inputs.get("selected_view_mapping")
    if not isinstance(selected_mapping, dict):
        raise RuntimeError(
            "Immutable MDL tournament baseline QA lacks selected_view_mapping"
        )

    reference_ids: list[str] = []
    for view in source_views:
        reference_id = view.get("id") if isinstance(view, dict) else None
        if (
            not isinstance(reference_id, str)
            or not reference_id
            or reference_id in reference_ids
        ):
            raise RuntimeError("Immutable MDL tournament reference manifest is invalid")
        reference_ids.append(reference_id)
    mapping = {
        reference_id: selected_mapping.get(reference_id)
        for reference_id in reference_ids
    }
    if (
        set(selected_mapping) != set(reference_ids)
        or any(
            not isinstance(render_id, str) or not render_id
            for render_id in mapping.values()
        )
        or len(set(mapping.values())) != len(mapping)
    ):
        raise RuntimeError(
            "Immutable MDL tournament requires a complete one-to-one QA mapping"
        )
    for reference_id, render_id in trusted_mapping.items():
        if mapping.get(reference_id) != render_id:
            raise RuntimeError(
                "Immutable MDL tournament QA mapping conflicts with trusted "
                "spatial registration"
            )

    available_render_ids = {
        view.get("view_id")
        for view in rendered_views
        if isinstance(view, dict) and isinstance(view.get("view_id"), str)
    }
    if not set(mapping.values()).issubset(available_render_ids):
        raise RuntimeError(
            "Immutable MDL tournament QA mapping references an unavailable render"
        )
    quality_by_reference = {
        view.get("reference_view_id"): view
        for view in quality_views
        if isinstance(view, dict) and isinstance(view.get("reference_view_id"), str)
    }
    if set(quality_by_reference) != set(reference_ids):
        raise RuntimeError(
            "Immutable MDL tournament QA does not cover every reference view"
        )

    minimum_auto_score = thresholds.get("minimum_auto_alignment_score")
    minimum_auto_margin = thresholds.get("minimum_auto_match_margin")
    if (
        isinstance(minimum_auto_score, bool)
        or not isinstance(minimum_auto_score, (int, float))
        or isinstance(minimum_auto_margin, bool)
        or not isinstance(minimum_auto_margin, (int, float))
    ):
        raise RuntimeError(
            "Immutable MDL tournament QA alignment thresholds are invalid"
        )
    for reference_id in reference_ids:
        quality_view = quality_by_reference[reference_id]
        mapping_audit = quality_view.get("mapping")
        if (
            quality_view.get("render_view_id") != mapping[reference_id]
            or quality_view.get("status") not in {"PASS", "REVIEW", "FAIL"}
            or not isinstance(quality_view.get("alignment"), dict)
            or not isinstance(quality_view.get("material_color"), dict)
            or not isinstance(mapping_audit, dict)
            or mapping_audit.get("selected_render_view_id") != mapping[reference_id]
            or mapping_audit.get("reasons") not in (None, [])
        ):
            raise RuntimeError(
                "Immutable MDL tournament QA mapping is not comparison-safe"
            )
        if mapping_audit.get("mode") == "auto_completion":
            best_score = mapping_audit.get("best_score")
            margin = mapping_audit.get("global_assignment_margin")
            if (
                mapping_audit.get("global_one_to_one_assignment") is not True
                or isinstance(best_score, bool)
                or not isinstance(best_score, (int, float))
                or float(best_score) < float(minimum_auto_score)
                or isinstance(margin, bool)
                or not isinstance(margin, (int, float))
                or float(margin) < float(minimum_auto_margin)
            ):
                raise RuntimeError(
                    "Immutable MDL tournament auto-completed view mapping is "
                    "not independently confident"
                )
        elif (
            mapping_audit.get("mode") != "explicit_locked"
            or mapping_audit.get("locked") is not True
            or reference_id not in trusted_mapping
        ):
            raise RuntimeError(
                "Immutable MDL tournament QA mapping has unsupported provenance"
            )
    if (
        aggregate.get("reference_view_coverage_status") != "PASS"
        or aggregate.get("comparable_view_count") != len(reference_ids)
        or aggregate.get("unscorable_view_count") != 0
    ):
        raise RuntimeError(
            "Immutable MDL tournament baseline QA view coverage is not safe"
        )
    return {reference_id: str(mapping[reference_id]) for reference_id in reference_ids}


__all__ = [
    "PART_ID_INAPPLICABLE_PALETTE_REASONS",
    "PART_ID_QUALITY_GATE_SCHEMA_VERSION",
    "PART_ID_QUALITY_VIEW_SCOPE_SCHEMA_VERSION",
    "QUALITY_STATUSES",
    "evaluate_part_id_quality_gate",
    "part_id_quality_scope_from_camera_alignment",
    "validated_exact_mdl_tournament_mapping",
]
