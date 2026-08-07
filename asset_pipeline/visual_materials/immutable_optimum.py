"""Acceptance contract for an exhausted immutable material library search.

This is deliberately narrower than a normal visual PASS.  It only recognizes
an otherwise complete REVIEW whose remaining discrepancy is foreground
brightness after every material identity and parameter has been frozen.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any


SCHEMA_VERSION = "asset-pipeline-immutable-library-optimum/v1"
DECISION = "ACCEPTED_IMMUTABLE_LIBRARY_OPTIMUM_WITH_REVIEW_VIEWS"
ALLOWED_REVIEW_REASONS = frozenset(
    {
        "foreground_value_similarity_below_pass_threshold",
    }
)
MINIMUM_TRUSTED_COLOR_RECALL = 0.90


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def evaluate_immutable_library_optimum(
    quality_report: Mapping[str, Any],
    *,
    minimum_aggregate_appearance_score: float,
    minimum_view_appearance_score: float,
) -> dict[str, Any]:
    """Return a hash-bound audit for a constrained immutable-library REVIEW."""

    reasons: list[str] = []
    aggregate = quality_report.get("aggregate")
    thresholds = quality_report.get("thresholds")
    raw_views = quality_report.get("views")
    if not isinstance(aggregate, Mapping):
        reasons.append("QUALITY_AGGREGATE_MISSING")
        aggregate = {}
    if not isinstance(thresholds, Mapping):
        reasons.append("QUALITY_THRESHOLDS_MISSING")
        thresholds = {}
    if not isinstance(raw_views, list) or not raw_views:
        reasons.append("QUALITY_VIEWS_MISSING")
        raw_views = []

    if aggregate.get("status") != "REVIEW":
        reasons.append("AGGREGATE_STATUS_IS_NOT_REVIEW")
    if aggregate.get("reference_view_coverage_status") != "PASS":
        reasons.append("REFERENCE_VIEW_COVERAGE_NOT_PASS")

    aggregate_appearance = _number(aggregate.get("material_appearance_score"))
    aggregate_color = _number(aggregate.get("material_color_score"))
    pass_color_score = _number(thresholds.get("pass_color_score"))
    strong_alignment_score = _number(thresholds.get("strong_alignment_score"))
    if (
        aggregate_appearance is None
        or aggregate_appearance < minimum_aggregate_appearance_score
    ):
        reasons.append("AGGREGATE_APPEARANCE_BELOW_FLOOR")
    if (
        aggregate_color is None
        or pass_color_score is None
        or aggregate_color < pass_color_score
    ):
        reasons.append("AGGREGATE_COLOR_BELOW_PASS_FLOOR")

    comparable_count = aggregate.get("comparable_view_count")
    passed_count = aggregate.get("passed_view_count")
    review_count = aggregate.get("review_view_count")
    failed_count = aggregate.get("failed_view_count")
    unscorable_count = aggregate.get("unscorable_view_count")
    counts = (
        comparable_count,
        passed_count,
        review_count,
        failed_count,
        unscorable_count,
    )
    if any(isinstance(value, bool) or not isinstance(value, int) for value in counts):
        reasons.append("AGGREGATE_VIEW_COUNTS_INVALID")
    elif (
        comparable_count != len(raw_views)
        or passed_count + review_count != comparable_count
        or failed_count != 0
        or unscorable_count != 0
    ):
        reasons.append("AGGREGATE_HAS_FAILED_OR_UNSCORABLE_VIEWS")

    view_records: list[dict[str, Any]] = []
    for index, raw_view in enumerate(raw_views):
        view_reasons: list[str] = []
        if not isinstance(raw_view, Mapping):
            reasons.append(f"VIEW_{index}_INVALID")
            continue
        view_id = str(raw_view.get("reference_view_id") or index)
        status = raw_view.get("status")
        declared_reasons = raw_view.get("reasons")
        if not isinstance(declared_reasons, list) or any(
            not isinstance(reason, str) for reason in declared_reasons
        ):
            view_reasons.append("VIEW_REASON_CODES_INVALID")
            declared_reason_set: set[str] = set()
        else:
            declared_reason_set = set(declared_reasons)
        if status not in {"PASS", "REVIEW"}:
            view_reasons.append("VIEW_STATUS_NOT_PASS_OR_REVIEW")
        if status == "PASS" and declared_reason_set:
            view_reasons.append("PASS_VIEW_HAS_REVIEW_REASONS")
        if status == "REVIEW" and (
            not declared_reason_set
            or not declared_reason_set.issubset(ALLOWED_REVIEW_REASONS)
        ):
            view_reasons.append("VIEW_HAS_NONPHOTOMETRIC_REVIEW_REASON")

        appearance = _number(raw_view.get("material_appearance_score"))
        color = raw_view.get("material_color")
        texture = raw_view.get("material_texture")
        alignment = raw_view.get("alignment")
        color_score = (
            _number(color.get("score")) if isinstance(color, Mapping) else None
        )
        trusted_color_recall = (
            _number(color.get("trusted_evidence_color_recall"))
            if isinstance(color, Mapping)
            else None
        )
        alignment_score = (
            _number(alignment.get("score"))
            if isinstance(alignment, Mapping)
            else None
        )
        if appearance is None or appearance < minimum_view_appearance_score:
            view_reasons.append("VIEW_APPEARANCE_BELOW_FLOOR")
        if (
            color_score is None
            or pass_color_score is None
            or color_score < pass_color_score
        ):
            view_reasons.append("VIEW_COLOR_BELOW_PASS_FLOOR")
        if (
            trusted_color_recall is None
            or trusted_color_recall < MINIMUM_TRUSTED_COLOR_RECALL
        ):
            view_reasons.append("VIEW_TRUSTED_COLOR_RECALL_BELOW_FLOOR")
        if (
            alignment_score is None
            or strong_alignment_score is None
            or alignment_score < strong_alignment_score
        ):
            view_reasons.append("VIEW_ALIGNMENT_BELOW_STRONG_FLOOR")
        if not isinstance(texture, Mapping) or texture.get("status") != "PASS":
            view_reasons.append("VIEW_TEXTURE_NOT_PASS")
        if view_reasons:
            reasons.extend(f"{view_id}:{reason}" for reason in view_reasons)
        view_records.append(
            {
                "reference_view_id": view_id,
                "raw_status": status,
                "raw_reason_codes": sorted(declared_reason_set),
                "appearance_score": appearance,
                "color_score": color_score,
                "trusted_color_recall": trusted_color_recall,
                "alignment_score": alignment_score,
                "accepted": not view_reasons,
                "reason_codes": sorted(set(view_reasons)),
            }
        )

    unique_reasons = sorted(set(reasons))
    accepted = not unique_reasons
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS" if accepted else "FAIL_CLOSED",
        "acceptance_allowed": accepted,
        "decision": DECISION if accepted else None,
        "reason_codes": unique_reasons,
        "quality_report_sha256": _canonical_sha256(quality_report),
        "policy": {
            "minimum_aggregate_appearance_score": (
                minimum_aggregate_appearance_score
            ),
            "minimum_view_appearance_score": minimum_view_appearance_score,
            "minimum_trusted_color_recall": MINIMUM_TRUSTED_COLOR_RECALL,
            "allowed_raw_aggregate_status": "REVIEW",
            "allowed_review_reason_codes": sorted(ALLOWED_REVIEW_REASONS),
            "requires_zero_failed_or_unscorable_views": True,
            "requires_color_pass_floor": True,
            "requires_texture_pass": True,
            "requires_strong_alignment": True,
        },
        "metrics": {
            "aggregate_appearance_score": aggregate_appearance,
            "aggregate_color_score": aggregate_color,
            "view_count": len(raw_views),
            "passed_view_count": passed_count,
            "review_view_count": review_count,
            "failed_view_count": failed_count,
            "unscorable_view_count": unscorable_count,
        },
        "views": view_records,
    }
