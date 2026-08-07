"""Dark, semantic and repeated-geometry reference diagnostics."""

from __future__ import annotations

import math
from typing import Any

from ..config import canonical_sha256
from ..policy_exact_cover import _require_exact_int
from .constants import (
    QUALITY_REPAIR_DARK_FOREGROUND_MAX_TOTAL_CONTRIBUTION_FACTOR,
    QUALITY_REPAIR_DARK_FOREGROUND_RESIDUAL_DEFICIT_SOURCE,
    QUALITY_REPAIR_DARK_FOREGROUND_THRESHOLDS,
    QUALITY_REPAIR_MULTIVIEW_DARK_IDENTITY_LANE,
    QUALITY_REPAIR_MULTIVIEW_SEMANTIC_REVIEW_MIN_SUPPORTS,
    QUALITY_REPAIR_REPEATED_GEOMETRY_DARK_RESIDUAL_LANE,
    QUALITY_REPAIR_REPEATED_GEOMETRY_DARK_THRESHOLDS,
    QUALITY_REPAIR_SEMANTIC_REVIEW_OVERRIDE_THRESHOLDS,
)
from .metrics import (
    _quality_finite_number,
    _quality_group_recalls,
    _quality_linear_quantile,
    _quality_sha256,
    _quality_unit,
)


def _quality_dark_reference_evidence(
    spatial_report: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    raw_records = spatial_report.get("reference_evidence")
    if not isinstance(raw_records, list):
        raise RuntimeError("Dark-foreground repair lacks spatial reference evidence")
    records: dict[str, dict[str, Any]] = {}
    for raw_record in raw_records:
        if not isinstance(raw_record, dict):
            raise RuntimeError("Dark-foreground spatial reference evidence is invalid")
        view_id = raw_record.get("view_id")
        if (
            not isinstance(view_id, str)
            or not view_id
            or view_id in records
            or not _quality_sha256(raw_record.get("raw_sha256"))
            or not _quality_sha256(raw_record.get("normalized_pixel_sha256"))
            or not isinstance(raw_record.get("content_cluster_id"), str)
            or not raw_record["content_cluster_id"]
        ):
            raise RuntimeError("Dark-foreground spatial reference evidence is invalid")
        records[view_id] = raw_record
    return records


def _quality_dark_spatial_part(
    spatial_report: dict[str, Any],
    part_id: str,
) -> dict[str, Any]:
    raw_parts = spatial_report.get("parts")
    if not isinstance(raw_parts, list):
        raise RuntimeError("Dark-foreground repair lacks spatial part evidence")
    matches = [
        item
        for item in raw_parts
        if isinstance(item, dict) and item.get("part_id") == part_id
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Dark-foreground spatial part evidence is not unique: {part_id}"
        )
    return matches[0]


def _validate_quality_multiview_semantic_review_support(
    *,
    spatial_report: dict[str, Any],
    spatial_gate_audit: dict[str, Any],
    mapping_consensus: dict[str, Any],
    part_id: str,
    canonical_group_id: str,
    deficit_view_ids: list[str],
) -> dict[str, list[str]]:
    """Independently recompute the three-view review-only localization lane."""

    references = _quality_dark_reference_evidence(spatial_report)
    spatial_part = _quality_dark_spatial_part(spatial_report, part_id)
    spatial_policy = spatial_report.get("policy")
    if not isinstance(spatial_policy, dict):
        raise RuntimeError("Semantic-review repair lacks a spatial policy")
    confidence_floor = _quality_unit(
        spatial_policy.get("minimum_semantic_conflict_confidence"),
        "Semantic-review conflict confidence floor",
    )
    raw_votes = spatial_part.get("semantic_votes")
    if not isinstance(raw_votes, list):
        raise RuntimeError("Semantic-review repair lacks semantic votes")

    support_views: set[str] = set()
    alternative_views: set[str] = set()
    seen_views: set[str] = set()
    for raw_vote in raw_votes:
        if not isinstance(raw_vote, dict):
            continue
        view_id = raw_vote.get("view_id")
        reference = references.get(view_id) if isinstance(view_id, str) else None
        if (
            not isinstance(view_id, str)
            or not isinstance(reference, dict)
            or reference.get("alignment_trusted") is not True
        ):
            continue
        if view_id in seen_views:
            raise RuntimeError("Semantic-review repair has duplicate view votes")
        seen_views.add(view_id)
        confidence = _quality_unit(
            raw_vote.get("effective_confidence"),
            "Semantic-review effective confidence",
        )
        eligible = (
            raw_vote.get("reference_sha256") == reference.get("raw_sha256")
            and raw_vote.get("normalized_pixel_sha256")
            == reference.get("normalized_pixel_sha256")
            and raw_vote.get("content_cluster_id")
            == reference.get("content_cluster_id")
            and raw_vote.get("pose_cluster_id") == reference.get("pose_cluster_id")
            and raw_vote.get("alignment_trusted") is True
            and raw_vote.get("cad_part_visibility_eligible") is True
            and raw_vote.get("unique_canonical_join") is True
            and raw_vote.get("pixel_gate_accepted") is True
            and raw_vote.get("status") in {"matched", "review"}
            and isinstance(raw_vote.get("canonical_group_id"), str)
            and confidence >= confidence_floor
        )
        if not eligible:
            continue
        if raw_vote.get("canonical_group_id") == canonical_group_id:
            if view_id in deficit_view_ids:
                support_views.add(view_id)
        else:
            alternative_views.add(view_id)
    if alternative_views:
        raise RuntimeError("Semantic-review repair has a trusted group conflict")

    observations = spatial_part.get("observations")
    if not isinstance(observations, list):
        raise RuntimeError("Semantic-review repair lacks spatial observations")
    for observation in observations:
        if not isinstance(observation, dict):
            continue
        view_id = observation.get("reference_view_id")
        reference = references.get(view_id) if isinstance(view_id, str) else None
        if not isinstance(reference, dict) or reference.get("alignment_trusted") is not True:
            continue
        observed_group = observation.get("canonical_group_id")
        if (
            observation.get("classification") == "resolved"
            and observation.get("registration_label_stable") is True
            and observation.get("perturbation_label_stable") is True
            and isinstance(observed_group, str)
            and observed_group != canonical_group_id
        ):
            raise RuntimeError(
                "Semantic-review repair conflicts with stable spatial evidence"
            )
        for field in ("small_part_diagnostic", "canonical_palette_diagnostic"):
            diagnostic = observation.get(field)
            if (
                isinstance(diagnostic, dict)
                and diagnostic.get("status") == "resolved"
                and diagnostic.get("reason_codes") == []
                and isinstance(diagnostic.get("canonical_group_id"), str)
                and diagnostic.get("canonical_group_id") != canonical_group_id
            ):
                raise RuntimeError(
                    "Semantic-review repair conflicts with a resolved diagnostic"
                )

    for label, audit in (
        ("spatial gate", spatial_gate_audit),
        ("mapping consensus", mapping_consensus),
    ):
        decisions = audit.get("decisions")
        if not isinstance(decisions, list):
            raise RuntimeError(f"Semantic-review {label} decisions are invalid")
        matches = [
            item
            for item in decisions
            if isinstance(item, dict) and item.get("part_id") == part_id
        ]
        if len(matches) > 1:
            raise RuntimeError(f"Semantic-review {label} decision is not unique")
        if (
            matches
            and matches[0].get("output_status") == "matched"
            and matches[0].get("output_group_id") != canonical_group_id
        ):
            raise RuntimeError(f"Semantic-review repair conflicts with {label}")

    expected_views = sorted(set(deficit_view_ids))
    if (
        len(expected_views) < QUALITY_REPAIR_MULTIVIEW_SEMANTIC_REVIEW_MIN_SUPPORTS
        or sorted(support_views) != expected_views
    ):
        raise RuntimeError(
            "Semantic-review repair does not cover three trusted QA-deficit views"
        )
    support_records = [references[view_id] for view_id in expected_views]
    for field in (
        "raw_sha256",
        "normalized_pixel_sha256",
        "content_cluster_id",
        "pose_cluster_id",
    ):
        values = [record.get(field) for record in support_records]
        if (
            any(not isinstance(value, str) or not value for value in values)
            or len(set(values)) != len(values)
        ):
            raise RuntimeError(
                "Semantic-review supporting references are not independent"
            )
    return {
        "supporting_view_ids": expected_views,
        "supporting_content_cluster_ids": sorted(
            {str(record["content_cluster_id"]) for record in support_records}
        ),
        "supporting_pose_cluster_ids": sorted(
            {str(record["pose_cluster_id"]) for record in support_records}
        ),
    }


def _quality_dark_spatial_observation(
    spatial_part: dict[str, Any],
    view_id: str,
) -> dict[str, Any]:
    raw_observations = spatial_part.get("observations")
    if not isinstance(raw_observations, list):
        raise RuntimeError("Dark-foreground repair lacks spatial observations")
    matches = [
        item
        for item in raw_observations
        if isinstance(item, dict) and item.get("reference_view_id") == view_id
    ]
    if len(matches) != 1:
        raise RuntimeError(
            "Dark-foreground spatial observation is not unique: "
            f"{spatial_part.get('part_id')}/{view_id}"
        )
    return matches[0]


def _quality_dark_alignment(
    spatial_report: dict[str, Any],
    view_id: str,
) -> dict[str, Any]:
    raw_alignments = spatial_report.get("view_alignments")
    if not isinstance(raw_alignments, list):
        raise RuntimeError("Dark-foreground repair lacks spatial alignments")
    matches = [
        item
        for item in raw_alignments
        if isinstance(item, dict) and item.get("reference_view_id") == view_id
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Dark-foreground spatial alignment is not unique: {view_id}"
        )
    return matches[0]


def _quality_group_share(
    raw_scores: Any,
    *,
    canonical_group_id: str,
    label: str,
) -> float:
    if not isinstance(raw_scores, list) or not raw_scores:
        raise RuntimeError(f"{label} group scores are invalid")
    total = 0.0
    matched = False
    for raw_score in raw_scores:
        if not isinstance(raw_score, dict):
            raise RuntimeError(f"{label} group score is invalid")
        share = _quality_unit(raw_score.get("color_share"), f"{label} color share")
        if raw_score.get("canonical_group_id") == canonical_group_id:
            total += share
            matched = True
    if not matched or total > 1.0 + 1e-9:
        raise RuntimeError(f"{label} canonical group share is invalid")
    return min(1.0, total)


def _validate_quality_semantic_review_override(
    *,
    spatial_report: dict[str, Any],
    spatial_gate_audit: dict[str, Any],
    mapping_consensus: dict[str, Any],
    part_id: str,
    canonical_group_id: str,
    target_view_id: str,
) -> tuple[list[str], list[str]]:
    """Recompute one review-only semantic override without trusting the compiler.

    A dominant residual may supersede a Qwen ``review`` vote only when the
    target projection is strong in the direct mask, bounding box, and all four
    registration perturbations; the rejected semantic group is absent from
    those same six pixel samples; and a distinct trusted reference carries a
    high-confidence semantic vote for the target group.  A matched alternative
    from either mapping gate remains a hard conflict.
    """

    thresholds = QUALITY_REPAIR_SEMANTIC_REVIEW_OVERRIDE_THRESHOLDS
    references = _quality_dark_reference_evidence(spatial_report)
    target_reference = references.get(target_view_id)
    if (
        not isinstance(target_reference, dict)
        or target_reference.get("alignment_trusted") is not True
    ):
        raise RuntimeError("Semantic-review override target reference is not trusted")
    spatial_part = _quality_dark_spatial_part(spatial_report, part_id)
    observation = _quality_dark_spatial_observation(spatial_part, target_view_id)
    alignment = _quality_dark_alignment(spatial_report, target_view_id)
    transform_audit = alignment.get("ecc_transform_audit")
    if (
        alignment.get("trusted") is not True
        or alignment.get("reason_codes") != []
        or alignment.get("ecc_status") != "success"
        or not isinstance(transform_audit, dict)
        or transform_audit.get("constraints_passed") is not True
        or _quality_unit(
            alignment.get("score"), "Semantic-review override alignment score"
        )
        < thresholds["minimum_alignment_score"]
        or _quality_unit(
            alignment.get("projection_iou"),
            "Semantic-review override projection IoU",
        )
        < thresholds["minimum_projection_iou"]
        or _quality_unit(
            alignment.get("ecc_correlation"),
            "Semantic-review override ECC correlation",
        )
        < thresholds["minimum_ecc_correlation"]
    ):
        raise RuntimeError("Semantic-review override alignment is below its floor")

    declared_pixels = _require_exact_int(
        observation.get("declared_visible_pixels"),
        "Semantic-review override declared pixels",
    )
    projected_pixels = _require_exact_int(
        observation.get("projected_part_pixels"),
        "Semantic-review override projected pixels",
    )
    minimum_pixels = int(thresholds["minimum_projected_pixels"])
    direct_scores = observation.get("group_scores")
    bbox_scores = observation.get("bbox_group_scores")
    direct_share = _quality_group_share(
        direct_scores,
        canonical_group_id=canonical_group_id,
        label="Semantic-review override direct",
    )
    bbox_share = _quality_group_share(
        bbox_scores,
        canonical_group_id=canonical_group_id,
        label="Semantic-review override bbox",
    )
    direct_margin = _quality_unit(
        observation.get("color_margin"), "Semantic-review override direct margin"
    )
    bbox_margin = _quality_unit(
        observation.get("bbox_color_margin"),
        "Semantic-review override bbox margin",
    )
    direct_winner = direct_scores[0] if isinstance(direct_scores, list) else None
    if (
        observation.get("registration_label_stable") is not True
        or observation.get("perturbation_label_stable") is not True
        or declared_pixels < minimum_pixels
        or projected_pixels < minimum_pixels
        or not isinstance(direct_winner, dict)
        or direct_winner.get("canonical_group_id") != canonical_group_id
        or direct_share < thresholds["minimum_direct_target_share"]
        or direct_margin < thresholds["minimum_direct_target_margin"]
        or observation.get("bbox_canonical_group_id") != canonical_group_id
        or bbox_share < thresholds["minimum_bbox_target_share"]
        or bbox_margin < thresholds["minimum_bbox_target_margin"]
    ):
        raise RuntimeError("Semantic-review override direct projection is not strong")

    required_offsets = {(-2, 0), (2, 0), (0, -2), (0, 2)}
    raw_perturbations = observation.get("projection_perturbations")
    if not isinstance(raw_perturbations, list) or len(raw_perturbations) != 4:
        raise RuntimeError("Semantic-review override perturbations are incomplete")
    perturbations_by_offset: dict[tuple[int, int], dict[str, Any]] = {}
    for raw_perturbation in raw_perturbations:
        if not isinstance(raw_perturbation, dict):
            raise RuntimeError("Semantic-review override perturbation is invalid")
        offset = raw_perturbation.get("offset_pixels")
        if (
            not isinstance(offset, list)
            or len(offset) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in offset
            )
            or tuple(offset) not in required_offsets
            or tuple(offset) in perturbations_by_offset
            or raw_perturbation.get("canonical_group_id") != canonical_group_id
            or raw_perturbation.get("diagnostic_canonical_group_id")
            != canonical_group_id
            or _require_exact_int(
                raw_perturbation.get("sampled_reference_pixels"),
                "Semantic-review override perturbation pixels",
            )
            < minimum_pixels
            or _quality_unit(
                raw_perturbation.get("best_color_share"),
                "Semantic-review override perturbation share",
            )
            < thresholds["minimum_perturbation_target_share"]
            or _quality_unit(
                raw_perturbation.get("color_margin"),
                "Semantic-review override perturbation margin",
            )
            < thresholds["minimum_perturbation_target_margin"]
        ):
            raise RuntimeError("Semantic-review override perturbation is not strong")
        perturbations_by_offset[tuple(offset)] = raw_perturbation
    if set(perturbations_by_offset) != required_offsets:
        raise RuntimeError("Semantic-review override perturbation coverage is invalid")

    raw_votes = spatial_part.get("semantic_votes")
    if not isinstance(raw_votes, list):
        raise RuntimeError("Semantic-review override votes are invalid")
    review_conflicts: list[dict[str, Any]] = []
    for raw_vote in raw_votes:
        if not isinstance(raw_vote, dict):
            raise RuntimeError("Semantic-review override vote is invalid")
        if (
            raw_vote.get("view_id") == target_view_id
            and raw_vote.get("status") == "review"
            and raw_vote.get("alignment_trusted") is True
            and raw_vote.get("unique_canonical_join") is True
            and raw_vote.get("pixel_gate_accepted") is True
            and isinstance(raw_vote.get("canonical_group_id"), str)
            and raw_vote.get("canonical_group_id") != canonical_group_id
        ):
            _quality_unit(
                raw_vote.get("effective_confidence"),
                "Semantic-review override conflict confidence",
            )
            review_conflicts.append(raw_vote)
    if not review_conflicts:
        raise RuntimeError("Semantic-review override lacks one review-only conflict")

    diagnostic = observation.get("canonical_palette_diagnostic")
    if not isinstance(diagnostic, dict):
        raise RuntimeError("Semantic-review override lacks canonical pixel samples")
    diagnostic_perturbations = diagnostic.get("projection_perturbations")
    pixel_samples = [
        diagnostic.get("direct_sample"),
        diagnostic.get("bbox_sample"),
        *(
            diagnostic_perturbations
            if isinstance(diagnostic_perturbations, list)
            else []
        ),
    ]
    if len(pixel_samples) != 6 or any(
        not isinstance(sample, dict) for sample in pixel_samples
    ):
        raise RuntimeError("Semantic-review override pixel samples are incomplete")
    for conflict in review_conflicts:
        rejected_group_id = str(conflict["canonical_group_id"])
        rejected_shares = [
            _quality_group_share(
                sample.get("group_scores"),
                canonical_group_id=rejected_group_id,
                label="Semantic-review override rejected group",
            )
            for sample in pixel_samples
        ]
        if (
            rejected_shares[0] > thresholds["maximum_rejected_group_share"]
            or rejected_shares[1] > thresholds["maximum_rejected_group_share"]
            or sum(
                share <= thresholds["maximum_rejected_group_share"]
                for share in rejected_shares
            )
            < thresholds["minimum_clean_rejected_group_samples"]
            or max(rejected_shares)
            > thresholds["maximum_rejected_group_outlier_share"]
        ):
            raise RuntimeError(
                "Semantic-review override did not pixel-disprove the review group"
            )

    spatial_policy = spatial_report.get("policy")
    if not isinstance(spatial_policy, dict):
        raise RuntimeError("Semantic-review override lacks a spatial policy")
    minimum_semantic_confidence = _quality_unit(
        spatial_policy.get("minimum_semantic_confidence"),
        "Semantic-review override semantic confidence floor",
    )
    if _quality_dark_matched_semantic_conflict(
        part_id=part_id,
        canonical_group_id=canonical_group_id,
        spatial_part=spatial_part,
        spatial_gate_audit=spatial_gate_audit,
        mapping_consensus=mapping_consensus,
        minimum_semantic_confidence=minimum_semantic_confidence,
    ):
        raise RuntimeError("Semantic-review override has a matched semantic conflict")

    independence_fields = (
        "raw_sha256",
        "normalized_pixel_sha256",
        "content_cluster_id",
        "pose_cluster_id",
    )
    anchor_view_ids: list[str] = []
    for raw_vote in raw_votes:
        view_id = raw_vote.get("view_id") if isinstance(raw_vote, dict) else None
        anchor_reference = references.get(view_id) if isinstance(view_id, str) else None
        if (
            not isinstance(raw_vote, dict)
            or not isinstance(view_id, str)
            or view_id == target_view_id
            or raw_vote.get("status") not in {"matched", "review"}
            or raw_vote.get("canonical_group_id") != canonical_group_id
            or raw_vote.get("alignment_trusted") is not True
            or raw_vote.get("unique_canonical_join") is not True
            or raw_vote.get("pixel_gate_accepted") is not True
            or not isinstance(anchor_reference, dict)
            or anchor_reference.get("alignment_trusted") is not True
        ):
            continue
        confidence = _quality_unit(
            raw_vote.get("effective_confidence"),
            "Semantic-review override anchor confidence",
        )
        if confidence < thresholds["minimum_anchor_effective_confidence"]:
            continue
        if all(
            isinstance(target_reference.get(field), str)
            and target_reference.get(field)
            and isinstance(anchor_reference.get(field), str)
            and anchor_reference.get(field)
            and target_reference[field] != anchor_reference[field]
            for field in independence_fields
        ):
            anchor_view_ids.append(view_id)
    anchor_view_ids = sorted(set(anchor_view_ids))
    if not anchor_view_ids:
        raise RuntimeError("Semantic-review override lacks an independent target anchor")
    return [target_view_id], anchor_view_ids


def _validate_quality_dark_foreground_diagnostic(
    *,
    diagnostic: dict[str, Any],
    observation: dict[str, Any],
    alignment: dict[str, Any],
    references: dict[str, dict[str, Any]],
    target_view_id: str,
    canonical_group_id: str,
    allow_moderate_cohort_alignment: bool = False,
) -> dict[str, Any]:
    """Validate every signed primitive in one dark-on-black projection proof."""

    expected_keys = {
        "status",
        "reason_codes",
        "evidence_scope",
        "canonical_group_id",
        "canonical_source_view_ids",
        "alternative_canonical_group_ids",
        "projected_part_pixels",
        "normalized_projected_pixels",
        "normalization",
        "alignment",
        "background",
        "thresholds",
        "near_black_pixels",
        "near_black_share",
        "non_background_pixels",
        "non_background_share",
        "dark_signal_pixels",
        "dark_signal_share",
        "dark_signal_purity",
        "core_pixels",
        "core_dark_signal_pixels",
        "core_dark_signal_share",
        "core_distance_pixels",
        "adaptive_edge_pixels",
        "adaptive_edge_density",
        "adaptive_edge_threshold",
        "border_gradient_p99",
        "canny_low_threshold",
        "canny_high_threshold",
        "canny_edge_pixels",
        "canny_edge_density",
        "null_shifts",
        "valid_null_shift_count",
        "null_dark_signal_share_q75",
        "dark_signal_null_margin",
        "normalized_reference_pixel_sha256",
        "normalized_projected_mask_sha256",
        "normalized_near_black_mask_sha256",
        "normalized_non_background_mask_sha256",
        "normalized_dark_signal_mask_sha256",
        "normalized_adaptive_edge_mask_sha256",
        "diagnostic_sha256",
    }
    if set(diagnostic) != expected_keys:
        raise RuntimeError("Dark-foreground diagnostic schema is incomplete")
    digest = diagnostic.get("diagnostic_sha256")
    unsigned = {
        key: value
        for key, value in diagnostic.items()
        if key != "diagnostic_sha256"
    }
    if not _quality_sha256(digest) or digest != canonical_sha256(unsigned):
        raise RuntimeError("Dark-foreground diagnostic hash is invalid")
    if allow_moderate_cohort_alignment:
        expected_status = "rejected"
        expected_reason_codes = [
            "DARK_CANONICAL_GROUP_CONFLICT",
            "DARK_ALIGNMENT_NOT_STRONG",
        ]
        alternative_group_ids = diagnostic.get(
            "alternative_canonical_group_ids"
        )
        alternatives_are_valid = (
            isinstance(alternative_group_ids, list)
            and bool(alternative_group_ids)
            and alternative_group_ids == sorted(set(alternative_group_ids))
            and all(
                isinstance(group_id, str) and group_id
                for group_id in alternative_group_ids
            )
        )
    else:
        expected_status = "resolved"
        expected_reason_codes = []
        alternatives_are_valid = (
            diagnostic.get("alternative_canonical_group_ids") == []
        )
    if (
        diagnostic.get("status") != expected_status
        or diagnostic.get("reason_codes") != expected_reason_codes
        or diagnostic.get("evidence_scope")
        != "dark_on_black_foreground_repair_only"
        or diagnostic.get("canonical_group_id") != canonical_group_id
        or not alternatives_are_valid
    ):
        raise RuntimeError("Dark-foreground diagnostic is not an exact resolution")

    source_view_ids = diagnostic.get("canonical_source_view_ids")
    if (
        not isinstance(source_view_ids, list)
        or source_view_ids != sorted(set(source_view_ids))
        or len(source_view_ids) < 2
        or any(view_id not in references for view_id in source_view_ids)
        or len({references[view_id]["raw_sha256"] for view_id in source_view_ids}) < 2
        or len(
            {
                references[view_id]["content_cluster_id"]
                for view_id in source_view_ids
            }
        )
        < 2
    ):
        raise RuntimeError(
            "Dark-foreground diagnostic lacks independent canonical sources"
        )
    target_reference = references.get(target_view_id)
    if (
        not isinstance(target_reference, dict)
        or target_reference.get("alignment_trusted") is not True
    ):
        raise RuntimeError("Dark-foreground target reference is not trusted")

    projected_pixels = _require_exact_int(
        diagnostic.get("projected_part_pixels"),
        "Dark-foreground projected pixels",
        minimum=1,
    )
    if projected_pixels != observation.get("projected_part_pixels"):
        raise RuntimeError("Dark-foreground projected-pixel evidence is inconsistent")
    normalized_pixels = _require_exact_int(
        diagnostic.get("normalized_projected_pixels"),
        "Dark-foreground normalized projected pixels",
        minimum=1,
    )

    normalization = diagnostic.get("normalization")
    if not isinstance(normalization, dict) or set(normalization) != {
        "long_edge_pixels",
        "original_size",
        "normalized_size",
        "scale",
    }:
        raise RuntimeError("Dark-foreground normalization audit is invalid")
    if normalization.get("long_edge_pixels") != 512:
        raise RuntimeError("Dark-foreground normalization long edge is invalid")
    original_size = normalization.get("original_size")
    normalized_size = normalization.get("normalized_size")
    if (
        not isinstance(original_size, list)
        or len(original_size) != 2
        or not isinstance(normalized_size, list)
        or len(normalized_size) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in [*original_size, *normalized_size]
        )
        or max(normalized_size) != 512
    ):
        raise RuntimeError("Dark-foreground normalization dimensions are invalid")
    scale = _quality_finite_number(
        normalization.get("scale"),
        "Dark-foreground normalization scale",
        minimum=0.0,
    )
    expected_scale = 512.0 / max(original_size)
    if (
        not math.isclose(scale, expected_scale, rel_tol=0.0, abs_tol=1e-9)
        or normalized_size
        != [max(1, round(value * expected_scale)) for value in original_size]
    ):
        raise RuntimeError("Dark-foreground normalization transform is inconsistent")

    diagnostic_alignment = diagnostic.get("alignment")
    expected_alignment_keys = {
        "trusted",
        "reason_codes_empty",
        "score",
        "projection_score",
        "projection_iou",
        "ecc_status",
        "ecc_correlation",
        "transform_constraints_passed",
        "strong",
    }
    if (
        not isinstance(diagnostic_alignment, dict)
        or set(diagnostic_alignment) != expected_alignment_keys
        or diagnostic_alignment.get("trusted") is not True
        or diagnostic_alignment.get("reason_codes_empty") is not True
        or diagnostic_alignment.get("ecc_status") != "success"
        or diagnostic_alignment.get("transform_constraints_passed") is not True
        or diagnostic_alignment.get("strong")
        is not (False if allow_moderate_cohort_alignment else True)
        or alignment.get("trusted") is not True
        or alignment.get("reason_codes") != []
        or alignment.get("ecc_status") != "success"
        or not isinstance(alignment.get("ecc_transform_audit"), dict)
        or alignment["ecc_transform_audit"].get("constraints_passed") is not True
    ):
        raise RuntimeError("Dark-foreground diagnostic alignment is not strong")
    alignment_minimums = (
        {
            "score": 0.80,
            "projection_score": 0.85,
            "projection_iou": 0.85,
            "ecc_correlation": 0.90,
        }
        if allow_moderate_cohort_alignment
        else {
            "score": 0.85,
            "projection_score": 0.85,
            "projection_iou": 0.85,
            "ecc_correlation": 0.90,
        }
    )
    for field, minimum in alignment_minimums.items():
        diagnostic_value = _quality_unit(
            diagnostic_alignment.get(field),
            f"Dark-foreground diagnostic alignment {field}",
        )
        source_value = _quality_unit(
            alignment.get(field),
            f"Dark-foreground source alignment {field}",
        )
        if diagnostic_value < minimum or not math.isclose(
            diagnostic_value,
            source_value,
            rel_tol=0.0,
            abs_tol=1e-8,
        ):
            raise RuntimeError("Dark-foreground diagnostic alignment is inconsistent")
    if not math.isclose(
        _quality_unit(
            target_reference.get("alignment_score"),
            "Dark-foreground reference alignment score",
        ),
        _quality_unit(diagnostic_alignment["score"], "Dark-foreground score"),
        rel_tol=0.0,
        abs_tol=1e-8,
    ):
        raise RuntimeError("Dark-foreground reference alignment is inconsistent")

    background = diagnostic.get("background")
    if not isinstance(background, dict) or set(background) != {
        "median_bgr",
        "border_distance_p95",
        "distance_threshold",
    }:
        raise RuntimeError("Dark-foreground background model is invalid")
    median_bgr = background.get("median_bgr")
    if (
        not isinstance(median_bgr, list)
        or len(median_bgr) != 3
        or any(
            _quality_finite_number(
                value,
                "Dark-foreground background channel",
                minimum=0.0,
                maximum=255.0,
            )
            < 0.0
            for value in median_bgr
        )
    ):
        raise RuntimeError("Dark-foreground background median is invalid")
    border_p95 = _quality_finite_number(
        background.get("border_distance_p95"),
        "Dark-foreground border distance",
        minimum=0.0,
    )
    distance_threshold = _quality_finite_number(
        background.get("distance_threshold"),
        "Dark-foreground background threshold",
        minimum=0.0,
    )
    if not math.isclose(
        distance_threshold,
        min(45.0, max(12.0, border_p95 + 6.0)),
        rel_tol=0.0,
        abs_tol=1e-7,
    ):
        raise RuntimeError("Dark-foreground background threshold is inconsistent")

    thresholds = diagnostic.get("thresholds")
    if thresholds != QUALITY_REPAIR_DARK_FOREGROUND_THRESHOLDS:
        raise RuntimeError("Dark-foreground diagnostic thresholds were changed")

    count_fields = (
        "near_black_pixels",
        "non_background_pixels",
        "dark_signal_pixels",
        "core_pixels",
        "core_dark_signal_pixels",
        "adaptive_edge_pixels",
        "canny_edge_pixels",
    )
    counts = {
        field: _require_exact_int(
            diagnostic.get(field),
            f"Dark-foreground {field}",
        )
        for field in count_fields
    }
    if any(
        counts[field] > normalized_pixels
        for field in (
            "near_black_pixels",
            "non_background_pixels",
            "dark_signal_pixels",
            "core_pixels",
            "adaptive_edge_pixels",
            "canny_edge_pixels",
        )
    ) or counts["core_dark_signal_pixels"] > counts["core_pixels"]:
        raise RuntimeError("Dark-foreground diagnostic counts are inconsistent")
    ratios = {
        field: _quality_unit(
            diagnostic.get(field),
            f"Dark-foreground {field}",
        )
        for field in (
            "near_black_share",
            "non_background_share",
            "dark_signal_share",
            "dark_signal_purity",
            "core_dark_signal_share",
            "adaptive_edge_density",
            "canny_edge_density",
        )
    }
    expected_ratios = {
        "near_black_share": counts["near_black_pixels"] / normalized_pixels,
        "non_background_share": (
            counts["non_background_pixels"] / normalized_pixels
        ),
        "dark_signal_share": counts["dark_signal_pixels"] / normalized_pixels,
        "dark_signal_purity": (
            counts["dark_signal_pixels"] / counts["non_background_pixels"]
            if counts["non_background_pixels"]
            else 0.0
        ),
        "core_dark_signal_share": (
            counts["core_dark_signal_pixels"] / counts["core_pixels"]
            if counts["core_pixels"]
            else 0.0
        ),
        "adaptive_edge_density": (
            counts["adaptive_edge_pixels"] / normalized_pixels
        ),
        "canny_edge_density": counts["canny_edge_pixels"] / normalized_pixels,
    }
    if any(
        not math.isclose(
            ratios[field],
            expected,
            rel_tol=0.0,
            abs_tol=1e-8,
        )
        for field, expected in expected_ratios.items()
    ):
        raise RuntimeError("Dark-foreground diagnostic ratios are inconsistent")
    if (
        normalized_pixels
        < QUALITY_REPAIR_DARK_FOREGROUND_THRESHOLDS[
            "minimum_normalized_projected_pixels"
        ]
        or ratios["near_black_share"]
        < QUALITY_REPAIR_DARK_FOREGROUND_THRESHOLDS["minimum_near_black_share"]
        or counts["non_background_pixels"]
        < QUALITY_REPAIR_DARK_FOREGROUND_THRESHOLDS[
            "minimum_non_background_pixels"
        ]
        or ratios["dark_signal_share"]
        < QUALITY_REPAIR_DARK_FOREGROUND_THRESHOLDS["minimum_dark_signal_share"]
        or ratios["dark_signal_purity"]
        < QUALITY_REPAIR_DARK_FOREGROUND_THRESHOLDS["minimum_dark_signal_purity"]
        or counts["core_pixels"]
        < QUALITY_REPAIR_DARK_FOREGROUND_THRESHOLDS["minimum_core_pixels"]
        or ratios["core_dark_signal_share"]
        < QUALITY_REPAIR_DARK_FOREGROUND_THRESHOLDS[
            "minimum_core_dark_signal_share"
        ]
        or ratios["adaptive_edge_density"]
        < QUALITY_REPAIR_DARK_FOREGROUND_THRESHOLDS[
            "minimum_adaptive_edge_density"
        ]
    ):
        raise RuntimeError("Dark-foreground diagnostic primitive gate failed")
    if diagnostic.get("core_distance_pixels") != 2.2:
        raise RuntimeError("Dark-foreground diagnostic core threshold is invalid")
    border_gradient_p99 = _quality_finite_number(
        diagnostic.get("border_gradient_p99"),
        "Dark-foreground border gradient",
        minimum=0.0,
    )
    adaptive_threshold = _quality_finite_number(
        diagnostic.get("adaptive_edge_threshold"),
        "Dark-foreground adaptive edge threshold",
        minimum=0.0,
    )
    if not math.isclose(
        adaptive_threshold,
        max(12.0, border_gradient_p99 + 6.0),
        rel_tol=0.0,
        abs_tol=1e-7,
    ):
        raise RuntimeError("Dark-foreground adaptive edge threshold is inconsistent")
    canny_low = _require_exact_int(
        diagnostic.get("canny_low_threshold"),
        "Dark-foreground Canny low threshold",
        minimum=1,
    )
    canny_high = _require_exact_int(
        diagnostic.get("canny_high_threshold"),
        "Dark-foreground Canny high threshold",
        minimum=1,
    )
    if (
        canny_low >= canny_high
        or canny_high > 255
        or canny_high != min(255, max(12, round(adaptive_threshold)))
        or canny_low != max(1, round(canny_high * 0.5))
    ):
        raise RuntimeError("Dark-foreground Canny thresholds are inconsistent")

    raw_nulls = diagnostic.get("null_shifts")
    if not isinstance(raw_nulls, list) or len(raw_nulls) != 8:
        raise RuntimeError("Dark-foreground null controls are incomplete")
    null_offsets: set[tuple[int, int]] = set()
    valid_null_shares: list[float] = []
    for raw_null in raw_nulls:
        if not isinstance(raw_null, dict) or set(raw_null) != {
            "offset_pixels",
            "retained_pixels",
            "valid_area_ratio",
            "valid",
            "dark_signal_pixels",
            "dark_signal_share",
            "mask_sha256",
        }:
            raise RuntimeError("Dark-foreground null control schema is invalid")
        offset = raw_null.get("offset_pixels")
        if (
            not isinstance(offset, list)
            or len(offset) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in offset
            )
            or tuple(offset) == (0, 0)
            or tuple(offset) in null_offsets
        ):
            raise RuntimeError("Dark-foreground null-control offset is invalid")
        null_offsets.add((offset[0], offset[1]))
        retained_pixels = _require_exact_int(
            raw_null.get("retained_pixels"),
            "Dark-foreground null retained pixels",
        )
        null_dark_pixels = _require_exact_int(
            raw_null.get("dark_signal_pixels"),
            "Dark-foreground null dark pixels",
        )
        valid_area_ratio = _quality_unit(
            raw_null.get("valid_area_ratio"),
            "Dark-foreground null valid-area ratio",
        )
        null_share = _quality_unit(
            raw_null.get("dark_signal_share"),
            "Dark-foreground null dark-signal share",
        )
        expected_valid = (
            valid_area_ratio
            >= QUALITY_REPAIR_DARK_FOREGROUND_THRESHOLDS[
                "minimum_null_valid_area_ratio"
            ]
        )
        if (
            raw_null.get("valid") is not expected_valid
            or retained_pixels > normalized_pixels
            or null_dark_pixels > retained_pixels
            or not math.isclose(
                valid_area_ratio,
                retained_pixels / normalized_pixels,
                rel_tol=0.0,
                abs_tol=1e-8,
            )
            or not math.isclose(
                null_share,
                null_dark_pixels / retained_pixels if retained_pixels else 0.0,
                rel_tol=0.0,
                abs_tol=1e-8,
            )
            or not _quality_sha256(raw_null.get("mask_sha256"))
        ):
            raise RuntimeError("Dark-foreground null control is inconsistent")
        if expected_valid:
            valid_null_shares.append(null_share)
    absolute_x = {abs(x) for x, _ in null_offsets if x}
    absolute_y = {abs(y) for _, y in null_offsets if y}
    if (
        len(absolute_x) != 1
        or len(absolute_y) != 1
        or min(absolute_x)
        < QUALITY_REPAIR_DARK_FOREGROUND_THRESHOLDS[
            "minimum_null_offset_pixels"
        ]
        or min(absolute_y)
        < QUALITY_REPAIR_DARK_FOREGROUND_THRESHOLDS[
            "minimum_null_offset_pixels"
        ]
    ):
        raise RuntimeError("Dark-foreground null-control geometry is invalid")
    offset_x = next(iter(absolute_x))
    offset_y = next(iter(absolute_y))
    expected_offsets = {
        (-offset_x, 0),
        (offset_x, 0),
        (0, -offset_y),
        (0, offset_y),
        (-offset_x, -offset_y),
        (-offset_x, offset_y),
        (offset_x, -offset_y),
        (offset_x, offset_y),
    }
    if null_offsets != expected_offsets:
        raise RuntimeError("Dark-foreground null-control geometry is incomplete")
    declared_valid_count = _require_exact_int(
        diagnostic.get("valid_null_shift_count"),
        "Dark-foreground valid null count",
    )
    q75 = _quality_unit(
        diagnostic.get("null_dark_signal_share_q75"),
        "Dark-foreground null q75",
    )
    recomputed_q75 = _quality_linear_quantile(valid_null_shares, 0.75)
    null_margin = _quality_finite_number(
        diagnostic.get("dark_signal_null_margin"),
        "Dark-foreground null margin",
    )
    if (
        declared_valid_count != len(valid_null_shares)
        or declared_valid_count
        < QUALITY_REPAIR_DARK_FOREGROUND_THRESHOLDS[
            "minimum_valid_null_shifts"
        ]
        or not math.isclose(q75, recomputed_q75, rel_tol=0.0, abs_tol=1e-8)
        or not math.isclose(
            null_margin,
            ratios["dark_signal_share"] - q75,
            rel_tol=0.0,
            abs_tol=1e-8,
        )
        or null_margin
        < QUALITY_REPAIR_DARK_FOREGROUND_THRESHOLDS["minimum_null_q75_margin"]
    ):
        raise RuntimeError("Dark-foreground null-control decision is invalid")

    hash_fields = (
        "normalized_reference_pixel_sha256",
        "normalized_projected_mask_sha256",
        "normalized_near_black_mask_sha256",
        "normalized_non_background_mask_sha256",
        "normalized_dark_signal_mask_sha256",
        "normalized_adaptive_edge_mask_sha256",
    )
    if any(not _quality_sha256(diagnostic.get(field)) for field in hash_fields):
        raise RuntimeError("Dark-foreground diagnostic mask hash is invalid")
    evidence_strength = (
        ratios["dark_signal_share"]
        + ratios["dark_signal_purity"]
        + null_margin
        + ratios["adaptive_edge_density"]
    )
    return {
        "diagnostic_sha256": digest,
        "projected_part_pixels": projected_pixels,
        "normalized_projected_pixels": normalized_pixels,
        "dark_signal_share": diagnostic["dark_signal_share"],
        "dark_signal_purity": diagnostic["dark_signal_purity"],
        "dark_signal_null_margin": diagnostic["dark_signal_null_margin"],
        "adaptive_edge_density": diagnostic["adaptive_edge_density"],
        "evidence_strength": evidence_strength,
    }


def _quality_dark_matched_semantic_conflict(
    *,
    part_id: str,
    canonical_group_id: str,
    spatial_part: dict[str, Any],
    spatial_gate_audit: dict[str, Any],
    mapping_consensus: dict[str, Any],
    minimum_semantic_confidence: float,
) -> bool:
    def matched_other(document: dict[str, Any]) -> bool:
        raw_decisions = document.get("decisions")
        if not isinstance(raw_decisions, list):
            raise RuntimeError("Dark-foreground mapping decisions are invalid")
        decisions = [
            item
            for item in raw_decisions
            if isinstance(item, dict) and item.get("part_id") == part_id
        ]
        if len(decisions) > 1:
            raise RuntimeError("Dark-foreground mapping decision is not unique")
        return bool(
            decisions
            and decisions[0].get("output_status") == "matched"
            and decisions[0].get("output_group_id") != canonical_group_id
        )

    if matched_other(spatial_gate_audit) or matched_other(mapping_consensus):
        return True
    raw_votes = spatial_part.get("semantic_votes")
    if not isinstance(raw_votes, list):
        raise RuntimeError("Dark-foreground semantic votes are invalid")
    for vote in raw_votes:
        if not isinstance(vote, dict):
            raise RuntimeError("Dark-foreground semantic vote is invalid")
        confidence = vote.get("effective_confidence")
        if (
            vote.get("status") == "matched"
            and vote.get("alignment_trusted") is True
            and vote.get("unique_canonical_join") is True
            and vote.get("pixel_gate_accepted") is True
            and isinstance(vote.get("canonical_group_id"), str)
            and vote.get("canonical_group_id") != canonical_group_id
            and not isinstance(confidence, bool)
            and isinstance(confidence, (int, float))
            and math.isfinite(float(confidence))
            and float(confidence) >= minimum_semantic_confidence
        ):
            return True
    return False


def _recompute_quality_multiview_dark_identity(
    *,
    part_id: str,
    canonical_group_id: str,
    canonical_group: dict[str, Any],
    spatial_part: dict[str, Any],
    spatial_gate_decision: dict[str, Any] | None,
    mapping_decision: dict[str, Any] | None,
    dark_residual_support: dict[str, Any],
    reference_evidence: dict[str, dict[str, Any]],
    spatial_policy: dict[str, Any],
    minimum_semantic_confidence: float,
) -> tuple[dict[str, Any], list[str], list[str]]:
    """Independently validate and rebuild a multiview dark-part diagnostic."""

    if str(canonical_group.get("base_color", "")).strip().casefold() != "black":
        raise RuntimeError("Multiview dark identity requires a black group")
    consensus = spatial_part.get("multiview_dark_consensus")
    support_view_ids = (
        consensus.get("supporting_view_ids")
        if isinstance(consensus, dict)
        else None
    )
    minimum_support = spatial_policy.get("minimum_spatial_support_views")
    if (
        not isinstance(consensus, dict)
        or consensus.get("status") != "resolved"
        or consensus.get("canonical_group_id") != canonical_group_id
        or consensus.get("evidence_contract")
        != "stable_projection_and_dark_interior_multiview_consensus"
        or isinstance(minimum_support, bool)
        or not isinstance(minimum_support, int)
        or minimum_support < 2
        or consensus.get("minimum_independent_support_views") != minimum_support
        or not isinstance(support_view_ids, list)
        or support_view_ids != sorted(set(support_view_ids))
        or len(support_view_ids) < minimum_support
        or any(
            not isinstance(view_id, str) or view_id not in reference_evidence
            for view_id in support_view_ids
        )
    ):
        raise RuntimeError("Multiview dark identity has an invalid consensus")
    support_views = [str(view_id) for view_id in support_view_ids]
    support_references = [reference_evidence[view_id] for view_id in support_views]
    for independence_field in (
        "raw_sha256",
        "normalized_pixel_sha256",
        "content_cluster_id",
        "pose_cluster_id",
    ):
        values = [record.get(independence_field) for record in support_references]
        if (
            any(not isinstance(value, str) or not value for value in values)
            or len(set(values)) != len(values)
        ):
            raise RuntimeError(
                "Multiview dark identity lacks independent reference evidence"
            )

    raw_observations = spatial_part.get("observations")
    if not isinstance(raw_observations, list):
        raise RuntimeError("Multiview dark identity lacks observations")
    observations_by_view: dict[str, dict[str, Any]] = {}
    for observation in raw_observations:
        if not isinstance(observation, dict):
            continue
        view_id = observation.get("reference_view_id")
        if isinstance(view_id, str) and view_id in support_views:
            if view_id in observations_by_view:
                raise RuntimeError(
                    "Multiview dark support observation is not unique"
                )
            observations_by_view[view_id] = observation
    if set(observations_by_view) != set(support_views):
        raise RuntimeError("Multiview dark support observations are incomplete")

    minimum_color_share = _quality_unit(
        spatial_policy.get("minimum_color_share"),
        "Multiview dark minimum color share",
    )
    evidence_rows: list[dict[str, Any]] = []
    for view_id in support_views:
        observation = observations_by_view[view_id]
        scores = observation.get("group_scores")
        winner = (
            scores[0]
            if isinstance(scores, list) and scores and isinstance(scores[0], dict)
            else None
        )
        diagnostic = observation.get("dark_foreground_diagnostic")
        if (
            observation.get("classification") != "resolved"
            or observation.get("reason_code")
            != "multiview_dark_consensus_resolved"
            or observation.get("canonical_group_id") != canonical_group_id
            or observation.get("bbox_canonical_group_id") != canonical_group_id
            or observation.get("registration_label_stable") is not True
            or observation.get("perturbation_label_stable") is not True
            or observation.get("multiview_dark_consensus") != consensus
            or not isinstance(winner, dict)
            or winner.get("canonical_group_id") != canonical_group_id
            or str(winner.get("base_color", "")).strip().casefold() != "black"
            or _quality_unit(
                winner.get("color_share"),
                "Multiview dark direct color share",
            )
            < minimum_color_share
            or not isinstance(diagnostic, dict)
        ):
            raise RuntimeError(
                f"Multiview dark support contract is invalid for {part_id}/{view_id}"
            )
        diagnostic_hash = diagnostic.get("diagnostic_sha256")
        unsigned_diagnostic = dict(diagnostic)
        unsigned_diagnostic.pop("diagnostic_sha256", None)
        projected_pixels = diagnostic.get("projected_part_pixels")
        normalized_pixels = diagnostic.get("normalized_projected_pixels")
        non_background_share = _quality_unit(
            diagnostic.get("non_background_share"),
            "Multiview dark non-background share",
        )
        dark_signal_share = _quality_unit(
            diagnostic.get("dark_signal_share"),
            "Multiview dark signal share",
        )
        dark_signal_purity = _quality_unit(
            diagnostic.get("dark_signal_purity"),
            "Multiview dark signal purity",
        )
        core_dark_share = _quality_unit(
            diagnostic.get("core_dark_signal_share"),
            "Multiview dark core share",
        )
        null_margin = _quality_finite_number(
            diagnostic.get("dark_signal_null_margin"),
            "Multiview dark null margin",
        )
        if (
            not _quality_sha256(diagnostic_hash)
            or canonical_sha256(unsigned_diagnostic) != diagnostic_hash
            or diagnostic.get("canonical_group_id") != canonical_group_id
            or isinstance(projected_pixels, bool)
            or not isinstance(projected_pixels, int)
            or projected_pixels < 1
            or projected_pixels != observation.get("projected_part_pixels")
            or isinstance(normalized_pixels, bool)
            or not isinstance(normalized_pixels, int)
            or normalized_pixels
            < int(
                QUALITY_REPAIR_DARK_FOREGROUND_THRESHOLDS[
                    "minimum_normalized_projected_pixels"
                ]
            )
            or non_background_share < 0.20
            or dark_signal_share
            < float(
                QUALITY_REPAIR_DARK_FOREGROUND_THRESHOLDS[
                    "minimum_dark_signal_share"
                ]
            )
            or dark_signal_purity
            < float(
                QUALITY_REPAIR_DARK_FOREGROUND_THRESHOLDS[
                    "minimum_dark_signal_purity"
                ]
            )
            or core_dark_share
            < float(
                QUALITY_REPAIR_DARK_FOREGROUND_THRESHOLDS[
                    "minimum_core_dark_signal_share"
                ]
            )
            or null_margin
            < float(
                QUALITY_REPAIR_DARK_FOREGROUND_THRESHOLDS[
                    "minimum_null_q75_margin"
                ]
            )
        ):
            raise RuntimeError(
                f"Multiview dark diagnostic is invalid for {part_id}/{view_id}"
            )
        evidence_rows.append(
            {
                "reference_view_id": view_id,
                "diagnostic_sha256": diagnostic_hash,
                "projected_part_pixels": projected_pixels,
                "normalized_projected_pixels": normalized_pixels,
                "dark_signal_share": dark_signal_share,
                "dark_signal_purity": dark_signal_purity,
                "core_dark_signal_share": core_dark_share,
                "dark_signal_null_margin": null_margin,
                "direct_black_share": float(winner["color_share"]),
            }
        )

    if any(
        observation.get("classification") == "resolved"
        and isinstance(observation.get("canonical_group_id"), str)
        and observation.get("canonical_group_id") != canonical_group_id
        for observation in raw_observations
        if isinstance(observation, dict)
    ):
        raise RuntimeError("Multiview dark identity has a resolved alternative")
    for decision in (spatial_gate_decision, mapping_decision):
        if (
            isinstance(decision, dict)
            and decision.get("output_status") == "matched"
            and decision.get("output_group_id") != canonical_group_id
        ):
            raise RuntimeError("Multiview dark identity conflicts with mapping")

    semantic_conflict_view_ids = sorted(
        {
            str(vote["view_id"])
            for vote in spatial_part.get("semantic_votes", [])
            if isinstance(vote, dict)
            and isinstance(vote.get("view_id"), str)
            and vote.get("alignment_trusted") is True
            and vote.get("unique_canonical_join") is True
            and vote.get("pixel_gate_accepted") is True
            and vote.get("status") == "matched"
            and isinstance(vote.get("canonical_group_id"), str)
            and vote.get("canonical_group_id") != canonical_group_id
            and isinstance(vote.get("effective_confidence"), (int, float))
            and not isinstance(vote.get("effective_confidence"), bool)
            and math.isfinite(float(vote["effective_confidence"]))
            and float(vote["effective_confidence"])
            >= minimum_semantic_confidence
        }
    )
    if len(semantic_conflict_view_ids) > 1:
        raise RuntimeError(
            "Multiview dark identity has multiple semantic conflicts"
        )
    reference_pixels = _require_exact_int(
        dark_residual_support.get("normalized_reference_pixels"),
        "Multiview dark normalized reference pixels",
        minimum=1,
    )
    render_pixels = _require_exact_int(
        dark_residual_support.get("render_foreground_pixels"),
        "Multiview dark render foreground pixels",
        minimum=1,
    )
    contribution = int(
        math.ceil(
            min(
                float(row["projected_part_pixels"])
                * float(row["dark_signal_share"])
                for row in evidence_rows
            )
            / render_pixels
            * reference_pixels
        )
    )
    evidence_strength = min(
        float(row["direct_black_share"])
        + float(row["dark_signal_share"])
        + float(row["dark_signal_purity"])
        + float(row["dark_signal_null_margin"])
        for row in evidence_rows
    )
    audit_body = {
        "lane": QUALITY_REPAIR_MULTIVIEW_DARK_IDENTITY_LANE,
        "canonical_group_id": canonical_group_id,
        "supporting_view_ids": support_views,
        "evidence_contract": consensus["evidence_contract"],
        "evidence_rows": evidence_rows,
    }
    return (
        {
            **audit_body,
            "diagnostic_sha256": canonical_sha256(audit_body),
            "estimated_contribution_pixels": contribution,
            "evidence_strength": evidence_strength,
        },
        support_views,
        semantic_conflict_view_ids,
    )


def _quality_dark_support_from_report(
    *,
    quality_report: dict[str, Any],
    view_id: str,
    canonical_group_id: str,
    spatial_reference: dict[str, Any],
) -> dict[str, Any]:
    aggregate = quality_report.get("aggregate")
    raw_views = quality_report.get("views")
    if (
        not isinstance(aggregate, dict)
        or aggregate.get("status") != "FAIL"
        or not isinstance(raw_views, list)
    ):
        raise RuntimeError("Dark-foreground residual lacks a failed QA report")
    matches = [
        item
        for item in raw_views
        if isinstance(item, dict) and item.get("reference_view_id") == view_id
    ]
    if len(matches) != 1:
        raise RuntimeError("Dark-foreground residual QA view is not unique")
    view = matches[0]
    render_view_id = view.get("render_view_id")
    mapping = view.get("mapping")
    reference = view.get("reference")
    render = view.get("render")
    alignment = view.get("alignment")
    material_color = view.get("material_color")
    if (
        view.get("status") not in {"FAIL", "REVIEW"}
        or not isinstance(render_view_id, str)
        or not render_view_id
        or render_view_id != spatial_reference.get("pose_cluster_id")
        or not isinstance(mapping, dict)
        or mapping.get("reasons") != []
        or mapping.get("selected_render_view_id") != render_view_id
        or not isinstance(reference, dict)
        or reference.get("image_sha256") != spatial_reference.get("raw_sha256")
        or not isinstance(reference.get("trusted_evidence"), dict)
        or reference["trusted_evidence"].get("usable") is not True
        or not isinstance(render, dict)
        or not isinstance(alignment, dict)
        or not isinstance(material_color, dict)
    ):
        raise RuntimeError("Dark-foreground residual QA view is not trusted")

    reference_foreground = reference.get("foreground")
    render_foreground = render.get("foreground")
    reference_pixels = _require_exact_int(
        (
            reference_foreground.get("pixel_count")
            if isinstance(reference_foreground, dict)
            else None
        ),
        "Dark-foreground reference foreground pixels",
        minimum=1,
    )
    render_pixels = _require_exact_int(
        (
            render_foreground.get("pixel_count")
            if isinstance(render_foreground, dict)
            else None
        ),
        "Dark-foreground render foreground pixels",
        minimum=1,
    )
    reference_distribution = material_color.get("reference_distribution")
    render_distribution = material_color.get("render_distribution")
    if not isinstance(reference_distribution, dict) or not isinstance(
        render_distribution, dict
    ):
        raise RuntimeError("Dark-foreground residual color distributions are missing")
    if (
        reference_distribution.get("sample_step") != 1
        or render_distribution.get("sample_step") != 1
        or reference_distribution.get("sampled_pixels") != reference_pixels
        or render_distribution.get("sampled_pixels") != render_pixels
    ):
        raise RuntimeError("Dark-foreground residual is not exact-pixel QA evidence")
    reference_categories = reference_distribution.get("category_distribution")
    render_categories = render_distribution.get("category_distribution")
    if not isinstance(reference_categories, dict) or not isinstance(
        render_categories, dict
    ):
        raise RuntimeError("Dark-foreground residual color categories are missing")
    try:
        reference_share = sum(
            _quality_unit(
                reference_categories[label],
                f"Dark-foreground reference {label}",
            )
            for label in ("black", "achromatic_dark")
        )
        render_share = sum(
            _quality_unit(
                render_categories[label],
                f"Dark-foreground render {label}",
            )
            for label in ("black", "achromatic_dark")
        )
    except KeyError as exc:
        raise RuntimeError(
            "Dark-foreground residual black-family categories are missing"
        ) from exc
    if reference_share > 1.0 + 1e-9 or render_share > 1.0 + 1e-9:
        raise RuntimeError("Dark-foreground residual black-family share exceeds one")
    deficit_share = max(0.0, reference_share - render_share)
    mass_recall = (
        min(1.0, render_share / reference_share)
        if reference_share > 0.0
        else 1.0
    )
    if deficit_share < 0.025 or mass_recall >= 0.80:
        raise RuntimeError("Dark-foreground baseline residual is below its QA floor")

    alignment_fields = {
        "score": 0.85,
        "silhouette_iou": 0.85,
        "edge_f1_tolerance_3px": 0.85,
        "profile_similarity": 0.90,
        "bbox_aspect_similarity": 0.90,
    }
    alignment_support: dict[str, float] = {}
    for field, minimum in alignment_fields.items():
        value = _quality_unit(
            alignment.get(field),
            f"Dark-foreground QA alignment {field}",
        )
        if value < minimum:
            raise RuntimeError("Dark-foreground QA alignment is below its floor")
        alignment_support[field] = value
    budget_pixels = int(math.ceil(deficit_share * reference_pixels))
    return {
        "reference_view_id": view_id,
        "local_group_id": f"__canonical_dark__:{canonical_group_id}",
        "reference_sha256": reference["image_sha256"],
        "content_cluster_id": spatial_reference["content_cluster_id"],
        "pose_cluster_id": spatial_reference["pose_cluster_id"],
        "recall": mass_recall,
        "mass_recall": mass_recall,
        "deficit_sources": [
            QUALITY_REPAIR_DARK_FOREGROUND_RESIDUAL_DEFICIT_SOURCE
        ],
        "reference_share": reference_share,
        "observed_render_share": render_share,
        "deficit_share": deficit_share,
        "normalized_reference_pixels": reference_pixels,
        "render_foreground_pixels": render_pixels,
        "budget_pixels": budget_pixels,
        "budget_limit_pixels": int(
            math.floor(
                QUALITY_REPAIR_DARK_FOREGROUND_MAX_TOTAL_CONTRIBUTION_FACTOR
                * budget_pixels
            )
        ),
        "alignment": dict(sorted(alignment_support.items())),
    }


def _quality_repeated_geometry_signature(part: dict[str, Any]) -> dict[str, Any]:
    point_count = _require_exact_int(
        part.get("point_count"), "Repeated-dark geometry point count", minimum=1
    )
    face_count = _require_exact_int(
        part.get("face_count"), "Repeated-dark geometry face count", minimum=1
    )
    raw_bbox = part.get("world_bbox")
    if (
        not isinstance(raw_bbox, list)
        or len(raw_bbox) != 2
        or any(not isinstance(bound, list) or len(bound) != 3 for bound in raw_bbox)
    ):
        raise RuntimeError("Repeated-dark geometry world bbox is invalid")
    bounds: list[list[float]] = []
    for raw_bound in raw_bbox:
        bound = [
            _quality_finite_number(
                value,
                "Repeated-dark geometry bbox coordinate",
            )
            for value in raw_bound
        ]
        bounds.append(bound)
    if any(bounds[1][axis] < bounds[0][axis] for axis in range(3)):
        raise RuntimeError("Repeated-dark geometry world bbox is inverted")
    extents = sorted(
        round(abs(bounds[1][axis] - bounds[0][axis]), 9) for axis in range(3)
    )
    if any(extent <= 0.0 for extent in extents):
        raise RuntimeError("Repeated-dark geometry has a degenerate extent")
    return {
        "point_count": point_count,
        "face_count": face_count,
        "sorted_bbox_extents": extents,
    }


def _quality_repeated_source_signature(part: dict[str, Any]) -> str:
    raw_properties = part.get("existing_visual_material_properties")
    if not isinstance(raw_properties, dict):
        raise RuntimeError("Repeated-dark source visual properties are invalid")
    stable_properties = {
        key: value for key, value in raw_properties.items() if key != "shader_path"
    }
    if not stable_properties:
        raise RuntimeError("Repeated-dark source visual signature is empty")
    return canonical_sha256(stable_properties)


def _quality_group_measurement(
    quality_report: dict[str, Any],
    *,
    view_id: str,
    local_group_id: str,
) -> dict[str, Any]:
    raw_views = quality_report.get("views")
    if not isinstance(raw_views, list):
        raise RuntimeError("Repeated-dark cohort lacks quality views")
    matches = [
        item
        for item in raw_views
        if isinstance(item, dict) and item.get("reference_view_id") == view_id
    ]
    if len(matches) != 1:
        raise RuntimeError("Repeated-dark quality view is not unique")
    view = matches[0]
    material_color = view.get("material_color")
    render = view.get("render")
    recall_audit = (
        material_color.get("trusted_evidence_group_recall")
        if isinstance(material_color, dict)
        else None
    )
    groups = recall_audit.get("groups") if isinstance(recall_audit, dict) else None
    render_foreground = render.get("foreground") if isinstance(render, dict) else None
    if not isinstance(groups, list) or not isinstance(render_foreground, dict):
        raise RuntimeError("Repeated-dark quality group evidence is incomplete")
    group_matches = [
        item
        for item in groups
        if isinstance(item, dict) and item.get("group_id") == local_group_id
    ]
    if len(group_matches) != 1:
        raise RuntimeError("Repeated-dark quality group is not unique")
    group = group_matches[0]
    base_colors = group.get("base_colors")
    render_bins = group.get("render_color_bins")
    required_share = _quality_unit(
        group.get("required_render_share"),
        "Repeated-dark required render share",
    )
    observed_share = _quality_unit(
        group.get("observed_render_share"),
        "Repeated-dark observed render share",
    )
    recall = _quality_unit(group.get("recall"), "Repeated-dark group recall")
    expected_recall = (
        min(1.0, observed_share / required_share) if required_share > 0.0 else 1.0
    )
    if (
        not isinstance(base_colors, list)
        or {str(value).strip().casefold() for value in base_colors} != {"black"}
        or not isinstance(render_bins, list)
        or set(render_bins) != {"black", "achromatic_dark"}
        or required_share <= observed_share
        or recall >= 0.5
        or not math.isclose(recall, expected_recall, rel_tol=0.0, abs_tol=1e-9)
    ):
        raise RuntimeError("Repeated-dark quality group is not a measured deficit")
    return {
        "required_render_share": required_share,
        "observed_render_share": observed_share,
        "recall": recall,
        "render_foreground_pixels": _require_exact_int(
            render_foreground.get("pixel_count"),
            "Repeated-dark render foreground pixels",
            minimum=1,
        ),
    }


def _quality_semantic_disproof_shares(
    *,
    spatial_part: dict[str, Any],
    view_id: str,
    rejected_group_id: str,
) -> dict[str, float]:
    observation = _quality_dark_spatial_observation(spatial_part, view_id)
    diagnostic = observation.get("canonical_palette_diagnostic")
    if not isinstance(diagnostic, dict):
        raise RuntimeError("Repeated-dark semantic alternative lacks pixel diagnostics")
    raw_perturbations = diagnostic.get("projection_perturbations")
    if not isinstance(raw_perturbations, list) or len(raw_perturbations) != 4:
        raise RuntimeError("Repeated-dark semantic disproof perturbations are incomplete")
    perturbations: dict[tuple[int, int], dict[str, Any]] = {}
    for raw_perturbation in raw_perturbations:
        if not isinstance(raw_perturbation, dict):
            raise RuntimeError("Repeated-dark semantic disproof perturbation is invalid")
        offset = raw_perturbation.get("offset_pixels")
        if (
            not isinstance(offset, list)
            or len(offset) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in offset
            )
            or tuple(offset) in perturbations
        ):
            raise RuntimeError("Repeated-dark semantic disproof offset is invalid")
        perturbations[tuple(offset)] = raw_perturbation
    required_offsets = {(-2, 0), (2, 0), (0, -2), (0, 2)}
    if set(perturbations) != required_offsets:
        raise RuntimeError("Repeated-dark semantic disproof offsets are incomplete")
    samples = {
        "direct": diagnostic.get("direct_sample"),
        "bbox": diagnostic.get("bbox_sample"),
        "offset_-2_0": perturbations[(-2, 0)],
        "offset_2_0": perturbations[(2, 0)],
        "offset_0_-2": perturbations[(0, -2)],
        "offset_0_2": perturbations[(0, 2)],
    }
    if any(not isinstance(sample, dict) for sample in samples.values()):
        raise RuntimeError("Repeated-dark semantic disproof samples are incomplete")
    return {
        label: _quality_group_share(
            sample.get("group_scores"),
            canonical_group_id=rejected_group_id,
            label=f"Repeated-dark semantic disproof {label}",
        )
        for label, sample in samples.items()
    }


def _quality_target_matching_pixels(
    raw_scores: Any,
    *,
    canonical_group_id: str,
) -> int:
    if not isinstance(raw_scores, list) or not raw_scores:
        raise RuntimeError("Repeated-dark direct group scores are invalid")
    total = 0
    matched = False
    for raw_score in raw_scores:
        if not isinstance(raw_score, dict):
            raise RuntimeError("Repeated-dark direct group score is invalid")
        if raw_score.get("canonical_group_id") != canonical_group_id:
            continue
        total += _require_exact_int(
            raw_score.get("matching_pixels"),
            "Repeated-dark direct matching pixels",
        )
        matched = True
    if not matched:
        raise RuntimeError("Repeated-dark direct target group is absent")
    return total


def _validate_repeated_geometry_dark_cohorts(
    *,
    audit: dict[str, Any],
    quality_report: dict[str, Any],
    spatial_report: dict[str, Any],
    spatial_gate_audit: dict[str, Any],
    mapping_consensus: dict[str, Any],
    geometry_risk: dict[str, Any],
    registry_parts: list[Any],
    canonical_groups: dict[str, dict[str, Any]],
    group_diagnostics: dict[str, dict[str, Any]],
    changes_by_part: dict[str, dict[str, Any]],
    localization_lanes_by_part: dict[str, dict[str, str]],
) -> dict[str, dict[str, Any]]:
    """Validate selected repeated-geometry dark cohorts from primary evidence."""

    thresholds = QUALITY_REPAIR_REPEATED_GEOMETRY_DARK_THRESHOLDS
    cohort_change_ids = {
        part_id
        for part_id, lane in localization_lanes_by_part.items()
        if lane["lane"] == QUALITY_REPAIR_REPEATED_GEOMETRY_DARK_RESIDUAL_LANE
    }
    raw_cohorts = audit.get("repeated_geometry_dark_cohorts")
    if not cohort_change_ids:
        if raw_cohorts not in (None, []):
            raise RuntimeError("Repeated-dark cohort audit has no authorized changes")
        return {}
    if not isinstance(raw_cohorts, list) or len(raw_cohorts) != 1:
        raise RuntimeError("Repeated-dark repair lacks an atomic cohort audit")

    registry_by_id: dict[str, dict[str, Any]] = {}
    for raw_part in registry_parts:
        if not isinstance(raw_part, dict):
            raise RuntimeError("Repeated-dark registry part is invalid")
        part_id = raw_part.get("part_id")
        if (
            not isinstance(part_id, str)
            or not part_id
            or part_id in registry_by_id
        ):
            raise RuntimeError("Repeated-dark registry identity is invalid")
        registry_by_id[part_id] = raw_part
    registry_count = len(registry_by_id)

    raw_risk_parts = geometry_risk.get("parts")
    if not isinstance(raw_risk_parts, list):
        raise RuntimeError("Repeated-dark repair lacks geometry-risk parts")
    risk_by_id: dict[str, bool] = {}
    for raw_risk in raw_risk_parts:
        if not isinstance(raw_risk, dict):
            raise RuntimeError("Repeated-dark geometry-risk part is invalid")
        part_id = raw_risk.get("part_id")
        risk = raw_risk.get("risk")
        if (
            not isinstance(part_id, str)
            or part_id in risk_by_id
            or not isinstance(risk, dict)
            or not isinstance(risk.get("multi_material_risk"), bool)
        ):
            raise RuntimeError("Repeated-dark geometry-risk evidence is invalid")
        risk_by_id[part_id] = bool(risk["multi_material_risk"])
    if set(risk_by_id) != set(registry_by_id):
        raise RuntimeError("Repeated-dark geometry-risk coverage is incomplete")

    references = _quality_dark_reference_evidence(spatial_report)
    spatial_policy = spatial_report.get("policy")
    if not isinstance(spatial_policy, dict):
        raise RuntimeError("Repeated-dark repair lacks a spatial policy")
    _quality_unit(
        spatial_policy.get("minimum_semantic_confidence"),
        "Repeated-dark semantic confidence floor",
    )

    eligible_black_groups = sorted(
        group_id
        for group_id, group in canonical_groups.items()
        if str(group.get("base_color", "")).strip().casefold() == "black"
        and group.get("singleton") is not True
        and isinstance(group.get("distinct_view_count"), int)
        and not isinstance(group.get("distinct_view_count"), bool)
        and int(group["distinct_view_count"]) >= 2
    )
    if len(eligible_black_groups) != 1:
        raise RuntimeError("Repeated-dark repair lacks one unique multiview black group")

    validated_parts: dict[str, dict[str, Any]] = {}
    seen_cohort_ids: set[str] = set()
    matched_alternative_groups: set[str] = set()
    matched_alternative_views: set[str] = set()
    expected_cohort_change_ids: set[str] = set()
    for raw_cohort in raw_cohorts:
        if not isinstance(raw_cohort, dict):
            raise RuntimeError("Repeated-dark cohort record is invalid")
        expected_keys = {
            "cohort_id",
            "canonical_group_id",
            "reference_view_id",
            "geometry_signature",
            "source_visual_stable_properties_signature_sha256",
            "registry_part_count",
            "cohort_size",
            "registry_fraction",
            "cohort_part_ids",
            "required_render_share",
            "observed_render_share",
            "render_foreground_pixels",
            "budget_pixels",
            "minimum_contribution_pixels",
            "maximum_contribution_pixels",
            "total_projected_part_pixels",
            "total_direct_target_matching_pixels",
            "selected",
            "reason_codes",
            "members",
        }
        if set(raw_cohort) != expected_keys:
            raise RuntimeError("Repeated-dark cohort schema is invalid")
        cohort_id = raw_cohort.get("cohort_id")
        group_id = raw_cohort.get("canonical_group_id")
        view_id = raw_cohort.get("reference_view_id")
        part_ids = raw_cohort.get("cohort_part_ids")
        if (
            not _quality_sha256(cohort_id)
            or cohort_id in seen_cohort_ids
            or group_id != eligible_black_groups[0]
            or not isinstance(view_id, str)
            or view_id not in references
            or not isinstance(part_ids, list)
            or part_ids != sorted(set(part_ids))
            or len(part_ids)
            < int(thresholds["minimum_cohort_size"])
            or len(part_ids)
            > int(thresholds["maximum_cohort_size"])
            or not set(part_ids) <= set(registry_by_id)
            or raw_cohort.get("selected") is not True
            or raw_cohort.get("reason_codes") != []
        ):
            raise RuntimeError("Repeated-dark selected cohort identity is invalid")
        seen_cohort_ids.add(str(cohort_id))
        if (
            raw_cohort.get("registry_part_count") != registry_count
            or raw_cohort.get("cohort_size") != len(part_ids)
            or not math.isclose(
                _quality_unit(
                    raw_cohort.get("registry_fraction"),
                    "Repeated-dark registry fraction",
                ),
                len(part_ids) / registry_count,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or len(part_ids) / registry_count
            > float(thresholds["maximum_registry_fraction"]) + 1e-12
        ):
            raise RuntimeError("Repeated-dark cohort rarity is invalid")

        raw_geometry = raw_cohort.get("geometry_signature")
        if not isinstance(raw_geometry, dict) or set(raw_geometry) != {
            "point_count",
            "face_count",
            "sorted_bbox_extents",
            "signature_sha256",
        }:
            raise RuntimeError("Repeated-dark geometry signature is invalid")
        geometry_payload = {
            "point_count": raw_geometry.get("point_count"),
            "face_count": raw_geometry.get("face_count"),
            "sorted_bbox_extents": raw_geometry.get("sorted_bbox_extents"),
        }
        geometry_sha = raw_geometry.get("signature_sha256")
        source_sha = raw_cohort.get(
            "source_visual_stable_properties_signature_sha256"
        )
        if (
            not _quality_sha256(geometry_sha)
            or geometry_sha != canonical_sha256(geometry_payload)
            or not _quality_sha256(source_sha)
        ):
            raise RuntimeError("Repeated-dark cohort signatures are invalid")
        complete_cohort_ids = sorted(
            part_id
            for part_id, registry_part in registry_by_id.items()
            if _quality_repeated_geometry_signature(registry_part) == geometry_payload
        )
        if (
            complete_cohort_ids != part_ids
            or any(
                _quality_repeated_source_signature(registry_by_id[part_id])
                != source_sha
                for part_id in part_ids
            )
            or any(risk_by_id[part_id] for part_id in part_ids)
        ):
            raise RuntimeError(
                "Repeated-dark cohort is not a complete safe geometry cohort"
            )
        expected_cohort_id = canonical_sha256(
            {
                "canonical_group_id": group_id,
                "reference_view_id": view_id,
                "geometry_signature_sha256": geometry_sha,
                "source_visual_stable_properties_signature_sha256": source_sha,
                "cohort_part_ids": part_ids,
            }
        )
        if cohort_id != expected_cohort_id:
            raise RuntimeError("Repeated-dark cohort ID is not reproducible")

        group_diagnostic = group_diagnostics.get(str(group_id))
        raw_supports = (
            group_diagnostic.get("supporting_views")
            if isinstance(group_diagnostic, dict)
            else None
        )
        support_matches = (
            [
                support
                for support in raw_supports
                if isinstance(support, dict)
                and support.get("reference_view_id") == view_id
            ]
            if isinstance(raw_supports, list)
            else []
        )
        if (
            not isinstance(group_diagnostic, dict)
            or group_diagnostic.get("single_view_spatial_repairable") is not True
            or len(support_matches) != 1
            or support_matches[0].get("deficit_sources") != ["group_recall"]
            or support_matches[0].get("reference_sha256")
            != references[view_id].get("raw_sha256")
        ):
            raise RuntimeError("Repeated-dark cohort lacks a single-view QA deficit")
        local_group_id = support_matches[0].get("local_group_id")
        if not isinstance(local_group_id, str) or not local_group_id:
            raise RuntimeError("Repeated-dark cohort QA group identity is invalid")
        measurement = _quality_group_measurement(
            quality_report,
            view_id=view_id,
            local_group_id=local_group_id,
        )
        quality_threshold, quality_recalls, _ = _quality_group_recalls(quality_report)
        if (
            quality_recalls.get(view_id, {}).get(local_group_id)
            != measurement["recall"]
            or measurement["recall"] >= quality_threshold
        ):
            raise RuntimeError("Repeated-dark cohort target is not a QA group deficit")
        budget_pixels = int(
            math.ceil(
                (
                    measurement["required_render_share"]
                    - measurement["observed_render_share"]
                )
                * measurement["render_foreground_pixels"]
            )
        )
        minimum_contribution = int(
            math.ceil(
                float(thresholds["minimum_budget_contribution_factor"])
                * budget_pixels
            )
        )
        maximum_contribution = int(
            math.floor(
                float(thresholds["maximum_budget_contribution_factor"])
                * budget_pixels
            )
        )
        if (
            not math.isclose(
                _quality_unit(
                    raw_cohort.get("required_render_share"),
                    "Repeated-dark audited required share",
                ),
                measurement["required_render_share"],
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            or not math.isclose(
                _quality_unit(
                    raw_cohort.get("observed_render_share"),
                    "Repeated-dark audited observed share",
                ),
                measurement["observed_render_share"],
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            or raw_cohort.get("render_foreground_pixels")
            != measurement["render_foreground_pixels"]
            or raw_cohort.get("budget_pixels") != budget_pixels
            or raw_cohort.get("minimum_contribution_pixels")
            != minimum_contribution
            or raw_cohort.get("maximum_contribution_pixels")
            != maximum_contribution
        ):
            raise RuntimeError("Repeated-dark cohort budget basis is inconsistent")

        raw_members = raw_cohort.get("members")
        if (
            not isinstance(raw_members, list)
            or [member.get("part_id") for member in raw_members if isinstance(member, dict)]
            != part_ids
            or len(raw_members) != len(part_ids)
        ):
            raise RuntimeError("Repeated-dark cohort members are not deterministic")
        total_projected = 0
        total_direct_matching = 0
        for raw_member, part_id in zip(raw_members, part_ids, strict=True):
            if not isinstance(raw_member, dict):
                raise RuntimeError("Repeated-dark cohort member is invalid")
            spatial_part = _quality_dark_spatial_part(spatial_report, part_id)
            observation = _quality_dark_spatial_observation(spatial_part, view_id)
            alignment = _quality_dark_alignment(spatial_report, view_id)
            direct_scores = observation.get("group_scores")
            bbox_scores = observation.get("bbox_group_scores")
            direct_share = _quality_group_share(
                direct_scores,
                canonical_group_id=str(group_id),
                label="Repeated-dark direct target",
            )
            bbox_share = _quality_group_share(
                bbox_scores,
                canonical_group_id=str(group_id),
                label="Repeated-dark bbox target",
            )
            direct_margin = _quality_unit(
                observation.get("color_margin"), "Repeated-dark direct margin"
            )
            bbox_margin = _quality_unit(
                observation.get("bbox_color_margin"), "Repeated-dark bbox margin"
            )
            projected_pixels = _require_exact_int(
                observation.get("projected_part_pixels"),
                "Repeated-dark projected pixels",
                minimum=1,
            )
            direct_matching = _quality_target_matching_pixels(
                direct_scores,
                canonical_group_id=str(group_id),
            )
            if (
                observation.get("registration_label_stable") is not True
                or observation.get("perturbation_label_stable") is not True
                or observation.get("canonical_group_id") != group_id
                or projected_pixels < int(thresholds["minimum_projected_pixels"])
                or not isinstance(direct_scores, list)
                or not isinstance(direct_scores[0], dict)
                or direct_scores[0].get("canonical_group_id") != group_id
                or direct_share < float(thresholds["minimum_direct_target_share"])
                or direct_margin < float(thresholds["minimum_direct_target_margin"])
                or observation.get("bbox_canonical_group_id") != group_id
                or bbox_share < float(thresholds["minimum_bbox_target_share"])
                or bbox_margin < float(thresholds["minimum_bbox_target_margin"])
            ):
                raise RuntimeError("Repeated-dark target projection is not strong")

            required_offsets = {(-2, 0), (2, 0), (0, -2), (0, 2)}
            raw_target_perturbations = observation.get("projection_perturbations")
            if (
                not isinstance(raw_target_perturbations, list)
                or len(raw_target_perturbations) != 4
            ):
                raise RuntimeError("Repeated-dark target perturbations are incomplete")
            perturbations_by_offset: dict[tuple[int, int], dict[str, Any]] = {}
            expected_perturbations: list[dict[str, Any]] = []
            for raw_perturbation in raw_target_perturbations:
                if not isinstance(raw_perturbation, dict):
                    raise RuntimeError("Repeated-dark target perturbation is invalid")
                offset = raw_perturbation.get("offset_pixels")
                if (
                    not isinstance(offset, list)
                    or len(offset) != 2
                    or any(
                        isinstance(value, bool) or not isinstance(value, int)
                        for value in offset
                    )
                    or tuple(offset) not in required_offsets
                    or tuple(offset) in perturbations_by_offset
                    or raw_perturbation.get("canonical_group_id") != group_id
                    or raw_perturbation.get("diagnostic_canonical_group_id") != group_id
                ):
                    raise RuntimeError("Repeated-dark target perturbation is invalid")
                sampled_pixels = _require_exact_int(
                    raw_perturbation.get("sampled_reference_pixels"),
                    "Repeated-dark target perturbation pixels",
                )
                target_share = _quality_unit(
                    raw_perturbation.get("best_color_share"),
                    "Repeated-dark target perturbation share",
                )
                target_margin = _quality_unit(
                    raw_perturbation.get("color_margin"),
                    "Repeated-dark target perturbation margin",
                )
                if (
                    sampled_pixels < int(thresholds["minimum_projected_pixels"])
                    or target_share
                    < float(thresholds["minimum_perturbation_target_share"])
                    or target_margin
                    < float(thresholds["minimum_perturbation_target_margin"])
                ):
                    raise RuntimeError(
                        "Repeated-dark target perturbation is below its floor"
                    )
                perturbations_by_offset[tuple(offset)] = raw_perturbation
            if set(perturbations_by_offset) != required_offsets:
                raise RuntimeError("Repeated-dark target perturbation coverage is invalid")
            for offset in sorted(required_offsets):
                raw_perturbation = perturbations_by_offset[offset]
                expected_perturbations.append(
                    {
                        "offset_pixels": list(offset),
                        "sampled_reference_pixels": raw_perturbation[
                            "sampled_reference_pixels"
                        ],
                        "target_share": raw_perturbation["best_color_share"],
                        "target_margin": raw_perturbation["color_margin"],
                    }
                )

            alignment_summary = {
                "score": alignment.get("score"),
                "projection_score": alignment.get("projection_score"),
                "projection_iou": alignment.get("projection_iou"),
                "ecc_correlation": alignment.get("ecc_correlation"),
                "ecc_status": alignment.get("ecc_status"),
            }
            transform_audit = alignment.get("ecc_transform_audit")
            if (
                alignment.get("trusted") is not True
                or alignment.get("reason_codes") != []
                or alignment_summary["ecc_status"] != "success"
                or not isinstance(transform_audit, dict)
                or transform_audit.get("constraints_passed") is not True
                or _quality_unit(
                    alignment_summary["score"],
                    "Repeated-dark alignment score",
                )
                < float(thresholds["minimum_alignment_score"])
                or _quality_unit(
                    alignment_summary["projection_score"],
                    "Repeated-dark projection score",
                )
                < float(thresholds["minimum_projection_score"])
                or _quality_unit(
                    alignment_summary["projection_iou"],
                    "Repeated-dark projection IoU",
                )
                < float(thresholds["minimum_projection_iou"])
                or _quality_unit(
                    alignment_summary["ecc_correlation"],
                    "Repeated-dark ECC correlation",
                )
                < float(thresholds["minimum_ecc_correlation"])
            ):
                raise RuntimeError("Repeated-dark alignment is below its floor")

            strict_member = {
                "part_id": part_id,
                "projected_part_pixels": projected_pixels,
                "direct_target_share": direct_share,
                "direct_target_margin": direct_margin,
                "direct_target_matching_pixels": direct_matching,
                "bbox_target_share": bbox_share,
                "bbox_target_margin": bbox_margin,
                "perturbations": expected_perturbations,
                "alignment": {
                    "score": alignment["score"],
                    "projection_iou": alignment["projection_iou"],
                    "ecc_correlation": alignment["ecc_correlation"],
                    "ecc_status": alignment["ecc_status"],
                },
                "evidence_contract": "strict_reference_space_projection",
                "semantic_alternative_disproofs": [],
            }
            strict_member["evidence_sha256"] = canonical_sha256(strict_member)
            if raw_member.get("evidence_contract") == (
                "strict_reference_space_projection"
            ):
                if raw_member != strict_member:
                    raise RuntimeError(
                        "Repeated-dark strict projection audit is inconsistent"
                    )
                total_projected += projected_pixels
                total_direct_matching += direct_matching
                expected_cohort_change_ids.add(part_id)
                validated_parts[part_id] = {
                    "cohort_id": cohort_id,
                    "cohort_part_ids": part_ids,
                    "provenance": {
                        "lane": QUALITY_REPAIR_REPEATED_GEOMETRY_DARK_RESIDUAL_LANE,
                        "cohort_id": cohort_id,
                        "canonical_group_id": group_id,
                        "reference_view_id": view_id,
                        "geometry_signature_sha256": geometry_sha,
                        "source_visual_stable_properties_signature_sha256": source_sha,
                        "cohort_part_ids": part_ids,
                        "budget_pixels": budget_pixels,
                        "minimum_contribution_pixels": minimum_contribution,
                        "maximum_contribution_pixels": maximum_contribution,
                        "total_projected_part_pixels": total_projected,
                        "total_direct_target_matching_pixels": total_direct_matching,
                        "member_evidence_contract": strict_member[
                            "evidence_contract"
                        ],
                        "member_evidence_sha256": strict_member["evidence_sha256"],
                    },
                }
                continue
            if raw_member.get("evidence_contract") != "dark_foreground_diagnostic":
                raise RuntimeError(
                    "Repeated-dark member evidence contract is unsupported"
                )

            raw_dark = observation.get("dark_foreground_diagnostic")
            if not isinstance(raw_dark, dict):
                raise RuntimeError("Repeated-dark foreground diagnostic is missing")
            canonical_source_view_ids = canonical_groups[str(group_id)].get(
                "source_view_ids"
            )
            if (
                not isinstance(canonical_source_view_ids, list)
                or raw_dark.get("canonical_source_view_ids")
                != sorted(set(canonical_source_view_ids))
            ):
                raise RuntimeError(
                    "Repeated-dark canonical source views are inconsistent"
                )
            dark_summary = _validate_quality_dark_foreground_diagnostic(
                diagnostic=raw_dark,
                observation=observation,
                alignment=alignment,
                references=references,
                target_view_id=view_id,
                canonical_group_id=str(group_id),
                allow_moderate_cohort_alignment=True,
            )

            raw_votes = spatial_part.get("semantic_votes")
            if not isinstance(raw_votes, list):
                raise RuntimeError("Repeated-dark semantic votes are invalid")
            expected_disproofs: list[dict[str, Any]] = []
            member_matched_alternatives = 0
            for raw_vote in raw_votes:
                if not isinstance(raw_vote, dict):
                    raise RuntimeError("Repeated-dark semantic vote is invalid")
                alternative_view_id = raw_vote.get("view_id")
                alternative_group_id = raw_vote.get("canonical_group_id")
                if not (
                    raw_vote.get("alignment_trusted") is True
                    and raw_vote.get("unique_canonical_join") is True
                    and raw_vote.get("pixel_gate_accepted") is True
                    and raw_vote.get("status") in {"matched", "review"}
                    and isinstance(alternative_view_id, str)
                    and isinstance(alternative_group_id, str)
                    and alternative_group_id != group_id
                ):
                    continue
                effective_confidence = _quality_unit(
                    raw_vote.get("effective_confidence"),
                    "Repeated-dark semantic alternative confidence",
                )
                sample_shares = _quality_semantic_disproof_shares(
                    spatial_part=spatial_part,
                    view_id=alternative_view_id,
                    rejected_group_id=alternative_group_id,
                )
                shares = list(sample_shares.values())
                if raw_vote.get("status") == "matched":
                    member_matched_alternatives += 1
                    if (
                        sum(
                            share
                            <= float(
                                thresholds[
                                    "maximum_semantic_rejected_group_share"
                                ]
                            )
                            for share in shares
                        )
                        < int(thresholds["minimum_clean_semantic_disproof_samples"])
                        or max(shares)
                        > float(
                            thresholds[
                                "maximum_semantic_rejected_group_outlier_share"
                            ]
                        )
                    ):
                        raise RuntimeError(
                            "Repeated-dark matched semantic alternative is not disproven"
                        )
                    if (
                        alternative_group_id in matched_alternative_groups
                        or alternative_view_id in matched_alternative_views
                    ):
                        raise RuntimeError(
                            "Repeated-dark cohort has a homogeneous semantic conflict"
                        )
                    matched_alternative_groups.add(alternative_group_id)
                    matched_alternative_views.add(alternative_view_id)
                elif (
                    sum(
                        share
                        <= float(thresholds["maximum_review_rejected_group_share"])
                        for share in shares
                    )
                    < int(thresholds["minimum_clean_semantic_disproof_samples"])
                    or max(shares)
                    > float(
                        thresholds[
                            "maximum_review_rejected_group_outlier_share"
                        ]
                    )
                ):
                    raise RuntimeError(
                        "Repeated-dark review semantic alternative is not disproven"
                    )
                expected_disproofs.append(
                    {
                        "view_id": alternative_view_id,
                        "canonical_group_id": alternative_group_id,
                        "status": raw_vote["status"],
                        "effective_confidence": effective_confidence,
                        "sample_shares": sample_shares,
                    }
                )
            if member_matched_alternatives > 1:
                raise RuntimeError(
                    "Repeated-dark member has multiple matched semantic alternatives"
                )
            expected_disproofs.sort(
                key=lambda item: (
                    item["view_id"],
                    item["canonical_group_id"],
                    item["status"],
                )
            )
            diagnostic_alignment = raw_dark["alignment"]
            alignment_summary = {
                "score": diagnostic_alignment["score"],
                "projection_score": diagnostic_alignment["projection_score"],
                "projection_iou": diagnostic_alignment["projection_iou"],
                "ecc_correlation": diagnostic_alignment["ecc_correlation"],
                "ecc_status": diagnostic_alignment["ecc_status"],
            }

            raw_mapping_decisions = mapping_consensus.get("decisions")
            raw_gate_decisions = spatial_gate_audit.get("decisions")
            if not isinstance(raw_mapping_decisions, list) or not isinstance(
                raw_gate_decisions, list
            ):
                raise RuntimeError("Repeated-dark mapping decisions are invalid")
            mapping_decisions = [
                item
                for item in raw_mapping_decisions
                if isinstance(item, dict) and item.get("part_id") == part_id
            ]
            gate_decisions = [
                item
                for item in raw_gate_decisions
                if isinstance(item, dict) and item.get("part_id") == part_id
            ]
            if len(mapping_decisions) != 1 or len(gate_decisions) > 1:
                raise RuntimeError("Repeated-dark mapping decision is not unique")
            mapping_decision = mapping_decisions[0]
            if (
                mapping_decision.get("main_group_id") != group_id
                or mapping_decision.get("main_status") not in {"matched", "review"}
                or _quality_unit(
                    mapping_decision.get("main_confidence"),
                    "Repeated-dark mapping confidence",
                )
                < float(thresholds["minimum_mapping_confidence"])
                or (
                    mapping_decision.get("output_status") == "matched"
                    and mapping_decision.get("output_group_id") != group_id
                )
                or (
                    gate_decisions
                    and gate_decisions[0].get("output_status") == "matched"
                    and gate_decisions[0].get("output_group_id") != group_id
                )
            ):
                raise RuntimeError(
                    "Repeated-dark cohort conflicts with its mapping contract"
                )

            expected_member = {
                "part_id": part_id,
                "projected_part_pixels": projected_pixels,
                "direct_target_share": direct_share,
                "direct_target_margin": direct_margin,
                "direct_target_matching_pixels": direct_matching,
                "bbox_target_share": bbox_share,
                "bbox_target_margin": bbox_margin,
                "perturbations": expected_perturbations,
                "alignment": alignment_summary,
                "dark_diagnostic_sha256": dark_summary["diagnostic_sha256"],
                "dark_signal_share": dark_summary["dark_signal_share"],
                "dark_signal_purity": dark_summary["dark_signal_purity"],
                "core_dark_signal_share": raw_dark["core_dark_signal_share"],
                "adaptive_edge_density": dark_summary["adaptive_edge_density"],
                "dark_signal_null_margin": dark_summary["dark_signal_null_margin"],
                "evidence_contract": "dark_foreground_diagnostic",
                "semantic_alternative_disproofs": expected_disproofs,
            }
            expected_member["evidence_sha256"] = canonical_sha256(expected_member)
            if raw_member != expected_member:
                raise RuntimeError("Repeated-dark cohort member audit is inconsistent")
            total_projected += projected_pixels
            total_direct_matching += direct_matching
            expected_cohort_change_ids.add(part_id)
            validated_parts[part_id] = {
                "cohort_id": cohort_id,
                "cohort_part_ids": part_ids,
                "provenance": {
                    "lane": QUALITY_REPAIR_REPEATED_GEOMETRY_DARK_RESIDUAL_LANE,
                    "cohort_id": cohort_id,
                    "canonical_group_id": group_id,
                    "reference_view_id": view_id,
                    "geometry_signature_sha256": geometry_sha,
                    "source_visual_stable_properties_signature_sha256": source_sha,
                    "cohort_part_ids": part_ids,
                    "budget_pixels": budget_pixels,
                    "minimum_contribution_pixels": minimum_contribution,
                    "maximum_contribution_pixels": maximum_contribution,
                    "total_projected_part_pixels": total_projected,
                    "total_direct_target_matching_pixels": total_direct_matching,
                    "member_evidence_contract": expected_member[
                        "evidence_contract"
                    ],
                    "member_evidence_sha256": expected_member["evidence_sha256"],
                    "dark_diagnostic_sha256": dark_summary["diagnostic_sha256"],
                },
            }
        if (
            raw_cohort.get("total_projected_part_pixels") != total_projected
            or raw_cohort.get("total_direct_target_matching_pixels")
            != total_direct_matching
            or total_projected < minimum_contribution
            or total_projected > maximum_contribution
        ):
            raise RuntimeError("Repeated-dark atomic cohort budget is inconsistent")
        # Totals become known only after the full cohort.  Rebuild every
        # member's provenance from that atomic total, not from prefix sums.
        for part_id in part_ids:
            validated_parts[part_id]["provenance"].update(
                {
                    "total_projected_part_pixels": total_projected,
                    "total_direct_target_matching_pixels": total_direct_matching,
                }
            )

    if expected_cohort_change_ids != cohort_change_ids:
        raise RuntimeError("Repeated-dark cohorts do not atomically cover their changes")
    if any(
        changes_by_part[part_id].get("repeated_geometry_dark_cohort_id")
        != validated_parts[part_id]["cohort_id"]
        or changes_by_part[part_id].get("cohort_part_ids")
        != validated_parts[part_id]["cohort_part_ids"]
        for part_id in cohort_change_ids
    ):
        raise RuntimeError("Repeated-dark change cohort identity is inconsistent")
    return validated_parts

__all__ = [
    "_quality_dark_alignment",
    "_quality_dark_matched_semantic_conflict",
    "_quality_dark_reference_evidence",
    "_quality_dark_spatial_observation",
    "_quality_dark_spatial_part",
    "_quality_dark_support_from_report",
    "_quality_group_measurement",
    "_quality_group_share",
    "_quality_repeated_geometry_signature",
    "_quality_repeated_source_signature",
    "_quality_semantic_disproof_shares",
    "_quality_target_matching_pixels",
    "_recompute_quality_multiview_dark_identity",
    "_validate_quality_dark_foreground_diagnostic",
    "_validate_quality_multiview_semantic_review_support",
    "_validate_quality_semantic_review_override",
    "_validate_repeated_geometry_dark_cohorts",
]
