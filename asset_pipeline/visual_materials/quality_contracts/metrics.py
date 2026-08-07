"""Primitive scores, hashes and render-evidence validation."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..references import sha256_file
from .constants import LIGHTING_STATISTICS_SCHEMA_VERSION, QUALITY_DOMINANT_THRESHOLD_FIELDS


def _quality_can_measure_lighting_statistics(
    quality_report: dict[str, Any],
) -> bool:
    """Return whether QA carries the image contract needed by measurement.

    Aggregate-only reports deliberately do not trigger the optional appearance
    round.  Neutral-anchor eligibility remains the measurement stage's job.
    """

    raw_views = quality_report.get("views")
    if not isinstance(raw_views, list) or not raw_views:
        return False
    for raw_view in raw_views:
        if not isinstance(raw_view, dict):
            continue
        reference = raw_view.get("reference")
        render = raw_view.get("render")
        if (
            isinstance(raw_view.get("material_color"), dict)
            and isinstance(reference, dict)
            and isinstance(reference.get("image"), str)
            and bool(reference["image"])
            and isinstance(render, dict)
            and isinstance(render.get("image"), str)
            and bool(render["image"])
            and isinstance(render.get("part_ids"), str)
            and bool(render["part_ids"])
        ):
            return True
    return False


def _quality_has_lighting_normalized_groups(
    quality_report: dict[str, Any],
) -> bool:
    """Detect measured groups without weakening their schema boundary."""

    raw_views = quality_report.get("views")
    if not isinstance(raw_views, list):
        return False
    for raw_view in raw_views:
        if not isinstance(raw_view, dict):
            continue
        material_color = raw_view.get("material_color")
        if not isinstance(material_color, dict):
            continue
        statistics = material_color.get("lighting_normalized_groups")
        if (
            isinstance(statistics, dict)
            and statistics.get("schema_version")
            == LIGHTING_STATISTICS_SCHEMA_VERSION
            and isinstance(statistics.get("groups"), list)
            and bool(statistics["groups"])
        ):
            return True
    return False


def _appearance_baseline_safety_reason(
    *,
    quality_gate_status: Any,
    lighting_profile: Any,
) -> str | None:
    """Return the first trust-boundary reason that blocks refinement."""

    if quality_gate_status != "PASS":
        return "QUALITY_GATE_IS_NOT_PASS"
    if lighting_profile != "material-neutral":
        return "BASELINE_LIGHTING_PROFILE_IS_NOT_MATERIAL_NEUTRAL"
    return None
def _quality_unit(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise RuntimeError(f"{label} must be a finite number in [0, 1]")
    return float(value)


def _quality_sorted_strings(value: Any, label: str) -> list[str]:
    if (
        not isinstance(value, list)
        or any(not isinstance(item, str) or not item for item in value)
        or value != sorted(set(value))
    ):
        raise RuntimeError(f"{label} must be a sorted unique string array")
    return list(value)


def _validate_quality_dominant_mass(
    report: dict[str, Any],
) -> dict[str, Any]:
    """Independently validate the optional dominant-family QA contract.

    The production comparator always emits this contract.  Complete legacy
    fixtures without any dominant fields remain readable, but partial fields,
    fabricated decisions, and threshold-less claims fail closed.
    """

    thresholds = report.get("thresholds")
    views = report.get("views")
    aggregate = report.get("aggregate")
    if thresholds is None and views is None and aggregate is None:
        return {"enabled": False, "thresholds": {}, "views": {}}
    if (
        not isinstance(thresholds, dict)
        or not isinstance(views, list)
        or not isinstance(aggregate, dict)
    ):
        raise RuntimeError("Visual quality dominant-mass report is incomplete")
    present = {
        field for field in QUALITY_DOMINANT_THRESHOLD_FIELDS if field in thresholds
    }
    aggregate_reasons = aggregate.get("reasons", [])
    if not isinstance(aggregate_reasons, list) or any(
        not isinstance(reason, str) for reason in aggregate_reasons
    ):
        raise RuntimeError("Visual quality aggregate reasons are invalid")
    if not present:
        for view in views:
            if not isinstance(view, dict):
                raise RuntimeError("Visual quality view is invalid")
            reasons = view.get("reasons", [])
            color = view.get("material_color")
            if (
                not isinstance(reasons, list)
                or "trusted_dominant_family_mass_deficit" in reasons
                or (
                    isinstance(color, dict)
                    and "trusted_evidence_dominant_mass" in color
                )
            ):
                raise RuntimeError(
                    "Visual quality dominant-mass evidence lacks thresholds"
                )
        if (
            "single_strong_view_confirms_dominant_family_mass_deficit"
            in aggregate_reasons
        ):
            raise RuntimeError(
                "Visual quality aggregate has an unbound dominant-mass claim"
            )
        return {"enabled": False, "thresholds": {}, "views": {}}
    if present != set(QUALITY_DOMINANT_THRESHOLD_FIELDS):
        raise RuntimeError(
            "Visual quality dominant-mass threshold contract is incomplete"
        )
    dominant_thresholds = {
        field: _quality_unit(
            thresholds[field], f"Visual quality threshold {field}"
        )
        for field in QUALITY_DOMINANT_THRESHOLD_FIELDS
    }
    strong_alignment = _quality_unit(
        thresholds.get("strong_alignment_score"),
        "Visual quality threshold strong_alignment_score",
    )

    view_records: dict[str, dict[str, Any]] = {}
    failed_view_ids: list[str] = []
    for raw_view in views:
        if not isinstance(raw_view, dict):
            raise RuntimeError("Visual quality dominant-mass view is invalid")
        view_id = raw_view.get("reference_view_id")
        status = raw_view.get("status")
        reasons = raw_view.get("reasons")
        if (
            not isinstance(view_id, str)
            or not view_id
            or view_id in view_records
            or not isinstance(status, str)
            or not isinstance(reasons, list)
            or any(not isinstance(reason, str) for reason in reasons)
        ):
            raise RuntimeError("Visual quality dominant-mass view is invalid")
        color = raw_view.get("material_color")
        if not isinstance(color, dict):
            if "trusted_dominant_family_mass_deficit" in reasons:
                raise RuntimeError(
                    "Unscored view claims a dominant-family material deficit"
                )
            view_records[view_id] = {"families": {}, "failed_family_keys": []}
            continue
        alignment = raw_view.get("alignment")
        dominant = color.get("trusted_evidence_dominant_mass")
        reference_distribution = color.get("reference_distribution")
        render_distribution = color.get("render_distribution")
        if (
            not isinstance(alignment, dict)
            or not isinstance(dominant, dict)
            or not isinstance(reference_distribution, dict)
            or not isinstance(render_distribution, dict)
            or not isinstance(
                reference_distribution.get("category_distribution"), dict
            )
            or not isinstance(
                render_distribution.get("category_distribution"), dict
            )
        ):
            raise RuntimeError(
                "Visual quality dominant-mass evidence is incomplete"
            )
        alignment_score = _quality_unit(
            alignment.get("score"),
            f"Visual quality dominant alignment score {view_id}",
        )
        silhouette_iou = _quality_unit(
            alignment.get("silhouette_iou"),
            f"Visual quality dominant silhouette IoU {view_id}",
        )
        raw_families = dominant.get("families")
        if not isinstance(raw_families, list):
            raise RuntimeError(
                "Visual quality dominant-mass families are invalid"
            )
        reference_shares = reference_distribution["category_distribution"]
        render_shares = render_distribution["category_distribution"]
        parsed: dict[str, dict[str, Any]] = {}
        for family in raw_families:
            if not isinstance(family, dict):
                raise RuntimeError(
                    "Visual quality dominant-mass family is invalid"
                )
            bins = _quality_sorted_strings(
                family.get("render_color_bins"),
                "Visual quality dominant render_color_bins",
            )
            local_group_ids = _quality_sorted_strings(
                family.get("local_group_ids"),
                "Visual quality dominant local_group_ids",
            )
            base_colors = _quality_sorted_strings(
                family.get("base_colors"),
                "Visual quality dominant base_colors",
            )
            if not bins or not local_group_ids or not base_colors:
                raise RuntimeError(
                    "Visual quality dominant-mass family identity is empty"
                )
            family_key = family.get("family_key")
            if (
                not isinstance(family_key, str)
                or family_key != "|".join(bins)
                or family_key in parsed
            ):
                raise RuntimeError(
                    "Visual quality dominant-mass family identity is invalid"
                )
            try:
                expected_reference_share = sum(
                    _quality_unit(
                        reference_shares[label],
                        f"Visual quality reference family share {label}",
                    )
                    for label in bins
                )
                expected_observed_share = sum(
                    _quality_unit(
                        render_shares[label],
                        f"Visual quality render family share {label}",
                    )
                    for label in bins
                )
            except KeyError as exc:
                raise RuntimeError(
                    "Visual quality dominant family bin is absent from distribution"
                ) from exc
            if (
                expected_reference_share > 1.0 + 1e-9
                or expected_observed_share > 1.0 + 1e-9
            ):
                raise RuntimeError(
                    "Visual quality dominant-family share exceeds one"
                )
            margin = family.get("reference_share_margin")
            if (
                isinstance(margin, bool)
                or not isinstance(margin, (int, float))
                or not math.isfinite(float(margin))
                or not -1.0 <= float(margin) <= 1.0
            ):
                raise RuntimeError(
                    "Visual quality dominant-family margin is invalid"
                )
            parsed[family_key] = {
                "raw": family,
                "family_key": family_key,
                "render_color_bins": bins,
                "local_group_ids": local_group_ids,
                "base_colors": base_colors,
                "reference_share": _quality_unit(
                    family.get("reference_share"),
                    "Visual quality dominant reference_share",
                ),
                "runner_up_reference_share": _quality_unit(
                    family.get("runner_up_reference_share"),
                    "Visual quality dominant runner_up_reference_share",
                ),
                "reference_share_margin": float(margin),
                "observed_render_share": _quality_unit(
                    family.get("observed_render_share"),
                    "Visual quality dominant observed_render_share",
                ),
                "deficit_share": _quality_unit(
                    family.get("deficit_share"),
                    "Visual quality dominant deficit_share",
                ),
                "mass_recall": _quality_unit(
                    family.get("mass_recall"),
                    "Visual quality dominant mass_recall",
                ),
                "expected_reference_share": expected_reference_share,
                "expected_observed_share": expected_observed_share,
            }

        failed_keys: list[str] = []
        eligible_count = 0
        for family_key, item in parsed.items():
            runner_up = max(
                (
                    other["reference_share"]
                    for other_key, other in parsed.items()
                    if other_key != family_key
                ),
                default=0.0,
            )
            reference_share = item["reference_share"]
            observed_share = item["observed_render_share"]
            expected_margin = reference_share - runner_up
            expected_deficit = max(0.0, reference_share - observed_share)
            expected_recall = (
                min(1.0, observed_share / reference_share)
                if reference_share > 0.0
                else 1.0
            )
            pairs = (
                (reference_share, item["expected_reference_share"]),
                (observed_share, item["expected_observed_share"]),
                (item["runner_up_reference_share"], runner_up),
                (item["reference_share_margin"], expected_margin),
                (item["deficit_share"], expected_deficit),
                (item["mass_recall"], expected_recall),
            )
            if any(
                not math.isclose(left, right, rel_tol=0.0, abs_tol=1e-9)
                for left, right in pairs
            ):
                raise RuntimeError(
                    "Visual quality dominant-mass numeric evidence is inconsistent"
                )
            eligibility_reasons: list[str] = []
            if (
                reference_share
                < dominant_thresholds["minimum_dominant_reference_share"]
            ):
                eligibility_reasons.append(
                    "REFERENCE_SHARE_BELOW_DOMINANT_FLOOR"
                )
            if (
                expected_margin
                < dominant_thresholds["minimum_dominant_share_margin"]
            ):
                eligibility_reasons.append(
                    "REFERENCE_DOMINANCE_MARGIN_BELOW_FLOOR"
                )
            if alignment_score < strong_alignment:
                eligibility_reasons.append("ALIGNMENT_NOT_STRONG")
            if (
                silhouette_iou
                < dominant_thresholds["minimum_dominant_silhouette_iou"]
            ):
                eligibility_reasons.append(
                    "SILHOUETTE_IOU_BELOW_DOMINANT_FLOOR"
                )
            eligible = not eligibility_reasons
            hard_failure = (
                eligible
                and item["mass_recall"]
                < dominant_thresholds["minimum_dominant_mass_recall"]
                and item["deficit_share"]
                >= dominant_thresholds["minimum_dominant_absolute_deficit"]
            )
            expected_status = (
                "FAIL"
                if hard_failure
                else "PASS"
                if eligible
                else "NOT_APPLICABLE"
            )
            expected_reasons = (
                ["DOMINANT_FAMILY_MASS_DEFICIT"]
                if hard_failure
                else eligibility_reasons
            )
            raw = item["raw"]
            if (
                raw.get("eligible") is not eligible
                or raw.get("status") != expected_status
                or raw.get("reason_codes") != expected_reasons
            ):
                raise RuntimeError(
                    "Visual quality dominant-mass family decision is inconsistent"
                )
            item["eligible"] = eligible
            item["status"] = expected_status
            item["reason_codes"] = expected_reasons
            if eligible:
                eligible_count += 1
            if hard_failure:
                failed_keys.append(family_key)

        raw_eligible_count = dominant.get("eligible_family_count")
        raw_failed_count = dominant.get("failed_family_count")
        expected_status = (
            "FAIL"
            if failed_keys
            else "PASS"
            if eligible_count
            else "NOT_APPLICABLE"
        )
        claims_failure = "trusted_dominant_family_mass_deficit" in reasons
        if (
            isinstance(raw_eligible_count, bool)
            or not isinstance(raw_eligible_count, int)
            or raw_eligible_count != eligible_count
            or isinstance(raw_failed_count, bool)
            or not isinstance(raw_failed_count, int)
            or raw_failed_count != len(failed_keys)
            or dominant.get("status") != expected_status
            or claims_failure != bool(failed_keys)
            or (failed_keys and status != "FAIL")
        ):
            raise RuntimeError(
                "Visual quality dominant-mass view decision is inconsistent"
            )
        if failed_keys:
            failed_view_ids.append(view_id)
        view_records[view_id] = {
            "families": {
                key: {
                    field: value
                    for field, value in item.items()
                    if field
                    not in {
                        "raw",
                        "expected_reference_share",
                        "expected_observed_share",
                    }
                }
                for key, item in parsed.items()
            },
            "failed_family_keys": sorted(failed_keys),
        }

    aggregate_claim = (
        "single_strong_view_confirms_dominant_family_mass_deficit"
        in aggregate_reasons
    )
    if (
        aggregate_claim != bool(failed_view_ids)
        or (failed_view_ids and aggregate.get("status") != "FAIL")
    ):
        raise RuntimeError(
            "Visual quality aggregate dominant-mass decision is inconsistent"
        )
    return {
        "enabled": True,
        "thresholds": dominant_thresholds,
        "views": view_records,
        "failed_view_ids": sorted(failed_view_ids),
    }


def _independent_quality_spatial_anchor_view_ids(
    *,
    spatial_report: dict[str, Any],
    part_id: str,
    canonical_group_id: str,
    target_view_id: str,
) -> list[str]:
    """Recompute the single-QA-view lane's independent pixel anchor."""

    raw_reference_evidence = spatial_report.get("reference_evidence")
    raw_parts = spatial_report.get("parts")
    if not isinstance(raw_reference_evidence, list) or not isinstance(
        raw_parts, list
    ):
        return []
    references: dict[str, dict[str, Any]] = {}
    for record in raw_reference_evidence:
        if not isinstance(record, dict):
            return []
        view_id = record.get("view_id")
        if not isinstance(view_id, str) or not view_id or view_id in references:
            return []
        references[view_id] = record
    target = references.get(target_view_id)
    if not isinstance(target, dict) or target.get("alignment_trusted") is not True:
        return []

    spatial_part = next(
        (
            record
            for record in raw_parts
            if isinstance(record, dict) and record.get("part_id") == part_id
        ),
        None,
    )
    if not isinstance(spatial_part, dict):
        return []
    observations = spatial_part.get("observations")
    if not isinstance(observations, list):
        return []

    anchors: set[str] = set()
    for observation in observations:
        if not isinstance(observation, dict):
            continue
        view_id = observation.get("reference_view_id")
        if not isinstance(view_id, str) or view_id == target_view_id:
            continue
        stable_target = (
            observation.get("classification") == "resolved"
            and observation.get("canonical_group_id") == canonical_group_id
            and observation.get("registration_label_stable") is True
            and observation.get("perturbation_label_stable") is True
        )
        diagnostic = observation.get("canonical_palette_diagnostic")
        diagnostic_target = (
            isinstance(diagnostic, dict)
            and diagnostic.get("status") == "resolved"
            and diagnostic.get("reason_codes") == []
            and diagnostic.get("evidence_scope")
            == "canonical_multiview_propagation_repair_only"
            and diagnostic.get("canonical_group_id") == canonical_group_id
            and diagnostic.get("bbox_canonical_group_id")
            == canonical_group_id
            and diagnostic.get("registration_label_stable") is True
            and diagnostic.get("perturbation_label_stable") is True
            and diagnostic.get("resolved_sample_count") == 6
            and diagnostic.get("target_sample_count") == 6
            and diagnostic.get("consensus_ratio") == 1.0
            and diagnostic.get("alternative_canonical_group_ids") == []
        )
        small_diagnostic = observation.get("small_part_diagnostic")
        small_diagnostic_target = (
            isinstance(small_diagnostic, dict)
            and small_diagnostic.get("status") == "resolved"
            and small_diagnostic.get("reason_codes") == []
            and small_diagnostic.get("canonical_group_id")
            == canonical_group_id
            and small_diagnostic.get("bbox_canonical_group_id")
            == canonical_group_id
            and small_diagnostic.get("registration_label_stable") is True
            and small_diagnostic.get("consensus_ratio") == 1.0
            and small_diagnostic.get("alternative_canonical_group_ids") == []
            and observation.get("perturbation_label_stable") is True
        )
        if (
            not stable_target
            and not diagnostic_target
            and not small_diagnostic_target
        ):
            continue
        anchor = references.get(view_id)
        if not isinstance(anchor, dict) or anchor.get("alignment_trusted") is not True:
            continue
        independence_fields = (
            "raw_sha256",
            "normalized_pixel_sha256",
            "content_cluster_id",
            "pose_cluster_id",
        )
        if all(
            isinstance(target.get(field), str)
            and isinstance(anchor.get(field), str)
            and target[field] != anchor[field]
            for field in independence_fields
        ):
            anchors.add(view_id)
    return sorted(anchors)


def _quality_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _quality_finite_number(
    value: Any,
    label: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or (minimum is not None and float(value) < minimum)
        or (maximum is not None and float(value) > maximum)
    ):
        raise RuntimeError(f"{label} is not a valid finite number")
    return float(value)


def _quality_linear_quantile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise RuntimeError("Dark-foreground null control has no valid samples")
    index = (len(ordered) - 1) * quantile
    lower = int(math.floor(index))
    upper = int(math.ceil(index))
    if lower == upper:
        return ordered[lower]
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight
def _quality_group_recalls(
    report: dict[str, Any],
) -> tuple[float, dict[str, dict[str, float]], dict[str, str]]:
    thresholds = report.get("thresholds")
    if not isinstance(thresholds, dict):
        raise RuntimeError("Visual quality report has no thresholds")
    threshold = thresholds.get("minimum_evidence_group_recall")
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not 0.0 <= float(threshold) <= 1.0
    ):
        raise RuntimeError("Visual quality group-recall threshold is invalid")
    recalls: dict[str, dict[str, float]] = {}
    statuses: dict[str, str] = {}
    views = report.get("views")
    if not isinstance(views, list):
        raise RuntimeError("Visual quality report has no view records")
    for view in views:
        if not isinstance(view, dict):
            raise RuntimeError("Visual quality report contains an invalid view")
        view_id = view.get("reference_view_id")
        status = view.get("status")
        if not isinstance(view_id, str) or not isinstance(status, str):
            raise RuntimeError("Visual quality view identity/status is invalid")
        if view_id in statuses:
            raise RuntimeError(f"Visual quality report repeats view {view_id}")
        statuses[view_id] = status
        material_color = view.get("material_color")
        group_report = (
            material_color.get("trusted_evidence_group_recall")
            if isinstance(material_color, dict)
            else None
        )
        groups = group_report.get("groups") if isinstance(group_report, dict) else None
        if not isinstance(groups, list):
            continue
        view_recalls: dict[str, float] = {}
        for group in groups:
            if not isinstance(group, dict):
                raise RuntimeError("Visual quality group-recall record is invalid")
            group_id = group.get("group_id")
            recall = group.get("recall")
            if (
                not isinstance(group_id, str)
                or group_id in view_recalls
                or isinstance(recall, bool)
                or not isinstance(recall, (int, float))
                or not 0.0 <= float(recall) <= 1.0
            ):
                raise RuntimeError("Visual quality group recall is invalid")
            view_recalls[group_id] = float(recall)
        recalls[view_id] = view_recalls
    return float(threshold), recalls, statuses


def _quality_evidence_contract(report: dict[str, Any]) -> dict[str, Any]:
    """Return the immutable reference/mapping contract for one QA report."""

    thresholds = report.get("thresholds")
    inputs = report.get("inputs")
    views = report.get("views")
    if (
        not isinstance(thresholds, dict)
        or not isinstance(inputs, dict)
        or not isinstance(views, list)
    ):
        raise RuntimeError("Visual quality report lacks its evidence contract")
    reference_manifest_sha256 = inputs.get("reference_manifest_sha256")
    mapping_mode = inputs.get("mapping_mode")
    selected_view_mapping = inputs.get("selected_view_mapping")
    if (
        not isinstance(reference_manifest_sha256, str)
        or len(reference_manifest_sha256) != 64
        or not isinstance(mapping_mode, str)
        or not mapping_mode
        or not isinstance(selected_view_mapping, dict)
    ):
        raise RuntimeError("Visual quality input evidence contract is invalid")

    view_contracts: dict[str, dict[str, Any]] = {}
    for view in views:
        if not isinstance(view, dict):
            raise RuntimeError("Visual quality view evidence is invalid")
        view_id = view.get("reference_view_id")
        reference = view.get("reference")
        mapping = view.get("mapping")
        if (
            not isinstance(view_id, str)
            or not view_id
            or view_id in view_contracts
            or not isinstance(reference, dict)
            or not isinstance(mapping, dict)
        ):
            raise RuntimeError("Visual quality view evidence is invalid")
        image_sha256 = reference.get("image_sha256")
        selected_render_view_id = mapping.get("selected_render_view_id")
        if (
            not isinstance(image_sha256, str)
            or len(image_sha256) != 64
            or (
                selected_render_view_id is not None
                and (
                    not isinstance(selected_render_view_id, str)
                    or not selected_render_view_id
                )
            )
            or selected_render_view_id != view.get("render_view_id")
        ):
            raise RuntimeError("Visual quality view evidence is invalid")
        view_contracts[view_id] = {
            "reference_image_sha256": image_sha256,
            "selected_render_view_id": selected_render_view_id,
        }
    return {
        "thresholds": thresholds,
        "reference_manifest_sha256": reference_manifest_sha256,
        "mapping_mode": mapping_mode,
        "selected_view_mapping": selected_view_mapping,
        "views": view_contracts,
    }


def _validate_quality_render_contract(
    *,
    quality_report: dict[str, Any],
    rendered_registry: dict[str, Any],
    rendered_registry_path: Path,
    spatial_report: dict[str, Any],
) -> None:
    """Bind final RGB/part-ID renders to the spatial evidence pose by hash."""

    if rendered_registry.get("schema_version") != "qwen-material-parts/v1":
        raise RuntimeError("Final quality registry has an unsupported schema_version")
    report_inputs = quality_report.get("inputs")
    render_set = rendered_registry.get("render_set")
    spatial_inputs = spatial_report.get("inputs")
    if (
        not isinstance(report_inputs, dict)
        or not isinstance(render_set, dict)
        or not isinstance(spatial_inputs, dict)
    ):
        raise RuntimeError("Final quality render contract is incomplete")

    try:
        resolved_registry_path = rendered_registry_path.expanduser().resolve(
            strict=True
        )
        reported_registry_path = Path(
            str(report_inputs.get("rendered_registry", ""))
        ).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise RuntimeError("Final quality registry path is invalid") from exc
    if (
        reported_registry_path != resolved_registry_path
        or report_inputs.get("rendered_registry_sha256")
        != sha256_file(resolved_registry_path)
    ):
        raise RuntimeError(
            "Final quality report is not hash-bound to its rendered registry"
        )

    raw_views = render_set.get("views")
    if not isinstance(raw_views, list) or len(raw_views) < 2:
        raise RuntimeError("Final quality registry has insufficient render views")
    registry_views: dict[str, dict[str, Any]] = {}
    for raw_view in raw_views:
        if not isinstance(raw_view, dict):
            raise RuntimeError("Final quality registry contains an invalid view")
        view_id = raw_view.get("view_id")
        if (
            not isinstance(view_id, str)
            or not view_id
            or view_id in registry_views
        ):
            raise RuntimeError("Final quality registry contains duplicate view IDs")
        registry_views[view_id] = raw_view

    raw_spatial_files = spatial_inputs.get("files")
    if not isinstance(raw_spatial_files, list):
        raise RuntimeError("Spatial report lacks its input-file ledger")
    spatial_part_ids: dict[str, str] = {}
    for item in raw_spatial_files:
        if not isinstance(item, dict):
            raise RuntimeError("Spatial input-file ledger is invalid")
        label = item.get("label")
        if not isinstance(label, str) or not label.startswith("part_ids:"):
            continue
        view_id = label.removeprefix("part_ids:")
        digest = item.get("sha256")
        if (
            not view_id
            or view_id in spatial_part_ids
            or not isinstance(digest, str)
            or len(digest) != 64
        ):
            raise RuntimeError("Spatial part-ID input ledger is invalid")
        spatial_part_ids[view_id] = digest
    if set(spatial_part_ids) != set(registry_views):
        raise RuntimeError(
            "Final quality render views differ from the spatial evidence poses"
        )

    registry_file_hashes: dict[str, tuple[str, str]] = {}
    for view_id, view in registry_views.items():
        rgb = view.get("rgb")
        part_ids = view.get("part_ids")
        if not isinstance(rgb, str) or not isinstance(part_ids, str):
            raise RuntimeError(
                f"Final quality registry view lacks render paths: {view_id}"
            )
        try:
            rgb_path = Path(rgb).expanduser().resolve(strict=True)
            part_ids_path = Path(part_ids).expanduser().resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise RuntimeError(
                f"Final quality render file is unavailable: {view_id}"
            ) from exc
        rgb_digest = sha256_file(rgb_path)
        part_ids_digest = sha256_file(part_ids_path)
        if part_ids_digest != spatial_part_ids[view_id]:
            raise RuntimeError(
                "Final material render changed geometry, camera pose, or part-ID "
                f"projection: {view_id}"
            )
        registry_file_hashes[view_id] = (rgb_digest, part_ids_digest)

    raw_quality_views = quality_report.get("views")
    if not isinstance(raw_quality_views, list):
        raise RuntimeError("Final quality report lacks view evidence")
    seen_reference_ids: set[str] = set()
    seen_render_ids: set[str] = set()
    for raw_view in raw_quality_views:
        if not isinstance(raw_view, dict):
            raise RuntimeError("Final quality report contains an invalid view")
        reference_id = raw_view.get("reference_view_id")
        render_view_id = raw_view.get("render_view_id")
        if (
            not isinstance(reference_id, str)
            or not reference_id
            or reference_id in seen_reference_ids
        ):
            raise RuntimeError("Final quality report repeats a reference view")
        seen_reference_ids.add(reference_id)
        if render_view_id is None:
            if raw_view.get("status") != "UNSCORABLE":
                raise RuntimeError(
                    "Final quality view has no render without being UNSCORABLE"
                )
            continue
        if (
            not isinstance(render_view_id, str)
            or render_view_id not in registry_views
            or render_view_id in seen_render_ids
        ):
            raise RuntimeError("Final quality report has an invalid render mapping")
        seen_render_ids.add(render_view_id)
        render = raw_view.get("render")
        if not isinstance(render, dict):
            raise RuntimeError("Final quality report lacks render evidence")
        registry_view = registry_views[render_view_id]
        try:
            reported_rgb = Path(str(render.get("image", ""))).expanduser().resolve(
                strict=True
            )
            reported_part_ids = Path(
                str(render.get("part_ids", ""))
            ).expanduser().resolve(strict=True)
            registry_rgb = Path(str(registry_view["rgb"])).expanduser().resolve(
                strict=True
            )
            registry_part_ids = Path(
                str(registry_view["part_ids"])
            ).expanduser().resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise RuntimeError("Final quality view render path is invalid") from exc
        rgb_digest, part_ids_digest = registry_file_hashes[render_view_id]
        if (
            reported_rgb != registry_rgb
            or reported_part_ids != registry_part_ids
            or render.get("image_sha256") != rgb_digest
            or render.get("part_ids_sha256") != part_ids_digest
        ):
            raise RuntimeError(
                "Final quality view is not hash-bound to its RGB and part-ID files: "
                f"{reference_id}"
            )

__all__ = [
    "_appearance_baseline_safety_reason",
    "_independent_quality_spatial_anchor_view_ids",
    "_quality_can_measure_lighting_statistics",
    "_quality_evidence_contract",
    "_quality_finite_number",
    "_quality_group_recalls",
    "_quality_has_lighting_normalized_groups",
    "_quality_linear_quantile",
    "_quality_sha256",
    "_quality_sorted_strings",
    "_quality_unit",
    "_validate_quality_dominant_mass",
    "_validate_quality_render_contract",
]
