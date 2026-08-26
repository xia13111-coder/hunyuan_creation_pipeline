#!/usr/bin/env python3
"""Deterministic lighting-normalized optimization for shared MDL parameters.

The ordinary reference/render comparison intentionally works on rendered RGB.
That is useful for delivery QA, but it cannot distinguish an incorrect material
tint from view-dependent illumination.  In particular, two images can remain
in the same coarse hue bin while their values differ substantially.

This module provides a separate, fail-closed optimization boundary:

* MVInverse albedo identifies the canonical material group and anchors hue.
* Existing dominant-group QA statistics can diagnose contradictory per-view
  brightness residuals, but can never authorize a parameter change.
* A parameter suggestion requires low-resolution target and achromatic-anchor
  medians for every eligible view.  Target luminance is divided by the neutral
  anchor luminance independently in the reference and render.
* A single bounded exposure adjustment is authored atomically to every member
  of the discovered shared-material cohort.
* A candidate is accepted only after a re-render proves objective improvement,
  exact view/cohort coverage, no per-view regression, and no aggregate QA
  regression.

The implementation is offline and deterministic.  It does not render, invoke a
model, change view matching, or special-case a part ID, material colour, or
asset.  ``reference_compare`` may emit the optional minimal statistics contract
documented by :data:`LIGHTING_STATISTICS_SCHEMA_VERSION`.
"""

from __future__ import annotations

import argparse
import colorsys
import copy
import hashlib
import json
import math
import os
import statistics
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps, UnidentifiedImageError

from qwen_material_pipeline.materials.parameters import srgb_to_linear
from qwen_material_pipeline.materials.tuning import (
    tune_selected_material_from_mvinverse,
    tuning_profile_for_material,
)
from qwen_material_pipeline.usd.material_common import canonical_sha256


PLAN_SCHEMA_VERSION = "1.0"
QUALITY_SCHEMA_VERSION = "qwen-reference-render-comparison/v1"
MVINVERSE_SCHEMA_VERSION = "qwen-mvinverse-pbr-evidence/v1"
PALETTE_FUSION_SCHEMA_VERSION = "qwen-multiview-palette-fusion/v1"
SPATIAL_REPORT_SCHEMA_VERSION = "qwen-spatial-mapping-audit/v1"
LIGHTING_STATISTICS_SCHEMA_VERSION = "qwen-lighting-normalized-group-statistics/v1"
MEASUREMENT_REPORT_SCHEMA_VERSION = "qwen-lighting-normalized-group-measurement/v1"
CONTRACT_SCHEMA_VERSION = "qwen-shared-material-appearance-optimization/v1"
APPLY_REPORT_SCHEMA_VERSION = "qwen-shared-material-appearance-optimization-apply/v1"
VALIDATION_SCHEMA_VERSION = "qwen-shared-material-appearance-optimization-validation/v1"
OPTIMIZATION_MODE = "lighting_normalized_shared_parameter_bounded_step/v1"

DECISION_ADJUST = "adjust_shared_material"
DECISION_PRESERVE = "preserve"
DECISION_LIGHTING_INCONSISTENT = "lighting_inconsistent"
DECISION_INSUFFICIENT = "insufficient_evidence"

EXIT_SUCCESS = 0
EXIT_INPUT_ERROR = 2
EXIT_REQUIRE_PASS_FAILED = 3

DEFAULT_POLICY: dict[str, float | int] = {
    "minimum_shared_part_count": 2,
    "minimum_trusted_view_count": 2,
    "minimum_alignment_score": 0.55,
    "minimum_dominant_reference_group_share": 0.50,
    "minimum_sample_pixels": 128,
    "minimum_neutral_linear_luminance": 0.05,
    "maximum_neutral_chromaticity_distance": 0.08,
    "maximum_target_albedo_chromaticity_distance": 0.15,
    "maximum_unanchored_raw_gain_span_stops": 0.45,
    "maximum_normalized_gain_span_stops": 0.25,
    "maximum_normalized_gain_mad_stops": 0.10,
    "minimum_adjustment_stops": 0.06,
    "maximum_adjustment_stops": 0.35,
    "minimum_relative_objective_improvement": 0.20,
    "minimum_absolute_objective_improvement_stops": 0.03,
    "maximum_view_regression_stops": 0.03,
    "maximum_candidate_residual_median_stops": 0.12,
    "maximum_candidate_span_regression_stops": 0.02,
}

_POLICY_INTEGER_FIELDS = frozenset(
    {
        "minimum_shared_part_count",
        "minimum_trusted_view_count",
        "minimum_sample_pixels",
    }
)


class AppearanceOptimizationError(ValueError):
    """Raised when an optimization artifact violates the trust boundary."""


def _object(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AppearanceOptimizationError(f"{label} must be an object")
    return value


def _array(value: Any, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise AppearanceOptimizationError(f"{label} must be an array")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AppearanceOptimizationError(f"{label} must be a non-empty string")
    return value.strip()


def _unit(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise AppearanceOptimizationError(
            f"{label} must be a finite number from 0 to 1"
        )
    return float(value)


def _positive(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise AppearanceOptimizationError(f"{label} must be a finite positive number")
    return float(value)


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise AppearanceOptimizationError(f"{label} must be an integer >= {minimum}")
    return value


def _rgb(value: Any, label: str) -> list[float]:
    raw = _array(value, label)
    if len(raw) != 3:
        raise AppearanceOptimizationError(f"{label} must contain three channels")
    return [_unit(channel, f"{label}[{index}]") for index, channel in enumerate(raw)]


def _sorted_unique_texts(value: Any, label: str) -> list[str]:
    result = [
        _text(item, f"{label}[{index}]")
        for index, item in enumerate(_array(value, label))
    ]
    if result != sorted(set(result)):
        raise AppearanceOptimizationError(f"{label} must be sorted with no duplicates")
    return result


def _schema(document: Mapping[str, Any], expected: str, label: str) -> None:
    if document.get("schema_version") != expected:
        raise AppearanceOptimizationError(f"{label} has unsupported schema_version")


def _effective_policy(overrides: Mapping[str, Any] | None) -> dict[str, float | int]:
    policy = dict(DEFAULT_POLICY)
    if overrides is not None:
        unknown = set(overrides) - set(policy)
        if unknown:
            raise AppearanceOptimizationError(
                f"unknown appearance optimization policy fields: {sorted(unknown)}"
            )
        policy.update(overrides)
    for name, value in policy.items():
        if name in _POLICY_INTEGER_FIELDS:
            _integer(value, f"policy.{name}", minimum=1)
        else:
            _unit(value, f"policy.{name}")
    if float(policy["minimum_adjustment_stops"]) >= float(
        policy["maximum_adjustment_stops"]
    ):
        raise AppearanceOptimizationError(
            "policy adjustment floor must be below its bound"
        )
    if float(policy["maximum_normalized_gain_mad_stops"]) > float(
        policy["maximum_normalized_gain_span_stops"]
    ):
        raise AppearanceOptimizationError(
            "policy normalized MAD bound cannot exceed its span bound"
        )
    return policy


def _luminance(rgb: Sequence[float]) -> float:
    return 0.2126 * float(rgb[0]) + 0.7152 * float(rgb[1]) + 0.0722 * float(rgb[2])


def _chromaticity(rgb: Sequence[float]) -> tuple[float, float, float]:
    total = sum(float(channel) for channel in rgb)
    if total <= 1e-9:
        raise AppearanceOptimizationError(
            "appearance sample is too dark for chromaticity"
        )
    return tuple(float(channel) / total for channel in rgb)  # type: ignore[return-value]


def _chromaticity_distance(left: Sequence[float], right: Sequence[float]) -> float:
    left_chroma = _chromaticity(left)
    right_chroma = _chromaticity(right)
    return math.sqrt(
        sum((left_chroma[index] - right_chroma[index]) ** 2 for index in range(3))
    )


def _gain_summary(records: Sequence[Mapping[str, Any]], field: str) -> dict[str, Any]:
    values = [float(record[field]) for record in records]
    median = statistics.median(values)
    absolute = [abs(value) for value in values]
    return {
        "median_stops": median,
        "mad_stops": statistics.median(abs(value - median) for value in values),
        "span_stops": max(values) - min(values),
        "objective_median_absolute_stops": statistics.median(absolute),
        "maximum_absolute_stops": max(absolute),
    }


def _numbers_equal(left: Any, right: Any, *, tolerance: float = 1e-9) -> bool:
    if (
        isinstance(left, Sequence)
        and not isinstance(left, (str, bytes))
        and isinstance(right, Sequence)
        and not isinstance(right, (str, bytes))
    ):
        return len(left) == len(right) and all(
            _numbers_equal(a, b, tolerance=tolerance) for a, b in zip(left, right)
        )
    if (
        isinstance(left, (int, float))
        and not isinstance(left, bool)
        and isinstance(right, (int, float))
        and not isinstance(right, bool)
    ):
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)
    return left == right


def _plan_assignments(
    final_plan: Mapping[str, Any],
) -> tuple[list[Mapping[str, Any]], dict[str, Mapping[str, Any]]]:
    _schema(final_plan, PLAN_SCHEMA_VERSION, "final plan")
    assignments = [
        _object(raw, f"final plan assignment[{index}]")
        for index, raw in enumerate(
            _array(final_plan.get("assignments"), "final plan assignments")
        )
    ]
    by_part: dict[str, Mapping[str, Any]] = {}
    for index, assignment in enumerate(assignments):
        part_id = _text(assignment.get("part_id"), f"assignment[{index}].part_id")
        if part_id in by_part:
            raise AppearanceOptimizationError(
                f"final plan repeats part assignment {part_id}"
            )
        by_part[part_id] = assignment
    return assignments, by_part


def _mvinverse_groups(
    evidence: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    _schema(evidence, MVINVERSE_SCHEMA_VERSION, "MVInverse evidence")
    groups: dict[str, Mapping[str, Any]] = {}
    for index, raw in enumerate(
        _array(evidence.get("groups"), "MVInverse evidence groups")
    ):
        group = _object(raw, f"MVInverse group[{index}]")
        group_id = _text(group.get("group_id"), f"MVInverse group[{index}].group_id")
        if group_id in groups:
            raise AppearanceOptimizationError(
                f"MVInverse evidence repeats group {group_id}"
            )
        groups[group_id] = group
    return groups


def _auto_mvinverse_group(
    group: Mapping[str, Any],
) -> tuple[list[float], list[str]] | None:
    suggestion = group.get("suggestion")
    if not isinstance(suggestion, Mapping):
        return None
    if (
        suggestion.get("decision") != "auto"
        or suggestion.get("auto_parameter_eligible") is not True
    ):
        return None
    color = _rgb(
        suggestion.get("base_color_srgb"),
        f"MVInverse group {group.get('group_id')}.base_color_srgb",
    )
    contributing = [
        _text(
            value,
            f"MVInverse group {group.get('group_id')}.contributing_view_ids",
        )
        for value in _array(
            group.get("contributing_view_ids"),
            f"MVInverse group {group.get('group_id')}.contributing_view_ids",
        )
    ]
    if len(contributing) < 2 or len(set(contributing)) != len(contributing):
        return None
    return color, sorted(contributing)


def _discover_shared_material_cohorts(
    *,
    final_plan: Mapping[str, Any],
    mvinverse_evidence: Mapping[str, Any],
    policy: Mapping[str, float | int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    assignments, _by_part = _plan_assignments(final_plan)
    groups = _mvinverse_groups(mvinverse_evidence)
    eligible_groups = {
        group_id: (group, eligible)
        for group_id, group in groups.items()
        if (eligible := _auto_mvinverse_group(group)) is not None
    }
    matched: dict[tuple[str, str, str], list[tuple[str, Mapping[str, Any]]]] = {}
    ambiguous_part_ids: list[str] = []
    for assignment in assignments:
        part_id = str(assignment["part_id"])
        material_id = assignment.get("material_id")
        parameters = assignment.get("parameters")
        if not isinstance(material_id, str) or not isinstance(parameters, Mapping):
            continue
        profile = tuning_profile_for_material(material_id)
        if profile is None:
            continue
        candidate_matches: list[
            tuple[str, Mapping[str, Any], dict[str, Any], dict[str, Any]]
        ] = []
        for group_id, (group, _eligible) in eligible_groups.items():
            try:
                expected, audit = tune_selected_material_from_mvinverse(
                    group,
                    group_id=group_id,
                    material_id=material_id,
                )
            except ValueError:
                continue
            if all(
                name in parameters and _numbers_equal(parameters[name], expected_value)
                for name, expected_value in expected.items()
            ):
                candidate_matches.append((group_id, group, expected, audit))
        if len(candidate_matches) != 1:
            if len(candidate_matches) > 1:
                ambiguous_part_ids.append(part_id)
            continue
        group_id, group, expected, audit = candidate_matches[0]
        relevant = {name: copy.deepcopy(parameters[name]) for name in expected}
        key = (group_id, material_id, canonical_sha256(relevant))
        matched.setdefault(key, []).append((part_id, group))

    cohorts: list[dict[str, Any]] = []
    minimum_members = int(policy["minimum_shared_part_count"])
    for (group_id, material_id, parameter_sha), raw_members in sorted(matched.items()):
        if len(raw_members) < minimum_members:
            continue
        part_ids = sorted(part_id for part_id, _group in raw_members)
        group = raw_members[0][1]
        auto_group = _auto_mvinverse_group(group)
        assert auto_group is not None
        albedo_srgb, contributing_view_ids = auto_group
        profile = tuning_profile_for_material(material_id)
        assert profile is not None
        first_assignment = next(
            assignment
            for assignment in assignments
            if assignment.get("part_id") == part_ids[0]
        )
        parameters = _object(
            first_assignment.get("parameters"),
            f"shared cohort {group_id} parameters",
        )
        current_parameters = {
            name: copy.deepcopy(parameters[name])
            for name in (*profile.color_parameters, profile.roughness_parameter)
        }
        identity = {
            "canonical_group_id": group_id,
            "material_id": material_id,
            "part_ids": part_ids,
            "current_parameter_sha256": parameter_sha,
        }
        cohorts.append(
            {
                "cohort_id": f"SMC_{canonical_sha256(identity)[:20]}",
                "canonical_group_id": group_id,
                "material_id": material_id,
                "tuning_profile_id": profile.profile_id,
                "part_ids": part_ids,
                "part_count": len(part_ids),
                "color_parameter_names": sorted(profile.color_parameters),
                "roughness_parameter_name": profile.roughness_parameter,
                "current_parameters": current_parameters,
                "current_parameter_sha256": parameter_sha,
                "mvinverse_albedo_srgb": albedo_srgb,
                "mvinverse_albedo_linear": srgb_to_linear(albedo_srgb),
                "mvinverse_contributing_view_ids": contributing_view_ids,
            }
        )
    return cohorts, {
        "eligible_mvinverse_group_count": len(eligible_groups),
        "ambiguous_part_ids": sorted(ambiguous_part_ids),
    }


def _quality_views(
    quality_report: Mapping[str, Any],
) -> tuple[Mapping[str, Any], list[Mapping[str, Any]]]:
    _schema(quality_report, QUALITY_SCHEMA_VERSION, "quality report")
    thresholds = _object(quality_report.get("thresholds"), "quality report thresholds")
    views: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(
        _array(quality_report.get("views"), "quality report views")
    ):
        view = _object(raw, f"quality report view[{index}]")
        view_id = _text(
            view.get("reference_view_id"),
            f"quality report view[{index}].reference_view_id",
        )
        if view_id in seen:
            raise AppearanceOptimizationError(
                f"quality report repeats reference view {view_id}"
            )
        seen.add(view_id)
        views.append(view)
    return thresholds, views


def _palette_view_group_maps(
    palette_fusion: Mapping[str, Any] | None,
) -> dict[str, dict[str, str]]:
    if palette_fusion is None:
        return {}
    _schema(palette_fusion, PALETTE_FUSION_SCHEMA_VERSION, "palette fusion")
    raw_maps = _object(
        palette_fusion.get("view_group_id_maps"),
        "palette fusion view_group_id_maps",
    )
    output: dict[str, dict[str, str]] = {}
    for raw_view_id, raw_mapping in raw_maps.items():
        view_id = _text(raw_view_id, "palette fusion view ID")
        mapping = _object(raw_mapping, f"palette fusion mapping for view {view_id}")
        translated: dict[str, str] = {}
        for raw_local, raw_canonical in mapping.items():
            local = _text(raw_local, f"palette fusion {view_id} local group")
            canonical = _text(
                raw_canonical, f"palette fusion {view_id} canonical group"
            )
            translated[local] = canonical
        output[view_id] = translated
    return output


def _dominant_raw_observations(
    *,
    quality_report: Mapping[str, Any],
    canonical_group_id: str,
    policy: Mapping[str, float | int],
    view_group_id_maps: Mapping[str, Mapping[str, str]] | None = None,
) -> list[dict[str, Any]]:
    thresholds, views = _quality_views(quality_report)
    quality_alignment_floor = thresholds.get("strong_alignment_score")
    alignment_floor = max(
        float(policy["minimum_alignment_score"]),
        _unit(
            quality_alignment_floor,
            "quality thresholds.strong_alignment_score",
        ),
    )
    records: list[dict[str, Any]] = []
    for view in views:
        material_color = view.get("material_color")
        alignment = view.get("alignment")
        render_view_id = view.get("render_view_id")
        if (
            not isinstance(material_color, Mapping)
            or not isinstance(alignment, Mapping)
            or not isinstance(render_view_id, str)
            or _unit(alignment.get("score"), "quality alignment score")
            < alignment_floor
        ):
            continue
        group_recall = material_color.get("trusted_evidence_group_recall")
        if not isinstance(group_recall, Mapping):
            continue
        raw_groups = _array(
            group_recall.get("groups"), "quality trusted evidence groups"
        )
        groups = [_object(raw, "quality trusted evidence group") for raw in raw_groups]
        view_id = str(view["reference_view_id"])
        local_to_canonical = (
            view_group_id_maps.get(view_id, {})
            if view_group_id_maps is not None
            else {}
        )

        def canonical_id(group: Mapping[str, Any]) -> Any:
            local_id = group.get("group_id")
            return local_to_canonical.get(local_id, local_id)

        matching = [
            group for group in groups if canonical_id(group) == canonical_group_id
        ]
        if len(matching) > 1:
            raise AppearanceOptimizationError(
                f"quality view {view['reference_view_id']} repeats group "
                f"{canonical_group_id}"
            )
        if not matching:
            continue
        shares = [
            _unit(
                group.get("reference_group_share"),
                "quality group reference share",
            )
            for group in groups
        ]
        target_share = _unit(
            matching[0].get("reference_group_share"),
            "target quality group reference share",
        )
        if target_share < float(
            policy["minimum_dominant_reference_group_share"]
        ) or target_share != max(shares):
            continue
        reference_distribution = _object(
            material_color.get("reference_distribution"),
            "quality reference distribution",
        )
        render_distribution = _object(
            material_color.get("render_distribution"),
            "quality render distribution",
        )
        reference_value = _positive(
            reference_distribution.get("median_value"),
            "quality reference median value",
        )
        render_value = _positive(
            render_distribution.get("median_value"),
            "quality render median value",
        )
        records.append(
            {
                "reference_view_id": str(view["reference_view_id"]),
                "render_view_id": render_view_id,
                "alignment_score": float(alignment["score"]),
                "reference_group_share": target_share,
                "reference_median_value": reference_value,
                "render_median_value": render_value,
                "raw_gain_stops": math.log2(reference_value / render_value),
            }
        )
    return sorted(records, key=lambda item: item["reference_view_id"])


def _sample(
    value: Any,
    label: str,
    *,
    minimum_pixels: int,
) -> dict[str, Any]:
    record = _object(value, label)
    pixels = _integer(
        record.get("sampled_pixels"), f"{label}.sampled_pixels", minimum=minimum_pixels
    )
    rgb = _rgb(record.get("median_linear_rgb"), f"{label}.median_linear_rgb")
    luminance = _positive(_luminance(rgb), f"{label} luminance")
    return {
        "sampled_pixels": pixels,
        "median_linear_rgb": rgb,
        "linear_luminance": luminance,
    }


def _normalized_observations(
    *,
    quality_report: Mapping[str, Any],
    canonical_group_id: str,
    mvinverse_albedo_linear: Sequence[float],
    policy: Mapping[str, float | int],
    eligible_raw_views: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    _thresholds, views = _quality_views(quality_report)
    raw_by_view = {
        str(record["reference_view_id"]): record for record in eligible_raw_views
    }
    records: list[dict[str, Any]] = []
    minimum_pixels = int(policy["minimum_sample_pixels"])
    for view in views:
        view_id = str(view["reference_view_id"])
        if view_id not in raw_by_view:
            continue
        material_color = view.get("material_color")
        if not isinstance(material_color, Mapping):
            continue
        statistics_document = material_color.get("lighting_normalized_groups")
        if statistics_document is None:
            continue
        statistics_mapping = _object(
            statistics_document,
            f"quality view {view_id} lighting statistics",
        )
        _schema(
            statistics_mapping,
            LIGHTING_STATISTICS_SCHEMA_VERSION,
            f"quality view {view_id} lighting statistics",
        )
        matching: list[Mapping[str, Any]] = []
        for index, raw_group in enumerate(
            _array(
                statistics_mapping.get("groups"),
                f"quality view {view_id} lighting statistic groups",
            )
        ):
            group = _object(
                raw_group,
                f"quality view {view_id} lighting statistic group[{index}]",
            )
            if group.get("canonical_group_id") == canonical_group_id:
                matching.append(group)
        if len(matching) > 1:
            raise AppearanceOptimizationError(
                f"quality view {view_id} repeats lighting group {canonical_group_id}"
            )
        if not matching:
            continue
        group = matching[0]
        reference = _sample(
            group.get("reference"),
            f"quality view {view_id} target reference",
            minimum_pixels=minimum_pixels,
        )
        render = _sample(
            group.get("render"),
            f"quality view {view_id} target render",
            minimum_pixels=minimum_pixels,
        )
        anchor = _object(
            group.get("neutral_anchor"),
            f"quality view {view_id} neutral anchor",
        )
        anchor_group_ids = _sorted_unique_texts(
            anchor.get("canonical_group_ids"),
            f"quality view {view_id} neutral anchor group IDs",
        )
        if not anchor_group_ids or canonical_group_id in anchor_group_ids:
            raise AppearanceOptimizationError(
                f"quality view {view_id} neutral anchor must use other groups"
            )
        anchor_reference = _sample(
            anchor.get("reference"),
            f"quality view {view_id} neutral reference",
            minimum_pixels=minimum_pixels,
        )
        anchor_render = _sample(
            anchor.get("render"),
            f"quality view {view_id} neutral render",
            minimum_pixels=minimum_pixels,
        )
        minimum_neutral = float(policy["minimum_neutral_linear_luminance"])
        if (
            anchor_reference["linear_luminance"] < minimum_neutral
            or anchor_render["linear_luminance"] < minimum_neutral
        ):
            raise AppearanceOptimizationError(
                f"quality view {view_id} neutral anchor is too dark"
            )
        neutral_reference_distance = _chromaticity_distance(
            anchor_reference["median_linear_rgb"], (1.0, 1.0, 1.0)
        )
        neutral_render_distance = _chromaticity_distance(
            anchor_render["median_linear_rgb"], (1.0, 1.0, 1.0)
        )
        if max(neutral_reference_distance, neutral_render_distance) > float(
            policy["maximum_neutral_chromaticity_distance"]
        ):
            raise AppearanceOptimizationError(
                f"quality view {view_id} anchor is not achromatic"
            )
        target_albedo_distance = _chromaticity_distance(
            reference["median_linear_rgb"], mvinverse_albedo_linear
        )
        render_albedo_distance = _chromaticity_distance(
            render["median_linear_rgb"], mvinverse_albedo_linear
        )
        if max(target_albedo_distance, render_albedo_distance) > float(
            policy["maximum_target_albedo_chromaticity_distance"]
        ):
            raise AppearanceOptimizationError(
                f"quality view {view_id} target chromaticity conflicts with "
                "MVInverse albedo"
            )
        reference_relative = (
            reference["linear_luminance"] / anchor_reference["linear_luminance"]
        )
        render_relative = render["linear_luminance"] / anchor_render["linear_luminance"]
        normalized_gain_stops = math.log2(reference_relative / render_relative)
        raw_record = raw_by_view[view_id]
        records.append(
            {
                "reference_view_id": view_id,
                "render_view_id": raw_record["render_view_id"],
                "alignment_score": raw_record["alignment_score"],
                "reference_group_share": raw_record["reference_group_share"],
                "reference_sampled_pixels": reference["sampled_pixels"],
                "render_sampled_pixels": render["sampled_pixels"],
                "neutral_reference_sampled_pixels": anchor_reference["sampled_pixels"],
                "neutral_render_sampled_pixels": anchor_render["sampled_pixels"],
                "neutral_anchor_group_ids": anchor_group_ids,
                "reference_target_linear_luminance": reference["linear_luminance"],
                "render_target_linear_luminance": render["linear_luminance"],
                "reference_neutral_linear_luminance": anchor_reference[
                    "linear_luminance"
                ],
                "render_neutral_linear_luminance": anchor_render["linear_luminance"],
                "raw_gain_stops": math.log2(
                    reference["linear_luminance"] / render["linear_luminance"]
                ),
                "normalized_gain_stops": normalized_gain_stops,
                "render_exposure_relative_to_reference_stops": math.log2(
                    anchor_render["linear_luminance"]
                    / anchor_reference["linear_luminance"]
                ),
                "reference_target_albedo_chromaticity_distance": (
                    target_albedo_distance
                ),
                "render_target_albedo_chromaticity_distance": (render_albedo_distance),
                "neutral_reference_chromaticity_distance": (neutral_reference_distance),
                "neutral_render_chromaticity_distance": neutral_render_distance,
            }
        )
    return sorted(records, key=lambda item: item["reference_view_id"])


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _part_color(part_id: str) -> tuple[int, int, int]:
    """Mirror the stable part-ID encoding used by the render pipeline."""

    number = int(part_id[1:]) if part_id[1:].isdigit() else sum(map(ord, part_id))
    hue = (number * 0.618033988749895) % 1.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.72, 0.95)
    return int(red * 255), int(green * 255), int(blue * 255)


def _image_data(image: Image.Image) -> Any:
    if hasattr(image, "get_flattened_data"):
        return image.get_flattened_data()
    return image.getdata()


def _open_rgb(path_value: Any, label: str) -> tuple[Path, Image.Image]:
    path = Path(_text(path_value, label)).expanduser().resolve(strict=True)
    try:
        with Image.open(path) as image:
            return path, ImageOps.exif_transpose(image).convert("RGB")
    except (OSError, UnidentifiedImageError) as exc:
        raise AppearanceOptimizationError(
            f"cannot decode {label} {path}: {exc}"
        ) from exc


def _weighted_median(values: Sequence[tuple[float, int]]) -> float:
    ordered = sorted(values, key=lambda item: item[0])
    total = sum(weight for _value, weight in ordered)
    threshold = total / 2.0
    cumulative = 0
    for value, weight in ordered:
        cumulative += weight
        if cumulative >= threshold:
            return value
    return ordered[-1][0]


def _reference_group_sample(
    *,
    quality_view: Mapping[str, Any],
    canonical_group_ids: set[str],
    local_to_canonical: Mapping[str, str],
    minimum_pixels: int,
) -> dict[str, Any] | None:
    reference = quality_view.get("reference")
    if not isinstance(reference, Mapping):
        return None
    trusted = reference.get("trusted_evidence")
    if not isinstance(trusted, Mapping) or trusted.get("usable") is not True:
        return None
    channels: list[list[tuple[float, int]]] = [[], [], []]
    sampled_pixels = 0
    contributing_local_group_ids: set[str] = set()
    for raw in _array(trusted.get("samples"), "trusted reference samples"):
        sample = _object(raw, "trusted reference sample")
        local_group_id = _text(
            sample.get("group_id"), "trusted reference local group ID"
        )
        canonical_group_id = local_to_canonical.get(local_group_id, local_group_id)
        if canonical_group_id not in canonical_group_ids:
            continue
        weight = _integer(
            sample.get("weight_pixels"),
            "trusted reference sample weight_pixels",
            minimum=1,
        )
        raw_rgb = _array(
            sample.get("representative_srgb"),
            "trusted reference representative_srgb",
        )
        if len(raw_rgb) != 3:
            raise AppearanceOptimizationError(
                "trusted reference representative_srgb needs three channels"
            )
        for index, raw_channel in enumerate(raw_rgb):
            if (
                isinstance(raw_channel, bool)
                or not isinstance(raw_channel, int)
                or not 0 <= raw_channel <= 255
            ):
                raise AppearanceOptimizationError(
                    "trusted reference representative_srgb must be uint8 RGB"
                )
            channels[index].append((raw_channel / 255.0, weight))
        sampled_pixels += weight
        contributing_local_group_ids.add(local_group_id)
    if sampled_pixels < minimum_pixels:
        return None
    median_srgb = [_weighted_median(channel) for channel in channels]
    return {
        "sampled_pixels": sampled_pixels,
        "median_linear_rgb": srgb_to_linear(median_srgb),
        "measurement_source": "trusted_palette_weighted_representatives",
        "local_group_ids": sorted(contributing_local_group_ids),
    }


def _render_part_sample(
    *,
    quality_view: Mapping[str, Any],
    part_ids: Sequence[str],
    minimum_pixels: int,
) -> dict[str, Any] | None:
    render = quality_view.get("render")
    if not isinstance(render, Mapping):
        return None
    rgb_path, rgb = _open_rgb(render.get("image"), "quality render image")
    ids_path, ids = _open_rgb(render.get("part_ids"), "quality part-ID image")
    if rgb.size != ids.size:
        raise AppearanceOptimizationError(
            "quality render and part-ID images have different sizes"
        )
    expected_rgb_sha = _text(render.get("image_sha256"), "quality render image_sha256")
    expected_ids_sha = _text(
        render.get("part_ids_sha256"), "quality render part_ids_sha256"
    )
    if _sha256_file(rgb_path) != expected_rgb_sha:
        raise AppearanceOptimizationError("quality render image hash is stale")
    if _sha256_file(ids_path) != expected_ids_sha:
        raise AppearanceOptimizationError("quality part-ID image hash is stale")
    allowed = {_part_color(part_id) for part_id in part_ids}
    selected = [
        pixel
        for pixel, part_pixel in zip(_image_data(rgb), _image_data(ids))
        if part_pixel in allowed
    ]
    if len(selected) < minimum_pixels:
        return None
    median_srgb = [
        statistics.median(pixel[index] for pixel in selected) / 255.0
        for index in range(3)
    ]
    return {
        "sampled_pixels": len(selected),
        "median_linear_rgb": srgb_to_linear(median_srgb),
        "measurement_source": "render_part_id_mask_channel_median",
        "part_ids": sorted(part_ids),
    }


def _canonical_palette_colors(
    palette_fusion: Mapping[str, Any],
) -> dict[str, str]:
    _schema(palette_fusion, PALETTE_FUSION_SCHEMA_VERSION, "palette fusion")
    canonical = _object(
        palette_fusion.get("canonical_palette"),
        "palette fusion canonical_palette",
    )
    groups: dict[str, str] = {}
    for index, raw in enumerate(
        _array(canonical.get("groups"), "canonical palette groups")
    ):
        group = _object(raw, f"canonical palette group[{index}]")
        group_id = _text(
            group.get("group_id"), f"canonical palette group[{index}].group_id"
        )
        if group_id in groups:
            raise AppearanceOptimizationError(
                f"canonical palette repeats group {group_id}"
            )
        groups[group_id] = _text(
            group.get("base_color"),
            f"canonical palette group {group_id}.base_color",
        ).lower()
    return groups


def _spatial_neutral_anchor_parts(
    *,
    spatial_report: Mapping[str, Any],
    neutral_group_ids: set[str],
    quality_render_views: Mapping[str, str],
    minimum_pixels: int,
) -> dict[str, dict[str, list[str]]]:
    _schema(spatial_report, SPATIAL_REPORT_SCHEMA_VERSION, "spatial report")
    output: dict[str, dict[str, list[str]]] = {}
    for index, raw_part in enumerate(
        _array(spatial_report.get("parts"), "spatial report parts")
    ):
        part = _object(raw_part, f"spatial part[{index}]")
        part_id = _text(part.get("part_id"), f"spatial part[{index}].part_id")
        for raw_observation in _array(
            part.get("observations"), f"spatial part {part_id}.observations"
        ):
            observation = _object(
                raw_observation, f"spatial part {part_id} observation"
            )
            reference_view_id = observation.get("reference_view_id")
            if (
                observation.get("classification") != "resolved"
                or not isinstance(reference_view_id, str)
                or observation.get("render_view_id")
                != quality_render_views.get(reference_view_id)
            ):
                continue
            projected = observation.get("projected_part_pixels")
            if (
                isinstance(projected, bool)
                or not isinstance(projected, int)
                or projected < minimum_pixels
            ):
                continue
            raw_scores = observation.get("group_scores")
            if (
                isinstance(raw_scores, (str, bytes))
                or not isinstance(raw_scores, Sequence)
                or not raw_scores
            ):
                continue
            winner = _object(raw_scores[0], "spatial neutral winner")
            canonical_group_id = winner.get("canonical_group_id")
            share = winner.get("color_share")
            margin = observation.get("color_margin")
            if (
                canonical_group_id not in neutral_group_ids
                or isinstance(share, bool)
                or not isinstance(share, (int, float))
                or float(share) < 0.60
                or isinstance(margin, bool)
                or not isinstance(margin, (int, float))
                or float(margin) < 0.30
            ):
                continue
            output.setdefault(reference_view_id, {}).setdefault(
                str(canonical_group_id), []
            ).append(part_id)
    return {
        view_id: {
            group_id: sorted(set(part_ids))
            for group_id, part_ids in sorted(groups.items())
        }
        for view_id, groups in sorted(output.items())
    }


def measure_lighting_normalized_group_statistics(
    *,
    final_plan: Mapping[str, Any],
    quality_report: Mapping[str, Any],
    mvinverse_evidence: Mapping[str, Any],
    palette_fusion: Mapping[str, Any],
    spatial_report: Mapping[str, Any],
    rendered_registry: Mapping[str, Any] | None = None,
    policy: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Add the minimal neutral-anchor statistics contract to a QA report.

    View identities come exclusively from the existing quality report.  This
    function never performs or changes view matching.
    """

    effective_policy = _effective_policy(policy)
    minimum_pixels = int(effective_policy["minimum_sample_pixels"])
    view_group_id_maps = _palette_view_group_maps(palette_fusion)
    palette_colors = _canonical_palette_colors(palette_fusion)
    neutral_group_ids = {
        group_id
        for group_id, color in palette_colors.items()
        if color in {"white", "gray", "grey", "silver"}
    }
    cohorts, discovery = _discover_shared_material_cohorts(
        final_plan=final_plan,
        mvinverse_evidence=mvinverse_evidence,
        policy=effective_policy,
    )
    _thresholds, quality_views = _quality_views(quality_report)
    quality_by_view = {str(view["reference_view_id"]): view for view in quality_views}
    quality_render_views = {
        view_id: _text(view.get("render_view_id"), f"quality view {view_id}.render")
        for view_id, view in quality_by_view.items()
        if isinstance(view.get("render_view_id"), str)
    }
    anchor_parts = _spatial_neutral_anchor_parts(
        spatial_report=spatial_report,
        neutral_group_ids=neutral_group_ids,
        quality_render_views=quality_render_views,
        minimum_pixels=minimum_pixels,
    )
    output = copy.deepcopy(dict(quality_report))
    output_by_view = {str(view["reference_view_id"]): view for view in output["views"]}
    measured: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for cohort in cohorts:
        group_id = str(cohort["canonical_group_id"])
        eligible = _dominant_raw_observations(
            quality_report=quality_report,
            canonical_group_id=group_id,
            policy=effective_policy,
            view_group_id_maps=view_group_id_maps,
        )
        for raw_observation in eligible:
            view_id = str(raw_observation["reference_view_id"])
            quality_view = quality_by_view[view_id]
            local_to_canonical = view_group_id_maps.get(view_id, {})
            available_anchor_groups = set(anchor_parts.get(view_id, {}))
            target_reference = _reference_group_sample(
                quality_view=quality_view,
                canonical_group_ids={group_id},
                local_to_canonical=local_to_canonical,
                minimum_pixels=minimum_pixels,
            )
            neutral_reference = _reference_group_sample(
                quality_view=quality_view,
                canonical_group_ids=available_anchor_groups,
                local_to_canonical=local_to_canonical,
                minimum_pixels=minimum_pixels,
            )
            target_render = _render_part_sample(
                quality_view=quality_view,
                part_ids=cohort["part_ids"],
                minimum_pixels=minimum_pixels,
            )
            neutral_part_ids = sorted(
                {
                    part_id
                    for anchor_group_id in available_anchor_groups
                    for part_id in anchor_parts[view_id][anchor_group_id]
                    if part_id not in set(cohort["part_ids"])
                }
            )
            neutral_render = (
                _render_part_sample(
                    quality_view=quality_view,
                    part_ids=neutral_part_ids,
                    minimum_pixels=minimum_pixels,
                )
                if neutral_part_ids
                else None
            )
            missing = [
                name
                for name, value in (
                    ("target_reference", target_reference),
                    ("target_render", target_render),
                    ("neutral_reference", neutral_reference),
                    ("neutral_render", neutral_render),
                )
                if value is None
            ]
            if missing:
                skipped.append(
                    {
                        "cohort_id": cohort["cohort_id"],
                        "canonical_group_id": group_id,
                        "reference_view_id": view_id,
                        "reason_code": "LIGHTING_STATISTICS_COMPONENT_MISSING",
                        "missing_components": missing,
                    }
                )
                continue
            assert target_reference is not None
            assert target_render is not None
            assert neutral_reference is not None
            assert neutral_render is not None
            statistic = {
                "canonical_group_id": group_id,
                "reference": target_reference,
                "render": target_render,
                "neutral_anchor": {
                    "canonical_group_ids": sorted(available_anchor_groups),
                    "reference": neutral_reference,
                    "render": neutral_render,
                },
            }
            output_view = output_by_view[view_id]
            material_color = output_view.get("material_color")
            if not isinstance(material_color, dict):
                raise AppearanceOptimizationError(
                    f"quality view {view_id} material_color is not mutable"
                )
            document = material_color.setdefault(
                "lighting_normalized_groups",
                {
                    "schema_version": LIGHTING_STATISTICS_SCHEMA_VERSION,
                    "groups": [],
                },
            )
            if not isinstance(document, dict):
                raise AppearanceOptimizationError(
                    f"quality view {view_id} lighting statistics is not mutable"
                )
            _schema(
                document,
                LIGHTING_STATISTICS_SCHEMA_VERSION,
                f"quality view {view_id} lighting statistics",
            )
            groups = document.get("groups")
            if not isinstance(groups, list):
                raise AppearanceOptimizationError(
                    f"quality view {view_id} lighting groups is not mutable"
                )
            groups[:] = [
                existing
                for existing in groups
                if not isinstance(existing, Mapping)
                or existing.get("canonical_group_id") != group_id
            ]
            groups.append(statistic)
            groups.sort(key=lambda item: str(item.get("canonical_group_id")))
            measured.append(
                {
                    "cohort_id": cohort["cohort_id"],
                    "canonical_group_id": group_id,
                    "reference_view_id": view_id,
                    "render_view_id": raw_observation["render_view_id"],
                    "neutral_anchor_group_ids": sorted(available_anchor_groups),
                    "target_reference_pixels": target_reference["sampled_pixels"],
                    "target_render_pixels": target_render["sampled_pixels"],
                    "neutral_reference_pixels": neutral_reference["sampled_pixels"],
                    "neutral_render_pixels": neutral_render["sampled_pixels"],
                }
            )
    report = {
        "schema_version": MEASUREMENT_REPORT_SCHEMA_VERSION,
        "mode": OPTIMIZATION_MODE,
        "input_hashes": {
            "final_plan_sha256": canonical_sha256(final_plan),
            "quality_report_sha256": canonical_sha256(quality_report),
            "mvinverse_evidence_sha256": canonical_sha256(mvinverse_evidence),
            "palette_fusion_sha256": canonical_sha256(palette_fusion),
            "spatial_report_sha256": canonical_sha256(spatial_report),
            "rendered_registry_sha256": (
                canonical_sha256(rendered_registry)
                if rendered_registry is not None
                else None
            ),
        },
        "output_quality_report_sha256": canonical_sha256(output),
        "policy": effective_policy,
        "summary": {
            "shared_cohort_count": len(cohorts),
            "measured_view_count": len(measured),
            "skipped_view_count": len(skipped),
            "neutral_canonical_group_ids": sorted(neutral_group_ids),
            **discovery,
        },
        "measured": measured,
        "skipped": skipped,
    }
    return output, report


def _suggest_parameters(
    cohort: Mapping[str, Any],
    *,
    adjustment_stops: float,
) -> dict[str, Any]:
    scale = 2.0**adjustment_stops
    current_parameters = _object(
        cohort.get("current_parameters"),
        f"cohort {cohort.get('cohort_id')} current parameters",
    )
    parameters: dict[str, Any] = {}
    for name in _sorted_unique_texts(
        cohort.get("color_parameter_names"),
        f"cohort {cohort.get('cohort_id')} color parameter names",
    ):
        current = _rgb(
            current_parameters.get(name),
            f"cohort {cohort.get('cohort_id')} parameter {name}",
        )
        candidate = [channel * scale for channel in current]
        if any(channel > 1.0 for channel in candidate):
            raise AppearanceOptimizationError(
                f"cohort {cohort.get('cohort_id')} bounded color adjustment "
                "would clip a linear channel"
            )
        parameters[name] = candidate
    return {
        "adjustment_stops": adjustment_stops,
        "linear_color_scale": scale,
        "authored_parameter_names": sorted(parameters),
        "parameters": parameters,
        "parameter_sha256": canonical_sha256(parameters),
    }


def build_shared_material_optimization_contract(
    *,
    final_plan: Mapping[str, Any],
    quality_report: Mapping[str, Any],
    mvinverse_evidence: Mapping[str, Any],
    rendered_registry: Mapping[str, Any] | None = None,
    palette_fusion: Mapping[str, Any] | None = None,
    policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one deterministic optimization contract from existing artifacts.

    Existing QA medians are sufficient to block unsafe shared-tint changes.
    Authoring is enabled only when every eligible view also contains
    ``material_color.lighting_normalized_groups`` with the v1 minimal field
    contract described in this module's docstring.
    """

    effective_policy = _effective_policy(policy)
    view_group_id_maps = _palette_view_group_maps(palette_fusion)
    cohorts, discovery = _discover_shared_material_cohorts(
        final_plan=final_plan,
        mvinverse_evidence=mvinverse_evidence,
        policy=effective_policy,
    )
    records: list[dict[str, Any]] = []
    minimum_views = int(effective_policy["minimum_trusted_view_count"])
    for cohort in cohorts:
        group_id = str(cohort["canonical_group_id"])
        raw = _dominant_raw_observations(
            quality_report=quality_report,
            canonical_group_id=group_id,
            policy=effective_policy,
            view_group_id_maps=view_group_id_maps,
        )
        normalized = _normalized_observations(
            quality_report=quality_report,
            canonical_group_id=group_id,
            mvinverse_albedo_linear=cohort["mvinverse_albedo_linear"],
            policy=effective_policy,
            eligible_raw_views=raw,
        )
        raw_summary = _gain_summary(raw, "raw_gain_stops") if raw else None
        normalized_summary = (
            _gain_summary(normalized, "normalized_gain_stops") if normalized else None
        )
        exact_normalized_coverage = len(raw) >= minimum_views and [
            record["reference_view_id"] for record in normalized
        ] == [record["reference_view_id"] for record in raw]
        decision = DECISION_INSUFFICIENT
        reasons: list[str] = []
        suggestion: dict[str, Any] | None = None
        if exact_normalized_coverage:
            assert normalized_summary is not None
            raw_normalized_summary = _gain_summary(normalized, "raw_gain_stops")
            exposure_summary = _gain_summary(
                normalized, "render_exposure_relative_to_reference_stops"
            )
            if float(normalized_summary["span_stops"]) > float(
                effective_policy["maximum_normalized_gain_span_stops"]
            ) or float(normalized_summary["mad_stops"]) > float(
                effective_policy["maximum_normalized_gain_mad_stops"]
            ):
                decision = DECISION_LIGHTING_INCONSISTENT
                reasons.extend(
                    [
                        "NORMALIZED_RESIDUAL_INCONSISTENT_ACROSS_VIEWS",
                        "SHARED_PARAMETER_CHANGE_BLOCKED",
                    ]
                )
            elif float(normalized_summary["objective_median_absolute_stops"]) < float(
                effective_policy["minimum_adjustment_stops"]
            ):
                decision = DECISION_PRESERVE
                reasons.append("LIGHTING_NORMALIZED_RESIDUAL_WITHIN_TOLERANCE")
            else:
                decision = DECISION_ADJUST
                adjustment = max(
                    -float(effective_policy["maximum_adjustment_stops"]),
                    min(
                        float(effective_policy["maximum_adjustment_stops"]),
                        float(normalized_summary["median_stops"]),
                    ),
                )
                suggestion = _suggest_parameters(cohort, adjustment_stops=adjustment)
                reasons.extend(
                    [
                        "MVINVERSE_ALBEDO_ANCHORED",
                        "NEUTRAL_ANCHOR_NORMALIZED",
                        "CROSS_VIEW_RESIDUAL_CONSISTENT",
                        "BOUNDED_SHARED_COLOR_STEP",
                    ]
                )
            if float(exposure_summary["span_stops"]) > float(
                effective_policy["maximum_unanchored_raw_gain_span_stops"]
            ):
                reasons.append("VIEW_LIGHTING_EXPOSURE_VARIES")
            diagnostics = {
                "source": "neutral_anchor_normalized",
                "raw_view_count": len(raw),
                "normalized_view_count": len(normalized),
                "raw_gain": raw_normalized_summary,
                "normalized_gain": normalized_summary,
                "view_exposure": exposure_summary,
                "views": normalized,
            }
        elif len(raw) >= minimum_views:
            assert raw_summary is not None
            if float(raw_summary["span_stops"]) > float(
                effective_policy["maximum_unanchored_raw_gain_span_stops"]
            ):
                decision = DECISION_LIGHTING_INCONSISTENT
                reasons.extend(
                    [
                        "VIEW_DEPENDENT_RAW_BRIGHTNESS_RESIDUAL",
                        "NEUTRAL_ANCHOR_STATISTICS_REQUIRED",
                        "SHARED_PARAMETER_CHANGE_BLOCKED",
                    ]
                )
            elif float(raw_summary["objective_median_absolute_stops"]) < float(
                effective_policy["minimum_adjustment_stops"]
            ):
                decision = DECISION_PRESERVE
                reasons.extend(
                    [
                        "RAW_BRIGHTNESS_RESIDUAL_WITHIN_TOLERANCE",
                        "NEUTRAL_ANCHOR_STATISTICS_NOT_NEEDED",
                    ]
                )
            else:
                reasons.extend(
                    [
                        "NEUTRAL_ANCHOR_STATISTICS_REQUIRED",
                        "RAW_RGB_CANNOT_AUTHOR_MATERIAL_PARAMETERS",
                    ]
                )
            diagnostics = {
                "source": "raw_dominant_group_proxy",
                "raw_view_count": len(raw),
                "normalized_view_count": len(normalized),
                "raw_gain": raw_summary,
                "normalized_gain": None,
                "view_exposure": None,
                "views": raw,
            }
        else:
            reasons.append("INSUFFICIENT_DOMINANT_TRUSTED_VIEWS")
            diagnostics = {
                "source": "insufficient",
                "raw_view_count": len(raw),
                "normalized_view_count": len(normalized),
                "raw_gain": raw_summary,
                "normalized_gain": normalized_summary,
                "view_exposure": None,
                "views": raw,
            }
        records.append(
            {
                **cohort,
                "decision": decision,
                "reason_codes": reasons,
                "diagnostics": diagnostics,
                "suggestion": suggestion,
            }
        )

    aggregate = _object(quality_report.get("aggregate"), "quality report aggregate")
    counts = {
        decision: sum(record["decision"] == decision for record in records)
        for decision in (
            DECISION_ADJUST,
            DECISION_PRESERVE,
            DECISION_LIGHTING_INCONSISTENT,
            DECISION_INSUFFICIENT,
        )
    }
    input_hashes = {
        "final_plan_sha256": canonical_sha256(final_plan),
        "quality_report_sha256": canonical_sha256(quality_report),
        "mvinverse_evidence_sha256": canonical_sha256(mvinverse_evidence),
        "rendered_registry_sha256": (
            canonical_sha256(rendered_registry)
            if rendered_registry is not None
            else None
        ),
        "palette_fusion_sha256": (
            canonical_sha256(palette_fusion) if palette_fusion is not None else None
        ),
    }
    return {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "mode": OPTIMIZATION_MODE,
        "input_hashes": input_hashes,
        "policy": effective_policy,
        "view_group_id_maps": view_group_id_maps,
        "baseline_quality": {
            "status": aggregate.get("status"),
            "comparable_view_count": _integer(
                aggregate.get("comparable_view_count"),
                "quality aggregate comparable_view_count",
            ),
            "failed_view_count": _integer(
                aggregate.get("failed_view_count"),
                "quality aggregate failed_view_count",
            ),
            "unscorable_view_count": _integer(
                aggregate.get("unscorable_view_count"),
                "quality aggregate unscorable_view_count",
            ),
        },
        "summary": {
            "shared_cohort_count": len(records),
            "adjustment_count": counts[DECISION_ADJUST],
            "preserve_count": counts[DECISION_PRESERVE],
            "lighting_inconsistent_count": counts[DECISION_LIGHTING_INCONSISTENT],
            "insufficient_evidence_count": counts[DECISION_INSUFFICIENT],
            **discovery,
        },
        "cohorts": records,
    }


def _contract_cohorts(
    contract: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    _schema(contract, CONTRACT_SCHEMA_VERSION, "optimization contract")
    if contract.get("mode") != OPTIMIZATION_MODE:
        raise AppearanceOptimizationError("optimization contract has unsupported mode")
    cohorts: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    part_owners: dict[str, str] = {}
    for index, raw in enumerate(
        _array(contract.get("cohorts"), "optimization contract cohorts")
    ):
        cohort = _object(raw, f"optimization cohort[{index}]")
        cohort_id = _text(cohort.get("cohort_id"), f"cohort[{index}].cohort_id")
        if cohort_id in seen:
            raise AppearanceOptimizationError(
                f"optimization contract repeats cohort {cohort_id}"
            )
        seen.add(cohort_id)
        part_ids = _sorted_unique_texts(
            cohort.get("part_ids"), f"cohort {cohort_id}.part_ids"
        )
        if len(part_ids) != _integer(
            cohort.get("part_count"), f"cohort {cohort_id}.part_count", minimum=1
        ):
            raise AppearanceOptimizationError(
                f"cohort {cohort_id} part count is inconsistent"
            )
        for part_id in part_ids:
            if part_id in part_owners:
                raise AppearanceOptimizationError(
                    f"part {part_id} occurs in multiple shared cohorts"
                )
            part_owners[part_id] = cohort_id
        decision = cohort.get("decision")
        if decision not in {
            DECISION_ADJUST,
            DECISION_PRESERVE,
            DECISION_LIGHTING_INCONSISTENT,
            DECISION_INSUFFICIENT,
        }:
            raise AppearanceOptimizationError(
                f"cohort {cohort_id} has invalid decision"
            )
        if (decision == DECISION_ADJUST) != isinstance(
            cohort.get("suggestion"), Mapping
        ):
            raise AppearanceOptimizationError(
                f"cohort {cohort_id} suggestion does not match its decision"
            )
        cohorts.append(cohort)
    return cohorts


def apply_shared_material_optimization(
    *,
    final_plan: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply every authorized shared adjustment atomically to a plan copy."""

    _assignments, by_part = _plan_assignments(final_plan)
    input_hashes = _object(
        contract.get("input_hashes"), "optimization contract input_hashes"
    )
    if input_hashes.get("final_plan_sha256") != canonical_sha256(final_plan):
        raise AppearanceOptimizationError(
            "optimization contract does not bind the supplied final plan"
        )
    cohorts = _contract_cohorts(contract)
    output = copy.deepcopy(dict(final_plan))
    output_assignments = {
        str(assignment["part_id"]): assignment for assignment in output["assignments"]
    }
    contract_sha = canonical_sha256(contract)
    changed_part_ids: list[str] = []
    applied_cohort_ids: list[str] = []
    for cohort in cohorts:
        if cohort["decision"] != DECISION_ADJUST:
            continue
        cohort_id = str(cohort["cohort_id"])
        material_id = _text(
            cohort.get("material_id"), f"cohort {cohort_id}.material_id"
        )
        current = _object(
            cohort.get("current_parameters"),
            f"cohort {cohort_id}.current_parameters",
        )
        suggestion = _object(cohort.get("suggestion"), f"cohort {cohort_id}.suggestion")
        suggested_parameters = _object(
            suggestion.get("parameters"),
            f"cohort {cohort_id}.suggestion.parameters",
        )
        authored_names = _sorted_unique_texts(
            suggestion.get("authored_parameter_names"),
            f"cohort {cohort_id}.suggestion.authored_parameter_names",
        )
        if authored_names != sorted(suggested_parameters):
            raise AppearanceOptimizationError(
                f"cohort {cohort_id} authored parameter set is inconsistent"
            )
        if suggestion.get("parameter_sha256") != canonical_sha256(suggested_parameters):
            raise AppearanceOptimizationError(
                f"cohort {cohort_id} suggested parameter hash is invalid"
            )
        for part_id in _sorted_unique_texts(
            cohort.get("part_ids"), f"cohort {cohort_id}.part_ids"
        ):
            source_assignment = by_part.get(part_id)
            assignment = output_assignments.get(part_id)
            if source_assignment is None or assignment is None:
                raise AppearanceOptimizationError(
                    f"cohort {cohort_id} references unknown part {part_id}"
                )
            if source_assignment.get("material_id") != material_id:
                raise AppearanceOptimizationError(
                    f"cohort {cohort_id} material changed for part {part_id}"
                )
            source_parameters = _object(
                source_assignment.get("parameters"),
                f"source assignment {part_id}.parameters",
            )
            for name, expected in current.items():
                if name not in source_parameters or not _numbers_equal(
                    source_parameters[name], expected
                ):
                    raise AppearanceOptimizationError(
                        f"cohort {cohort_id} source parameters diverge at "
                        f"{part_id}.{name}"
                    )
            candidate_parameters = _object(
                assignment.get("parameters"),
                f"candidate assignment {part_id}.parameters",
            )
            for name in authored_names:
                candidate_parameters[name] = copy.deepcopy(suggested_parameters[name])
            provenance = assignment.get("provenance")
            if provenance is None:
                provenance = {}
                assignment["provenance"] = provenance
            if not isinstance(provenance, dict):
                raise AppearanceOptimizationError(
                    f"assignment {part_id} provenance is not mutable"
                )
            provenance["shared_material_appearance_optimization"] = {
                "contract_sha256": contract_sha,
                "cohort_id": cohort_id,
                "canonical_group_id": cohort["canonical_group_id"],
                "reason_codes": list(cohort["reason_codes"]),
                "adjustment_stops": suggestion["adjustment_stops"],
                "linear_color_scale": suggestion["linear_color_scale"],
                "authored_parameter_names": authored_names,
            }
            changed_part_ids.append(part_id)
        applied_cohort_ids.append(cohort_id)
    changed_part_ids.sort()
    report = {
        "schema_version": APPLY_REPORT_SCHEMA_VERSION,
        "mode": OPTIMIZATION_MODE,
        "input_plan_sha256": canonical_sha256(final_plan),
        "contract_sha256": contract_sha,
        "output_plan_sha256": canonical_sha256(output),
        "applied_cohort_ids": sorted(applied_cohort_ids),
        "changed_part_ids": changed_part_ids,
        "changed_part_count": len(changed_part_ids),
    }
    return output, report


def validate_shared_material_optimization_result(
    *,
    source_plan: Mapping[str, Any],
    contract: Mapping[str, Any],
    candidate_plan: Mapping[str, Any],
    candidate_quality_report: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the exact candidate plan and its lighting-normalized re-render."""

    expected_plan, apply_report = apply_shared_material_optimization(
        final_plan=source_plan,
        contract=contract,
    )
    if canonical_sha256(candidate_plan) != canonical_sha256(expected_plan):
        raise AppearanceOptimizationError(
            "candidate plan is not the exact atomic contract application"
        )
    cohorts = _contract_cohorts(contract)
    policy = _effective_policy(
        _object(contract.get("policy"), "optimization contract policy")
    )
    raw_view_group_maps = _object(
        contract.get("view_group_id_maps"),
        "optimization contract view_group_id_maps",
    )
    view_group_id_maps = {
        _text(raw_view_id, "optimization view group map ID"): {
            _text(raw_local, "optimization local group ID"): _text(
                raw_canonical, "optimization canonical group ID"
            )
            for raw_local, raw_canonical in _object(
                raw_mapping, "optimization view group mapping"
            ).items()
        }
        for raw_view_id, raw_mapping in raw_view_group_maps.items()
    }
    baseline_quality = _object(
        contract.get("baseline_quality"), "optimization baseline quality"
    )
    _thresholds, _views = _quality_views(candidate_quality_report)
    candidate_aggregate = _object(
        candidate_quality_report.get("aggregate"),
        "candidate quality aggregate",
    )
    aggregate_regression = (
        _integer(
            candidate_aggregate.get("failed_view_count"),
            "candidate failed_view_count",
        )
        > _integer(
            baseline_quality.get("failed_view_count"),
            "baseline failed_view_count",
        )
        or _integer(
            candidate_aggregate.get("unscorable_view_count"),
            "candidate unscorable_view_count",
        )
        > _integer(
            baseline_quality.get("unscorable_view_count"),
            "baseline unscorable_view_count",
        )
        or _integer(
            candidate_aggregate.get("comparable_view_count"),
            "candidate comparable_view_count",
        )
        < _integer(
            baseline_quality.get("comparable_view_count"),
            "baseline comparable_view_count",
        )
    )
    validations: list[dict[str, Any]] = []
    for cohort in cohorts:
        if cohort["decision"] != DECISION_ADJUST:
            continue
        cohort_id = str(cohort["cohort_id"])
        diagnostics = _object(
            cohort.get("diagnostics"), f"cohort {cohort_id}.diagnostics"
        )
        baseline_views = [
            _object(raw, f"cohort {cohort_id} baseline view")
            for raw in _array(
                diagnostics.get("views"), f"cohort {cohort_id} baseline views"
            )
        ]
        if diagnostics.get("source") != "neutral_anchor_normalized":
            raise AppearanceOptimizationError(
                f"adjustable cohort {cohort_id} lacks normalized evidence"
            )
        candidate_raw = _dominant_raw_observations(
            quality_report=candidate_quality_report,
            canonical_group_id=str(cohort["canonical_group_id"]),
            policy=policy,
            view_group_id_maps=view_group_id_maps,
        )
        candidate_views = _normalized_observations(
            quality_report=candidate_quality_report,
            canonical_group_id=str(cohort["canonical_group_id"]),
            mvinverse_albedo_linear=_rgb(
                cohort.get("mvinverse_albedo_linear"),
                f"cohort {cohort_id}.mvinverse_albedo_linear",
            ),
            policy=policy,
            eligible_raw_views=candidate_raw,
        )
        expected_view_ids = [
            str(record["reference_view_id"]) for record in baseline_views
        ]
        if [
            str(record["reference_view_id"]) for record in candidate_views
        ] != expected_view_ids:
            raise AppearanceOptimizationError(
                f"candidate cohort {cohort_id} does not exactly cover baseline views"
            )
        baseline_by_view = {
            str(record["reference_view_id"]): record for record in baseline_views
        }
        for candidate in candidate_views:
            baseline = baseline_by_view[str(candidate["reference_view_id"])]
            if (
                candidate["render_view_id"] != baseline["render_view_id"]
                or candidate["neutral_anchor_group_ids"]
                != baseline["neutral_anchor_group_ids"]
            ):
                raise AppearanceOptimizationError(
                    f"candidate cohort {cohort_id} changed its view/anchor identity"
                )
        baseline_summary = _gain_summary(baseline_views, "normalized_gain_stops")
        candidate_summary = _gain_summary(candidate_views, "normalized_gain_stops")
        baseline_objective = float(baseline_summary["objective_median_absolute_stops"])
        candidate_objective = float(
            candidate_summary["objective_median_absolute_stops"]
        )
        absolute_improvement = baseline_objective - candidate_objective
        relative_improvement = (
            absolute_improvement / baseline_objective
            if baseline_objective > 0.0
            else 0.0
        )
        view_regressions = [
            str(candidate["reference_view_id"])
            for candidate in candidate_views
            if abs(float(candidate["normalized_gain_stops"]))
            > abs(
                float(
                    baseline_by_view[str(candidate["reference_view_id"])][
                        "normalized_gain_stops"
                    ]
                )
            )
            + float(policy["maximum_view_regression_stops"])
        ]
        reasons: list[str] = []
        if aggregate_regression:
            reasons.append("AGGREGATE_QUALITY_REGRESSION")
        if absolute_improvement < float(
            policy["minimum_absolute_objective_improvement_stops"]
        ):
            reasons.append("ABSOLUTE_OBJECTIVE_IMPROVEMENT_BELOW_FLOOR")
        if relative_improvement < float(
            policy["minimum_relative_objective_improvement"]
        ):
            reasons.append("RELATIVE_OBJECTIVE_IMPROVEMENT_BELOW_FLOOR")
        if candidate_objective > float(
            policy["maximum_candidate_residual_median_stops"]
        ):
            reasons.append("CANDIDATE_RESIDUAL_ABOVE_CEILING")
        if float(candidate_summary["span_stops"]) > float(
            baseline_summary["span_stops"]
        ) + float(policy["maximum_candidate_span_regression_stops"]):
            reasons.append("CROSS_VIEW_RESIDUAL_SPAN_REGRESSION")
        if view_regressions:
            reasons.append("PER_VIEW_NORMALIZED_RESIDUAL_REGRESSION")
        validations.append(
            {
                "cohort_id": cohort_id,
                "canonical_group_id": cohort["canonical_group_id"],
                "part_ids": list(cohort["part_ids"]),
                "status": "PASS" if not reasons else "FAIL_CLOSED",
                "reason_codes": reasons,
                "baseline_objective_stops": baseline_objective,
                "candidate_objective_stops": candidate_objective,
                "absolute_improvement_stops": absolute_improvement,
                "relative_improvement": relative_improvement,
                "baseline_gain": baseline_summary,
                "candidate_gain": candidate_summary,
                "regressed_view_ids": view_regressions,
            }
        )
    if not validations:
        status = "NOT_APPLICABLE"
    elif all(record["status"] == "PASS" for record in validations):
        status = "PASS"
    else:
        status = "FAIL_CLOSED"
    return {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "mode": OPTIMIZATION_MODE,
        "status": status,
        "input_hashes": {
            "source_plan_sha256": canonical_sha256(source_plan),
            "contract_sha256": canonical_sha256(contract),
            "candidate_plan_sha256": canonical_sha256(candidate_plan),
            "candidate_quality_report_sha256": canonical_sha256(
                candidate_quality_report
            ),
        },
        "apply_report": apply_report,
        "aggregate_quality_regression": aggregate_regression,
        "validated_cohort_count": len(validations),
        "cohorts": validations,
    }


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.expanduser().resolve(strict=True).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise AppearanceOptimizationError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise AppearanceOptimizationError(f"{label} must contain an object")
    return value


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    output = path.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build/apply/validate bounded shared-material optimization"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build")
    build.add_argument("--final-plan", type=Path, required=True)
    build.add_argument("--quality-report", type=Path, required=True)
    build.add_argument("--mvinverse-evidence", type=Path, required=True)
    build.add_argument("--rendered-registry", type=Path)
    build.add_argument("--palette-fusion", type=Path)
    build.add_argument("--output", type=Path, required=True)

    measure = subparsers.add_parser("measure")
    measure.add_argument("--final-plan", type=Path, required=True)
    measure.add_argument("--quality-report", type=Path, required=True)
    measure.add_argument("--mvinverse-evidence", type=Path, required=True)
    measure.add_argument("--palette-fusion", type=Path, required=True)
    measure.add_argument("--spatial-report", type=Path, required=True)
    measure.add_argument("--rendered-registry", type=Path)
    measure.add_argument("--output-quality-report", type=Path, required=True)
    measure.add_argument("--output-report", type=Path, required=True)

    apply = subparsers.add_parser("apply")
    apply.add_argument("--final-plan", type=Path, required=True)
    apply.add_argument("--contract", type=Path, required=True)
    apply.add_argument("--output-plan", type=Path, required=True)
    apply.add_argument("--output-report", type=Path, required=True)

    validate = subparsers.add_parser("validate")
    validate.add_argument("--source-plan", type=Path, required=True)
    validate.add_argument("--contract", type=Path, required=True)
    validate.add_argument("--candidate-plan", type=Path, required=True)
    validate.add_argument("--candidate-quality-report", type=Path, required=True)
    validate.add_argument("--output", type=Path, required=True)
    validate.add_argument("--require-pass", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "build":
            rendered_registry = (
                _read_json(args.rendered_registry, "rendered registry")
                if args.rendered_registry is not None
                else None
            )
            palette_fusion = (
                _read_json(args.palette_fusion, "palette fusion")
                if args.palette_fusion is not None
                else None
            )
            result = build_shared_material_optimization_contract(
                final_plan=_read_json(args.final_plan, "final plan"),
                quality_report=_read_json(args.quality_report, "quality report"),
                mvinverse_evidence=_read_json(
                    args.mvinverse_evidence, "MVInverse evidence"
                ),
                rendered_registry=rendered_registry,
                palette_fusion=palette_fusion,
            )
            _write_json_atomic(args.output, result)
            return EXIT_SUCCESS
        if args.command == "measure":
            rendered_registry = (
                _read_json(args.rendered_registry, "rendered registry")
                if args.rendered_registry is not None
                else None
            )
            quality, report = measure_lighting_normalized_group_statistics(
                final_plan=_read_json(args.final_plan, "final plan"),
                quality_report=_read_json(args.quality_report, "quality report"),
                mvinverse_evidence=_read_json(
                    args.mvinverse_evidence, "MVInverse evidence"
                ),
                palette_fusion=_read_json(args.palette_fusion, "palette fusion"),
                spatial_report=_read_json(args.spatial_report, "spatial report"),
                rendered_registry=rendered_registry,
            )
            _write_json_atomic(args.output_quality_report, quality)
            _write_json_atomic(args.output_report, report)
            return EXIT_SUCCESS
        if args.command == "apply":
            plan, report = apply_shared_material_optimization(
                final_plan=_read_json(args.final_plan, "final plan"),
                contract=_read_json(args.contract, "optimization contract"),
            )
            _write_json_atomic(args.output_plan, plan)
            _write_json_atomic(args.output_report, report)
            return EXIT_SUCCESS
        result = validate_shared_material_optimization_result(
            source_plan=_read_json(args.source_plan, "source plan"),
            contract=_read_json(args.contract, "optimization contract"),
            candidate_plan=_read_json(args.candidate_plan, "candidate plan"),
            candidate_quality_report=_read_json(
                args.candidate_quality_report, "candidate quality report"
            ),
        )
        _write_json_atomic(args.output, result)
        if args.require_pass and result["status"] != "PASS":
            return EXIT_REQUIRE_PASS_FAILED
        return EXIT_SUCCESS
    except AppearanceOptimizationError as exc:
        print(f"appearance optimization error: {exc}")
        return EXIT_INPUT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "APPLY_REPORT_SCHEMA_VERSION",
    "CONTRACT_SCHEMA_VERSION",
    "DEFAULT_POLICY",
    "LIGHTING_STATISTICS_SCHEMA_VERSION",
    "MEASUREMENT_REPORT_SCHEMA_VERSION",
    "VALIDATION_SCHEMA_VERSION",
    "AppearanceOptimizationError",
    "apply_shared_material_optimization",
    "build_shared_material_optimization_contract",
    "main",
    "measure_lighting_normalized_group_statistics",
    "validate_shared_material_optimization_result",
]
