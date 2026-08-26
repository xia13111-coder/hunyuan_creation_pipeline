"""Resolve final visual-material QA without disguising geometry/pose limits.

This module is deliberately separate from material repair.  It never changes a
material plan, relaxes a comparator threshold, or turns an unverified colour
count into spatial ownership.  A regular comparator PASS remains PASS.

For a non-PASS report, a limitation is accepted only when hash-bound evidence
proves one of three cases:

* a small, trusted, multiview chromatic reference group is missing;
* a rare repeated source-visual cohort is independently corroborated by the
  policy audit, safe for uniform material use, and either exactly invisible
  or visibly smaller than the reference in the target pose; or
* the selected material is bound to the same canonical group, is already
  delivered in at least two other reference poses, and is partly delivered in
  the target pose, while the remaining deficit is bounded to a weak
  camera/configuration correspondence.

The resolver never invents spatial ownership from global colour counts and
never changes the comparator's PASS thresholds.  The separate limited-pass
floor only permits a small, explicitly enumerated residual after two complete
reference views already pass.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..evidence.color_semantics import (
    evidence_color_labels,
    fusion_color_label,
)
from .policy_exact_cover import (
    CORROBORATED_SOURCE_MDL_TIER,
)
from ..usd.material_common import (
    SOURCE_VISUAL_PRESERVE_ACTION,
    canonical_sha256,
    validate_source_visual_preserve,
)


PLAN_SCHEMA_VERSION = "1.0"
POLICY_AUDIT_SCHEMA_VERSION = "qwen-policy-exact-cover-report/v1"
QUALITY_SCHEMA_VERSION = "qwen-reference-render-comparison/v1"
PALETTE_FUSION_SCHEMA_VERSION = "qwen-multiview-palette-fusion/v1"
CANONICAL_PALETTE_SCHEMA_VERSION = "qwen-canonical-material-palette/v1"
SPATIAL_REPORT_SCHEMA_VERSION = "qwen-spatial-mapping-audit/v1"
GEOMETRY_RISK_SCHEMA_VERSION = "qwen-geometry-uniform-material-risk/v1"
REGISTRY_SCHEMA_VERSION = "qwen-material-parts/v1"
REPORT_SCHEMA_VERSION = "qwen-visual-quality-resolution/v1"

PASS = "PASS"
FAIL_CLOSED = "FAIL_CLOSED"
LIMITED_PASS = "MATERIAL_ACCEPTED_WITH_GEOMETRY_POSE_LIMITATION"
LIMITATION_CLASSIFICATION = "NOT_OBSERVABLE_GEOMETRY_POSE"
LIMITATION_REASON = "POSE_OR_OCCLUSION_MISMATCH"
COVERAGE_CLASSIFICATION = "OBSERVABLE_GEOMETRY_COVERAGE_MISMATCH"
COVERAGE_REASON = "CAMERA_OR_ASSEMBLY_COVERAGE_MISMATCH"

MIN_ALIGNMENT_SCORE = 0.75
MIN_PROJECTION_IOU = 0.80
MIN_ECC_CORRELATION = 0.85
MIN_COVERAGE_PROJECTION_IOU = 0.70
MIN_COVERAGE_ECC_CORRELATION = 0.75
MIN_MAPPING_PREVIEW_SCORE = 0.60
MIN_MAPPING_PREVIEW_SILHOUETTE_IOU = 0.50
MIN_REFERENCE_EVIDENCE_PIXELS = 128
MAX_REFERENCE_GROUP_SHARE = 0.05
MAX_LIMITED_SHARE_PER_VIEW = 0.075
MIN_LIMITED_AGGREGATE_COLOR_SCORE = 0.60
MIN_REPEATED_CANDIDATE_COUNT = 4
MIN_OTHER_VISIBLE_RENDER_VIEWS = 2
MIN_SOURCE_COVERAGE_RECALL = 0.35
MIN_CROSS_VIEW_COVERAGE_RECALL = 0.10
MIN_CROSS_VIEW_DELIVERED_VIEWS = 2
MIN_OWNER_PROJECTED_PIXELS = 256
MIN_OWNER_COLOR_SHARE = 0.70
MIN_OWNER_COLOR_MARGIN = 0.60
MAX_TARGET_SHARE_IN_OWNER = 0.05
MIN_ACCEPTED_BOX_OWNER_OVERLAP = 0.50
_CHROMATIC_FAMILIES = frozenset(
    {"red", "orange", "yellow", "green", "blue", "pink"}
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PERTURBATION_OFFSETS = frozenset({(-2, 0), (2, 0), (0, -2), (0, 2)})


class QualityResolutionError(ValueError):
    """Raised when an input document violates the resolution schema."""


def _object(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise QualityResolutionError(f"{label} must be an object")
    return value


def _array(value: Any, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise QualityResolutionError(f"{label} must be an array")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise QualityResolutionError(f"{label} must be a non-empty string")
    return value.strip()


def _unit(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise QualityResolutionError(
            f"{label} must be a finite number from 0 to 1"
        )
    return float(value)


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
    ):
        raise QualityResolutionError(
            f"{label} must be an integer >= {minimum}"
        )
    return value


def _schema(
    document: Mapping[str, Any],
    expected: str,
    label: str,
) -> None:
    if document.get("schema_version") != expected:
        raise QualityResolutionError(
            f"{label} has an unsupported schema_version"
        )


def _sorted_unique_texts(value: Any, label: str) -> list[str]:
    result = [
        _text(item, f"{label}[{index}]")
        for index, item in enumerate(_array(value, label))
    ]
    if result != sorted(set(result)):
        raise QualityResolutionError(
            f"{label} must be sorted and contain no duplicates"
        )
    return result


def _hashes(
    *,
    final_plan: Mapping[str, Any],
    policy_audit: Mapping[str, Any],
    quality_report: Mapping[str, Any],
    palette_fusion: Mapping[str, Any],
    spatial_report: Mapping[str, Any],
    geometry_risk: Mapping[str, Any],
    rendered_registry: Mapping[str, Any],
) -> dict[str, str]:
    return {
        "final_plan_sha256": canonical_sha256(final_plan),
        "policy_audit_sha256": canonical_sha256(policy_audit),
        "quality_report_sha256": canonical_sha256(quality_report),
        "palette_fusion_sha256": canonical_sha256(palette_fusion),
        "spatial_report_sha256": canonical_sha256(spatial_report),
        "geometry_risk_sha256": canonical_sha256(geometry_risk),
        "rendered_registry_sha256": canonical_sha256(rendered_registry),
    }


def _canonical_groups(
    palette_fusion: Mapping[str, Any],
) -> tuple[
    dict[str, Mapping[str, Any]],
    dict[str, dict[str, str]],
    set[str],
]:
    _schema(
        palette_fusion,
        PALETTE_FUSION_SCHEMA_VERSION,
        "palette fusion",
    )
    canonical = _object(
        palette_fusion.get("canonical_palette"),
        "palette_fusion.canonical_palette",
    )
    _schema(
        canonical,
        CANONICAL_PALETTE_SCHEMA_VERSION,
        "canonical palette",
    )
    groups: dict[str, Mapping[str, Any]] = {}
    qualifying_by_color: dict[str, list[str]] = {}
    for index, raw_group in enumerate(
        _array(canonical.get("groups"), "canonical palette groups")
    ):
        group = _object(raw_group, f"canonical palette group {index}")
        group_id = _text(group.get("group_id"), f"canonical group {index}.group_id")
        if group_id in groups:
            raise QualityResolutionError(
                f"canonical palette repeats group {group_id}"
            )
        source_views = _sorted_unique_texts(
            group.get("source_view_ids"),
            f"canonical group {group_id}.source_view_ids",
        )
        distinct_count = _integer(
            group.get("distinct_view_count"),
            f"canonical group {group_id}.distinct_view_count",
        )
        singleton = group.get("singleton")
        if not isinstance(singleton, bool) or distinct_count != len(source_views):
            raise QualityResolutionError(
                f"canonical group {group_id} has inconsistent support"
            )
        groups[group_id] = group
        color = fusion_color_label(
            _text(
                group.get("base_color"),
                f"canonical group {group_id}.base_color",
            )
        )
        if (
            not singleton
            and distinct_count >= 2
            and color in _CHROMATIC_FAMILIES
        ):
            qualifying_by_color.setdefault(color, []).append(group_id)
    unique_chromatic = {
        values[0]
        for values in qualifying_by_color.values()
        if len(values) == 1
    }

    raw_maps = _object(
        palette_fusion.get("view_group_id_maps"),
        "palette_fusion.view_group_id_maps",
    )
    view_maps: dict[str, dict[str, str]] = {}
    for raw_view_id, raw_mapping in raw_maps.items():
        view_id = _text(raw_view_id, "palette view ID")
        mapping = _object(raw_mapping, f"palette map {view_id}")
        normalized: dict[str, str] = {}
        for raw_local, raw_canonical in mapping.items():
            local_id = _text(raw_local, f"palette map {view_id} local ID")
            canonical_id = _text(
                raw_canonical,
                f"palette map {view_id}/{local_id} canonical ID",
            )
            if canonical_id not in groups:
                raise QualityResolutionError(
                    f"palette map {view_id}/{local_id} targets an unknown group"
                )
            normalized[local_id] = canonical_id
        view_maps[view_id] = normalized
    return groups, view_maps, unique_chromatic


def _registry(
    rendered_registry: Mapping[str, Any],
) -> tuple[
    dict[str, Mapping[str, Any]],
    dict[str, dict[str, int]],
    dict[str, str],
]:
    _schema(rendered_registry, REGISTRY_SCHEMA_VERSION, "rendered registry")
    parts: dict[str, Mapping[str, Any]] = {}
    for index, raw_part in enumerate(
        _array(rendered_registry.get("parts"), "rendered registry parts")
    ):
        part = _object(raw_part, f"rendered registry part {index}")
        part_id = _text(part.get("part_id"), f"rendered part {index}.part_id")
        if part_id in parts:
            raise QualityResolutionError(
                f"rendered registry repeats part {part_id}"
            )
        parts[part_id] = part
    declared_count = _integer(
        rendered_registry.get("part_count"),
        "rendered registry part_count",
    )
    if declared_count != len(parts):
        raise QualityResolutionError(
            "rendered registry part_count does not match parts"
        )

    render_set = _object(
        rendered_registry.get("render_set"),
        "rendered_registry.render_set",
    )
    visible_by_view: dict[str, dict[str, int]] = {}
    part_ids_sha_by_view: dict[str, str] = {}
    for index, raw_view in enumerate(
        _array(render_set.get("views"), "render set views")
    ):
        view = _object(raw_view, f"render set view {index}")
        view_id = _text(view.get("view_id"), f"render set view {index}.view_id")
        if view_id in visible_by_view:
            raise QualityResolutionError(f"render set repeats view {view_id}")
        part_ids_sha = view.get("part_ids_sha256")
        if part_ids_sha is None:
            # Existing registries bind the file through spatial_report.inputs.
            # A future producer may duplicate the digest directly on the view.
            part_ids_sha = ""
        elif not isinstance(part_ids_sha, str) or not _SHA256.fullmatch(
            part_ids_sha
        ):
            raise QualityResolutionError(
                f"render set view {view_id} has invalid part_ids_sha256"
            )
        visibility: dict[str, int] = {}
        for item_index, raw_visible in enumerate(
            _array(
                view.get("visible_parts"),
                f"render set view {view_id}.visible_parts",
            )
        ):
            visible = _object(
                raw_visible,
                f"render set view {view_id}.visible_parts[{item_index}]",
            )
            part_id = _text(
                visible.get("part_id"),
                f"render set view {view_id} visible part ID",
            )
            pixels = _integer(
                visible.get("pixels"),
                f"render set view {view_id}/{part_id}.pixels",
                minimum=1,
            )
            if part_id not in parts or part_id in visibility:
                raise QualityResolutionError(
                    f"render set view {view_id} has invalid part {part_id}"
                )
            visibility[part_id] = pixels
        visible_by_view[view_id] = visibility
        part_ids_sha_by_view[view_id] = part_ids_sha
    return parts, visible_by_view, part_ids_sha_by_view


def _assignments(
    final_plan: Mapping[str, Any],
    *,
    registry_ids: set[str],
) -> dict[str, Mapping[str, Any]]:
    _schema(final_plan, PLAN_SCHEMA_VERSION, "final plan")
    assignments: dict[str, Mapping[str, Any]] = {}
    for index, raw_assignment in enumerate(
        _array(final_plan.get("assignments"), "final plan assignments")
    ):
        assignment = _object(raw_assignment, f"final assignment {index}")
        part_id = _text(
            assignment.get("part_id"),
            f"final assignment {index}.part_id",
        )
        if part_id in assignments:
            raise QualityResolutionError(f"final plan repeats part {part_id}")
        assignments[part_id] = assignment
    if set(assignments) != registry_ids:
        raise QualityResolutionError(
            "final plan does not exactly cover the rendered registry"
        )
    return assignments


def _geometry_risks(
    geometry_risk: Mapping[str, Any],
) -> dict[str, bool]:
    _schema(
        geometry_risk,
        GEOMETRY_RISK_SCHEMA_VERSION,
        "geometry risk",
    )
    result: dict[str, bool] = {}
    for index, raw_part in enumerate(
        _array(geometry_risk.get("parts"), "geometry risk parts")
    ):
        part = _object(raw_part, f"geometry risk part {index}")
        part_id = _text(part.get("part_id"), f"geometry risk part {index}.part_id")
        if part_id in result:
            raise QualityResolutionError(
                f"geometry risk repeats part {part_id}"
            )
        risk = _object(
            part.get("risk"),
            f"geometry risk part {part_id}.risk",
        ).get("multi_material_risk")
        if not isinstance(risk, bool):
            raise QualityResolutionError(
                f"geometry risk part {part_id} has invalid risk flag"
            )
        result[part_id] = risk
    return result


def _spatial(
    spatial_report: Mapping[str, Any],
) -> tuple[
    Mapping[str, Any],
    dict[str, Mapping[str, Any]],
    dict[str, Mapping[str, Any]],
    dict[str, Mapping[str, Any]],
    dict[str, str],
]:
    _schema(
        spatial_report,
        SPATIAL_REPORT_SCHEMA_VERSION,
        "spatial report",
    )
    policy = _object(spatial_report.get("policy"), "spatial_report.policy")
    parts: dict[str, Mapping[str, Any]] = {}
    for index, raw_part in enumerate(
        _array(spatial_report.get("parts"), "spatial report parts")
    ):
        part = _object(raw_part, f"spatial part {index}")
        part_id = _text(part.get("part_id"), f"spatial part {index}.part_id")
        if part_id in parts:
            raise QualityResolutionError(f"spatial report repeats {part_id}")
        parts[part_id] = part

    references: dict[str, Mapping[str, Any]] = {}
    for index, raw_reference in enumerate(
        _array(
            spatial_report.get("reference_evidence"),
            "spatial reference evidence",
        )
    ):
        reference = _object(raw_reference, f"spatial reference {index}")
        view_id = _text(
            reference.get("view_id"),
            f"spatial reference {index}.view_id",
        )
        if view_id in references:
            raise QualityResolutionError(
                f"spatial report repeats reference {view_id}"
            )
        references[view_id] = reference

    alignments: dict[str, Mapping[str, Any]] = {}
    for index, raw_alignment in enumerate(
        _array(
            spatial_report.get("view_alignments"),
            "spatial view alignments",
        )
    ):
        alignment = _object(raw_alignment, f"spatial alignment {index}")
        view_id = _text(
            alignment.get("reference_view_id"),
            f"spatial alignment {index}.reference_view_id",
        )
        if view_id in alignments:
            raise QualityResolutionError(
                f"spatial report repeats alignment {view_id}"
            )
        alignments[view_id] = alignment

    part_ids_sha_by_view: dict[str, str] = {}
    inputs = _object(spatial_report.get("inputs"), "spatial_report.inputs")
    for index, raw_file in enumerate(
        _array(inputs.get("files"), "spatial input files")
    ):
        file_record = _object(raw_file, f"spatial input file {index}")
        label = _text(
            file_record.get("label"),
            f"spatial input file {index}.label",
        )
        if not label.startswith("part_ids:"):
            continue
        digest = _text(
            file_record.get("sha256"),
            f"spatial input file {label}.sha256",
        )
        if not _SHA256.fullmatch(digest):
            raise QualityResolutionError(
                f"spatial input file {label} has invalid sha256"
            )
        part_ids_sha_by_view[label.removeprefix("part_ids:")] = digest
    return policy, parts, references, alignments, part_ids_sha_by_view


def _quality_views(
    quality_report: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any], dict[str, Mapping[str, Any]]]:
    _schema(quality_report, QUALITY_SCHEMA_VERSION, "quality report")
    aggregate = _object(quality_report.get("aggregate"), "quality aggregate")
    thresholds = _object(quality_report.get("thresholds"), "quality thresholds")
    views: dict[str, Mapping[str, Any]] = {}
    for index, raw_view in enumerate(
        _array(quality_report.get("views"), "quality views")
    ):
        view = _object(raw_view, f"quality view {index}")
        view_id = _text(
            view.get("reference_view_id"),
            f"quality view {index}.reference_view_id",
        )
        if view_id in views:
            raise QualityResolutionError(f"quality report repeats view {view_id}")
        views[view_id] = view
    return aggregate, thresholds, views


def _policy_source_groups(
    policy_audit: Mapping[str, Any],
) -> tuple[dict[str, Mapping[str, Any]], set[str], list[str]]:
    _schema(
        policy_audit,
        POLICY_AUDIT_SCHEMA_VERSION,
        "policy audit",
    )
    source = policy_audit.get("corroborated_source_visual")
    if not isinstance(source, Mapping):
        return {}, set(), ["SOURCE_VISUAL_CORROBORATION_AUDIT_MISSING"]
    applied = set(
        _sorted_unique_texts(
            source.get("applied_part_ids"),
            "corroborated source applied_part_ids",
        )
    )
    groups: dict[str, Mapping[str, Any]] = {}
    for index, raw_group in enumerate(
        _array(source.get("groups"), "corroborated source groups")
    ):
        group = _object(raw_group, f"corroborated source group {index}")
        group_id = _text(
            group.get("group_id"),
            f"corroborated source group {index}.group_id",
        )
        if group_id in groups:
            raise QualityResolutionError(
                f"corroborated source repeats group {group_id}"
            )
        groups[group_id] = group
    return groups, applied, []


def _visibility(
    visible_by_view: Mapping[str, Mapping[str, int]],
    *,
    part_id: str,
    view_id: str,
) -> int:
    return int(visible_by_view.get(view_id, {}).get(part_id, 0))


def _candidate_geometry(
    *,
    group_id: str,
    target_render_view_id: str,
    source_groups: Mapping[str, Mapping[str, Any]],
    applied_source_ids: set[str],
    assignments: Mapping[str, Mapping[str, Any]],
    registry_parts: Mapping[str, Mapping[str, Any]],
    visible_by_view: Mapping[str, Mapping[str, int]],
    risks: Mapping[str, bool],
) -> tuple[dict[str, Any] | None, list[str]]:
    reasons: list[str] = []
    source_group = source_groups.get(group_id)
    if source_group is None:
        return None, ["CORROBORATED_SOURCE_GROUP_MISSING"]
    eligible_ids = set(
        _sorted_unique_texts(
            source_group.get("eligible_part_ids"),
            f"corroborated source group {group_id}.eligible_part_ids",
        )
    )
    source_signature_count = _integer(
        source_group.get("source_signature_count"),
        f"corroborated source group {group_id}.source_signature_count",
    )
    registry_fraction = _unit(
        source_group.get("registry_fraction"),
        f"corroborated source group {group_id}.registry_fraction",
    )
    if source_signature_count < MIN_REPEATED_CANDIDATE_COUNT:
        reasons.append("SOURCE_SIGNATURE_COUNT_BELOW_FLOOR")
    if registry_fraction > MAX_REFERENCE_GROUP_SHARE:
        reasons.append("SOURCE_SIGNATURE_TOO_COMMON")
    if not eligible_ids <= applied_source_ids:
        reasons.append("SOURCE_ELIGIBLE_PARTS_NOT_ALL_APPLIED")

    binding_errors: dict[str, str] = {}
    target_visibility: dict[str, int] = {}
    for part_id in sorted(eligible_ids):
        assignment = assignments.get(part_id)
        registry_part = registry_parts.get(part_id)
        if assignment is None or registry_part is None:
            binding_errors[part_id] = "PLAN_OR_REGISTRY_PART_MISSING"
            continue
        provenance = assignment.get("provenance")
        corroboration = (
            provenance.get("source_visual_corroboration")
            if isinstance(provenance, Mapping)
            else None
        )
        source_preserved = (
            assignment.get("apply_action") == SOURCE_VISUAL_PRESERVE_ACTION
        )
        source_represented_by_mdl = (
            assignment.get("apply_action") is None
            and isinstance(provenance, Mapping)
            and provenance.get("tier") == CORROBORATED_SOURCE_MDL_TIER
            and isinstance(corroboration, Mapping)
            and corroboration.get("confirmed_material_id")
            == assignment.get("material_id")
        )
        if (
            not isinstance(corroboration, Mapping)
            or corroboration.get("canonical_group_id") != group_id
            or not (source_preserved or source_represented_by_mdl)
        ):
            binding_errors[part_id] = "SOURCE_PRESERVE_CORROBORATION_MISMATCH"
            continue
        if source_preserved:
            try:
                validate_source_visual_preserve(
                    part_id,
                    assignment,
                    registry_part,
                )
            except ValueError as exc:
                binding_errors[part_id] = str(exc)
        target_visibility[part_id] = _visibility(
            visible_by_view,
            part_id=part_id,
            view_id=target_render_view_id,
        )
    if binding_errors:
        reasons.append("SOURCE_VISUAL_BINDING_VALIDATION_FAILED")

    zero_visible_cohorts: list[dict[str, Any]] = []
    visible_cohorts: list[dict[str, Any]] = []
    raw_cohorts = _array(
        source_group.get("geometry_cohorts"),
        f"corroborated source group {group_id}.geometry_cohorts",
    )
    for cohort_index, raw_cohort in enumerate(raw_cohorts):
        cohort = _object(
            raw_cohort,
            f"corroborated source group {group_id}.cohort[{cohort_index}]",
        )
        part_ids = _sorted_unique_texts(
            cohort.get("part_ids"),
            f"corroborated source group {group_id}.cohort[{cohort_index}].part_ids",
        )
        repeat_count = _integer(
            cohort.get("repeat_count"),
            f"corroborated source group {group_id}.cohort repeat_count",
        )
        signature_sha = _text(
            cohort.get("geometry_signature_sha256"),
            f"corroborated source group {group_id}.cohort signature",
        )
        if not _SHA256.fullmatch(signature_sha) or repeat_count != len(part_ids):
            raise QualityResolutionError(
                f"corroborated source group {group_id} has invalid geometry cohort"
            )
        if not set(part_ids) <= eligible_ids:
            raise QualityResolutionError(
                f"corroborated source group {group_id} cohort is not eligible"
            )
        other_views: dict[str, list[str]] = {}
        cohort_reasons: list[str] = []
        for part_id in part_ids:
            if risks.get(part_id) is not False:
                cohort_reasons.append("GEOMETRY_MULTI_MATERIAL_RISK")
            visible_views = sorted(
                view_id
                for view_id, visibility in visible_by_view.items()
                if view_id != target_render_view_id
                and int(visibility.get(part_id, 0)) > 0
            )
            other_views[part_id] = visible_views
            if len(visible_views) < MIN_OTHER_VISIBLE_RENDER_VIEWS:
                cohort_reasons.append(
                    "CANDIDATE_NOT_VISIBLE_IN_TWO_OTHER_RENDER_POSES"
                )
            if part_id in binding_errors:
                cohort_reasons.append("CANDIDATE_SOURCE_BINDING_INVALID")
        if repeat_count < MIN_REPEATED_CANDIDATE_COUNT:
            cohort_reasons.append("GEOMETRY_REPEAT_COUNT_BELOW_FLOOR")
        if not cohort_reasons:
            cohort_record = {
                "geometry_signature_sha256": signature_sha,
                "part_ids": part_ids,
                "repeat_count": repeat_count,
                "other_visible_render_view_ids_by_part": other_views,
            }
            cohort_target_pixels = [
                target_visibility.get(part_id, 0) for part_id in part_ids
            ]
            if all(pixels == 0 for pixels in cohort_target_pixels):
                zero_visible_cohorts.append(cohort_record)
            elif all(pixels > 0 for pixels in cohort_target_pixels):
                visible_cohorts.append(cohort_record)
    if not zero_visible_cohorts and not visible_cohorts:
        reasons.append("NO_SAFE_REPEATED_SOURCE_GEOMETRY_COHORT")
    zero_visible_ids = sorted(
        {
            part_id
            for cohort in zero_visible_cohorts
            for part_id in cohort["part_ids"]
        }
    )
    visible_ids = sorted(
        {
            part_id
            for cohort in visible_cohorts
            for part_id in cohort["part_ids"]
        }
    )
    record = {
        "source_visual_signature_sha256": source_group.get(
            "source_visual_signature_sha256"
        ),
        "source_signature_count": source_signature_count,
        "registry_fraction": registry_fraction,
        "eligible_part_ids": sorted(eligible_ids),
        "target_visible_pixels_by_part": dict(sorted(target_visibility.items())),
        "binding_errors": dict(sorted(binding_errors.items())),
        "safe_geometry_cohorts": zero_visible_cohorts,
        "safe_part_ids": zero_visible_ids,
        "visible_geometry_cohorts": visible_cohorts,
        "visible_part_ids": visible_ids,
    }
    return record, sorted(set(reasons))


def _observation_for_view(
    spatial_part: Mapping[str, Any],
    *,
    view_id: str,
) -> Mapping[str, Any] | None:
    found: Mapping[str, Any] | None = None
    for index, raw_observation in enumerate(
        _array(
            spatial_part.get("observations"),
            f"spatial part {spatial_part.get('part_id')}.observations",
        )
    ):
        observation = _object(
            raw_observation,
            f"spatial observation {spatial_part.get('part_id')}[{index}]",
        )
        if observation.get("reference_view_id") != view_id:
            continue
        if found is not None:
            raise QualityResolutionError(
                f"spatial part {spatial_part.get('part_id')} repeats view {view_id}"
            )
        found = observation
    return found


def _score_for_group(
    scores: Any,
    *,
    group_id: str,
    label: str,
) -> tuple[Mapping[str, Any] | None, Mapping[str, Any] | None]:
    normalized = [
        _object(item, f"{label}[{index}]")
        for index, item in enumerate(_array(scores, label))
    ]
    winner = normalized[0] if normalized else None
    matches = [
        item for item in normalized if item.get("canonical_group_id") == group_id
    ]
    if len(matches) > 1:
        raise QualityResolutionError(
            f"{label} repeats canonical group {group_id}"
        )
    return winner, matches[0] if matches else None


def _accepted_box_overlap(
    observation: Mapping[str, Any],
    *,
    view_id: str,
    local_group_id: str,
    canonical_group_id: str,
    evidence_pixels: int,
    accepted_palette_evidence: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any] | None, list[str]]:
    raw_records = observation.get("accepted_evidence_box_overlaps")
    if raw_records is None:
        return None, ["ACCEPTED_BOX_OWNER_OVERLAP_AUDIT_MISSING"]
    records = [
        _object(item, f"accepted box overlap {index}")
        for index, item in enumerate(
            _array(raw_records, "accepted box group overlaps")
        )
        if isinstance(item, Mapping)
        and item.get("local_group_id") == local_group_id
        and item.get("canonical_group_id") == canonical_group_id
    ]
    if len(records) != 1:
        return None, ["ACCEPTED_BOX_OWNER_OVERLAP_NOT_UNIQUE"]
    record = records[0]
    accepted_pixels = _integer(
        record.get("evidence_pixel_count"),
        "accepted box overlap evidence_pixel_count",
        minimum=1,
    )
    overlap_pixels = _integer(
        record.get("projected_overlap_pixels"),
        "accepted box overlap projected_overlap_pixels",
    )
    overlap_share = _unit(
        record.get("projected_overlap_share"),
        "accepted box overlap projected_overlap_share",
    )
    evidence_audit_sha = _text(
        record.get("evidence_audit_sha256"),
        "accepted box overlap evidence_audit_sha256",
    )
    if not _SHA256.fullmatch(evidence_audit_sha):
        raise QualityResolutionError(
            "accepted box overlap has invalid evidence_audit_sha256"
        )
    reasons: list[str] = []
    evidence_records = [
        item
        for item in accepted_palette_evidence
        if item.get("local_group_id") == local_group_id
        and item.get("canonical_group_id") == canonical_group_id
    ]
    if len(evidence_records) != 1:
        reasons.append("ACCEPTED_PALETTE_EVIDENCE_NOT_UNIQUE")
        reference_record: Mapping[str, Any] | None = None
    else:
        reference_record = evidence_records[0]
        accepted_boxes = reference_record.get("accepted_boxes")
        base_color = reference_record.get("base_color")
        reference_count = reference_record.get("evidence_pixel_count")
        reference_audit_sha = reference_record.get("evidence_audit_sha256")
        if (
            reference_record.get("view_id") != view_id
            or not isinstance(base_color, str)
            or not isinstance(accepted_boxes, Sequence)
            or isinstance(accepted_boxes, (str, bytes))
            or isinstance(reference_count, bool)
            or not isinstance(reference_count, int)
            or reference_count < 1
            or not isinstance(reference_audit_sha, str)
            or not _SHA256.fullmatch(reference_audit_sha)
        ):
            reasons.append("ACCEPTED_PALETTE_EVIDENCE_INVALID")
        else:
            payload = {
                "view_id": view_id,
                "local_group_id": local_group_id,
                "canonical_group_id": canonical_group_id,
                "base_color": base_color,
                "accepted_boxes": list(accepted_boxes),
                "evidence_pixel_count": reference_count,
            }
            if canonical_sha256(payload) != reference_audit_sha:
                reasons.append("ACCEPTED_PALETTE_EVIDENCE_SHA_MISMATCH")
            if evidence_audit_sha != reference_audit_sha:
                reasons.append("ACCEPTED_BOX_OVERLAP_AUDIT_SHA_MISMATCH")
            if record.get("base_color") != base_color:
                reasons.append("ACCEPTED_BOX_OVERLAP_BASE_COLOR_MISMATCH")
            if accepted_pixels != reference_count:
                reasons.append("ACCEPTED_BOX_OVERLAP_REFERENCE_COUNT_MISMATCH")
    if accepted_pixels != evidence_pixels:
        reasons.append("ACCEPTED_BOX_OVERLAP_EVIDENCE_COUNT_MISMATCH")
    if overlap_pixels > accepted_pixels:
        reasons.append("ACCEPTED_BOX_OVERLAP_PIXELS_EXCEED_EVIDENCE")
    expected_share = round(overlap_pixels / accepted_pixels, 8)
    if overlap_share != expected_share:
        reasons.append("ACCEPTED_BOX_OVERLAP_SHARE_INCONSISTENT")
    if overlap_share < MIN_ACCEPTED_BOX_OWNER_OVERLAP:
        reasons.append("ACCEPTED_BOX_OWNER_OVERLAP_BELOW_FLOOR")
    return (
        {
            "base_color": record.get("base_color"),
            "evidence_pixel_count": accepted_pixels,
            "projected_overlap_pixels": overlap_pixels,
            "projected_overlap_share": overlap_share,
            "evidence_audit_sha256": evidence_audit_sha,
        },
        sorted(set(reasons)),
    )


def _owner(
    *,
    target_group_id: str,
    local_group_id: str,
    view_id: str,
    evidence_pixels: int,
    spatial_parts: Mapping[str, Mapping[str, Any]],
    assignments: Mapping[str, Mapping[str, Any]],
    candidate_part_ids: set[str],
    spatial_policy: Mapping[str, Any],
    accepted_palette_evidence: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any] | None, list[str], list[dict[str, Any]]]:
    diagnostic_floor = _integer(
        spatial_policy.get("minimum_diagnostic_visible_pixels"),
        "spatial minimum_diagnostic_visible_pixels",
        minimum=1,
    )
    owner_candidates: list[dict[str, Any]] = []
    visible_same_group: list[str] = []
    for part_id, spatial_part in sorted(spatial_parts.items()):
        observation = _observation_for_view(spatial_part, view_id=view_id)
        if observation is None:
            continue
        declared = observation.get("declared_visible_pixels")
        declared_pixels = (
            int(declared)
            if isinstance(declared, int) and not isinstance(declared, bool)
            else -1
        )
        projected = observation.get("projected_part_pixels")
        projected_pixels = (
            int(projected)
            if isinstance(projected, int) and not isinstance(projected, bool)
            else -1
        )
        if (
            declared_pixels >= diagnostic_floor
            and (
                observation.get("canonical_group_id") == target_group_id
                or (
                    isinstance(observation.get("small_part_diagnostic"), Mapping)
                    and observation["small_part_diagnostic"].get("status")
                    == "resolved"
                    and observation["small_part_diagnostic"].get(
                        "canonical_group_id"
                    )
                    == target_group_id
                )
            )
        ):
            visible_same_group.append(part_id)

        if (
            observation.get("classification") != "resolved"
            or not isinstance(observation.get("canonical_group_id"), str)
            or observation.get("canonical_group_id") == target_group_id
            or observation.get("registration_label_stable") is not True
            or observation.get("perturbation_label_stable") is not True
            or declared_pixels < MIN_OWNER_PROJECTED_PIXELS
            or projected_pixels < MIN_OWNER_PROJECTED_PIXELS
        ):
            continue
        owner_group_id = str(observation["canonical_group_id"])
        direct_winner, target_score = _score_for_group(
            observation.get("group_scores"),
            group_id=target_group_id,
            label=f"spatial owner {part_id} direct group_scores",
        )
        bbox_winner, _ = _score_for_group(
            observation.get("bbox_group_scores"),
            group_id=target_group_id,
            label=f"spatial owner {part_id} bbox group_scores",
        )
        if (
            direct_winner is None
            or bbox_winner is None
            or target_score is None
            or direct_winner.get("canonical_group_id") != owner_group_id
            or bbox_winner.get("canonical_group_id") != owner_group_id
            or observation.get("bbox_canonical_group_id") != owner_group_id
        ):
            continue
        direct_share = _unit(
            direct_winner.get("color_share"),
            f"spatial owner {part_id} direct share",
        )
        bbox_share = _unit(
            bbox_winner.get("color_share"),
            f"spatial owner {part_id} bbox share",
        )
        direct_margin = _unit(
            observation.get("color_margin"),
            f"spatial owner {part_id} direct margin",
        )
        bbox_margin = _unit(
            observation.get("bbox_color_margin"),
            f"spatial owner {part_id} bbox margin",
        )
        target_share = _unit(
            target_score.get("color_share"),
            f"spatial owner {part_id} target share",
        )
        target_matching_pixels = _integer(
            target_score.get("matching_pixels"),
            f"spatial owner {part_id} target matching_pixels",
        )
        perturbations = _array(
            observation.get("projection_perturbations"),
            f"spatial owner {part_id} perturbations",
        )
        perturbation_records: list[dict[str, Any]] = []
        offsets: set[tuple[int, int]] = set()
        perturbations_valid = len(perturbations) == 4
        for index, raw_perturbation in enumerate(perturbations):
            perturbation = _object(
                raw_perturbation,
                f"spatial owner {part_id} perturbation {index}",
            )
            offset = perturbation.get("offset_pixels")
            if (
                isinstance(offset, Sequence)
                and not isinstance(offset, (str, bytes))
                and len(offset) == 2
                and all(
                    isinstance(value, int) and not isinstance(value, bool)
                    for value in offset
                )
            ):
                normalized_offset = (int(offset[0]), int(offset[1]))
                offsets.add(normalized_offset)
            else:
                perturbations_valid = False
                normalized_offset = (0, 0)
            share = _unit(
                perturbation.get("best_color_share"),
                f"spatial owner {part_id} perturbation share",
            )
            margin = _unit(
                perturbation.get("color_margin"),
                f"spatial owner {part_id} perturbation margin",
            )
            if (
                perturbation.get("canonical_group_id") != owner_group_id
                or perturbation.get("diagnostic_canonical_group_id")
                != owner_group_id
                or share < MIN_OWNER_COLOR_SHARE
                or margin < MIN_OWNER_COLOR_MARGIN
            ):
                perturbations_valid = False
            perturbation_records.append(
                {
                    "offset_pixels": list(normalized_offset),
                    "canonical_group_id": perturbation.get(
                        "canonical_group_id"
                    ),
                    "diagnostic_canonical_group_id": perturbation.get(
                        "diagnostic_canonical_group_id"
                    ),
                    "best_color_share": share,
                    "color_margin": margin,
                }
            )
        perturbations_valid = (
            perturbations_valid and offsets == _PERTURBATION_OFFSETS
        )
        assignment = assignments.get(part_id)
        assignment_provenance = (
            assignment.get("provenance")
            if isinstance(assignment, Mapping)
            else None
        )
        final_group_id = (
            assignment_provenance.get("canonical_group_id")
            if isinstance(assignment_provenance, Mapping)
            else None
        )
        overlap, overlap_reasons = _accepted_box_overlap(
            observation,
            view_id=view_id,
            local_group_id=local_group_id,
            canonical_group_id=target_group_id,
            evidence_pixels=evidence_pixels,
            accepted_palette_evidence=accepted_palette_evidence,
        )
        reasons: list[str] = list(overlap_reasons)
        if direct_share < MIN_OWNER_COLOR_SHARE:
            reasons.append("OWNER_DIRECT_COLOR_SHARE_BELOW_FLOOR")
        if bbox_share < MIN_OWNER_COLOR_SHARE:
            reasons.append("OWNER_BBOX_COLOR_SHARE_BELOW_FLOOR")
        if direct_margin < MIN_OWNER_COLOR_MARGIN:
            reasons.append("OWNER_DIRECT_COLOR_MARGIN_BELOW_FLOOR")
        if bbox_margin < MIN_OWNER_COLOR_MARGIN:
            reasons.append("OWNER_BBOX_COLOR_MARGIN_BELOW_FLOOR")
        if target_share > MAX_TARGET_SHARE_IN_OWNER:
            reasons.append("TARGET_COLOR_SHARE_IN_OWNER_ABOVE_CEILING")
        if target_matching_pixels < diagnostic_floor:
            reasons.append("OWNER_TARGET_COLOR_PIXELS_BELOW_DIAGNOSTIC_FLOOR")
        if not perturbations_valid:
            reasons.append("OWNER_PERTURBATION_CONTRACT_FAILED")
        if final_group_id != owner_group_id:
            reasons.append("FINAL_PLAN_OWNER_GROUP_MISMATCH")
        if part_id in candidate_part_ids:
            reasons.append("OWNER_IS_TARGET_SOURCE_CANDIDATE")
        owner_candidates.append(
            {
                "part_id": part_id,
                "canonical_group_id": owner_group_id,
                "declared_visible_pixels": declared_pixels,
                "projected_part_pixels": projected_pixels,
                "direct_color_share": direct_share,
                "direct_color_margin": direct_margin,
                "bbox_color_share": bbox_share,
                "bbox_color_margin": bbox_margin,
                "target_matching_pixels": target_matching_pixels,
                "target_color_share": target_share,
                "perturbations": perturbation_records,
                "accepted_box_overlap": overlap,
                "eligible": not reasons,
                "reason_codes": sorted(set(reasons)),
            }
        )

    reasons: list[str] = []
    if visible_same_group:
        reasons.append("VISIBLE_TRUSTED_SAME_GROUP_PART_EXISTS")
    eligible = [item for item in owner_candidates if item["eligible"]]
    if len(eligible) != 1:
        reasons.append(
            "STABLE_ACCEPTED_BOX_FOREIGN_OWNER_NOT_UNIQUE"
            if eligible
            else "STABLE_ACCEPTED_BOX_FOREIGN_OWNER_MISSING"
        )
    selected = eligible[0] if len(eligible) == 1 else None
    return selected, sorted(set(reasons)), owner_candidates


def _reference_alignment(
    *,
    view: Mapping[str, Any],
    view_id: str,
    spatial_reference: Mapping[str, Any] | None,
    spatial_alignment: Mapping[str, Any] | None,
) -> tuple[dict[str, Any] | None, list[str]]:
    reasons: list[str] = []
    mapping = view.get("mapping")
    reference = view.get("reference")
    if not isinstance(mapping, Mapping):
        return None, ["QUALITY_VIEW_MAPPING_MISSING"]
    if not isinstance(reference, Mapping):
        return None, ["QUALITY_VIEW_REFERENCE_MISSING"]
    render_view_id = mapping.get("selected_render_view_id")
    mapping_reasons = mapping.get("reasons")
    if (
        not isinstance(render_view_id, str)
        or not render_view_id
        or not isinstance(mapping_reasons, Sequence)
        or isinstance(mapping_reasons, (str, bytes))
        or list(mapping_reasons)
    ):
        reasons.append("QUALITY_VIEW_MAPPING_NOT_TRUSTED")
    reference_sha = reference.get("image_sha256")
    if not isinstance(reference_sha, str) or not _SHA256.fullmatch(reference_sha):
        reasons.append("QUALITY_REFERENCE_SHA_INVALID")
    trusted_reference = reference.get("trusted_evidence")
    if (
        not isinstance(trusted_reference, Mapping)
        or trusted_reference.get("usable") is not True
        or trusted_reference.get("reasons") != []
    ):
        reasons.append("QUALITY_REFERENCE_PALETTE_NOT_TRUSTED")
    if spatial_reference is None:
        reasons.append("SPATIAL_REFERENCE_EVIDENCE_MISSING")
    else:
        if spatial_reference.get("alignment_trusted") is not True:
            reasons.append("SPATIAL_REFERENCE_ALIGNMENT_NOT_TRUSTED")
        if spatial_reference.get("raw_sha256") != reference_sha:
            reasons.append("QUALITY_SPATIAL_REFERENCE_SHA_MISMATCH")
        if (
            spatial_reference.get("selected_render_view_id") != render_view_id
            or spatial_reference.get("pose_cluster_id") != render_view_id
        ):
            reasons.append("QUALITY_SPATIAL_RENDER_POSE_MISMATCH")
    if spatial_alignment is None:
        reasons.append("SPATIAL_ALIGNMENT_AUDIT_MISSING")
        alignment_record = None
    else:
        alignment_score = _unit(
            spatial_alignment.get("score"),
            f"spatial alignment {view_id}.score",
        )
        projection_iou = _unit(
            spatial_alignment.get("projection_iou"),
            f"spatial alignment {view_id}.projection_iou",
        )
        ecc_correlation = _unit(
            spatial_alignment.get("ecc_correlation"),
            f"spatial alignment {view_id}.ecc_correlation",
        )
        if (
            spatial_alignment.get("trusted") is not True
            or spatial_alignment.get("reason_codes") != []
            or spatial_alignment.get("selected_render_view_id")
            != render_view_id
        ):
            reasons.append("SPATIAL_ALIGNMENT_NOT_TRUSTED")
        if alignment_score < MIN_ALIGNMENT_SCORE:
            reasons.append("SPATIAL_ALIGNMENT_SCORE_BELOW_FLOOR")
        if projection_iou < MIN_PROJECTION_IOU:
            reasons.append("SPATIAL_PROJECTION_IOU_BELOW_FLOOR")
        if (
            spatial_alignment.get("ecc_status") != "success"
            or ecc_correlation < MIN_ECC_CORRELATION
        ):
            reasons.append("SPATIAL_ECC_BELOW_FLOOR")
        transform = spatial_alignment.get("ecc_transform_audit")
        if (
            not isinstance(transform, Mapping)
            or transform.get("constraints_passed") is not True
            or transform.get("constraint_failures") != []
        ):
            reasons.append("SPATIAL_ECC_TRANSFORM_NOT_TRUSTED")
        alignment_record = {
            "score": alignment_score,
            "projection_iou": projection_iou,
            "ecc_correlation": ecc_correlation,
        }
    record = {
        "reference_sha256": reference_sha,
        "selected_render_view_id": render_view_id,
        "alignment": alignment_record,
    }
    return record, sorted(set(reasons))


def _dominant_or_excess_failure(view: Mapping[str, Any]) -> bool:
    material = view.get("material_color")
    if not isinstance(material, Mapping):
        return True
    dominant = material.get("trusted_evidence_dominant_mass")
    if isinstance(dominant, Mapping):
        if dominant.get("status") == "FAIL":
            return True
        families = dominant.get("families")
        if isinstance(families, Sequence) and not isinstance(
            families, (str, bytes)
        ):
            if any(
                isinstance(item, Mapping) and item.get("status") == "FAIL"
                for item in families
            ):
                return True
    excess = material.get("unreferenced_render_chromatic_mass")
    return isinstance(excess, Mapping) and excess.get("status") == "FAIL"


def _missing_groups(
    *,
    view: Mapping[str, Any],
    view_id: str,
    recall_threshold: float,
) -> list[Mapping[str, Any]]:
    material = _object(
        view.get("material_color"),
        f"quality view {view_id}.material_color",
    )
    recall = _object(
        material.get("trusted_evidence_group_recall"),
        f"quality view {view_id}.trusted_evidence_group_recall",
    )
    result: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    for index, raw_group in enumerate(
        _array(recall.get("groups"), f"quality view {view_id} recall groups")
    ):
        group = _object(raw_group, f"quality view {view_id} group {index}")
        group_id = _text(
            group.get("group_id"),
            f"quality view {view_id} group {index}.group_id",
        )
        if group_id in seen:
            raise QualityResolutionError(
                f"quality view {view_id} repeats group {group_id}"
            )
        seen.add(group_id)
        recall = _unit(
            group.get("recall"),
            f"quality view {view_id}/{group_id}.recall",
        )
        delivery_presence_status = group.get("delivery_presence_status")
        if delivery_presence_status is not None and delivery_presence_status not in {
            "PRESENT",
            "LOW_EVIDENCE_NEAR_THRESHOLD_PRESENT",
            "MISSING",
        }:
            raise QualityResolutionError(
                f"quality view {view_id}/{group_id} has invalid delivery presence"
            )
        if (
            delivery_presence_status == "MISSING"
            or (
                delivery_presence_status is None
                and recall < recall_threshold
            )
        ):
            result.append(group)
    return result


def _cross_view_delivery(
    *,
    canonical_group_id: str,
    target_view_id: str,
    quality_views: Mapping[str, Mapping[str, Any]],
    view_maps: Mapping[str, Mapping[str, str]],
    recall_threshold: float,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    delivered_view_ids: list[str] = []
    for other_view_id, other_view in sorted(quality_views.items()):
        if other_view_id == target_view_id:
            continue
        material = other_view.get("material_color")
        recall_block = (
            material.get("trusted_evidence_group_recall")
            if isinstance(material, Mapping)
            else None
        )
        raw_groups = (
            recall_block.get("groups")
            if isinstance(recall_block, Mapping)
            else None
        )
        if not isinstance(raw_groups, Sequence) or isinstance(
            raw_groups, (str, bytes)
        ):
            continue
        matching = [
            _object(
                raw_group,
                f"quality view {other_view_id} cross-view group",
            )
            for raw_group in raw_groups
            if isinstance(raw_group, Mapping)
            and view_maps.get(other_view_id, {}).get(raw_group.get("group_id"))
            == canonical_group_id
        ]
        if len(matching) > 1:
            raise QualityResolutionError(
                f"quality view {other_view_id} repeats canonical group "
                f"{canonical_group_id}"
            )
        if not matching:
            continue
        group = matching[0]
        local_group_id = _text(
            group.get("group_id"),
            f"quality view {other_view_id} cross-view local group",
        )
        required_share = _unit(
            group.get("required_render_share"),
            f"quality view {other_view_id}/{local_group_id}.required_render_share",
        )
        observed_share = _unit(
            group.get("observed_render_share"),
            f"quality view {other_view_id}/{local_group_id}.observed_render_share",
        )
        recall = _unit(
            group.get("recall"),
            f"quality view {other_view_id}/{local_group_id}.recall",
        )
        presence = group.get("delivery_presence_status")
        delivered = (
            presence
            in {"PRESENT", "LOW_EVIDENCE_NEAR_THRESHOLD_PRESENT"}
            or observed_share >= required_share
            or recall >= recall_threshold
        )
        records.append(
            {
                "reference_view_id": other_view_id,
                "local_group_id": local_group_id,
                "view_status": other_view.get("status"),
                "required_render_share": required_share,
                "observed_render_share": observed_share,
                "recall": recall,
                "delivery_presence_status": presence,
                "delivered": delivered,
            }
        )
        if delivered:
            delivered_view_ids.append(other_view_id)
    return {
        "canonical_group_id": canonical_group_id,
        "delivered_view_ids": sorted(delivered_view_ids),
        "delivered_view_count": len(delivered_view_ids),
        "views": records,
    }


def _qa_repair_assignments(
    *,
    canonical_group_id: str,
    target_render_view_id: str,
    assignments: Mapping[str, Mapping[str, Any]],
    visible_by_view: Mapping[str, Mapping[str, int]],
    risks: Mapping[str, bool],
) -> list[dict[str, Any]]:
    confirmed_reasons = {
        "QA_MISSING_CANONICAL_GROUP_MULTI_VIEW",
        "QA_TRUSTED_PART_GROUP_LOCALIZATION",
        "QA_CONFIRMED_WHITELIST_MATERIAL",
    }
    provisional_reasons = {
        "QA_MISSING_CANONICAL_GROUP_MULTI_VIEW",
        "QA_TRUSTED_PART_GROUP_LOCALIZATION",
        "QA_HIGH_CONFIDENCE_WHITELIST_MATERIAL_CANDIDATE",
        "QA_POST_RENDER_VALIDATION_REQUIRED",
    }
    result: list[dict[str, Any]] = []
    for part_id, assignment in sorted(assignments.items()):
        provenance = assignment.get("provenance")
        if (
            not isinstance(provenance, Mapping)
            or provenance.get("tier") != "qa_repair_candidate"
            or provenance.get("canonical_group_id") != canonical_group_id
            or risks.get(part_id) is not False
        ):
            continue
        reason_codes = provenance.get("reason_codes")
        if (
            not isinstance(reason_codes, Sequence)
            or isinstance(reason_codes, (str, bytes))
        ):
            continue
        reason_set = set(reason_codes)
        provisional = reason_set == provisional_reasons
        confirmed = reason_set == confirmed_reasons
        if not confirmed and not (
            provisional
            and provenance.get("material_selection_basis")
            == "high_confidence_whitelist_candidate_pending_render_qa"
        ):
            continue
        supporting_view_ids = provenance.get("supporting_view_ids")
        if (
            not isinstance(supporting_view_ids, list)
            or supporting_view_ids != sorted(set(supporting_view_ids))
            or len(supporting_view_ids) < MIN_CROSS_VIEW_DELIVERED_VIEWS
        ):
            continue
        target_pixels = int(
            visible_by_view.get(target_render_view_id, {}).get(part_id, 0)
        )
        if target_pixels <= 0:
            continue
        result.append(
            {
                "part_id": part_id,
                "material_id": assignment.get("material_id"),
                "selection_lane": (
                    "post_render_validated_high_confidence_candidate"
                    if provisional
                    else "confirmed_whitelist_material"
                ),
                "supporting_view_ids": supporting_view_ids,
                "target_visible_pixels": target_pixels,
            }
        )
    return result


def _coverage_alignment(
    *,
    view: Mapping[str, Any],
    strict_alignment: Mapping[str, Any] | None,
    strict_reasons: Sequence[str],
) -> tuple[dict[str, Any] | None, list[str]]:
    reasons = [
        reason
        for reason in strict_reasons
        if reason
        not in {
            "SPATIAL_PROJECTION_IOU_BELOW_FLOOR",
            "SPATIAL_ECC_BELOW_FLOOR",
        }
    ]
    mapping = view.get("mapping")
    preview = (
        mapping.get("alignment_preview")
        if isinstance(mapping, Mapping)
        else None
    )
    alignment = (
        strict_alignment.get("alignment")
        if isinstance(strict_alignment, Mapping)
        else None
    )
    if not isinstance(alignment, Mapping):
        reasons.append("COVERAGE_ALIGNMENT_AUDIT_MISSING")
        return None, sorted(set(reasons))
    score = _unit(alignment.get("score"), "coverage alignment score")
    projection_iou = _unit(
        alignment.get("projection_iou"), "coverage projection IoU"
    )
    ecc_correlation = _unit(
        alignment.get("ecc_correlation"), "coverage ECC correlation"
    )
    if score < MIN_ALIGNMENT_SCORE:
        reasons.append("COVERAGE_ALIGNMENT_SCORE_BELOW_FLOOR")
    if projection_iou < MIN_COVERAGE_PROJECTION_IOU:
        reasons.append("COVERAGE_PROJECTION_IOU_BELOW_FLOOR")
    if ecc_correlation < MIN_COVERAGE_ECC_CORRELATION:
        reasons.append("COVERAGE_ECC_BELOW_FLOOR")
    if not isinstance(preview, Mapping):
        reasons.append("MAPPING_ALIGNMENT_PREVIEW_MISSING")
        preview_record = None
    else:
        preview_score = _unit(
            preview.get("score"), "mapping alignment preview score"
        )
        preview_iou = _unit(
            preview.get("silhouette_iou"),
            "mapping alignment preview silhouette IoU",
        )
        if preview_score < MIN_MAPPING_PREVIEW_SCORE:
            reasons.append("MAPPING_PREVIEW_SCORE_BELOW_FLOOR")
        if preview_iou < MIN_MAPPING_PREVIEW_SILHOUETTE_IOU:
            reasons.append("MAPPING_PREVIEW_SILHOUETTE_IOU_BELOW_FLOOR")
        preview_record = {
            "score": preview_score,
            "silhouette_iou": preview_iou,
        }
    return (
        {
            "score": score,
            "projection_iou": projection_iou,
            "ecc_correlation": ecc_correlation,
            "mapping_preview": preview_record,
        },
        sorted(set(reasons)),
    )


def _limitation_candidate(
    *,
    view_id: str,
    view: Mapping[str, Any],
    local_group: Mapping[str, Any],
    quality_views: Mapping[str, Mapping[str, Any]],
    canonical_groups: Mapping[str, Mapping[str, Any]],
    view_maps: Mapping[str, Mapping[str, str]],
    unique_chromatic_groups: set[str],
    spatial_policy: Mapping[str, Any],
    spatial_parts: Mapping[str, Mapping[str, Any]],
    spatial_references: Mapping[str, Mapping[str, Any]],
    spatial_alignments: Mapping[str, Mapping[str, Any]],
    spatial_part_ids_sha: Mapping[str, str],
    registry_part_ids_sha: Mapping[str, str],
    source_groups: Mapping[str, Mapping[str, Any]],
    applied_source_ids: set[str],
    assignments: Mapping[str, Mapping[str, Any]],
    registry_parts: Mapping[str, Mapping[str, Any]],
    visible_by_view: Mapping[str, Mapping[str, int]],
    risks: Mapping[str, bool],
    recall_threshold: float,
) -> tuple[dict[str, Any], bool]:
    local_group_id = _text(
        local_group.get("group_id"),
        f"quality view {view_id} missing local group",
    )
    reasons: list[str] = []
    canonical_group_id = view_maps.get(view_id, {}).get(local_group_id)
    if canonical_group_id is None:
        return (
            {
                "reference_view_id": view_id,
                "local_group_id": local_group_id,
                "canonical_group_id": None,
                "eligible": False,
                "reason_codes": ["LOCAL_GROUP_NOT_CANONICALIZED"],
            },
            False,
        )
    canonical_group = canonical_groups[canonical_group_id]
    if canonical_group_id not in unique_chromatic_groups:
        reasons.append("CANONICAL_GROUP_NOT_UNIQUE_MULTIVIEW_CHROMATIC")
    expected_labels = evidence_color_labels(
        _text(
            canonical_group.get("base_color"),
            f"canonical group {canonical_group_id}.base_color",
        )
    )
    base_colors = _sorted_unique_texts(
        local_group.get("base_colors"),
        f"quality view {view_id}/{local_group_id}.base_colors",
    )
    if not base_colors or any(
        not evidence_color_labels(color) & expected_labels
        for color in base_colors
    ):
        reasons.append("LOCAL_CANONICAL_COLOR_CONFLICT")

    evidence_pixels = _integer(
        local_group.get("reference_evidence_weight"),
        f"quality view {view_id}/{local_group_id}.reference_evidence_weight",
    )
    reference_group_share = _unit(
        local_group.get("reference_group_share"),
        f"quality view {view_id}/{local_group_id}.reference_group_share",
    )
    required_share = _unit(
        local_group.get("required_render_share"),
        f"quality view {view_id}/{local_group_id}.required_render_share",
    )
    observed_share = _unit(
        local_group.get("observed_render_share"),
        f"quality view {view_id}/{local_group_id}.observed_render_share",
    )
    recall = _unit(
        local_group.get("recall"),
        f"quality view {view_id}/{local_group_id}.recall",
    )
    if evidence_pixels < MIN_REFERENCE_EVIDENCE_PIXELS:
        reasons.append("REFERENCE_GROUP_EVIDENCE_PIXELS_BELOW_FLOOR")
    if reference_group_share > MAX_REFERENCE_GROUP_SHARE:
        reasons.append("REFERENCE_GROUP_SHARE_ABOVE_LIMITATION_CEILING")
    if recall >= recall_threshold or observed_share >= required_share:
        reasons.append("REFERENCE_GROUP_IS_NOT_A_RENDER_DEFICIT")

    strict_alignment, strict_alignment_reasons = _reference_alignment(
        view=view,
        view_id=view_id,
        spatial_reference=spatial_references.get(view_id),
        spatial_alignment=spatial_alignments.get(view_id),
    )
    render_view_id = (
        strict_alignment.get("selected_render_view_id")
        if isinstance(strict_alignment, Mapping)
        else None
    )
    reference_sha = (
        strict_alignment.get("reference_sha256")
        if isinstance(strict_alignment, Mapping)
        else None
    )
    candidate_record: dict[str, Any] | None = None
    candidate_reasons: list[str] = []
    owner_record: dict[str, Any] | None = None
    owner_diagnostics: list[dict[str, Any]] = []
    cross_view_delivery = _cross_view_delivery(
        canonical_group_id=canonical_group_id,
        target_view_id=view_id,
        quality_views=quality_views,
        view_maps=view_maps,
        recall_threshold=recall_threshold,
    )
    qa_repair_assignments: list[dict[str, Any]] = []
    limitation_lane: str | None = None
    classification = LIMITATION_CLASSIFICATION
    reason_code = LIMITATION_REASON
    selected_alignment: Mapping[str, Any] | None = (
        strict_alignment.get("alignment")
        if isinstance(strict_alignment, Mapping)
        else None
    )
    if isinstance(render_view_id, str):
        spatial_digest = spatial_part_ids_sha.get(render_view_id)
        registry_digest = registry_part_ids_sha.get(render_view_id)
        if spatial_digest is None:
            reasons.append("SPATIAL_TARGET_PART_IDS_SHA_MISSING")
        if registry_digest and spatial_digest != registry_digest:
            reasons.append("FINAL_REGISTRY_TARGET_PART_IDS_SHA_MISMATCH")
        candidate_record, candidate_reasons = _candidate_geometry(
            group_id=canonical_group_id,
            target_render_view_id=render_view_id,
            source_groups=source_groups,
            applied_source_ids=applied_source_ids,
            assignments=assignments,
            registry_parts=registry_parts,
            visible_by_view=visible_by_view,
            risks=risks,
        )
        safe_part_ids = set(
            candidate_record.get("safe_part_ids", [])
            if isinstance(candidate_record, Mapping)
            else []
        )
        visible_part_ids = set(
            candidate_record.get("visible_part_ids", [])
            if isinstance(candidate_record, Mapping)
            else []
        )
        qa_repair_assignments = _qa_repair_assignments(
            canonical_group_id=canonical_group_id,
            target_render_view_id=render_view_id,
            assignments=assignments,
            visible_by_view=visible_by_view,
            risks=risks,
        )
        if (
            isinstance(reference_sha, str)
            and isinstance(spatial_digest, str)
            and safe_part_ids
        ):
            owner_record, owner_reasons, owner_diagnostics = _owner(
                target_group_id=canonical_group_id,
                local_group_id=local_group_id,
                view_id=view_id,
                evidence_pixels=evidence_pixels,
                spatial_parts=spatial_parts,
                assignments=assignments,
                candidate_part_ids=safe_part_ids,
                spatial_policy=spatial_policy,
                accepted_palette_evidence=[
                    _object(
                        item,
                        f"spatial reference {view_id} accepted palette evidence",
                    )
                    for item in _array(
                        spatial_references[view_id].get(
                            "accepted_palette_evidence"
                        ),
                        (
                            f"spatial reference {view_id} "
                            "accepted_palette_evidence"
                        ),
                    )
                ],
            )
            # The owner remains useful diagnostic evidence, but an exactly
            # invisible, source-bound repeated cohort is already sufficient
            # proof that changing its material cannot make it appear.
            del owner_reasons

        if safe_part_ids:
            limitation_lane = "source_bound_zero_visible_repeated_geometry"
            reasons.extend(candidate_reasons)
            reasons.extend(strict_alignment_reasons)
        elif (
            visible_part_ids
            and observed_share > 0.0
            and recall >= MIN_SOURCE_COVERAGE_RECALL
        ):
            limitation_lane = "source_bound_visible_repeated_geometry"
            classification = COVERAGE_CLASSIFICATION
            reason_code = COVERAGE_REASON
            selected_alignment, coverage_reasons = _coverage_alignment(
                view=view,
                strict_alignment=strict_alignment,
                strict_reasons=strict_alignment_reasons,
            )
            reasons.extend(candidate_reasons)
            reasons.extend(coverage_reasons)
        elif (
            qa_repair_assignments
            and cross_view_delivery["delivered_view_count"]
            >= MIN_CROSS_VIEW_DELIVERED_VIEWS
            and observed_share > 0.0
            and recall >= MIN_CROSS_VIEW_COVERAGE_RECALL
        ):
            limitation_lane = "cross_view_material_delivery"
            classification = COVERAGE_CLASSIFICATION
            reason_code = COVERAGE_REASON
            selected_alignment, coverage_reasons = _coverage_alignment(
                view=view,
                strict_alignment=strict_alignment,
                strict_reasons=strict_alignment_reasons,
            )
            reasons.extend(coverage_reasons)
        else:
            reasons.extend(candidate_reasons)
            reasons.extend(strict_alignment_reasons)
            if observed_share <= 0.0:
                reasons.append("TARGET_GROUP_NOT_OBSERVED_IN_RENDER")
            if visible_part_ids and recall < MIN_SOURCE_COVERAGE_RECALL:
                reasons.append("SOURCE_COVERAGE_RECALL_BELOW_FLOOR")
            if qa_repair_assignments and (
                cross_view_delivery["delivered_view_count"]
                < MIN_CROSS_VIEW_DELIVERED_VIEWS
            ):
                reasons.append("CROSS_VIEW_MATERIAL_DELIVERY_BELOW_FLOOR")
            if qa_repair_assignments and recall < MIN_CROSS_VIEW_COVERAGE_RECALL:
                reasons.append("CROSS_VIEW_COVERAGE_RECALL_BELOW_FLOOR")
            reasons.append("NO_VERIFIED_GEOMETRY_COVERAGE_LANE")
    else:
        reasons.append("TARGET_RENDER_VIEW_UNAVAILABLE")
        reasons.extend(strict_alignment_reasons)

    core = {
        "classification": classification,
        "reason_code": reason_code,
        "limitation_lane": limitation_lane,
        "canonical_group_id": canonical_group_id,
        "reference_view_id": view_id,
        "local_group_id": local_group_id,
        "reference_sha256": reference_sha,
        "selected_render_view_id": render_view_id,
        "alignment": selected_alignment,
        "reference_group_evidence": {
            "base_colors": base_colors,
            "evidence_pixels": evidence_pixels,
            "reference_group_share": reference_group_share,
            "required_render_share": required_share,
            "observed_render_share": observed_share,
            "recall": recall,
            "minimum_recall": recall_threshold,
        },
        "candidate_geometry": candidate_record,
        "cross_view_delivery": cross_view_delivery,
        "qa_repair_assignments": qa_repair_assignments,
        "foreign_owner": owner_record,
        "foreign_owner_diagnostics": owner_diagnostics,
    }
    eligible = not reasons
    record = {
        **core,
        "eligible": eligible,
        "reason_codes": sorted(set(reasons)),
    }
    if eligible:
        record["evidence_sha256"] = canonical_sha256(core)
    return record, eligible


def build_quality_resolution(
    *,
    final_plan: Mapping[str, Any],
    policy_audit: Mapping[str, Any],
    quality_report: Mapping[str, Any],
    palette_fusion: Mapping[str, Any],
    spatial_report: Mapping[str, Any],
    geometry_risk: Mapping[str, Any],
    rendered_registry: Mapping[str, Any],
) -> dict[str, Any]:
    """Return one deterministic final QA resolution report."""

    input_hashes = _hashes(
        final_plan=final_plan,
        policy_audit=policy_audit,
        quality_report=quality_report,
        palette_fusion=palette_fusion,
        spatial_report=spatial_report,
        geometry_risk=geometry_risk,
        rendered_registry=rendered_registry,
    )
    aggregate, thresholds, quality_views = _quality_views(quality_report)
    raw_status = _text(aggregate.get("status"), "quality aggregate status")
    registry_parts, visible_by_view, registry_part_ids_sha = _registry(
        rendered_registry
    )
    assignments = _assignments(
        final_plan,
        registry_ids=set(registry_parts),
    )
    risks = _geometry_risks(geometry_risk)
    (
        spatial_policy,
        spatial_parts,
        spatial_references,
        spatial_alignments,
        spatial_part_ids_sha,
    ) = _spatial(spatial_report)
    if set(spatial_parts) != set(registry_parts):
        raise QualityResolutionError(
            "spatial report does not exactly cover the rendered registry"
        )

    report_thresholds = {
        "minimum_alignment_score": MIN_ALIGNMENT_SCORE,
        "minimum_projection_iou": MIN_PROJECTION_IOU,
        "minimum_ecc_correlation": MIN_ECC_CORRELATION,
        "minimum_coverage_projection_iou": MIN_COVERAGE_PROJECTION_IOU,
        "minimum_coverage_ecc_correlation": MIN_COVERAGE_ECC_CORRELATION,
        "minimum_mapping_preview_score": MIN_MAPPING_PREVIEW_SCORE,
        "minimum_mapping_preview_silhouette_iou": (
            MIN_MAPPING_PREVIEW_SILHOUETTE_IOU
        ),
        "minimum_reference_evidence_pixels": MIN_REFERENCE_EVIDENCE_PIXELS,
        "maximum_reference_group_share": MAX_REFERENCE_GROUP_SHARE,
        "maximum_limited_share_per_view": MAX_LIMITED_SHARE_PER_VIEW,
        "minimum_limited_aggregate_color_score": (
            MIN_LIMITED_AGGREGATE_COLOR_SCORE
        ),
        "minimum_repeated_candidate_count": MIN_REPEATED_CANDIDATE_COUNT,
        "minimum_other_visible_render_views": MIN_OTHER_VISIBLE_RENDER_VIEWS,
        "minimum_source_coverage_recall": MIN_SOURCE_COVERAGE_RECALL,
        "minimum_cross_view_coverage_recall": (
            MIN_CROSS_VIEW_COVERAGE_RECALL
        ),
        "minimum_cross_view_delivered_views": (
            MIN_CROSS_VIEW_DELIVERED_VIEWS
        ),
        "minimum_owner_projected_pixels": MIN_OWNER_PROJECTED_PIXELS,
        "minimum_owner_color_share": MIN_OWNER_COLOR_SHARE,
        "minimum_owner_color_margin": MIN_OWNER_COLOR_MARGIN,
        "maximum_target_share_in_owner": MAX_TARGET_SHARE_IN_OWNER,
        "minimum_accepted_box_owner_overlap": (
            MIN_ACCEPTED_BOX_OWNER_OVERLAP
        ),
    }
    if raw_status == PASS:
        return {
            "schema_version": REPORT_SCHEMA_VERSION,
            "input_hashes": input_hashes,
            "raw_quality_status": raw_status,
            "resolution_status": PASS,
            "material_stage_accepted": True,
            "reason_codes": ["RAW_VISUAL_QA_PASS"],
            "thresholds": report_thresholds,
            "limitations": [],
            "limitation_candidates": [],
            "summary": {
                "quality_view_count": len(quality_views),
                "passed_view_count": sum(
                    view.get("status") == PASS
                    for view in quality_views.values()
                ),
                "accepted_limitation_count": 0,
                "fail_closed_candidate_count": 0,
            },
        }

    recall_threshold = _unit(
        thresholds.get("minimum_evidence_group_recall"),
        "minimum_evidence_group_recall",
    )
    _unit(thresholds.get("pass_color_score"), "pass_color_score")
    aggregate_color_score = _unit(
        aggregate.get("material_color_score"),
        "aggregate material_color_score",
    )
    (
        canonical_groups,
        view_maps,
        unique_chromatic_groups,
    ) = _canonical_groups(palette_fusion)
    source_groups, applied_source_ids, source_audit_reasons = (
        _policy_source_groups(policy_audit)
    )

    candidates: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []
    top_reasons: list[str] = list(source_audit_reasons)
    passed_view_count = sum(
        view.get("status") == PASS for view in quality_views.values()
    )
    if passed_view_count < 2:
        top_reasons.append("FEWER_THAN_TWO_PASSING_COMPARABLE_VIEWS")
    if aggregate_color_score < MIN_LIMITED_AGGREGATE_COLOR_SCORE:
        top_reasons.append("AGGREGATE_COLOR_SCORE_BELOW_LIMITED_PASS_FLOOR")

    limited_share_by_view: Counter[str] = Counter()
    nonpass_comparable_views: set[str] = set()
    for view_id, view in sorted(quality_views.items()):
        view_status = view.get("status")
        if view_status == "UNSCORABLE":
            continue
        if view_status != PASS:
            nonpass_comparable_views.add(view_id)
        if _dominant_or_excess_failure(view):
            top_reasons.append(
                f"DOMINANT_OR_UNREFERENCED_CHROMATIC_FAILURE:{view_id}"
            )
        missing = _missing_groups(
            view=view,
            view_id=view_id,
            recall_threshold=recall_threshold,
        )
        if not missing:
            if view_status != PASS:
                top_reasons.append(
                    f"NONPASS_VIEW_WITHOUT_LIMITABLE_GROUP:{view_id}"
                )
            continue
        if (
            view_status != "FAIL"
            or view.get("reasons")
            != ["trusted_palette_group_missing_from_render"]
        ):
            top_reasons.append(
                f"MISSING_GROUP_VIEW_HAS_OTHER_FAILURES:{view_id}"
            )
        for local_group in missing:
            candidate, eligible = _limitation_candidate(
                view_id=view_id,
                view=view,
                local_group=local_group,
                quality_views=quality_views,
                canonical_groups=canonical_groups,
                view_maps=view_maps,
                unique_chromatic_groups=unique_chromatic_groups,
                spatial_policy=spatial_policy,
                spatial_parts=spatial_parts,
                spatial_references=spatial_references,
                spatial_alignments=spatial_alignments,
                spatial_part_ids_sha=spatial_part_ids_sha,
                registry_part_ids_sha=registry_part_ids_sha,
                source_groups=source_groups,
                applied_source_ids=applied_source_ids,
                assignments=assignments,
                registry_parts=registry_parts,
                visible_by_view=visible_by_view,
                risks=risks,
                recall_threshold=recall_threshold,
            )
            candidates.append(candidate)
            if eligible:
                accepted.append(candidate)
                limited_share_by_view[view_id] += float(
                    candidate["reference_group_evidence"][
                        "reference_group_share"
                    ]
                )

    for view_id, share in sorted(limited_share_by_view.items()):
        if share > MAX_LIMITED_SHARE_PER_VIEW:
            top_reasons.append(
                f"LIMITED_REFERENCE_SHARE_ABOVE_VIEW_CEILING:{view_id}"
            )
    covered_views = {
        str(item["reference_view_id"]) for item in accepted
    }
    if not nonpass_comparable_views <= covered_views:
        top_reasons.append("NONPASS_COMPARABLE_VIEW_NOT_COVERED")
    if any(not item["eligible"] for item in candidates):
        top_reasons.append("LIMITATION_CANDIDATE_FAILED_CLOSED")
    if not accepted:
        top_reasons.append("NO_ACCEPTED_GEOMETRY_OR_POSE_LIMITATION")

    top_reasons = sorted(set(top_reasons))
    accepted_resolution = not top_reasons
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "input_hashes": input_hashes,
        "raw_quality_status": raw_status,
        "resolution_status": LIMITED_PASS if accepted_resolution else FAIL_CLOSED,
        "material_stage_accepted": accepted_resolution,
        "reason_codes": top_reasons,
        "thresholds": report_thresholds,
        "limitations": accepted if accepted_resolution else [],
        "limitation_candidates": candidates,
        "summary": {
            "quality_view_count": len(quality_views),
            "passed_view_count": passed_view_count,
            "accepted_limitation_count": (
                len(accepted) if accepted_resolution else 0
            ),
            "fail_closed_candidate_count": sum(
                item.get("eligible") is not True for item in candidates
            ),
        },
    }


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        resolved = path.expanduser().resolve(strict=True)
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise QualityResolutionError(f"unable to read {label}: {exc}") from exc
    return dict(_object(value, label))


def _write_json_new(path: Path, value: Mapping[str, Any]) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.exists():
        raise QualityResolutionError(f"refusing to overwrite output: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved.with_name(f".{resolved.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(
                value,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, resolved)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return resolved


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-plan", type=Path, required=True)
    parser.add_argument("--policy-audit", type=Path, required=True)
    parser.add_argument("--quality-report", type=Path, required=True)
    parser.add_argument("--palette-fusion", type=Path, required=True)
    parser.add_argument("--spatial-report", type=Path, required=True)
    parser.add_argument("--geometry-risk", type=Path, required=True)
    parser.add_argument("--rendered-registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_path = args.output.expanduser().resolve()
    if output_path.exists():
        raise QualityResolutionError(
            f"refusing to overwrite output: {output_path}"
        )
    report = build_quality_resolution(
        final_plan=_read_json(args.final_plan, "final plan"),
        policy_audit=_read_json(args.policy_audit, "policy audit"),
        quality_report=_read_json(args.quality_report, "quality report"),
        palette_fusion=_read_json(args.palette_fusion, "palette fusion"),
        spatial_report=_read_json(args.spatial_report, "spatial report"),
        geometry_risk=_read_json(args.geometry_risk, "geometry risk"),
        rendered_registry=_read_json(
            args.rendered_registry,
            "rendered registry",
        ),
    )
    output = _write_json_new(output_path, report)
    print(
        json.dumps(
            {
                "output": str(output),
                "raw_quality_status": report["raw_quality_status"],
                "resolution_status": report["resolution_status"],
                **report["summary"],
            },
            ensure_ascii=False,
            allow_nan=False,
        ),
        flush=True,
    )
    return 0


__all__ = [
    "COVERAGE_CLASSIFICATION",
    "COVERAGE_REASON",
    "FAIL_CLOSED",
    "LIMITATION_CLASSIFICATION",
    "LIMITATION_REASON",
    "LIMITED_PASS",
    "PASS",
    "QualityResolutionError",
    "REPORT_SCHEMA_VERSION",
    "build_quality_resolution",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
