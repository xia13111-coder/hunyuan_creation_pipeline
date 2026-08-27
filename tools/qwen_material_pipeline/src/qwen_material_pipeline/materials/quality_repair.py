"""Compile one conservative material-plan repair from final visual QA.

This command is intentionally narrower than material inference.  It does not
rerun Qwen, relax a threshold, invent a material, or turn review evidence into
``auto`` confidence.  It can only replace a neutral/default
``policy_fallback`` assignment when all of the following are already proven by
hash-bound pipeline artefacts:

* the same canonical palette group is missing from at least two trusted,
  content-distinct reference views registered to different CAD poses, or it is
  a multi-view palette group missing from exactly one trusted QA view;
* the part is independently localized to that group in those views, by one
  exact perturbation-stable projection in the single-view lane, or by one
  high-purity direct projection whose canonical group already has another
  accepted multi-view repair anchor covering every QA-deficit view; a dominant
  single-view deficit may also override one same-view semantic vote when a
  second trusted view semantically anchors the target and the rejected
  semantic group is nearly absent from the projected mask; the multi-view
  anchor lane may defer that second semantic anchor to its independently
  accepted spatial anchors.  Semantic votes from views where the part is
  explicitly below the diagnostic visibility floor are ignored; two
  content/pose/hash-distinct bounded pixel diagnostics may override only a
  single-view semantic disagreement, while any independently corroborated
  alternative or unresolved spatial/mapping conflict remains a hard block;
* uniform assignment is not flagged as a multi-material geometry risk; and
* the group's material choice is confirmed and belongs to the whitelist.

Eligible repairs keep ``status=policy_fallback`` and ``confidence=0``.  The
repair is a deterministic, explicitly authorized candidate for one subsequent
Apply/Render/Compare round; it is not a promotion of model confidence.  If any
required localizing schema or evidence is absent, the result is a safe no-op
with ``changed_count=0``.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from qwen_material_pipeline.evidence.color_semantics import evidence_color_labels
from qwen_material_pipeline.materials.policy_exact_cover import (
    _mvinverse_parameterizations,
)
from qwen_material_pipeline.usd.material_common import (
    POLICY_EXACT_COVER_MODE,
    POLICY_FALLBACK_CONFIDENCE_BASIS,
    SOURCE_VISUAL_PRESERVE_ACTION,
    SOURCE_VISUAL_PRESERVE_TIER,
)


PLAN_SCHEMA_VERSION = "1.0"
REGISTRY_SCHEMA_VERSION = "qwen-material-parts/v1"
POLICY_AUDIT_SCHEMA_VERSION = "qwen-policy-exact-cover-report/v1"
QUALITY_SCHEMA_VERSION = "qwen-reference-render-comparison/v1"
PALETTE_FUSION_SCHEMA_VERSION = "qwen-multiview-palette-fusion/v1"
SPATIAL_REPORT_SCHEMA_VERSION = "qwen-spatial-mapping-audit/v1"
SPATIAL_GATE_SCHEMA_VERSION = "qwen-spatial-mapping-gate/v1"
MAPPING_CONSENSUS_SCHEMA_VERSION = "qwen-mapping-consensus-audit/v1"
GEOMETRY_RISK_SCHEMA_VERSION = "qwen-geometry-uniform-material-risk/v1"
GROUP_MATERIALS_SCHEMA_VERSION = "qwen-palette-material/v1"
REPORT_SCHEMA_VERSION = "qwen-quality-repair-report/v1"
REPAIR_MODE = "quality_missing_canonical_group_repair/v1"
REPAIR_PROVENANCE_FIELD = "quality_repair"
ANCHORED_SINGLE_VIEW_LANE = "exact_spatial_single_view_with_multiview_anchor"
SPATIAL_ANCHOR_SINGLE_VIEW_LANE = "exact_spatial_single_qa_view_with_spatial_anchor"
QA_CONFIRMED_MULTIVIEW_SEMANTIC_REVIEW_LANE = "qa_confirmed_three_view_semantic_review"
QA_CONFIRMED_MULTIVIEW_SEMANTIC_REVIEW_MIN_SUPPORTS = 3
SOURCE_IDENTITY_ANCHORED_DIAGNOSTIC_LANE = (
    "source_identity_anchored_single_view_diagnostic"
)
SOURCE_IDENTITY_COHORT_CONSENSUS_LANE = "source_identity_cohort_multiview_consensus"
DOMINANT_ASSEMBLY_COHORT_EXPANSION_LANE = "dominant_assembly_cohort_expansion"
BOUNDED_SIGNATURE_SIBLING_COHORT_LANE = "single_strict_anchor_bounded_signature_sibling"
BOUNDED_SIGNATURE_SIBLING_PROPOSAL_POLICY = (
    "single_strict_anchor_bounded_signature_sibling/v1"
)
MATERIAL_SELECTION_OBJECTIVE_SEMANTIC = "semantic_compatible_visual"
MATERIAL_SELECTION_OBJECTIVE_VISUAL = "visual_similarity"
MATERIAL_SELECTION_OBJECTIVES = {
    MATERIAL_SELECTION_OBJECTIVE_SEMANTIC,
    MATERIAL_SELECTION_OBJECTIVE_VISUAL,
}
SOURCE_IDENTITY_MAX_REGISTRY_FRACTION = 0.05
SOURCE_IDENTITY_MIN_SIGNATURE_COUNT = 2
SOURCE_IDENTITY_MAX_ASSEMBLY_COHORT_SIZE = 4
DOMINANT_ASSEMBLY_MIN_GROUP_CONFIDENCE = 0.80
DOMINANT_ASSEMBLY_MIN_GROUP_SOURCE_VIEWS = 3
DOMINANT_ASSEMBLY_MIN_QA_VIEWS = 3
DOMINANT_ASSEMBLY_MIN_REFERENCE_SHARE = 0.50
DOMINANT_ASSEMBLY_MIN_REFERENCE_SHARE_MARGIN = 0.25
DOMINANT_ASSEMBLY_MIN_DEFICIT_SHARE = 0.20
DOMINANT_ASSEMBLY_MAX_MASS_RECALL = 0.50
DOMINANT_ASSEMBLY_MIN_ALIGNMENT_SCORE = 0.82
DOMINANT_ASSEMBLY_MIN_PROJECTION_IOU = 0.80
DOMINANT_ASSEMBLY_MIN_ECC_CORRELATION = 0.85
DOMINANT_ASSEMBLY_MIN_ANCHOR_PROJECTED_PIXELS = 1024
DOMINANT_ASSEMBLY_MIN_ANCHOR_COLOR_SHARE = 0.85
DOMINANT_ASSEMBLY_MIN_ANCHOR_COLOR_MARGIN = 0.75
DOMINANT_ASSEMBLY_MIN_PERTURBATION_PIXELS = 256
DOMINANT_ASSEMBLY_MIN_PERTURBATION_COLOR_SHARE = 0.65
DOMINANT_ASSEMBLY_MIN_PERTURBATION_COLOR_MARGIN = 0.30
DOMINANT_ASSEMBLY_MIN_ANCHOR_VIEWS = 2
DOMINANT_ASSEMBLY_MIN_ANCHOR_PARTS = 3
DOMINANT_ASSEMBLY_MIN_ANCHOR_CHILD_BRANCHES = 3
DOMINANT_ASSEMBLY_MIN_COHORT_SIZE = 6
DOMINANT_ASSEMBLY_MAX_COHORT_SIZE = 128
DOMINANT_ASSEMBLY_MAX_REGISTRY_FRACTION = 0.20
DOMINANT_ASSEMBLY_MIN_SIGNATURE_PART_SHARE = 0.90
DOMINANT_ASSEMBLY_MIN_SIGNATURE_FACE_SHARE = 0.90
DOMINANT_ASSEMBLY_ALTERNATIVE_MIN_PROJECTED_PIXELS = 256
DOMINANT_ASSEMBLY_ALTERNATIVE_MIN_COLOR_SHARE = 0.65
DOMINANT_ASSEMBLY_ALTERNATIVE_MIN_COLOR_MARGIN = 0.30
DOMINANT_ASSEMBLY_MAX_VETO_PART_SHARE = 0.10
DOMINANT_ASSEMBLY_MAX_VETO_FACE_SHARE = 0.05
DOMINANT_ASSEMBLY_MEMBERSHIP_STATUS = "PROVISIONAL_PENDING_GROUP_TOURNAMENT"
BOUNDED_SIGNATURE_MIN_GROUP_CONFIDENCE = 0.60
BOUNDED_SIGNATURE_MIN_GROUP_SOURCE_VIEWS = 3
BOUNDED_SIGNATURE_MIN_QA_VIEWS = 3
BOUNDED_SIGNATURE_REQUIRED_GLOBAL_COHORT_SIZE = 2
BOUNDED_SIGNATURE_REQUIRED_DIAGNOSTIC_SAMPLE_COUNT = 6
BOUNDED_SIGNATURE_MIN_TARGET_SAMPLE_COUNT = 5
BOUNDED_SIGNATURE_MIN_ANCHOR_DIRECT_COLOR_SHARE = 0.70
BOUNDED_SIGNATURE_MIN_ANCHOR_DIRECT_COLOR_MARGIN = 0.60
BOUNDED_SIGNATURE_MIN_ANCHOR_BBOX_COLOR_SHARE = 0.65
BOUNDED_SIGNATURE_MIN_ANCHOR_BBOX_COLOR_MARGIN = 0.55
BOUNDED_SIGNATURE_MIN_ANCHOR_PERTURBATION_SHARE = 0.50
BOUNDED_SIGNATURE_MIN_ANCHOR_PERTURBATION_MARGIN = 0.50
BOUNDED_SIGNATURE_MIN_DIRECT_COLOR_SHARE = 0.40
BOUNDED_SIGNATURE_MIN_DIRECT_COLOR_MARGIN = 0.20
BOUNDED_SIGNATURE_MIN_BBOX_COLOR_SHARE = 0.40
BOUNDED_SIGNATURE_MIN_BBOX_COLOR_MARGIN = 0.20
DOMINANT_ASSEMBLY_CHROMATIC_BASE_COLORS = frozenset(
    {
        "red",
        "orange",
        "orange_brown",
        "brown",
        "yellow",
        "green",
        "cyan",
        "cyan_blue",
        "blue",
        "purple",
    }
)
DOMINANT_ASSEMBLY_NEUTRAL_TIERS = frozenset(
    {
        "neutral_default",
        "source_preserve_unavailable_neutral_fallback",
    }
)
AUTHORITATIVE_CANONICAL_GROUP_LOCK_REASON = (
    "BASELINE_AUTHORITATIVE_CANONICAL_GROUP_LOCKED"
)
DOMINANT_RESIDUAL_SINGLE_VIEW_LANE = "dominant_chromatic_residual_exact_single_view"
DARK_FOREGROUND_RESIDUAL_LANE = "dark_foreground_achromatic_residual_exact_projection"
MULTIVIEW_DARK_IDENTITY_LANE = "multiview_dark_part_identity_consensus"
REPEATED_GEOMETRY_DARK_RESIDUAL_LANE = (
    "repeated_geometry_dark_residual_exact_projection"
)
_DARK_RESIDUAL_LANES = frozenset(
    {
        DARK_FOREGROUND_RESIDUAL_LANE,
        MULTIVIEW_DARK_IDENTITY_LANE,
    }
)
DARK_FOREGROUND_RESIDUAL_DEFICIT_SOURCE = "dark_foreground_achromatic_residual"
DARK_FOREGROUND_MIN_ALIGNMENT_SCORE = 0.85
DARK_FOREGROUND_MIN_SILHOUETTE_IOU = 0.85
DARK_FOREGROUND_MIN_EDGE_F1 = 0.85
DARK_FOREGROUND_MIN_PROFILE_SIMILARITY = 0.90
DARK_FOREGROUND_MIN_BBOX_ASPECT_SIMILARITY = 0.90
DARK_FOREGROUND_MIN_DEFICIT_SHARE = 0.025
DARK_FOREGROUND_MAX_MASS_RECALL = 0.80
DARK_FOREGROUND_MAX_SINGLE_CONTRIBUTION_FACTOR = 1.25
DARK_FOREGROUND_MAX_TOTAL_CONTRIBUTION_FACTOR = 1.35
REPEATED_DARK_MIN_COHORT_SIZE = 2
REPEATED_DARK_MAX_COHORT_SIZE = 4
REPEATED_DARK_MAX_REGISTRY_FRACTION = 0.05
REPEATED_DARK_MIN_ALIGNMENT_SCORE = 0.80
REPEATED_DARK_MIN_PROJECTION_SCORE = 0.85
REPEATED_DARK_MIN_PROJECTION_IOU = 0.85
REPEATED_DARK_MIN_ECC_CORRELATION = 0.90
REPEATED_DARK_MIN_PROJECTED_PIXELS = 256
REPEATED_DARK_MIN_DIRECT_SHARE = 0.60
REPEATED_DARK_MIN_DIRECT_MARGIN = 0.25
REPEATED_DARK_MIN_BBOX_SHARE = 0.90
REPEATED_DARK_MIN_BBOX_MARGIN = 0.80
REPEATED_DARK_MIN_PERTURBATION_SHARE = 0.60
REPEATED_DARK_MIN_PERTURBATION_MARGIN = 0.20
REPEATED_DARK_MIN_MAPPING_CONFIDENCE = 0.60
REPEATED_DARK_MIN_BUDGET_FACTOR = 0.75
REPEATED_DARK_MAX_BUDGET_FACTOR = 1.35
REPEATED_DARK_ALLOWED_DIAGNOSTIC_REASONS = frozenset(
    {"DARK_CANONICAL_GROUP_CONFLICT", "DARK_ALIGNMENT_NOT_STRONG"}
)
DARK_FOREGROUND_DIAGNOSTIC_THRESHOLDS: dict[str, float | int] = {
    "normalized_long_edge_pixels": 512,
    "minimum_normalized_projected_pixels": 96,
    "near_black_max_channel_exclusive": 97,
    "near_black_max_channel_spread": 32,
    "minimum_near_black_share": 0.60,
    "minimum_non_background_pixels": 24,
    "minimum_dark_signal_share": 0.20,
    "minimum_dark_signal_purity": 0.45,
    "core_distance_pixels": 2.2,
    "minimum_core_pixels": 16,
    "minimum_core_dark_signal_share": 0.25,
    "minimum_adaptive_edge_density": 0.25,
    "minimum_null_offset_pixels": 7,
    "minimum_null_valid_area_ratio": 0.80,
    "minimum_valid_null_shifts": 4,
    "minimum_null_q75_margin": 0.10,
    "minimum_alignment_score": 0.85,
    "minimum_projection_score": 0.85,
    "minimum_projection_iou": 0.85,
    "minimum_ecc_correlation": 0.90,
}
ANCHORED_SINGLE_MIN_PROJECTED_PIXELS = 256
ANCHORED_SINGLE_MIN_COLOR_SHARE = 0.85
ANCHORED_SINGLE_MIN_PERTURBATION_COLOR_SHARE = 0.80
ANCHORED_SINGLE_MIN_COLOR_MARGIN = 0.70
SLENDER_DIRECT_BOX_MIN_COLOR_SHARE = 0.65
SLENDER_DIRECT_BOX_MIN_COLOR_MARGIN = 0.60
SLENDER_DIRECT_BOX_MIN_EVIDENCE_OVERLAP_SHARE = 0.25
DOMINANT_RESIDUAL_MIN_ALIGNMENT_SCORE = 0.82
DOMINANT_RESIDUAL_MAX_SILHOUETTE_GAP = 0.025
DOMINANT_RESIDUAL_MIN_EDGE_F1 = 0.85
DOMINANT_RESIDUAL_MIN_PROFILE_SIMILARITY = 0.90
DOMINANT_RESIDUAL_MIN_BBOX_ASPECT_SIMILARITY = 0.90
DOMINANT_RESIDUAL_MIN_PROJECTED_PIXELS = 4096
DOMINANT_RESIDUAL_MIN_RENDER_FOREGROUND_SHARE = 0.20
DOMINANT_RESIDUAL_MIN_PROJECTION_IOU = 0.80
DOMINANT_RESIDUAL_MIN_ECC_CORRELATION = 0.85
DOMINANT_RESIDUAL_MIN_COLOR_SHARE = 0.95
DOMINANT_RESIDUAL_MIN_COLOR_MARGIN = 0.90
DOMINANT_RESIDUAL_MIN_SEMANTIC_CONFIDENCE = 0.90
DOMINANT_RESIDUAL_REVIEW_OVERRIDE_MIN_ANCHOR_CONFIDENCE = 0.80
DOMINANT_RESIDUAL_DEFICIT_SOURCE = "dominant_mass_local_projection"
_CHROMATIC_DOMINANT_BINS = frozenset(
    {"red", "orange_brown", "yellow", "green", "cyan_blue", "purple"}
)
SEMANTIC_OVERRIDE_SINGLE_VIEW_LANE = "exact_spatial_single_qa_view_with_semantic_anchor"
SEMANTIC_OVERRIDE_MIN_ALIGNMENT_SCORE = 0.75
SEMANTIC_OVERRIDE_MIN_PROJECTION_IOU = 0.80
SEMANTIC_OVERRIDE_MIN_ECC_CORRELATION = 0.85
SEMANTIC_OVERRIDE_MIN_PROJECTED_PIXELS = 1024
SEMANTIC_OVERRIDE_MIN_COLOR_SHARE = 0.70
SEMANTIC_OVERRIDE_MIN_COLOR_MARGIN = 0.60
SEMANTIC_OVERRIDE_MIN_BBOX_COLOR_SHARE = 0.65
SEMANTIC_OVERRIDE_MIN_BBOX_COLOR_MARGIN = 0.55
SEMANTIC_OVERRIDE_MAX_REJECTED_GROUP_SHARE = 0.05
SEMANTIC_OVERRIDE_MAX_REJECTED_GROUP_OUTLIER_SHARE = 0.15
SEMANTIC_OVERRIDE_MIN_CLEAN_REJECTED_GROUP_SAMPLES = 5
REPAIR_REASON_CODES = (
    "QA_MISSING_CANONICAL_GROUP_MULTI_VIEW",
    "QA_TRUSTED_PART_GROUP_LOCALIZATION",
    "QA_CONFIRMED_WHITELIST_MATERIAL",
)
PROVISIONAL_REPAIR_REASON_CODES = (
    "QA_MISSING_CANONICAL_GROUP_MULTI_VIEW",
    "QA_TRUSTED_PART_GROUP_LOCALIZATION",
    "QA_HIGH_CONFIDENCE_WHITELIST_MATERIAL_CANDIDATE",
    "QA_POST_RENDER_VALIDATION_REQUIRED",
)
DARK_FOREGROUND_REPAIR_REASON_CODES = (
    "QA_DARK_FOREGROUND_ACHROMATIC_RESIDUAL",
    "QA_TRUSTED_PART_GROUP_LOCALIZATION",
    "QA_CONFIRMED_WHITELIST_MATERIAL",
)
REPEATED_GEOMETRY_DARK_REPAIR_REASON_CODES = (
    "QA_MISSING_CANONICAL_GROUP_SINGLE_VIEW",
    "QA_REPEATED_GEOMETRY_COHORT_EXACT_PROJECTION",
    "QA_CONFIRMED_WHITELIST_MATERIAL",
)
DOMINANT_ASSEMBLY_REPAIR_REASON_CODES = (
    "QA_DOMINANT_CHROMATIC_GROUP_DEFICIT",
    "QA_STRICT_MULTIVIEW_SPATIAL_ASSEMBLY_ANCHORS",
    "QA_HASH_BOUND_ASSEMBLY_COHORT_EXPANSION",
    "QA_POST_RENDER_MEMBERSHIP_VALIDATION_REQUIRED",
)
BOUNDED_SIGNATURE_SIBLING_REPAIR_REASON_CODES = (
    "QA_MISSING_CANONICAL_GROUP_MULTI_VIEW",
    "QA_EXACT_DIAGNOSTIC_SOURCE_SIGNATURE_ANCHOR",
    "QA_BOUNDED_SOURCE_SIGNATURE_SIBLING",
    "QA_POST_RENDER_MEMBERSHIP_VALIDATION_REQUIRED",
)

POLICY_FALLBACK_STATUS = "policy_fallback"
MIN_PROVISIONAL_MATERIAL_CONFIDENCE = 0.85
MIN_CONFIRMED_MATERIAL_CONFIDENCE = 0.60
NEUTRAL_FALLBACK_TIERS = frozenset(
    {
        "neutral_default",
        "source_preserve_unavailable_neutral_fallback",
        SOURCE_VISUAL_PRESERVE_TIER,
    }
)
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_CONFLICT_FIELDS = (
    "conflicting_view_ids",
    "semantic_conflicting_view_ids",
    "semantic_unresolved_view_ids",
    "semantic_multi_material_view_ids",
    "semantic_nondeterministic_content_cluster_ids",
)


class QualityRepairError(ValueError):
    """Raised when a quality-repair input violates its trust contract."""


def _observation_diagnostic_floor(
    observation: Mapping[str, Any],
    spatial_policy: Mapping[str, Any],
) -> int | None:
    """Return the authenticated source-pixel floor for one observation."""

    raw_floor = spatial_policy.get(
        "minimum_diagnostic_visible_pixels",
        spatial_policy.get("minimum_visible_pixels"),
    )
    if isinstance(raw_floor, bool) or not isinstance(raw_floor, int) or raw_floor < 1:
        return None
    if observation.get("evidence_mode") != "isolated_mask_multiview_diagnostic":
        return raw_floor
    isolated_floor = spatial_policy.get("minimum_isolated_source_visible_pixels")
    minimum_view_count = spatial_policy.get("minimum_isolated_source_view_count")
    source_view_count = observation.get("isolated_source_view_count")
    digest = observation.get("isolated_evidence_sha256")
    if (
        isinstance(isolated_floor, bool)
        or not isinstance(isolated_floor, int)
        or isolated_floor < 1
        or isolated_floor > raw_floor
        or isinstance(minimum_view_count, bool)
        or not isinstance(minimum_view_count, int)
        or minimum_view_count < 2
        or isinstance(source_view_count, bool)
        or not isinstance(source_view_count, int)
        or source_view_count < minimum_view_count
        or not isinstance(digest, str)
        or _SHA256_PATTERN.fullmatch(digest) is None
    ):
        return raw_floor
    return isolated_floor


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise QualityRepairError(f"{label} must be an object")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise QualityRepairError(f"{label} must be an array")
    return value


def _explicitly_invisible_semantic_view_ids(
    *,
    spatial_part: Mapping[str, Any],
    spatial_policy: Mapping[str, Any],
    part_id: str,
) -> set[str]:
    """Return views whose rendered part mask is explicitly too small to trust.

    Qwen can still emit a per-view semantic label when a part is fully
    occluded in the CAD pose used for that reference.  Such a label is not
    independent contradictory evidence about that part.  Only explicit
    visibility measurements below the diagnostic floor are suppressed; a
    missing or invalid measurement remains fail-closed.
    """

    invisible: set[str] = set()
    for raw_observation in _sequence(
        spatial_part.get("observations", []),
        f"spatial part {part_id}.observations",
    ):
        if not isinstance(raw_observation, Mapping):
            continue
        view_id = raw_observation.get("reference_view_id")
        declared_pixels = raw_observation.get("declared_visible_pixels")
        raw_floor = _observation_diagnostic_floor(raw_observation, spatial_policy)
        if (
            isinstance(view_id, str)
            and view_id
            and isinstance(raw_floor, int)
            and isinstance(declared_pixels, int)
            and not isinstance(declared_pixels, bool)
            and 0 <= declared_pixels < raw_floor
        ):
            invisible.add(view_id)
    return invisible


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise QualityRepairError(f"{label} must be a non-empty string")
    return value.strip()


def _unit(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise QualityRepairError(f"{label} must be a finite number from 0 to 1")
    return float(value)


def _optional_unit(value: Any) -> float | None:
    """Return one finite unit interval value, or ``None`` when unavailable."""

    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        return None
    return float(value)


def _string_array(value: Any, label: str) -> list[str]:
    result = [
        _text(item, f"{label}[{index}]")
        for index, item in enumerate(_sequence(value, label))
    ]
    if len(result) != len(set(result)):
        raise QualityRepairError(f"{label} must not contain duplicates")
    return result


def _canonical_sha256(value: Any) -> str:
    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise QualityRepairError(f"input is not canonical JSON: {exc}") from exc
    return hashlib.sha256(payload).hexdigest()


def _require_schema(document: Mapping[str, Any], expected: str, label: str) -> None:
    if document.get("schema_version") != expected:
        raise QualityRepairError(
            f"{label} has an unsupported schema_version: "
            f"{document.get('schema_version')!r}"
        )


def _registry_part_ids(registry: Mapping[str, Any]) -> list[str]:
    _require_schema(registry, REGISTRY_SCHEMA_VERSION, "registry")
    part_ids: list[str] = []
    for index, raw_part in enumerate(
        _sequence(registry.get("parts"), "registry.parts")
    ):
        part = _mapping(raw_part, f"registry.parts[{index}]")
        part_ids.append(_text(part.get("part_id"), f"registry.parts[{index}].part_id"))
    if not part_ids or len(part_ids) != len(set(part_ids)):
        raise QualityRepairError(
            "registry.parts must contain unique non-empty part IDs"
        )
    part_count = registry.get("part_count")
    if (
        isinstance(part_count, bool)
        or not isinstance(part_count, int)
        or part_count != len(part_ids)
    ):
        raise QualityRepairError("registry.part_count does not match registry.parts")
    return part_ids


def _whitelist_ids(whitelist: Mapping[str, Any]) -> set[str]:
    if whitelist.get("schema_version") != 1:
        raise QualityRepairError("whitelist.schema_version must be 1")
    material_ids = _string_array(
        whitelist.get("material_ids"), "whitelist.material_ids"
    )
    if not material_ids:
        raise QualityRepairError("whitelist.material_ids cannot be empty")
    return set(material_ids)


def _baseline_assignments(
    baseline_plan: Mapping[str, Any],
    *,
    registry_part_ids: Sequence[str],
    whitelist_ids: set[str],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    if baseline_plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise QualityRepairError("baseline plan schema_version must be '1.0'")
    raw_assignments = _sequence(
        baseline_plan.get("assignments"), "baseline_plan.assignments"
    )
    assignments: list[dict[str, Any]] = []
    by_part: dict[str, dict[str, Any]] = {}
    for index, raw_assignment in enumerate(raw_assignments):
        assignment = copy.deepcopy(
            dict(_mapping(raw_assignment, f"baseline_plan.assignments[{index}]"))
        )
        part_id = _text(
            assignment.get("part_id"),
            f"baseline_plan.assignments[{index}].part_id",
        )
        if part_id in by_part:
            raise QualityRepairError(f"duplicate baseline assignment: {part_id}")
        material_id = _text(
            assignment.get("material_id"),
            f"baseline assignment {part_id}.material_id",
        )
        if material_id not in whitelist_ids:
            raise QualityRepairError(
                f"baseline assignment {part_id} is outside the whitelist"
            )
        by_part[part_id] = assignment
        assignments.append(assignment)
    if set(by_part) != set(registry_part_ids):
        missing = sorted(set(registry_part_ids) - set(by_part))
        unexpected = sorted(set(by_part) - set(registry_part_ids))
        raise QualityRepairError(
            "baseline plan is not an exact cover of the registry: "
            f"missing={missing[:20]} unexpected={unexpected[:20]}"
        )
    return assignments, by_part


def _validate_policy_audit(
    baseline_plan: Mapping[str, Any],
    policy_audit: Mapping[str, Any],
    *,
    registry_part_count: int,
) -> None:
    _require_schema(policy_audit, POLICY_AUDIT_SCHEMA_VERSION, "baseline policy audit")
    baseline_hash = _canonical_sha256(baseline_plan)
    if policy_audit.get("output_plan_sha256") != baseline_hash:
        raise QualityRepairError(
            "baseline policy audit output hash does not match baseline plan"
        )
    provenance = _mapping(baseline_plan.get("provenance"), "baseline_plan.provenance")
    if provenance.get("mode") != "explicit_best_effort_policy_exact_cover":
        raise QualityRepairError(
            "quality repair requires an exact-cover policy baseline"
        )
    if policy_audit.get("input_hashes") != provenance:
        raise QualityRepairError(
            "baseline policy audit input hashes do not match plan provenance"
        )
    summary = _mapping(policy_audit.get("summary"), "baseline policy audit summary")
    if (
        summary.get("exact_cover") is not True
        or summary.get("all_materials_in_industrial_whitelist") is not True
        or summary.get("output_assignment_count") != registry_part_count
        or summary.get("registry_part_count") != registry_part_count
    ):
        raise QualityRepairError(
            "baseline policy audit does not prove an exact whitelisted cover"
        )


def _canonical_groups(
    palette_fusion: Mapping[str, Any],
) -> tuple[dict[str, Mapping[str, Any]], dict[str, dict[str, str]]]:
    _require_schema(palette_fusion, PALETTE_FUSION_SCHEMA_VERSION, "palette fusion")
    canonical_palette = _mapping(
        palette_fusion.get("canonical_palette"),
        "palette_fusion.canonical_palette",
    )
    groups: dict[str, Mapping[str, Any]] = {}
    for index, raw_group in enumerate(
        _sequence(canonical_palette.get("groups"), "canonical_palette.groups")
    ):
        group = _mapping(raw_group, f"canonical_palette.groups[{index}]")
        group_id = _text(
            group.get("group_id"), f"canonical_palette.groups[{index}].group_id"
        )
        if group_id in groups:
            raise QualityRepairError(f"duplicate canonical group: {group_id}")
        _text(group.get("base_color"), f"canonical group {group_id}.base_color")
        distinct_views = group.get("distinct_view_count")
        if (
            isinstance(distinct_views, bool)
            or not isinstance(distinct_views, int)
            or distinct_views < 1
        ):
            raise QualityRepairError(
                f"canonical group {group_id}.distinct_view_count is invalid"
            )
        groups[group_id] = group
    if not groups:
        raise QualityRepairError("canonical palette has no groups")

    raw_maps = _mapping(
        palette_fusion.get("view_group_id_maps"),
        "palette_fusion.view_group_id_maps",
    )
    view_maps: dict[str, dict[str, str]] = {}
    for raw_view_id, raw_group_map in raw_maps.items():
        view_id = _text(raw_view_id, "view_group_id_maps view ID")
        group_map = _mapping(raw_group_map, f"view_group_id_maps[{view_id}]")
        normalized: dict[str, str] = {}
        for raw_local_id, raw_canonical_id in group_map.items():
            local_id = _text(raw_local_id, f"{view_id} local group ID")
            canonical_id = _text(
                raw_canonical_id, f"{view_id}/{local_id} canonical group ID"
            )
            if canonical_id not in groups:
                raise QualityRepairError(
                    f"{view_id}/{local_id} maps to unknown canonical group "
                    f"{canonical_id}"
                )
            normalized[local_id] = canonical_id
        view_maps[view_id] = normalized
    return groups, view_maps


def _reference_evidence(
    spatial_report: Mapping[str, Any],
) -> dict[str, dict[str, str]]:
    _require_schema(spatial_report, SPATIAL_REPORT_SCHEMA_VERSION, "spatial report")
    integrity = _mapping(spatial_report.get("integrity"), "spatial_report.integrity")
    expected_hash = _text(
        integrity.get("report_sha256"), "spatial_report.integrity.report_sha256"
    )
    unsigned = copy.deepcopy(dict(spatial_report))
    unsigned.pop("integrity", None)
    if _canonical_sha256(unsigned) != expected_hash:
        raise QualityRepairError("spatial report integrity hash mismatch")

    raw_records = spatial_report.get("reference_evidence")
    if not isinstance(raw_records, Sequence) or isinstance(raw_records, (str, bytes)):
        # Older reports do not contain content/pose clusters.  They remain
        # valid audit inputs, but cannot safely authorize a repair.
        return {}
    records: dict[str, dict[str, str]] = {}
    for index, raw_record in enumerate(raw_records):
        record = _mapping(raw_record, f"spatial_report.reference_evidence[{index}]")
        view_id = _text(
            record.get("view_id"),
            f"spatial_report.reference_evidence[{index}].view_id",
        )
        if view_id in records:
            raise QualityRepairError(f"duplicate spatial reference evidence: {view_id}")
        if record.get("alignment_trusted") is not True:
            continue
        raw_sha = _text(
            record.get("raw_sha256"), f"reference evidence {view_id}.raw_sha256"
        )
        if not _SHA256_PATTERN.fullmatch(raw_sha):
            raise QualityRepairError(
                f"reference evidence {view_id} has invalid raw_sha256"
            )
        normalized_sha = _text(
            record.get("normalized_pixel_sha256"),
            f"reference evidence {view_id}.normalized_pixel_sha256",
        )
        if not _SHA256_PATTERN.fullmatch(normalized_sha):
            raise QualityRepairError(
                f"reference evidence {view_id} has invalid normalized_pixel_sha256"
            )
        content_cluster = _text(
            record.get("content_cluster_id"),
            f"reference evidence {view_id}.content_cluster_id",
        )
        pose_cluster = _text(
            record.get("pose_cluster_id"),
            f"reference evidence {view_id}.pose_cluster_id",
        )
        selected_render = _text(
            record.get("selected_render_view_id"),
            f"reference evidence {view_id}.selected_render_view_id",
        )
        if pose_cluster != selected_render:
            raise QualityRepairError(
                f"reference evidence {view_id} pose/render mismatch"
            )
        records[view_id] = {
            "raw_sha256": raw_sha,
            "normalized_pixel_sha256": normalized_sha,
            "content_cluster_id": content_cluster,
            "pose_cluster_id": pose_cluster,
        }
    return records


def _all_reference_evidence(
    spatial_report: Mapping[str, Any],
) -> dict[str, dict[str, str]]:
    """Return hash/content evidence even for views not aligned to a CAD pose.

    An unaligned view cannot localize a part, but it can remain an independent
    source of a canonical multi-view palette group.  The spatial report
    integrity is validated by ``_reference_evidence`` before this parser is
    called.
    """

    raw_records = spatial_report.get("reference_evidence")
    if not isinstance(raw_records, Sequence) or isinstance(raw_records, (str, bytes)):
        return {}
    records: dict[str, dict[str, str]] = {}
    for index, raw_record in enumerate(raw_records):
        record = _mapping(raw_record, f"spatial_report.reference_evidence[{index}]")
        view_id = _text(
            record.get("view_id"),
            f"spatial_report.reference_evidence[{index}].view_id",
        )
        if view_id in records:
            raise QualityRepairError(f"duplicate spatial reference evidence: {view_id}")
        raw_sha = _text(
            record.get("raw_sha256"), f"reference evidence {view_id}.raw_sha256"
        )
        normalized_sha = _text(
            record.get("normalized_pixel_sha256"),
            f"reference evidence {view_id}.normalized_pixel_sha256",
        )
        if not _SHA256_PATTERN.fullmatch(raw_sha) or not _SHA256_PATTERN.fullmatch(
            normalized_sha
        ):
            raise QualityRepairError(
                f"reference evidence {view_id} has an invalid image hash"
            )
        records[view_id] = {
            "raw_sha256": raw_sha,
            "normalized_pixel_sha256": normalized_sha,
            "content_cluster_id": _text(
                record.get("content_cluster_id"),
                f"reference evidence {view_id}.content_cluster_id",
            ),
        }
    return records


def _trusted_alignment_audits(
    spatial_report: Mapping[str, Any],
    *,
    reference_evidence: Mapping[str, Mapping[str, str]],
) -> dict[str, dict[str, Any]]:
    """Return validated refined registration metrics for trusted references."""

    audits: dict[str, dict[str, Any]] = {}
    raw_alignments = spatial_report.get("view_alignments")
    if not isinstance(raw_alignments, Sequence) or isinstance(
        raw_alignments, (str, bytes)
    ):
        return audits
    for index, raw_alignment in enumerate(raw_alignments):
        alignment = _mapping(raw_alignment, f"spatial_report.view_alignments[{index}]")
        view_id = _text(
            alignment.get("reference_view_id"),
            f"spatial_report.view_alignments[{index}].reference_view_id",
        )
        if view_id in audits:
            raise QualityRepairError(f"duplicate spatial view alignment: {view_id}")
        if alignment.get("trusted") is not True or view_id not in reference_evidence:
            continue
        render_view_id = _text(
            alignment.get("selected_render_view_id"),
            f"spatial alignment {view_id}.selected_render_view_id",
        )
        if render_view_id != reference_evidence[view_id]["pose_cluster_id"]:
            raise QualityRepairError(
                f"spatial alignment {view_id} disagrees with reference evidence"
            )
        audits[view_id] = {
            "selected_render_view_id": render_view_id,
            "score": _unit(
                alignment.get("score"), f"spatial alignment {view_id}.score"
            ),
            "projection_iou": _unit(
                alignment.get("projection_iou"),
                f"spatial alignment {view_id}.projection_iou",
            ),
            "ecc_correlation": _unit(
                alignment.get("ecc_correlation"),
                f"spatial alignment {view_id}.ecc_correlation",
            ),
            "ecc_status": alignment.get("ecc_status"),
        }
    return audits


_DOMINANT_THRESHOLD_FIELDS = (
    "minimum_dominant_reference_share",
    "minimum_dominant_share_margin",
    "minimum_dominant_mass_recall",
    "minimum_dominant_absolute_deficit",
    "minimum_dominant_silhouette_iou",
)


def _dominant_thresholds(
    thresholds: Mapping[str, Any],
) -> dict[str, float] | None:
    present = {field for field in _DOMINANT_THRESHOLD_FIELDS if field in thresholds}
    if not present:
        return None
    if present != set(_DOMINANT_THRESHOLD_FIELDS):
        raise QualityRepairError(
            "quality dominant-mass threshold contract is incomplete"
        )
    return {
        field: _unit(thresholds.get(field), f"quality threshold {field}")
        for field in _DOMINANT_THRESHOLD_FIELDS
    }


def _dominant_mass_failures(
    *,
    view: Mapping[str, Any],
    color: Mapping[str, Any],
    alignment: Mapping[str, Any],
    thresholds: Mapping[str, Any],
    dominant_thresholds: Mapping[str, float] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate and return hard-failing dominant-family records.

    Reports without the complete threshold contract are accepted only as
    legacy inputs and must not contain dominant-mass fields or reason codes.
    Once enabled, every numeric field and decision is independently
    recomputed from the report's colour distributions and alignment audit.
    """

    raw_dominant = color.get("trusted_evidence_dominant_mass")
    view_reasons = view.get("reasons")
    if not isinstance(view_reasons, Sequence) or isinstance(view_reasons, (str, bytes)):
        raise QualityRepairError("quality view reasons must be an array")
    claims_failure = "trusted_dominant_family_mass_deficit" in view_reasons
    if dominant_thresholds is None:
        if raw_dominant is not None or claims_failure:
            raise QualityRepairError(
                "quality dominant-mass evidence lacks its threshold contract"
            )
        return [], []
    dominant = _mapping(
        raw_dominant,
        "quality view material_color.trusted_evidence_dominant_mass",
    )
    families = _sequence(
        dominant.get("families"),
        "quality dominant-mass families",
    )
    reference_distribution = _mapping(
        color.get("reference_distribution"),
        "quality dominant-mass reference_distribution",
    )
    render_distribution = _mapping(
        color.get("render_distribution"),
        "quality dominant-mass render_distribution",
    )
    reference_shares = _mapping(
        reference_distribution.get("category_distribution"),
        "quality dominant-mass reference category_distribution",
    )
    render_shares = _mapping(
        render_distribution.get("category_distribution"),
        "quality dominant-mass render category_distribution",
    )
    alignment_score = _unit(
        alignment.get("score"), "quality dominant-mass alignment.score"
    )
    silhouette_iou = _unit(
        alignment.get("silhouette_iou"),
        "quality dominant-mass alignment.silhouette_iou",
    )
    edge_f1 = _optional_unit(alignment.get("edge_f1_tolerance_3px"))
    profile_similarity = _optional_unit(alignment.get("profile_similarity"))
    bbox_aspect_similarity = _optional_unit(alignment.get("bbox_aspect_similarity"))
    render = view.get("render")
    foreground = render.get("foreground") if isinstance(render, Mapping) else None
    raw_render_foreground_pixels = (
        foreground.get("pixel_count") if isinstance(foreground, Mapping) else None
    )
    render_foreground_pixels = (
        int(raw_render_foreground_pixels)
        if isinstance(raw_render_foreground_pixels, int)
        and not isinstance(raw_render_foreground_pixels, bool)
        and raw_render_foreground_pixels > 0
        else None
    )
    strong_alignment = _unit(
        thresholds.get("strong_alignment_score"),
        "quality threshold strong_alignment_score",
    )

    parsed: list[dict[str, Any]] = []
    family_keys: set[str] = set()
    for index, raw_family in enumerate(families):
        family = _mapping(raw_family, f"quality dominant-mass families[{index}]")
        bins = sorted(
            _string_array(
                family.get("render_color_bins"),
                f"quality dominant-mass families[{index}].render_color_bins",
            )
        )
        if not bins or family.get("render_color_bins") != bins:
            raise QualityRepairError(
                "quality dominant-mass render_color_bins must be sorted and non-empty"
            )
        family_key = _text(
            family.get("family_key"),
            f"quality dominant-mass families[{index}].family_key",
        )
        if family_key != "|".join(bins) or family_key in family_keys:
            raise QualityRepairError("quality dominant-mass family identity is invalid")
        family_keys.add(family_key)
        local_group_ids = sorted(
            _string_array(
                family.get("local_group_ids"),
                f"quality dominant-mass families[{index}].local_group_ids",
            )
        )
        base_colors = sorted(
            _string_array(
                family.get("base_colors"),
                f"quality dominant-mass families[{index}].base_colors",
            )
        )
        if (
            not local_group_ids
            or family.get("local_group_ids") != local_group_ids
            or not base_colors
            or family.get("base_colors") != base_colors
        ):
            raise QualityRepairError(
                "quality dominant-mass group/color identity is invalid"
            )
        try:
            expected_reference_share = sum(
                _unit(
                    reference_shares[label],
                    f"quality dominant-mass reference share {label}",
                )
                for label in bins
            )
            expected_observed_share = sum(
                _unit(
                    render_shares[label],
                    f"quality dominant-mass render share {label}",
                )
                for label in bins
            )
        except KeyError as exc:
            raise QualityRepairError(
                f"quality dominant-mass bin is absent from a distribution: {exc}"
            ) from exc
        if (
            expected_reference_share > 1.0 + 1e-9
            or expected_observed_share > 1.0 + 1e-9
        ):
            raise QualityRepairError("quality dominant-mass family shares exceed one")
        reference_share = _unit(
            family.get("reference_share"),
            f"quality dominant-mass families[{index}].reference_share",
        )
        runner_up_share = _unit(
            family.get("runner_up_reference_share"),
            f"quality dominant-mass families[{index}].runner_up_reference_share",
        )
        observed_share = _unit(
            family.get("observed_render_share"),
            f"quality dominant-mass families[{index}].observed_render_share",
        )
        deficit_share = _unit(
            family.get("deficit_share"),
            f"quality dominant-mass families[{index}].deficit_share",
        )
        mass_recall = _unit(
            family.get("mass_recall"),
            f"quality dominant-mass families[{index}].mass_recall",
        )
        raw_margin = family.get("reference_share_margin")
        if (
            isinstance(raw_margin, bool)
            or not isinstance(raw_margin, (int, float))
            or not math.isfinite(float(raw_margin))
            or not -1.0 <= float(raw_margin) <= 1.0
        ):
            raise QualityRepairError(
                "quality dominant-mass reference_share_margin is invalid"
            )
        parsed.append(
            {
                "raw": family,
                "family_key": family_key,
                "local_group_ids": local_group_ids,
                "base_colors": base_colors,
                "bins": bins,
                "reference_share": reference_share,
                "runner_up_share": runner_up_share,
                "reference_share_margin": float(raw_margin),
                "observed_share": observed_share,
                "deficit_share": deficit_share,
                "mass_recall": mass_recall,
                "expected_reference_share": expected_reference_share,
                "expected_observed_share": expected_observed_share,
            }
        )

    failures: list[dict[str, Any]] = []
    local_projection_residuals: list[dict[str, Any]] = []
    eligible_count = 0
    for item in parsed:
        runner_up = max(
            (
                other["reference_share"]
                for other in parsed
                if other["family_key"] != item["family_key"]
            ),
            default=0.0,
        )
        reference_share = item["reference_share"]
        observed_share = item["observed_share"]
        expected_deficit = max(0.0, reference_share - observed_share)
        expected_recall = (
            min(1.0, observed_share / reference_share) if reference_share > 0.0 else 1.0
        )
        expected_margin = reference_share - runner_up
        numeric_pairs = (
            (reference_share, item["expected_reference_share"]),
            (observed_share, item["expected_observed_share"]),
            (item["runner_up_share"], runner_up),
            (item["reference_share_margin"], expected_margin),
            (item["deficit_share"], expected_deficit),
            (item["mass_recall"], expected_recall),
        )
        if any(
            not math.isclose(left, right, rel_tol=0.0, abs_tol=1e-9)
            for left, right in numeric_pairs
        ):
            raise QualityRepairError(
                "quality dominant-mass numeric evidence is inconsistent"
            )
        eligibility_reasons: list[str] = []
        if reference_share < dominant_thresholds["minimum_dominant_reference_share"]:
            eligibility_reasons.append("REFERENCE_SHARE_BELOW_DOMINANT_FLOOR")
        if expected_margin < dominant_thresholds["minimum_dominant_share_margin"]:
            eligibility_reasons.append("REFERENCE_DOMINANCE_MARGIN_BELOW_FLOOR")
        if alignment_score < strong_alignment:
            eligibility_reasons.append("ALIGNMENT_NOT_STRONG")
        if silhouette_iou < dominant_thresholds["minimum_dominant_silhouette_iou"]:
            eligibility_reasons.append("SILHOUETTE_IOU_BELOW_DOMINANT_FLOOR")
        eligible = not eligibility_reasons
        hard_failure = (
            eligible
            and item["mass_recall"]
            < dominant_thresholds["minimum_dominant_mass_recall"]
            and item["deficit_share"]
            >= dominant_thresholds["minimum_dominant_absolute_deficit"]
        )
        expected_status = (
            "FAIL" if hard_failure else "PASS" if eligible else "NOT_APPLICABLE"
        )
        expected_reasons = (
            ["DOMINANT_FAMILY_MASS_DEFICIT"] if hard_failure else eligibility_reasons
        )
        raw = item["raw"]
        if (
            raw.get("eligible") is not eligible
            or raw.get("status") != expected_status
            or raw.get("reason_codes") != expected_reasons
        ):
            raise QualityRepairError("quality dominant-mass decision is inconsistent")
        if eligible:
            eligible_count += 1
        if hard_failure:
            failures.append(item)
        local_projection_residual = (
            not eligible
            and eligibility_reasons == ["SILHOUETTE_IOU_BELOW_DOMINANT_FLOOR"]
            and alignment_score
            >= max(
                strong_alignment,
                DOMINANT_RESIDUAL_MIN_ALIGNMENT_SCORE,
            )
            and (
                dominant_thresholds["minimum_dominant_silhouette_iou"] - silhouette_iou
                <= DOMINANT_RESIDUAL_MAX_SILHOUETTE_GAP
            )
            and edge_f1 is not None
            and edge_f1 >= DOMINANT_RESIDUAL_MIN_EDGE_F1
            and profile_similarity is not None
            and profile_similarity >= DOMINANT_RESIDUAL_MIN_PROFILE_SIMILARITY
            and bbox_aspect_similarity is not None
            and bbox_aspect_similarity >= DOMINANT_RESIDUAL_MIN_BBOX_ASPECT_SIMILARITY
            and render_foreground_pixels is not None
            and set(item["bins"]) <= _CHROMATIC_DOMINANT_BINS
            and item["mass_recall"]
            < dominant_thresholds["minimum_dominant_mass_recall"]
            and item["deficit_share"]
            >= dominant_thresholds["minimum_dominant_absolute_deficit"]
        )
        if local_projection_residual:
            item.update(
                {
                    "alignment_score": alignment_score,
                    "silhouette_iou": silhouette_iou,
                    "silhouette_floor": dominant_thresholds[
                        "minimum_dominant_silhouette_iou"
                    ],
                    "silhouette_gap": (
                        dominant_thresholds["minimum_dominant_silhouette_iou"]
                        - silhouette_iou
                    ),
                    "edge_f1_tolerance_3px": edge_f1,
                    "profile_similarity": profile_similarity,
                    "bbox_aspect_similarity": bbox_aspect_similarity,
                    "render_foreground_pixels": render_foreground_pixels,
                }
            )
            local_projection_residuals.append(item)

    expected_status = (
        "FAIL" if failures else "PASS" if eligible_count else "NOT_APPLICABLE"
    )
    if (
        dominant.get("status") != expected_status
        or dominant.get("eligible_family_count") != eligible_count
        or dominant.get("failed_family_count") != len(failures)
        or claims_failure != bool(failures)
        or (bool(failures) and view.get("status") != "FAIL")
    ):
        raise QualityRepairError(
            "quality dominant-mass summary/view decision is inconsistent"
        )
    return failures, local_projection_residuals


def _trusted_missing_groups(
    quality_report: Mapping[str, Any],
    *,
    canonical_groups: Mapping[str, Mapping[str, Any]],
    view_group_maps: Mapping[str, Mapping[str, str]],
    reference_evidence: Mapping[str, Mapping[str, str]],
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[str],
]:
    _require_schema(quality_report, QUALITY_SCHEMA_VERSION, "quality report")
    aggregate = _mapping(quality_report.get("aggregate"), "quality_report.aggregate")
    if aggregate.get("status") != "FAIL":
        return {}, {}, [], [], ["QUALITY_STATUS_NOT_FAIL"]
    thresholds = _mapping(quality_report.get("thresholds"), "quality_report.thresholds")
    group_recall_threshold = _unit(
        thresholds.get("minimum_evidence_group_recall"),
        "quality threshold minimum_evidence_group_recall",
    )
    strong_alignment = _unit(
        thresholds.get("strong_alignment_score"),
        "quality threshold strong_alignment_score",
    )
    dominant_threshold_contract = _dominant_thresholds(thresholds)

    raw_support: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    raw_residual_support: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    view_diagnostics: list[dict[str, Any]] = []
    seen_views: set[str] = set()
    dominant_failure_view_count = 0
    for index, raw_view in enumerate(
        _sequence(quality_report.get("views"), "quality_report.views")
    ):
        view = _mapping(raw_view, f"quality_report.views[{index}]")
        view_id = _text(
            view.get("reference_view_id"),
            f"quality_report.views[{index}].reference_view_id",
        )
        if view_id in seen_views:
            raise QualityRepairError(f"duplicate quality reference view: {view_id}")
        seen_views.add(view_id)
        reasons: list[str] = []
        view_status = view.get("status")
        if view_status != "FAIL":
            reasons.append("VIEW_STATUS_NOT_FAIL")
        render_view_id = view.get("render_view_id")
        if not isinstance(render_view_id, str) or not render_view_id:
            reasons.append("MISSING_RENDER_VIEW")
        mapping = view.get("mapping")
        if not isinstance(mapping, Mapping):
            reasons.append("MAPPING_AUDIT_MISSING")
        else:
            mapping_reasons = mapping.get("reasons")
            if not isinstance(mapping_reasons, Sequence) or isinstance(
                mapping_reasons, (str, bytes)
            ):
                reasons.append("MAPPING_REASONS_INVALID")
            elif list(mapping_reasons):
                reasons.append("MAPPING_NOT_TRUSTED")
            if mapping.get("selected_render_view_id") != render_view_id:
                reasons.append("MAPPING_RENDER_MISMATCH")
        alignment = view.get("alignment")
        if not isinstance(alignment, Mapping):
            reasons.append("ALIGNMENT_MISSING")
        else:
            score = alignment.get("score")
            if (
                isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not math.isfinite(float(score))
                or float(score) < strong_alignment
            ):
                reasons.append("ALIGNMENT_NOT_STRONG")
        reference = view.get("reference")
        image_sha: str | None = None
        if not isinstance(reference, Mapping):
            reasons.append("REFERENCE_AUDIT_MISSING")
        else:
            raw_sha = reference.get("image_sha256")
            if not isinstance(raw_sha, str) or not _SHA256_PATTERN.fullmatch(raw_sha):
                reasons.append("REFERENCE_SHA256_INVALID")
            else:
                image_sha = raw_sha
            trusted = reference.get("trusted_evidence")
            if not isinstance(trusted, Mapping) or trusted.get("usable") is not True:
                reasons.append("PALETTE_EVIDENCE_NOT_TRUSTED")
        spatial_reference = reference_evidence.get(view_id)
        if spatial_reference is None:
            reasons.append("CONTENT_POSE_EVIDENCE_MISSING")
        elif image_sha is not None and spatial_reference["raw_sha256"] != image_sha:
            reasons.append("REFERENCE_SHA256_SPATIAL_MISMATCH")
        elif render_view_id != spatial_reference["pose_cluster_id"]:
            reasons.append("QUALITY_SPATIAL_POSE_MISMATCH")

        color = view.get("material_color")
        group_recall: Mapping[str, Any] | None = None
        dominant_failures: list[dict[str, Any]] = []
        dominant_local_projection_residuals: list[dict[str, Any]] = []
        if not isinstance(color, Mapping):
            reasons.append("MATERIAL_COLOR_AUDIT_MISSING")
        else:
            raw_group_recall = color.get("trusted_evidence_group_recall")
            if isinstance(raw_group_recall, Mapping):
                group_recall = raw_group_recall
            else:
                reasons.append("GROUP_RECALL_MISSING")
            if isinstance(alignment, Mapping):
                (
                    dominant_failures,
                    dominant_local_projection_residuals,
                ) = _dominant_mass_failures(
                    view=view,
                    color=color,
                    alignment=alignment,
                    thresholds=thresholds,
                    dominant_thresholds=dominant_threshold_contract,
                )
                if dominant_failures:
                    dominant_failure_view_count += 1
        local_map = view_group_maps.get(view_id)
        if local_map is None:
            reasons.append("CANONICAL_GROUP_MAP_MISSING")
        residual_reasons = [
            reason for reason in reasons if reason != "VIEW_STATUS_NOT_FAIL"
        ]
        if view_status not in {"FAIL", "REVIEW"}:
            residual_reasons.append("VIEW_STATUS_NOT_FAIL_OR_REVIEW")

        missing_records: list[dict[str, Any]] = []
        pending_supports: list[tuple[str, dict[str, Any]]] = []
        dominant_mass_skips: list[dict[str, Any]] = []
        if not reasons and group_recall is not None and local_map is not None:
            for group_index, raw_group in enumerate(
                _sequence(
                    group_recall.get("groups"),
                    f"quality view {view_id} group recall",
                )
            ):
                group = _mapping(
                    raw_group,
                    f"quality view {view_id} group recall[{group_index}]",
                )
                local_group_id = _text(
                    group.get("group_id"),
                    f"quality view {view_id} group recall[{group_index}].group_id",
                )
                recall = _unit(
                    group.get("recall"),
                    f"quality view {view_id}/{local_group_id}.recall",
                )
                if recall >= group_recall_threshold:
                    continue
                canonical_group_id = local_map.get(local_group_id)
                if canonical_group_id is None:
                    reasons.append(f"LOCAL_GROUP_NOT_CANONICALIZED:{local_group_id}")
                    continue
                canonical_group = canonical_groups[canonical_group_id]
                base_colors = _string_array(
                    group.get("base_colors"),
                    f"quality view {view_id}/{local_group_id}.base_colors",
                )
                expected_colors = evidence_color_labels(
                    _text(
                        canonical_group.get("base_color"),
                        f"canonical group {canonical_group_id}.base_color",
                    )
                )
                if not base_colors or any(
                    not evidence_color_labels(color) & expected_colors
                    for color in base_colors
                ):
                    reasons.append(f"LOCAL_CANONICAL_COLOR_CONFLICT:{local_group_id}")
                    continue
                assert spatial_reference is not None
                support = {
                    "reference_view_id": view_id,
                    "local_group_id": local_group_id,
                    "reference_sha256": image_sha,
                    "content_cluster_id": spatial_reference["content_cluster_id"],
                    "pose_cluster_id": spatial_reference["pose_cluster_id"],
                    "recall": recall,
                    "deficit_sources": ["group_recall"],
                }
                pending_supports.append((canonical_group_id, support))
                missing_records.append(
                    {
                        "local_group_id": local_group_id,
                        "canonical_group_id": canonical_group_id,
                        "recall": recall,
                        "deficit_sources": ["group_recall"],
                    }
                )
            for failure in dominant_failures:
                deficit_source = "dominant_mass"
                local_group_ids = failure["local_group_ids"]
                if len(local_group_ids) != 1:
                    dominant_mass_skips.append(
                        {
                            "family_key": failure["family_key"],
                            "local_group_ids": local_group_ids,
                            "reason_code": (
                                "DOMINANT_FAMILY_LOCAL_GROUP_JOIN_NOT_UNIQUE"
                            ),
                        }
                    )
                    continue
                local_group_id = local_group_ids[0]
                canonical_group_id = local_map.get(local_group_id)
                if canonical_group_id is None:
                    dominant_mass_skips.append(
                        {
                            "family_key": failure["family_key"],
                            "local_group_ids": local_group_ids,
                            "reason_code": "LOCAL_GROUP_NOT_CANONICALIZED",
                        }
                    )
                    continue
                canonical_group = canonical_groups[canonical_group_id]
                expected_colors = evidence_color_labels(
                    _text(
                        canonical_group.get("base_color"),
                        f"canonical group {canonical_group_id}.base_color",
                    )
                )
                if not failure["base_colors"] or any(
                    not evidence_color_labels(color) & expected_colors
                    for color in failure["base_colors"]
                ):
                    dominant_mass_skips.append(
                        {
                            "family_key": failure["family_key"],
                            "local_group_ids": local_group_ids,
                            "reason_code": "LOCAL_CANONICAL_COLOR_CONFLICT",
                        }
                    )
                    continue
                assert spatial_reference is not None
                support = {
                    "reference_view_id": view_id,
                    "local_group_id": local_group_id,
                    "reference_sha256": image_sha,
                    "content_cluster_id": spatial_reference["content_cluster_id"],
                    "pose_cluster_id": spatial_reference["pose_cluster_id"],
                    "recall": failure["mass_recall"],
                    "deficit_sources": [deficit_source],
                    "dominant_mass_family_key": failure["family_key"],
                }
                pending_supports.append((canonical_group_id, support))
                missing_records.append(
                    {
                        "local_group_id": local_group_id,
                        "canonical_group_id": canonical_group_id,
                        "recall": failure["mass_recall"],
                        "deficit_sources": [deficit_source],
                        "dominant_mass_family_key": failure["family_key"],
                    }
                )
        if (
            not residual_reasons
            and local_map is not None
            and spatial_reference is not None
        ):
            for failure in dominant_local_projection_residuals:
                local_group_ids = failure["local_group_ids"]
                if len(local_group_ids) != 1:
                    dominant_mass_skips.append(
                        {
                            "family_key": failure["family_key"],
                            "local_group_ids": local_group_ids,
                            "reason_code": (
                                "DOMINANT_FAMILY_LOCAL_GROUP_JOIN_NOT_UNIQUE"
                            ),
                        }
                    )
                    continue
                local_group_id = local_group_ids[0]
                canonical_group_id = local_map.get(local_group_id)
                if canonical_group_id is None:
                    dominant_mass_skips.append(
                        {
                            "family_key": failure["family_key"],
                            "local_group_ids": local_group_ids,
                            "reason_code": "LOCAL_GROUP_NOT_CANONICALIZED",
                        }
                    )
                    continue
                canonical_group = canonical_groups[canonical_group_id]
                expected_colors = evidence_color_labels(
                    _text(
                        canonical_group.get("base_color"),
                        f"canonical group {canonical_group_id}.base_color",
                    )
                )
                if not failure["base_colors"] or any(
                    not evidence_color_labels(color) & expected_colors
                    for color in failure["base_colors"]
                ):
                    dominant_mass_skips.append(
                        {
                            "family_key": failure["family_key"],
                            "local_group_ids": local_group_ids,
                            "reason_code": "LOCAL_CANONICAL_COLOR_CONFLICT",
                        }
                    )
                    continue
                support = {
                    "reference_view_id": view_id,
                    "local_group_id": local_group_id,
                    "reference_sha256": image_sha,
                    "content_cluster_id": spatial_reference["content_cluster_id"],
                    "pose_cluster_id": spatial_reference["pose_cluster_id"],
                    "recall": failure["mass_recall"],
                    "deficit_sources": [DOMINANT_RESIDUAL_DEFICIT_SOURCE],
                    "dominant_mass_family_key": failure["family_key"],
                    "requires_strict_local_projection": True,
                    "reference_share": failure["reference_share"],
                    "reference_share_margin": failure["reference_share_margin"],
                    "observed_render_share": failure["observed_share"],
                    "deficit_share": failure["deficit_share"],
                    "mass_recall": failure["mass_recall"],
                    "alignment_score": failure["alignment_score"],
                    "silhouette_iou": failure["silhouette_iou"],
                    "silhouette_floor": failure["silhouette_floor"],
                    "silhouette_gap": failure["silhouette_gap"],
                    "edge_f1_tolerance_3px": failure["edge_f1_tolerance_3px"],
                    "profile_similarity": failure["profile_similarity"],
                    "bbox_aspect_similarity": failure["bbox_aspect_similarity"],
                    "render_foreground_pixels": failure["render_foreground_pixels"],
                }
                existing_residual = raw_residual_support[canonical_group_id].get(
                    view_id
                )
                if existing_residual is not None:
                    raise QualityRepairError(
                        "quality residuals disagree on one canonical view"
                    )
                raw_residual_support[canonical_group_id][view_id] = support
                missing_records.append(
                    {
                        "local_group_id": local_group_id,
                        "canonical_group_id": canonical_group_id,
                        "recall": failure["mass_recall"],
                        "deficit_sources": [DOMINANT_RESIDUAL_DEFICIT_SOURCE],
                        "dominant_mass_family_key": failure["family_key"],
                    }
                )
        if not reasons:
            for canonical_group_id, support in pending_supports:
                existing = raw_support[canonical_group_id].get(view_id)
                if existing is None:
                    raw_support[canonical_group_id][view_id] = support
                    continue
                if existing["local_group_id"] != support["local_group_id"]:
                    raise QualityRepairError(
                        "quality deficits disagree on a local canonical-group join"
                    )
                sources = sorted(
                    set(existing.get("deficit_sources", []))
                    | set(support.get("deficit_sources", []))
                )
                existing["deficit_sources"] = sources
                existing["recall"] = min(
                    float(existing["recall"]), float(support["recall"])
                )
                dominant_key = support.get("dominant_mass_family_key")
                if dominant_key is not None:
                    previous_key = existing.get("dominant_mass_family_key")
                    if previous_key not in {None, dominant_key}:
                        raise QualityRepairError(
                            "quality deficits disagree on dominant family identity"
                        )
                    existing["dominant_mass_family_key"] = dominant_key
        merged_missing: dict[tuple[str, str], dict[str, Any]] = {}
        for record in missing_records:
            key = (record["canonical_group_id"], record["local_group_id"])
            existing = merged_missing.get(key)
            if existing is None:
                merged_missing[key] = record
                continue
            existing["recall"] = min(float(existing["recall"]), float(record["recall"]))
            existing["deficit_sources"] = sorted(
                set(existing["deficit_sources"]) | set(record["deficit_sources"])
            )
            dominant_key = record.get("dominant_mass_family_key")
            if dominant_key is not None:
                existing["dominant_mass_family_key"] = dominant_key
        view_diagnostics.append(
            {
                "reference_view_id": view_id,
                "trusted_for_repair": not reasons,
                "trusted_for_dominant_residual": not residual_reasons,
                "reason_codes": sorted(set(reasons)),
                "dominant_residual_reason_codes": sorted(set(residual_reasons)),
                "missing_groups": sorted(
                    merged_missing.values(),
                    key=lambda item: (
                        item["canonical_group_id"],
                        item["local_group_id"],
                        item["deficit_sources"],
                    ),
                ),
                "dominant_mass_skips": dominant_mass_skips,
            }
        )

    aggregate_reason_values = aggregate.get("reasons")
    if dominant_threshold_contract is not None:
        if not isinstance(aggregate_reason_values, Sequence) or isinstance(
            aggregate_reason_values, (str, bytes)
        ):
            raise QualityRepairError("quality aggregate reasons must be an array")
        claims_dominant_failure = (
            "single_strong_view_confirms_dominant_family_mass_deficit"
            in aggregate_reason_values
        )
        if claims_dominant_failure != bool(dominant_failure_view_count):
            raise QualityRepairError(
                "quality aggregate dominant-mass decision is inconsistent"
            )

    repairable: dict[str, dict[str, Any]] = {}
    dominant_residual_repairable: dict[str, dict[str, Any]] = {}
    group_diagnostics: list[dict[str, Any]] = []
    for group_id, canonical_group in sorted(canonical_groups.items()):
        supports = sorted(
            raw_support.get(group_id, {}).values(),
            key=lambda item: item["reference_view_id"],
        )
        residual_supports = sorted(
            raw_residual_support.get(group_id, {}).values(),
            key=lambda item: item["reference_view_id"],
        )
        content_clusters = sorted(
            {str(item["content_cluster_id"]) for item in supports}
        )
        pose_clusters = sorted({str(item["pose_cluster_id"]) for item in supports})
        raw_hashes = sorted({str(item["reference_sha256"]) for item in supports})
        reasons: list[str] = []
        if (
            canonical_group.get("singleton") is True
            or int(canonical_group.get("distinct_view_count", 0)) < 2
        ):
            reasons.append("CANONICAL_GROUP_NOT_MULTIVIEW")
        if len(supports) < 2:
            reasons.append("MISSING_IN_FEWER_THAN_TWO_TRUSTED_VIEWS")
        if len(content_clusters) < 2:
            reasons.append("REFERENCE_CONTENT_NOT_DISTINCT")
        if len(raw_hashes) < 2:
            reasons.append("REFERENCE_BYTES_NOT_DISTINCT")
        if len(pose_clusters) < 2:
            reasons.append("CAD_POSE_NOT_DISTINCT")
        single_view_spatial_repairable = (
            canonical_group.get("singleton") is not True
            and int(canonical_group.get("distinct_view_count", 0)) >= 2
            and len(supports) == 1
        )
        residual_reasons: list[str] = []
        if (
            canonical_group.get("singleton") is True
            or int(canonical_group.get("distinct_view_count", 0)) < 2
        ):
            residual_reasons.append("CANONICAL_GROUP_NOT_MULTIVIEW")
        if len(residual_supports) != 1:
            residual_reasons.append(
                "DOMINANT_RESIDUAL_REQUIRES_EXACTLY_ONE_TRUSTED_VIEW"
            )
        record = {
            "canonical_group_id": group_id,
            "repairable": not reasons,
            "single_view_spatial_repairable": (single_view_spatial_repairable),
            "dominant_residual_repairable": not residual_reasons,
            "reason_codes": reasons,
            "dominant_residual_reason_codes": residual_reasons,
            "supporting_views": supports,
            "dominant_residual_supporting_views": residual_supports,
            "supporting_content_cluster_ids": content_clusters,
            "supporting_pose_cluster_ids": pose_clusters,
            "supporting_reference_sha256s": raw_hashes,
            "local_projection_residual_view_ids": (
                [str(item["reference_view_id"]) for item in residual_supports]
            ),
        }
        group_diagnostics.append(record)
        if not reasons or single_view_spatial_repairable:
            repairable[group_id] = record
        if not residual_reasons:
            dominant_residual_repairable[group_id] = {
                **copy.deepcopy(record),
                "repairable": False,
                "single_view_spatial_repairable": False,
                "supporting_views": copy.deepcopy(residual_supports),
                "supporting_content_cluster_ids": sorted(
                    {str(item["content_cluster_id"]) for item in residual_supports}
                ),
                "supporting_pose_cluster_ids": sorted(
                    {str(item["pose_cluster_id"]) for item in residual_supports}
                ),
                "supporting_reference_sha256s": sorted(
                    {str(item["reference_sha256"]) for item in residual_supports}
                ),
            }
    aggregate_reasons = (
        []
        if repairable or dominant_residual_repairable
        else ["NO_REPAIRABLE_CANONICAL_GROUP"]
    )
    return (
        repairable,
        dominant_residual_repairable,
        group_diagnostics,
        view_diagnostics,
        aggregate_reasons,
    )


def _trusted_dark_foreground_residual_groups(
    quality_report: Mapping[str, Any],
    *,
    canonical_groups: Mapping[str, Mapping[str, Any]],
    reference_evidence: Mapping[str, Mapping[str, str]],
) -> tuple[
    dict[str, dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Recover a black-family QA residual even when local Qwen missed black.

    This is deliberately independent of ``view_group_id_maps``: those maps
    cannot contain a black group that the local palette omitted.  The join is
    instead permitted only when the canonical palette contains exactly one
    multi-view black group and the QA distributions, image hashes, mapping,
    foreground counts, and strong registration all agree.
    """

    _require_schema(quality_report, QUALITY_SCHEMA_VERSION, "quality report")
    aggregate = _mapping(quality_report.get("aggregate"), "quality_report.aggregate")
    eligible_black_groups = [
        group_id
        for group_id, group in sorted(canonical_groups.items())
        if str(group.get("base_color", "")).strip().casefold() == "black"
        and group.get("singleton") is not True
        and isinstance(group.get("distinct_view_count"), int)
        and not isinstance(group.get("distinct_view_count"), bool)
        and int(group["distinct_view_count"]) >= 2
    ]
    unique_group_id = (
        eligible_black_groups[0] if len(eligible_black_groups) == 1 else None
    )
    raw_supports: list[dict[str, Any]] = []
    view_diagnostics: list[dict[str, Any]] = []
    seen_views: set[str] = set()

    for index, raw_view in enumerate(
        _sequence(quality_report.get("views"), "quality_report.views")
    ):
        view = _mapping(raw_view, f"quality_report.views[{index}]")
        view_id = _text(
            view.get("reference_view_id"),
            f"quality_report.views[{index}].reference_view_id",
        )
        if view_id in seen_views:
            raise QualityRepairError(f"duplicate quality reference view: {view_id}")
        seen_views.add(view_id)
        reasons: list[str] = []
        if aggregate.get("status") != "FAIL":
            reasons.append("QUALITY_STATUS_NOT_FAIL")
        if unique_group_id is None:
            reasons.append("CANONICAL_BLACK_GROUP_NOT_UNIQUE_MULTIVIEW")
        if view.get("status") not in {"FAIL", "REVIEW"}:
            reasons.append("VIEW_STATUS_NOT_FAIL_OR_REVIEW")

        render_view_id = view.get("render_view_id")
        if not isinstance(render_view_id, str) or not render_view_id:
            reasons.append("MISSING_RENDER_VIEW")
        mapping = view.get("mapping")
        if not isinstance(mapping, Mapping):
            reasons.append("MAPPING_AUDIT_MISSING")
        else:
            mapping_reasons = mapping.get("reasons")
            if (
                not isinstance(mapping_reasons, Sequence)
                or isinstance(mapping_reasons, (str, bytes))
                or list(mapping_reasons)
            ):
                reasons.append("MAPPING_NOT_TRUSTED")
            if mapping.get("selected_render_view_id") != render_view_id:
                reasons.append("MAPPING_RENDER_MISMATCH")

        reference = view.get("reference")
        image_sha: str | None = None
        reference_pixels: int | None = None
        if not isinstance(reference, Mapping):
            reasons.append("REFERENCE_AUDIT_MISSING")
        else:
            raw_sha = reference.get("image_sha256")
            if not isinstance(raw_sha, str) or not _SHA256_PATTERN.fullmatch(raw_sha):
                reasons.append("REFERENCE_SHA256_INVALID")
            else:
                image_sha = raw_sha
            trusted = reference.get("trusted_evidence")
            if not isinstance(trusted, Mapping) or trusted.get("usable") is not True:
                reasons.append("PALETTE_EVIDENCE_NOT_TRUSTED")
            foreground = reference.get("foreground")
            raw_pixels = (
                foreground.get("pixel_count")
                if isinstance(foreground, Mapping)
                else None
            )
            if (
                isinstance(raw_pixels, int)
                and not isinstance(raw_pixels, bool)
                and raw_pixels > 0
            ):
                reference_pixels = raw_pixels
            else:
                reasons.append("REFERENCE_FOREGROUND_PIXELS_INVALID")

        render = view.get("render")
        render_pixels: int | None = None
        if isinstance(render, Mapping):
            foreground = render.get("foreground")
            raw_pixels = (
                foreground.get("pixel_count")
                if isinstance(foreground, Mapping)
                else None
            )
            if (
                isinstance(raw_pixels, int)
                and not isinstance(raw_pixels, bool)
                and raw_pixels > 0
            ):
                render_pixels = raw_pixels
        if render_pixels is None:
            reasons.append("RENDER_FOREGROUND_PIXELS_INVALID")

        spatial_reference = reference_evidence.get(view_id)
        if spatial_reference is None:
            reasons.append("CONTENT_POSE_EVIDENCE_MISSING")
        elif image_sha is not None and spatial_reference["raw_sha256"] != image_sha:
            reasons.append("REFERENCE_SHA256_SPATIAL_MISMATCH")
        elif render_view_id != spatial_reference["pose_cluster_id"]:
            reasons.append("QUALITY_SPATIAL_POSE_MISMATCH")

        alignment = view.get("alignment")
        alignment_values: dict[str, float] = {}
        alignment_requirements = {
            "score": DARK_FOREGROUND_MIN_ALIGNMENT_SCORE,
            "silhouette_iou": DARK_FOREGROUND_MIN_SILHOUETTE_IOU,
            "edge_f1_tolerance_3px": DARK_FOREGROUND_MIN_EDGE_F1,
            "profile_similarity": DARK_FOREGROUND_MIN_PROFILE_SIMILARITY,
            "bbox_aspect_similarity": (DARK_FOREGROUND_MIN_BBOX_ASPECT_SIMILARITY),
        }
        if not isinstance(alignment, Mapping):
            reasons.append("ALIGNMENT_MISSING")
        else:
            for field, minimum in alignment_requirements.items():
                parsed = _optional_unit(alignment.get(field))
                if parsed is None or parsed < minimum:
                    reasons.append(f"DARK_ALIGNMENT_BELOW_FLOOR:{field}")
                else:
                    alignment_values[field] = parsed

        reference_share: float | None = None
        render_share: float | None = None
        deficit_share: float | None = None
        mass_recall: float | None = None
        sampled_reference_pixels: int | None = None
        color = view.get("material_color")
        if not isinstance(color, Mapping):
            reasons.append("MATERIAL_COLOR_AUDIT_MISSING")
        else:
            reference_distribution = color.get("reference_distribution")
            render_distribution = color.get("render_distribution")
            if not isinstance(reference_distribution, Mapping) or not isinstance(
                render_distribution, Mapping
            ):
                reasons.append("COLOR_DISTRIBUTIONS_MISSING")
            else:
                raw_sampled = reference_distribution.get("sampled_pixels")
                raw_render_sampled = render_distribution.get("sampled_pixels")
                if (
                    isinstance(raw_sampled, int)
                    and not isinstance(raw_sampled, bool)
                    and raw_sampled > 0
                    and raw_sampled == reference_pixels
                    and reference_distribution.get("sample_step") == 1
                ):
                    sampled_reference_pixels = raw_sampled
                else:
                    reasons.append("REFERENCE_COLOR_SAMPLE_NOT_EXACT")
                if (
                    not isinstance(raw_render_sampled, int)
                    or isinstance(raw_render_sampled, bool)
                    or raw_render_sampled <= 0
                    or raw_render_sampled != render_pixels
                    or render_distribution.get("sample_step") != 1
                ):
                    reasons.append("RENDER_COLOR_SAMPLE_NOT_EXACT")
                reference_categories = reference_distribution.get(
                    "category_distribution"
                )
                render_categories = render_distribution.get("category_distribution")
                if not isinstance(reference_categories, Mapping) or not isinstance(
                    render_categories, Mapping
                ):
                    reasons.append("COLOR_CATEGORY_DISTRIBUTIONS_MISSING")
                else:
                    try:
                        reference_share = sum(
                            _unit(
                                reference_categories[label],
                                f"quality dark reference share {view_id}/{label}",
                            )
                            for label in ("black", "achromatic_dark")
                        )
                        render_share = sum(
                            _unit(
                                render_categories[label],
                                f"quality dark render share {view_id}/{label}",
                            )
                            for label in ("black", "achromatic_dark")
                        )
                    except KeyError:
                        reasons.append("BLACK_FAMILY_CATEGORY_MISSING")
        if reference_share is not None and render_share is not None:
            if reference_share > 1.0 + 1e-9 or render_share > 1.0 + 1e-9:
                raise QualityRepairError("quality dark-family shares exceed one")
            deficit_share = max(0.0, reference_share - render_share)
            mass_recall = (
                min(1.0, render_share / reference_share)
                if reference_share > 0.0
                else 1.0
            )
            if deficit_share < DARK_FOREGROUND_MIN_DEFICIT_SHARE:
                reasons.append("DARK_FOREGROUND_DEFICIT_BELOW_FLOOR")
            if mass_recall >= DARK_FOREGROUND_MAX_MASS_RECALL:
                reasons.append("DARK_FOREGROUND_MASS_RECALL_NOT_LOW")

        support: dict[str, Any] | None = None
        if (
            not reasons
            and unique_group_id is not None
            and spatial_reference is not None
            and image_sha is not None
            and sampled_reference_pixels is not None
            and render_pixels is not None
            and reference_share is not None
            and render_share is not None
            and deficit_share is not None
            and mass_recall is not None
        ):
            budget_pixels = int(math.ceil(deficit_share * sampled_reference_pixels))
            support = {
                "reference_view_id": view_id,
                "local_group_id": f"__canonical_dark__:{unique_group_id}",
                "reference_sha256": image_sha,
                "content_cluster_id": spatial_reference["content_cluster_id"],
                "pose_cluster_id": spatial_reference["pose_cluster_id"],
                "recall": mass_recall,
                "mass_recall": mass_recall,
                "deficit_sources": [DARK_FOREGROUND_RESIDUAL_DEFICIT_SOURCE],
                "reference_share": reference_share,
                "observed_render_share": render_share,
                "deficit_share": deficit_share,
                "normalized_reference_pixels": sampled_reference_pixels,
                "render_foreground_pixels": render_pixels,
                "budget_pixels": budget_pixels,
                "budget_limit_pixels": int(
                    math.floor(
                        DARK_FOREGROUND_MAX_TOTAL_CONTRIBUTION_FACTOR * budget_pixels
                    )
                ),
                "alignment": dict(sorted(alignment_values.items())),
            }
            raw_supports.append(support)
        view_diagnostics.append(
            {
                "reference_view_id": view_id,
                "canonical_group_id": unique_group_id,
                "trusted_for_dark_foreground_residual": not reasons,
                "reason_codes": sorted(set(reasons)),
                "support": copy.deepcopy(support),
            }
        )

    repairable: dict[str, dict[str, Any]] = {}
    group_diagnostics: list[dict[str, Any]] = []
    for group_id, group in sorted(canonical_groups.items()):
        supports = (
            sorted(raw_supports, key=lambda item: item["reference_view_id"])
            if group_id == unique_group_id
            else []
        )
        reasons: list[str] = []
        if group_id != unique_group_id:
            reasons.append("NOT_UNIQUE_MULTIVIEW_BLACK_CANONICAL_GROUP")
        elif len(supports) != 1:
            reasons.append("DARK_FOREGROUND_RESIDUAL_REQUIRES_EXACTLY_ONE_TRUSTED_VIEW")
        record = {
            "canonical_group_id": group_id,
            "dark_residual_repairable": not reasons,
            "dark_residual_reason_codes": reasons,
            "dark_residual_supporting_views": supports,
        }
        group_diagnostics.append(record)
        if not reasons:
            repairable[group_id] = {
                **copy.deepcopy(record),
                "supporting_views": copy.deepcopy(supports),
                "supporting_content_cluster_ids": sorted(
                    {str(item["content_cluster_id"]) for item in supports}
                ),
                "supporting_pose_cluster_ids": sorted(
                    {str(item["pose_cluster_id"]) for item in supports}
                ),
                "supporting_reference_sha256s": sorted(
                    {str(item["reference_sha256"]) for item in supports}
                ),
            }
    return repairable, group_diagnostics, view_diagnostics


def _confirmed_group_materials(
    group_materials: Mapping[str, Any],
    *,
    canonical_groups: Mapping[str, Mapping[str, Any]],
    whitelist_ids: set[str],
    allow_unconfirmed_visual_tournament_seeds: bool,
) -> tuple[dict[str, str], dict[str, str], set[str]]:
    """Return materials that are safe for one bounded render-verified repair.

    A forward/reverse Qwen disagreement is never a completed material choice.
    Immutable mode may use one disputed choice as a zero-confidence rendered
    seed because the orchestrator subsequently requires the exact-MDL
    disagreement tournament before it can create the material lock.  Mutable
    mode has no such finalization gate, so it must reject the disputed choice
    instead of silently treating the forward answer as a repair.
    """

    _require_schema(group_materials, GROUP_MATERIALS_SCHEMA_VERSION, "group materials")
    if not isinstance(allow_unconfirmed_visual_tournament_seeds, bool):
        raise QualityRepairError(
            "allow_unconfirmed_visual_tournament_seeds must be boolean"
        )
    confirmed: dict[str, str] = {}
    unavailable: dict[str, str] = {}
    provisional: set[str] = set()
    seen: set[str] = set()
    for index, raw_selection in enumerate(
        _sequence(group_materials.get("selections"), "group_materials.selections")
    ):
        selection = _mapping(raw_selection, f"group_materials.selections[{index}]")
        group_id = _text(
            selection.get("group_id"),
            f"group_materials.selections[{index}].group_id",
        )
        if group_id in seen:
            raise QualityRepairError(f"duplicate group material selection: {group_id}")
        seen.add(group_id)
        if group_id not in canonical_groups:
            raise QualityRepairError(
                f"group material selection references unknown group: {group_id}"
            )
        material_id = _text(
            selection.get("material_id"),
            f"group material selection {group_id}.material_id",
        )
        raw_confidence = selection.get("confidence")
        if (
            isinstance(raw_confidence, bool)
            or not isinstance(raw_confidence, (int, float))
            or not math.isfinite(float(raw_confidence))
            or not 0.0 <= float(raw_confidence) <= 1.0
        ):
            raise QualityRepairError(
                f"group material selection {group_id}.confidence is invalid"
            )
        if material_id not in whitelist_ids:
            unavailable[group_id] = "CONFIRMED_MATERIAL_NOT_WHITELISTED"
        elif (
            selection.get("confirmed") is True
            and float(raw_confidence) >= MIN_CONFIRMED_MATERIAL_CONFIDENCE
        ):
            confirmed[group_id] = material_id
        elif (
            allow_unconfirmed_visual_tournament_seeds
            and float(raw_confidence) >= MIN_PROVISIONAL_MATERIAL_CONFIDENCE
        ):
            confirmed[group_id] = material_id
            provisional.add(group_id)
        else:
            unavailable[group_id] = (
                "MATERIAL_SELECTION_REQUIRES_EXACT_MDL_VISUAL_TOURNAMENT"
                if float(raw_confidence) >= MIN_PROVISIONAL_MATERIAL_CONFIDENCE
                else "MATERIAL_SELECTION_BELOW_DERIVED_REVIEW_CONFIDENCE"
                if selection.get("confirmed") is True
                else "MATERIAL_SELECTION_NOT_CONFIRMED"
            )
    for group_id in canonical_groups:
        if group_id not in seen:
            unavailable[group_id] = "GROUP_MATERIAL_SELECTION_MISSING"
    return confirmed, unavailable, provisional


def _policy_collapse_recovery_group_ids(
    baseline_policy_audit: Mapping[str, Any],
) -> set[str]:
    """Read groups that the baseline explicitly barred from authoring."""

    raw_recovery = baseline_policy_audit.get("material_collapse_recovery")
    if raw_recovery is None:
        return set()
    recovery = _mapping(
        raw_recovery,
        "baseline_policy_audit.material_collapse_recovery",
    )
    return {
        _text(
            raw_group_id,
            (
                "baseline_policy_audit.material_collapse_recovery."
                f"excluded_group_ids[{index}]"
            ),
        )
        for index, raw_group_id in enumerate(
            _sequence(
                recovery.get("excluded_group_ids"),
                (
                    "baseline_policy_audit.material_collapse_recovery."
                    "excluded_group_ids"
                ),
            )
        )
    }


def _geometry_risks(
    geometry_risk: Mapping[str, Any], *, registry_part_ids: Sequence[str]
) -> dict[str, bool]:
    _require_schema(geometry_risk, GEOMETRY_RISK_SCHEMA_VERSION, "geometry risk report")
    risks: dict[str, bool] = {}
    for index, raw_part in enumerate(
        _sequence(geometry_risk.get("parts"), "geometry_risk.parts")
    ):
        part = _mapping(raw_part, f"geometry_risk.parts[{index}]")
        part_id = _text(part.get("part_id"), f"geometry_risk.parts[{index}].part_id")
        if part_id in risks:
            raise QualityRepairError(f"duplicate geometry risk part: {part_id}")
        risk = _mapping(part.get("risk"), f"geometry risk {part_id}.risk")
        multi = risk.get("multi_material_risk")
        if not isinstance(multi, bool):
            raise QualityRepairError(
                f"geometry risk {part_id}.multi_material_risk must be boolean"
            )
        risks[part_id] = multi
    if set(risks) != set(registry_part_ids):
        raise QualityRepairError(
            "geometry risk report does not exactly cover the registry"
        )
    return risks


def _spatial_parts(
    spatial_report: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    parts: dict[str, Mapping[str, Any]] = {}
    for index, raw_part in enumerate(
        _sequence(spatial_report.get("parts"), "spatial_report.parts")
    ):
        part = _mapping(raw_part, f"spatial_report.parts[{index}]")
        part_id = _text(part.get("part_id"), f"spatial_report.parts[{index}].part_id")
        if part_id in parts:
            raise QualityRepairError(f"duplicate spatial report part: {part_id}")
        parts[part_id] = part
    return parts


def _spatial_gate_decisions(
    spatial_gate_audit: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    _require_schema(
        spatial_gate_audit, SPATIAL_GATE_SCHEMA_VERSION, "spatial gate audit"
    )
    decisions: dict[str, Mapping[str, Any]] = {}
    for index, raw_decision in enumerate(
        _sequence(spatial_gate_audit.get("decisions"), "spatial_gate_audit.decisions")
    ):
        decision = _mapping(raw_decision, f"spatial_gate_audit.decisions[{index}]")
        part_id = _text(
            decision.get("part_id"),
            f"spatial_gate_audit.decisions[{index}].part_id",
        )
        if part_id in decisions:
            raise QualityRepairError(f"duplicate spatial gate decision: {part_id}")
        decisions[part_id] = decision
    return decisions


def _mapping_decisions(
    mapping_consensus: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    _require_schema(
        mapping_consensus,
        MAPPING_CONSENSUS_SCHEMA_VERSION,
        "mapping consensus audit",
    )
    decisions: dict[str, Mapping[str, Any]] = {}
    for index, raw_decision in enumerate(
        _sequence(mapping_consensus.get("decisions"), "mapping_consensus.decisions")
    ):
        decision = _mapping(raw_decision, f"mapping_consensus.decisions[{index}]")
        part_id = _text(
            decision.get("part_id"),
            f"mapping_consensus.decisions[{index}].part_id",
        )
        if part_id in decisions:
            raise QualityRepairError(f"duplicate mapping decision: {part_id}")
        decisions[part_id] = decision
    return decisions


def _has_conflict(decision: Mapping[str, Any]) -> bool:
    for field in _CONFLICT_FIELDS:
        value = decision.get(field, [])
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            return True
        if value:
            return True
    return False


def _independent_support_reasons(
    *,
    view_ids: set[str],
    reference_evidence: Mapping[str, Mapping[str, str]],
) -> list[str]:
    content_clusters = {
        reference_evidence[view_id]["content_cluster_id"]
        for view_id in view_ids
        if view_id in reference_evidence
    }
    pose_clusters = {
        reference_evidence[view_id]["pose_cluster_id"]
        for view_id in view_ids
        if view_id in reference_evidence
    }
    raw_hashes = {
        reference_evidence[view_id]["raw_sha256"]
        for view_id in view_ids
        if view_id in reference_evidence
    }
    normalized_hashes = {
        reference_evidence[view_id]["normalized_pixel_sha256"]
        for view_id in view_ids
        if view_id in reference_evidence
    }
    reasons: list[str] = []
    if len(view_ids) < 2:
        reasons.append("PART_LOCALIZED_IN_FEWER_THAN_TWO_REFERENCE_VIEWS")
    if len(raw_hashes) < 2:
        reasons.append("PART_SUPPORT_REFERENCE_BYTES_NOT_DISTINCT")
    if len(normalized_hashes) < 2:
        reasons.append("PART_SUPPORT_NORMALIZED_PIXELS_NOT_DISTINCT")
    if len(content_clusters) < 2:
        reasons.append("PART_SUPPORT_CONTENT_NOT_DISTINCT")
    if len(pose_clusters) < 2:
        reasons.append("PART_SUPPORT_POSE_NOT_DISTINCT")
    return reasons


def _qa_confirmed_multiview_semantic_review_support_views(
    *,
    part_id: str,
    group_id: str,
    spatial_part: Mapping[str, Any],
    spatial_gate_decision: Mapping[str, Any] | None,
    mapping_decision: Mapping[str, Any] | None,
    reference_evidence: Mapping[str, Mapping[str, str]],
    spatial_policy: Mapping[str, Any],
    deficit: Mapping[str, Any] | None,
) -> tuple[list[str], list[str]]:
    """Return one conservative review-only semantic localization.

    Qwen ``review`` labels are never promoted during ordinary mapping.  After
    render QA independently proves the same canonical group missing, however,
    three hash/content/pose-distinct trusted views that all identify the same
    part/group pair can authorize one zero-confidence repair candidate.  The
    lane fails closed on any trusted semantic alternative, stable spatial
    alternative, or matched mapping-gate alternative.
    """

    reasons: list[str] = []
    if deficit is None or deficit.get("repairable") is not True:
        return [], ["SEMANTIC_REVIEW_REQUIRES_REPAIRABLE_QA_DEFICIT"]

    raw_confidence_floor = spatial_policy.get("minimum_semantic_conflict_confidence")
    if (
        isinstance(raw_confidence_floor, bool)
        or not isinstance(raw_confidence_floor, (int, float))
        or not math.isfinite(float(raw_confidence_floor))
        or not 0.0 <= float(raw_confidence_floor) <= 1.0
    ):
        return [], ["SEMANTIC_REVIEW_CONFIDENCE_FLOOR_INVALID"]
    confidence_floor = float(raw_confidence_floor)

    deficit_views = {
        str(item["reference_view_id"])
        for item in _sequence(
            deficit.get("supporting_views"),
            f"deficit {group_id}.supporting_views",
        )
        if isinstance(item, Mapping) and isinstance(item.get("reference_view_id"), str)
    }
    explicitly_invisible_views = _explicitly_invisible_semantic_view_ids(
        spatial_part=spatial_part,
        spatial_policy=spatial_policy,
        part_id=part_id,
    )
    support_views: set[str] = set()
    semantic_alternative_views: set[str] = set()
    seen_vote_views: set[str] = set()
    duplicate_vote_views: set[str] = set()
    for raw_vote in _sequence(
        spatial_part.get("semantic_votes", []),
        f"spatial part {part_id}.semantic_votes",
    ):
        if not isinstance(raw_vote, Mapping):
            continue
        vote = raw_vote
        view_id = vote.get("view_id")
        if not isinstance(view_id, str) or view_id not in reference_evidence:
            continue
        if view_id in seen_vote_views:
            duplicate_vote_views.add(view_id)
            continue
        seen_vote_views.add(view_id)
        reference = reference_evidence[view_id]
        authenticated_reference = (
            vote.get("reference_sha256") == reference["raw_sha256"]
            and vote.get("normalized_pixel_sha256")
            == reference["normalized_pixel_sha256"]
            and vote.get("content_cluster_id") == reference["content_cluster_id"]
            and vote.get("pose_cluster_id") == reference["pose_cluster_id"]
        )
        confidence = vote.get("effective_confidence")
        eligible = (
            authenticated_reference
            and view_id not in explicitly_invisible_views
            and vote.get("alignment_trusted") is True
            and vote.get("unique_canonical_join") is True
            and vote.get("pixel_gate_accepted") is True
            and vote.get("status") in {"matched", "review"}
            and isinstance(vote.get("canonical_group_id"), str)
            and isinstance(confidence, (int, float))
            and not isinstance(confidence, bool)
            and math.isfinite(float(confidence))
            and float(confidence) >= confidence_floor
        )
        if not eligible:
            continue
        if vote.get("canonical_group_id") == group_id:
            if view_id in deficit_views:
                support_views.add(view_id)
        else:
            semantic_alternative_views.add(view_id)

    if duplicate_vote_views:
        reasons.append("SEMANTIC_REVIEW_DUPLICATE_VIEW_VOTES")
    if semantic_alternative_views:
        reasons.append("SEMANTIC_REVIEW_TRUSTED_GROUP_CONFLICT")

    stable_spatial_alternatives: set[str] = set()
    resolved_diagnostic_alternatives: set[str] = set()
    for raw_observation in _sequence(
        spatial_part.get("observations", []),
        f"spatial part {part_id}.observations",
    ):
        observation = _mapping(
            raw_observation,
            f"spatial part {part_id}.observation",
        )
        view_id = observation.get("reference_view_id")
        canonical_group_id = observation.get("canonical_group_id")
        if (
            isinstance(view_id, str)
            and view_id in reference_evidence
            and observation.get("classification") == "resolved"
            and observation.get("registration_label_stable") is True
            and observation.get("perturbation_label_stable") is True
            and isinstance(canonical_group_id, str)
            and canonical_group_id != group_id
        ):
            stable_spatial_alternatives.add(view_id)
        for field in ("small_part_diagnostic", "canonical_palette_diagnostic"):
            diagnostic = observation.get(field)
            if (
                isinstance(view_id, str)
                and view_id in reference_evidence
                and isinstance(diagnostic, Mapping)
                and diagnostic.get("status") == "resolved"
                and diagnostic.get("reason_codes") == []
                and isinstance(diagnostic.get("canonical_group_id"), str)
                and diagnostic.get("canonical_group_id") != group_id
            ):
                resolved_diagnostic_alternatives.add(view_id)
    if stable_spatial_alternatives:
        reasons.append("SEMANTIC_REVIEW_STABLE_SPATIAL_GROUP_CONFLICT")
    if resolved_diagnostic_alternatives:
        reasons.append("SEMANTIC_REVIEW_RESOLVED_DIAGNOSTIC_GROUP_CONFLICT")

    if spatial_gate_decision is not None and (
        spatial_gate_decision.get("output_status") == "matched"
        and spatial_gate_decision.get("output_group_id") != group_id
    ):
        reasons.append("SEMANTIC_REVIEW_MATCHED_SPATIAL_GATE_CONFLICT")
    if mapping_decision is not None and (
        mapping_decision.get("output_status") == "matched"
        and mapping_decision.get("output_group_id") != group_id
    ):
        reasons.append("SEMANTIC_REVIEW_MATCHED_MAPPING_CONFLICT")

    if len(support_views) < QA_CONFIRMED_MULTIVIEW_SEMANTIC_REVIEW_MIN_SUPPORTS:
        reasons.append("SEMANTIC_REVIEW_REQUIRES_THREE_TRUSTED_QA_VIEWS")
    if support_views != deficit_views:
        reasons.append("SEMANTIC_REVIEW_DOES_NOT_COVER_QA_DEFICIT_VIEWS")
    reasons.extend(
        _independent_support_reasons(
            view_ids=support_views,
            reference_evidence=reference_evidence,
        )
    )
    return sorted(support_views), sorted(set(reasons))


def _stable_spatial_support_views(
    *,
    part_id: str,
    group_id: str,
    spatial_part: Mapping[str, Any],
    spatial_gate_decision: Mapping[str, Any] | None,
    mapping_decision: Mapping[str, Any] | None,
    reference_evidence: Mapping[str, Mapping[str, str]],
    deficit: Mapping[str, Any],
) -> tuple[list[str], list[str]]:
    """Return two-view, perturbation-stable part/group projections.

    This lane is stronger than the downgrade-only spatial gate: it is used
    only after final QA has failed and can only replace a neutral fallback.
    Review-level semantic disagreement is inconclusive; a matched mapping to a
    different group remains a hard contradiction.
    """

    reasons: list[str] = []
    supports: set[str] = set()
    conflicting_groups: set[str] = set()
    for raw_observation in _sequence(
        spatial_part.get("observations", []),
        f"spatial part {part_id}.observations",
    ):
        observation = _mapping(raw_observation, f"spatial part {part_id}.observation")
        canonical_group_id = observation.get("canonical_group_id")
        is_stable = (
            observation.get("classification") == "resolved"
            and observation.get("registration_label_stable") is True
            and observation.get("perturbation_label_stable") is True
            and isinstance(observation.get("reference_view_id"), str)
        )
        if not is_stable:
            continue
        if canonical_group_id == group_id:
            supports.add(str(observation["reference_view_id"]))
        elif isinstance(canonical_group_id, str) and canonical_group_id:
            conflicting_groups.add(canonical_group_id)
    if conflicting_groups:
        reasons.append("STABLE_SPATIAL_GROUP_CONFLICT")
    if spatial_gate_decision is not None and _has_conflict(spatial_gate_decision):
        reasons.append("SPATIAL_OR_SEMANTIC_CONFLICT")
    if mapping_decision is not None and (
        mapping_decision.get("output_status") == "matched"
        and mapping_decision.get("output_group_id") != group_id
    ):
        reasons.append("MATCHED_MAPPING_CONSENSUS_GROUP_CONFLICT")
    deficit_views = {
        str(item["reference_view_id"])
        for item in _sequence(
            deficit.get("supporting_views"),
            f"deficit {group_id}.supporting_views",
        )
        if isinstance(item, Mapping) and isinstance(item.get("reference_view_id"), str)
    }
    supports &= deficit_views
    reasons.extend(
        _independent_support_reasons(
            view_ids=supports,
            reference_evidence=reference_evidence,
        )
    )
    return sorted(supports), sorted(set(reasons))


def _slender_part_direct_box_projection(
    *,
    observation: Mapping[str, Any],
    group_id: str,
    policy: Mapping[str, Any],
) -> bool:
    """Accept one bounded projection whose rectangular sample is inapplicable.

    A thin or curved part can have a precise projected mask while most of its
    axis-aligned bounding rectangle covers unrelated background.  This lane
    treats that single bbox disagreement as inconclusive only when the direct
    mask, all four registration perturbations, and one independently audited
    palette evidence region agree exactly.  Every redundant count, ratio,
    local/canonical identifier, and digest is checked so malformed or
    contradictory evidence remains a safe rejection.
    """

    if (
        observation.get("evidence_mode") != "source_projection"
        or observation.get("classification") != "insufficient_visibility"
        or observation.get("reason_code") != "part_visible_pixels_below_floor"
        or observation.get("canonical_group_id") is not None
        or observation.get("bbox_canonical_group_id") is not None
        or observation.get("registration_label_stable") is not None
        or observation.get("perturbation_label_stable") is not True
    ):
        return False

    diagnostic_floor = _observation_diagnostic_floor(observation, policy)
    automatic_floor = policy.get("minimum_visible_pixels")
    minimum_samples = policy.get("minimum_diagnostic_resolved_samples")
    minimum_ratio = policy.get("minimum_diagnostic_consensus_ratio")
    declared_pixels = observation.get("declared_visible_pixels")
    projected_pixels = observation.get("projected_part_pixels")
    sampled_pixels = observation.get("sampled_reference_pixels")
    integer_values = (
        diagnostic_floor,
        automatic_floor,
        minimum_samples,
        declared_pixels,
        projected_pixels,
        sampled_pixels,
    )
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in integer_values
    ):
        return False
    if (
        isinstance(minimum_ratio, bool)
        or not isinstance(minimum_ratio, (int, float))
        or not math.isfinite(float(minimum_ratio))
        or not 0.0 <= float(minimum_ratio) <= 1.0
        or int(diagnostic_floor) < 1
        or int(automatic_floor) <= int(diagnostic_floor)
        or int(minimum_samples) < 1
        or int(declared_pixels) < int(diagnostic_floor)
        or int(projected_pixels) < int(diagnostic_floor)
        or int(sampled_pixels) != int(projected_pixels)
        or (
            int(declared_pixels) >= int(automatic_floor)
            and int(projected_pixels) >= int(automatic_floor)
        )
    ):
        return False

    # A valid resolved canonical diagnostic was already handled by the
    # stronger lane above this helper.  Do not use this exception to bypass a
    # malformed resolved canonical diagnostic or a resolved alternative.
    canonical_diagnostic = observation.get("canonical_palette_diagnostic")
    if canonical_diagnostic is not None:
        if not isinstance(canonical_diagnostic, Mapping):
            return False
        if canonical_diagnostic.get("status") == "resolved":
            return False

    diagnostic = observation.get("small_part_diagnostic")
    if not isinstance(diagnostic, Mapping):
        return False
    local_group_id = diagnostic.get("local_group_id")
    resolved_count = diagnostic.get("resolved_sample_count")
    target_count = diagnostic.get("target_sample_count")
    ratio = diagnostic.get("consensus_ratio")
    if (
        diagnostic.get("status") != "rejected"
        or diagnostic.get("reason_codes") != ["DIAGNOSTIC_BBOX_SAMPLE_DISAGREES"]
        or not isinstance(local_group_id, str)
        or not local_group_id
        or diagnostic.get("canonical_group_id") != group_id
        or diagnostic.get("bbox_canonical_group_id") is not None
        or diagnostic.get("registration_label_stable") is not None
        or isinstance(resolved_count, bool)
        or not isinstance(resolved_count, int)
        or isinstance(target_count, bool)
        or not isinstance(target_count, int)
        or isinstance(ratio, bool)
        or not isinstance(ratio, (int, float))
        or not math.isfinite(float(ratio))
        or resolved_count < int(minimum_samples)
        or target_count != resolved_count
        or float(ratio) != 1.0
        or float(ratio) < float(minimum_ratio)
        or round(target_count / resolved_count, 8) != float(ratio)
        or diagnostic.get("alternative_canonical_group_ids") != []
    ):
        return False

    raw_scores = observation.get("group_scores")
    if (
        not isinstance(raw_scores, Sequence)
        or isinstance(raw_scores, (str, bytes))
        or not raw_scores
    ):
        return False
    canonical_ids: set[str] = set()
    local_ids: set[str] = set()
    normalized_scores: list[tuple[str, str, float]] = []
    for raw_score in raw_scores:
        if not isinstance(raw_score, Mapping):
            return False
        canonical_id = raw_score.get("canonical_group_id")
        score_local_id = raw_score.get("local_group_id")
        matching_pixels = raw_score.get("matching_pixels")
        share = raw_score.get("color_share")
        if (
            not isinstance(canonical_id, str)
            or not canonical_id
            or canonical_id in canonical_ids
            or not isinstance(score_local_id, str)
            or not score_local_id
            or score_local_id in local_ids
            or isinstance(matching_pixels, bool)
            or not isinstance(matching_pixels, int)
            or not 0 <= matching_pixels <= int(projected_pixels)
            or isinstance(share, bool)
            or not isinstance(share, (int, float))
            or not math.isfinite(float(share))
            or not 0.0 <= float(share) <= 1.0
            or round(matching_pixels / int(projected_pixels), 8) != float(share)
            or raw_score.get("evidence_scope") != "view_local_palette"
        ):
            return False
        canonical_ids.add(canonical_id)
        local_ids.add(score_local_id)
        normalized_scores.append((canonical_id, score_local_id, float(share)))
    if any(
        normalized_scores[index][2] < normalized_scores[index + 1][2]
        for index in range(len(normalized_scores) - 1)
    ):
        return False
    winner_group_id, winner_local_id, winner_share = normalized_scores[0]
    runner_share = normalized_scores[1][2] if len(normalized_scores) > 1 else 0.0
    direct_margin = observation.get("color_margin")
    if (
        winner_group_id != group_id
        or winner_local_id != local_group_id
        or winner_share < SLENDER_DIRECT_BOX_MIN_COLOR_SHARE
        or isinstance(direct_margin, bool)
        or not isinstance(direct_margin, (int, float))
        or not math.isfinite(float(direct_margin))
        or not 0.0 <= float(direct_margin) <= 1.0
        or round(winner_share - runner_share, 8) != float(direct_margin)
        or float(direct_margin) < SLENDER_DIRECT_BOX_MIN_COLOR_MARGIN
    ):
        return False

    required_offsets = {(-2, 0), (2, 0), (0, -2), (0, 2)}
    perturbations = observation.get("projection_perturbations")
    if (
        not isinstance(perturbations, Sequence)
        or isinstance(perturbations, (str, bytes))
        or len(perturbations) != len(required_offsets)
    ):
        return False
    actual_offsets: set[tuple[int, int]] = set()
    for raw_perturbation in perturbations:
        if not isinstance(raw_perturbation, Mapping):
            return False
        offset = raw_perturbation.get("offset_pixels")
        perturbation_pixels = raw_perturbation.get("sampled_reference_pixels")
        share = raw_perturbation.get("best_color_share")
        margin = raw_perturbation.get("color_margin")
        if (
            not isinstance(offset, Sequence)
            or isinstance(offset, (str, bytes))
            or len(offset) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in offset
            )
            or raw_perturbation.get("canonical_group_id") != group_id
            or raw_perturbation.get("diagnostic_canonical_group_id") != group_id
            or isinstance(perturbation_pixels, bool)
            or not isinstance(perturbation_pixels, int)
            or perturbation_pixels != int(projected_pixels)
            or isinstance(share, bool)
            or not isinstance(share, (int, float))
            or not math.isfinite(float(share))
            or not 0.0 <= float(share) <= 1.0
            or float(share) < SLENDER_DIRECT_BOX_MIN_COLOR_MARGIN
            or isinstance(margin, bool)
            or not isinstance(margin, (int, float))
            or not math.isfinite(float(margin))
            or not 0.0 <= float(margin) <= 1.0
            or float(margin) < SLENDER_DIRECT_BOX_MIN_COLOR_MARGIN
        ):
            return False
        actual_offsets.add((int(offset[0]), int(offset[1])))
    if actual_offsets != required_offsets or resolved_count != 1 + len(
        required_offsets
    ):
        return False

    raw_overlaps = observation.get("accepted_evidence_box_overlaps")
    if not isinstance(raw_overlaps, Sequence) or isinstance(raw_overlaps, (str, bytes)):
        return False
    if any(not isinstance(raw_overlap, Mapping) for raw_overlap in raw_overlaps):
        return False
    target_overlaps = [
        raw_overlap
        for raw_overlap in raw_overlaps
        if raw_overlap.get("canonical_group_id") == group_id
    ]
    if len(target_overlaps) != 1:
        return False
    overlap = target_overlaps[0]
    evidence_pixels = overlap.get("evidence_pixel_count")
    overlap_pixels = overlap.get("projected_overlap_pixels")
    overlap_share = overlap.get("projected_overlap_share")
    evidence_digest = overlap.get("evidence_audit_sha256")
    if (
        overlap.get("local_group_id") != local_group_id
        or isinstance(evidence_pixels, bool)
        or not isinstance(evidence_pixels, int)
        or evidence_pixels < 1
        or isinstance(overlap_pixels, bool)
        or not isinstance(overlap_pixels, int)
        or overlap_pixels < int(diagnostic_floor)
        or overlap_pixels > evidence_pixels
        or isinstance(overlap_share, bool)
        or not isinstance(overlap_share, (int, float))
        or not math.isfinite(float(overlap_share))
        or not 0.0 <= float(overlap_share) <= 1.0
        or round(overlap_pixels / evidence_pixels, 8) != float(overlap_share)
        or float(overlap_share) < SLENDER_DIRECT_BOX_MIN_EVIDENCE_OVERLAP_SHARE
        or not isinstance(evidence_digest, str)
        or _SHA256_PATTERN.fullmatch(evidence_digest) is None
    ):
        return False
    return True


def _diagnostic_projection(
    *,
    observation: Mapping[str, Any],
    group_id: str,
    policy: Mapping[str, Any],
) -> tuple[bool, bool]:
    """Validate one bounded diagnostic and report whether it has alternatives."""

    canonical_diagnostic = observation.get("canonical_palette_diagnostic")
    if isinstance(canonical_diagnostic, Mapping):
        minimum_overlap = policy.get("minimum_canonical_supplement_foreground_overlap")
        diagnostic_floor = policy.get("minimum_diagnostic_visible_pixels")
        minimum_color_share = policy.get("minimum_diagnostic_color_share")
        minimum_color_margin = policy.get("minimum_diagnostic_color_margin")
        source_view_ids = canonical_diagnostic.get("canonical_source_view_ids")
        resolved_count = canonical_diagnostic.get("resolved_sample_count")
        target_count = canonical_diagnostic.get("target_sample_count")
        ratio = canonical_diagnostic.get("consensus_ratio")
        samples = [
            canonical_diagnostic.get("direct_sample"),
            canonical_diagnostic.get("bbox_sample"),
            *_sequence(
                canonical_diagnostic.get("projection_perturbations", []),
                "canonical palette diagnostic projection_perturbations",
            ),
        ]
        sample_contract_valid = len(samples) == 6 and all(
            isinstance(sample, Mapping) for sample in samples
        )
        required_offsets = {(-2, 0), (2, 0), (0, -2), (0, 2)}
        actual_offsets: set[tuple[int, int]] = set()
        if sample_contract_valid:
            for sample in samples[2:]:
                offset = sample.get("offset_pixels")
                if (
                    not isinstance(offset, Sequence)
                    or isinstance(offset, (str, bytes))
                    or len(offset) != 2
                    or any(
                        isinstance(value, bool) or not isinstance(value, int)
                        for value in offset
                    )
                ):
                    sample_contract_valid = False
                    break
                actual_offsets.add((int(offset[0]), int(offset[1])))
        sample_contract_valid = (
            sample_contract_valid and actual_offsets == required_offsets
        )
        numeric_contract_valid = (
            isinstance(diagnostic_floor, int)
            and not isinstance(diagnostic_floor, bool)
            and diagnostic_floor >= 1
            and isinstance(minimum_overlap, (int, float))
            and not isinstance(minimum_overlap, bool)
            and math.isfinite(float(minimum_overlap))
            and 0.0 <= float(minimum_overlap) <= 1.0
            and isinstance(minimum_color_share, (int, float))
            and not isinstance(minimum_color_share, bool)
            and math.isfinite(float(minimum_color_share))
            and 0.0 <= float(minimum_color_share) <= 1.0
            and isinstance(minimum_color_margin, (int, float))
            and not isinstance(minimum_color_margin, bool)
            and math.isfinite(float(minimum_color_margin))
            and 0.0 <= float(minimum_color_margin) <= 1.0
            and isinstance(resolved_count, int)
            and not isinstance(resolved_count, bool)
            and isinstance(target_count, int)
            and not isinstance(target_count, bool)
            and isinstance(ratio, (int, float))
            and not isinstance(ratio, bool)
            and math.isfinite(float(ratio))
        )
        if sample_contract_valid and numeric_contract_valid:
            for sample in samples:
                projected = sample.get("sampled_projection_pixels")
                foreground = sample.get("sampled_foreground_pixels")
                overlap = sample.get("foreground_overlap_ratio")
                share = sample.get("best_color_share")
                margin = sample.get("color_margin")
                if (
                    isinstance(projected, bool)
                    or not isinstance(projected, int)
                    or isinstance(foreground, bool)
                    or not isinstance(foreground, int)
                    or projected < foreground
                    or foreground < int(diagnostic_floor)
                    or isinstance(overlap, bool)
                    or not isinstance(overlap, (int, float))
                    or not math.isfinite(float(overlap))
                    or round(
                        foreground / projected if projected else 0.0,
                        8,
                    )
                    != float(overlap)
                    or float(overlap) < float(minimum_overlap)
                    or isinstance(share, bool)
                    or not isinstance(share, (int, float))
                    or not math.isfinite(float(share))
                    or not 0.0 <= float(share) <= 1.0
                    or float(share) < float(minimum_color_share)
                    or isinstance(margin, bool)
                    or not isinstance(margin, (int, float))
                    or not math.isfinite(float(margin))
                    or not 0.0 <= float(margin) <= 1.0
                    or float(margin) < float(minimum_color_margin)
                    or sample.get("canonical_group_id") != group_id
                ):
                    sample_contract_valid = False
                    break
        alternatives = canonical_diagnostic.get("alternative_canonical_group_ids")
        source_view_ids_valid = (
            isinstance(source_view_ids, Sequence)
            and not isinstance(source_view_ids, (str, bytes))
            and all(isinstance(value, str) and value for value in source_view_ids)
            and list(source_view_ids) == sorted(set(source_view_ids))
            and len(source_view_ids) >= 2
        )
        alternatives_valid = (
            isinstance(alternatives, Sequence)
            and not isinstance(alternatives, (str, bytes))
            and all(isinstance(value, str) and value for value in alternatives)
            and list(alternatives) == sorted(set(alternatives))
        )
        canonical_valid = (
            canonical_diagnostic.get("status") == "resolved"
            and canonical_diagnostic.get("reason_codes") == []
            and canonical_diagnostic.get("evidence_scope")
            == "canonical_multiview_propagation_repair_only"
            and canonical_diagnostic.get("canonical_group_id") == group_id
            and canonical_diagnostic.get("bbox_canonical_group_id") == group_id
            and canonical_diagnostic.get("registration_label_stable") is True
            and canonical_diagnostic.get("perturbation_label_stable") is True
            and canonical_diagnostic.get("minimum_foreground_overlap")
            == minimum_overlap
            and numeric_contract_valid
            and sample_contract_valid
            and resolved_count == 6
            and target_count == 6
            and float(ratio) == 1.0
            and source_view_ids_valid
            and alternatives_valid
        )
        if canonical_valid:
            return True, bool(alternatives)

    if _slender_part_direct_box_projection(
        observation=observation,
        group_id=group_id,
        policy=policy,
    ):
        return True, False

    diagnostic = observation.get("small_part_diagnostic")
    if not isinstance(diagnostic, Mapping):
        return False, False
    if (
        diagnostic.get("status") != "resolved"
        or diagnostic.get("reason_codes") != []
        or diagnostic.get("canonical_group_id") != group_id
        or diagnostic.get("bbox_canonical_group_id") != group_id
        or diagnostic.get("registration_label_stable") is not True
    ):
        return False, False
    resolved_count = diagnostic.get("resolved_sample_count")
    target_count = diagnostic.get("target_sample_count")
    ratio = diagnostic.get("consensus_ratio")
    minimum_samples = policy.get("minimum_diagnostic_resolved_samples")
    minimum_ratio = policy.get("minimum_diagnostic_consensus_ratio")
    diagnostic_floor = _observation_diagnostic_floor(observation, policy)
    automatic_floor = policy.get("minimum_visible_pixels")
    declared_pixels = observation.get("declared_visible_pixels")
    projected_pixels = observation.get("projected_part_pixels")
    numeric_values = (
        minimum_samples,
        diagnostic_floor,
        automatic_floor,
        resolved_count,
        target_count,
        declared_pixels,
        projected_pixels,
    )
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in numeric_values
    ):
        return False, False
    if (
        isinstance(minimum_ratio, bool)
        or not isinstance(minimum_ratio, (int, float))
        or isinstance(ratio, bool)
        or not isinstance(ratio, (int, float))
        or not math.isfinite(float(minimum_ratio))
        or not math.isfinite(float(ratio))
    ):
        return False, False
    if (
        int(minimum_samples) < 1
        or int(diagnostic_floor) < 1
        or int(automatic_floor) < int(diagnostic_floor)
        or int(resolved_count) < 1
        or int(target_count) < 0
        or int(declared_pixels) < 0
        or int(projected_pixels) < 0
        or not 0.0 <= float(minimum_ratio) <= 1.0
        or not 0.0 <= float(ratio) <= 1.0
        or int(resolved_count) < int(minimum_samples)
        or int(target_count) > int(resolved_count)
        or int(declared_pixels) < int(diagnostic_floor)
        or int(projected_pixels) < int(diagnostic_floor)
        or float(ratio) < float(minimum_ratio)
        or round(int(target_count) / int(resolved_count), 8) != float(ratio)
    ):
        return False, False
    bounded_case = (
        int(declared_pixels) < int(automatic_floor)
        or int(projected_pixels) < int(automatic_floor)
        or observation.get("perturbation_label_stable") is False
    )
    if not bounded_case:
        return False, False
    alternatives = diagnostic.get("alternative_canonical_group_ids")
    if (
        not isinstance(alternatives, Sequence)
        or isinstance(alternatives, (str, bytes))
        or any(not isinstance(value, str) or not value for value in alternatives)
        or list(alternatives) != sorted(set(alternatives))
    ):
        return False, False
    return True, bool(alternatives)


def _diagnostic_rejects_semantic_group(
    *,
    samples: Sequence[Any],
    rejected_group_id: str,
) -> bool:
    """Return whether six foreground samples robustly reject one group.

    Direct and bounding-box samples are mandatory clean observations.  Five
    of all six samples must remain below the strict rejection ceiling, while
    one registration-neighbourhood outlier is tolerated only up to a bounded
    ceiling.
    """

    if len(samples) != 6 or any(not isinstance(sample, Mapping) for sample in samples):
        return False
    rejected_shares: list[float] = []
    for sample in samples:
        scores = sample.get("group_scores")
        if not isinstance(scores, Sequence) or isinstance(scores, (str, bytes)):
            return False
        shares = [
            row.get("color_share")
            for row in scores
            if isinstance(row, Mapping)
            and row.get("canonical_group_id") == rejected_group_id
        ]
        if not shares or any(
            isinstance(share, bool)
            or not isinstance(share, (int, float))
            or not math.isfinite(float(share))
            for share in shares
        ):
            return False
        total = sum(float(share) for share in shares)
        if not 0.0 <= total <= 1.0:
            return False
        rejected_shares.append(total)

    return (
        rejected_shares[0] <= SEMANTIC_OVERRIDE_MAX_REJECTED_GROUP_SHARE
        and rejected_shares[1] <= SEMANTIC_OVERRIDE_MAX_REJECTED_GROUP_SHARE
        and sum(
            share <= SEMANTIC_OVERRIDE_MAX_REJECTED_GROUP_SHARE
            for share in rejected_shares
        )
        >= SEMANTIC_OVERRIDE_MIN_CLEAN_REJECTED_GROUP_SAMPLES
        and max(rejected_shares) <= SEMANTIC_OVERRIDE_MAX_REJECTED_GROUP_OUTLIER_SHARE
    )


def _bounded_spatial_support_views(
    *,
    part_id: str,
    group_id: str,
    spatial_part: Mapping[str, Any],
    spatial_gate_decision: Mapping[str, Any] | None,
    mapping_decision: Mapping[str, Any] | None,
    reference_evidence: Mapping[str, Mapping[str, str]],
    spatial_policy: Mapping[str, Any],
    minimum_semantic_confidence: float,
    deficit: Mapping[str, Any] | None,
) -> tuple[list[str], list[str], list[str]]:
    """Combine strong projections with independently corroborated diagnostics.

    One strong view plus one diagnostic requires the diagnostic to have no
    competing resolved label.  Two diagnostic-only views may tolerate the
    bounded majority encoded in each diagnostic, but only when final QA
    independently reports that same canonical group missing in both views.
    """

    reasons: list[str] = []
    stable_views: set[str] = set()
    diagnostic_views: set[str] = set()
    diagnostic_with_alternatives: set[str] = set()
    conflicting_groups: set[str] = set()
    for raw_observation in _sequence(
        spatial_part.get("observations", []),
        f"spatial part {part_id}.observations",
    ):
        observation = _mapping(raw_observation, f"spatial part {part_id}.observation")
        view_id = observation.get("reference_view_id")
        if not isinstance(view_id, str):
            continue
        canonical_group_id = observation.get("canonical_group_id")
        is_stable = (
            observation.get("classification") == "resolved"
            and observation.get("registration_label_stable") is True
            and observation.get("perturbation_label_stable") is True
        )
        if is_stable:
            if canonical_group_id == group_id:
                stable_views.add(view_id)
            elif isinstance(canonical_group_id, str) and canonical_group_id:
                conflicting_groups.add(canonical_group_id)
        diagnostic_ok, has_alternatives = _diagnostic_projection(
            observation=observation,
            group_id=group_id,
            policy=spatial_policy,
        )
        if diagnostic_ok:
            diagnostic_views.add(view_id)
            if has_alternatives:
                diagnostic_with_alternatives.add(view_id)
        raw_diagnostic = observation.get("small_part_diagnostic")
        if (
            isinstance(raw_diagnostic, Mapping)
            and raw_diagnostic.get("status") == "resolved"
            and isinstance(raw_diagnostic.get("canonical_group_id"), str)
            and raw_diagnostic.get("canonical_group_id") != group_id
        ):
            conflicting_groups.add(str(raw_diagnostic["canonical_group_id"]))
        canonical_diagnostic = observation.get("canonical_palette_diagnostic")
        if (
            isinstance(canonical_diagnostic, Mapping)
            and canonical_diagnostic.get("status") == "resolved"
            and canonical_diagnostic.get("reason_codes") == []
            and isinstance(canonical_diagnostic.get("canonical_group_id"), str)
            and canonical_diagnostic.get("canonical_group_id") != group_id
        ):
            conflicting_groups.add(str(canonical_diagnostic["canonical_group_id"]))

    explicitly_invisible_views = _explicitly_invisible_semantic_view_ids(
        spatial_part=spatial_part,
        spatial_policy=spatial_policy,
        part_id=part_id,
    )
    semantic_conflict_groups: dict[str, set[str]] = defaultdict(set)
    for raw_vote in _sequence(
        spatial_part.get("semantic_votes", []),
        f"spatial part {part_id}.semantic_votes",
    ):
        if not isinstance(raw_vote, Mapping):
            continue
        vote = raw_vote
        if not (
            vote.get("alignment_trusted") is True
            and vote.get("unique_canonical_join") is True
            and vote.get("status") == "matched"
            and isinstance(vote.get("canonical_group_id"), str)
            and vote.get("canonical_group_id") != group_id
            and isinstance(vote.get("effective_confidence"), (int, float))
            and not isinstance(vote.get("effective_confidence"), bool)
            and float(vote["effective_confidence"]) >= minimum_semantic_confidence
            and isinstance(vote.get("view_id"), str)
            and vote.get("view_id") not in explicitly_invisible_views
        ):
            continue
        semantic_conflict_groups[str(vote["canonical_group_id"])].add(
            str(vote["view_id"])
        )
    semantic_conflicts = {
        view_id
        for view_ids in semantic_conflict_groups.values()
        for view_id in view_ids
    }
    raw_semantic_conflict_floor = spatial_policy.get(
        "minimum_semantic_conflict_confidence"
    )
    semantic_alternative_views = {
        str(vote["view_id"])
        for raw_vote in _sequence(
            spatial_part.get("semantic_votes", []),
            f"spatial part {part_id}.semantic_votes",
        )
        if isinstance(raw_vote, Mapping)
        for vote in [raw_vote]
        if (
            vote.get("alignment_trusted") is True
            and vote.get("unique_canonical_join") is True
            and vote.get("status") in {"matched", "review"}
            and isinstance(vote.get("canonical_group_id"), str)
            and vote.get("canonical_group_id") != group_id
            and isinstance(vote.get("effective_confidence"), (int, float))
            and not isinstance(vote.get("effective_confidence"), bool)
            and isinstance(raw_semantic_conflict_floor, (int, float))
            and not isinstance(raw_semantic_conflict_floor, bool)
            and float(vote["effective_confidence"])
            >= float(raw_semantic_conflict_floor)
            and isinstance(vote.get("view_id"), str)
            and vote.get("view_id") not in explicitly_invisible_views
        )
    }
    if conflicting_groups:
        reasons.append("SPATIAL_DIAGNOSTIC_GROUP_CONFLICT")
    if mapping_decision is not None and (
        mapping_decision.get("output_status") == "matched"
        and mapping_decision.get("output_group_id") != group_id
    ):
        reasons.append("MATCHED_MAPPING_CONSENSUS_GROUP_CONFLICT")

    supports = stable_views | diagnostic_views
    if not diagnostic_views:
        reasons.append("NO_BOUNDED_SPATIAL_DIAGNOSTIC")
    if deficit is None:
        reasons.append("SPATIAL_REPAIR_REQUIRES_QA_GROUP_DEFICIT")
        supports.clear()
        stable_views.clear()
        diagnostic_views.clear()
    else:
        deficit_views = {
            str(item["reference_view_id"])
            for item in _sequence(
                deficit.get("supporting_views"),
                f"deficit {group_id}.supporting_views",
            )
            if isinstance(item, Mapping)
            and isinstance(item.get("reference_view_id"), str)
        }
        supports &= deficit_views
        stable_views &= deficit_views
        diagnostic_views &= deficit_views
    independent_pixel_support_reasons = _independent_support_reasons(
        view_ids=supports,
        reference_evidence=reference_evidence,
    )
    independently_confirmed_alternative = any(
        len(view_ids) >= 2
        and not _independent_support_reasons(
            view_ids=set(view_ids),
            reference_evidence=reference_evidence,
        )
        for view_ids in semantic_conflict_groups.values()
    )
    semantic_override_applied = (
        bool(semantic_conflicts)
        and len(supports) >= 2
        and not independent_pixel_support_reasons
        and semantic_conflicts <= supports
        and not independently_confirmed_alternative
        and not conflicting_groups
        and not diagnostic_with_alternatives
    )
    if semantic_conflicts and not semantic_override_applied:
        reasons.append("HIGH_CONFIDENCE_SEMANTIC_GROUP_CONFLICT")
    if spatial_gate_decision is not None and _has_conflict(spatial_gate_decision):
        gate_semantic_conflicts = spatial_gate_decision.get(
            "semantic_conflicting_view_ids", []
        )
        other_gate_conflicts = any(
            spatial_gate_decision.get(field, [])
            for field in _CONFLICT_FIELDS
            if field != "semantic_conflicting_view_ids"
        )
        gate_override_applied = (
            isinstance(gate_semantic_conflicts, Sequence)
            and not isinstance(gate_semantic_conflicts, (str, bytes))
            and set(gate_semantic_conflicts) - explicitly_invisible_views
            <= semantic_alternative_views
            and (not semantic_conflicts or semantic_override_applied)
            and not other_gate_conflicts
            and spatial_gate_decision.get("output_group_id") in {None, group_id}
            and spatial_gate_decision.get("output_status") != "matched"
        )
        if not gate_override_applied:
            reasons.append("SPATIAL_OR_SEMANTIC_CONFLICT")
    if stable_views:
        if diagnostic_with_alternatives:
            reasons.append("CORROBORATING_DIAGNOSTIC_HAS_COMPETING_GROUP")
    else:
        if len(diagnostic_views) < 2:
            reasons.append("DIAGNOSTIC_ONLY_REQUIRES_TWO_DEFICIT_VIEW_SUPPORTS")
    reasons.extend(
        _independent_support_reasons(
            view_ids=supports,
            reference_evidence=reference_evidence,
        )
    )
    return (
        sorted(supports),
        sorted(set(reasons)),
        sorted(semantic_conflicts) if semantic_override_applied else [],
    )


def _single_view_spatial_support_views(
    *,
    part_id: str,
    group_id: str,
    spatial_part: Mapping[str, Any],
    spatial_gate_decision: Mapping[str, Any] | None,
    mapping_decision: Mapping[str, Any] | None,
    spatial_policy: Mapping[str, Any],
    minimum_semantic_confidence: float,
    deficit: Mapping[str, Any],
    alignment_audits: Mapping[str, Mapping[str, Any]],
    reference_evidence: Mapping[str, Mapping[str, str]],
    allow_deferred_semantic_anchor: bool = False,
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Return one exact, strongly aligned spatial localization.

    This lane exists only for a canonical appearance seen by the palette in
    multiple source images but missing from exactly one strongly aligned QA
    view.  It requires a perturbation-stable direct projection, or a 100%
    bounded diagnostic without alternatives.  Any resolved spatial
    contradiction or a trusted matched semantic contradiction normally
    remains a hard block.  One same-view semantic contradiction may be
    superseded only by the stricter dominant-deficit lane below: the projected
    pixels must nearly exclude the rejected group and a second independent
    reference must semantically anchor the target.  It never authorizes a face
    subset or a non-neutral baseline delta.
    """

    reasons: list[str] = []
    if deficit.get("single_view_spatial_repairable") is not True:
        reasons.append("GROUP_NOT_SINGLE_VIEW_SPATIAL_REPAIRABLE")
    deficit_views = {
        str(item["reference_view_id"])
        for item in _sequence(
            deficit.get("supporting_views"),
            f"deficit {group_id}.supporting_views",
        )
        if isinstance(item, Mapping) and isinstance(item.get("reference_view_id"), str)
    }
    if len(deficit_views) != 1:
        reasons.append("SINGLE_VIEW_REPAIR_REQUIRES_ONE_QA_DEFICIT")

    supports: set[str] = set()
    spatial_anchor_candidates: set[str] = set()
    canonical_spatial_anchor_candidates: set[str] = set()
    observations_by_view: dict[str, Mapping[str, Any]] = {}
    conflicting_groups: set[str] = set()
    direct_support_observations: dict[str, Mapping[str, Any]] = {}
    automatic_floor = spatial_policy.get("minimum_visible_pixels")
    minimum_margin = spatial_policy.get("minimum_color_margin")
    if (
        isinstance(automatic_floor, bool)
        or not isinstance(automatic_floor, int)
        or isinstance(minimum_margin, bool)
        or not isinstance(minimum_margin, (int, float))
        or not math.isfinite(float(minimum_margin))
    ):
        reasons.append("SINGLE_VIEW_SPATIAL_POLICY_INVALID")
    else:
        for raw_observation in _sequence(
            spatial_part.get("observations", []),
            f"spatial part {part_id}.observations",
        ):
            observation = _mapping(
                raw_observation, f"spatial part {part_id}.observation"
            )
            view_id = observation.get("reference_view_id")
            if not isinstance(view_id, str):
                continue
            observations_by_view[view_id] = observation
            canonical_group_id = observation.get("canonical_group_id")
            declared_pixels = observation.get("declared_visible_pixels")
            projected_pixels = observation.get("projected_part_pixels")
            color_margin = observation.get("color_margin")
            stable = (
                observation.get("classification") == "resolved"
                and observation.get("registration_label_stable") is True
                and observation.get("perturbation_label_stable") is True
                and isinstance(declared_pixels, int)
                and not isinstance(declared_pixels, bool)
                and isinstance(projected_pixels, int)
                and not isinstance(projected_pixels, bool)
                and declared_pixels >= automatic_floor
                and projected_pixels >= automatic_floor
                and isinstance(color_margin, (int, float))
                and not isinstance(color_margin, bool)
                and math.isfinite(float(color_margin))
                and float(color_margin) >= float(minimum_margin)
            )
            diagnostic_ok, has_alternatives = _diagnostic_projection(
                observation=observation,
                group_id=group_id,
                policy=spatial_policy,
            )
            raw_small_diagnostic = observation.get("small_part_diagnostic")
            raw_canonical_diagnostic = observation.get("canonical_palette_diagnostic")
            exact_diagnostic = (
                diagnostic_ok
                and not has_alternatives
                and observation.get("perturbation_label_stable") is True
                and isinstance(raw_small_diagnostic, Mapping)
                and raw_small_diagnostic.get("consensus_ratio") == 1.0
            )
            exact_anchor_diagnostic = exact_diagnostic or (
                diagnostic_ok
                and not has_alternatives
                and isinstance(raw_canonical_diagnostic, Mapping)
                and raw_canonical_diagnostic.get("status") == "resolved"
                and raw_canonical_diagnostic.get("consensus_ratio") == 1.0
            )
            if view_id in deficit_views and (
                (stable and canonical_group_id == group_id) or exact_diagnostic
            ):
                supports.add(view_id)
                if stable and canonical_group_id == group_id:
                    direct_support_observations[view_id] = observation
            if (
                view_id not in deficit_views
                and not has_alternatives
                and (
                    (stable and canonical_group_id == group_id)
                    or exact_anchor_diagnostic
                )
            ):
                spatial_anchor_candidates.add(view_id)
                if (
                    isinstance(raw_canonical_diagnostic, Mapping)
                    and raw_canonical_diagnostic.get("status") == "resolved"
                    and exact_anchor_diagnostic
                ):
                    canonical_spatial_anchor_candidates.add(view_id)
            if stable and isinstance(canonical_group_id, str):
                if canonical_group_id != group_id:
                    conflicting_groups.add(canonical_group_id)
            raw_diagnostic = observation.get("small_part_diagnostic")
            if (
                isinstance(raw_diagnostic, Mapping)
                and raw_diagnostic.get("status") == "resolved"
                and isinstance(raw_diagnostic.get("canonical_group_id"), str)
                and raw_diagnostic.get("canonical_group_id") != group_id
            ):
                conflicting_groups.add(str(raw_diagnostic["canonical_group_id"]))
            canonical_diagnostic = observation.get("canonical_palette_diagnostic")
            if (
                isinstance(canonical_diagnostic, Mapping)
                and canonical_diagnostic.get("status") == "resolved"
                and canonical_diagnostic.get("reason_codes") == []
                and isinstance(canonical_diagnostic.get("canonical_group_id"), str)
                and canonical_diagnostic.get("canonical_group_id") != group_id
            ):
                conflicting_groups.add(str(canonical_diagnostic["canonical_group_id"]))

    explicitly_invisible_views = _explicitly_invisible_semantic_view_ids(
        spatial_part=spatial_part,
        spatial_policy=spatial_policy,
        part_id=part_id,
    )
    semantic_conflict_rows = [
        vote
        for raw_vote in _sequence(
            spatial_part.get("semantic_votes", []),
            f"spatial part {part_id}.semantic_votes",
        )
        if isinstance(raw_vote, Mapping)
        for vote in [raw_vote]
        if (
            vote.get("alignment_trusted") is True
            and vote.get("unique_canonical_join") is True
            and vote.get("status") == "matched"
            and isinstance(vote.get("canonical_group_id"), str)
            and vote.get("canonical_group_id") != group_id
            and isinstance(vote.get("effective_confidence"), (int, float))
            and not isinstance(vote.get("effective_confidence"), bool)
            and float(vote["effective_confidence"]) >= minimum_semantic_confidence
            and isinstance(vote.get("view_id"), str)
            and vote.get("view_id") not in explicitly_invisible_views
        )
    ]
    semantic_conflicts = {str(vote["view_id"]) for vote in semantic_conflict_rows}
    semantic_support_views = {
        str(vote["view_id"])
        for raw_vote in _sequence(
            spatial_part.get("semantic_votes", []),
            f"spatial part {part_id}.semantic_votes",
        )
        if isinstance(raw_vote, Mapping)
        for vote in [raw_vote]
        if (
            vote.get("alignment_trusted") is True
            and vote.get("unique_canonical_join") is True
            and vote.get("status") == "matched"
            and vote.get("canonical_group_id") == group_id
            and isinstance(vote.get("effective_confidence"), (int, float))
            and not isinstance(vote.get("effective_confidence"), bool)
            and math.isfinite(float(vote["effective_confidence"]))
            and float(vote["effective_confidence"]) >= minimum_semantic_confidence
            and isinstance(vote.get("view_id"), str)
            and vote.get("view_id") not in explicitly_invisible_views
        )
    }
    raw_semantic_conflict_floor = spatial_policy.get(
        "minimum_semantic_conflict_confidence"
    )
    semantic_alternative_views = {
        str(vote["view_id"])
        for raw_vote in _sequence(
            spatial_part.get("semantic_votes", []),
            f"spatial part {part_id}.semantic_votes",
        )
        if isinstance(raw_vote, Mapping)
        for vote in [raw_vote]
        if (
            vote.get("alignment_trusted") is True
            and vote.get("unique_canonical_join") is True
            and vote.get("status") in {"matched", "review"}
            and isinstance(vote.get("canonical_group_id"), str)
            and vote.get("canonical_group_id") != group_id
            and isinstance(vote.get("effective_confidence"), (int, float))
            and not isinstance(vote.get("effective_confidence"), bool)
            and isinstance(raw_semantic_conflict_floor, (int, float))
            and not isinstance(raw_semantic_conflict_floor, bool)
            and float(vote["effective_confidence"])
            >= float(raw_semantic_conflict_floor)
            and isinstance(vote.get("view_id"), str)
            and vote.get("view_id") not in explicitly_invisible_views
        )
    }

    semantic_override_views: list[str] = []
    dominant_support = next(
        (
            item
            for item in _sequence(
                deficit.get("supporting_views"),
                f"deficit {group_id}.supporting_views",
            )
            if isinstance(item, Mapping)
            and item.get("reference_view_id") in deficit_views
            and isinstance(item.get("deficit_sources"), Sequence)
            and not isinstance(item.get("deficit_sources"), (str, bytes))
            and "dominant_mass" in item["deficit_sources"]
        ),
        None,
    )
    direct_view_id = next(iter(supports), None) if len(supports) == 1 else None
    direct_observation = (
        direct_support_observations.get(direct_view_id)
        if direct_view_id is not None
        else None
    )

    def group_share(raw_scores: Any, canonical_group_id: str) -> float | None:
        if not isinstance(raw_scores, Sequence) or isinstance(raw_scores, (str, bytes)):
            return None
        total = 0.0
        for raw_score in raw_scores:
            if not isinstance(raw_score, Mapping):
                return None
            share = raw_score.get("color_share")
            if (
                isinstance(share, bool)
                or not isinstance(share, (int, float))
                or not math.isfinite(float(share))
                or not 0.0 <= float(share) <= 1.0
            ):
                return None
            if raw_score.get("canonical_group_id") == canonical_group_id:
                total += float(share)
        return total if total <= 1.0 + 1e-9 else None

    strict_projection = False
    if direct_observation is not None and direct_view_id is not None:
        alignment = alignment_audits.get(direct_view_id)
        group_scores = direct_observation.get("group_scores")
        bbox_scores = direct_observation.get("bbox_group_scores")
        winner = (
            group_scores[0]
            if isinstance(group_scores, Sequence)
            and not isinstance(group_scores, (str, bytes))
            and group_scores
            and isinstance(group_scores[0], Mapping)
            else None
        )
        winner_share = group_share(group_scores, group_id)
        bbox_share = group_share(bbox_scores, group_id)
        declared_pixels = direct_observation.get("declared_visible_pixels")
        projected_pixels = direct_observation.get("projected_part_pixels")
        direct_margin = direct_observation.get("color_margin")
        bbox_margin = direct_observation.get("bbox_color_margin")
        required_offsets = {(-2, 0), (2, 0), (0, -2), (0, 2)}
        perturbation_offsets: set[tuple[int, int]] = set()
        perturbations = direct_observation.get("projection_perturbations")
        perturbations_exact = (
            isinstance(perturbations, Sequence)
            and not isinstance(perturbations, (str, bytes))
            and len(perturbations) == len(required_offsets)
        )
        if perturbations_exact:
            for raw_perturbation in perturbations:
                if not isinstance(raw_perturbation, Mapping):
                    perturbations_exact = False
                    break
                offset = raw_perturbation.get("offset_pixels")
                sampled_pixels = raw_perturbation.get("sampled_reference_pixels")
                share = raw_perturbation.get("best_color_share")
                margin = raw_perturbation.get("color_margin")
                if (
                    not isinstance(offset, Sequence)
                    or isinstance(offset, (str, bytes))
                    or len(offset) != 2
                    or any(
                        isinstance(value, bool) or not isinstance(value, int)
                        for value in offset
                    )
                    or raw_perturbation.get("canonical_group_id") != group_id
                    or raw_perturbation.get("diagnostic_canonical_group_id") != group_id
                    or isinstance(sampled_pixels, bool)
                    or not isinstance(sampled_pixels, int)
                    or sampled_pixels < SEMANTIC_OVERRIDE_MIN_PROJECTED_PIXELS
                    or isinstance(share, bool)
                    or not isinstance(share, (int, float))
                    or not math.isfinite(float(share))
                    or float(share) < SEMANTIC_OVERRIDE_MIN_COLOR_SHARE
                    or isinstance(margin, bool)
                    or not isinstance(margin, (int, float))
                    or not math.isfinite(float(margin))
                    or float(margin) < SEMANTIC_OVERRIDE_MIN_COLOR_MARGIN
                ):
                    perturbations_exact = False
                    break
                perturbation_offsets.add((int(offset[0]), int(offset[1])))
            perturbations_exact = (
                perturbations_exact and perturbation_offsets == required_offsets
            )
        strict_projection = (
            isinstance(alignment, Mapping)
            and alignment.get("ecc_status") == "success"
            and float(alignment.get("score", -1.0))
            >= SEMANTIC_OVERRIDE_MIN_ALIGNMENT_SCORE
            and float(alignment.get("projection_iou", -1.0))
            >= SEMANTIC_OVERRIDE_MIN_PROJECTION_IOU
            and float(alignment.get("ecc_correlation", -1.0))
            >= SEMANTIC_OVERRIDE_MIN_ECC_CORRELATION
            and isinstance(declared_pixels, int)
            and not isinstance(declared_pixels, bool)
            and declared_pixels >= SEMANTIC_OVERRIDE_MIN_PROJECTED_PIXELS
            and isinstance(projected_pixels, int)
            and not isinstance(projected_pixels, bool)
            and projected_pixels >= SEMANTIC_OVERRIDE_MIN_PROJECTED_PIXELS
            and winner is not None
            and winner.get("canonical_group_id") == group_id
            and winner_share is not None
            and winner_share >= SEMANTIC_OVERRIDE_MIN_COLOR_SHARE
            and isinstance(direct_margin, (int, float))
            and not isinstance(direct_margin, bool)
            and math.isfinite(float(direct_margin))
            and float(direct_margin) >= SEMANTIC_OVERRIDE_MIN_COLOR_MARGIN
            and direct_observation.get("bbox_canonical_group_id") == group_id
            and bbox_share is not None
            and bbox_share >= SEMANTIC_OVERRIDE_MIN_BBOX_COLOR_SHARE
            and isinstance(bbox_margin, (int, float))
            and not isinstance(bbox_margin, bool)
            and math.isfinite(float(bbox_margin))
            and float(bbox_margin) >= SEMANTIC_OVERRIDE_MIN_BBOX_COLOR_MARGIN
            and perturbations_exact
        )

    rejected_group_nearly_absent = direct_observation is not None and all(
        (
            share := group_share(
                direct_observation.get("group_scores"),
                str(vote["canonical_group_id"]),
            )
        )
        is not None
        and share <= SEMANTIC_OVERRIDE_MAX_REJECTED_GROUP_SHARE
        for vote in semantic_conflict_rows
    )
    independent_semantic_anchor = any(
        not _independent_support_reasons(
            view_ids={str(direct_view_id), anchor_view_id},
            reference_evidence=reference_evidence,
        )
        for anchor_view_id in semantic_support_views - {str(direct_view_id)}
    )
    spatial_anchor_views = sorted(
        anchor_view_id
        for anchor_view_id in spatial_anchor_candidates
        if direct_view_id is not None
        and not _independent_support_reasons(
            view_ids={str(direct_view_id), anchor_view_id},
            reference_evidence=reference_evidence,
        )
    )

    def rejected_semantic_group_nearly_absent(
        conflict_vote: Mapping[str, Any],
    ) -> bool:
        view_id = conflict_vote.get("view_id")
        rejected_group_id = conflict_vote.get("canonical_group_id")
        if (
            not isinstance(view_id, str)
            or view_id not in canonical_spatial_anchor_candidates
            or not isinstance(rejected_group_id, str)
        ):
            return False
        observation = observations_by_view.get(view_id)
        diagnostic = (
            observation.get("canonical_palette_diagnostic")
            if isinstance(observation, Mapping)
            else None
        )
        if not isinstance(diagnostic, Mapping):
            return False
        samples = [
            diagnostic.get("direct_sample"),
            diagnostic.get("bbox_sample"),
            *_sequence(
                diagnostic.get("projection_perturbations", []),
                f"spatial part {part_id} canonical diagnostic perturbations",
            ),
        ]
        return _diagnostic_rejects_semantic_group(
            samples=samples,
            rejected_group_id=rejected_group_id,
        )

    if (
        direct_view_id is not None
        and dominant_support is None
        and not independent_semantic_anchor
        and not spatial_anchor_views
    ):
        reasons.append("SINGLE_VIEW_REQUIRES_DOMINANT_OR_INDEPENDENT_ANCHOR")
    dominant_semantic_override_applied = (
        dominant_support is not None
        and direct_view_id is not None
        and semantic_conflicts == {direct_view_id}
        and strict_projection
        and rejected_group_nearly_absent
        and (independent_semantic_anchor or allow_deferred_semantic_anchor)
        and not conflicting_groups
    )
    spatial_anchor_semantic_override_applied = (
        bool(semantic_conflicts)
        and direct_view_id is not None
        and bool(spatial_anchor_views)
        and semantic_conflicts <= set(spatial_anchor_views)
        and all(
            rejected_semantic_group_nearly_absent(vote)
            for vote in semantic_conflict_rows
        )
        and not conflicting_groups
    )
    semantic_override_applied = (
        dominant_semantic_override_applied or spatial_anchor_semantic_override_applied
    )
    if semantic_override_applied:
        semantic_override_views = sorted(semantic_conflicts)

    if conflicting_groups:
        reasons.append("SINGLE_VIEW_STABLE_SPATIAL_GROUP_CONFLICT")
    if semantic_conflicts and not semantic_override_applied:
        reasons.append("SINGLE_VIEW_MATCHED_SEMANTIC_GROUP_CONFLICT")
    if spatial_gate_decision is not None and _has_conflict(spatial_gate_decision):
        gate_semantic_conflicts = spatial_gate_decision.get(
            "semantic_conflicting_view_ids", []
        )
        other_gate_conflicts = any(
            spatial_gate_decision.get(field, [])
            for field in _CONFLICT_FIELDS
            if field != "semantic_conflicting_view_ids"
        )
        gate_override_applied = (
            isinstance(gate_semantic_conflicts, Sequence)
            and not isinstance(gate_semantic_conflicts, (str, bytes))
            and set(gate_semantic_conflicts) - explicitly_invisible_views
            <= semantic_alternative_views
            and (not semantic_conflicts or semantic_override_applied)
            and not other_gate_conflicts
            and spatial_gate_decision.get("output_group_id") in {None, group_id}
            and spatial_gate_decision.get("output_status") != "matched"
        )
        if not gate_override_applied:
            reasons.append("SPATIAL_OR_SEMANTIC_CONFLICT")
    if mapping_decision is not None and (
        mapping_decision.get("output_status") == "matched"
        and mapping_decision.get("output_group_id") != group_id
    ):
        reasons.append("MATCHED_MAPPING_CONSENSUS_GROUP_CONFLICT")
    if supports != deficit_views:
        reasons.append("SINGLE_VIEW_EXACT_SPATIAL_SUPPORT_MISSING")
    return (
        sorted(supports),
        sorted(set(reasons)),
        semantic_override_views,
        spatial_anchor_views,
    )


def _dominant_residual_spatial_support_views(
    *,
    part_id: str,
    group_id: str,
    spatial_part: Mapping[str, Any],
    spatial_gate_decision: Mapping[str, Any] | None,
    mapping_decision: Mapping[str, Any] | None,
    minimum_semantic_confidence: float,
    deficit: Mapping[str, Any],
    alignment_audits: Mapping[str, Mapping[str, Any]],
    spatial_policy: Mapping[str, Any],
    reference_evidence: Mapping[str, Mapping[str, str]],
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Localize one large chromatic residual with strict local evidence.

    This lane does not relax the global silhouette gate.  It handles a
    dominant chromatic family whose global comparison is silhouette-limited
    by requiring one large part to independently prove the same target with
    direct, bounding-box, perturbation, alignment, and same-view semantic
    evidence.  Any trusted conflicting semantic or spatial evidence blocks
    the lane.
    """

    reasons: list[str] = []
    residual_supports = {
        str(item["reference_view_id"]): item
        for item in _sequence(
            deficit.get("supporting_views"),
            f"deficit {group_id}.supporting_views",
        )
        if isinstance(item, Mapping)
        and isinstance(item.get("reference_view_id"), str)
        and item.get("requires_strict_local_projection") is True
        and DOMINANT_RESIDUAL_DEFICIT_SOURCE in item.get("deficit_sources", [])
    }
    residual_views = set(residual_supports)
    if len(residual_views) != 1:
        reasons.append("DOMINANT_RESIDUAL_REQUIRES_ONE_LOCAL_VIEW")

    supports: set[str] = set()
    conflicting_groups: set[str] = set()
    strict_observations: dict[str, Mapping[str, Any]] = {}
    required_offsets = {(-2, 0), (2, 0), (0, -2), (0, 2)}
    for raw_observation in _sequence(
        spatial_part.get("observations", []),
        f"spatial part {part_id}.observations",
    ):
        observation = _mapping(
            raw_observation,
            f"spatial part {part_id}.observation",
        )
        view_id = observation.get("reference_view_id")
        if not isinstance(view_id, str):
            continue
        canonical_group_id = observation.get("canonical_group_id")
        if (
            observation.get("classification") == "resolved"
            and isinstance(canonical_group_id, str)
            and canonical_group_id != group_id
        ):
            conflicting_groups.add(canonical_group_id)

        if view_id not in residual_views:
            continue
        alignment = alignment_audits.get(view_id)
        group_scores = observation.get("group_scores")
        bbox_scores = observation.get("bbox_group_scores")
        winner = (
            group_scores[0]
            if isinstance(group_scores, Sequence)
            and not isinstance(group_scores, (str, bytes))
            and group_scores
            and isinstance(group_scores[0], Mapping)
            else None
        )
        bbox_winner = (
            bbox_scores[0]
            if isinstance(bbox_scores, Sequence)
            and not isinstance(bbox_scores, (str, bytes))
            and bbox_scores
            and isinstance(bbox_scores[0], Mapping)
            else None
        )
        perturbations = observation.get("projection_perturbations")
        perturbation_offsets: set[tuple[int, int]] = set()
        perturbations_exact = (
            isinstance(perturbations, Sequence)
            and not isinstance(perturbations, (str, bytes))
            and len(perturbations) == len(required_offsets)
        )
        if perturbations_exact:
            for raw_perturbation in perturbations:
                if not isinstance(raw_perturbation, Mapping):
                    perturbations_exact = False
                    break
                offset = raw_perturbation.get("offset_pixels")
                pixels = raw_perturbation.get("sampled_reference_pixels")
                share = raw_perturbation.get("best_color_share")
                margin = raw_perturbation.get("color_margin")
                if (
                    not isinstance(offset, Sequence)
                    or isinstance(offset, (str, bytes))
                    or len(offset) != 2
                    or any(
                        isinstance(value, bool) or not isinstance(value, int)
                        for value in offset
                    )
                    or raw_perturbation.get("canonical_group_id") != group_id
                    or raw_perturbation.get("diagnostic_canonical_group_id") != group_id
                    or isinstance(pixels, bool)
                    or not isinstance(pixels, int)
                    or pixels < DOMINANT_RESIDUAL_MIN_PROJECTED_PIXELS
                    or isinstance(share, bool)
                    or not isinstance(share, (int, float))
                    or not math.isfinite(float(share))
                    or not DOMINANT_RESIDUAL_MIN_COLOR_SHARE <= float(share) <= 1.0
                    or isinstance(margin, bool)
                    or not isinstance(margin, (int, float))
                    or not math.isfinite(float(margin))
                    or not DOMINANT_RESIDUAL_MIN_COLOR_MARGIN <= float(margin) <= 1.0
                ):
                    perturbations_exact = False
                    break
                perturbation_offsets.add((int(offset[0]), int(offset[1])))
            perturbations_exact = (
                perturbations_exact and perturbation_offsets == required_offsets
            )

        declared_pixels = observation.get("declared_visible_pixels")
        projected_pixels = observation.get("projected_part_pixels")
        bbox_sampled_pixels = observation.get("bbox_sampled_reference_pixels")
        color_margin = observation.get("color_margin")
        bbox_margin = observation.get("bbox_color_margin")
        render_foreground_pixels = residual_supports.get(view_id, {}).get(
            "render_foreground_pixels"
        )
        alignment_score = (
            _optional_unit(alignment.get("score"))
            if isinstance(alignment, Mapping)
            else None
        )
        projection_iou = (
            _optional_unit(alignment.get("projection_iou"))
            if isinstance(alignment, Mapping)
            else None
        )
        ecc_correlation = (
            _optional_unit(alignment.get("ecc_correlation"))
            if isinstance(alignment, Mapping)
            else None
        )
        winner_share = (
            _optional_unit(winner.get("color_share"))
            if isinstance(winner, Mapping)
            else None
        )
        bbox_winner_share = (
            _optional_unit(bbox_winner.get("color_share"))
            if isinstance(bbox_winner, Mapping)
            else None
        )
        direct_margin = _optional_unit(color_margin)
        direct_bbox_margin = _optional_unit(bbox_margin)
        strict = (
            isinstance(alignment, Mapping)
            and alignment_score is not None
            and alignment_score >= DOMINANT_RESIDUAL_MIN_ALIGNMENT_SCORE
            and projection_iou is not None
            and projection_iou >= DOMINANT_RESIDUAL_MIN_PROJECTION_IOU
            and ecc_correlation is not None
            and ecc_correlation >= DOMINANT_RESIDUAL_MIN_ECC_CORRELATION
            and observation.get("classification") == "resolved"
            and canonical_group_id == group_id
            and observation.get("registration_label_stable") is True
            and observation.get("perturbation_label_stable") is True
            and isinstance(declared_pixels, int)
            and not isinstance(declared_pixels, bool)
            and declared_pixels >= DOMINANT_RESIDUAL_MIN_PROJECTED_PIXELS
            and isinstance(projected_pixels, int)
            and not isinstance(projected_pixels, bool)
            and projected_pixels >= DOMINANT_RESIDUAL_MIN_PROJECTED_PIXELS
            and isinstance(render_foreground_pixels, int)
            and not isinstance(render_foreground_pixels, bool)
            and render_foreground_pixels > 0
            and projected_pixels / render_foreground_pixels
            >= DOMINANT_RESIDUAL_MIN_RENDER_FOREGROUND_SHARE
            and winner is not None
            and winner.get("canonical_group_id") == group_id
            and winner_share is not None
            and winner_share >= DOMINANT_RESIDUAL_MIN_COLOR_SHARE
            and direct_margin is not None
            and direct_margin >= DOMINANT_RESIDUAL_MIN_COLOR_MARGIN
            and observation.get("bbox_canonical_group_id") == group_id
            and isinstance(bbox_sampled_pixels, int)
            and not isinstance(bbox_sampled_pixels, bool)
            and bbox_sampled_pixels >= DOMINANT_RESIDUAL_MIN_PROJECTED_PIXELS
            and bbox_winner is not None
            and bbox_winner.get("canonical_group_id") == group_id
            and bbox_winner_share is not None
            and bbox_winner_share >= DOMINANT_RESIDUAL_MIN_COLOR_SHARE
            and direct_bbox_margin is not None
            and direct_bbox_margin >= DOMINANT_RESIDUAL_MIN_COLOR_MARGIN
            and perturbations_exact
        )
        if strict:
            supports.add(view_id)
            strict_observations[view_id] = observation

    explicitly_invisible_views = _explicitly_invisible_semantic_view_ids(
        spatial_part=spatial_part,
        spatial_policy=spatial_policy,
        part_id=part_id,
    )
    raw_conflict_confidence = spatial_policy.get("minimum_semantic_conflict_confidence")
    semantic_conflict_confidence = (
        float(raw_conflict_confidence)
        if _optional_unit(raw_conflict_confidence) is not None
        else minimum_semantic_confidence
    )
    required_semantic_confidence = max(
        minimum_semantic_confidence,
        DOMINANT_RESIDUAL_MIN_SEMANTIC_CONFIDENCE,
    )
    semantic_supports = {
        str(vote["view_id"])
        for raw_vote in _sequence(
            spatial_part.get("semantic_votes", []),
            f"spatial part {part_id}.semantic_votes",
        )
        if isinstance(raw_vote, Mapping)
        for vote in [raw_vote]
        if (
            vote.get("alignment_trusted") is True
            and vote.get("unique_canonical_join") is True
            and vote.get("pixel_gate_accepted") is True
            and vote.get("status") == "matched"
            and vote.get("canonical_group_id") == group_id
            and isinstance(vote.get("effective_confidence"), (int, float))
            and not isinstance(vote.get("effective_confidence"), bool)
            and math.isfinite(float(vote["effective_confidence"]))
            and float(vote["effective_confidence"]) >= required_semantic_confidence
            and isinstance(vote.get("view_id"), str)
            and vote.get("view_id") in residual_views
            and vote.get("view_id") not in explicitly_invisible_views
            and vote.get("reference_sha256")
            == residual_supports[str(vote["view_id"])].get("reference_sha256")
        )
    }
    semantic_conflict_rows = [
        vote
        for raw_vote in _sequence(
            spatial_part.get("semantic_votes", []),
            f"spatial part {part_id}.semantic_votes",
        )
        if isinstance(raw_vote, Mapping)
        for vote in [raw_vote]
        if (
            vote.get("alignment_trusted") is True
            and vote.get("unique_canonical_join") is True
            and vote.get("pixel_gate_accepted") is True
            and vote.get("status") in {"matched", "review"}
            and isinstance(vote.get("canonical_group_id"), str)
            and vote.get("canonical_group_id") != group_id
            and isinstance(vote.get("effective_confidence"), (int, float))
            and not isinstance(vote.get("effective_confidence"), bool)
            and math.isfinite(float(vote["effective_confidence"]))
            and float(vote["effective_confidence"]) >= semantic_conflict_confidence
            and isinstance(vote.get("view_id"), str)
            and vote.get("view_id") not in explicitly_invisible_views
        )
    ]
    semantic_conflicts = {str(vote["view_id"]) for vote in semantic_conflict_rows}

    review_override_view_ids: list[str] = []
    semantic_anchor_view_ids: list[str] = []
    if len(residual_views) == 1 and supports == residual_views:
        residual_view_id = next(iter(residual_views))
        residual_observation = strict_observations.get(residual_view_id)
        same_view_alternatives = [
            vote
            for vote in semantic_conflict_rows
            if vote.get("view_id") == residual_view_id
        ]
        canonical_diagnostic = (
            residual_observation.get("canonical_palette_diagnostic")
            if isinstance(residual_observation, Mapping)
            else None
        )
        diagnostic_samples: list[Any] = []
        if isinstance(canonical_diagnostic, Mapping):
            diagnostic_samples = [
                canonical_diagnostic.get("direct_sample"),
                canonical_diagnostic.get("bbox_sample"),
                *_sequence(
                    canonical_diagnostic.get("projection_perturbations", []),
                    (
                        f"spatial part {part_id} dominant residual "
                        "canonical perturbations"
                    ),
                ),
            ]
        alternatives_disproved = (
            len(same_view_alternatives) == 1
            and same_view_alternatives[0].get("status") == "review"
            and len(diagnostic_samples) == 6
            and all(isinstance(sample, Mapping) for sample in diagnostic_samples)
            and _diagnostic_rejects_semantic_group(
                samples=[
                    _mapping(sample, "dominant residual diagnostic sample")
                    for sample in diagnostic_samples
                ],
                rejected_group_id=str(same_view_alternatives[0]["canonical_group_id"]),
            )
        )
        anchor_candidates = {
            str(vote["view_id"])
            for raw_vote in _sequence(
                spatial_part.get("semantic_votes", []),
                f"spatial part {part_id}.semantic_votes",
            )
            if isinstance(raw_vote, Mapping)
            for vote in [raw_vote]
            if (
                vote.get("alignment_trusted") is True
                and vote.get("unique_canonical_join") is True
                and vote.get("pixel_gate_accepted") is True
                and vote.get("status") in {"matched", "review"}
                and vote.get("canonical_group_id") == group_id
                and _optional_unit(vote.get("effective_confidence")) is not None
                and float(vote["effective_confidence"])
                >= DOMINANT_RESIDUAL_REVIEW_OVERRIDE_MIN_ANCHOR_CONFIDENCE
                and isinstance(vote.get("view_id"), str)
                and vote.get("view_id") != residual_view_id
                and vote.get("view_id") in alignment_audits
                and vote.get("view_id") in reference_evidence
                and vote.get("reference_sha256")
                == reference_evidence[str(vote["view_id"])]["raw_sha256"]
                and vote.get("view_id") not in explicitly_invisible_views
            )
        }
        if alternatives_disproved and anchor_candidates:
            review_override_view_ids = [residual_view_id]
            semantic_anchor_view_ids = sorted(anchor_candidates)

    semantic_review_override_applied = bool(
        review_override_view_ids
        and semantic_anchor_view_ids
        and semantic_conflicts == set(review_override_view_ids)
    )
    if supports != residual_views:
        reasons.append("DOMINANT_RESIDUAL_STRICT_PROJECTION_MISSING")
    if residual_views - semantic_supports and not semantic_review_override_applied:
        reasons.append("DOMINANT_RESIDUAL_SEMANTIC_CONFIRMATION_MISSING")
    if semantic_conflicts and not semantic_review_override_applied:
        reasons.append("DOMINANT_RESIDUAL_SEMANTIC_GROUP_CONFLICT")
    if conflicting_groups:
        reasons.append("DOMINANT_RESIDUAL_STABLE_SPATIAL_GROUP_CONFLICT")
    if spatial_gate_decision is not None and _has_conflict(spatial_gate_decision):
        gate_semantic_conflicts = spatial_gate_decision.get(
            "semantic_conflicting_view_ids", []
        )
        other_gate_conflicts = any(
            spatial_gate_decision.get(field, [])
            for field in _CONFLICT_FIELDS
            if field != "semantic_conflicting_view_ids"
        )
        gate_override_applied = (
            semantic_review_override_applied
            and isinstance(gate_semantic_conflicts, Sequence)
            and not isinstance(gate_semantic_conflicts, (str, bytes))
            and set(gate_semantic_conflicts) <= set(review_override_view_ids)
            and not other_gate_conflicts
            and spatial_gate_decision.get("output_group_id") in {None, group_id}
            and spatial_gate_decision.get("output_status") != "matched"
        )
        if not gate_override_applied:
            reasons.append("SPATIAL_OR_SEMANTIC_CONFLICT")
    if mapping_decision is not None and (
        mapping_decision.get("output_status") == "matched"
        and mapping_decision.get("output_group_id") != group_id
    ):
        reasons.append("MATCHED_MAPPING_CONSENSUS_GROUP_CONFLICT")
    return (
        sorted(supports),
        sorted(set(reasons)),
        review_override_view_ids if not reasons else [],
        semantic_anchor_view_ids if not reasons else [],
    )


def _linear_quantile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise QualityRepairError("cannot compute a quantile of no values")
    index = (len(ordered) - 1) * quantile
    lower = int(math.floor(index))
    upper = int(math.ceil(index))
    if lower == upper:
        return ordered[lower]
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _dark_foreground_spatial_support(
    *,
    part_id: str,
    group_id: str,
    canonical_group: Mapping[str, Any],
    spatial_part: Mapping[str, Any],
    spatial_gate_decision: Mapping[str, Any] | None,
    mapping_decision: Mapping[str, Any] | None,
    deficit: Mapping[str, Any],
    alignment_audits: Mapping[str, Mapping[str, Any]],
    reference_evidence: Mapping[str, Mapping[str, str]],
    canonical_reference_evidence: Mapping[str, Mapping[str, str]],
    minimum_semantic_confidence: float,
) -> tuple[list[str], list[str], dict[str, Any] | None]:
    """Validate one hash-bound dark-object projection against black QA mass."""

    reasons: list[str] = []
    supports = [
        item
        for item in _sequence(
            deficit.get("supporting_views"),
            f"dark deficit {group_id}.supporting_views",
        )
        if isinstance(item, Mapping)
        and DARK_FOREGROUND_RESIDUAL_DEFICIT_SOURCE in item.get("deficit_sources", [])
    ]
    if len(supports) != 1:
        return [], ["DARK_RESIDUAL_REQUIRES_ONE_SUPPORTING_VIEW"], None
    support = supports[0]
    view_id = support.get("reference_view_id")
    if not isinstance(view_id, str):
        return [], ["DARK_RESIDUAL_SUPPORT_VIEW_INVALID"], None
    if str(canonical_group.get("base_color", "")).strip().casefold() != "black":
        reasons.append("DARK_RESIDUAL_CANONICAL_GROUP_NOT_BLACK")

    matching_observations = [
        observation
        for raw_observation in _sequence(
            spatial_part.get("observations", []),
            f"spatial part {part_id}.observations",
        )
        for observation in [
            _mapping(raw_observation, f"spatial part {part_id}.observation")
        ]
        if observation.get("reference_view_id") == view_id
    ]
    if len(matching_observations) != 1:
        reasons.append("DARK_DIAGNOSTIC_VIEW_OBSERVATION_NOT_UNIQUE")
        return [], sorted(set(reasons)), None
    observation = matching_observations[0]
    diagnostic = observation.get("dark_foreground_diagnostic")
    if not isinstance(diagnostic, Mapping):
        return [], ["DARK_FOREGROUND_DIAGNOSTIC_MISSING"], None

    diagnostic_hash = diagnostic.get("diagnostic_sha256")
    unsigned_diagnostic = copy.deepcopy(dict(diagnostic))
    unsigned_diagnostic.pop("diagnostic_sha256", None)
    if (
        not isinstance(diagnostic_hash, str)
        or not _SHA256_PATTERN.fullmatch(diagnostic_hash)
        or _canonical_sha256(unsigned_diagnostic) != diagnostic_hash
    ):
        reasons.append("DARK_DIAGNOSTIC_HASH_INVALID")
    if (
        diagnostic.get("status") != "resolved"
        or diagnostic.get("reason_codes") != []
        or diagnostic.get("evidence_scope") != "dark_on_black_foreground_repair_only"
        or diagnostic.get("canonical_group_id") != group_id
        or diagnostic.get("alternative_canonical_group_ids") != []
    ):
        reasons.append("DARK_DIAGNOSTIC_NOT_EXACT_RESOLVED_TARGET")

    source_view_ids = diagnostic.get("canonical_source_view_ids")
    canonical_source_view_ids = canonical_group.get("source_view_ids")
    source_view_ids_valid = (
        isinstance(source_view_ids, Sequence)
        and not isinstance(source_view_ids, (str, bytes))
        and list(source_view_ids) == sorted(set(source_view_ids))
        and len(source_view_ids) >= 2
        and isinstance(canonical_source_view_ids, Sequence)
        and not isinstance(canonical_source_view_ids, (str, bytes))
        and list(source_view_ids)
        == sorted(set(str(value) for value in canonical_source_view_ids))
        and all(
            isinstance(source_view_id, str)
            and source_view_id in canonical_reference_evidence
            for source_view_id in source_view_ids
        )
    )
    if not source_view_ids_valid:
        reasons.append("DARK_DIAGNOSTIC_CANONICAL_SOURCES_INVALID")
    else:
        source_hashes = {
            canonical_reference_evidence[str(source_view_id)]["raw_sha256"]
            for source_view_id in source_view_ids
        }
        source_contents = {
            canonical_reference_evidence[str(source_view_id)]["content_cluster_id"]
            for source_view_id in source_view_ids
        }
        if len(source_hashes) < 2 or len(source_contents) < 2:
            reasons.append("DARK_DIAGNOSTIC_CANONICAL_SOURCES_NOT_DISTINCT")

    projected_pixels = diagnostic.get("projected_part_pixels")
    normalized_pixels = diagnostic.get("normalized_projected_pixels")
    if (
        isinstance(projected_pixels, bool)
        or not isinstance(projected_pixels, int)
        or projected_pixels < 1
        or projected_pixels != observation.get("projected_part_pixels")
    ):
        reasons.append("DARK_DIAGNOSTIC_PROJECTED_PIXELS_INVALID")
    if (
        isinstance(normalized_pixels, bool)
        or not isinstance(normalized_pixels, int)
        or normalized_pixels < 96
    ):
        reasons.append("DARK_DIAGNOSTIC_NORMALIZED_PIXELS_INVALID")

    normalization = diagnostic.get("normalization")
    if not isinstance(normalization, Mapping):
        reasons.append("DARK_DIAGNOSTIC_NORMALIZATION_INVALID")
    else:
        original_size = normalization.get("original_size")
        normalized_size = normalization.get("normalized_size")
        scale = normalization.get("scale")
        size_contract = (
            normalization.get("long_edge_pixels") == 512
            and isinstance(original_size, Sequence)
            and not isinstance(original_size, (str, bytes))
            and len(original_size) == 2
            and isinstance(normalized_size, Sequence)
            and not isinstance(normalized_size, (str, bytes))
            and len(normalized_size) == 2
            and all(
                isinstance(value, int) and not isinstance(value, bool) and value > 0
                for value in [*original_size, *normalized_size]
            )
            and max(int(value) for value in normalized_size) == 512
            and isinstance(scale, (int, float))
            and not isinstance(scale, bool)
            and math.isfinite(float(scale))
            and float(scale) > 0.0
        )
        if not size_contract:
            reasons.append("DARK_DIAGNOSTIC_NORMALIZATION_INVALID")

    diagnostic_alignment = diagnostic.get("alignment")
    trusted_alignment = alignment_audits.get(view_id)
    if not isinstance(diagnostic_alignment, Mapping) or not isinstance(
        trusted_alignment, Mapping
    ):
        reasons.append("DARK_DIAGNOSTIC_ALIGNMENT_INVALID")
    else:
        alignment_contract = (
            diagnostic_alignment.get("trusted") is True
            and diagnostic_alignment.get("strong") is True
            and diagnostic_alignment.get("ecc_status") == "success"
            and diagnostic_alignment.get("reason_codes_empty") is True
            and diagnostic_alignment.get("transform_constraints_passed") is True
        )
        for field, minimum in (
            ("score", DARK_FOREGROUND_MIN_ALIGNMENT_SCORE),
            ("projection_iou", 0.85),
            ("ecc_correlation", 0.90),
        ):
            value = _optional_unit(diagnostic_alignment.get(field))
            trusted_value = _optional_unit(trusted_alignment.get(field))
            if (
                value is None
                or value < minimum
                or trusted_value is None
                or not math.isclose(value, trusted_value, abs_tol=5e-8)
            ):
                alignment_contract = False
        projection_score = _optional_unit(diagnostic_alignment.get("projection_score"))
        if projection_score is None or projection_score < 0.85:
            alignment_contract = False
        if not alignment_contract:
            reasons.append("DARK_DIAGNOSTIC_ALIGNMENT_INVALID")

    thresholds = diagnostic.get("thresholds")
    if (
        not isinstance(thresholds, Mapping)
        or dict(thresholds) != DARK_FOREGROUND_DIAGNOSTIC_THRESHOLDS
    ):
        reasons.append("DARK_DIAGNOSTIC_THRESHOLDS_INVALID")
    selected_thresholds = DARK_FOREGROUND_DIAGNOSTIC_THRESHOLDS

    def count(field: str) -> int | None:
        value = diagnostic.get(field)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
        reasons.append(f"DARK_DIAGNOSTIC_COUNT_INVALID:{field}")
        return None

    def unit(field: str) -> float | None:
        value = _optional_unit(diagnostic.get(field))
        if value is None:
            reasons.append(f"DARK_DIAGNOSTIC_RATIO_INVALID:{field}")
        return value

    near_black_pixels = count("near_black_pixels")
    non_background_pixels = count("non_background_pixels")
    dark_signal_pixels = count("dark_signal_pixels")
    core_pixels = count("core_pixels")
    dark_core_pixels = count("core_dark_signal_pixels")
    edge_support_pixels = count("adaptive_edge_pixels")
    near_black_share = unit("near_black_share")
    non_background_share = unit("non_background_share")
    dark_signal_share = unit("dark_signal_share")
    dark_signal_purity = unit("dark_signal_purity")
    dark_core_share = unit("core_dark_signal_share")
    edge_density = unit("adaptive_edge_density")
    count_values = (
        near_black_pixels,
        non_background_pixels,
        dark_signal_pixels,
        core_pixels,
        dark_core_pixels,
        edge_support_pixels,
    )
    if normalized_pixels is not None and all(
        value is not None for value in count_values
    ):
        numeric_contracts = (
            (
                near_black_share,
                near_black_pixels / normalized_pixels,
            ),
            (
                non_background_share,
                non_background_pixels / normalized_pixels,
            ),
            (
                dark_signal_share,
                dark_signal_pixels / normalized_pixels,
            ),
            (
                dark_signal_purity,
                (
                    dark_signal_pixels / non_background_pixels
                    if non_background_pixels
                    else 0.0
                ),
            ),
            (
                dark_core_share,
                dark_core_pixels / core_pixels if core_pixels else 0.0,
            ),
            (
                edge_density,
                edge_support_pixels / normalized_pixels,
            ),
        )
        if any(
            actual is None
            or not math.isclose(float(actual), float(expected), abs_tol=1e-8)
            for actual, expected in numeric_contracts
        ):
            reasons.append("DARK_DIAGNOSTIC_NUMERIC_EVIDENCE_INCONSISTENT")
        if (
            normalized_pixels
            < selected_thresholds["minimum_normalized_projected_pixels"]
            or near_black_share < selected_thresholds["minimum_near_black_share"]
            or non_background_pixels
            < selected_thresholds["minimum_non_background_pixels"]
            or dark_signal_share < selected_thresholds["minimum_dark_signal_share"]
            or dark_signal_purity < selected_thresholds["minimum_dark_signal_purity"]
            or core_pixels < selected_thresholds["minimum_core_pixels"]
            or dark_core_share < selected_thresholds["minimum_core_dark_signal_share"]
            or edge_density < selected_thresholds["minimum_adaptive_edge_density"]
        ):
            reasons.append("DARK_DIAGNOSTIC_PRIMITIVE_GATE_FAILED")

    core_distance = diagnostic.get("core_distance_pixels")
    if (
        not isinstance(core_distance, (int, float))
        or isinstance(core_distance, bool)
        or not math.isfinite(float(core_distance))
        or (float(core_distance) != selected_thresholds["core_distance_pixels"])
    ):
        reasons.append("DARK_DIAGNOSTIC_CORE_DISTANCE_INVALID")
    for field in ("canny_low_threshold", "canny_high_threshold"):
        value = diagnostic.get(field)
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or not 0 <= value <= 255
        ):
            reasons.append(f"DARK_DIAGNOSTIC_EDGE_THRESHOLD_INVALID:{field}")
    if (
        isinstance(diagnostic.get("canny_low_threshold"), int)
        and isinstance(diagnostic.get("canny_high_threshold"), int)
        and diagnostic["canny_low_threshold"] >= diagnostic["canny_high_threshold"]
    ):
        reasons.append("DARK_DIAGNOSTIC_EDGE_THRESHOLDS_NOT_ORDERED")

    raw_nulls = diagnostic.get("null_shifts")
    null_shares: list[float] = []
    valid_null_count = 0
    if (
        not isinstance(raw_nulls, Sequence)
        or isinstance(raw_nulls, (str, bytes))
        or len(raw_nulls) != 8
    ):
        reasons.append("DARK_DIAGNOSTIC_NULL_SHIFTS_INVALID")
    else:
        offsets: set[tuple[int, int]] = set()
        for raw_null in raw_nulls:
            if not isinstance(raw_null, Mapping):
                reasons.append("DARK_DIAGNOSTIC_NULL_SHIFTS_INVALID")
                continue
            offset = raw_null.get("offset_pixels")
            ratio = _optional_unit(raw_null.get("valid_area_ratio"))
            null_pixels = raw_null.get("dark_signal_pixels")
            share = _optional_unit(raw_null.get("dark_signal_share"))
            retained_pixels = raw_null.get("retained_pixels")
            valid = raw_null.get("valid")
            if (
                not isinstance(offset, Sequence)
                or isinstance(offset, (str, bytes))
                or len(offset) != 2
                or any(
                    not isinstance(value, int) or isinstance(value, bool)
                    for value in offset
                )
                or (int(offset[0]), int(offset[1])) == (0, 0)
                or max(abs(int(offset[0])), abs(int(offset[1])))
                < selected_thresholds["minimum_null_offset_pixels"]
                or (int(offset[0]), int(offset[1])) in offsets
                or ratio is None
                or not isinstance(retained_pixels, int)
                or isinstance(retained_pixels, bool)
                or retained_pixels < 0
                or not isinstance(null_pixels, int)
                or isinstance(null_pixels, bool)
                or null_pixels < 0
                or share is None
                or not isinstance(valid, bool)
                or valid
                != (ratio >= selected_thresholds["minimum_null_valid_area_ratio"])
                or normalized_pixels is None
                or not math.isclose(
                    ratio,
                    retained_pixels / normalized_pixels,
                    abs_tol=1e-8,
                )
                or not math.isclose(
                    share,
                    null_pixels / retained_pixels if retained_pixels else 0.0,
                    abs_tol=1e-8,
                )
                or not isinstance(raw_null.get("mask_sha256"), str)
                or not _SHA256_PATTERN.fullmatch(str(raw_null.get("mask_sha256")))
            ):
                reasons.append("DARK_DIAGNOSTIC_NULL_SHIFTS_INVALID")
                continue
            offsets.add((int(offset[0]), int(offset[1])))
            if valid:
                valid_null_count += 1
                null_shares.append(share)
        declared_count = diagnostic.get("valid_null_shift_count")
        q75 = _optional_unit(diagnostic.get("null_dark_signal_share_q75"))
        margin = diagnostic.get("dark_signal_null_margin")
        recomputed_q75 = _linear_quantile(null_shares, 0.75) if null_shares else None
        if (
            not isinstance(declared_count, int)
            or isinstance(declared_count, bool)
            or declared_count != valid_null_count
            or q75 is None
            or recomputed_q75 is None
            or not math.isclose(q75, recomputed_q75, abs_tol=1e-8)
            or not isinstance(margin, (int, float))
            or isinstance(margin, bool)
            or not math.isfinite(float(margin))
            or dark_signal_share is None
            or not math.isclose(
                float(margin),
                dark_signal_share - recomputed_q75,
                abs_tol=1e-8,
            )
            or (
                valid_null_count < selected_thresholds["minimum_valid_null_shifts"]
                or float(margin) < selected_thresholds["minimum_null_q75_margin"]
            )
        ):
            reasons.append("DARK_DIAGNOSTIC_NULL_CONTROL_INVALID")

    for hash_field in (
        "normalized_reference_pixel_sha256",
        "normalized_projected_mask_sha256",
        "normalized_near_black_mask_sha256",
        "normalized_non_background_mask_sha256",
        "normalized_dark_signal_mask_sha256",
        "normalized_adaptive_edge_mask_sha256",
    ):
        value = diagnostic.get(hash_field)
        if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
            reasons.append(f"DARK_DIAGNOSTIC_HASH_INVALID:{hash_field}")

    trusted_mapping_conflict = (
        mapping_decision is not None
        and mapping_decision.get("output_status") == "matched"
        and mapping_decision.get("output_group_id") != group_id
    )
    trusted_gate_conflict = (
        spatial_gate_decision is not None
        and spatial_gate_decision.get("output_status") == "matched"
        and spatial_gate_decision.get("output_group_id") != group_id
    )
    trusted_semantic_conflict = any(
        vote.get("alignment_trusted") is True
        and vote.get("unique_canonical_join") is True
        and vote.get("pixel_gate_accepted") is True
        and vote.get("status") == "matched"
        and isinstance(vote.get("canonical_group_id"), str)
        and vote.get("canonical_group_id") != group_id
        and _optional_unit(vote.get("effective_confidence")) is not None
        and float(vote["effective_confidence"]) >= minimum_semantic_confidence
        for vote in _sequence(
            spatial_part.get("semantic_votes", []),
            f"spatial part {part_id}.semantic_votes",
        )
        if isinstance(vote, Mapping)
    )
    if trusted_mapping_conflict or trusted_gate_conflict or trusted_semantic_conflict:
        reasons.append("DARK_RESIDUAL_MATCHED_SEMANTIC_GROUP_CONFLICT")

    reference_record = reference_evidence.get(view_id)
    if reference_record is None or reference_record["raw_sha256"] != support.get(
        "reference_sha256"
    ):
        reasons.append("DARK_RESIDUAL_REFERENCE_HASH_MISMATCH")

    if reasons:
        return [], sorted(set(reasons)), None
    assert isinstance(projected_pixels, int)
    assert dark_signal_share is not None
    reference_pixels = int(support["normalized_reference_pixels"])
    render_pixels = int(support["render_foreground_pixels"])
    contribution = int(
        math.ceil(
            projected_pixels / render_pixels * reference_pixels * dark_signal_share
        )
    )
    evidence_strength = (
        dark_signal_share
        + float(diagnostic["dark_signal_purity"])
        + float(diagnostic["dark_signal_null_margin"])
        + float(diagnostic["adaptive_edge_density"])
    )
    audit = {
        "diagnostic_sha256": diagnostic_hash,
        "projected_part_pixels": projected_pixels,
        "normalized_projected_pixels": normalized_pixels,
        "dark_signal_share": dark_signal_share,
        "dark_signal_purity": diagnostic["dark_signal_purity"],
        "dark_signal_null_margin": diagnostic["dark_signal_null_margin"],
        "adaptive_edge_density": diagnostic["adaptive_edge_density"],
        "estimated_contribution_pixels": contribution,
        "evidence_strength": evidence_strength,
    }
    return [view_id], [], audit


def _multiview_dark_identity_support(
    *,
    part_id: str,
    group_id: str,
    canonical_group: Mapping[str, Any],
    spatial_part: Mapping[str, Any],
    spatial_gate_decision: Mapping[str, Any] | None,
    mapping_decision: Mapping[str, Any] | None,
    deficit: Mapping[str, Any],
    reference_evidence: Mapping[str, Mapping[str, str]],
    spatial_policy: Mapping[str, Any],
    minimum_semantic_confidence: float,
) -> tuple[list[str], list[str], list[str], dict[str, Any] | None]:
    """Authorize a dark repair from independent, view-invariant part identity.

    The QA view that proves a global black-material deficit need not be the
    same view that best exposes a particular black part.  This lane therefore
    separates those two facts: ``deficit`` authorizes only the missing
    canonical group, while two independent registered projections must agree
    on the part identity.  It remains fail-closed for a single projection,
    unstable labels, an alternative resolved group, or more than one trusted
    semantic disagreement.
    """

    reasons: list[str] = []
    if str(canonical_group.get("base_color", "")).strip().casefold() != "black":
        reasons.append("MULTIVIEW_DARK_CANONICAL_GROUP_NOT_BLACK")

    deficit_supports = [
        item
        for item in _sequence(
            deficit.get("supporting_views"),
            f"dark deficit {group_id}.supporting_views",
        )
        if isinstance(item, Mapping)
        and DARK_FOREGROUND_RESIDUAL_DEFICIT_SOURCE in item.get("deficit_sources", [])
    ]
    if len(deficit_supports) != 1:
        return (
            [],
            ["MULTIVIEW_DARK_REQUIRES_ONE_GLOBAL_DEFICIT_VIEW"],
            [],
            None,
        )
    deficit_support = deficit_supports[0]

    consensus = spatial_part.get("multiview_dark_consensus")
    if not isinstance(consensus, Mapping):
        return [], ["MULTIVIEW_DARK_CONSENSUS_MISSING"], [], None
    support_view_ids = consensus.get("supporting_view_ids")
    minimum_support = spatial_policy.get("minimum_spatial_support_views")
    consensus_contract_valid = (
        consensus.get("status") == "resolved"
        and consensus.get("canonical_group_id") == group_id
        and consensus.get("evidence_contract")
        == "stable_projection_and_dark_interior_multiview_consensus"
        and isinstance(minimum_support, int)
        and not isinstance(minimum_support, bool)
        and minimum_support >= 2
        and consensus.get("minimum_independent_support_views") == minimum_support
        and isinstance(support_view_ids, Sequence)
        and not isinstance(support_view_ids, (str, bytes))
        and list(support_view_ids)
        == sorted(set(str(view_id) for view_id in support_view_ids))
        and len(support_view_ids) >= minimum_support
    )
    if not consensus_contract_valid:
        return [], ["MULTIVIEW_DARK_CONSENSUS_CONTRACT_INVALID"], [], None
    support_views = [str(view_id) for view_id in support_view_ids]
    reasons.extend(
        _independent_support_reasons(
            view_ids=set(support_views),
            reference_evidence=reference_evidence,
        )
    )

    observations_by_view: dict[str, Mapping[str, Any]] = {}
    for raw_observation in _sequence(
        spatial_part.get("observations", []),
        f"spatial part {part_id}.observations",
    ):
        observation = _mapping(raw_observation, f"spatial part {part_id}.observation")
        view_id = observation.get("reference_view_id")
        if isinstance(view_id, str) and view_id in support_views:
            if view_id in observations_by_view:
                reasons.append("MULTIVIEW_DARK_SUPPORT_OBSERVATION_NOT_UNIQUE")
            observations_by_view[view_id] = observation
    if set(observations_by_view) != set(support_views):
        reasons.append("MULTIVIEW_DARK_SUPPORT_OBSERVATION_MISSING")

    minimum_color_share = float(spatial_policy.get("minimum_color_share", 1.0))
    evidence_rows: list[dict[str, Any]] = []
    for view_id in support_views:
        observation = observations_by_view.get(view_id)
        if observation is None:
            continue
        nested_consensus = observation.get("multiview_dark_consensus")
        scores = observation.get("group_scores")
        winner = (
            scores[0]
            if isinstance(scores, Sequence)
            and not isinstance(scores, (str, bytes))
            and scores
            and isinstance(scores[0], Mapping)
            else None
        )
        diagnostic = observation.get("dark_foreground_diagnostic")
        if (
            observation.get("classification") != "resolved"
            or observation.get("reason_code") != "multiview_dark_consensus_resolved"
            or observation.get("canonical_group_id") != group_id
            or observation.get("bbox_canonical_group_id") != group_id
            or observation.get("registration_label_stable") is not True
            or observation.get("perturbation_label_stable") is not True
            or not isinstance(nested_consensus, Mapping)
            or dict(nested_consensus) != dict(consensus)
            or winner is None
            or winner.get("canonical_group_id") != group_id
            or str(winner.get("base_color", "")).strip().casefold() != "black"
            or _optional_unit(winner.get("color_share")) is None
            or float(winner["color_share"]) < minimum_color_share
            or not isinstance(diagnostic, Mapping)
        ):
            reasons.append(f"MULTIVIEW_DARK_SUPPORT_CONTRACT_INVALID:{view_id}")
            continue
        diagnostic_hash = diagnostic.get("diagnostic_sha256")
        unsigned_diagnostic = copy.deepcopy(dict(diagnostic))
        unsigned_diagnostic.pop("diagnostic_sha256", None)
        projected_pixels = diagnostic.get("projected_part_pixels")
        normalized_pixels = diagnostic.get("normalized_projected_pixels")
        non_background_share = _optional_unit(diagnostic.get("non_background_share"))
        dark_signal_share = _optional_unit(diagnostic.get("dark_signal_share"))
        dark_signal_purity = _optional_unit(diagnostic.get("dark_signal_purity"))
        core_dark_share = _optional_unit(diagnostic.get("core_dark_signal_share"))
        null_margin = diagnostic.get("dark_signal_null_margin")
        if (
            not isinstance(diagnostic_hash, str)
            or not _SHA256_PATTERN.fullmatch(diagnostic_hash)
            or _canonical_sha256(unsigned_diagnostic) != diagnostic_hash
            or diagnostic.get("canonical_group_id") != group_id
            or isinstance(projected_pixels, bool)
            or not isinstance(projected_pixels, int)
            or projected_pixels < 1
            or projected_pixels != observation.get("projected_part_pixels")
            or isinstance(normalized_pixels, bool)
            or not isinstance(normalized_pixels, int)
            or normalized_pixels
            < int(
                DARK_FOREGROUND_DIAGNOSTIC_THRESHOLDS[
                    "minimum_normalized_projected_pixels"
                ]
            )
            or non_background_share is None
            or non_background_share < 0.20
            or dark_signal_share is None
            or dark_signal_share
            < float(DARK_FOREGROUND_DIAGNOSTIC_THRESHOLDS["minimum_dark_signal_share"])
            or dark_signal_purity is None
            or dark_signal_purity
            < float(DARK_FOREGROUND_DIAGNOSTIC_THRESHOLDS["minimum_dark_signal_purity"])
            or core_dark_share is None
            or core_dark_share
            < float(
                DARK_FOREGROUND_DIAGNOSTIC_THRESHOLDS["minimum_core_dark_signal_share"]
            )
            or not isinstance(null_margin, (int, float))
            or isinstance(null_margin, bool)
            or not math.isfinite(float(null_margin))
            or float(null_margin)
            < float(DARK_FOREGROUND_DIAGNOSTIC_THRESHOLDS["minimum_null_q75_margin"])
        ):
            reasons.append(f"MULTIVIEW_DARK_DIAGNOSTIC_INVALID:{view_id}")
            continue
        evidence_rows.append(
            {
                "reference_view_id": view_id,
                "diagnostic_sha256": diagnostic_hash,
                "projected_part_pixels": projected_pixels,
                "normalized_projected_pixels": normalized_pixels,
                "dark_signal_share": dark_signal_share,
                "dark_signal_purity": dark_signal_purity,
                "core_dark_signal_share": core_dark_share,
                "dark_signal_null_margin": float(null_margin),
                "direct_black_share": float(winner["color_share"]),
            }
        )

    resolved_alternatives = {
        str(observation.get("canonical_group_id"))
        for observation in (
            _mapping(raw, f"spatial part {part_id}.observation")
            for raw in _sequence(
                spatial_part.get("observations", []),
                f"spatial part {part_id}.observations",
            )
        )
        if observation.get("classification") == "resolved"
        and isinstance(observation.get("canonical_group_id"), str)
        and observation.get("canonical_group_id") != group_id
    }
    if resolved_alternatives:
        reasons.append("MULTIVIEW_DARK_RESOLVED_ALTERNATIVE_GROUP")
    if (
        spatial_gate_decision is not None
        and spatial_gate_decision.get("output_status") == "matched"
        and spatial_gate_decision.get("output_group_id") != group_id
    ):
        reasons.append("MULTIVIEW_DARK_MATCHED_SPATIAL_GATE_CONFLICT")
    if (
        mapping_decision is not None
        and mapping_decision.get("output_status") == "matched"
        and mapping_decision.get("output_group_id") != group_id
    ):
        reasons.append("MULTIVIEW_DARK_MATCHED_MAPPING_CONFLICT")

    semantic_conflict_view_ids = sorted(
        {
            str(vote["view_id"])
            for vote in _sequence(
                spatial_part.get("semantic_votes", []),
                f"spatial part {part_id}.semantic_votes",
            )
            if isinstance(vote, Mapping)
            and isinstance(vote.get("view_id"), str)
            and vote.get("alignment_trusted") is True
            and vote.get("unique_canonical_join") is True
            and vote.get("pixel_gate_accepted") is True
            and vote.get("status") == "matched"
            and isinstance(vote.get("canonical_group_id"), str)
            and vote.get("canonical_group_id") != group_id
            and _optional_unit(vote.get("effective_confidence")) is not None
            and float(vote["effective_confidence"]) >= minimum_semantic_confidence
        }
    )
    if len(semantic_conflict_view_ids) > 1:
        reasons.append("MULTIVIEW_DARK_MULTIPLE_SEMANTIC_CONFLICTS")

    if reasons:
        return (
            support_views,
            sorted(set(reasons)),
            [],
            None,
        )
    reference_pixels = int(deficit_support["normalized_reference_pixels"])
    render_pixels = int(deficit_support["render_foreground_pixels"])
    conservative_projected_dark_pixels = min(
        float(row["projected_part_pixels"]) * float(row["dark_signal_share"])
        for row in evidence_rows
    )
    contribution = int(
        math.ceil(conservative_projected_dark_pixels / render_pixels * reference_pixels)
    )
    evidence_strength = min(
        float(row["direct_black_share"])
        + float(row["dark_signal_share"])
        + float(row["dark_signal_purity"])
        + float(row["dark_signal_null_margin"])
        for row in evidence_rows
    )
    audit_body = {
        "lane": MULTIVIEW_DARK_IDENTITY_LANE,
        "canonical_group_id": group_id,
        "supporting_view_ids": support_views,
        "evidence_contract": consensus["evidence_contract"],
        "evidence_rows": evidence_rows,
    }
    audit = {
        **audit_body,
        "diagnostic_sha256": _canonical_sha256(audit_body),
        "estimated_contribution_pixels": contribution,
        "evidence_strength": evidence_strength,
    }
    return support_views, [], semantic_conflict_view_ids, audit


def _anchored_single_view_spatial_support_views(
    *,
    part_id: str,
    group_id: str,
    spatial_part: Mapping[str, Any],
    spatial_gate_decision: Mapping[str, Any] | None,
    mapping_decision: Mapping[str, Any] | None,
    spatial_policy: Mapping[str, Any],
    minimum_semantic_confidence: float,
    deficit: Mapping[str, Any],
    alignment_audits: Mapping[str, Mapping[str, Any]],
    reference_evidence: Mapping[str, Mapping[str, str]],
) -> tuple[list[str], list[str], list[str]]:
    """Return one high-purity direct projection pending a multi-view anchor.

    Unlike the exact single-QA-view lane, this lane targets a group missing in
    multiple trusted QA views.  A single part is only provisionally localized
    here; a second compilation phase must find an independently accepted
    stable/bounded proposals for the same group whose combined supports cover
    every deficit view.  Diagnostic-only observations are deliberately
    ineligible.  The stricter semantic-anchor lane may supply the one direct
    projection when a same-view Qwen localization is disproven by its pixels.
    """

    reasons: list[str] = []
    if deficit.get("repairable") is not True:
        reasons.append("ANCHORED_SINGLE_REQUIRES_REPAIRABLE_MULTIVIEW_GROUP")
    deficit_views = {
        str(item["reference_view_id"])
        for item in _sequence(
            deficit.get("supporting_views"),
            f"deficit {group_id}.supporting_views",
        )
        if isinstance(item, Mapping) and isinstance(item.get("reference_view_id"), str)
    }
    if len(deficit_views) < 2:
        reasons.append("ANCHORED_SINGLE_REQUIRES_MULTIVIEW_QA_DEFICIT")

    automatic_floor = spatial_policy.get("minimum_visible_pixels")
    configured_margin = spatial_policy.get("minimum_color_margin")
    if (
        isinstance(automatic_floor, bool)
        or not isinstance(automatic_floor, int)
        or automatic_floor < 1
        or isinstance(configured_margin, bool)
        or not isinstance(configured_margin, (int, float))
        or not math.isfinite(float(configured_margin))
    ):
        reasons.append("ANCHORED_SINGLE_SPATIAL_POLICY_INVALID")
        required_pixels = ANCHORED_SINGLE_MIN_PROJECTED_PIXELS
        required_margin = ANCHORED_SINGLE_MIN_COLOR_MARGIN
    else:
        required_pixels = max(
            automatic_floor,
            ANCHORED_SINGLE_MIN_PROJECTED_PIXELS,
        )
        required_margin = max(
            float(configured_margin),
            ANCHORED_SINGLE_MIN_COLOR_MARGIN,
        )

    supports: set[str] = set()
    conflicting_groups: set[str] = set()
    required_offsets = {(-2, 0), (2, 0), (0, -2), (0, 2)}
    for raw_observation in _sequence(
        spatial_part.get("observations", []),
        f"spatial part {part_id}.observations",
    ):
        observation = _mapping(raw_observation, f"spatial part {part_id}.observation")
        view_id = observation.get("reference_view_id")
        if not isinstance(view_id, str):
            continue
        canonical_group_id = observation.get("canonical_group_id")
        classification_resolved = observation.get("classification") == "resolved"
        if (
            classification_resolved
            and isinstance(canonical_group_id, str)
            and canonical_group_id
            and canonical_group_id != group_id
        ):
            conflicting_groups.add(canonical_group_id)

        declared_pixels = observation.get("declared_visible_pixels")
        projected_pixels = observation.get("projected_part_pixels")
        color_margin = observation.get("color_margin")
        group_scores = observation.get("group_scores")
        winner = (
            group_scores[0]
            if isinstance(group_scores, Sequence)
            and not isinstance(group_scores, (str, bytes))
            and group_scores
            and isinstance(group_scores[0], Mapping)
            else None
        )
        winner_share = winner.get("color_share") if winner is not None else None
        perturbations = observation.get("projection_perturbations")
        perturbation_offsets: set[tuple[int, int]] = set()
        perturbations_exact = (
            isinstance(perturbations, Sequence)
            and not isinstance(perturbations, (str, bytes))
            and len(perturbations) == len(required_offsets)
        )
        if perturbations_exact:
            for raw_perturbation in perturbations:
                if not isinstance(raw_perturbation, Mapping):
                    perturbations_exact = False
                    break
                offset = raw_perturbation.get("offset_pixels")
                sampled_pixels = raw_perturbation.get("sampled_reference_pixels")
                best_color_share = raw_perturbation.get("best_color_share")
                perturbation_margin = raw_perturbation.get("color_margin")
                if (
                    not isinstance(offset, Sequence)
                    or isinstance(offset, (str, bytes))
                    or len(offset) != 2
                    or any(
                        isinstance(value, bool) or not isinstance(value, int)
                        for value in offset
                    )
                    or raw_perturbation.get("canonical_group_id") != group_id
                    or raw_perturbation.get("diagnostic_canonical_group_id") != group_id
                    or isinstance(sampled_pixels, bool)
                    or not isinstance(sampled_pixels, int)
                    or sampled_pixels < required_pixels
                    or isinstance(best_color_share, bool)
                    or not isinstance(best_color_share, (int, float))
                    or not math.isfinite(float(best_color_share))
                    or not (
                        ANCHORED_SINGLE_MIN_PERTURBATION_COLOR_SHARE
                        <= float(best_color_share)
                        <= 1.0
                    )
                    or isinstance(perturbation_margin, bool)
                    or not isinstance(perturbation_margin, (int, float))
                    or not math.isfinite(float(perturbation_margin))
                    or float(perturbation_margin) < required_margin
                ):
                    perturbations_exact = False
                    break
                perturbation_offsets.add((int(offset[0]), int(offset[1])))
            perturbations_exact = (
                perturbations_exact and perturbation_offsets == required_offsets
            )

        exact_direct_projection = (
            classification_resolved
            and canonical_group_id == group_id
            and observation.get("registration_label_stable") is True
            and observation.get("perturbation_label_stable") is True
            and isinstance(declared_pixels, int)
            and not isinstance(declared_pixels, bool)
            and isinstance(projected_pixels, int)
            and not isinstance(projected_pixels, bool)
            and declared_pixels >= required_pixels
            and projected_pixels >= required_pixels
            and isinstance(color_margin, (int, float))
            and not isinstance(color_margin, bool)
            and math.isfinite(float(color_margin))
            and float(color_margin) >= required_margin
            and observation.get("bbox_canonical_group_id") == group_id
            and winner is not None
            and winner.get("canonical_group_id") == group_id
            and isinstance(winner_share, (int, float))
            and not isinstance(winner_share, bool)
            and math.isfinite(float(winner_share))
            and ANCHORED_SINGLE_MIN_COLOR_SHARE <= float(winner_share) <= 1.0
            and perturbations_exact
        )
        if view_id in deficit_views and exact_direct_projection:
            supports.add(view_id)

        raw_diagnostic = observation.get("small_part_diagnostic")
        if (
            isinstance(raw_diagnostic, Mapping)
            and raw_diagnostic.get("status") == "resolved"
            and isinstance(raw_diagnostic.get("canonical_group_id"), str)
            and raw_diagnostic.get("canonical_group_id") != group_id
        ):
            conflicting_groups.add(str(raw_diagnostic["canonical_group_id"]))
        canonical_diagnostic = observation.get("canonical_palette_diagnostic")
        if (
            isinstance(canonical_diagnostic, Mapping)
            and canonical_diagnostic.get("status") == "resolved"
            and canonical_diagnostic.get("reason_codes") == []
            and isinstance(canonical_diagnostic.get("canonical_group_id"), str)
            and canonical_diagnostic.get("canonical_group_id") != group_id
        ):
            conflicting_groups.add(str(canonical_diagnostic["canonical_group_id"]))

    explicitly_invisible_views = _explicitly_invisible_semantic_view_ids(
        spatial_part=spatial_part,
        spatial_policy=spatial_policy,
        part_id=part_id,
    )
    semantic_conflicts = {
        str(vote["view_id"])
        for raw_vote in _sequence(
            spatial_part.get("semantic_votes", []),
            f"spatial part {part_id}.semantic_votes",
        )
        if isinstance(raw_vote, Mapping)
        for vote in [raw_vote]
        if (
            vote.get("alignment_trusted") is True
            and vote.get("unique_canonical_join") is True
            and vote.get("status") == "matched"
            and isinstance(vote.get("canonical_group_id"), str)
            and vote.get("canonical_group_id") != group_id
            and isinstance(vote.get("effective_confidence"), (int, float))
            and not isinstance(vote.get("effective_confidence"), bool)
            and float(vote["effective_confidence"]) >= minimum_semantic_confidence
            and isinstance(vote.get("view_id"), str)
            and vote.get("view_id") not in explicitly_invisible_views
        )
    }

    semantic_override_candidates: list[tuple[list[str], list[str]]] = []
    for raw_support in _sequence(
        deficit.get("supporting_views"),
        f"deficit {group_id}.supporting_views",
    ):
        if not isinstance(raw_support, Mapping):
            continue
        sources = raw_support.get("deficit_sources")
        if (
            not isinstance(sources, Sequence)
            or isinstance(sources, (str, bytes))
            or "dominant_mass" not in sources
        ):
            continue
        override_deficit = copy.deepcopy(dict(deficit))
        override_deficit["single_view_spatial_repairable"] = True
        override_deficit["supporting_views"] = [copy.deepcopy(dict(raw_support))]
        (
            override_supports,
            override_reasons,
            override_views,
            _override_spatial_anchor_views,
        ) = _single_view_spatial_support_views(
            part_id=part_id,
            group_id=group_id,
            spatial_part=spatial_part,
            spatial_gate_decision=spatial_gate_decision,
            mapping_decision=mapping_decision,
            spatial_policy=spatial_policy,
            minimum_semantic_confidence=minimum_semantic_confidence,
            deficit=override_deficit,
            alignment_audits=alignment_audits,
            reference_evidence=reference_evidence,
            allow_deferred_semantic_anchor=True,
        )
        if not override_reasons and override_views:
            semantic_override_candidates.append((override_supports, override_views))
    semantic_override_applied = len(semantic_override_candidates) == 1
    semantic_override_views: list[str] = []
    if semantic_override_applied:
        override_supports, semantic_override_views = semantic_override_candidates[0]
        supports.update(override_supports)

    if conflicting_groups:
        reasons.append("ANCHORED_SINGLE_STABLE_SPATIAL_GROUP_CONFLICT")
    if semantic_conflicts and not semantic_override_applied:
        reasons.append("ANCHORED_SINGLE_MATCHED_SEMANTIC_GROUP_CONFLICT")
    if (
        spatial_gate_decision is not None
        and _has_conflict(spatial_gate_decision)
        and not semantic_override_applied
    ):
        reasons.append("SPATIAL_OR_SEMANTIC_CONFLICT")
    if mapping_decision is not None and (
        mapping_decision.get("output_status") == "matched"
        and mapping_decision.get("output_group_id") != group_id
    ):
        reasons.append("MATCHED_MAPPING_CONSENSUS_GROUP_CONFLICT")
    if len(supports) != 1:
        reasons.append("ANCHORED_SINGLE_REQUIRES_EXACTLY_ONE_DIRECT_SUPPORT")
    return (
        sorted(supports),
        sorted(set(reasons)),
        sorted(semantic_override_views),
    )


def _group_color_share(raw_scores: Any, group_id: str) -> float | None:
    if not isinstance(raw_scores, Sequence) or isinstance(raw_scores, (str, bytes)):
        return None
    total = 0.0
    for raw_score in raw_scores:
        if not isinstance(raw_score, Mapping):
            return None
        share = _optional_unit(raw_score.get("color_share"))
        if share is None:
            return None
        if raw_score.get("canonical_group_id") == group_id:
            total += share
    return total if total <= 1.0 + 1e-9 else None


def _repeated_geometry_registry_cohorts(
    registry: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Return complete, translation-invariant registry geometry cohorts."""

    registry_parts = _sequence(registry.get("parts"), "registry.parts")
    registry_count = len(registry_parts)
    by_signature: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, raw_part in enumerate(registry_parts):
        part = _mapping(raw_part, f"registry.parts[{index}]")
        part_id = part.get("part_id")
        point_count = part.get("point_count")
        face_count = part.get("face_count")
        bbox = part.get("world_bbox")
        if (
            not isinstance(part_id, str)
            or not part_id
            or not isinstance(point_count, int)
            or isinstance(point_count, bool)
            or point_count < 1
            or not isinstance(face_count, int)
            or isinstance(face_count, bool)
            or face_count < 1
            or not isinstance(bbox, Sequence)
            or isinstance(bbox, (str, bytes))
            or len(bbox) != 2
            or any(
                not isinstance(corner, Sequence)
                or isinstance(corner, (str, bytes))
                or len(corner) != 3
                for corner in bbox
            )
        ):
            continue
        coordinates = [value for corner in bbox for value in corner]
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in coordinates
        ):
            continue
        extents = sorted(
            round(abs(float(bbox[1][axis]) - float(bbox[0][axis])), 9)
            for axis in range(3)
        )
        if any(value <= 0.0 for value in extents):
            continue
        geometry_signature = {
            "point_count": point_count,
            "face_count": face_count,
            "sorted_bbox_extents": extents,
        }
        geometry_sha = _canonical_sha256(geometry_signature)
        properties = part.get("existing_visual_material_properties")
        if not isinstance(properties, Mapping):
            continue
        stable_properties = {
            str(key): copy.deepcopy(value)
            for key, value in properties.items()
            if str(key) != "shader_path"
        }
        if not stable_properties:
            continue
        try:
            properties_sha = _canonical_sha256(stable_properties)
        except QualityRepairError:
            continue
        by_signature[geometry_sha].append(
            {
                "part_id": part_id,
                "geometry_signature": geometry_signature,
                "geometry_signature_sha256": geometry_sha,
                "source_visual_stable_properties_signature_sha256": properties_sha,
            }
        )

    cohorts: list[dict[str, Any]] = []
    for geometry_sha, members in sorted(by_signature.items()):
        part_ids = sorted(str(member["part_id"]) for member in members)
        properties_hashes = {
            str(member["source_visual_stable_properties_signature_sha256"])
            for member in members
        }
        fraction = len(members) / registry_count if registry_count else 1.0
        if (
            not REPEATED_DARK_MIN_COHORT_SIZE
            <= len(members)
            <= REPEATED_DARK_MAX_COHORT_SIZE
            or fraction > REPEATED_DARK_MAX_REGISTRY_FRACTION
            or len(properties_hashes) != 1
        ):
            continue
        cohorts.append(
            {
                "geometry_signature": copy.deepcopy(members[0]["geometry_signature"]),
                "geometry_signature_sha256": geometry_sha,
                "source_visual_stable_properties_signature_sha256": next(
                    iter(properties_hashes)
                ),
                "registry_part_count": registry_count,
                "cohort_size": len(members),
                "registry_fraction": fraction,
                "cohort_part_ids": part_ids,
            }
        )
    return cohorts


def _source_visual_stable_properties_signature(
    part: Mapping[str, Any],
) -> str | None:
    properties = part.get("existing_visual_material_properties")
    if not isinstance(properties, Mapping):
        return None
    stable_properties = {
        str(key): copy.deepcopy(value)
        for key, value in properties.items()
        if str(key) != "shader_path"
    }
    if not stable_properties:
        return None
    try:
        return _canonical_sha256(stable_properties)
    except QualityRepairError:
        return None


def _dominant_assembly_strict_anchor_views(
    *,
    part_id: str,
    group_id: str,
    spatial_part: Mapping[str, Any],
    alignment_audits: Mapping[str, Mapping[str, Any]],
    reference_evidence: Mapping[str, Mapping[str, str]],
) -> tuple[list[str], list[dict[str, Any]], list[str]]:
    """Return only high-purity, perturbation-stable assembly anchor views."""

    required_offsets = {(-2, 0), (2, 0), (0, -2), (0, 2)}
    supporting_views: set[str] = set()
    evidence: list[dict[str, Any]] = []
    stable_alternative_group_ids: set[str] = set()
    for raw_observation in _sequence(
        spatial_part.get("observations", []),
        f"spatial part {part_id}.observations",
    ):
        observation = _mapping(
            raw_observation,
            f"spatial part {part_id}.observation",
        )
        view_id = observation.get("reference_view_id")
        canonical_group_id = observation.get("canonical_group_id")
        stable = (
            observation.get("classification") == "resolved"
            and observation.get("registration_label_stable") is True
            and observation.get("perturbation_label_stable") is True
        )
        if (
            stable
            and isinstance(canonical_group_id, str)
            and canonical_group_id != group_id
        ):
            stable_alternative_group_ids.add(canonical_group_id)
        if (
            not stable
            or canonical_group_id != group_id
            or not isinstance(view_id, str)
            or view_id not in reference_evidence
        ):
            continue

        alignment = alignment_audits.get(view_id)
        group_scores = observation.get("group_scores")
        bbox_scores = observation.get("bbox_group_scores")
        winner = (
            group_scores[0]
            if isinstance(group_scores, Sequence)
            and not isinstance(group_scores, (str, bytes))
            and group_scores
            and isinstance(group_scores[0], Mapping)
            else None
        )
        bbox_winner = (
            bbox_scores[0]
            if isinstance(bbox_scores, Sequence)
            and not isinstance(bbox_scores, (str, bytes))
            and bbox_scores
            and isinstance(bbox_scores[0], Mapping)
            else None
        )
        projected_pixels = observation.get("projected_part_pixels")
        direct_share = (
            _optional_unit(winner.get("color_share"))
            if isinstance(winner, Mapping)
            else None
        )
        direct_margin = _optional_unit(observation.get("color_margin"))
        bbox_share = (
            _optional_unit(bbox_winner.get("color_share"))
            if isinstance(bbox_winner, Mapping)
            else None
        )
        bbox_margin = _optional_unit(observation.get("bbox_color_margin"))
        alignment_score = (
            _optional_unit(alignment.get("score"))
            if isinstance(alignment, Mapping)
            else None
        )
        projection_iou = (
            _optional_unit(alignment.get("projection_iou"))
            if isinstance(alignment, Mapping)
            else None
        )
        ecc_correlation = (
            _optional_unit(alignment.get("ecc_correlation"))
            if isinstance(alignment, Mapping)
            else None
        )

        perturbation_evidence: list[dict[str, Any]] = []
        perturbation_offsets: set[tuple[int, int]] = set()
        raw_perturbations = observation.get("projection_perturbations")
        perturbations_valid = (
            isinstance(raw_perturbations, Sequence)
            and not isinstance(raw_perturbations, (str, bytes))
            and len(raw_perturbations) == len(required_offsets)
        )
        if perturbations_valid:
            for raw_perturbation in raw_perturbations:
                if not isinstance(raw_perturbation, Mapping):
                    perturbations_valid = False
                    break
                offset = raw_perturbation.get("offset_pixels")
                sampled_pixels = raw_perturbation.get("sampled_reference_pixels")
                share = _optional_unit(raw_perturbation.get("best_color_share"))
                margin = _optional_unit(raw_perturbation.get("color_margin"))
                if (
                    not isinstance(offset, Sequence)
                    or isinstance(offset, (str, bytes))
                    or len(offset) != 2
                    or any(
                        isinstance(value, bool) or not isinstance(value, int)
                        for value in offset
                    )
                    or raw_perturbation.get("canonical_group_id") != group_id
                    or raw_perturbation.get("diagnostic_canonical_group_id") != group_id
                    or isinstance(sampled_pixels, bool)
                    or not isinstance(sampled_pixels, int)
                    or sampled_pixels < DOMINANT_ASSEMBLY_MIN_PERTURBATION_PIXELS
                    or share is None
                    or share < DOMINANT_ASSEMBLY_MIN_PERTURBATION_COLOR_SHARE
                    or margin is None
                    or margin < DOMINANT_ASSEMBLY_MIN_PERTURBATION_COLOR_MARGIN
                ):
                    perturbations_valid = False
                    break
                normalized_offset = (int(offset[0]), int(offset[1]))
                perturbation_offsets.add(normalized_offset)
                perturbation_evidence.append(
                    {
                        "offset_pixels": list(normalized_offset),
                        "sampled_reference_pixels": sampled_pixels,
                        "target_color_share": share,
                        "target_color_margin": margin,
                    }
                )
            perturbations_valid = (
                perturbations_valid and perturbation_offsets == required_offsets
            )

        strict = (
            isinstance(alignment, Mapping)
            and alignment_score is not None
            and alignment_score >= DOMINANT_ASSEMBLY_MIN_ALIGNMENT_SCORE
            and projection_iou is not None
            and projection_iou >= DOMINANT_ASSEMBLY_MIN_PROJECTION_IOU
            and ecc_correlation is not None
            and ecc_correlation >= DOMINANT_ASSEMBLY_MIN_ECC_CORRELATION
            and alignment.get("ecc_status") == "success"
            and isinstance(projected_pixels, int)
            and not isinstance(projected_pixels, bool)
            and projected_pixels >= DOMINANT_ASSEMBLY_MIN_ANCHOR_PROJECTED_PIXELS
            and isinstance(winner, Mapping)
            and winner.get("canonical_group_id") == group_id
            and direct_share is not None
            and direct_share >= DOMINANT_ASSEMBLY_MIN_ANCHOR_COLOR_SHARE
            and direct_margin is not None
            and direct_margin >= DOMINANT_ASSEMBLY_MIN_ANCHOR_COLOR_MARGIN
            and isinstance(bbox_winner, Mapping)
            and bbox_winner.get("canonical_group_id") == group_id
            and observation.get("bbox_canonical_group_id") == group_id
            and perturbations_valid
        )
        if not strict:
            continue
        supporting_views.add(view_id)
        record = {
            "part_id": part_id,
            "reference_view_id": view_id,
            "canonical_group_id": group_id,
            "projected_part_pixels": projected_pixels,
            "direct_target_share": direct_share,
            "direct_target_margin": direct_margin,
            "bbox_target_share": bbox_share,
            "bbox_target_margin": bbox_margin,
            "alignment": {
                "score": alignment_score,
                "projection_iou": projection_iou,
                "ecc_correlation": ecc_correlation,
                "ecc_status": alignment["ecc_status"],
            },
            "projection_perturbations": sorted(
                perturbation_evidence,
                key=lambda item: item["offset_pixels"],
            ),
        }
        record["evidence_sha256"] = _canonical_sha256(record)
        evidence.append(record)

    reasons: list[str] = []
    if stable_alternative_group_ids:
        reasons.append("DOMINANT_ASSEMBLY_ANCHOR_STABLE_GROUP_CONFLICT")
    if len(supporting_views) < DOMINANT_ASSEMBLY_MIN_ANCHOR_VIEWS:
        reasons.append("DOMINANT_ASSEMBLY_ANCHOR_REQUIRES_TWO_STRICT_VIEWS")
    reasons.extend(
        _independent_support_reasons(
            view_ids=supporting_views,
            reference_evidence=reference_evidence,
        )
    )
    if reasons:
        return [], [], sorted(set(reasons))
    return (
        sorted(supporting_views),
        sorted(evidence, key=lambda item: item["reference_view_id"]),
        [],
    )


def _dominant_assembly_ancestor_paths(parent_path: str) -> list[str]:
    components = [component for component in parent_path.split("/") if component]
    return ["/" + "/".join(components[:end]) for end in range(1, len(components) + 1)]


def _dominant_assembly_child_branch(
    *, parent_path: str, assembly_path: str
) -> str | None:
    if parent_path == assembly_path:
        return "__assembly_self__"
    prefix = f"{assembly_path}/"
    if not parent_path.startswith(prefix):
        return None
    suffix = parent_path[len(prefix) :]
    return suffix.split("/", 1)[0] if suffix else "__assembly_self__"


def _dominant_assembly_member_veto(
    *,
    part_id: str,
    target_group_id: str,
    target_material_id: str,
    target_source_signature: str,
    registry_part: Mapping[str, Any],
    spatial_part: Mapping[str, Any],
    spatial_gate_decision: Mapping[str, Any] | None,
    mapping_decision: Mapping[str, Any] | None,
    geometry_risk: bool,
    baseline_assignment: Mapping[str, Any],
    existing_proposals: Sequence[Mapping[str, Any]],
) -> tuple[list[str], list[dict[str, Any]]]:
    """Return hard, non-semantic reasons that forbid cohort propagation."""

    reasons: list[str] = []
    evidence: list[dict[str, Any]] = []
    source_signature = _source_visual_stable_properties_signature(registry_part)
    if source_signature != target_source_signature:
        reasons.append("DOMINANT_ASSEMBLY_SOURCE_SIGNATURE_MISMATCH")
    if baseline_assignment.get("status") != POLICY_FALLBACK_STATUS:
        reasons.append("DOMINANT_ASSEMBLY_BASELINE_NOT_POLICY_FALLBACK")
    provenance = baseline_assignment.get("provenance")
    baseline_tier = provenance.get("tier") if isinstance(provenance, Mapping) else None
    if baseline_tier not in DOMINANT_ASSEMBLY_NEUTRAL_TIERS:
        reasons.append("DOMINANT_ASSEMBLY_BASELINE_NOT_SAFE_NEUTRAL")
    if (
        isinstance(provenance, Mapping)
        and provenance.get("canonical_group_id") is not None
    ):
        reasons.append(AUTHORITATIVE_CANONICAL_GROUP_LOCK_REASON)
    raw_face_subsets = baseline_assignment.get("face_subsets", [])
    if (
        not isinstance(raw_face_subsets, Sequence)
        or isinstance(raw_face_subsets, (str, bytes))
        or bool(raw_face_subsets)
    ):
        reasons.append("DOMINANT_ASSEMBLY_FACE_SUBSETS_FORBID_UNIFORM_REPAIR")
    if geometry_risk:
        reasons.append("DOMINANT_ASSEMBLY_MULTI_MATERIAL_GEOMETRY_RISK")
    if len(existing_proposals) > 1:
        reasons.append("DOMINANT_ASSEMBLY_EXISTING_PROPOSALS_AMBIGUOUS")
    for proposal in existing_proposals:
        if (
            proposal.get("canonical_group_id") != target_group_id
            or proposal.get("material_id") != target_material_id
        ):
            reasons.append("DOMINANT_ASSEMBLY_EXISTING_PROPOSAL_CONFLICT")

    for raw_observation in _sequence(
        spatial_part.get("observations", []),
        f"spatial part {part_id}.observations",
    ):
        observation = _mapping(
            raw_observation,
            f"spatial part {part_id}.observation",
        )
        view_id = observation.get("reference_view_id")
        observation_group_id = observation.get("canonical_group_id")
        if (
            observation.get("classification") == "resolved"
            and observation.get("registration_label_stable") is True
            and observation.get("perturbation_label_stable") is True
            and isinstance(observation_group_id, str)
            and observation_group_id != target_group_id
        ):
            reasons.append("DOMINANT_ASSEMBLY_STABLE_SPATIAL_ALTERNATIVE")
            evidence.append(
                {
                    "kind": "stable_spatial_alternative",
                    "reference_view_id": view_id,
                    "canonical_group_id": observation_group_id,
                }
            )

        group_scores = observation.get("group_scores")
        bbox_scores = observation.get("bbox_group_scores")
        winner = (
            group_scores[0]
            if isinstance(group_scores, Sequence)
            and not isinstance(group_scores, (str, bytes))
            and group_scores
            and isinstance(group_scores[0], Mapping)
            else None
        )
        bbox_winner = (
            bbox_scores[0]
            if isinstance(bbox_scores, Sequence)
            and not isinstance(bbox_scores, (str, bytes))
            and bbox_scores
            and isinstance(bbox_scores[0], Mapping)
            else None
        )
        projected_pixels = observation.get("projected_part_pixels")
        winner_group_id = (
            winner.get("canonical_group_id") if isinstance(winner, Mapping) else None
        )
        direct_share = (
            _optional_unit(winner.get("color_share"))
            if isinstance(winner, Mapping)
            else None
        )
        direct_margin = _optional_unit(observation.get("color_margin"))
        bbox_group_id = (
            bbox_winner.get("canonical_group_id")
            if isinstance(bbox_winner, Mapping)
            else None
        )
        if (
            isinstance(projected_pixels, int)
            and not isinstance(projected_pixels, bool)
            and projected_pixels >= DOMINANT_ASSEMBLY_ALTERNATIVE_MIN_PROJECTED_PIXELS
            and isinstance(winner_group_id, str)
            and winner_group_id != target_group_id
            and bbox_group_id == winner_group_id
            and observation.get("bbox_canonical_group_id") == winner_group_id
            and direct_share is not None
            and direct_share >= DOMINANT_ASSEMBLY_ALTERNATIVE_MIN_COLOR_SHARE
            and direct_margin is not None
            and direct_margin >= DOMINANT_ASSEMBLY_ALTERNATIVE_MIN_COLOR_MARGIN
        ):
            reasons.append("DOMINANT_ASSEMBLY_STRONG_DIRECT_ALTERNATIVE")
            evidence.append(
                {
                    "kind": "strong_direct_alternative",
                    "reference_view_id": view_id,
                    "canonical_group_id": winner_group_id,
                    "projected_part_pixels": projected_pixels,
                    "color_share": direct_share,
                    "color_margin": direct_margin,
                }
            )

        for diagnostic_field in (
            "small_part_diagnostic",
            "canonical_palette_diagnostic",
        ):
            diagnostic = observation.get(diagnostic_field)
            diagnostic_group_id = (
                diagnostic.get("canonical_group_id")
                if isinstance(diagnostic, Mapping)
                else None
            )
            if (
                isinstance(diagnostic, Mapping)
                and diagnostic.get("status") == "resolved"
                and diagnostic.get("reason_codes") == []
                and isinstance(diagnostic_group_id, str)
                and diagnostic_group_id != target_group_id
            ):
                reasons.append("DOMINANT_ASSEMBLY_RESOLVED_DIAGNOSTIC_ALTERNATIVE")
                evidence.append(
                    {
                        "kind": diagnostic_field,
                        "reference_view_id": view_id,
                        "canonical_group_id": diagnostic_group_id,
                    }
                )

    dark_consensus = spatial_part.get("multiview_dark_consensus")
    dark_group_id = (
        dark_consensus.get("canonical_group_id")
        if isinstance(dark_consensus, Mapping)
        else None
    )
    if (
        isinstance(dark_consensus, Mapping)
        and dark_consensus.get("status") == "resolved"
        and isinstance(dark_group_id, str)
        and dark_group_id != target_group_id
    ):
        reasons.append("DOMINANT_ASSEMBLY_MULTIVIEW_DARK_ALTERNATIVE")
        evidence.append(
            {
                "kind": "multiview_dark_consensus",
                "canonical_group_id": dark_group_id,
                "supporting_view_ids": sorted(
                    str(view_id)
                    for view_id in dark_consensus.get("supporting_view_ids", [])
                    if isinstance(view_id, str)
                ),
            }
        )

    for source, decision in (
        ("spatial_gate", spatial_gate_decision),
        ("mapping_consensus", mapping_decision),
    ):
        if (
            isinstance(decision, Mapping)
            and decision.get("output_status") == "matched"
            and isinstance(decision.get("output_group_id"), str)
            and decision.get("output_group_id") != target_group_id
        ):
            reasons.append("DOMINANT_ASSEMBLY_MATCHED_ALTERNATIVE_DECISION")
            evidence.append(
                {
                    "kind": source,
                    "canonical_group_id": decision["output_group_id"],
                }
            )
    return (
        sorted(set(reasons)),
        sorted(
            evidence,
            key=lambda item: (
                str(item.get("kind", "")),
                str(item.get("reference_view_id", "")),
                str(item.get("canonical_group_id", "")),
            ),
        ),
    )


def _dominant_assembly_group_gate(
    *,
    group_id: str,
    canonical_group: Mapping[str, Any],
    repairable_group: Mapping[str, Any] | None,
    dominant_residual_group: Mapping[str, Any] | None,
    reference_evidence: Mapping[str, Mapping[str, str]],
    material_id: str | None,
) -> list[str]:
    """Validate the target before assembly structure can add any members."""

    reasons: list[str] = []
    base_color = str(canonical_group.get("base_color", "")).strip().casefold()
    if base_color not in DOMINANT_ASSEMBLY_CHROMATIC_BASE_COLORS:
        reasons.append("DOMINANT_ASSEMBLY_GROUP_NOT_CHROMATIC")
    confidence = _optional_unit(canonical_group.get("confidence"))
    if confidence is None or confidence < DOMINANT_ASSEMBLY_MIN_GROUP_CONFIDENCE:
        reasons.append("DOMINANT_ASSEMBLY_GROUP_CONFIDENCE_BELOW_FLOOR")
    raw_source_view_ids = canonical_group.get("source_view_ids")
    source_view_ids = (
        {str(view_id) for view_id in raw_source_view_ids if isinstance(view_id, str)}
        if isinstance(raw_source_view_ids, Sequence)
        and not isinstance(raw_source_view_ids, (str, bytes))
        else set()
    )
    distinct_view_count = canonical_group.get("distinct_view_count")
    if (
        len(source_view_ids) < DOMINANT_ASSEMBLY_MIN_GROUP_SOURCE_VIEWS
        or isinstance(distinct_view_count, bool)
        or not isinstance(distinct_view_count, int)
        or distinct_view_count < DOMINANT_ASSEMBLY_MIN_GROUP_SOURCE_VIEWS
    ):
        reasons.append("DOMINANT_ASSEMBLY_GROUP_REQUIRES_THREE_SOURCE_VIEWS")
    if material_id is None:
        reasons.append("DOMINANT_ASSEMBLY_CONFIRMED_MATERIAL_MISSING")

    if (
        not isinstance(repairable_group, Mapping)
        or repairable_group.get("repairable") is not True
    ):
        reasons.append("DOMINANT_ASSEMBLY_MULTIVIEW_QA_DEFICIT_MISSING")
    else:
        qa_view_ids = {
            str(item["reference_view_id"])
            for item in repairable_group.get("supporting_views", [])
            if isinstance(item, Mapping)
            and isinstance(item.get("reference_view_id"), str)
            and item.get("reference_view_id") in reference_evidence
        }
        qa_content_ids = {
            reference_evidence[view_id]["content_cluster_id"] for view_id in qa_view_ids
        }
        qa_pose_ids = {
            reference_evidence[view_id]["pose_cluster_id"] for view_id in qa_view_ids
        }
        qa_raw_hashes = {
            reference_evidence[view_id]["raw_sha256"] for view_id in qa_view_ids
        }
        qa_normalized_hashes = {
            reference_evidence[view_id]["normalized_pixel_sha256"]
            for view_id in qa_view_ids
        }
        if (
            len(qa_view_ids) < DOMINANT_ASSEMBLY_MIN_QA_VIEWS
            or len(qa_content_ids) < DOMINANT_ASSEMBLY_MIN_QA_VIEWS
            or len(qa_pose_ids) < DOMINANT_ASSEMBLY_MIN_QA_VIEWS
            or len(qa_raw_hashes) < DOMINANT_ASSEMBLY_MIN_QA_VIEWS
            or len(qa_normalized_hashes) < DOMINANT_ASSEMBLY_MIN_QA_VIEWS
        ):
            reasons.append("DOMINANT_ASSEMBLY_QA_VIEWS_NOT_INDEPENDENT")

    residual_supports = (
        dominant_residual_group.get("supporting_views", [])
        if isinstance(dominant_residual_group, Mapping)
        else []
    )
    qualifying_residuals: list[Mapping[str, Any]] = []
    if isinstance(residual_supports, Sequence) and not isinstance(
        residual_supports, (str, bytes)
    ):
        for raw_support in residual_supports:
            if not isinstance(raw_support, Mapping):
                continue
            reference_share = _optional_unit(raw_support.get("reference_share"))
            reference_share_margin = raw_support.get("reference_share_margin")
            deficit_share = _optional_unit(raw_support.get("deficit_share"))
            mass_recall = _optional_unit(raw_support.get("mass_recall"))
            if (
                reference_share is not None
                and reference_share >= DOMINANT_ASSEMBLY_MIN_REFERENCE_SHARE
                and isinstance(reference_share_margin, (int, float))
                and not isinstance(reference_share_margin, bool)
                and math.isfinite(float(reference_share_margin))
                and float(reference_share_margin)
                >= DOMINANT_ASSEMBLY_MIN_REFERENCE_SHARE_MARGIN
                and deficit_share is not None
                and deficit_share >= DOMINANT_ASSEMBLY_MIN_DEFICIT_SHARE
                and mass_recall is not None
                and mass_recall <= DOMINANT_ASSEMBLY_MAX_MASS_RECALL
                and raw_support.get("requires_strict_local_projection") is True
                and DOMINANT_RESIDUAL_DEFICIT_SOURCE
                in raw_support.get("deficit_sources", [])
            ):
                qualifying_residuals.append(raw_support)
    if len(qualifying_residuals) != 1:
        reasons.append("DOMINANT_ASSEMBLY_DOMINANT_QA_DEFICIT_MISSING")
    return sorted(set(reasons))


def _dominant_assembly_cohort_expansions(
    *,
    canonical_groups: Mapping[str, Mapping[str, Any]],
    repairable_groups: Mapping[str, Mapping[str, Any]],
    dominant_residual_groups: Mapping[str, Mapping[str, Any]],
    registry_by_part: Mapping[str, Mapping[str, Any]],
    spatial_parts: Mapping[str, Mapping[str, Any]],
    alignment_audits: Mapping[str, Mapping[str, Any]],
    reference_evidence: Mapping[str, Mapping[str, str]],
    gate_decisions: Mapping[str, Mapping[str, Any]],
    mapping_decisions: Mapping[str, Mapping[str, Any]],
    geometry_risks: Mapping[str, bool],
    baseline_by_part: Mapping[str, Mapping[str, Any]],
    occupied_proposals_by_part: Mapping[str, Sequence[Mapping[str, Any]]],
    confirmed_materials: Mapping[str, str],
    provisional_material_groups: set[str],
    input_hashes: Mapping[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Expand strict spatial anchors through one dominant authored assembly.

    Qwen part votes are deliberately absent from this contract.  They can
    neither create an anchor nor authorize a propagated member.  Hash-bound
    palette/QA evidence chooses the group, strict projections anchor it, and
    source appearance plus authored hierarchy define the cohort.
    """

    registry_count = len(registry_by_part)
    if not registry_count:
        return [], [], []
    maximum_subtree_size = max(
        DOMINANT_ASSEMBLY_MIN_COHORT_SIZE,
        min(
            DOMINANT_ASSEMBLY_MAX_COHORT_SIZE,
            math.floor(DOMINANT_ASSEMBLY_MAX_REGISTRY_FRACTION * registry_count),
        ),
    )
    source_signatures = {
        part_id: _source_visual_stable_properties_signature(part)
        for part_id, part in registry_by_part.items()
    }
    parent_paths: dict[str, str] = {}
    face_counts: dict[str, int] = {}
    for part_id, part in registry_by_part.items():
        parent_path = part.get("parent_path")
        face_count = part.get("face_count")
        if isinstance(parent_path, str) and parent_path.startswith("/"):
            parent_paths[part_id] = parent_path.rstrip("/") or "/"
        if (
            isinstance(face_count, int)
            and not isinstance(face_count, bool)
            and face_count > 0
        ):
            face_counts[part_id] = face_count

    strict_anchors_by_group: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    observed_group_ids_by_part: dict[str, set[str]] = defaultdict(set)
    for part_id, spatial_part in spatial_parts.items():
        for raw_observation in spatial_part.get("observations", []):
            if not isinstance(raw_observation, Mapping):
                continue
            observed_group_id = raw_observation.get("canonical_group_id")
            if isinstance(observed_group_id, str):
                observed_group_ids_by_part[part_id].add(observed_group_id)
    for part_id, observed_group_ids in sorted(observed_group_ids_by_part.items()):
        spatial_part = spatial_parts.get(part_id)
        if not isinstance(spatial_part, Mapping):
            continue
        for group_id in sorted(observed_group_ids):
            views, evidence, reasons = _dominant_assembly_strict_anchor_views(
                part_id=part_id,
                group_id=group_id,
                spatial_part=spatial_part,
                alignment_audits=alignment_audits,
                reference_evidence=reference_evidence,
            )
            if not reasons:
                strict_anchors_by_group[group_id][part_id] = {
                    "part_id": part_id,
                    "supporting_view_ids": views,
                    "evidence": evidence,
                }

    structural_candidates: list[dict[str, Any]] = []
    for group_id, canonical_group in sorted(canonical_groups.items()):
        material_id = confirmed_materials.get(group_id)
        group_reasons = _dominant_assembly_group_gate(
            group_id=group_id,
            canonical_group=canonical_group,
            repairable_group=repairable_groups.get(group_id),
            dominant_residual_group=dominant_residual_groups.get(group_id),
            reference_evidence=reference_evidence,
            material_id=material_id,
        )
        if group_reasons:
            continue
        assert material_id is not None
        anchors_by_signature: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for anchor_part_id, anchor in sorted(
            strict_anchors_by_group.get(group_id, {}).items()
        ):
            signature = source_signatures.get(anchor_part_id)
            if signature is None or anchor_part_id not in parent_paths:
                continue
            veto_reasons, _ = _dominant_assembly_member_veto(
                part_id=anchor_part_id,
                target_group_id=group_id,
                target_material_id=material_id,
                target_source_signature=signature,
                registry_part=registry_by_part[anchor_part_id],
                spatial_part=spatial_parts[anchor_part_id],
                spatial_gate_decision=gate_decisions.get(anchor_part_id),
                mapping_decision=mapping_decisions.get(anchor_part_id),
                geometry_risk=geometry_risks.get(anchor_part_id, True),
                baseline_assignment=baseline_by_part[anchor_part_id],
                existing_proposals=occupied_proposals_by_part.get(anchor_part_id, []),
            )
            if not veto_reasons:
                anchors_by_signature[signature].append(anchor)

        for signature, signature_anchors in sorted(anchors_by_signature.items()):
            ancestor_anchor_ids: dict[str, set[str]] = defaultdict(set)
            for anchor in signature_anchors:
                anchor_part_id = str(anchor["part_id"])
                for ancestor_path in _dominant_assembly_ancestor_paths(
                    parent_paths[anchor_part_id]
                ):
                    ancestor_anchor_ids[ancestor_path].add(anchor_part_id)
            for assembly_path, raw_anchor_part_ids in sorted(
                ancestor_anchor_ids.items()
            ):
                anchor_part_ids = sorted(raw_anchor_part_ids)
                if len(anchor_part_ids) < DOMINANT_ASSEMBLY_MIN_ANCHOR_PARTS:
                    continue
                subtree_part_ids = sorted(
                    part_id
                    for part_id, parent_path in parent_paths.items()
                    if (
                        parent_path == assembly_path
                        or parent_path.startswith(f"{assembly_path}/")
                    )
                )
                if not (
                    DOMINANT_ASSEMBLY_MIN_COHORT_SIZE
                    <= len(subtree_part_ids)
                    <= maximum_subtree_size
                ):
                    continue
                if any(part_id not in face_counts for part_id in subtree_part_ids):
                    continue
                child_branches = sorted(
                    {
                        branch
                        for part_id in anchor_part_ids
                        for branch in [
                            _dominant_assembly_child_branch(
                                parent_path=parent_paths[part_id],
                                assembly_path=assembly_path,
                            )
                        ]
                        if branch is not None
                    }
                )
                if len(child_branches) < DOMINANT_ASSEMBLY_MIN_ANCHOR_CHILD_BRANCHES:
                    continue
                anchor_view_ids = {
                    str(view_id)
                    for part_id in anchor_part_ids
                    for view_id in strict_anchors_by_group[group_id][part_id][
                        "supporting_view_ids"
                    ]
                }
                anchor_content_ids = {
                    reference_evidence[view_id]["content_cluster_id"]
                    for view_id in anchor_view_ids
                    if view_id in reference_evidence
                }
                anchor_pose_ids = {
                    reference_evidence[view_id]["pose_cluster_id"]
                    for view_id in anchor_view_ids
                    if view_id in reference_evidence
                }
                anchor_raw_hashes = {
                    reference_evidence[view_id]["raw_sha256"]
                    for view_id in anchor_view_ids
                    if view_id in reference_evidence
                }
                if (
                    len(anchor_view_ids) < DOMINANT_ASSEMBLY_MIN_QA_VIEWS
                    or len(anchor_content_ids) < DOMINANT_ASSEMBLY_MIN_QA_VIEWS
                    or len(anchor_pose_ids) < DOMINANT_ASSEMBLY_MIN_QA_VIEWS
                    or len(anchor_raw_hashes) < DOMINANT_ASSEMBLY_MIN_QA_VIEWS
                ):
                    continue

                signature_part_ids = sorted(
                    part_id
                    for part_id in subtree_part_ids
                    if source_signatures.get(part_id) == signature
                )
                subtree_face_count = sum(
                    face_counts[part_id] for part_id in subtree_part_ids
                )
                signature_face_count = sum(
                    face_counts[part_id] for part_id in signature_part_ids
                )
                signature_part_share = len(signature_part_ids) / len(subtree_part_ids)
                signature_face_share = (
                    signature_face_count / subtree_face_count
                    if subtree_face_count
                    else 0.0
                )
                if (
                    signature_part_share < DOMINANT_ASSEMBLY_MIN_SIGNATURE_PART_SHARE
                    or signature_face_share < DOMINANT_ASSEMBLY_MIN_SIGNATURE_FACE_SHARE
                ):
                    continue
                competing_anchors = sorted(
                    {
                        competitor_part_id
                        for competitor_group_id, competitor_anchors in (
                            strict_anchors_by_group.items()
                        )
                        if competitor_group_id != group_id
                        for competitor_part_id in competitor_anchors
                        if (
                            competitor_part_id in subtree_part_ids
                            and source_signatures.get(competitor_part_id) == signature
                        )
                    }
                )
                if competing_anchors:
                    continue
                structural_candidates.append(
                    {
                        "canonical_group_id": group_id,
                        "material_id": material_id,
                        "provisional_material_candidate": (
                            group_id in provisional_material_groups
                        ),
                        "assembly_path": assembly_path,
                        "source_visual_stable_properties_signature_sha256": (signature),
                        "anchor_part_ids": anchor_part_ids,
                        "anchor_supporting_view_ids": sorted(anchor_view_ids),
                        "anchor_child_branches": child_branches,
                        "subtree_part_ids": subtree_part_ids,
                        "signature_part_ids": signature_part_ids,
                        "subtree_face_count": subtree_face_count,
                        "signature_face_count": signature_face_count,
                        "signature_part_share": signature_part_share,
                        "signature_face_share": signature_face_share,
                    }
                )

    # A valid descendant is more specific than its valid ancestor.  Keep all
    # disjoint deepest assemblies, but never allow overlapping contracts.
    deepest_candidates = [
        candidate
        for candidate in structural_candidates
        if not any(
            other is not candidate
            and other["canonical_group_id"] == candidate["canonical_group_id"]
            and other["source_visual_stable_properties_signature_sha256"]
            == candidate["source_visual_stable_properties_signature_sha256"]
            and str(other["assembly_path"]).startswith(f"{candidate['assembly_path']}/")
            for other in structural_candidates
        )
    ]
    accepted_records: list[dict[str, Any]] = []
    cohort_audits: list[dict[str, Any]] = []
    member_skips: list[dict[str, Any]] = []
    claimed_part_ids: set[str] = set()
    for candidate in sorted(
        deepest_candidates,
        key=lambda item: (
            -str(item["assembly_path"]).count("/"),
            str(item["canonical_group_id"]),
            str(item["assembly_path"]),
        ),
    ):
        group_id = str(candidate["canonical_group_id"])
        material_id = str(candidate["material_id"])
        signature = str(candidate["source_visual_stable_properties_signature_sha256"])
        excluded_members: list[dict[str, Any]] = []
        accepted_part_ids: list[str] = []
        veto_part_ids: list[str] = []
        veto_face_count = 0
        for part_id in candidate["subtree_part_ids"]:
            veto_reasons, veto_evidence = _dominant_assembly_member_veto(
                part_id=part_id,
                target_group_id=group_id,
                target_material_id=material_id,
                target_source_signature=signature,
                registry_part=registry_by_part[part_id],
                spatial_part=spatial_parts[part_id],
                spatial_gate_decision=gate_decisions.get(part_id),
                mapping_decision=mapping_decisions.get(part_id),
                geometry_risk=geometry_risks.get(part_id, True),
                baseline_assignment=baseline_by_part[part_id],
                existing_proposals=occupied_proposals_by_part.get(part_id, []),
            )
            if veto_reasons:
                excluded_members.append(
                    {
                        "part_id": part_id,
                        "face_count": face_counts[part_id],
                        "reason_codes": veto_reasons,
                        "alternative_evidence": veto_evidence,
                    }
                )
                if source_signatures.get(part_id) == signature:
                    veto_part_ids.append(part_id)
                    veto_face_count += face_counts[part_id]
                continue
            accepted_part_ids.append(part_id)

        signature_part_count = len(candidate["signature_part_ids"])
        veto_part_share = (
            len(veto_part_ids) / signature_part_count if signature_part_count else 1.0
        )
        veto_face_share = (
            veto_face_count / int(candidate["signature_face_count"])
            if candidate["signature_face_count"]
            else 1.0
        )
        if (
            len(accepted_part_ids) < DOMINANT_ASSEMBLY_MIN_COHORT_SIZE
            or not set(candidate["anchor_part_ids"]) <= set(accepted_part_ids)
            or veto_part_share > DOMINANT_ASSEMBLY_MAX_VETO_PART_SHARE
            or veto_face_share > DOMINANT_ASSEMBLY_MAX_VETO_FACE_SHARE
            or claimed_part_ids & set(accepted_part_ids)
        ):
            continue

        expanded_member_part_ids = sorted(
            set(accepted_part_ids) - set(candidate["anchor_part_ids"])
        )
        identity_payload = {
            "schema_version": "qwen-dominant-assembly-cohort/v1",
            "candidate_kind": "dominant_assembly",
            "canonical_group_id": group_id,
            "assembly_path": candidate["assembly_path"],
            "source_visual_stable_properties_signature_sha256": signature,
            "anchor_part_ids": candidate["anchor_part_ids"],
            "cohort_part_ids": accepted_part_ids,
            "input_hashes": dict(sorted(input_hashes.items())),
        }
        cohort_id = _canonical_sha256(identity_payload)
        anchor_evidence = sorted(
            (
                copy.deepcopy(evidence)
                for anchor_part_id in candidate["anchor_part_ids"]
                for evidence in strict_anchors_by_group[group_id][anchor_part_id][
                    "evidence"
                ]
            ),
            key=lambda item: (
                item["part_id"],
                item["reference_view_id"],
            ),
        )
        contract: dict[str, Any] = {
            "schema_version": "qwen-dominant-assembly-cohort/v1",
            "candidate_kind": "dominant_assembly",
            "cohort_id": cohort_id,
            "canonical_group_id": group_id,
            "material_id": material_id,
            "material_selection_status": (
                "PROVISIONAL_PENDING_EXACT_MDL_TOURNAMENT"
                if candidate["provisional_material_candidate"]
                else "CONFIRMED"
            ),
            "assembly_path": candidate["assembly_path"],
            "source_visual_stable_properties_signature_sha256": signature,
            "membership_status": DOMINANT_ASSEMBLY_MEMBERSHIP_STATUS,
            "anchor_part_ids": candidate["anchor_part_ids"],
            "anchor_supporting_view_ids": candidate["anchor_supporting_view_ids"],
            "anchor_child_branches": candidate["anchor_child_branches"],
            "subtree_part_ids": candidate["subtree_part_ids"],
            "candidate_signature_part_ids": candidate["signature_part_ids"],
            "accepted_part_ids": accepted_part_ids,
            "cohort_part_ids": accepted_part_ids,
            "expanded_member_part_ids": expanded_member_part_ids,
            "excluded_members": sorted(
                excluded_members, key=lambda item: item["part_id"]
            ),
            "anchor_evidence": anchor_evidence,
            "signature_dominance": {
                "subtree_part_count": len(candidate["subtree_part_ids"]),
                "signature_part_count": len(candidate["signature_part_ids"]),
                "part_share": candidate["signature_part_share"],
                "subtree_face_count": candidate["subtree_face_count"],
                "signature_face_count": candidate["signature_face_count"],
                "face_share": candidate["signature_face_share"],
            },
            "veto_budget": {
                "veto_part_ids": sorted(veto_part_ids),
                "veto_part_count": len(veto_part_ids),
                "veto_part_share": veto_part_share,
                "veto_face_count": veto_face_count,
                "veto_face_share": veto_face_share,
            },
            "thresholds": {
                "minimum_group_confidence": (DOMINANT_ASSEMBLY_MIN_GROUP_CONFIDENCE),
                "minimum_group_source_views": (
                    DOMINANT_ASSEMBLY_MIN_GROUP_SOURCE_VIEWS
                ),
                "minimum_qa_views": DOMINANT_ASSEMBLY_MIN_QA_VIEWS,
                "minimum_reference_share": (DOMINANT_ASSEMBLY_MIN_REFERENCE_SHARE),
                "minimum_reference_share_margin": (
                    DOMINANT_ASSEMBLY_MIN_REFERENCE_SHARE_MARGIN
                ),
                "minimum_deficit_share": DOMINANT_ASSEMBLY_MIN_DEFICIT_SHARE,
                "maximum_mass_recall": DOMINANT_ASSEMBLY_MAX_MASS_RECALL,
                "minimum_anchor_parts": DOMINANT_ASSEMBLY_MIN_ANCHOR_PARTS,
                "minimum_anchor_views": DOMINANT_ASSEMBLY_MIN_ANCHOR_VIEWS,
                "minimum_anchor_child_branches": (
                    DOMINANT_ASSEMBLY_MIN_ANCHOR_CHILD_BRANCHES
                ),
                "minimum_signature_part_share": (
                    DOMINANT_ASSEMBLY_MIN_SIGNATURE_PART_SHARE
                ),
                "minimum_signature_face_share": (
                    DOMINANT_ASSEMBLY_MIN_SIGNATURE_FACE_SHARE
                ),
                "maximum_veto_part_share": (DOMINANT_ASSEMBLY_MAX_VETO_PART_SHARE),
                "maximum_veto_face_share": (DOMINANT_ASSEMBLY_MAX_VETO_FACE_SHARE),
            },
            "input_hashes": dict(sorted(input_hashes.items())),
        }
        contract["contract_sha256"] = _canonical_sha256(contract)
        cohort_audits.append(copy.deepcopy(contract))
        claimed_part_ids.update(accepted_part_ids)
        for excluded in excluded_members:
            member_skips.append(
                {
                    "part_id": excluded["part_id"],
                    "canonical_group_id": group_id,
                    "reason_codes": excluded["reason_codes"],
                }
            )
        for part_id in accepted_part_ids:
            accepted_records.append(
                {
                    "part_id": part_id,
                    "canonical_group_id": group_id,
                    "material_id": material_id,
                    "provisional_material_candidate": bool(
                        candidate["provisional_material_candidate"]
                    ),
                    "localization_lane": (DOMINANT_ASSEMBLY_COHORT_EXPANSION_LANE),
                    "member_role": (
                        "strict_spatial_anchor"
                        if part_id in candidate["anchor_part_ids"]
                        else "expanded_member"
                    ),
                    "contract": copy.deepcopy(contract),
                    "baseline_material_id": str(
                        baseline_by_part[part_id]["material_id"]
                    ),
                }
            )
    return (
        sorted(accepted_records, key=lambda item: item["part_id"]),
        sorted(cohort_audits, key=lambda item: item["cohort_id"]),
        sorted(
            member_skips,
            key=lambda item: (
                item["part_id"],
                item["canonical_group_id"],
            ),
        ),
    )


def _bounded_signature_group_gate(
    *,
    canonical_group: Mapping[str, Any],
    repairable_group: Mapping[str, Any] | None,
    reference_evidence: Mapping[str, Mapping[str, str]],
    material_id: str | None,
    provisional_material: bool,
) -> list[str]:
    """Validate a multi-view missing chromatic group for one atomic pair."""

    reasons: list[str] = []
    base_color = str(canonical_group.get("base_color", "")).strip().casefold()
    if base_color not in DOMINANT_ASSEMBLY_CHROMATIC_BASE_COLORS:
        reasons.append("BOUNDED_SIGNATURE_GROUP_NOT_CHROMATIC")
    confidence = _optional_unit(canonical_group.get("confidence"))
    if confidence is None or confidence < BOUNDED_SIGNATURE_MIN_GROUP_CONFIDENCE:
        reasons.append("BOUNDED_SIGNATURE_GROUP_CONFIDENCE_BELOW_FLOOR")
    distinct_view_count = canonical_group.get("distinct_view_count")
    raw_source_view_ids = canonical_group.get("source_view_ids")
    source_view_ids = (
        {str(view_id) for view_id in raw_source_view_ids if isinstance(view_id, str)}
        if isinstance(raw_source_view_ids, Sequence)
        and not isinstance(raw_source_view_ids, (str, bytes))
        else set()
    )
    if (
        canonical_group.get("singleton") is True
        or isinstance(distinct_view_count, bool)
        or not isinstance(distinct_view_count, int)
        or distinct_view_count < BOUNDED_SIGNATURE_MIN_GROUP_SOURCE_VIEWS
        or len(source_view_ids) < BOUNDED_SIGNATURE_MIN_GROUP_SOURCE_VIEWS
    ):
        reasons.append("BOUNDED_SIGNATURE_GROUP_NOT_MULTIVIEW")
    if material_id is None:
        reasons.append("BOUNDED_SIGNATURE_EXACT_MDL_MISSING")
    if provisional_material:
        reasons.append("BOUNDED_SIGNATURE_REQUIRES_CONFIRMED_EXACT_MDL")
    if (
        not isinstance(repairable_group, Mapping)
        or repairable_group.get("repairable") is not True
    ):
        reasons.append("BOUNDED_SIGNATURE_MULTIVIEW_QA_DEFICIT_MISSING")
    else:
        qa_view_ids = {
            str(item["reference_view_id"])
            for item in repairable_group.get("supporting_views", [])
            if isinstance(item, Mapping)
            and isinstance(item.get("reference_view_id"), str)
            and item.get("reference_view_id") in reference_evidence
        }
        qa_content_ids = {
            reference_evidence[view_id]["content_cluster_id"] for view_id in qa_view_ids
        }
        qa_pose_ids = {
            reference_evidence[view_id]["pose_cluster_id"] for view_id in qa_view_ids
        }
        qa_raw_hashes = {
            reference_evidence[view_id]["raw_sha256"] for view_id in qa_view_ids
        }
        qa_normalized_hashes = {
            reference_evidence[view_id]["normalized_pixel_sha256"]
            for view_id in qa_view_ids
        }
        if (
            len(qa_view_ids) < BOUNDED_SIGNATURE_MIN_QA_VIEWS
            or len(qa_content_ids) < BOUNDED_SIGNATURE_MIN_QA_VIEWS
            or len(qa_pose_ids) < BOUNDED_SIGNATURE_MIN_QA_VIEWS
            or len(qa_raw_hashes) < BOUNDED_SIGNATURE_MIN_QA_VIEWS
            or len(qa_normalized_hashes) < BOUNDED_SIGNATURE_MIN_QA_VIEWS
        ):
            reasons.append("BOUNDED_SIGNATURE_QA_VIEWS_NOT_INDEPENDENT")
    return sorted(set(reasons))


def _bounded_signature_exact_anchor_evidence(
    *,
    part_id: str,
    group_id: str,
    spatial_part: Mapping[str, Any],
    spatial_gate_decision: Mapping[str, Any] | None,
    mapping_decision: Mapping[str, Any] | None,
    spatial_policy: Mapping[str, Any],
    minimum_semantic_confidence: float,
    deficit: Mapping[str, Any],
    reference_evidence: Mapping[str, Mapping[str, str]],
) -> tuple[dict[str, Any] | None, list[str]]:
    """Return one exact six-sample diagnostic with no alternative label."""

    support_views, reasons = _source_identity_diagnostic_support_views(
        part_id=part_id,
        group_id=group_id,
        spatial_part=spatial_part,
        spatial_gate_decision=spatial_gate_decision,
        mapping_decision=mapping_decision,
        spatial_policy=spatial_policy,
        minimum_semantic_confidence=minimum_semantic_confidence,
        deficit=deficit,
        reference_evidence=reference_evidence,
    )
    if reasons or len(support_views) != 1:
        return None, reasons
    view_id = support_views[0]
    observations = [
        observation
        for raw_observation in spatial_part.get("observations", [])
        if isinstance(raw_observation, Mapping)
        for observation in [raw_observation]
        if observation.get("reference_view_id") == view_id
    ]
    if len(observations) != 1:
        return None, ["BOUNDED_SIGNATURE_ANCHOR_OBSERVATION_NOT_UNIQUE"]
    observation = observations[0]
    diagnostic = observation.get("small_part_diagnostic")
    group_scores = observation.get("group_scores")
    bbox_scores = observation.get("bbox_group_scores")
    winner = (
        group_scores[0]
        if isinstance(group_scores, Sequence)
        and not isinstance(group_scores, (str, bytes))
        and group_scores
        and isinstance(group_scores[0], Mapping)
        else None
    )
    bbox_winner = (
        bbox_scores[0]
        if isinstance(bbox_scores, Sequence)
        and not isinstance(bbox_scores, (str, bytes))
        and bbox_scores
        and isinstance(bbox_scores[0], Mapping)
        else None
    )
    direct_share = (
        _optional_unit(winner.get("color_share"))
        if isinstance(winner, Mapping)
        else None
    )
    direct_margin = _optional_unit(observation.get("color_margin"))
    bbox_share = (
        _optional_unit(bbox_winner.get("color_share"))
        if isinstance(bbox_winner, Mapping)
        else None
    )
    bbox_margin = _optional_unit(observation.get("bbox_color_margin"))
    projected_pixels = observation.get("projected_part_pixels")
    diagnostic_floor = _observation_diagnostic_floor(observation, spatial_policy)
    required_offsets = {(-2, 0), (2, 0), (0, -2), (0, 2)}
    perturbations = observation.get("projection_perturbations")
    perturbation_evidence: list[dict[str, Any]] = []
    perturbation_offsets: set[tuple[int, int]] = set()
    perturbations_valid = (
        isinstance(perturbations, Sequence)
        and not isinstance(perturbations, (str, bytes))
        and len(perturbations) == len(required_offsets)
    )
    if perturbations_valid:
        for raw_perturbation in perturbations:
            if not isinstance(raw_perturbation, Mapping):
                perturbations_valid = False
                break
            offset = raw_perturbation.get("offset_pixels")
            sampled_pixels = raw_perturbation.get("sampled_reference_pixels")
            share = _optional_unit(raw_perturbation.get("best_color_share"))
            margin = _optional_unit(raw_perturbation.get("color_margin"))
            if (
                not isinstance(offset, Sequence)
                or isinstance(offset, (str, bytes))
                or len(offset) != 2
                or any(
                    isinstance(value, bool) or not isinstance(value, int)
                    for value in offset
                )
                or raw_perturbation.get("canonical_group_id") != group_id
                or raw_perturbation.get("diagnostic_canonical_group_id") != group_id
                or isinstance(sampled_pixels, bool)
                or not isinstance(sampled_pixels, int)
                or not isinstance(diagnostic_floor, int)
                or sampled_pixels < diagnostic_floor
                or share is None
                or share < BOUNDED_SIGNATURE_MIN_ANCHOR_PERTURBATION_SHARE
                or margin is None
                or margin < BOUNDED_SIGNATURE_MIN_ANCHOR_PERTURBATION_MARGIN
            ):
                perturbations_valid = False
                break
            normalized_offset = (int(offset[0]), int(offset[1]))
            perturbation_offsets.add(normalized_offset)
            perturbation_evidence.append(
                {
                    "offset_pixels": list(normalized_offset),
                    "sampled_reference_pixels": sampled_pixels,
                    "target_color_share": share,
                    "target_color_margin": margin,
                }
            )
        perturbations_valid = (
            perturbations_valid and perturbation_offsets == required_offsets
        )
    strict = (
        isinstance(diagnostic, Mapping)
        and diagnostic.get("status") == "resolved"
        and diagnostic.get("reason_codes") == []
        and diagnostic.get("canonical_group_id") == group_id
        and diagnostic.get("bbox_canonical_group_id") == group_id
        and diagnostic.get("registration_label_stable") is True
        and diagnostic.get("resolved_sample_count")
        == BOUNDED_SIGNATURE_REQUIRED_DIAGNOSTIC_SAMPLE_COUNT
        and diagnostic.get("target_sample_count")
        == BOUNDED_SIGNATURE_REQUIRED_DIAGNOSTIC_SAMPLE_COUNT
        and diagnostic.get("consensus_ratio") == 1.0
        and diagnostic.get("alternative_canonical_group_ids") == []
        and observation.get("registration_label_stable") is True
        and observation.get("perturbation_label_stable") is True
        and isinstance(projected_pixels, int)
        and not isinstance(projected_pixels, bool)
        and isinstance(diagnostic_floor, int)
        and projected_pixels >= diagnostic_floor
        and isinstance(winner, Mapping)
        and winner.get("canonical_group_id") == group_id
        and direct_share is not None
        and direct_share >= BOUNDED_SIGNATURE_MIN_ANCHOR_DIRECT_COLOR_SHARE
        and direct_margin is not None
        and direct_margin >= BOUNDED_SIGNATURE_MIN_ANCHOR_DIRECT_COLOR_MARGIN
        and isinstance(bbox_winner, Mapping)
        and bbox_winner.get("canonical_group_id") == group_id
        and observation.get("bbox_canonical_group_id") == group_id
        and bbox_share is not None
        and bbox_share >= BOUNDED_SIGNATURE_MIN_ANCHOR_BBOX_COLOR_SHARE
        and bbox_margin is not None
        and bbox_margin >= BOUNDED_SIGNATURE_MIN_ANCHOR_BBOX_COLOR_MARGIN
        and perturbations_valid
    )
    if not strict:
        return None, ["BOUNDED_SIGNATURE_STRICT_ANCHOR_CONTRACT_FAILED"]
    record = {
        "part_id": part_id,
        "reference_view_id": view_id,
        "canonical_group_id": group_id,
        "projected_part_pixels": projected_pixels,
        "diagnostic_floor_pixels": diagnostic_floor,
        "direct_target_share": direct_share,
        "direct_target_margin": direct_margin,
        "bbox_target_share": bbox_share,
        "bbox_target_margin": bbox_margin,
        "resolved_sample_count": diagnostic["resolved_sample_count"],
        "target_sample_count": diagnostic["target_sample_count"],
        "consensus_ratio": diagnostic["consensus_ratio"],
        "projection_perturbations": sorted(
            perturbation_evidence,
            key=lambda item: item["offset_pixels"],
        ),
    }
    record["evidence_sha256"] = _canonical_sha256(record)
    return record, []


def _bounded_signature_sibling_evidence(
    *,
    part_id: str,
    group_id: str,
    spatial_part: Mapping[str, Any],
    spatial_policy: Mapping[str, Any],
    deficit: Mapping[str, Any],
    reference_evidence: Mapping[str, Mapping[str, str]],
) -> tuple[dict[str, Any] | None, list[str]]:
    """Return one five-of-six diagnostic that requires render adjudication."""

    deficit_views = {
        str(item["reference_view_id"])
        for item in deficit.get("supporting_views", [])
        if isinstance(item, Mapping) and isinstance(item.get("reference_view_id"), str)
    }
    candidates: list[dict[str, Any]] = []
    for raw_observation in spatial_part.get("observations", []):
        if not isinstance(raw_observation, Mapping):
            continue
        observation = raw_observation
        view_id = observation.get("reference_view_id")
        if (
            not isinstance(view_id, str)
            or view_id not in deficit_views
            or view_id not in reference_evidence
        ):
            continue
        diagnostic_ok, has_alternatives = _diagnostic_projection(
            observation=observation,
            group_id=group_id,
            policy=spatial_policy,
        )
        diagnostic = observation.get("small_part_diagnostic")
        if (
            not diagnostic_ok
            or not has_alternatives
            or not isinstance(diagnostic, Mapping)
        ):
            continue
        alternatives = diagnostic.get("alternative_canonical_group_ids")
        group_scores = observation.get("group_scores")
        bbox_scores = observation.get("bbox_group_scores")
        winner = (
            group_scores[0]
            if isinstance(group_scores, Sequence)
            and not isinstance(group_scores, (str, bytes))
            and group_scores
            and isinstance(group_scores[0], Mapping)
            else None
        )
        bbox_winner = (
            bbox_scores[0]
            if isinstance(bbox_scores, Sequence)
            and not isinstance(bbox_scores, (str, bytes))
            and bbox_scores
            and isinstance(bbox_scores[0], Mapping)
            else None
        )
        direct_share = (
            _optional_unit(winner.get("color_share"))
            if isinstance(winner, Mapping)
            else None
        )
        direct_margin = _optional_unit(observation.get("color_margin"))
        bbox_share = (
            _optional_unit(bbox_winner.get("color_share"))
            if isinstance(bbox_winner, Mapping)
            else None
        )
        bbox_margin = _optional_unit(observation.get("bbox_color_margin"))
        diagnostic_ratio = _optional_unit(diagnostic.get("consensus_ratio"))
        projected_pixels = observation.get("projected_part_pixels")
        diagnostic_floor = _observation_diagnostic_floor(observation, spatial_policy)
        bounded = (
            diagnostic.get("status") == "resolved"
            and diagnostic.get("reason_codes") == []
            and diagnostic.get("canonical_group_id") == group_id
            and diagnostic.get("bbox_canonical_group_id") == group_id
            and diagnostic.get("registration_label_stable") is True
            and diagnostic.get("resolved_sample_count")
            == BOUNDED_SIGNATURE_REQUIRED_DIAGNOSTIC_SAMPLE_COUNT
            and diagnostic.get("target_sample_count")
            == BOUNDED_SIGNATURE_MIN_TARGET_SAMPLE_COUNT
            and diagnostic_ratio is not None
            and math.isclose(
                diagnostic_ratio,
                BOUNDED_SIGNATURE_MIN_TARGET_SAMPLE_COUNT
                / BOUNDED_SIGNATURE_REQUIRED_DIAGNOSTIC_SAMPLE_COUNT,
                rel_tol=0.0,
                abs_tol=1e-8,
            )
            and isinstance(alternatives, Sequence)
            and not isinstance(alternatives, (str, bytes))
            and len(alternatives) == 1
            and isinstance(alternatives[0], str)
            and alternatives[0] != group_id
            and isinstance(projected_pixels, int)
            and not isinstance(projected_pixels, bool)
            and isinstance(diagnostic_floor, int)
            and projected_pixels >= diagnostic_floor
            and isinstance(winner, Mapping)
            and winner.get("canonical_group_id") == group_id
            and direct_share is not None
            and direct_share >= BOUNDED_SIGNATURE_MIN_DIRECT_COLOR_SHARE
            and direct_margin is not None
            and direct_margin >= BOUNDED_SIGNATURE_MIN_DIRECT_COLOR_MARGIN
            and isinstance(bbox_winner, Mapping)
            and bbox_winner.get("canonical_group_id") == group_id
            and observation.get("bbox_canonical_group_id") == group_id
            and bbox_share is not None
            and bbox_share >= BOUNDED_SIGNATURE_MIN_BBOX_COLOR_SHARE
            and bbox_margin is not None
            and bbox_margin >= BOUNDED_SIGNATURE_MIN_BBOX_COLOR_MARGIN
        )
        if not bounded:
            continue
        record = {
            "part_id": part_id,
            "reference_view_id": view_id,
            "canonical_group_id": group_id,
            "projected_part_pixels": projected_pixels,
            "diagnostic_floor_pixels": diagnostic_floor,
            "direct_target_share": direct_share,
            "direct_target_margin": direct_margin,
            "bbox_target_share": bbox_share,
            "bbox_target_margin": bbox_margin,
            "resolved_sample_count": diagnostic["resolved_sample_count"],
            "target_sample_count": diagnostic["target_sample_count"],
            "consensus_ratio": diagnostic["consensus_ratio"],
            "alternative_canonical_group_ids": list(alternatives),
        }
        record["evidence_sha256"] = _canonical_sha256(record)
        candidates.append(record)
    if len(candidates) != 1:
        return None, ["BOUNDED_SIGNATURE_REQUIRES_ONE_BOUNDED_DIAGNOSTIC"]
    return candidates[0], []


def _bounded_signature_sibling_cohort_expansions(
    *,
    canonical_groups: Mapping[str, Mapping[str, Any]],
    repairable_groups: Mapping[str, Mapping[str, Any]],
    registry_by_part: Mapping[str, Mapping[str, Any]],
    spatial_parts: Mapping[str, Mapping[str, Any]],
    reference_evidence: Mapping[str, Mapping[str, str]],
    gate_decisions: Mapping[str, Mapping[str, Any]],
    mapping_decisions: Mapping[str, Mapping[str, Any]],
    geometry_risks: Mapping[str, bool],
    baseline_by_part: Mapping[str, Mapping[str, Any]],
    occupied_proposals_by_part: Mapping[str, Sequence[Mapping[str, Any]]],
    confirmed_materials: Mapping[str, str],
    provisional_material_groups: set[str],
    spatial_policy: Mapping[str, Any],
    minimum_semantic_confidence: float,
    input_hashes: Mapping[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Compile a two-member M0/M1 proposal; never finalize it locally."""

    source_signatures = {
        part_id: _source_visual_stable_properties_signature(part)
        for part_id, part in registry_by_part.items()
    }
    signature_part_ids: dict[str, list[str]] = defaultdict(list)
    for part_id, signature in source_signatures.items():
        if signature is not None:
            signature_part_ids[signature].append(part_id)
    raw_candidates: list[dict[str, Any]] = []
    for signature, raw_part_ids in sorted(signature_part_ids.items()):
        part_ids = sorted(raw_part_ids)
        if len(part_ids) != BOUNDED_SIGNATURE_REQUIRED_GLOBAL_COHORT_SIZE:
            continue
        parent_paths: dict[str, str] = {}
        for part_id in part_ids:
            parent_path = registry_by_part[part_id].get("parent_path")
            if isinstance(parent_path, str) and parent_path.startswith("/"):
                parent_paths[part_id] = parent_path.rstrip("/")
        if len(parent_paths) != len(part_ids):
            continue
        assembly_paths = {
            parent_path.rsplit("/", 1)[0] if "/" in parent_path[1:] else "/"
            for parent_path in parent_paths.values()
        }
        if len(assembly_paths) != 1 or len(set(parent_paths.values())) != 2:
            continue
        assembly_path = next(iter(assembly_paths))
        assembly_signature_part_ids = sorted(
            candidate_part_id
            for candidate_part_id, candidate_part in registry_by_part.items()
            if (
                source_signatures.get(candidate_part_id) == signature
                and isinstance(candidate_part.get("parent_path"), str)
                and (
                    candidate_part["parent_path"].rsplit("/", 1)[0]
                    if "/" in candidate_part["parent_path"][1:]
                    else "/"
                )
                == assembly_path
            )
        )
        if assembly_signature_part_ids != part_ids:
            continue
        if any(
            geometry_risks.get(part_id, True)
            or part_id not in spatial_parts
            or isinstance(registry_by_part[part_id].get("face_count"), bool)
            or not isinstance(registry_by_part[part_id].get("face_count"), int)
            or int(registry_by_part[part_id]["face_count"]) < 1
            for part_id in part_ids
        ):
            continue

        for group_id, canonical_group in sorted(canonical_groups.items()):
            material_id = confirmed_materials.get(group_id)
            deficit = repairable_groups.get(group_id)
            group_reasons = _bounded_signature_group_gate(
                canonical_group=canonical_group,
                repairable_group=deficit,
                reference_evidence=reference_evidence,
                material_id=material_id,
                provisional_material=group_id in provisional_material_groups,
            )
            if group_reasons or not isinstance(deficit, Mapping):
                continue
            assert material_id is not None
            vetoes: dict[str, list[str]] = {}
            for part_id in part_ids:
                veto_reasons, _ = _dominant_assembly_member_veto(
                    part_id=part_id,
                    target_group_id=group_id,
                    target_material_id=material_id,
                    target_source_signature=signature,
                    registry_part=registry_by_part[part_id],
                    spatial_part=spatial_parts[part_id],
                    spatial_gate_decision=gate_decisions.get(part_id),
                    mapping_decision=mapping_decisions.get(part_id),
                    geometry_risk=geometry_risks[part_id],
                    baseline_assignment=baseline_by_part[part_id],
                    existing_proposals=occupied_proposals_by_part.get(part_id, []),
                )
                explicitly_invisible = _explicitly_invisible_semantic_view_ids(
                    spatial_part=spatial_parts[part_id],
                    spatial_policy=spatial_policy,
                    part_id=part_id,
                )
                matched_semantic_alternative = any(
                    vote.get("alignment_trusted") is True
                    and vote.get("unique_canonical_join") is True
                    and vote.get("status") == "matched"
                    and isinstance(vote.get("canonical_group_id"), str)
                    and vote.get("canonical_group_id") != group_id
                    and isinstance(vote.get("view_id"), str)
                    and vote.get("view_id") not in explicitly_invisible
                    for raw_vote in spatial_parts[part_id].get("semantic_votes", [])
                    if isinstance(raw_vote, Mapping)
                    for vote in [raw_vote]
                )
                if matched_semantic_alternative:
                    veto_reasons.append(
                        "BOUNDED_SIGNATURE_MATCHED_SEMANTIC_ALTERNATIVE"
                    )
                if veto_reasons:
                    vetoes[part_id] = sorted(set(veto_reasons))
            if vetoes:
                continue
            anchors: dict[str, dict[str, Any]] = {}
            siblings: dict[str, dict[str, Any]] = {}
            for part_id in part_ids:
                anchor_evidence, anchor_reasons = (
                    _bounded_signature_exact_anchor_evidence(
                        part_id=part_id,
                        group_id=group_id,
                        spatial_part=spatial_parts[part_id],
                        spatial_gate_decision=gate_decisions.get(part_id),
                        mapping_decision=mapping_decisions.get(part_id),
                        spatial_policy=spatial_policy,
                        minimum_semantic_confidence=(minimum_semantic_confidence),
                        deficit=deficit,
                        reference_evidence=reference_evidence,
                    )
                )
                if not anchor_reasons and anchor_evidence is not None:
                    anchors[part_id] = anchor_evidence
                sibling_evidence, sibling_reasons = _bounded_signature_sibling_evidence(
                    part_id=part_id,
                    group_id=group_id,
                    spatial_part=spatial_parts[part_id],
                    spatial_policy=spatial_policy,
                    deficit=deficit,
                    reference_evidence=reference_evidence,
                )
                if not sibling_reasons and sibling_evidence is not None:
                    siblings[part_id] = sibling_evidence
            if (
                len(anchors) != 1
                or len(siblings) != 1
                or set(anchors) & set(siblings)
                or set(anchors) | set(siblings) != set(part_ids)
            ):
                continue
            anchor_part_id = next(iter(anchors))
            sibling_part_id = next(iter(siblings))
            raw_candidates.append(
                {
                    "canonical_group_id": group_id,
                    "material_id": material_id,
                    "assembly_path": assembly_path,
                    "source_visual_stable_properties_signature_sha256": (signature),
                    "anchor_part_id": anchor_part_id,
                    "sibling_part_id": sibling_part_id,
                    "anchor_evidence": anchors[anchor_part_id],
                    "sibling_evidence": siblings[sibling_part_id],
                    "cohort_part_ids": part_ids,
                }
            )

    candidate_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for candidate in raw_candidates:
        candidate_groups[
            (
                str(candidate["assembly_path"]),
                str(candidate["source_visual_stable_properties_signature_sha256"]),
            )
        ].append(candidate)
    accepted_records: list[dict[str, Any]] = []
    cohort_audits: list[dict[str, Any]] = []
    for candidates in candidate_groups.values():
        if len(candidates) != 1:
            continue
        candidate = candidates[0]
        group_id = str(candidate["canonical_group_id"])
        material_id = str(candidate["material_id"])
        signature = str(candidate["source_visual_stable_properties_signature_sha256"])
        anchor_part_id = str(candidate["anchor_part_id"])
        sibling_part_id = str(candidate["sibling_part_id"])
        cohort_part_ids = list(candidate["cohort_part_ids"])
        anchor_supporting_view_ids = [
            str(candidate["anchor_evidence"]["reference_view_id"])
        ]
        anchor_branch = _dominant_assembly_child_branch(
            parent_path=str(registry_by_part[anchor_part_id]["parent_path"]),
            assembly_path=str(candidate["assembly_path"]),
        )
        if anchor_branch is None:
            continue
        identity_payload = {
            "schema_version": "qwen-dominant-assembly-cohort/v1",
            "candidate_kind": "rare_source_identity_pair",
            "proposal_policy": BOUNDED_SIGNATURE_SIBLING_PROPOSAL_POLICY,
            "canonical_group_id": group_id,
            "assembly_path": candidate["assembly_path"],
            "source_visual_stable_properties_signature_sha256": signature,
            "anchor_part_ids": [anchor_part_id],
            "cohort_part_ids": cohort_part_ids,
            "input_hashes": dict(sorted(input_hashes.items())),
        }
        cohort_id = _canonical_sha256(identity_payload)
        assembly_subtree_part_ids = sorted(
            part_id
            for part_id, part in registry_by_part.items()
            if (
                isinstance(part.get("parent_path"), str)
                and (
                    part["parent_path"] == candidate["assembly_path"]
                    or part["parent_path"].startswith(f"{candidate['assembly_path']}/")
                )
            )
        )
        contract: dict[str, Any] = {
            "schema_version": "qwen-dominant-assembly-cohort/v1",
            "candidate_kind": "rare_source_identity_pair",
            "proposal_policy": BOUNDED_SIGNATURE_SIBLING_PROPOSAL_POLICY,
            "cohort_id": cohort_id,
            "canonical_group_id": group_id,
            "material_id": material_id,
            "material_selection_status": "CONFIRMED",
            "assembly_path": candidate["assembly_path"],
            "source_visual_stable_properties_signature_sha256": signature,
            "membership_status": DOMINANT_ASSEMBLY_MEMBERSHIP_STATUS,
            "anchor_part_ids": [anchor_part_id],
            "anchor_supporting_view_ids": anchor_supporting_view_ids,
            "anchor_child_branches": [anchor_branch],
            "subtree_part_ids": assembly_subtree_part_ids,
            "candidate_signature_part_ids": cohort_part_ids,
            "accepted_part_ids": cohort_part_ids,
            "cohort_part_ids": cohort_part_ids,
            "expanded_member_part_ids": [sibling_part_id],
            "excluded_members": [],
            "anchor_evidence": [copy.deepcopy(candidate["anchor_evidence"])],
            "bounded_sibling_evidence": copy.deepcopy(candidate["sibling_evidence"]),
            "source_identity_contract": {
                "global_signature_part_count": len(cohort_part_ids),
                "assembly_signature_part_count": len(cohort_part_ids),
                "globally_exact_two_member_signature": True,
            },
            "render_membership_requirement": {
                "minimum_trusted_reference_view_count": 2,
                "all_available_reference_views_required": True,
                "non_target_group_regression_forces_m0": True,
                "incomplete_or_ambiguous_evidence_forces_m0": True,
            },
            "thresholds": {
                "minimum_group_confidence": (BOUNDED_SIGNATURE_MIN_GROUP_CONFIDENCE),
                "minimum_group_source_views": (
                    BOUNDED_SIGNATURE_MIN_GROUP_SOURCE_VIEWS
                ),
                "minimum_qa_views": BOUNDED_SIGNATURE_MIN_QA_VIEWS,
                "required_global_signature_part_count": (
                    BOUNDED_SIGNATURE_REQUIRED_GLOBAL_COHORT_SIZE
                ),
                "required_diagnostic_sample_count": (
                    BOUNDED_SIGNATURE_REQUIRED_DIAGNOSTIC_SAMPLE_COUNT
                ),
                "minimum_bounded_target_sample_count": (
                    BOUNDED_SIGNATURE_MIN_TARGET_SAMPLE_COUNT
                ),
                "minimum_bounded_direct_color_share": (
                    BOUNDED_SIGNATURE_MIN_DIRECT_COLOR_SHARE
                ),
                "minimum_bounded_direct_color_margin": (
                    BOUNDED_SIGNATURE_MIN_DIRECT_COLOR_MARGIN
                ),
                "minimum_bounded_bbox_color_share": (
                    BOUNDED_SIGNATURE_MIN_BBOX_COLOR_SHARE
                ),
                "minimum_bounded_bbox_color_margin": (
                    BOUNDED_SIGNATURE_MIN_BBOX_COLOR_MARGIN
                ),
            },
            "input_hashes": dict(sorted(input_hashes.items())),
        }
        contract["contract_sha256"] = _canonical_sha256(contract)
        cohort_audits.append(copy.deepcopy(contract))
        for part_id in cohort_part_ids:
            accepted_records.append(
                {
                    "part_id": part_id,
                    "canonical_group_id": group_id,
                    "material_id": material_id,
                    "provisional_material_candidate": False,
                    "localization_lane": (BOUNDED_SIGNATURE_SIBLING_COHORT_LANE),
                    "member_role": (
                        "strict_spatial_anchor"
                        if part_id == anchor_part_id
                        else "expanded_member"
                    ),
                    "contract": copy.deepcopy(contract),
                    "baseline_material_id": str(
                        baseline_by_part[part_id]["material_id"]
                    ),
                }
            )
    return (
        sorted(accepted_records, key=lambda item: item["part_id"]),
        sorted(cohort_audits, key=lambda item: item["cohort_id"]),
    )


def _source_identity_diagnostic_support_views(
    *,
    part_id: str,
    group_id: str,
    spatial_part: Mapping[str, Any],
    spatial_gate_decision: Mapping[str, Any] | None,
    mapping_decision: Mapping[str, Any] | None,
    spatial_policy: Mapping[str, Any],
    minimum_semantic_confidence: float,
    deficit: Mapping[str, Any],
    reference_evidence: Mapping[str, Mapping[str, str]],
) -> tuple[list[str], list[str]]:
    """Return one exact diagnostic pending a source-identity sibling anchor."""

    reasons: list[str] = []
    if deficit.get("repairable") is not True:
        reasons.append("SOURCE_IDENTITY_REQUIRES_REPAIRABLE_MULTIVIEW_GROUP")
    deficit_views = {
        str(item["reference_view_id"])
        for item in _sequence(
            deficit.get("supporting_views"),
            f"deficit {group_id}.supporting_views",
        )
        if isinstance(item, Mapping) and isinstance(item.get("reference_view_id"), str)
    }
    if len(deficit_views) < 2:
        reasons.append("SOURCE_IDENTITY_REQUIRES_MULTIVIEW_QA_DEFICIT")

    supports: set[str] = set()
    spatial_conflicts: set[str] = set()
    for raw_observation in _sequence(
        spatial_part.get("observations", []),
        f"spatial part {part_id}.observations",
    ):
        observation = _mapping(raw_observation, f"spatial part {part_id}.observation")
        view_id = observation.get("reference_view_id")
        if not isinstance(view_id, str):
            continue
        canonical_group_id = observation.get("canonical_group_id")
        if (
            observation.get("classification") == "resolved"
            and isinstance(canonical_group_id, str)
            and canonical_group_id != group_id
        ):
            spatial_conflicts.add(canonical_group_id)
        diagnostic_ok, has_alternatives = _diagnostic_projection(
            observation=observation,
            group_id=group_id,
            policy=spatial_policy,
        )
        if (
            view_id in deficit_views
            and diagnostic_ok
            and not has_alternatives
            and view_id in reference_evidence
        ):
            supports.add(view_id)
        for field in ("small_part_diagnostic", "canonical_palette_diagnostic"):
            diagnostic = observation.get(field)
            if (
                isinstance(diagnostic, Mapping)
                and diagnostic.get("status") == "resolved"
                and diagnostic.get("reason_codes") == []
                and isinstance(diagnostic.get("canonical_group_id"), str)
                and diagnostic.get("canonical_group_id") != group_id
            ):
                spatial_conflicts.add(str(diagnostic["canonical_group_id"]))

    explicitly_invisible = _explicitly_invisible_semantic_view_ids(
        spatial_part=spatial_part,
        spatial_policy=spatial_policy,
        part_id=part_id,
    )
    matched_semantic_conflicts = {
        str(vote["canonical_group_id"])
        for raw_vote in _sequence(
            spatial_part.get("semantic_votes", []),
            f"spatial part {part_id}.semantic_votes",
        )
        if isinstance(raw_vote, Mapping)
        for vote in [raw_vote]
        if (
            vote.get("alignment_trusted") is True
            and vote.get("unique_canonical_join") is True
            and vote.get("status") == "matched"
            and isinstance(vote.get("canonical_group_id"), str)
            and vote.get("canonical_group_id") != group_id
            and isinstance(vote.get("effective_confidence"), (int, float))
            and not isinstance(vote.get("effective_confidence"), bool)
            and float(vote["effective_confidence"]) >= minimum_semantic_confidence
            and isinstance(vote.get("view_id"), str)
            and vote.get("view_id") not in explicitly_invisible
        )
    }
    if spatial_conflicts:
        reasons.append("SOURCE_IDENTITY_SPATIAL_GROUP_CONFLICT")
    if matched_semantic_conflicts:
        reasons.append("SOURCE_IDENTITY_MATCHED_SEMANTIC_GROUP_CONFLICT")
    if mapping_decision is not None and (
        mapping_decision.get("output_status") == "matched"
        and mapping_decision.get("output_group_id") != group_id
    ):
        reasons.append("SOURCE_IDENTITY_MAPPING_GROUP_CONFLICT")
    if spatial_gate_decision is not None and (
        spatial_gate_decision.get("output_status") == "matched"
        and spatial_gate_decision.get("output_group_id") != group_id
    ):
        reasons.append("SOURCE_IDENTITY_GATE_GROUP_CONFLICT")
    if len(supports) != 1:
        reasons.append("SOURCE_IDENTITY_REQUIRES_ONE_EXACT_DIAGNOSTIC")
    return sorted(supports), sorted(set(reasons))


def _ordinary_black_budget_support(
    *,
    quality_report: Mapping[str, Any],
    deficit: Mapping[str, Any],
    group_id: str,
) -> tuple[dict[str, Any] | None, list[str]]:
    reasons: list[str] = []
    supports = [
        item
        for item in _sequence(
            deficit.get("supporting_views"),
            f"deficit {group_id}.supporting_views",
        )
        if isinstance(item, Mapping) and item.get("deficit_sources") == ["group_recall"]
    ]
    if len(supports) != 1:
        return None, ["REPEATED_DARK_REQUIRES_ONE_ORDINARY_QA_DEFICIT"]
    support = supports[0]
    view_id = support.get("reference_view_id")
    local_group_id = support.get("local_group_id")
    quality_views = [
        view
        for raw_view in _sequence(quality_report.get("views"), "quality_report.views")
        for view in [_mapping(raw_view, "quality report view")]
        if view.get("reference_view_id") == view_id
    ]
    if len(quality_views) != 1 or not isinstance(local_group_id, str):
        return None, ["REPEATED_DARK_QA_VIEW_NOT_UNIQUE"]
    view = quality_views[0]
    material_color = view.get("material_color")
    group_recall = (
        material_color.get("trusted_evidence_group_recall")
        if isinstance(material_color, Mapping)
        else None
    )
    rows = (
        [
            row
            for raw_row in group_recall.get("groups", [])
            for row in [_mapping(raw_row, "quality group recall row")]
            if row.get("group_id") == local_group_id
        ]
        if isinstance(group_recall, Mapping)
        else []
    )
    if len(rows) != 1:
        return None, ["REPEATED_DARK_QA_GROUP_ROW_NOT_UNIQUE"]
    row = rows[0]
    base_colors = row.get("base_colors")
    bins = row.get("render_color_bins")
    required_share = _optional_unit(row.get("required_render_share"))
    observed_share = _optional_unit(row.get("observed_render_share"))
    recall = _optional_unit(row.get("recall"))
    render = view.get("render")
    foreground = render.get("foreground") if isinstance(render, Mapping) else None
    render_pixels = (
        foreground.get("pixel_count") if isinstance(foreground, Mapping) else None
    )
    if (
        not isinstance(base_colors, Sequence)
        or isinstance(base_colors, (str, bytes))
        or {str(value).strip().casefold() for value in base_colors} != {"black"}
        or not isinstance(bins, Sequence)
        or isinstance(bins, (str, bytes))
        or set(str(value) for value in bins) != {"black", "achromatic_dark"}
        or required_share is None
        or observed_share is None
        or required_share <= observed_share
        or recall is None
        or recall >= 0.5
        or not isinstance(render_pixels, int)
        or isinstance(render_pixels, bool)
        or render_pixels <= 0
    ):
        reasons.append("REPEATED_DARK_QA_BUDGET_EVIDENCE_INVALID")
    budget_pixels = (
        int(math.ceil((required_share - observed_share) * render_pixels))
        if not reasons
        else 0
    )
    if budget_pixels < 1:
        reasons.append("REPEATED_DARK_QA_BUDGET_NOT_POSITIVE")
    if reasons:
        return None, sorted(set(reasons))
    assert isinstance(view_id, str)
    return (
        {
            "reference_view_id": view_id,
            "local_group_id": local_group_id,
            "reference_sha256": support.get("reference_sha256"),
            "required_render_share": required_share,
            "observed_render_share": observed_share,
            "render_foreground_pixels": render_pixels,
            "budget_pixels": budget_pixels,
            "minimum_contribution_pixels": int(
                math.ceil(REPEATED_DARK_MIN_BUDGET_FACTOR * budget_pixels)
            ),
            "maximum_contribution_pixels": int(
                math.floor(REPEATED_DARK_MAX_BUDGET_FACTOR * budget_pixels)
            ),
        },
        [],
    )


def _repeated_dark_member_audit(
    *,
    part_id: str,
    group_id: str,
    canonical_group: Mapping[str, Any],
    view_id: str,
    spatial_part: Mapping[str, Any],
    spatial_gate_decision: Mapping[str, Any] | None,
    mapping_decision: Mapping[str, Any] | None,
    alignment_audits: Mapping[str, Mapping[str, Any]],
    reference_evidence: Mapping[str, Mapping[str, str]],
    canonical_reference_evidence: Mapping[str, Mapping[str, str]],
) -> tuple[dict[str, Any] | None, list[str]]:
    """Validate one member of an atomic repeated-geometry black cohort."""

    reasons: list[str] = []
    observations = [
        observation
        for raw_observation in _sequence(
            spatial_part.get("observations", []),
            f"spatial part {part_id}.observations",
        )
        for observation in [_mapping(raw_observation, f"{part_id} observation")]
        if observation.get("reference_view_id") == view_id
    ]
    if len(observations) != 1:
        return None, ["REPEATED_DARK_TARGET_OBSERVATION_NOT_UNIQUE"]
    observation = observations[0]
    projected_pixels = observation.get("projected_part_pixels")
    direct_share = _group_color_share(observation.get("group_scores"), group_id)
    direct_margin = _optional_unit(observation.get("color_margin"))
    direct_matching_pixels = sum(
        int(score["matching_pixels"])
        for score in observation.get("group_scores", [])
        if isinstance(score, Mapping)
        and score.get("canonical_group_id") == group_id
        and isinstance(score.get("matching_pixels"), int)
        and not isinstance(score.get("matching_pixels"), bool)
        and int(score["matching_pixels"]) >= 0
    )
    bbox_share = _group_color_share(observation.get("bbox_group_scores"), group_id)
    bbox_margin = _optional_unit(observation.get("bbox_color_margin"))
    if (
        not isinstance(projected_pixels, int)
        or isinstance(projected_pixels, bool)
        or projected_pixels < REPEATED_DARK_MIN_PROJECTED_PIXELS
        or observation.get("canonical_group_id") != group_id
        or observation.get("registration_label_stable") is not True
        or observation.get("perturbation_label_stable") is not True
        or direct_share is None
        or direct_share < REPEATED_DARK_MIN_DIRECT_SHARE
        or direct_margin is None
        or direct_margin < REPEATED_DARK_MIN_DIRECT_MARGIN
        or observation.get("bbox_canonical_group_id") != group_id
        or bbox_share is None
        or bbox_share < REPEATED_DARK_MIN_BBOX_SHARE
        or bbox_margin is None
        or bbox_margin < REPEATED_DARK_MIN_BBOX_MARGIN
    ):
        reasons.append("REPEATED_DARK_DIRECT_OR_BBOX_PROJECTION_INVALID")

    required_offsets = {(-2, 0), (2, 0), (0, -2), (0, 2)}
    perturbation_audits: list[dict[str, Any]] = []
    offsets: set[tuple[int, int]] = set()
    perturbations = observation.get("projection_perturbations")
    if not isinstance(perturbations, Sequence) or isinstance(
        perturbations, (str, bytes)
    ):
        perturbations = []
    for raw_perturbation in perturbations:
        if not isinstance(raw_perturbation, Mapping):
            reasons.append("REPEATED_DARK_PERTURBATION_INVALID")
            continue
        offset = raw_perturbation.get("offset_pixels")
        share = _optional_unit(raw_perturbation.get("best_color_share"))
        margin = _optional_unit(raw_perturbation.get("color_margin"))
        pixels = raw_perturbation.get("sampled_reference_pixels")
        valid_offset = (
            isinstance(offset, Sequence)
            and not isinstance(offset, (str, bytes))
            and len(offset) == 2
            and all(
                isinstance(value, int) and not isinstance(value, bool)
                for value in offset
            )
        )
        offset_tuple = (int(offset[0]), int(offset[1])) if valid_offset else (0, 0)
        if (
            not valid_offset
            or offset_tuple not in required_offsets
            or offset_tuple in offsets
            or raw_perturbation.get("canonical_group_id") != group_id
            or raw_perturbation.get("diagnostic_canonical_group_id") != group_id
            or not isinstance(pixels, int)
            or isinstance(pixels, bool)
            or pixels < REPEATED_DARK_MIN_PROJECTED_PIXELS
            or share is None
            or share < REPEATED_DARK_MIN_PERTURBATION_SHARE
            or margin is None
            or margin < REPEATED_DARK_MIN_PERTURBATION_MARGIN
        ):
            reasons.append("REPEATED_DARK_PERTURBATION_INVALID")
            continue
        offsets.add(offset_tuple)
        perturbation_audits.append(
            {
                "offset_pixels": [offset_tuple[0], offset_tuple[1]],
                "sampled_reference_pixels": pixels,
                "target_share": share,
                "target_margin": margin,
            }
        )
    if offsets != required_offsets or len(perturbation_audits) != 4:
        reasons.append("REPEATED_DARK_PERTURBATION_INVALID")

    alignment = alignment_audits.get(view_id)
    strict_reference_alignment = (
        isinstance(alignment, Mapping)
        and alignment.get("ecc_status") == "success"
        and (_optional_unit(alignment.get("score")) or 0.0)
        >= REPEATED_DARK_MIN_ALIGNMENT_SCORE
        and (_optional_unit(alignment.get("projection_iou")) or 0.0)
        >= REPEATED_DARK_MIN_PROJECTION_IOU
        and (_optional_unit(alignment.get("ecc_correlation")) or 0.0)
        >= REPEATED_DARK_MIN_ECC_CORRELATION
    )
    # An exact direct/bbox/four-perturbation projection is more local than a
    # Qwen semantic vote over an assembly crop.  When every member of a rare
    # repeated-geometry cohort proves that same projection and the complete
    # cohort fits the independently measured QA deficit budget, allow the
    # cohort to proceed even when the optional dark-on-black diagnostic was
    # not emitted.  The outer compiler still rejects partial cohorts and
    # multiple eligible cohorts atomically.
    if not reasons and strict_reference_alignment:
        assert isinstance(projected_pixels, int)
        assert direct_share is not None
        assert direct_margin is not None
        assert bbox_share is not None
        assert bbox_margin is not None
        strict_record = {
            "part_id": part_id,
            "projected_part_pixels": projected_pixels,
            "direct_target_share": direct_share,
            "direct_target_margin": direct_margin,
            "direct_target_matching_pixels": direct_matching_pixels,
            "bbox_target_share": bbox_share,
            "bbox_target_margin": bbox_margin,
            "perturbations": sorted(
                perturbation_audits,
                key=lambda item: item["offset_pixels"],
            ),
            "alignment": {
                "score": alignment["score"],
                "projection_iou": alignment["projection_iou"],
                "ecc_correlation": alignment["ecc_correlation"],
                "ecc_status": alignment["ecc_status"],
            },
            "evidence_contract": "strict_reference_space_projection",
            "semantic_alternative_disproofs": [],
        }
        strict_record["evidence_sha256"] = _canonical_sha256(strict_record)
        return strict_record, []

    diagnostic = observation.get("dark_foreground_diagnostic")
    if not isinstance(diagnostic, Mapping):
        return None, sorted(set([*reasons, "DARK_FOREGROUND_DIAGNOSTIC_MISSING"]))
    diagnostic_alignment = diagnostic.get("alignment")
    if not isinstance(alignment, Mapping) or not isinstance(
        diagnostic_alignment, Mapping
    ):
        reasons.append("REPEATED_DARK_ALIGNMENT_INVALID")
    else:
        alignment_contract = (
            diagnostic_alignment.get("trusted") is True
            and diagnostic_alignment.get("strong") is False
            and diagnostic_alignment.get("reason_codes_empty") is True
            and diagnostic_alignment.get("transform_constraints_passed") is True
            and diagnostic_alignment.get("ecc_status") == "success"
        )
        for field, minimum in (
            ("score", REPEATED_DARK_MIN_ALIGNMENT_SCORE),
            ("projection_iou", REPEATED_DARK_MIN_PROJECTION_IOU),
            ("ecc_correlation", REPEATED_DARK_MIN_ECC_CORRELATION),
        ):
            value = _optional_unit(diagnostic_alignment.get(field))
            trusted_value = _optional_unit(alignment.get(field))
            if (
                value is None
                or value < minimum
                or trusted_value is None
                or not math.isclose(value, trusted_value, abs_tol=5e-8)
            ):
                alignment_contract = False
        projection_score = _optional_unit(diagnostic_alignment.get("projection_score"))
        if (
            projection_score is None
            or projection_score < REPEATED_DARK_MIN_PROJECTION_SCORE
        ):
            alignment_contract = False
        if not alignment_contract:
            reasons.append("REPEATED_DARK_ALIGNMENT_INVALID")

    diagnostic_hash = diagnostic.get("diagnostic_sha256")
    unsigned_diagnostic = copy.deepcopy(dict(diagnostic))
    unsigned_diagnostic.pop("diagnostic_sha256", None)
    if (
        not isinstance(diagnostic_hash, str)
        or not _SHA256_PATTERN.fullmatch(diagnostic_hash)
        or _canonical_sha256(unsigned_diagnostic) != diagnostic_hash
    ):
        reasons.append("DARK_DIAGNOSTIC_HASH_INVALID")
    if (
        diagnostic.get("status") != "rejected"
        or set(diagnostic.get("reason_codes", []))
        != REPEATED_DARK_ALLOWED_DIAGNOSTIC_REASONS
        or diagnostic.get("evidence_scope") != "dark_on_black_foreground_repair_only"
        or diagnostic.get("canonical_group_id") != group_id
    ):
        reasons.append("REPEATED_DARK_DIAGNOSTIC_REASON_CONTRACT_INVALID")
    source_ids = diagnostic.get("canonical_source_view_ids")
    expected_source_ids = canonical_group.get("source_view_ids")
    if (
        not isinstance(source_ids, Sequence)
        or isinstance(source_ids, (str, bytes))
        or list(source_ids) != sorted(set(source_ids))
        or len(source_ids) < 2
        or not isinstance(expected_source_ids, Sequence)
        or isinstance(expected_source_ids, (str, bytes))
        or list(source_ids)
        != sorted(set(str(source_id) for source_id in expected_source_ids))
        or any(
            source_id not in canonical_reference_evidence for source_id in source_ids
        )
        or len(
            {
                canonical_reference_evidence[str(source_id)]["raw_sha256"]
                for source_id in source_ids
            }
        )
        < 2
        or len(
            {
                canonical_reference_evidence[str(source_id)]["content_cluster_id"]
                for source_id in source_ids
            }
        )
        < 2
    ):
        reasons.append("DARK_DIAGNOSTIC_CANONICAL_SOURCES_INVALID")
    if (
        diagnostic.get("projected_part_pixels") != projected_pixels
        or not isinstance(diagnostic.get("normalized_projected_pixels"), int)
        or isinstance(diagnostic.get("normalized_projected_pixels"), bool)
        or int(diagnostic["normalized_projected_pixels"])
        < int(
            DARK_FOREGROUND_DIAGNOSTIC_THRESHOLDS["minimum_normalized_projected_pixels"]
        )
        or diagnostic.get("thresholds") != DARK_FOREGROUND_DIAGNOSTIC_THRESHOLDS
    ):
        reasons.append("DARK_DIAGNOSTIC_SCHEMA_INVALID")
    normalization = diagnostic.get("normalization")
    background = diagnostic.get("background")
    original_size = (
        normalization.get("original_size")
        if isinstance(normalization, Mapping)
        else None
    )
    normalized_size = (
        normalization.get("normalized_size")
        if isinstance(normalization, Mapping)
        else None
    )
    scale = normalization.get("scale") if isinstance(normalization, Mapping) else None
    median_bgr = (
        background.get("median_bgr") if isinstance(background, Mapping) else None
    )
    background_numbers = (
        [
            background.get("border_distance_p95"),
            background.get("distance_threshold"),
        ]
        if isinstance(background, Mapping)
        else []
    )
    if (
        not isinstance(normalization, Mapping)
        or normalization.get("long_edge_pixels") != 512
        or not isinstance(original_size, Sequence)
        or isinstance(original_size, (str, bytes))
        or len(original_size) != 2
        or not isinstance(normalized_size, Sequence)
        or isinstance(normalized_size, (str, bytes))
        or len(normalized_size) != 2
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in [*original_size, *normalized_size]
        )
        or max(int(value) for value in normalized_size) != 512
        or not isinstance(scale, (int, float))
        or isinstance(scale, bool)
        or not math.isfinite(float(scale))
        or float(scale) <= 0.0
        or not isinstance(background, Mapping)
        or not isinstance(median_bgr, Sequence)
        or isinstance(median_bgr, (str, bytes))
        or len(median_bgr) != 3
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 255.0
            for value in median_bgr
        )
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
            for value in background_numbers
        )
        or float(background_numbers[1]) <= 0.0
    ):
        reasons.append("DARK_DIAGNOSTIC_SCHEMA_INVALID")

    normalized_pixels = diagnostic.get("normalized_projected_pixels")

    def diagnostic_count(field: str) -> int | None:
        value = diagnostic.get(field)
        return (
            value
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0
            else None
        )

    count_ratio_fields = (
        ("near_black_pixels", "near_black_share"),
        ("non_background_pixels", "non_background_share"),
        ("dark_signal_pixels", "dark_signal_share"),
        ("core_dark_signal_pixels", "core_dark_signal_share"),
        ("adaptive_edge_pixels", "adaptive_edge_density"),
    )
    counts = {field: diagnostic_count(field) for field, _ in count_ratio_fields}
    core_pixels = diagnostic_count("core_pixels")
    primitive_valid = isinstance(normalized_pixels, int) and not isinstance(
        normalized_pixels, bool
    )
    for count_field, ratio_field in count_ratio_fields:
        count_value = counts[count_field]
        ratio_value = _optional_unit(diagnostic.get(ratio_field))
        denominator = (
            diagnostic_count("non_background_pixels")
            if ratio_field == "dark_signal_purity"
            else core_pixels
            if ratio_field == "core_dark_signal_share"
            else normalized_pixels
        )
        if (
            count_value is None
            or ratio_value is None
            or not isinstance(denominator, int)
            or denominator < 0
            or not math.isclose(
                ratio_value,
                count_value / denominator if denominator else 0.0,
                abs_tol=1e-8,
            )
        ):
            primitive_valid = False
    dark_signal_pixels = diagnostic_count("dark_signal_pixels")
    non_background_pixels = diagnostic_count("non_background_pixels")
    dark_signal_purity = _optional_unit(diagnostic.get("dark_signal_purity"))
    if (
        dark_signal_pixels is None
        or non_background_pixels is None
        or dark_signal_purity is None
        or not math.isclose(
            dark_signal_purity,
            dark_signal_pixels / non_background_pixels
            if non_background_pixels
            else 0.0,
            abs_tol=1e-8,
        )
    ):
        primitive_valid = False
    threshold_checks = (
        _optional_unit(diagnostic.get("near_black_share")),
        _optional_unit(diagnostic.get("dark_signal_share")),
        dark_signal_purity,
        _optional_unit(diagnostic.get("core_dark_signal_share")),
        _optional_unit(diagnostic.get("adaptive_edge_density")),
    )
    if (
        not primitive_valid
        or any(value is None for value in threshold_checks)
        or threshold_checks[0]
        < float(DARK_FOREGROUND_DIAGNOSTIC_THRESHOLDS["minimum_near_black_share"])
        or non_background_pixels
        < int(DARK_FOREGROUND_DIAGNOSTIC_THRESHOLDS["minimum_non_background_pixels"])
        or threshold_checks[1]
        < float(DARK_FOREGROUND_DIAGNOSTIC_THRESHOLDS["minimum_dark_signal_share"])
        or threshold_checks[2]
        < float(DARK_FOREGROUND_DIAGNOSTIC_THRESHOLDS["minimum_dark_signal_purity"])
        or core_pixels
        < int(DARK_FOREGROUND_DIAGNOSTIC_THRESHOLDS["minimum_core_pixels"])
        or threshold_checks[3]
        < float(DARK_FOREGROUND_DIAGNOSTIC_THRESHOLDS["minimum_core_dark_signal_share"])
        or threshold_checks[4]
        < float(DARK_FOREGROUND_DIAGNOSTIC_THRESHOLDS["minimum_adaptive_edge_density"])
    ):
        reasons.append("DARK_DIAGNOSTIC_PRIMITIVE_GATE_FAILED")
    core_distance = diagnostic.get("core_distance_pixels")
    canny_low = diagnostic.get("canny_low_threshold")
    canny_high = diagnostic.get("canny_high_threshold")
    if (
        not isinstance(core_distance, (int, float))
        or isinstance(core_distance, bool)
        or not math.isfinite(float(core_distance))
        or float(core_distance)
        != float(DARK_FOREGROUND_DIAGNOSTIC_THRESHOLDS["core_distance_pixels"])
        or not isinstance(canny_low, int)
        or isinstance(canny_low, bool)
        or not 0 <= canny_low <= 255
        or not isinstance(canny_high, int)
        or isinstance(canny_high, bool)
        or not 0 <= canny_high <= 255
        or canny_low >= canny_high
    ):
        reasons.append("DARK_DIAGNOSTIC_EDGE_OR_CORE_SCHEMA_INVALID")

    nulls = diagnostic.get("null_shifts")
    valid_shares: list[float] = []
    null_valid = (
        isinstance(nulls, Sequence)
        and not isinstance(nulls, (str, bytes))
        and len(nulls) == 8
    )
    null_offsets: set[tuple[int, int]] = set()
    if null_valid:
        for raw_null in nulls:
            if not isinstance(raw_null, Mapping):
                null_valid = False
                continue
            offset = raw_null.get("offset_pixels")
            retained = raw_null.get("retained_pixels")
            pixels = raw_null.get("dark_signal_pixels")
            area_ratio = _optional_unit(raw_null.get("valid_area_ratio"))
            share = _optional_unit(raw_null.get("dark_signal_share"))
            valid = raw_null.get("valid")
            if (
                not isinstance(offset, Sequence)
                or isinstance(offset, (str, bytes))
                or len(offset) != 2
                or any(
                    not isinstance(value, int) or isinstance(value, bool)
                    for value in offset
                )
                or (int(offset[0]), int(offset[1])) == (0, 0)
                or max(abs(int(offset[0])), abs(int(offset[1])))
                < int(
                    DARK_FOREGROUND_DIAGNOSTIC_THRESHOLDS["minimum_null_offset_pixels"]
                )
                or (int(offset[0]), int(offset[1])) in null_offsets
                or not isinstance(retained, int)
                or isinstance(retained, bool)
                or retained < 0
                or not isinstance(pixels, int)
                or isinstance(pixels, bool)
                or pixels < 0
                or area_ratio is None
                or share is None
                or not isinstance(valid, bool)
                or not isinstance(normalized_pixels, int)
                or not math.isclose(
                    area_ratio, retained / normalized_pixels, abs_tol=1e-8
                )
                or not math.isclose(
                    share, pixels / retained if retained else 0.0, abs_tol=1e-8
                )
                or valid
                != (
                    area_ratio
                    >= float(
                        DARK_FOREGROUND_DIAGNOSTIC_THRESHOLDS[
                            "minimum_null_valid_area_ratio"
                        ]
                    )
                )
                or not isinstance(raw_null.get("mask_sha256"), str)
                or not _SHA256_PATTERN.fullmatch(str(raw_null.get("mask_sha256")))
            ):
                null_valid = False
                continue
            null_offsets.add((int(offset[0]), int(offset[1])))
            if valid:
                valid_shares.append(share)
    q75 = _optional_unit(diagnostic.get("null_dark_signal_share_q75"))
    margin = diagnostic.get("dark_signal_null_margin")
    recomputed_q75 = _linear_quantile(valid_shares, 0.75) if valid_shares else None
    if (
        not null_valid
        or diagnostic.get("valid_null_shift_count") != len(valid_shares)
        or len(valid_shares)
        < int(DARK_FOREGROUND_DIAGNOSTIC_THRESHOLDS["minimum_valid_null_shifts"])
        or q75 is None
        or recomputed_q75 is None
        or not math.isclose(q75, recomputed_q75, abs_tol=1e-8)
        or not isinstance(margin, (int, float))
        or isinstance(margin, bool)
        or not math.isfinite(float(margin))
        or threshold_checks[1] is None
        or not math.isclose(
            float(margin), float(threshold_checks[1]) - recomputed_q75, abs_tol=1e-8
        )
        or float(margin)
        < float(DARK_FOREGROUND_DIAGNOSTIC_THRESHOLDS["minimum_null_q75_margin"])
    ):
        reasons.append("DARK_DIAGNOSTIC_NULL_CONTROL_INVALID")
    for hash_field in (
        "normalized_reference_pixel_sha256",
        "normalized_projected_mask_sha256",
        "normalized_near_black_mask_sha256",
        "normalized_non_background_mask_sha256",
        "normalized_dark_signal_mask_sha256",
        "normalized_adaptive_edge_mask_sha256",
    ):
        value = diagnostic.get(hash_field)
        if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
            reasons.append(f"DARK_DIAGNOSTIC_HASH_INVALID:{hash_field}")
    if view_id not in reference_evidence:
        reasons.append("REPEATED_DARK_REFERENCE_EVIDENCE_MISSING")

    if mapping_decision is None:
        reasons.append("REPEATED_DARK_MAPPING_DECISION_MISSING")
    else:
        main_confidence = _optional_unit(mapping_decision.get("main_confidence"))
        if (
            mapping_decision.get("main_group_id") != group_id
            or mapping_decision.get("main_status") not in {"matched", "review"}
            or main_confidence is None
            or main_confidence < REPEATED_DARK_MIN_MAPPING_CONFIDENCE
            or (
                mapping_decision.get("output_status") == "matched"
                and mapping_decision.get("output_group_id") != group_id
            )
        ):
            reasons.append("REPEATED_DARK_MAPPING_CONTRACT_INVALID")
    if (
        spatial_gate_decision is not None
        and spatial_gate_decision.get("output_status") == "matched"
        and spatial_gate_decision.get("output_group_id") != group_id
    ):
        reasons.append("REPEATED_DARK_GATE_MATCHED_OTHER_GROUP")

    alternatives: list[dict[str, Any]] = []
    for raw_vote in _sequence(
        spatial_part.get("semantic_votes", []),
        f"spatial part {part_id}.semantic_votes",
    ):
        if not isinstance(raw_vote, Mapping):
            continue
        vote = raw_vote
        if not (
            vote.get("alignment_trusted") is True
            and vote.get("unique_canonical_join") is True
            and vote.get("pixel_gate_accepted") is True
            and vote.get("status") in {"matched", "review"}
            and isinstance(vote.get("canonical_group_id"), str)
            and vote.get("canonical_group_id") != group_id
            and isinstance(vote.get("view_id"), str)
        ):
            continue
        alternative_view_id = str(vote["view_id"])
        alternative_group_id = str(vote["canonical_group_id"])
        alternative_observations = [
            item
            for raw_item in _sequence(
                spatial_part.get("observations", []),
                f"spatial part {part_id}.observations",
            )
            for item in [_mapping(raw_item, f"{part_id} alternative observation")]
            if item.get("reference_view_id") == alternative_view_id
        ]
        if len(alternative_observations) != 1:
            reasons.append("REPEATED_DARK_SEMANTIC_ALTERNATIVE_NOT_DISPROVED")
            continue
        canonical_diagnostic = alternative_observations[0].get(
            "canonical_palette_diagnostic"
        )
        samples: list[Any] = []
        if isinstance(canonical_diagnostic, Mapping):
            raw_semantic_perturbations = canonical_diagnostic.get(
                "projection_perturbations"
            )
            by_offset: dict[tuple[int, int], Mapping[str, Any]] = {}
            if isinstance(raw_semantic_perturbations, Sequence) and not isinstance(
                raw_semantic_perturbations, (str, bytes)
            ):
                for raw_sample in raw_semantic_perturbations:
                    if not isinstance(raw_sample, Mapping):
                        continue
                    offset = raw_sample.get("offset_pixels")
                    if (
                        isinstance(offset, Sequence)
                        and not isinstance(offset, (str, bytes))
                        and len(offset) == 2
                        and all(
                            isinstance(value, int) and not isinstance(value, bool)
                            for value in offset
                        )
                    ):
                        by_offset[(int(offset[0]), int(offset[1]))] = raw_sample
            ordered_offsets = ((-2, 0), (2, 0), (0, -2), (0, 2))
            if set(by_offset) == set(ordered_offsets):
                samples = [
                    canonical_diagnostic.get("direct_sample"),
                    canonical_diagnostic.get("bbox_sample"),
                    *(by_offset[offset] for offset in ordered_offsets),
                ]
        sample_mappings = [
            _mapping(sample, "repeated dark semantic disproof sample")
            for sample in samples
            if isinstance(sample, Mapping)
        ]
        sample_shares = [
            _group_color_share(sample.get("group_scores"), alternative_group_id)
            for sample in sample_mappings
        ]
        semantic_status = str(vote.get("status"))
        bounded_disproof = (
            len(samples) == 6
            and len(sample_mappings) == 6
            and all(share is not None for share in sample_shares)
            and (
                (
                    semantic_status == "matched"
                    and sum(float(share) <= 0.05 for share in sample_shares) >= 5
                    and max(float(share) for share in sample_shares) <= 0.15
                )
                or (
                    semantic_status == "review"
                    and sum(float(share) <= 0.15 for share in sample_shares) >= 5
                    and max(float(share) for share in sample_shares) <= 0.25
                )
            )
        )
        if not bounded_disproof:
            reasons.append("REPEATED_DARK_SEMANTIC_ALTERNATIVE_NOT_DISPROVED")
            continue
        labels = (
            "direct",
            "bbox",
            "offset_-2_0",
            "offset_2_0",
            "offset_0_-2",
            "offset_0_2",
        )
        alternatives.append(
            {
                "view_id": alternative_view_id,
                "canonical_group_id": alternative_group_id,
                "status": semantic_status,
                "effective_confidence": vote.get("effective_confidence"),
                "sample_shares": {
                    label: share
                    for label, share in zip(labels, sample_shares, strict=True)
                },
            }
        )
    matched_alternatives = [
        alternative
        for alternative in alternatives
        if alternative["status"] == "matched"
    ]
    if len(matched_alternatives) > 1:
        reasons.append("REPEATED_DARK_TOO_MANY_MATCHED_SEMANTIC_ALTERNATIVES")
    if reasons:
        return None, sorted(set(reasons))
    assert isinstance(projected_pixels, int)
    assert direct_share is not None
    assert direct_margin is not None
    assert bbox_share is not None
    assert bbox_margin is not None
    assert isinstance(diagnostic_hash, str)
    record = {
        "part_id": part_id,
        "projected_part_pixels": projected_pixels,
        "direct_target_share": direct_share,
        "direct_target_margin": direct_margin,
        "direct_target_matching_pixels": direct_matching_pixels,
        "bbox_target_share": bbox_share,
        "bbox_target_margin": bbox_margin,
        "perturbations": sorted(
            perturbation_audits, key=lambda item: item["offset_pixels"]
        ),
        "alignment": {
            "score": diagnostic_alignment["score"],
            "projection_score": diagnostic_alignment["projection_score"],
            "projection_iou": diagnostic_alignment["projection_iou"],
            "ecc_correlation": diagnostic_alignment["ecc_correlation"],
            "ecc_status": diagnostic_alignment["ecc_status"],
        },
        "evidence_contract": "dark_foreground_diagnostic",
        "dark_diagnostic_sha256": diagnostic_hash,
        "dark_signal_share": diagnostic["dark_signal_share"],
        "dark_signal_purity": diagnostic["dark_signal_purity"],
        "core_dark_signal_share": diagnostic["core_dark_signal_share"],
        "adaptive_edge_density": diagnostic["adaptive_edge_density"],
        "dark_signal_null_margin": diagnostic["dark_signal_null_margin"],
        "semantic_alternative_disproofs": alternatives,
    }
    record["evidence_sha256"] = _canonical_sha256(record)
    return record, []


def build_quality_repair_plan(
    *,
    baseline_plan: Mapping[str, Any],
    baseline_policy_audit: Mapping[str, Any],
    quality_report: Mapping[str, Any],
    palette_fusion: Mapping[str, Any],
    spatial_report: Mapping[str, Any],
    spatial_gate_audit: Mapping[str, Any],
    mapping_consensus: Mapping[str, Any],
    geometry_risk: Mapping[str, Any],
    group_materials: Mapping[str, Any],
    registry: Mapping[str, Any],
    whitelist: Mapping[str, Any],
    mvinverse_pbr_evidence: Mapping[str, Any] | None = None,
    allow_parameter_writes: bool = True,
    material_selection_objective: str = MATERIAL_SELECTION_OBJECTIVE_SEMANTIC,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return one hash-bound repair candidate and its audit.

    A successful compilation may change many independently proven parts, but
    it is one atomic repair plan intended for at most one orchestrator retry.
    """

    registry_part_ids = _registry_part_ids(registry)
    registry_by_part = {
        _text(part.get("part_id"), "registry part_id"): part
        for part in (
            _mapping(raw_part, "registry part")
            for raw_part in _sequence(registry.get("parts"), "registry.parts")
        )
    }
    allowed_material_ids = _whitelist_ids(whitelist)
    assignments, baseline_by_part = _baseline_assignments(
        baseline_plan,
        registry_part_ids=registry_part_ids,
        whitelist_ids=allowed_material_ids,
    )
    if not isinstance(allow_parameter_writes, bool):
        raise QualityRepairError("allow_parameter_writes must be a boolean")
    if material_selection_objective not in MATERIAL_SELECTION_OBJECTIVES:
        raise QualityRepairError(
            "material_selection_objective must be one of "
            f"{sorted(MATERIAL_SELECTION_OBJECTIVES)}"
        )
    visual_similarity_first = (
        material_selection_objective == MATERIAL_SELECTION_OBJECTIVE_VISUAL
    )
    if not allow_parameter_writes:
        for assignment in assignments:
            part_id = _text(
                assignment.get("part_id"),
                "immutable quality-repair assignment part_id",
            )
            parameters = assignment.get("parameters")
            if isinstance(parameters, Mapping) and parameters:
                raise QualityRepairError(
                    "immutable MDL selection received inherited parameter "
                    f"overrides for part {part_id}"
                )
            raw_subsets = assignment.get("face_subsets", [])
            if isinstance(raw_subsets, Sequence) and not isinstance(
                raw_subsets, (str, bytes)
            ):
                for raw_subset in raw_subsets:
                    subset = raw_subset if isinstance(raw_subset, Mapping) else {}
                    subset_parameters = subset.get("parameters")
                    if isinstance(subset_parameters, Mapping) and subset_parameters:
                        raise QualityRepairError(
                            "immutable MDL selection received inherited face-subset "
                            f"parameter overrides for part {part_id}"
                        )
    _validate_policy_audit(
        baseline_plan,
        baseline_policy_audit,
        registry_part_count=len(registry_part_ids),
    )
    canonical_groups, view_group_maps = _canonical_groups(palette_fusion)
    reference_evidence = _reference_evidence(spatial_report)
    canonical_reference_evidence = _all_reference_evidence(spatial_report)
    alignment_audits = _trusted_alignment_audits(
        spatial_report,
        reference_evidence=reference_evidence,
    )
    (
        repairable_groups,
        dominant_residual_groups,
        group_diagnostics,
        view_diagnostics,
        top_reason_codes,
    ) = _trusted_missing_groups(
        quality_report,
        canonical_groups=canonical_groups,
        view_group_maps=view_group_maps,
        reference_evidence=reference_evidence,
    )
    (
        dark_residual_groups,
        dark_group_diagnostics,
        dark_view_diagnostics,
    ) = _trusted_dark_foreground_residual_groups(
        quality_report,
        canonical_groups=canonical_groups,
        reference_evidence=reference_evidence,
    )
    dark_group_diagnostics_by_id = {
        str(item["canonical_group_id"]): item for item in dark_group_diagnostics
    }
    for group_diagnostic in group_diagnostics:
        dark_diagnostic = dark_group_diagnostics_by_id[
            str(group_diagnostic["canonical_group_id"])
        ]
        group_diagnostic.update(
            {
                key: copy.deepcopy(value)
                for key, value in dark_diagnostic.items()
                if key != "canonical_group_id"
            }
        )
    if dark_residual_groups:
        top_reason_codes = []
    (
        confirmed_materials,
        material_unavailable,
        provisional_material_groups,
    ) = _confirmed_group_materials(
        group_materials,
        canonical_groups=canonical_groups,
        whitelist_ids=allowed_material_ids,
        allow_unconfirmed_visual_tournament_seeds=not allow_parameter_writes,
    )
    collapse_recovery_excluded_group_ids = (
        _policy_collapse_recovery_group_ids(baseline_policy_audit)
    )
    for group_id in collapse_recovery_excluded_group_ids:
        confirmed_materials.pop(group_id, None)
        provisional_material_groups.discard(group_id)
        material_unavailable[group_id] = "MATERIAL_COLLAPSE_RECOVERY_REQUIRED"
    if allow_parameter_writes:
        parameterizations, parameterization_skips = _mvinverse_parameterizations(
            group_materials=group_materials,
            mvinverse_pbr_evidence=mvinverse_pbr_evidence,
            allowed_material_ids=allowed_material_ids,
            palette_fusion=palette_fusion,
            key_by_group=True,
            excluded_group_ids=collapse_recovery_excluded_group_ids,
        )
    else:
        parameterizations = {}
        parameterization_skips = [
            {
                "group_id": group_id,
                "reason_code": "SELECTED_MDL_LIBRARY_DEFAULTS_REQUIRED",
            }
            for group_id in sorted(canonical_groups)
        ]
    geometry_risks = _geometry_risks(geometry_risk, registry_part_ids=registry_part_ids)
    spatial_parts = _spatial_parts(spatial_report)
    gate_decisions = _spatial_gate_decisions(spatial_gate_audit)
    mapping_decisions = _mapping_decisions(mapping_consensus)
    if set(spatial_parts) != set(registry_part_ids):
        raise QualityRepairError("spatial report does not exactly cover the registry")
    input_hashes = {
        "baseline_plan_sha256": _canonical_sha256(baseline_plan),
        "baseline_policy_audit_sha256": _canonical_sha256(baseline_policy_audit),
        "quality_report_sha256": _canonical_sha256(quality_report),
        "palette_fusion_sha256": _canonical_sha256(palette_fusion),
        "spatial_report_sha256": _canonical_sha256(spatial_report),
        "spatial_gate_audit_sha256": _canonical_sha256(spatial_gate_audit),
        "mapping_consensus_sha256": _canonical_sha256(mapping_consensus),
        "geometry_risk_sha256": _canonical_sha256(geometry_risk),
        "group_materials_sha256": _canonical_sha256(group_materials),
        "registry_sha256": _canonical_sha256(registry),
        "whitelist_sha256": _canonical_sha256(whitelist),
    }
    if mvinverse_pbr_evidence is not None:
        input_hashes["mvinverse_pbr_evidence_sha256"] = _canonical_sha256(
            mvinverse_pbr_evidence
        )
    if visual_similarity_first:
        input_hashes["material_selection_objective"] = material_selection_objective
    dominant_assembly_input_hashes = {
        key: input_hashes[key]
        for key in (
            "baseline_plan_sha256",
            "quality_report_sha256",
            "palette_fusion_sha256",
            "spatial_report_sha256",
            "spatial_gate_audit_sha256",
            "mapping_consensus_sha256",
            "geometry_risk_sha256",
            "group_materials_sha256",
            "registry_sha256",
        )
    }
    spatial_policy = _mapping(spatial_report.get("policy"), "spatial_report.policy")
    raw_semantic_confidence = spatial_policy.get("minimum_semantic_confidence")
    if (
        isinstance(raw_semantic_confidence, (int, float))
        and not isinstance(raw_semantic_confidence, bool)
        and math.isfinite(float(raw_semantic_confidence))
        and 0.0 <= float(raw_semantic_confidence) <= 1.0
    ):
        minimum_semantic_confidence = float(raw_semantic_confidence)
    else:
        # Reports produced before semantic localization was added may still
        # contain usable spatial projections.  Disable only the semantic lane;
        # do not turn a legacy, otherwise valid no-op input into an exception.
        minimum_semantic_confidence = math.inf

    multiview_group_ids = {
        group_id
        for group_id, group in canonical_groups.items()
        if group.get("singleton") is not True
        and isinstance(group.get("distinct_view_count"), int)
        and not isinstance(group.get("distinct_view_count"), bool)
        and int(group["distinct_view_count"]) >= 2
    }
    proposals: dict[str, list[dict[str, Any]]] = defaultdict(list)
    pending_anchored_proposals: dict[str, list[dict[str, Any]]] = defaultdict(list)
    pending_source_identity_proposals: dict[str, list[dict[str, Any]]] = defaultdict(
        list
    )
    pending_dark_proposals: dict[str, list[dict[str, Any]]] = defaultdict(list)
    skips: list[dict[str, Any]] = []
    for part_id in sorted(registry_part_ids):
        gate = gate_decisions.get(part_id)
        mapping_decision = mapping_decisions.get(part_id)
        spatial_part = spatial_parts.get(part_id)
        if spatial_part is None:
            continue
        evidence_group_ids: set[str] = set()
        for raw_observation in _sequence(
            spatial_part.get("observations", []),
            f"spatial part {part_id}.observations",
        ):
            observation = _mapping(
                raw_observation, f"spatial part {part_id}.observation"
            )
            canonical_group_id = observation.get("canonical_group_id")
            if isinstance(canonical_group_id, str):
                evidence_group_ids.add(canonical_group_id)
            diagnostic = observation.get("small_part_diagnostic")
            if isinstance(diagnostic, Mapping) and isinstance(
                diagnostic.get("canonical_group_id"), str
            ):
                evidence_group_ids.add(str(diagnostic["canonical_group_id"]))
            canonical_diagnostic = observation.get("canonical_palette_diagnostic")
            if isinstance(canonical_diagnostic, Mapping) and isinstance(
                canonical_diagnostic.get("canonical_group_id"), str
            ):
                evidence_group_ids.add(str(canonical_diagnostic["canonical_group_id"]))
            dark_diagnostic = observation.get("dark_foreground_diagnostic")
            if isinstance(dark_diagnostic, Mapping) and isinstance(
                dark_diagnostic.get("canonical_group_id"), str
            ):
                evidence_group_ids.add(str(dark_diagnostic["canonical_group_id"]))
        if gate is not None and isinstance(gate.get("output_group_id"), str):
            evidence_group_ids.add(str(gate["output_group_id"]))
        candidate_group_ids = sorted(
            evidence_group_ids
            & multiview_group_ids
            & (
                set(repairable_groups)
                | set(dominant_residual_groups)
                | set(dark_residual_groups)
            )
        )
        if not candidate_group_ids:
            continue
        assignment = baseline_by_part[part_id]
        baseline_reasons: list[str] = []
        provenance = assignment.get("provenance")
        tier = provenance.get("tier") if isinstance(provenance, Mapping) else None
        if assignment.get("status") != POLICY_FALLBACK_STATUS:
            baseline_reasons.append("BASELINE_STATUS_NOT_POLICY_FALLBACK")
        confidence = assignment.get("confidence")
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or float(confidence) != 0.0
        ):
            baseline_reasons.append("BASELINE_FALLBACK_CONFIDENCE_NOT_ZERO")
        if assignment.get("evidence_views") != []:
            baseline_reasons.append("BASELINE_FALLBACK_EVIDENCE_NOT_EMPTY")
        if tier not in NEUTRAL_FALLBACK_TIERS:
            baseline_reasons.append("BASELINE_TIER_NOT_NEUTRAL_DEFAULT")
        if (
            not isinstance(provenance, Mapping)
            or provenance.get("output_confidence_basis")
            != POLICY_FALLBACK_CONFIDENCE_BASIS
        ):
            baseline_reasons.append("BASELINE_FALLBACK_CONFIDENCE_BASIS_INVALID")
        if (
            isinstance(provenance, Mapping)
            and provenance.get("canonical_group_id") is not None
        ):
            baseline_reasons.append(AUTHORITATIVE_CANONICAL_GROUP_LOCK_REASON)
        if "face_subsets" in assignment:
            baseline_reasons.append("BASELINE_HAS_FACE_SUBSETS")
        if "parameters" in assignment:
            baseline_reasons.append("BASELINE_HAS_PARAMETERS")
        if geometry_risks[part_id]:
            baseline_reasons.append("GEOMETRY_MULTI_MATERIAL_RISK")

        for group_id in candidate_group_ids:
            part_reasons = list(baseline_reasons)
            material_id = confirmed_materials.get(group_id)
            if material_id is None:
                part_reasons.append(
                    material_unavailable.get(
                        group_id, "CONFIRMED_GROUP_MATERIAL_UNAVAILABLE"
                    )
                )
            elif (
                material_id == assignment.get("material_id")
                and assignment.get("apply_action") != SOURCE_VISUAL_PRESERVE_ACTION
            ):
                part_reasons.append("MATERIAL_ALREADY_ASSIGNED")

            localization_attempts: list[
                tuple[str, list[str], list[str], list[str]]
            ] = []
            single_spatial_anchor_views: list[str] = []
            anchored_single_views: list[str] = []
            anchored_single_reasons: list[str] = ["ORDINARY_DEFICIT_UNAVAILABLE"]
            anchored_semantic_override_views: list[str] = []
            semantic_anchor_view_ids: list[str] = []
            dark_spatial_audit: dict[str, Any] | None = None
            ordinary_deficit = repairable_groups.get(group_id)
            if ordinary_deficit is not None:
                stable_views, stable_reasons = _stable_spatial_support_views(
                    part_id=part_id,
                    group_id=group_id,
                    spatial_part=spatial_part,
                    spatial_gate_decision=gate,
                    mapping_decision=mapping_decision,
                    reference_evidence=reference_evidence,
                    deficit=ordinary_deficit,
                )
                localization_attempts.append(
                    (
                        "stable_spatial_multiview",
                        stable_views,
                        stable_reasons,
                        [],
                    )
                )
                (
                    bounded_views,
                    bounded_reasons,
                    bounded_semantic_override_views,
                ) = _bounded_spatial_support_views(
                    part_id=part_id,
                    group_id=group_id,
                    spatial_part=spatial_part,
                    spatial_gate_decision=gate,
                    mapping_decision=mapping_decision,
                    reference_evidence=reference_evidence,
                    spatial_policy=spatial_policy,
                    minimum_semantic_confidence=minimum_semantic_confidence,
                    deficit=ordinary_deficit,
                )
                localization_attempts.append(
                    (
                        "bounded_spatial_multiview",
                        bounded_views,
                        bounded_reasons,
                        bounded_semantic_override_views,
                    )
                )
                (
                    semantic_review_views,
                    semantic_review_reasons,
                ) = _qa_confirmed_multiview_semantic_review_support_views(
                    part_id=part_id,
                    group_id=group_id,
                    spatial_part=spatial_part,
                    spatial_gate_decision=gate,
                    mapping_decision=mapping_decision,
                    reference_evidence=reference_evidence,
                    spatial_policy=spatial_policy,
                    deficit=ordinary_deficit,
                )
                localization_attempts.append(
                    (
                        QA_CONFIRMED_MULTIVIEW_SEMANTIC_REVIEW_LANE,
                        semantic_review_views,
                        semantic_review_reasons,
                        [],
                    )
                )
                (
                    source_identity_views,
                    source_identity_reasons,
                ) = _source_identity_diagnostic_support_views(
                    part_id=part_id,
                    group_id=group_id,
                    spatial_part=spatial_part,
                    spatial_gate_decision=gate,
                    mapping_decision=mapping_decision,
                    spatial_policy=spatial_policy,
                    minimum_semantic_confidence=minimum_semantic_confidence,
                    deficit=ordinary_deficit,
                    reference_evidence=reference_evidence,
                )
                localization_attempts.append(
                    (
                        SOURCE_IDENTITY_ANCHORED_DIAGNOSTIC_LANE,
                        source_identity_views,
                        source_identity_reasons,
                        [],
                    )
                )
                (
                    single_views,
                    single_reasons,
                    single_semantic_override_views,
                    single_spatial_anchor_views,
                ) = _single_view_spatial_support_views(
                    part_id=part_id,
                    group_id=group_id,
                    spatial_part=spatial_part,
                    spatial_gate_decision=gate,
                    mapping_decision=mapping_decision,
                    spatial_policy=spatial_policy,
                    minimum_semantic_confidence=minimum_semantic_confidence,
                    deficit=ordinary_deficit,
                    alignment_audits=alignment_audits,
                    reference_evidence=reference_evidence,
                )
                localization_attempts.append(
                    (
                        (
                            SPATIAL_ANCHOR_SINGLE_VIEW_LANE
                            if single_spatial_anchor_views
                            else (
                                SEMANTIC_OVERRIDE_SINGLE_VIEW_LANE
                                if single_semantic_override_views
                                else "exact_spatial_single_qa_view"
                            )
                        ),
                        single_views,
                        single_reasons,
                        single_semantic_override_views,
                    )
                )
                (
                    anchored_single_views,
                    anchored_single_reasons,
                    anchored_semantic_override_views,
                ) = _anchored_single_view_spatial_support_views(
                    part_id=part_id,
                    group_id=group_id,
                    spatial_part=spatial_part,
                    spatial_gate_decision=gate,
                    mapping_decision=mapping_decision,
                    spatial_policy=spatial_policy,
                    minimum_semantic_confidence=minimum_semantic_confidence,
                    deficit=ordinary_deficit,
                    alignment_audits=alignment_audits,
                    reference_evidence=reference_evidence,
                )
            accepted_localization = next(
                (
                    (lane, support_views, semantic_override_views)
                    for (
                        lane,
                        support_views,
                        reasons,
                        semantic_override_views,
                    ) in localization_attempts
                    if not reasons
                ),
                None,
            )
            residual_deficit = dominant_residual_groups.get(group_id)
            if accepted_localization is None and residual_deficit is not None:
                (
                    dominant_residual_views,
                    dominant_residual_reasons,
                    dominant_semantic_override_views,
                    semantic_anchor_view_ids,
                ) = _dominant_residual_spatial_support_views(
                    part_id=part_id,
                    group_id=group_id,
                    spatial_part=spatial_part,
                    spatial_gate_decision=gate,
                    mapping_decision=mapping_decision,
                    minimum_semantic_confidence=minimum_semantic_confidence,
                    deficit=residual_deficit,
                    alignment_audits=alignment_audits,
                    spatial_policy=spatial_policy,
                    reference_evidence=reference_evidence,
                )
                localization_attempts.append(
                    (
                        DOMINANT_RESIDUAL_SINGLE_VIEW_LANE,
                        dominant_residual_views,
                        dominant_residual_reasons,
                        dominant_semantic_override_views,
                    )
                )
                if not dominant_residual_reasons:
                    accepted_localization = (
                        DOMINANT_RESIDUAL_SINGLE_VIEW_LANE,
                        dominant_residual_views,
                        dominant_semantic_override_views,
                    )
            dark_deficit = dark_residual_groups.get(group_id)
            if accepted_localization is None and dark_deficit is not None:
                (
                    dark_views,
                    dark_reasons,
                    dark_spatial_audit,
                ) = _dark_foreground_spatial_support(
                    part_id=part_id,
                    group_id=group_id,
                    canonical_group=canonical_groups[group_id],
                    spatial_part=spatial_part,
                    spatial_gate_decision=gate,
                    mapping_decision=mapping_decision,
                    deficit=dark_deficit,
                    alignment_audits=alignment_audits,
                    reference_evidence=reference_evidence,
                    canonical_reference_evidence=canonical_reference_evidence,
                    minimum_semantic_confidence=minimum_semantic_confidence,
                )
                localization_attempts.append(
                    (
                        DARK_FOREGROUND_RESIDUAL_LANE,
                        dark_views,
                        dark_reasons,
                        [],
                    )
                )
                if not dark_reasons:
                    accepted_localization = (
                        DARK_FOREGROUND_RESIDUAL_LANE,
                        dark_views,
                        [],
                    )
                else:
                    (
                        multiview_dark_views,
                        multiview_dark_reasons,
                        multiview_dark_semantic_override_views,
                        multiview_dark_spatial_audit,
                    ) = _multiview_dark_identity_support(
                        part_id=part_id,
                        group_id=group_id,
                        canonical_group=canonical_groups[group_id],
                        spatial_part=spatial_part,
                        spatial_gate_decision=gate,
                        mapping_decision=mapping_decision,
                        deficit=dark_deficit,
                        reference_evidence=reference_evidence,
                        spatial_policy=spatial_policy,
                        minimum_semantic_confidence=minimum_semantic_confidence,
                    )
                    localization_attempts.append(
                        (
                            MULTIVIEW_DARK_IDENTITY_LANE,
                            multiview_dark_views,
                            multiview_dark_reasons,
                            multiview_dark_semantic_override_views,
                        )
                    )
                    if not multiview_dark_reasons:
                        accepted_localization = (
                            MULTIVIEW_DARK_IDENTITY_LANE,
                            multiview_dark_views,
                            multiview_dark_semantic_override_views,
                        )
                        dark_spatial_audit = multiview_dark_spatial_audit
            pending_multiview_anchor = (
                ordinary_deficit is not None
                and accepted_localization is None
                and not anchored_single_reasons
            )
            if pending_multiview_anchor:
                accepted_localization = (
                    ANCHORED_SINGLE_VIEW_LANE,
                    anchored_single_views,
                    anchored_semantic_override_views,
                )
            if accepted_localization is None:
                part_reasons.extend(
                    reason
                    for _, _, reasons, _ in localization_attempts
                    for reason in reasons
                )
                part_reasons.extend(anchored_single_reasons)
            if part_reasons:
                skips.append(
                    {
                        "part_id": part_id,
                        "canonical_group_id": group_id,
                        "reason_codes": sorted(set(part_reasons)),
                    }
                )
                continue
            assert accepted_localization is not None
            (
                localization_lane,
                support_views,
                semantic_conflict_override_view_ids,
            ) = accepted_localization
            support_records = [reference_evidence[view_id] for view_id in support_views]
            assert material_id is not None
            output_material_id = material_id
            parameters: dict[str, Any] | None = None
            parameter_audit: dict[str, Any] | None = None
            parameterization = parameterizations.get(group_id)
            if (
                parameterization is not None
                and localization_lane not in _DARK_RESIDUAL_LANES
            ):
                raw_parameters, raw_parameter_audit = parameterization
                if raw_parameter_audit.get("group_id") == group_id:
                    parameters = copy.deepcopy(dict(raw_parameters))
                    parameter_audit = copy.deepcopy(dict(raw_parameter_audit))
            if (
                output_material_id == assignment.get("material_id")
                and parameters is None
            ):
                skips.append(
                    {
                        "part_id": part_id,
                        "canonical_group_id": group_id,
                        "reason_codes": ["PARAMETERIZED_MATERIAL_ALREADY_ASSIGNED"],
                    }
                )
                continue
            proposal = {
                "part_id": part_id,
                "canonical_group_id": group_id,
                "material_id": output_material_id,
                "supporting_view_ids": support_views,
                "supporting_content_cluster_ids": sorted(
                    {item["content_cluster_id"] for item in support_records}
                ),
                "supporting_pose_cluster_ids": sorted(
                    {item["pose_cluster_id"] for item in support_records}
                ),
                "_localization_lane": localization_lane,
                "_semantic_conflict_override_view_ids": (
                    semantic_conflict_override_view_ids
                ),
                "_semantic_anchor_view_ids": semantic_anchor_view_ids,
                "_spatial_anchor_view_ids": (
                    single_spatial_anchor_views
                    if localization_lane == SPATIAL_ANCHOR_SINGLE_VIEW_LANE
                    else []
                ),
                "_confirmed_source_material_id": material_id,
                "_provisional_material_candidate": (
                    group_id in provisional_material_groups
                ),
                "_parameters": parameters,
                "_parameter_audit": parameter_audit,
                "_dark_spatial_audit": (
                    copy.deepcopy(dark_spatial_audit)
                    if localization_lane in _DARK_RESIDUAL_LANES
                    else None
                ),
                "_dark_residual_support": (
                    copy.deepcopy(dark_residual_groups[group_id]["supporting_views"][0])
                    if localization_lane in _DARK_RESIDUAL_LANES
                    else None
                ),
            }
            target = (
                pending_dark_proposals
                if localization_lane in _DARK_RESIDUAL_LANES
                else (
                    pending_source_identity_proposals
                    if localization_lane == SOURCE_IDENTITY_ANCHORED_DIAGNOSTIC_LANE
                    else (
                        pending_anchored_proposals
                        if pending_multiview_anchor
                        else proposals
                    )
                )
            )
            target[part_id].append(proposal)

    # A repeated-geometry dark cohort is compiled atomically from an ordinary
    # single-view group-recall deficit.  No member can bootstrap another:
    # registry completeness, source-display equivalence, every member's
    # projection, and the complete cohort contribution are all checked before
    # any proposal is emitted.
    repeated_geometry_dark_cohorts: list[dict[str, Any]] = []
    unique_multiview_black_group_ids = [
        group_id
        for group_id, group in sorted(canonical_groups.items())
        if str(group.get("base_color", "")).strip().casefold() == "black"
        and group.get("singleton") is not True
        and isinstance(group.get("distinct_view_count"), int)
        and not isinstance(group.get("distinct_view_count"), bool)
        and int(group["distinct_view_count"]) >= 2
        and group_id in repairable_groups
        and repairable_groups[group_id].get("single_view_spatial_repairable") is True
    ]
    if len(unique_multiview_black_group_ids) == 1:
        repeated_group_id = unique_multiview_black_group_ids[0]
        repeated_deficit = repairable_groups[repeated_group_id]
        repeated_support, repeated_support_reasons = _ordinary_black_budget_support(
            quality_report=quality_report,
            deficit=repeated_deficit,
            group_id=repeated_group_id,
        )
        if repeated_support is not None and not repeated_support_reasons:
            repeated_view_id = str(repeated_support["reference_view_id"])
            repeated_reference = reference_evidence.get(repeated_view_id)
            if (
                repeated_reference is None
                or repeated_reference["raw_sha256"]
                != repeated_support["reference_sha256"]
            ):
                repeated_support = None
        if repeated_support is not None and not repeated_support_reasons:
            repeated_view_id = str(repeated_support["reference_view_id"])
            for raw_cohort in _repeated_geometry_registry_cohorts(registry):
                cohort = copy.deepcopy(raw_cohort)
                cohort_part_ids = list(cohort["cohort_part_ids"])
                cohort_reasons: list[str] = []
                member_audits: list[dict[str, Any]] = []
                for cohort_part_id in cohort_part_ids:
                    assignment = baseline_by_part[cohort_part_id]
                    provenance = assignment.get("provenance")
                    confidence = assignment.get("confidence")
                    if (
                        assignment.get("status") != POLICY_FALLBACK_STATUS
                        or isinstance(confidence, bool)
                        or not isinstance(confidence, (int, float))
                        or float(confidence) != 0.0
                        or assignment.get("evidence_views") != []
                        or not isinstance(provenance, Mapping)
                        or provenance.get("tier") not in NEUTRAL_FALLBACK_TIERS
                        or provenance.get("output_confidence_basis")
                        != POLICY_FALLBACK_CONFIDENCE_BASIS
                        or provenance.get("canonical_group_id") is not None
                        or "face_subsets" in assignment
                        or "parameters" in assignment
                        or geometry_risks[cohort_part_id]
                    ):
                        cohort_reasons.append(
                            "REPEATED_DARK_COHORT_MEMBER_NOT_SAFE_NEUTRAL_FALLBACK"
                        )
                        continue
                    member_audit, member_reasons = _repeated_dark_member_audit(
                        part_id=cohort_part_id,
                        group_id=repeated_group_id,
                        canonical_group=canonical_groups[repeated_group_id],
                        view_id=repeated_view_id,
                        spatial_part=spatial_parts[cohort_part_id],
                        spatial_gate_decision=gate_decisions.get(cohort_part_id),
                        mapping_decision=mapping_decisions.get(cohort_part_id),
                        alignment_audits=alignment_audits,
                        reference_evidence=reference_evidence,
                        canonical_reference_evidence=canonical_reference_evidence,
                    )
                    if member_reasons:
                        cohort_reasons.extend(member_reasons)
                    elif member_audit is not None:
                        member_audits.append(member_audit)

                semantic_alternative_groups = [
                    str(disproof["canonical_group_id"])
                    for member in member_audits
                    for disproof in member["semantic_alternative_disproofs"]
                    if disproof["status"] == "matched"
                ]
                semantic_alternative_views = [
                    str(disproof["view_id"])
                    for member in member_audits
                    for disproof in member["semantic_alternative_disproofs"]
                    if disproof["status"] == "matched"
                ]
                if len(semantic_alternative_groups) != len(
                    set(semantic_alternative_groups)
                ) or len(semantic_alternative_views) != len(
                    set(semantic_alternative_views)
                ):
                    cohort_reasons.append(
                        "REPEATED_DARK_COHORT_SEMANTIC_CONFLICT_CORROBORATED"
                    )
                total_projected = sum(
                    int(member["projected_part_pixels"]) for member in member_audits
                )
                total_matching = sum(
                    int(member["direct_target_matching_pixels"])
                    for member in member_audits
                )
                if len(member_audits) != len(cohort_part_ids):
                    cohort_reasons.append(
                        "REPEATED_DARK_COHORT_MEMBER_EVIDENCE_INCOMPLETE"
                    )
                if not (
                    int(repeated_support["minimum_contribution_pixels"])
                    <= total_projected
                    <= int(repeated_support["maximum_contribution_pixels"])
                ):
                    cohort_reasons.append(
                        "REPEATED_DARK_COHORT_CONTRIBUTION_OUTSIDE_BUDGET"
                    )
                cohort_id_payload = {
                    "canonical_group_id": repeated_group_id,
                    "reference_view_id": repeated_view_id,
                    "geometry_signature_sha256": cohort["geometry_signature_sha256"],
                    "source_visual_stable_properties_signature_sha256": cohort[
                        "source_visual_stable_properties_signature_sha256"
                    ],
                    "cohort_part_ids": cohort_part_ids,
                }
                cohort_id = _canonical_sha256(cohort_id_payload)
                selected = not cohort_reasons
                cohort_audit = {
                    "cohort_id": cohort_id,
                    "canonical_group_id": repeated_group_id,
                    "reference_view_id": repeated_view_id,
                    "geometry_signature": {
                        **copy.deepcopy(cohort["geometry_signature"]),
                        "signature_sha256": cohort["geometry_signature_sha256"],
                    },
                    "source_visual_stable_properties_signature_sha256": cohort[
                        "source_visual_stable_properties_signature_sha256"
                    ],
                    "registry_part_count": cohort["registry_part_count"],
                    "cohort_size": cohort["cohort_size"],
                    "registry_fraction": cohort["registry_fraction"],
                    "cohort_part_ids": cohort_part_ids,
                    "required_render_share": repeated_support["required_render_share"],
                    "observed_render_share": repeated_support["observed_render_share"],
                    "render_foreground_pixels": repeated_support[
                        "render_foreground_pixels"
                    ],
                    "budget_pixels": repeated_support["budget_pixels"],
                    "minimum_contribution_pixels": repeated_support[
                        "minimum_contribution_pixels"
                    ],
                    "maximum_contribution_pixels": repeated_support[
                        "maximum_contribution_pixels"
                    ],
                    "total_projected_part_pixels": total_projected,
                    "total_direct_target_matching_pixels": total_matching,
                    "selected": selected,
                    "reason_codes": sorted(set(cohort_reasons)),
                    "members": sorted(member_audits, key=lambda item: item["part_id"]),
                }
                repeated_geometry_dark_cohorts.append(cohort_audit)
                if not selected:
                    for cohort_part_id in cohort_part_ids:
                        skips.append(
                            {
                                "part_id": cohort_part_id,
                                "canonical_group_id": repeated_group_id,
                                "reason_codes": sorted(set(cohort_reasons)),
                            }
                        )
                    continue

                material_id = confirmed_materials.get(repeated_group_id)
                if material_id is None:
                    for cohort_part_id in cohort_part_ids:
                        skips.append(
                            {
                                "part_id": cohort_part_id,
                                "canonical_group_id": repeated_group_id,
                                "reason_codes": [
                                    material_unavailable.get(
                                        repeated_group_id,
                                        "CONFIRMED_GROUP_MATERIAL_UNAVAILABLE",
                                    )
                                ],
                            }
                        )
                    cohort_audit["selected"] = False
                    cohort_audit["reason_codes"] = [
                        "CONFIRMED_GROUP_MATERIAL_UNAVAILABLE"
                    ]
                    continue
                support_record = reference_evidence[repeated_view_id]
                if any(
                    cohort_part_id in proposals for cohort_part_id in cohort_part_ids
                ):
                    cohort_audit["selected"] = False
                    cohort_audit["reason_codes"] = [
                        "REPEATED_DARK_COHORT_OVERLAPS_EXISTING_PROPOSAL"
                    ]
                    for cohort_part_id in cohort_part_ids:
                        skips.append(
                            {
                                "part_id": cohort_part_id,
                                "canonical_group_id": repeated_group_id,
                                "reason_codes": list(cohort_audit["reason_codes"]),
                            }
                        )
                    continue
                for member_audit in member_audits:
                    cohort_part_id = str(member_audit["part_id"])
                    proposals[cohort_part_id].append(
                        {
                            "part_id": cohort_part_id,
                            "canonical_group_id": repeated_group_id,
                            "material_id": material_id,
                            "supporting_view_ids": [repeated_view_id],
                            "supporting_content_cluster_ids": [
                                support_record["content_cluster_id"]
                            ],
                            "supporting_pose_cluster_ids": [
                                support_record["pose_cluster_id"]
                            ],
                            "_localization_lane": (
                                REPEATED_GEOMETRY_DARK_RESIDUAL_LANE
                            ),
                            "_semantic_conflict_override_view_ids": [],
                            "_semantic_anchor_view_ids": [],
                            "_spatial_anchor_view_ids": [],
                            "_confirmed_source_material_id": material_id,
                            "_provisional_material_candidate": False,
                            "_parameters": None,
                            "_parameter_audit": None,
                            "_dark_spatial_audit": None,
                            "_dark_residual_support": None,
                            "_repeated_dark_cohort_audit": copy.deepcopy(cohort_audit),
                            "_repeated_dark_member_audit": copy.deepcopy(member_audit),
                        }
                    )

    selected_repeated_cohorts = [
        cohort
        for cohort in repeated_geometry_dark_cohorts
        if cohort.get("selected") is True and cohort.get("reason_codes") == []
    ]
    if len(selected_repeated_cohorts) > 1:
        ambiguous_cohort_ids = {
            str(cohort["cohort_id"]) for cohort in selected_repeated_cohorts
        }
        for cohort in selected_repeated_cohorts:
            cohort["selected"] = False
            cohort["reason_codes"] = ["REPEATED_DARK_COHORT_LOCALIZATION_NOT_UNIQUE"]
        for proposal_part_id, part_proposals in list(proposals.items()):
            kept = [
                proposal
                for proposal in part_proposals
                if not (
                    proposal.get("_localization_lane")
                    == REPEATED_GEOMETRY_DARK_RESIDUAL_LANE
                    and isinstance(proposal.get("_repeated_dark_cohort_audit"), Mapping)
                    and str(proposal["_repeated_dark_cohort_audit"].get("cohort_id"))
                    in ambiguous_cohort_ids
                )
            ]
            if len(kept) != len(part_proposals):
                skips.append(
                    {
                        "part_id": proposal_part_id,
                        "canonical_group_id": (
                            unique_multiview_black_group_ids[0]
                            if unique_multiview_black_group_ids
                            else None
                        ),
                        "reason_codes": [
                            "REPEATED_DARK_COHORT_LOCALIZATION_NOT_UNIQUE"
                        ],
                    }
                )
            if kept:
                proposals[proposal_part_id] = kept
            else:
                proposals.pop(proposal_part_id, None)

    # A silhouette-limited dominant residual is intentionally repaired by one
    # uniquely localized large part.  Multiple independently plausible parts
    # would make the amount of material to add ambiguous and could overshoot
    # the trusted reference mass, so that group fails closed.
    residual_parts_by_group: dict[str, set[str]] = defaultdict(set)
    for proposal_part_id, part_proposals in proposals.items():
        for proposal in part_proposals:
            if proposal.get("_localization_lane") == DOMINANT_RESIDUAL_SINGLE_VIEW_LANE:
                residual_parts_by_group[str(proposal["canonical_group_id"])].add(
                    proposal_part_id
                )
    ambiguous_residual_groups = {
        group_id
        for group_id, part_ids in residual_parts_by_group.items()
        if len(part_ids) != 1
    }
    if ambiguous_residual_groups:
        for proposal_part_id, part_proposals in list(proposals.items()):
            kept: list[dict[str, Any]] = []
            for proposal in part_proposals:
                group_id = str(proposal["canonical_group_id"])
                if (
                    group_id in ambiguous_residual_groups
                    and proposal.get("_localization_lane")
                    == DOMINANT_RESIDUAL_SINGLE_VIEW_LANE
                ):
                    skips.append(
                        {
                            "part_id": proposal_part_id,
                            "canonical_group_id": group_id,
                            "reason_codes": [
                                "DOMINANT_RESIDUAL_PART_LOCALIZATION_NOT_UNIQUE"
                            ],
                        }
                    )
                else:
                    kept.append(proposal)
            if kept:
                proposals[proposal_part_id] = kept
            else:
                proposals.pop(proposal_part_id, None)

    # The anchored-single lane is intentionally resolved only after every
    # ordinary proposal is known.  A provisional part cannot bootstrap itself
    # or another provisional part: its anchor must be an unambiguous,
    # independently accepted stable/bounded proposals whose union covers the
    # complete trusted QA deficit for that same canonical group.
    anchors_by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for anchor_part_id, anchor_proposals in sorted(proposals.items()):
        if len(anchor_proposals) != 1:
            continue
        anchor = anchor_proposals[0]
        anchor_group_id = str(anchor["canonical_group_id"])
        if anchor.get("_localization_lane") not in {
            "stable_spatial_multiview",
            "bounded_spatial_multiview",
        }:
            continue
        anchors_by_group[anchor_group_id].append(
            {
                "part_id": anchor_part_id,
                "supporting_view_ids": sorted(
                    str(view_id) for view_id in anchor["supporting_view_ids"]
                ),
            }
        )

    for part_id, pending_proposals in sorted(pending_anchored_proposals.items()):
        for proposal in pending_proposals:
            group_id = str(proposal["canonical_group_id"])
            anchors = anchors_by_group.get(group_id, [])
            deficit_view_ids = {
                str(item["reference_view_id"])
                for item in _sequence(
                    repairable_groups[group_id].get("supporting_views"),
                    f"deficit {group_id}.supporting_views",
                )
                if isinstance(item, Mapping)
                and isinstance(item.get("reference_view_id"), str)
            }
            covered_view_ids = {
                str(view_id)
                for anchor in anchors
                for view_id in anchor["supporting_view_ids"]
            }
            if not anchors or covered_view_ids != deficit_view_ids:
                skips.append(
                    {
                        "part_id": part_id,
                        "canonical_group_id": group_id,
                        "reason_codes": ["MULTIVIEW_ANCHOR_PROPOSAL_MISSING"],
                    }
                )
                continue
            proposal["anchor_part_ids"] = sorted(
                str(anchor["part_id"]) for anchor in anchors
            )
            proposal["anchor_supporting_view_ids"] = sorted(covered_view_ids)
            proposals[part_id].append(proposal)

    # A second curved/segmented child of the same authored subassembly can be
    # too small for three independent semantic votes even when one sibling is
    # fully localized.  Permit exactly one diagnostic-only sibling when the
    # accepted anchor covers the complete QA deficit and the registry proves a
    # rare, complete source-appearance cohort under the same assembly parent.
    # Source display colour alone is never sufficient.
    source_identity_anchors_by_group: dict[str, list[dict[str, Any]]] = defaultdict(
        list
    )
    for anchor_part_id, anchor_proposals in sorted(proposals.items()):
        if len(anchor_proposals) != 1:
            continue
        anchor = anchor_proposals[0]
        if anchor.get("_localization_lane") not in {
            "stable_spatial_multiview",
            "bounded_spatial_multiview",
            QA_CONFIRMED_MULTIVIEW_SEMANTIC_REVIEW_LANE,
        }:
            continue
        source_identity_anchors_by_group[str(anchor["canonical_group_id"])].append(
            {
                "part_id": anchor_part_id,
                "supporting_view_ids": sorted(
                    str(view_id) for view_id in anchor["supporting_view_ids"]
                ),
            }
        )

    registry_count = len(registry_by_part)
    source_signatures = {
        part_id: _source_visual_stable_properties_signature(part)
        for part_id, part in registry_by_part.items()
    }
    source_signature_counts = Counter(
        signature for signature in source_signatures.values() if signature is not None
    )
    pending_source_identity_by_cohort: dict[
        tuple[str, str, str], list[tuple[str, dict[str, Any]]]
    ] = defaultdict(list)
    for pending_part_id, pending_proposals in sorted(
        pending_source_identity_proposals.items()
    ):
        candidate_part = registry_by_part[pending_part_id]
        candidate_signature = source_signatures.get(pending_part_id)
        candidate_parent = _text(
            candidate_part.get("parent_path"),
            f"registry part {pending_part_id}.parent_path",
        )
        assembly_path = (
            candidate_parent.rsplit("/", 1)[0] if "/" in candidate_parent[1:] else "/"
        )
        if candidate_signature is None:
            continue
        for proposal in pending_proposals:
            pending_source_identity_by_cohort[
                (
                    str(proposal["canonical_group_id"]),
                    candidate_signature,
                    assembly_path,
                )
            ].append((pending_part_id, proposal))

    accepted_source_identity_consensus_parts: set[str] = set()
    for (
        group_id,
        candidate_signature,
        assembly_path,
    ), cohort_proposals in sorted(pending_source_identity_by_cohort.items()):
        proposal_part_ids = sorted(part_id for part_id, _proposal in cohort_proposals)
        if len(set(proposal_part_ids)) != len(proposal_part_ids):
            continue
        cohort_part_ids = sorted(
            candidate_id
            for candidate_id, candidate in registry_by_part.items()
            if (
                source_signatures.get(candidate_id) == candidate_signature
                and (
                    (
                        _text(
                            candidate.get("parent_path"),
                            f"registry part {candidate_id}.parent_path",
                        ).rsplit("/", 1)[0]
                        if "/"
                        in _text(
                            candidate.get("parent_path"),
                            f"registry part {candidate_id}.parent_path",
                        )[1:]
                        else "/"
                    )
                    == assembly_path
                )
            )
        )
        global_signature_count = int(
            source_signature_counts.get(candidate_signature, 0)
        )
        maximum_signature_count = max(
            SOURCE_IDENTITY_MIN_SIGNATURE_COUNT,
            math.floor(SOURCE_IDENTITY_MAX_REGISTRY_FRACTION * registry_count),
        )
        if (
            not SOURCE_IDENTITY_MIN_SIGNATURE_COUNT
            <= len(cohort_part_ids)
            <= SOURCE_IDENTITY_MAX_ASSEMBLY_COHORT_SIZE
            or proposal_part_ids != cohort_part_ids
            or not SOURCE_IDENTITY_MIN_SIGNATURE_COUNT
            <= global_signature_count
            <= maximum_signature_count
        ):
            continue

        consensus_confidence_floor = _unit(
            spatial_policy.get("minimum_semantic_conflict_confidence"),
            "spatial policy.minimum_semantic_conflict_confidence",
        )
        consensus_view_ids: set[str] = set()
        direct_consensus_view_ids: set[str] = set()
        conflict_group_ids: set[str] = set()
        for cohort_part_id in cohort_part_ids:
            spatial_part = spatial_parts[cohort_part_id]
            for raw_vote in _sequence(
                spatial_part.get("semantic_votes", []),
                f"spatial part {cohort_part_id}.semantic_votes",
            ):
                if not isinstance(raw_vote, Mapping):
                    continue
                vote = raw_vote
                effective_confidence = _optional_unit(vote.get("effective_confidence"))
                view_id = vote.get("view_id")
                vote_group_id = vote.get("canonical_group_id")
                if not (
                    vote.get("alignment_trusted") is True
                    and vote.get("unique_canonical_join") is True
                    and isinstance(view_id, str)
                    and view_id in reference_evidence
                    and isinstance(vote_group_id, str)
                    and effective_confidence is not None
                    and effective_confidence >= consensus_confidence_floor
                ):
                    continue
                if vote_group_id == group_id and vote.get("status") in {
                    "matched",
                    "review",
                }:
                    consensus_view_ids.add(view_id)
                    if vote.get("reason_code") == "direct_visual_match":
                        direct_consensus_view_ids.add(view_id)
                elif vote_group_id != group_id and vote.get("status") == "matched":
                    conflict_group_ids.add(vote_group_id)
        consensus_content_cluster_ids = sorted(
            {
                reference_evidence[view_id]["content_cluster_id"]
                for view_id in consensus_view_ids
            }
        )
        consensus_pose_cluster_ids = sorted(
            {
                reference_evidence[view_id]["pose_cluster_id"]
                for view_id in consensus_view_ids
            }
        )
        deficit_view_ids = {
            str(item["reference_view_id"])
            for item in _sequence(
                repairable_groups[group_id].get("supporting_views"),
                f"deficit {group_id}.supporting_views",
            )
            if isinstance(item, Mapping)
            and isinstance(item.get("reference_view_id"), str)
        }
        visual_qa_bridge = (
            visual_similarity_first
            and len(deficit_view_ids) >= 2
            and len(consensus_view_ids) >= 1
            and len(direct_consensus_view_ids) >= 1
            and len(consensus_content_cluster_ids) >= 1
            and len(consensus_pose_cluster_ids) >= 1
        )
        independent_multiview_consensus = (
            len(consensus_view_ids) >= 2
            and len(consensus_content_cluster_ids) >= 2
            and len(consensus_pose_cluster_ids) >= 2
        )
        if conflict_group_ids or not (
            independent_multiview_consensus or visual_qa_bridge
        ):
            continue

        consensus_mode = (
            "independent_multiview_semantic_consensus"
            if independent_multiview_consensus
            else "direct_visual_anchor_plus_multiview_qa_deficit"
        )
        for cohort_part_id, proposal in cohort_proposals:
            proposal["_localization_lane"] = SOURCE_IDENTITY_COHORT_CONSENSUS_LANE
            proposal["source_identity_anchor_part_ids"] = sorted(
                set(cohort_part_ids) - {cohort_part_id}
            )
            proposal["source_identity_anchor_supporting_view_ids"] = sorted(
                consensus_view_ids
            )
            proposal["source_visual_stable_properties_signature_sha256"] = (
                candidate_signature
            )
            proposal["source_identity_assembly_path"] = assembly_path
            proposal["source_identity_cohort_part_ids"] = cohort_part_ids
            proposal["source_identity_signature_count"] = global_signature_count
            proposal["source_identity_consensus_view_ids"] = sorted(consensus_view_ids)
            proposal["source_identity_consensus_content_cluster_ids"] = (
                consensus_content_cluster_ids
            )
            proposal["source_identity_consensus_pose_cluster_ids"] = (
                consensus_pose_cluster_ids
            )
            proposal["source_identity_consensus_mode"] = consensus_mode
            proposals[cohort_part_id].append(proposal)
            accepted_source_identity_consensus_parts.add(cohort_part_id)

    for part_id, pending_proposals in sorted(pending_source_identity_proposals.items()):
        if part_id in accepted_source_identity_consensus_parts:
            continue
        for proposal in pending_proposals:
            group_id = str(proposal["canonical_group_id"])
            deficit_view_ids = {
                str(item["reference_view_id"])
                for item in _sequence(
                    repairable_groups[group_id].get("supporting_views"),
                    f"deficit {group_id}.supporting_views",
                )
                if isinstance(item, Mapping)
                and isinstance(item.get("reference_view_id"), str)
            }
            candidate_part = registry_by_part[part_id]
            candidate_signature = source_signatures.get(part_id)
            candidate_parent = _text(
                candidate_part.get("parent_path"),
                f"registry part {part_id}.parent_path",
            )
            assembly_path = (
                candidate_parent.rsplit("/", 1)[0]
                if "/" in candidate_parent[1:]
                else "/"
            )
            matching_anchors: list[dict[str, Any]] = []
            for anchor in source_identity_anchors_by_group.get(group_id, []):
                anchor_part_id = str(anchor["part_id"])
                anchor_part = registry_by_part[anchor_part_id]
                anchor_parent = _text(
                    anchor_part.get("parent_path"),
                    f"registry part {anchor_part_id}.parent_path",
                )
                anchor_assembly = (
                    anchor_parent.rsplit("/", 1)[0] if "/" in anchor_parent[1:] else "/"
                )
                if (
                    candidate_signature is not None
                    and source_signatures.get(anchor_part_id) == candidate_signature
                    and anchor_assembly == assembly_path
                    and set(anchor["supporting_view_ids"]) == deficit_view_ids
                ):
                    matching_anchors.append(anchor)
            cohort_part_ids = sorted(
                candidate_id
                for candidate_id, candidate in registry_by_part.items()
                if (
                    source_signatures.get(candidate_id) == candidate_signature
                    and (
                        (
                            _text(
                                candidate.get("parent_path"),
                                f"registry part {candidate_id}.parent_path",
                            ).rsplit("/", 1)[0]
                            if "/"
                            in _text(
                                candidate.get("parent_path"),
                                f"registry part {candidate_id}.parent_path",
                            )[1:]
                            else "/"
                        )
                        == assembly_path
                    )
                )
            )
            global_signature_count = int(
                source_signature_counts.get(candidate_signature, 0)
            )
            anchor_part_ids = sorted(
                str(anchor["part_id"]) for anchor in matching_anchors
            )
            reasons: list[str] = []
            if candidate_signature is None:
                reasons.append("SOURCE_IDENTITY_PROPERTIES_UNAVAILABLE")
            if not (
                SOURCE_IDENTITY_MIN_SIGNATURE_COUNT
                <= global_signature_count
                <= max(
                    SOURCE_IDENTITY_MIN_SIGNATURE_COUNT,
                    math.floor(SOURCE_IDENTITY_MAX_REGISTRY_FRACTION * registry_count),
                )
            ):
                reasons.append("SOURCE_IDENTITY_SIGNATURE_NOT_RARE")
            if not matching_anchors:
                reasons.append("SOURCE_IDENTITY_COMPLETE_ANCHOR_MISSING")
            if not SOURCE_IDENTITY_MIN_SIGNATURE_COUNT <= len(
                cohort_part_ids
            ) <= SOURCE_IDENTITY_MAX_ASSEMBLY_COHORT_SIZE or set(cohort_part_ids) != {
                part_id,
                *anchor_part_ids,
            }:
                reasons.append("SOURCE_IDENTITY_ASSEMBLY_COHORT_NOT_EXACT")
            if reasons:
                skips.append(
                    {
                        "part_id": part_id,
                        "canonical_group_id": group_id,
                        "reason_codes": sorted(set(reasons)),
                    }
                )
                continue
            proposal["source_identity_anchor_part_ids"] = anchor_part_ids
            proposal["source_identity_anchor_supporting_view_ids"] = sorted(
                deficit_view_ids
            )
            proposal["source_visual_stable_properties_signature_sha256"] = (
                candidate_signature
            )
            proposal["source_identity_assembly_path"] = assembly_path
            proposal["source_identity_cohort_part_ids"] = cohort_part_ids
            proposal["source_identity_signature_count"] = global_signature_count
            proposals[part_id].append(proposal)

    (
        dominant_assembly_records,
        dominant_assembly_cohorts,
        dominant_assembly_skips,
    ) = _dominant_assembly_cohort_expansions(
        canonical_groups=canonical_groups,
        repairable_groups=repairable_groups,
        dominant_residual_groups=dominant_residual_groups,
        registry_by_part=registry_by_part,
        spatial_parts=spatial_parts,
        alignment_audits=alignment_audits,
        reference_evidence=reference_evidence,
        gate_decisions=gate_decisions,
        mapping_decisions=mapping_decisions,
        geometry_risks=geometry_risks,
        baseline_by_part=baseline_by_part,
        occupied_proposals_by_part=proposals,
        confirmed_materials=confirmed_materials,
        provisional_material_groups=provisional_material_groups,
        input_hashes=dominant_assembly_input_hashes,
    )
    (
        bounded_signature_records,
        bounded_signature_cohorts,
    ) = _bounded_signature_sibling_cohort_expansions(
        canonical_groups=canonical_groups,
        repairable_groups=repairable_groups,
        registry_by_part=registry_by_part,
        spatial_parts=spatial_parts,
        reference_evidence=reference_evidence,
        gate_decisions=gate_decisions,
        mapping_decisions=mapping_decisions,
        geometry_risks=geometry_risks,
        baseline_by_part=baseline_by_part,
        occupied_proposals_by_part=proposals,
        confirmed_materials=confirmed_materials,
        provisional_material_groups=provisional_material_groups,
        spatial_policy=spatial_policy,
        minimum_semantic_confidence=minimum_semantic_confidence,
        input_hashes=dominant_assembly_input_hashes,
    )
    dominant_part_ids = {str(record["part_id"]) for record in dominant_assembly_records}
    overlapping_bounded_cohort_ids = {
        str(record["contract"]["cohort_id"])
        for record in bounded_signature_records
        if str(record["part_id"]) in dominant_part_ids
    }
    bounded_signature_records = [
        record
        for record in bounded_signature_records
        if str(record["contract"]["cohort_id"]) not in overlapping_bounded_cohort_ids
    ]
    bounded_signature_cohorts = [
        cohort
        for cohort in bounded_signature_cohorts
        if str(cohort["cohort_id"]) not in overlapping_bounded_cohort_ids
    ]
    dominant_assembly_records.extend(bounded_signature_records)
    dominant_assembly_cohorts.extend(bounded_signature_cohorts)
    dominant_assembly_records.sort(key=lambda item: item["part_id"])
    dominant_assembly_cohorts.sort(key=lambda item: item["cohort_id"])
    skips.extend(dominant_assembly_skips)
    for record in dominant_assembly_records:
        part_id = str(record["part_id"])
        group_id = str(record["canonical_group_id"])
        material_id = str(record["material_id"])
        contract = _mapping(
            record.get("contract"),
            f"dominant assembly cohort record {part_id}.contract",
        )
        support_views = sorted(
            str(view_id) for view_id in contract["anchor_supporting_view_ids"]
        )
        support_records = [reference_evidence[view_id] for view_id in support_views]
        existing = proposals.get(part_id, [])
        if len(existing) == 1:
            proposal = existing[0]
        elif not existing:
            proposal = {
                "part_id": part_id,
                "canonical_group_id": group_id,
                "material_id": material_id,
                "supporting_view_ids": support_views,
                "supporting_content_cluster_ids": sorted(
                    {item["content_cluster_id"] for item in support_records}
                ),
                "supporting_pose_cluster_ids": sorted(
                    {item["pose_cluster_id"] for item in support_records}
                ),
                "_semantic_conflict_override_view_ids": [],
                "_semantic_anchor_view_ids": [],
                "_spatial_anchor_view_ids": [],
                "_confirmed_source_material_id": material_id,
                "_provisional_material_candidate": bool(
                    record["provisional_material_candidate"]
                ),
                "_parameters": None,
                "_parameter_audit": None,
                "_dark_spatial_audit": None,
                "_dark_residual_support": None,
            }
            proposals[part_id].append(proposal)
        else:
            raise AssertionError(
                "dominant assembly cohort admitted an ambiguous proposal"
            )
        if (
            proposal.get("canonical_group_id") != group_id
            or proposal.get("material_id") != material_id
        ):
            raise AssertionError(
                "dominant assembly cohort admitted a conflicting proposal"
            )
        proposal.update(
            {
                "supporting_view_ids": support_views,
                "supporting_content_cluster_ids": sorted(
                    {item["content_cluster_id"] for item in support_records}
                ),
                "supporting_pose_cluster_ids": sorted(
                    {item["pose_cluster_id"] for item in support_records}
                ),
                "_localization_lane": str(record["localization_lane"]),
                "_semantic_conflict_override_view_ids": [],
                "_semantic_anchor_view_ids": [],
                "_spatial_anchor_view_ids": [],
                "_confirmed_source_material_id": material_id,
                "_provisional_material_candidate": bool(
                    record["provisional_material_candidate"]
                ),
                "_parameters": None,
                "_parameter_audit": None,
                "_dark_spatial_audit": None,
                "_dark_residual_support": None,
                "_dominant_assembly_cohort_audit": copy.deepcopy(dict(contract)),
                "_dominant_assembly_member_role": str(record["member_role"]),
            }
        )

    # Dark-on-black recovery is mass-bounded after all ordinary repairs are
    # known.  Ordinary proposals already turning the same projected parts
    # black consume the budget first; new diagnostic candidates are then
    # selected by deterministic evidence strength without exceeding either
    # the per-part or total residual cap.
    dark_residual_budgets: list[dict[str, Any]] = []
    for group_id, deficit in sorted(dark_residual_groups.items()):
        support = copy.deepcopy(deficit["supporting_views"][0])
        view_id = str(support["reference_view_id"])
        reference_pixels = int(support["normalized_reference_pixels"])
        render_pixels = int(support["render_foreground_pixels"])
        budget_pixels = int(support["budget_pixels"])
        budget_limit_pixels = int(support["budget_limit_pixels"])
        per_part_limit = int(
            math.floor(DARK_FOREGROUND_MAX_SINGLE_CONTRIBUTION_FACTOR * budget_pixels)
        )

        existing_records: list[dict[str, Any]] = []
        existing_part_ids: set[str] = set()
        for proposal_part_id, part_proposals in sorted(proposals.items()):
            if proposal_part_id in existing_part_ids:
                continue
            matching = [
                proposal
                for proposal in part_proposals
                if proposal.get("canonical_group_id") == group_id
                and proposal.get("_localization_lane") not in _DARK_RESIDUAL_LANES
            ]
            if not matching:
                continue
            observation = next(
                (
                    item
                    for raw_observation in _sequence(
                        spatial_parts[proposal_part_id].get("observations", []),
                        (f"spatial part {proposal_part_id}.observations"),
                    )
                    for item in [
                        _mapping(
                            raw_observation,
                            f"spatial part {proposal_part_id}.observation",
                        )
                    ]
                    if item.get("reference_view_id") == view_id
                ),
                None,
            )
            projected = (
                observation.get("projected_part_pixels")
                if isinstance(observation, Mapping)
                else None
            )
            if (
                not isinstance(projected, int)
                or isinstance(projected, bool)
                or projected <= 0
            ):
                continue
            contribution = int(math.ceil(projected / render_pixels * reference_pixels))
            existing_part_ids.add(proposal_part_id)
            existing_records.append(
                {
                    "part_id": proposal_part_id,
                    "projected_part_pixels": projected,
                    "estimated_contribution_pixels": contribution,
                }
            )
        existing_contribution = sum(
            int(item["estimated_contribution_pixels"]) for item in existing_records
        )
        cumulative = existing_contribution
        candidate_records: list[dict[str, Any]] = []
        candidates = sorted(
            (
                proposal
                for part_proposals in pending_dark_proposals.values()
                for proposal in part_proposals
                if proposal.get("canonical_group_id") == group_id
                and isinstance(proposal.get("_dark_spatial_audit"), Mapping)
            ),
            key=lambda item: (
                -float(item["_dark_spatial_audit"]["evidence_strength"]),
                str(item["part_id"]),
            ),
        )
        selected_part_ids: list[str] = []
        selected_contribution = 0
        for proposal in candidates:
            audit = proposal["_dark_spatial_audit"]
            contribution = int(audit["estimated_contribution_pixels"])
            selected = True
            reason_code: str | None = None
            if contribution <= 0:
                selected = False
                reason_code = "DARK_RESIDUAL_CONTRIBUTION_NOT_POSITIVE"
            elif contribution > per_part_limit:
                selected = False
                reason_code = "DARK_RESIDUAL_SINGLE_PART_BUDGET_EXCEEDED"
            elif cumulative + contribution > budget_limit_pixels:
                selected = False
                reason_code = "DARK_RESIDUAL_TOTAL_BUDGET_EXCEEDED"
            if selected:
                cumulative += contribution
                selected_contribution += contribution
                selected_part_ids.append(str(proposal["part_id"]))
                proposal["_dark_budget_audit"] = {
                    "budget_pixels": budget_pixels,
                    "budget_limit_pixels": budget_limit_pixels,
                    "existing_contribution_pixels": existing_contribution,
                    "estimated_contribution_pixels": contribution,
                    "selected_contribution_pixels": contribution,
                    "cumulative_contribution_pixels": cumulative,
                }
                proposals[str(proposal["part_id"])].append(proposal)
            else:
                skips.append(
                    {
                        "part_id": str(proposal["part_id"]),
                        "canonical_group_id": group_id,
                        "reason_codes": [str(reason_code)],
                    }
                )
            candidate_records.append(
                {
                    "part_id": str(proposal["part_id"]),
                    "evidence_strength": audit["evidence_strength"],
                    "diagnostic_sha256": audit["diagnostic_sha256"],
                    "estimated_contribution_pixels": contribution,
                    "selected": selected,
                    "reason_code": reason_code,
                    "cumulative_contribution_pixels": cumulative,
                }
            )
        dark_residual_budgets.append(
            {
                "canonical_group_id": group_id,
                "reference_view_id": view_id,
                "dark_residual_support": support,
                "budget_pixels": budget_pixels,
                "budget_limit_pixels": budget_limit_pixels,
                "per_part_limit_pixels": per_part_limit,
                "existing_contribution_pixels": existing_contribution,
                "existing_parts": existing_records,
                "candidates": candidate_records,
                "selected_part_ids": sorted(selected_part_ids),
                "selected_contribution_pixels": selected_contribution,
                "total_contribution_pixels": cumulative,
            }
        )

    changes: list[dict[str, Any]] = []
    localization_lanes: list[dict[str, str]] = []
    output_by_part = {
        part_id: copy.deepcopy(item) for part_id, item in baseline_by_part.items()
    }
    for part_id, part_proposals in sorted(proposals.items()):
        group_ids = {item["canonical_group_id"] for item in part_proposals}
        material_ids = {item["material_id"] for item in part_proposals}
        if len(group_ids) != 1 or len(material_ids) != 1:
            skips.append(
                {
                    "part_id": part_id,
                    "canonical_group_id": None,
                    "reason_codes": ["MULTIPLE_REPAIR_PROPOSALS_CONFLICT"],
                }
            )
            continue
        proposal = part_proposals[0]
        baseline_provenance = baseline_by_part[part_id].get("provenance")
        if (
            isinstance(baseline_provenance, Mapping)
            and baseline_provenance.get("canonical_group_id") is not None
        ):
            skips.append(
                {
                    "part_id": part_id,
                    "canonical_group_id": proposal["canonical_group_id"],
                    "reason_codes": [
                        AUTHORITATIVE_CANONICAL_GROUP_LOCK_REASON
                    ],
                }
            )
            continue
        localization_lane = str(proposal.pop("_localization_lane"))
        semantic_conflict_override_view_ids = list(
            proposal.pop("_semantic_conflict_override_view_ids")
        )
        semantic_anchor_view_ids = list(proposal.pop("_semantic_anchor_view_ids", []))
        spatial_anchor_view_ids = list(proposal.pop("_spatial_anchor_view_ids"))
        confirmed_source_material_id = str(
            proposal.pop("_confirmed_source_material_id")
        )
        provisional_material_candidate = bool(
            proposal.pop("_provisional_material_candidate")
        )
        parameters = proposal.pop("_parameters")
        parameter_audit = proposal.pop("_parameter_audit")
        dark_spatial_audit = proposal.pop("_dark_spatial_audit")
        dark_residual_support = proposal.pop("_dark_residual_support")
        dark_budget_audit = proposal.pop("_dark_budget_audit", None)
        repeated_cohort_audit = proposal.pop("_repeated_dark_cohort_audit", None)
        repeated_member_audit = proposal.pop("_repeated_dark_member_audit", None)
        dominant_assembly_cohort_audit = proposal.pop(
            "_dominant_assembly_cohort_audit", None
        )
        dominant_assembly_member_role = proposal.pop(
            "_dominant_assembly_member_role", None
        )
        assignment = output_by_part[part_id]
        old_material_id = str(assignment["material_id"])
        canonical_group = canonical_groups[proposal["canonical_group_id"]]
        assignment["material_id"] = proposal["material_id"]
        assignment["semantic"] = _text(
            canonical_group.get("visual_description"),
            f"canonical group {proposal['canonical_group_id']}.visual_description",
        )
        assignment["confidence"] = 0.0
        assignment["evidence_views"] = []
        assignment["status"] = POLICY_FALLBACK_STATUS
        assignment.pop("apply_action", None)
        assignment.pop("source_visual_material_prim_path", None)
        assignment.pop("source_visual_material_binding_sha256", None)
        if parameters is None:
            assignment.pop("parameters", None)
        else:
            assignment["parameters"] = copy.deepcopy(parameters)
        assignment["provenance"] = {
            "tier": "qa_repair_candidate",
            "reason_codes": list(
                BOUNDED_SIGNATURE_SIBLING_REPAIR_REASON_CODES
                if localization_lane == BOUNDED_SIGNATURE_SIBLING_COHORT_LANE
                else (
                    DOMINANT_ASSEMBLY_REPAIR_REASON_CODES
                    if localization_lane == DOMINANT_ASSEMBLY_COHORT_EXPANSION_LANE
                    else (
                        DARK_FOREGROUND_REPAIR_REASON_CODES
                        if localization_lane in _DARK_RESIDUAL_LANES
                        else (
                            REPEATED_GEOMETRY_DARK_REPAIR_REASON_CODES
                            if localization_lane == REPEATED_GEOMETRY_DARK_RESIDUAL_LANE
                            else (
                                PROVISIONAL_REPAIR_REASON_CODES
                                if provisional_material_candidate
                                else REPAIR_REASON_CODES
                            )
                        )
                    )
                )
            ),
            "output_confidence_basis": POLICY_FALLBACK_CONFIDENCE_BASIS,
            "sources": [],
            "canonical_group_id": proposal["canonical_group_id"],
            "baseline_material_id": old_material_id,
            "baseline_tier": baseline_by_part[part_id]["provenance"]["tier"],
            "supporting_view_ids": proposal["supporting_view_ids"],
            "supporting_content_cluster_ids": (
                proposal["supporting_content_cluster_ids"]
            ),
            "supporting_pose_cluster_ids": proposal["supporting_pose_cluster_ids"],
        }
        if provisional_material_candidate:
            proposal["material_selection_basis"] = (
                "high_confidence_whitelist_candidate_pending_render_qa"
            )
            assignment["provenance"]["material_selection_basis"] = (
                "high_confidence_whitelist_candidate_pending_render_qa"
            )
        if localization_lane in _DARK_RESIDUAL_LANES:
            if (
                not isinstance(dark_spatial_audit, Mapping)
                or not isinstance(dark_residual_support, Mapping)
                or not isinstance(dark_budget_audit, Mapping)
            ):
                raise AssertionError("dark residual proposal lost its audit")
            proposal["dark_residual_support"] = copy.deepcopy(
                dict(dark_residual_support)
            )
            proposal["dark_foreground_diagnostic"] = copy.deepcopy(
                dict(dark_spatial_audit)
            )
            proposal.update(copy.deepcopy(dict(dark_budget_audit)))
            assignment["provenance"]["dark_foreground_residual"] = {
                "lane": localization_lane,
                "support": copy.deepcopy(dict(dark_residual_support)),
                "diagnostic_sha256": dark_spatial_audit["diagnostic_sha256"],
                **copy.deepcopy(dict(dark_budget_audit)),
            }
        if localization_lane == REPEATED_GEOMETRY_DARK_RESIDUAL_LANE:
            if not isinstance(repeated_cohort_audit, Mapping) or not isinstance(
                repeated_member_audit, Mapping
            ):
                raise AssertionError("repeated dark cohort proposal lost its audit")
            cohort_part_ids = list(repeated_cohort_audit["cohort_part_ids"])
            proposal["repeated_geometry_dark_cohort_id"] = repeated_cohort_audit[
                "cohort_id"
            ]
            proposal["cohort_part_ids"] = cohort_part_ids
            repeated_provenance = {
                "lane": REPEATED_GEOMETRY_DARK_RESIDUAL_LANE,
                "cohort_id": repeated_cohort_audit["cohort_id"],
                "canonical_group_id": proposal["canonical_group_id"],
                "reference_view_id": repeated_cohort_audit["reference_view_id"],
                "geometry_signature_sha256": repeated_cohort_audit[
                    "geometry_signature"
                ]["signature_sha256"],
                "source_visual_stable_properties_signature_sha256": (
                    repeated_cohort_audit[
                        "source_visual_stable_properties_signature_sha256"
                    ]
                ),
                "cohort_part_ids": cohort_part_ids,
                "budget_pixels": repeated_cohort_audit["budget_pixels"],
                "minimum_contribution_pixels": repeated_cohort_audit[
                    "minimum_contribution_pixels"
                ],
                "maximum_contribution_pixels": repeated_cohort_audit[
                    "maximum_contribution_pixels"
                ],
                "total_projected_part_pixels": repeated_cohort_audit[
                    "total_projected_part_pixels"
                ],
                "total_direct_target_matching_pixels": repeated_cohort_audit[
                    "total_direct_target_matching_pixels"
                ],
                "member_evidence_contract": repeated_member_audit["evidence_contract"],
                "member_evidence_sha256": repeated_member_audit["evidence_sha256"],
            }
            dark_diagnostic_sha256 = repeated_member_audit.get("dark_diagnostic_sha256")
            if isinstance(dark_diagnostic_sha256, str):
                repeated_provenance["dark_diagnostic_sha256"] = dark_diagnostic_sha256
            assignment["provenance"]["repeated_geometry_dark_residual"] = (
                repeated_provenance
            )
        if localization_lane in {
            DOMINANT_ASSEMBLY_COHORT_EXPANSION_LANE,
            BOUNDED_SIGNATURE_SIBLING_COHORT_LANE,
        }:
            if not isinstance(
                dominant_assembly_cohort_audit, Mapping
            ) or dominant_assembly_member_role not in {
                "strict_spatial_anchor",
                "expanded_member",
            }:
                raise AssertionError(
                    "dominant assembly proposal lost its immutable cohort audit"
                )
            unsigned_contract = copy.deepcopy(dict(dominant_assembly_cohort_audit))
            contract_sha256 = unsigned_contract.pop("contract_sha256", None)
            if (
                not isinstance(contract_sha256, str)
                or _canonical_sha256(unsigned_contract) != contract_sha256
            ):
                raise AssertionError("dominant assembly cohort contract hash mismatch")
            dominant_provenance = {
                "schema_version": dominant_assembly_cohort_audit["schema_version"],
                "candidate_kind": dominant_assembly_cohort_audit["candidate_kind"],
                "cohort_id": dominant_assembly_cohort_audit["cohort_id"],
                "contract_sha256": contract_sha256,
                "canonical_group_id": dominant_assembly_cohort_audit[
                    "canonical_group_id"
                ],
                "assembly_path": dominant_assembly_cohort_audit["assembly_path"],
                "source_visual_stable_properties_signature_sha256": (
                    dominant_assembly_cohort_audit[
                        "source_visual_stable_properties_signature_sha256"
                    ]
                ),
                "anchor_part_ids": list(
                    dominant_assembly_cohort_audit["anchor_part_ids"]
                ),
                "anchor_supporting_view_ids": list(
                    dominant_assembly_cohort_audit["anchor_supporting_view_ids"]
                ),
                "anchor_child_branches": list(
                    dominant_assembly_cohort_audit["anchor_child_branches"]
                ),
                "cohort_part_ids": list(
                    dominant_assembly_cohort_audit["cohort_part_ids"]
                ),
                "expanded_member_part_ids": list(
                    dominant_assembly_cohort_audit["expanded_member_part_ids"]
                ),
                "member_role": dominant_assembly_member_role,
                "membership_status": dominant_assembly_cohort_audit[
                    "membership_status"
                ],
                "baseline_material_id": old_material_id,
                "input_hashes": copy.deepcopy(
                    dominant_assembly_cohort_audit["input_hashes"]
                ),
            }
            proposal_policy = dominant_assembly_cohort_audit.get("proposal_policy")
            if isinstance(proposal_policy, str):
                dominant_provenance["proposal_policy"] = proposal_policy
            assignment["provenance"]["dominant_assembly_cohort"] = dominant_provenance
            proposal["dominant_assembly_cohort_id"] = dominant_assembly_cohort_audit[
                "cohort_id"
            ]
            proposal["dominant_assembly_member_role"] = dominant_assembly_member_role
        if parameter_audit is not None:
            assignment["provenance"]["mvinverse"] = {
                "source_material_id": confirmed_source_material_id,
                "output_material_id": proposal["material_id"],
                "group_id": parameter_audit["group_id"],
                "tuning_profile_id": parameter_audit["tuning_profile_id"],
                "parameterization_mode": parameter_audit.get(
                    "parameterization_mode", "full_mvinverse_pbr"
                ),
                "contributing_view_ids": list(parameter_audit["contributing_view_ids"]),
                "base_color_srgb": list(parameter_audit["base_color_srgb"]),
                "base_color_linear": list(parameter_audit["base_color_linear"]),
                "observed_metallic": parameter_audit["observed_metallic"],
                "authored_metallic": parameter_audit["authored_metallic"],
                "roughness": parameter_audit["roughness"],
                "authored_parameter_names": list(
                    parameter_audit["authored_parameter_names"]
                ),
                "reason_code": (
                    "MVINVERSE_AUTO_PARAMETER_ELIGIBLE"
                    if parameter_audit.get("parameterization_mode")
                    != ("multiview_palette_corroborated_color_only")
                    else "MVINVERSE_COLOR_CORROBORATED_BY_MULTIVIEW_PALETTE"
                ),
            }
            proposal["confirmed_source_material_id"] = confirmed_source_material_id
            proposal["mvinverse_parameterized"] = True
        if semantic_conflict_override_view_ids:
            proposal["semantic_conflict_override_view_ids"] = (
                semantic_conflict_override_view_ids
            )
        if semantic_anchor_view_ids:
            proposal["semantic_anchor_view_ids"] = semantic_anchor_view_ids
            assignment["provenance"]["semantic_review_override"] = {
                "conflict_view_ids": semantic_conflict_override_view_ids,
                "anchor_view_ids": semantic_anchor_view_ids,
            }
        if spatial_anchor_view_ids:
            proposal["spatial_anchor_view_ids"] = spatial_anchor_view_ids
        if localization_lane in {
            SOURCE_IDENTITY_ANCHORED_DIAGNOSTIC_LANE,
            SOURCE_IDENTITY_COHORT_CONSENSUS_LANE,
        }:
            source_identity_fields = (
                "source_identity_anchor_part_ids",
                "source_identity_anchor_supporting_view_ids",
                "source_visual_stable_properties_signature_sha256",
                "source_identity_assembly_path",
                "source_identity_cohort_part_ids",
                "source_identity_signature_count",
            )
            if localization_lane == SOURCE_IDENTITY_COHORT_CONSENSUS_LANE:
                source_identity_fields = (
                    *source_identity_fields,
                    "source_identity_consensus_view_ids",
                    "source_identity_consensus_content_cluster_ids",
                    "source_identity_consensus_pose_cluster_ids",
                    "source_identity_consensus_mode",
                )
            if any(field not in proposal for field in source_identity_fields):
                raise AssertionError(
                    "source-identity proposal lost its immutable cohort audit"
                )
            assignment["provenance"]["source_identity_anchor"] = {
                "lane": localization_lane,
                **{
                    field: copy.deepcopy(proposal[field])
                    for field in source_identity_fields
                },
            }
        changes.append(
            {
                **proposal,
                "old_material_id": old_material_id,
                "new_material_id": proposal["material_id"],
            }
        )
        localization_lanes.append(
            {
                "part_id": part_id,
                "canonical_group_id": proposal["canonical_group_id"],
                "lane": localization_lane,
            }
        )

    if changes:
        output_assignments = [
            output_by_part[str(item["part_id"])] for item in assignments
        ]
        baseline_provenance = copy.deepcopy(
            dict(_mapping(baseline_plan.get("provenance"), "baseline_plan.provenance"))
        )
        if baseline_provenance.get("mode") != POLICY_EXACT_COVER_MODE:
            raise AssertionError("quality repair lost exact-cover authorization")
        if REPAIR_PROVENANCE_FIELD in baseline_provenance:
            raise QualityRepairError(
                "baseline plan already contains quality-repair provenance"
            )
        baseline_provenance[REPAIR_PROVENANCE_FIELD] = {
            "mode": REPAIR_MODE,
            "input_hashes": copy.deepcopy(input_hashes),
            "changed_part_ids": sorted(item["part_id"] for item in changes),
        }
        output_plan = {
            "schema_version": PLAN_SCHEMA_VERSION,
            "assignments": output_assignments,
            "provenance": baseline_provenance,
        }
        top_reason_codes = ["QUALITY_REPAIR_PLAN_COMPILED"]
    else:
        # Byte-equivalent canonical JSON keeps a no-op incapable of changing
        # the material plan even if a caller ignores changed_count.
        output_plan = copy.deepcopy(dict(baseline_plan))
        if not top_reason_codes:
            top_reason_codes = ["NO_ELIGIBLE_NEUTRAL_FALLBACK_PARTS"]

    output_assignment_ids = [
        _text(item.get("part_id"), "output assignment part_id")
        for item in _sequence(output_plan.get("assignments"), "output assignments")
        if isinstance(item, Mapping)
    ]
    if len(output_assignment_ids) != len(registry_part_ids) or set(
        output_assignment_ids
    ) != set(registry_part_ids):
        raise AssertionError("quality repair broke exact-cover invariants")
    output_materials = {
        _text(item.get("material_id"), "output assignment material_id")
        for item in _sequence(output_plan.get("assignments"), "output assignments")
        if isinstance(item, Mapping)
    }
    if not output_materials <= allowed_material_ids:
        raise AssertionError("quality repair emitted a non-whitelisted material")

    skip_counts = Counter(
        reason
        for item in skips
        for reason in _sequence(item["reason_codes"], "skip.reason_codes")
    )
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "material_selection_objective": material_selection_objective,
        "summary": {
            "status": "REPAIRED" if changes else "SAFE_NOOP",
            "baseline_assignment_count": len(assignments),
            "repairable_group_count": len(repairable_groups),
            "candidate_part_count": len(proposals),
            "changed_count": len(changes),
            "no_op": not changes,
            "exact_cover": True,
            "all_materials_in_whitelist": True,
            "maximum_orchestrator_retry_count": 1,
        },
        "reason_codes": top_reason_codes,
        "input_hashes": input_hashes,
        "view_diagnostics": view_diagnostics,
        "dark_residual_view_diagnostics": dark_view_diagnostics,
        "group_diagnostics": group_diagnostics,
        "dark_residual_budgets": dark_residual_budgets,
        "repeated_geometry_dark_cohorts": sorted(
            (
                item
                for item in repeated_geometry_dark_cohorts
                if item.get("selected") is True and item.get("reason_codes") == []
            ),
            key=lambda item: item["cohort_id"],
        ),
        "dominant_assembly_cohorts": dominant_assembly_cohorts,
        "changes": sorted(changes, key=lambda item: item["part_id"]),
        "localization_lanes": sorted(
            localization_lanes, key=lambda item: item["part_id"]
        ),
        "skips": sorted(
            skips,
            key=lambda item: (
                item["part_id"],
                str(item.get("canonical_group_id") or ""),
            ),
        ),
        "skip_reason_counts": dict(sorted(skip_counts.items())),
        "mvinverse": {
            "enabled": mvinverse_pbr_evidence is not None,
            "parameterized_part_ids": sorted(
                item["part_id"]
                for item in changes
                if item.get("mvinverse_parameterized") is True
            ),
            "skipped": parameterization_skips,
        },
        **(
            {
                "material_collapse_recovery": {
                    "excluded_group_ids": sorted(
                        collapse_recovery_excluded_group_ids
                    )
                }
            }
            if collapse_recovery_excluded_group_ids
            else {}
        ),
        "provisional_material_candidate_group_ids": sorted(
            {
                str(item["canonical_group_id"])
                for item in changes
                if item.get("material_selection_basis")
                == "high_confidence_whitelist_candidate_pending_render_qa"
            }
        ),
        "output_plan_sha256": _canonical_sha256(output_plan),
    }
    if not allow_parameter_writes:
        report["summary"]["selected_mdl_library_defaults_required"] = True
        report["mvinverse"]["parameter_writes_allowed"] = False
    return output_plan, report


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        resolved = path.expanduser().resolve(strict=True)
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise QualityRepairError(f"unable to read {label}: {exc}") from exc
    return dict(_mapping(value, label))


def _write_json_new(path: Path, value: Mapping[str, Any]) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.exists():
        raise QualityRepairError(f"refusing to overwrite output: {resolved}")
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
    parser.add_argument("--baseline-plan", type=Path, required=True)
    parser.add_argument("--baseline-policy-audit", type=Path, required=True)
    parser.add_argument("--quality-report", type=Path, required=True)
    parser.add_argument("--palette-fusion", type=Path, required=True)
    parser.add_argument("--spatial-report", type=Path, required=True)
    parser.add_argument("--spatial-gate-audit", type=Path, required=True)
    parser.add_argument("--mapping-consensus", type=Path, required=True)
    parser.add_argument("--geometry-risk", type=Path, required=True)
    parser.add_argument("--group-materials", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--whitelist", type=Path, required=True)
    parser.add_argument("--mvinverse-pbr-evidence", type=Path)
    parser.add_argument(
        "--immutable-mdl-after-selection",
        action="store_true",
        help=(
            "Treat this render-guided pass as final MDL selection: choose only "
            "existing NVIDIA MDL exports and do not author color or PBR parameters."
        ),
    )
    parser.add_argument(
        "--material-selection-objective",
        choices=sorted(MATERIAL_SELECTION_OBJECTIVES),
        default=MATERIAL_SELECTION_OBJECTIVE_SEMANTIC,
        help=(
            "Keep the conservative semantic contract by default, or permit "
            "a direct visual cohort anchor to bridge a multiview QA deficit."
        ),
    )
    parser.add_argument("--output-plan", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    requested_outputs = [
        args.output_plan.expanduser().resolve(),
        args.audit.expanduser().resolve(),
    ]
    if requested_outputs[0] == requested_outputs[1]:
        raise QualityRepairError("--output-plan and --audit must be different files")
    existing_outputs = [path for path in requested_outputs if path.exists()]
    if existing_outputs:
        raise QualityRepairError(f"refusing to overwrite output: {existing_outputs[0]}")
    plan, report = build_quality_repair_plan(
        baseline_plan=_read_json(args.baseline_plan, "baseline plan"),
        baseline_policy_audit=_read_json(
            args.baseline_policy_audit, "baseline policy audit"
        ),
        quality_report=_read_json(args.quality_report, "quality report"),
        palette_fusion=_read_json(args.palette_fusion, "palette fusion"),
        spatial_report=_read_json(args.spatial_report, "spatial report"),
        spatial_gate_audit=_read_json(args.spatial_gate_audit, "spatial gate audit"),
        mapping_consensus=_read_json(args.mapping_consensus, "mapping consensus"),
        geometry_risk=_read_json(args.geometry_risk, "geometry risk"),
        group_materials=_read_json(args.group_materials, "group materials"),
        registry=_read_json(args.registry, "registry"),
        whitelist=_read_json(args.whitelist, "whitelist"),
        mvinverse_pbr_evidence=(
            _read_json(
                args.mvinverse_pbr_evidence,
                "MVInverse PBR evidence",
            )
            if args.mvinverse_pbr_evidence is not None
            else None
        ),
        allow_parameter_writes=not args.immutable_mdl_after_selection,
        material_selection_objective=args.material_selection_objective,
    )
    output_plan = _write_json_new(args.output_plan, plan)
    audit = _write_json_new(args.audit, report)
    print(
        json.dumps(
            {
                "output_plan": str(output_plan),
                "audit": str(audit),
                **report["summary"],
            },
            ensure_ascii=False,
            allow_nan=False,
        ),
        flush=True,
    )
    return 0


__all__ = [
    "QualityRepairError",
    "REPORT_SCHEMA_VERSION",
    "REPAIR_MODE",
    "build_quality_repair_plan",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
