#!/usr/bin/env python3
"""Deterministic, fail-closed confidence gate for staged Qwen assignments.

The gate is deliberately an opt-in, read-only post-processing step.  It does
not modify ``run_staged_local`` output, the human review document, or an USD
stage.  A decision of ``auto`` means only that the evidence satisfies this
module's strict policy; callers still decide whether to consume the emitted
``auto_material_plan``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

if __package__:
    from qwen_material_pipeline.evidence.geometry import (
        GeometryRiskError,
        validate_geometry_risk,
    )
else:  # Keep the documented ``python confidence_gate.py`` CLI usable.
    from geometry_risk import GeometryRiskError, validate_geometry_risk


SCHEMA_VERSION = "qwen-material-confidence-gate/v1"
VIEW_EVIDENCE_SCHEMA_VERSION = "qwen-material-view-evidence/v1"
STAGED_SCHEMA_VERSION = "qwen-staged-material-result/v1"
REGISTRY_SCHEMA_VERSION = "qwen-material-parts/v1"
MATERIAL_PLAN_SCHEMA_VERSION = "1.0"
SPATIAL_GATE_AUDIT_SCHEMA_VERSION = "qwen-spatial-mapping-gate/v1"
ISOLATED_EVIDENCE_SCHEMA_VERSION = "qwen-isolated-part-evidence/v1"
MATERIAL_SELECTION_CONFIDENCE_SCHEMA_VERSION = (
    "qwen-derived-material-selection-confidence/v1"
)

_PART_ID_RE = re.compile(r"P[0-9]{4,8}")
_GEOMETRY_VIEW_PREFIXES = (
    "cad_",
    "part_ids_",
    "part_contact_",
    "part_highlight_",
    "batch_parts_",
)
_DECISION_ORDER = {"auto": 0, "review": 1, "preserve": 2}
_SEMANTIC_RETRIEVAL_STRATEGY_V5 = "family_gated_semantic_mvinverse_similarity_score/v5"
_SEMANTIC_RETRIEVAL_STRATEGY_V6 = "family_gated_semantic_mvinverse_similarity_score/v6"
_SEMANTIC_RETRIEVAL_STRATEGY_V7 = "family_gated_semantic_mvinverse_similarity_score/v7"
_SEMANTIC_RETRIEVAL_STRATEGY_V8 = "family_gated_semantic_mvinverse_similarity_score/v8"
_SEMANTIC_RETRIEVAL_STRATEGY_V9 = "family_gated_semantic_mvinverse_similarity_score/v9"
_SEMANTIC_RETRIEVAL_STRATEGY_V10 = "family_gated_semantic_mvinverse_similarity_score/v10"
_SEMANTIC_RETRIEVAL_STRATEGY_V11 = "family_gated_semantic_mvinverse_similarity_score/v11"
_SEMANTIC_RETRIEVAL_STRATEGY = "family_gated_semantic_mvinverse_similarity_score/v12"
_VISUAL_RETRIEVAL_STRATEGY = "siglip2_full_catalog_plus_dinov2_masked_rrf/v1"
_BASE_BANK_RETRIEVAL_STRATEGY = (
    "base_observation_bank_siglip2_dinov2_color_mvinverse_rrf/v1"
)
_VISUAL_RETRIEVAL_STRATEGIES = {
    _VISUAL_RETRIEVAL_STRATEGY,
    _BASE_BANK_RETRIEVAL_STRATEGY,
}
_VISUAL_RETRIEVAL_FINAL_AUTHORITY = "exact_mdl_render_tournament"
_SEMANTIC_AUTO_THRESHOLD = 0.85
_SEMANTIC_REVIEW_THRESHOLD = 0.60
_SEMANTIC_MINIMUM_SUPPORT_VIEWS = 2
_UNRESOLVED_SEMANTICS = frozenset({"other", "unknown"})
_INTRINSIC_METAL_COLORS = frozenset({"orange", "brown", "yellow"})
_MATERIAL_SELECTION_OBJECTIVES = frozenset(
    {"semantic_compatible_visual", "visual_similarity"}
)
_BASE_PAINT_PRIMARY_RE = re.compile(
    r"^mdl:Miscellaneous/Paint_(?:Gloss|Matte|Satin)\.mdl#"
    r"Paint_(?:Gloss|Matte|Satin)$"
)
_BASE_PAINT_EQUIVALENCE_RE = re.compile(
    r"^mdl:Miscellaneous/(Paint_(?:Gloss|Matte|Satin))"
    r"(?:_Finish)?\.mdl#Paint_(?:Gloss|Matte|Satin)(?:_Finish)?$"
)
_SURFACE_INTERPRETATIONS = (
    "conversion_coating",
    "applied_paint",
    "bare_metal",
)
_COATING_PHYSICS_TEMPLATES = (
    "painted_engineering_metal",
    "generic_applied_paint",
    "conversion_coating",
)
_FAMILY_HINTS = frozenset(
    {"metal", "plastic", "rubber", "glass", "fabric", "ceramic", "other", "unknown"}
)
_BASE_COLORS = frozenset(
    {
        "white",
        "yellow",
        "cyan",
        "pink",
        "black",
        "gray",
        "silver",
        "red",
        "blue",
        "green",
        "orange",
        "brown",
        "clear",
        "other",
        "unknown",
    }
)
_FINISH_HINTS = frozenset(
    {
        "painted",
        "bare",
        "matte",
        "glossy",
        "brushed",
        "polished",
        "translucent",
        "rough",
        "smooth",
        "other",
        "unknown",
    }
)


class ConfidenceGateError(ValueError):
    """Raised when an input contract is malformed or internally inconsistent."""


@dataclass(frozen=True)
class GatePolicy:
    """Strict defaults suitable for automatic visual-material application."""

    auto_model_confidence: float = 0.90
    review_model_confidence: float = 0.60
    auto_mapping_confidence: float = 0.90
    review_mapping_confidence: float = 0.60
    independently_validated_auto_confidence: float = 0.85
    minimum_independent_references: int = 2
    review_visible_pixels: int = 64
    auto_visible_pixels: int = 256
    auto_visible_view_count: int = 2
    visible_view_pixel_floor: int = 64
    isolated_source_visible_pixels: int = 12
    isolated_auto_visible_view_count: int = 2
    auto_material_choice_confidence: float = 0.85
    review_material_choice_confidence: float = 0.60
    minimum_candidate_margin: float = 0.15

    def validate(self) -> None:
        confidence_fields = (
            "auto_model_confidence",
            "review_model_confidence",
            "auto_mapping_confidence",
            "review_mapping_confidence",
            "independently_validated_auto_confidence",
            "auto_material_choice_confidence",
            "review_material_choice_confidence",
            "minimum_candidate_margin",
        )
        for name in confidence_fields:
            value = getattr(self, name)
            _confidence(value, f"policy.{name}")
        if self.review_model_confidence > self.auto_model_confidence:
            raise ConfidenceGateError(
                "policy review_model_confidence cannot exceed auto threshold"
            )
        if self.review_mapping_confidence > self.auto_mapping_confidence:
            raise ConfidenceGateError(
                "policy review_mapping_confidence cannot exceed auto threshold"
            )
        if not (
            max(
                self.review_model_confidence,
                self.review_mapping_confidence,
            )
            <= self.independently_validated_auto_confidence
            <= min(
                self.auto_model_confidence,
                self.auto_mapping_confidence,
            )
        ):
            raise ConfidenceGateError(
                "policy independently_validated_auto_confidence must be between "
                "the review and ordinary auto thresholds"
            )
        if (
            self.review_material_choice_confidence
            > self.auto_material_choice_confidence
        ):
            raise ConfidenceGateError(
                "policy review material confidence cannot exceed auto threshold"
            )
        for name in (
            "minimum_independent_references",
            "review_visible_pixels",
            "auto_visible_pixels",
            "auto_visible_view_count",
            "visible_view_pixel_floor",
            "isolated_source_visible_pixels",
            "isolated_auto_visible_view_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ConfidenceGateError(f"policy.{name} must be a positive integer")
        if self.review_visible_pixels > self.auto_visible_pixels:
            raise ConfidenceGateError(
                "policy review_visible_pixels cannot exceed auto_visible_pixels"
            )


def _object(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfidenceGateError(f"{label} must be an object")
    return value


def _array(value: Any, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ConfidenceGateError(f"{label} must be an array")
    return value


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfidenceGateError(f"{label} must be a non-empty string")
    return value.strip()


def _part_id(value: Any, label: str) -> str:
    identifier = _string(value, label)
    if not _PART_ID_RE.fullmatch(identifier):
        raise ConfidenceGateError(f"{label} must use P followed by 4..8 digits")
    return identifier


def _confidence(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0.0 <= value <= 1.0
    ):
        raise ConfidenceGateError(f"{label} must be a finite number from 0 to 1")
    return float(value)


def _finite_number(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ConfidenceGateError(f"{label} must be a finite number")
    return float(value)


def _nonnegative_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ConfidenceGateError(f"{label} must be a non-negative integer")
    return value


def _unique_strings(value: Any, label: str) -> list[str]:
    records = _array(value, label)
    result = [_string(item, f"{label}[{index}]") for index, item in enumerate(records)]
    if len(set(result)) != len(result):
        raise ConfidenceGateError(f"{label} contains duplicate values")
    return result


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ConfidenceGateError(f"{label} must be boolean")
    return value


def _exact_fields(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ConfidenceGateError(
            f"{label} fields are invalid; "
            f"unexpected={sorted(actual - expected)}, "
            f"missing={sorted(expected - actual)}"
        )


def _reference_view_id(view_id: str) -> bool:
    lowered = view_id.lower()
    return not (
        lowered.startswith(_GEOMETRY_VIEW_PREFIXES)
        or "support" in lowered
        or "contact" in lowered
        or "sheet" in lowered
    )


def _registry_parts(registry: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    if registry.get("schema_version") != REGISTRY_SCHEMA_VERSION:
        raise ConfidenceGateError(
            f"rendered registry schema_version must be {REGISTRY_SCHEMA_VERSION!r}"
        )
    records = _array(registry.get("parts"), "rendered_registry.parts")
    if not records:
        raise ConfidenceGateError("rendered_registry.parts cannot be empty")
    parts: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(records):
        part = _object(raw, f"rendered_registry.parts[{index}]")
        part_id = _part_id(
            part.get("part_id"), f"rendered_registry.parts[{index}].part_id"
        )
        if part_id in parts:
            raise ConfidenceGateError(f"duplicate registry part_id: {part_id}")
        renders = _array(part.get("renders", []), f"registry part {part_id}.renders")
        seen_views: set[str] = set()
        canonical_renders: list[dict[str, Any]] = []
        for render_index, raw_render in enumerate(renders):
            render = _object(
                raw_render, f"registry part {part_id}.renders[{render_index}]"
            )
            view_id = _string(
                render.get("view_id"), f"registry part {part_id} render view_id"
            )
            if view_id in seen_views:
                raise ConfidenceGateError(
                    f"registry part {part_id} has duplicate render view_id: {view_id}"
                )
            seen_views.add(view_id)
            pixels = _nonnegative_integer(
                render.get("visible_pixels"),
                f"registry part {part_id} render {view_id}.visible_pixels",
            )
            canonical_renders.append({"view_id": view_id, "visible_pixels": pixels})
        isolated = part.get("isolated_evidence")
        canonical_isolated: dict[str, Any] | None = None
        if isolated is not None:
            isolated = _object(
                isolated,
                f"registry part {part_id}.isolated_evidence",
            )
            if isolated.get("schema_version") != ISOLATED_EVIDENCE_SCHEMA_VERSION:
                raise ConfidenceGateError(
                    f"registry part {part_id} isolated evidence schema is unsupported"
                )
            selected_views = _unique_strings(
                isolated.get("selected_view_ids"),
                f"registry part {part_id} isolated selected_view_ids",
            )
            source_by_view_raw = _object(
                isolated.get("source_visible_pixels_by_view"),
                f"registry part {part_id} isolated source pixels",
            )
            normalized_by_view_raw = _object(
                isolated.get("normalized_visible_pixels_by_view"),
                f"registry part {part_id} isolated normalized pixels",
            )
            if set(source_by_view_raw) != set(selected_views) or set(
                normalized_by_view_raw
            ) != set(selected_views):
                raise ConfidenceGateError(
                    f"registry part {part_id} isolated evidence view sets differ"
                )
            source_by_view = {
                view_id: _nonnegative_integer(
                    source_by_view_raw[view_id],
                    f"registry part {part_id} isolated source {view_id}",
                )
                for view_id in selected_views
            }
            normalized_by_view = {
                view_id: _nonnegative_integer(
                    normalized_by_view_raw[view_id],
                    f"registry part {part_id} isolated normalized {view_id}",
                )
                for view_id in selected_views
            }
            raw_render_pixels = {
                render["view_id"]: render["visible_pixels"]
                for render in canonical_renders
            }
            if any(
                view_id not in raw_render_pixels
                or raw_render_pixels[view_id] != source_by_view[view_id]
                for view_id in selected_views
            ):
                raise ConfidenceGateError(
                    f"registry part {part_id} isolated/source projection pixels differ"
                )
            source_floor = _nonnegative_integer(
                isolated.get("source_pixel_floor"),
                f"registry part {part_id} isolated source_pixel_floor",
            )
            if source_floor < 1:
                raise ConfidenceGateError(
                    f"registry part {part_id} isolated source_pixel_floor must be positive"
                )
            source_max = max(source_by_view.values(), default=0)
            normalized_max = max(normalized_by_view.values(), default=0)
            source_evidence_views = sorted(
                view_id
                for view_id, pixels in source_by_view.items()
                if pixels >= source_floor
            )
            declared_evidence_views = _unique_strings(
                isolated.get("source_evidence_view_ids"),
                f"registry part {part_id} isolated source_evidence_view_ids",
            )
            if (
                isolated.get("source_max_visible_pixels") != source_max
                or isolated.get("normalized_max_visible_pixels") != normalized_max
                or isolated.get("source_evidence_view_count")
                != len(source_evidence_views)
                or declared_evidence_views != source_evidence_views
                or isolated.get("material_neutralized") is not True
                or isolated.get("background_removed") is not True
            ):
                raise ConfidenceGateError(
                    f"registry part {part_id} isolated evidence summary is inconsistent"
                )
            digest = _string(
                isolated.get("sha256"),
                f"registry part {part_id} isolated sha256",
            )
            if not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ConfidenceGateError(
                    f"registry part {part_id} isolated sha256 is invalid"
                )
            canonical_isolated = {
                "schema_version": ISOLATED_EVIDENCE_SCHEMA_VERSION,
                "sha256": digest,
                "source_visible_pixels_by_view": source_by_view,
                "normalized_visible_pixels_by_view": normalized_by_view,
                "source_max_visible_pixels": source_max,
                "normalized_max_visible_pixels": normalized_max,
                "source_evidence_view_count": len(source_evidence_views),
                "source_evidence_view_ids": source_evidence_views,
                "source_pixel_floor": source_floor,
                "material_neutralized": True,
                "background_removed": True,
            }
        parts[part_id] = {
            "part_id": part_id,
            "renders": canonical_renders,
            "isolated_evidence": canonical_isolated,
        }
    declared_count = registry.get("part_count")
    if declared_count is not None and declared_count != len(parts):
        raise ConfidenceGateError(
            "rendered registry part_count does not match the parts array"
        )
    return parts


def _staged_parts(
    staged_result: Mapping[str, Any], registry_ids: set[str]
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    if staged_result.get("schema_version") != STAGED_SCHEMA_VERSION:
        raise ConfidenceGateError(
            f"staged_result.schema_version must be {STAGED_SCHEMA_VERSION!r}"
        )
    material_plan = _object(
        staged_result.get("material_plan"), "staged_result.material_plan"
    )
    if material_plan.get("schema_version") != MATERIAL_PLAN_SCHEMA_VERSION:
        raise ConfidenceGateError("staged material plan has unsupported schema_version")
    raw_assignments = _array(
        material_plan.get("assignments"), "material_plan.assignments"
    )
    raw_unknown = _array(
        staged_result.get("unknown_parts"), "staged_result.unknown_parts"
    )
    assignments: dict[str, dict[str, Any]] = {}
    unknown: dict[str, str] = {}
    for index, raw in enumerate(raw_assignments):
        assignment = _object(raw, f"material_plan.assignments[{index}]")
        part_id = _part_id(assignment.get("part_id"), f"assignment[{index}].part_id")
        if part_id in assignments or part_id in unknown:
            raise ConfidenceGateError(f"duplicate staged part_id: {part_id}")
        material_id = _string(
            assignment.get("material_id"), f"assignment {part_id}.material_id"
        )
        confidence = _confidence(
            assignment.get("confidence"), f"assignment {part_id}.confidence"
        )
        status = assignment.get("status")
        if status not in {"auto", "review"}:
            raise ConfidenceGateError(
                f"assignment {part_id}.status must be 'auto' or 'review'"
            )
        evidence_views = _unique_strings(
            assignment.get("evidence_views", []), f"assignment {part_id}.evidence_views"
        )
        candidate_margin = assignment.get("candidate_margin")
        if candidate_margin is not None:
            candidate_margin = _confidence(
                candidate_margin, f"assignment {part_id}.candidate_margin"
            )
        assignments[part_id] = {
            "part_id": part_id,
            "material_id": material_id,
            "semantic": assignment.get("semantic"),
            "confidence": confidence,
            "status": status,
            "evidence_views": evidence_views,
            "candidate_margin": candidate_margin,
            "source_assignment": dict(assignment),
        }
    for index, raw in enumerate(raw_unknown):
        record = _object(raw, f"unknown_parts[{index}]")
        part_id = _part_id(record.get("part_id"), f"unknown_parts[{index}].part_id")
        if part_id in assignments or part_id in unknown:
            raise ConfidenceGateError(f"duplicate staged part_id: {part_id}")
        unknown[part_id] = _string(
            record.get("reason_code"), f"unknown part {part_id}.reason_code"
        )
    staged_ids = set(assignments) | set(unknown)
    if staged_ids != registry_ids:
        raise ConfidenceGateError(
            "staged result does not exactly cover rendered registry; "
            f"missing={sorted(registry_ids - staged_ids)}, "
            f"unexpected={sorted(staged_ids - registry_ids)}"
        )
    return assignments, unknown


def _batch_mappings(
    batches: Sequence[Mapping[str, Any]] | None,
    registry_ids: set[str],
) -> dict[str, dict[str, Any]]:
    if batches is None:
        return {}
    result: dict[str, dict[str, Any]] = {}
    seen_batch_ids: set[str] = set()
    for batch_index, raw_batch in enumerate(batches):
        batch = _object(raw_batch, f"batches[{batch_index}]")
        if batch.get("schema_version") != "qwen-part-palette-map/v1":
            raise ConfidenceGateError(
                f"batches[{batch_index}] has unsupported schema_version"
            )
        batch_id = _string(batch.get("batch_id"), f"batches[{batch_index}].batch_id")
        if batch_id in seen_batch_ids:
            raise ConfidenceGateError(f"duplicate batch_id: {batch_id}")
        seen_batch_ids.add(batch_id)
        mappings = _array(batch.get("mappings"), f"batch {batch_id}.mappings")
        for index, raw_mapping in enumerate(mappings):
            mapping = _object(raw_mapping, f"batch {batch_id}.mappings[{index}]")
            part_id = _part_id(
                mapping.get("part_id"), f"batch {batch_id} mapping part_id"
            )
            if part_id not in registry_ids:
                raise ConfidenceGateError(
                    f"batch {batch_id} contains unknown part_id: {part_id}"
                )
            if part_id in result:
                raise ConfidenceGateError(
                    f"duplicate batch mapping for part_id: {part_id}"
                )
            status = mapping.get("status")
            if status not in {"matched", "review", "unknown"}:
                raise ConfidenceGateError(
                    f"batch mapping {part_id}.status is unsupported: {status!r}"
                )
            confidence = _confidence(
                mapping.get("mapping_confidence"),
                f"batch mapping {part_id}.mapping_confidence",
            )
            group_id = mapping.get("group_id")
            if group_id is not None:
                group_id = _string(group_id, f"batch mapping {part_id}.group_id")
            reason_code = _string(
                mapping.get("reason_code"), f"batch mapping {part_id}.reason_code"
            )
            result[part_id] = {
                "batch_id": batch_id,
                "group_id": group_id,
                "status": status,
                "confidence": confidence,
                "reason_code": reason_code,
            }
    return result


def _independent_mapping_validations(
    audit: Mapping[str, Any] | None,
    *,
    registry_ids: set[str],
    mappings: Mapping[str, Mapping[str, Any]],
    minimum_confidence: float,
    maximum_conflict_confidence: float,
) -> dict[str, dict[str, Any]]:
    """Validate the stronger spatial/semantic gate before honoring its result."""

    if audit is None:
        return {}
    document = _object(audit, "independent_validation_audit")
    if document.get("schema_version") != SPATIAL_GATE_AUDIT_SCHEMA_VERSION:
        raise ConfidenceGateError(
            "independent validation audit has an unsupported schema_version"
        )
    audit_policy = _object(
        document.get("policy"), "independent_validation_audit.policy"
    )
    semantic_support_threshold = _confidence(
        audit_policy.get("minimum_semantic_confidence"),
        "independent validation minimum_semantic_confidence",
    )
    semantic_conflict_threshold = _confidence(
        audit_policy.get("minimum_semantic_conflict_confidence"),
        "independent validation minimum_semantic_conflict_confidence",
    )
    if (
        semantic_support_threshold < minimum_confidence
        or semantic_conflict_threshold > maximum_conflict_confidence
    ):
        raise ConfidenceGateError(
            "independent validation audit uses unsafe semantic thresholds"
        )
    decisions = _array(
        document.get("decisions"), "independent_validation_audit.decisions"
    )
    result: dict[str, dict[str, Any]] = {}
    seen_part_ids: set[str] = set()
    for index, raw_decision in enumerate(decisions):
        decision = _object(
            raw_decision, f"independent_validation_audit.decisions[{index}]"
        )
        part_id = _part_id(
            decision.get("part_id"),
            f"independent validation decision[{index}].part_id",
        )
        if part_id not in registry_ids:
            raise ConfidenceGateError(
                f"independent validation contains unknown part_id: {part_id}"
            )
        if part_id in seen_part_ids:
            raise ConfidenceGateError(
                f"independent validation contains duplicate part_id: {part_id}"
            )
        seen_part_ids.add(part_id)
        if decision.get("decision") != "kept_auto":
            continue
        group_id = _string(
            decision.get("output_group_id"),
            f"independent validation {part_id}.output_group_id",
        )
        if decision.get("output_status") != "matched":
            raise ConfidenceGateError(
                f"independent validation {part_id} kept_auto is not matched"
            )
        output_confidence = _confidence(
            decision.get("output_confidence"),
            f"independent validation {part_id}.output_confidence",
        )
        if output_confidence < minimum_confidence:
            raise ConfidenceGateError(
                f"independent validation {part_id} confidence is below the "
                "validated-auto threshold"
            )
        mapping = mappings.get(part_id)
        if (
            mapping is None
            or mapping.get("batch_id") != decision.get("batch_id")
            or mapping.get("status") != "matched"
            or mapping.get("group_id") != group_id
            or not math.isclose(
                float(mapping.get("confidence", -1.0)),
                output_confidence,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        ):
            raise ConfidenceGateError(
                f"independent validation {part_id} does not match the supplied "
                "post-gate batch mapping"
            )

        lanes = _array(
            decision.get("validation_lanes"),
            f"independent validation {part_id}.validation_lanes",
        )
        if (
            not lanes
            or any(
                lane not in {"spatial_projection", "semantic_multiview"}
                for lane in lanes
            )
            or len(set(lanes)) != len(lanes)
        ):
            raise ConfidenceGateError(
                f"independent validation {part_id} has invalid validation lanes"
            )
        conflict_fields = (
            "conflicting_view_ids",
            "semantic_conflicting_view_ids",
            "semantic_unresolved_view_ids",
            "semantic_multi_material_view_ids",
            "semantic_nondeterministic_content_cluster_ids",
        )
        if any(
            _array(
                decision.get(field, []),
                f"independent validation {part_id}.{field}",
            )
            for field in conflict_fields
        ):
            raise ConfidenceGateError(
                f"independent validation {part_id} contains a material conflict"
            )

        if "spatial_projection" in lanes:
            supporting_views = _array(
                decision.get("supporting_view_ids"),
                f"independent validation {part_id}.supporting_view_ids",
            )
            if any(
                not isinstance(view_id, str) or not view_id
                for view_id in supporting_views
            ):
                raise ConfidenceGateError(
                    f"independent validation {part_id} has invalid spatial support IDs"
                )
            if len(set(supporting_views)) < 2:
                raise ConfidenceGateError(
                    f"independent validation {part_id} has insufficient spatial support"
                )
        else:
            supporting_views = []
        if "semantic_multiview" in lanes:
            semantic_supporting_views = _array(
                decision.get("semantic_supporting_view_ids"),
                f"independent validation {part_id}.semantic_supporting_view_ids",
            )
            if any(
                not isinstance(view_id, str) or not view_id
                for view_id in semantic_supporting_views
            ):
                raise ConfidenceGateError(
                    f"independent validation {part_id} has invalid semantic support IDs"
                )
            if len(set(semantic_supporting_views)) < 2:
                raise ConfidenceGateError(
                    f"independent validation {part_id} has insufficient distinct "
                    "semantic support views"
                )
            for field in (
                "semantic_supporting_pixel_sha256s",
                "semantic_supporting_content_cluster_ids",
                "semantic_supporting_pose_cluster_ids",
            ):
                values = _array(
                    decision.get(field),
                    f"independent validation {part_id}.{field}",
                )
                if any(not isinstance(value, str) or not value for value in values):
                    raise ConfidenceGateError(
                        f"independent validation {part_id} has invalid values "
                        f"in {field}"
                    )
                if len(set(values)) < 2:
                    raise ConfidenceGateError(
                        f"independent validation {part_id} has insufficient "
                        f"distinct semantic evidence in {field}"
                    )
        else:
            semantic_supporting_views = []
        result[part_id] = {
            "batch_id": mapping["batch_id"],
            "group_id": group_id,
            "confidence": output_confidence,
            "threshold_profile": "independently_validated",
            "auto_confidence_threshold": minimum_confidence,
            "validation_lanes": list(lanes),
            "supporting_view_ids": sorted(
                set(supporting_views) | set(semantic_supporting_views)
            ),
        }
    return result


def _review_assignments(
    review_plan: Mapping[str, Any] | None, registry_ids: set[str]
) -> dict[str, dict[str, Any]] | None:
    if review_plan is None:
        return None
    if review_plan.get("schema_version") != MATERIAL_PLAN_SCHEMA_VERSION:
        raise ConfidenceGateError("review plan has unsupported schema_version")
    records = _array(review_plan.get("assignments"), "review_plan.assignments")
    result: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(records):
        assignment = _object(raw, f"review_plan.assignments[{index}]")
        part_id = _part_id(
            assignment.get("part_id"), f"review assignment[{index}].part_id"
        )
        if part_id not in registry_ids:
            raise ConfidenceGateError(
                f"review plan contains unknown part_id: {part_id}"
            )
        if part_id in result:
            raise ConfidenceGateError(
                f"review plan contains duplicate part_id: {part_id}"
            )
        material_id = _string(
            assignment.get("material_id"), f"review assignment {part_id}.material_id"
        )
        subsets = assignment.get("face_subsets", [])
        subsets = _array(subsets, f"review assignment {part_id}.face_subsets")
        result[part_id] = {
            "material_id": material_id,
            "status": assignment.get("status"),
            "face_subset_count": len(subsets),
        }
        if result[part_id]["status"] != "approved":
            raise ConfidenceGateError(
                f"review assignment {part_id}.status must be 'approved'"
            )
    return result


def _choice_confidence(choice: Mapping[str, Any], label: str) -> float:
    return _confidence(choice.get("confidence"), f"{label}.confidence")


def _selection_group(
    value: Any,
    *,
    group_id: str,
) -> dict[str, Any]:
    label = f"material choice {group_id}.selection_group"
    group = _object(value, label)
    expected_fields = {
        "group_id",
        "family_hint",
        "base_color",
        "finish_hint",
        "visual_description",
        "boxes",
        "confidence",
    }
    if "material_selection_objective" in group:
        expected_fields.add("material_selection_objective")
    _exact_fields(group, expected_fields, label)
    selected_group_id = _string(group["group_id"], f"{label}.group_id")
    if selected_group_id != group_id:
        raise ConfidenceGateError(
            f"{label}.group_id does not match its material choice key"
        )
    family_hint = _string(group["family_hint"], f"{label}.family_hint")
    if family_hint not in _FAMILY_HINTS:
        raise ConfidenceGateError(
            f"{label}.family_hint must be one of {sorted(_FAMILY_HINTS)}"
        )
    base_color = _string(group["base_color"], f"{label}.base_color")
    if base_color not in _BASE_COLORS:
        raise ConfidenceGateError(
            f"{label}.base_color must be one of {sorted(_BASE_COLORS)}"
        )
    finish_hint = _string(group["finish_hint"], f"{label}.finish_hint")
    if finish_hint not in _FINISH_HINTS:
        raise ConfidenceGateError(
            f"{label}.finish_hint must be one of {sorted(_FINISH_HINTS)}"
        )
    boxes = _array(group["boxes"], f"{label}.boxes")
    if not 1 <= len(boxes) <= 4:
        raise ConfidenceGateError(f"{label}.boxes must contain 1..4 boxes")
    canonical_boxes: list[list[int]] = []
    for index, value in enumerate(boxes):
        box_label = f"{label}.boxes[{index}]"
        box = _array(value, box_label)
        if len(box) != 4:
            raise ConfidenceGateError(
                f"{box_label} must contain four normalized coordinates"
            )
        if any(
            isinstance(coordinate, bool)
            or not isinstance(coordinate, int)
            or not 0 <= coordinate <= 1000
            for coordinate in box
        ):
            raise ConfidenceGateError(
                f"{box_label} coordinates must be integers from 0 to 1000"
            )
        x0, y0, x1, y1 = box
        if x0 >= x1 or y0 >= y1:
            raise ConfidenceGateError(f"{box_label} must have positive area")
        canonical_boxes.append(list(box))
    material_selection_objective = _string(
        group.get(
            "material_selection_objective",
            "semantic_compatible_visual",
        ),
        f"{label}.material_selection_objective",
    )
    if material_selection_objective not in _MATERIAL_SELECTION_OBJECTIVES:
        raise ConfidenceGateError(
            f"{label}.material_selection_objective must be one of "
            f"{sorted(_MATERIAL_SELECTION_OBJECTIVES)}"
        )
    return {
        "group_id": selected_group_id,
        "family_hint": family_hint,
        "base_color": base_color,
        "finish_hint": finish_hint,
        "visual_description": _string(
            group["visual_description"], f"{label}.visual_description"
        ),
        "boxes": canonical_boxes,
        "confidence": _confidence(group["confidence"], f"{label}.confidence"),
        "material_selection_objective": material_selection_objective,
    }


def _semantic_reliability(
    value: Any,
    *,
    selection_group: Mapping[str, Any],
    group_id: str,
) -> dict[str, Any]:
    label = f"material choice {group_id}.semantic_reliability"
    reliability = _object(value, label)
    _exact_fields(
        reliability,
        {
            "policy",
            "finish_hint",
            "visual_description",
            "selection_context_modified",
            "canonical_group_preserved",
            "reason_codes",
        },
        label,
    )

    policy_label = f"{label}.policy"
    policy = _object(reliability["policy"], policy_label)
    _exact_fields(
        policy,
        {
            "automatic_confidence_threshold",
            "review_confidence_threshold",
            "minimum_independent_support_views",
            "unresolved_values",
        },
        policy_label,
    )
    automatic_threshold = _confidence(
        policy["automatic_confidence_threshold"],
        f"{policy_label}.automatic_confidence_threshold",
    )
    review_threshold = _confidence(
        policy["review_confidence_threshold"],
        f"{policy_label}.review_confidence_threshold",
    )
    minimum_support = policy["minimum_independent_support_views"]
    if (
        isinstance(minimum_support, bool)
        or not isinstance(minimum_support, int)
        or minimum_support < 1
    ):
        raise ConfidenceGateError(
            f"{policy_label}.minimum_independent_support_views "
            "must be a positive integer"
        )
    unresolved_values = _unique_strings(
        policy["unresolved_values"], f"{policy_label}.unresolved_values"
    )
    expected_policy = (
        math.isclose(
            automatic_threshold,
            _SEMANTIC_AUTO_THRESHOLD,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and math.isclose(
            review_threshold,
            _SEMANTIC_REVIEW_THRESHOLD,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and minimum_support == _SEMANTIC_MINIMUM_SUPPORT_VIEWS
        and unresolved_values == sorted(_UNRESOLVED_SEMANTICS)
    )
    if not expected_policy:
        raise ConfidenceGateError(
            f"{policy_label} does not match the staged semantic policy"
        )

    finish_label = f"{label}.finish_hint"
    finish = _object(reliability["finish_hint"], finish_label)
    _exact_fields(
        finish,
        {
            "canonical_value",
            "selection_value",
            "reliable",
            "supporting_view_ids",
            "conflicting_view_ids",
            "conflicting_values",
            "maximum_support_confidence",
            "multiview_confirmed",
            "high_confidence_confirmed",
        },
        finish_label,
    )
    canonical_finish = _string(
        finish["canonical_value"], f"{finish_label}.canonical_value"
    )
    if canonical_finish not in _FINISH_HINTS:
        raise ConfidenceGateError(
            f"{finish_label}.canonical_value must be one of {sorted(_FINISH_HINTS)}"
        )
    selected_finish = _string(
        finish["selection_value"], f"{finish_label}.selection_value"
    )
    if selected_finish not in _FINISH_HINTS:
        raise ConfidenceGateError(
            f"{finish_label}.selection_value must be one of {sorted(_FINISH_HINTS)}"
        )
    finish_reliable = _boolean(finish["reliable"], f"{finish_label}.reliable")
    supporting_view_ids = _unique_strings(
        finish["supporting_view_ids"], f"{finish_label}.supporting_view_ids"
    )
    conflicting_view_ids = _unique_strings(
        finish["conflicting_view_ids"], f"{finish_label}.conflicting_view_ids"
    )
    conflicting_values = _unique_strings(
        finish["conflicting_values"], f"{finish_label}.conflicting_values"
    )
    if any(
        value in _UNRESOLVED_SEMANTICS or value == canonical_finish
        for value in conflicting_values
    ):
        raise ConfidenceGateError(
            f"{finish_label}.conflicting_values contains a non-conflicting value"
        )
    if bool(conflicting_view_ids) != bool(conflicting_values):
        raise ConfidenceGateError(
            f"{finish_label} conflict view/value evidence is inconsistent"
        )
    maximum_support_confidence = _confidence(
        finish["maximum_support_confidence"],
        f"{finish_label}.maximum_support_confidence",
    )
    if not supporting_view_ids and maximum_support_confidence != 0.0:
        raise ConfidenceGateError(
            f"{finish_label}.maximum_support_confidence must be zero "
            "without supporting views"
        )
    if "canonical" in supporting_view_ids and (
        supporting_view_ids != ["canonical"]
        or conflicting_view_ids
        or not math.isclose(
            maximum_support_confidence,
            selection_group["confidence"],
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise ConfidenceGateError(
            f"{finish_label} canonical fallback evidence is inconsistent"
        )
    multiview_confirmed = _boolean(
        finish["multiview_confirmed"], f"{finish_label}.multiview_confirmed"
    )
    expected_multiview = len(supporting_view_ids) >= minimum_support
    if multiview_confirmed != expected_multiview:
        raise ConfidenceGateError(f"{finish_label}.multiview_confirmed is inconsistent")
    high_confidence_confirmed = _boolean(
        finish["high_confidence_confirmed"],
        f"{finish_label}.high_confidence_confirmed",
    )
    expected_high_confidence = maximum_support_confidence >= automatic_threshold
    if high_confidence_confirmed != expected_high_confidence:
        raise ConfidenceGateError(
            f"{finish_label}.high_confidence_confirmed is inconsistent"
        )
    expected_finish_reliable = (
        canonical_finish not in _UNRESOLVED_SEMANTICS
        and not conflicting_view_ids
        and (multiview_confirmed or high_confidence_confirmed)
    )
    if finish_reliable != expected_finish_reliable:
        raise ConfidenceGateError(f"{finish_label}.reliable is inconsistent")
    expected_selected_finish = canonical_finish if finish_reliable else "other"
    if selected_finish != expected_selected_finish:
        raise ConfidenceGateError(f"{finish_label}.selection_value is inconsistent")
    if selection_group["finish_hint"] != selected_finish:
        raise ConfidenceGateError(
            f"{finish_label}.selection_value does not match selection_group"
        )

    description_label = f"{label}.visual_description"
    description = _object(reliability["visual_description"], description_label)
    _exact_fields(
        description,
        {
            "canonical_value",
            "selection_value",
            "reliable",
            "canonical_confidence",
            "requires_reliable_finish",
        },
        description_label,
    )
    canonical_description = _string(
        description["canonical_value"], f"{description_label}.canonical_value"
    )
    selected_description = _string(
        description["selection_value"], f"{description_label}.selection_value"
    )
    description_reliable = _boolean(
        description["reliable"], f"{description_label}.reliable"
    )
    canonical_confidence = _confidence(
        description["canonical_confidence"],
        f"{description_label}.canonical_confidence",
    )
    if not _boolean(
        description["requires_reliable_finish"],
        f"{description_label}.requires_reliable_finish",
    ):
        raise ConfidenceGateError(
            f"{description_label}.requires_reliable_finish must be true"
        )
    expected_description_reliable = (
        finish_reliable and canonical_confidence >= automatic_threshold
    )
    if description_reliable != expected_description_reliable:
        raise ConfidenceGateError(f"{description_label}.reliable is inconsistent")
    if description_reliable and selected_description != canonical_description:
        raise ConfidenceGateError(
            f"{description_label}.selection_value must preserve reliable evidence"
        )
    if selection_group["visual_description"] != selected_description:
        raise ConfidenceGateError(
            f"{description_label}.selection_value does not match selection_group"
        )
    if not math.isclose(
        selection_group["confidence"],
        canonical_confidence,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ConfidenceGateError(
            f"{description_label}.canonical_confidence does not match selection_group"
        )

    selection_context_modified = _boolean(
        reliability["selection_context_modified"],
        f"{label}.selection_context_modified",
    )
    expected_modified = (
        canonical_finish != selected_finish
        or canonical_description != selected_description
    )
    if selection_context_modified != expected_modified:
        raise ConfidenceGateError(f"{label}.selection_context_modified is inconsistent")
    if not _boolean(
        reliability["canonical_group_preserved"],
        f"{label}.canonical_group_preserved",
    ):
        raise ConfidenceGateError(f"{label}.canonical_group_preserved must be true")

    reason_codes = _unique_strings(reliability["reason_codes"], f"{label}.reason_codes")
    expected_reasons: list[str] = []
    if canonical_finish in _UNRESOLVED_SEMANTICS:
        expected_reasons.append("finish_hint_already_unresolved")
    if conflicting_view_ids:
        expected_reasons.append("resolved_finish_conflict")
    if (
        canonical_finish not in _UNRESOLVED_SEMANTICS
        and not conflicting_view_ids
        and not supporting_view_ids
    ):
        expected_reasons.append("no_resolved_finish_support")
    if (
        len(supporting_view_ids) == 1
        and not high_confidence_confirmed
        and not conflicting_view_ids
    ):
        expected_reasons.append("single_review_confidence_finish_source")
    if multiview_confirmed and not conflicting_view_ids:
        expected_reasons.append("finish_confirmed_by_independent_views")
    elif high_confidence_confirmed and not conflicting_view_ids:
        expected_reasons.append("finish_confirmed_by_high_confidence_source")
    if canonical_confidence < automatic_threshold:
        expected_reasons.append("visual_description_below_auto_confidence")
    if not finish_reliable:
        expected_reasons.append("visual_description_depends_on_unreliable_finish")
    if reason_codes != expected_reasons:
        raise ConfidenceGateError(f"{label}.reason_codes are inconsistent")

    return {
        "policy": {
            "automatic_confidence_threshold": automatic_threshold,
            "review_confidence_threshold": review_threshold,
            "minimum_independent_support_views": minimum_support,
            "unresolved_values": unresolved_values,
        },
        "finish_hint": {
            "canonical_value": canonical_finish,
            "selection_value": selected_finish,
            "reliable": finish_reliable,
            "supporting_view_ids": supporting_view_ids,
            "conflicting_view_ids": conflicting_view_ids,
            "conflicting_values": conflicting_values,
            "maximum_support_confidence": maximum_support_confidence,
            "multiview_confirmed": multiview_confirmed,
            "high_confidence_confirmed": high_confidence_confirmed,
        },
        "visual_description": {
            "canonical_value": canonical_description,
            "selection_value": selected_description,
            "reliable": description_reliable,
            "canonical_confidence": canonical_confidence,
            "requires_reliable_finish": True,
        },
        "selection_context_modified": selection_context_modified,
        "canonical_group_preserved": True,
        "reason_codes": reason_codes,
    }


def _semantic_retrieval_context(
    record: Mapping[str, Any],
    audit: Mapping[str, Any],
    *,
    group_id: str,
) -> dict[str, Any]:
    if "selection_group" not in record or "semantic_reliability" not in record:
        raise ConfidenceGateError(
            f"material choice {group_id} semantic retrieval context "
            "must be supplied together"
        )
    selection_group = _selection_group(record["selection_group"], group_id=group_id)
    top_level_reliability = record["semantic_reliability"]
    if audit["semantic_reliability"] != top_level_reliability:
        raise ConfidenceGateError(
            f"material choice {group_id} retrieval semantic_reliability "
            "does not match the material choice audit"
        )
    semantic_reliability = _semantic_reliability(
        top_level_reliability,
        selection_group=selection_group,
        group_id=group_id,
    )
    finish_evidence_used = _boolean(
        audit["finish_evidence_used"],
        f"material choice {group_id} retrieval finish_evidence_used",
    )
    description_evidence_used = _boolean(
        audit["description_evidence_used"],
        f"material choice {group_id} retrieval description_evidence_used",
    )
    intrinsic_surface_ambiguity = _boolean(
        audit["intrinsic_surface_ambiguity"],
        f"material choice {group_id} retrieval intrinsic_surface_ambiguity",
    )
    if finish_evidence_used != semantic_reliability["finish_hint"]["reliable"]:
        raise ConfidenceGateError(
            f"material choice {group_id} retrieval finish_evidence_used is inconsistent"
        )
    if (
        description_evidence_used
        != semantic_reliability["visual_description"]["reliable"]
    ):
        raise ConfidenceGateError(
            f"material choice {group_id} retrieval description_evidence_used "
            "is inconsistent"
        )
    expected_ambiguity = (
        selection_group["family_hint"] == "metal"
        and selection_group["base_color"] in _INTRINSIC_METAL_COLORS
        and not finish_evidence_used
    )
    if intrinsic_surface_ambiguity != expected_ambiguity:
        raise ConfidenceGateError(
            f"material choice {group_id} retrieval intrinsic_surface_ambiguity "
            "is inconsistent"
        )
    return {
        "semantic_reliability": semantic_reliability,
        "finish_evidence_used": finish_evidence_used,
        "description_evidence_used": description_evidence_used,
        "intrinsic_surface_ambiguity": intrinsic_surface_ambiguity,
    }


def _surface_interpretation_context(
    value: Any,
    *,
    group_id: str,
) -> dict[str, Any]:
    policy = _object(
        value,
        f"material choice {group_id} retrieval surface_interpretation_policy",
    )
    expected_fields = {
        "mode",
        "active",
        "family_reliable",
        "semantic_surface_class",
        "semantic_numeric_conflict",
        "multi_view_albedo_reliable",
        "albedo_median",
        "albedo_luminance",
        "dark_multiview_color",
        "metallic_reliable",
        "metallicity_class",
        "roughness_reliable",
        "observed_roughness",
        "roughness_class",
        "required_interpretations",
        "available_interpretation_counts",
        "selected_material_ids_by_interpretation",
        "complete_required_coverage",
    }
    optional_fields = {
        "required_intrinsic_material_identities",
        "required_coating_physics_templates",
        "available_coating_physics_template_counts",
        "selected_material_ids_by_coating_physics_template",
    }
    if not expected_fields <= set(policy) or set(policy) - expected_fields - optional_fields:
        raise ConfidenceGateError(
            f"material choice {group_id} retrieval surface policy fields are invalid"
        )
    mode = _string(policy["mode"], f"material choice {group_id} retrieval surface mode")
    if mode not in {
        "score_ranked",
        "balanced_dark_metal_surface_interpretations",
        "balanced_intrinsic_metal_identities",
        "balanced_confirmed_applied_coating_physics",
    }:
        raise ConfidenceGateError(
            f"material choice {group_id} retrieval surface mode is invalid"
        )
    active = _boolean(
        policy["active"], f"material choice {group_id} retrieval surface active"
    )
    family_reliable = _boolean(
        policy["family_reliable"],
        f"material choice {group_id} retrieval surface family_reliable",
    )
    semantic_surface_class = _string(
        policy["semantic_surface_class"],
        f"material choice {group_id} retrieval semantic surface class",
    )
    if semantic_surface_class not in {"coating", "bare", "ambiguous"}:
        raise ConfidenceGateError(
            f"material choice {group_id} retrieval semantic surface class is invalid"
        )
    multi_view_albedo_reliable = _boolean(
        policy["multi_view_albedo_reliable"],
        f"material choice {group_id} retrieval multi-view albedo reliability",
    )
    raw_albedo = policy["albedo_median"]
    raw_luminance = policy["albedo_luminance"]
    if multi_view_albedo_reliable:
        albedo = [
            _confidence(
                component,
                f"material choice {group_id} retrieval albedo_median[{index}]",
            )
            for index, component in enumerate(
                _array(
                    raw_albedo,
                    f"material choice {group_id} retrieval albedo_median",
                )
            )
        ]
        if len(albedo) != 3:
            raise ConfidenceGateError(
                f"material choice {group_id} retrieval albedo_median must have 3 values"
            )
        albedo_luminance = _confidence(
            raw_luminance,
            f"material choice {group_id} retrieval albedo_luminance",
        )
        expected_luminance = (
            0.2126 * albedo[0] + 0.7152 * albedo[1] + 0.0722 * albedo[2]
        )
        if not math.isclose(
            albedo_luminance,
            expected_luminance,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise ConfidenceGateError(
                f"material choice {group_id} retrieval albedo_luminance is inconsistent"
            )
    else:
        if raw_albedo is not None or raw_luminance is not None:
            raise ConfidenceGateError(
                f"material choice {group_id} unreliable albedo must use null statistics"
            )
        albedo = None
        albedo_luminance = None
    dark_multiview_color = _boolean(
        policy["dark_multiview_color"],
        f"material choice {group_id} retrieval dark_multiview_color",
    )
    if dark_multiview_color != (
        multi_view_albedo_reliable
        and albedo_luminance is not None
        and albedo_luminance <= 0.35
    ):
        raise ConfidenceGateError(
            f"material choice {group_id} retrieval dark color decision is inconsistent"
        )

    metallic_reliable = _boolean(
        policy["metallic_reliable"],
        f"material choice {group_id} retrieval metallic reliability",
    )
    metallicity_class = _string(
        policy["metallicity_class"],
        f"material choice {group_id} retrieval metallicity class",
    )
    if metallicity_class not in {
        "dielectric",
        "conductive",
        "ambiguous",
        "unknown",
    }:
        raise ConfidenceGateError(
            f"material choice {group_id} retrieval metallicity class is invalid"
        )
    if not metallic_reliable and metallicity_class != "unknown":
        raise ConfidenceGateError(
            f"material choice {group_id} unreliable metallicity must be unknown"
        )

    roughness_reliable = _boolean(
        policy["roughness_reliable"],
        f"material choice {group_id} retrieval roughness reliability",
    )
    observed_roughness = (
        _confidence(
            policy["observed_roughness"],
            f"material choice {group_id} retrieval observed roughness",
        )
        if policy["observed_roughness"] is not None
        else None
    )
    roughness_class = _string(
        policy["roughness_class"],
        f"material choice {group_id} retrieval roughness class",
    )
    if roughness_class not in {"glossy", "satin", "matte", "unknown"}:
        raise ConfidenceGateError(
            f"material choice {group_id} retrieval roughness class is invalid"
        )
    expected_roughness_class = (
        "unknown"
        if not roughness_reliable or observed_roughness is None
        else "glossy"
        if observed_roughness <= 0.30
        else "matte"
        if observed_roughness >= 0.58
        else "satin"
    )
    if roughness_class != expected_roughness_class:
        raise ConfidenceGateError(
            f"material choice {group_id} retrieval roughness class is inconsistent"
        )

    semantic_numeric_conflict = _boolean(
        policy["semantic_numeric_conflict"],
        f"material choice {group_id} retrieval semantic_numeric_conflict",
    )
    expected_conflict = (
        semantic_surface_class == "coating"
        and metallicity_class == "conductive"
        or semantic_surface_class == "bare"
        and metallicity_class == "dielectric"
    )
    if semantic_numeric_conflict != expected_conflict:
        raise ConfidenceGateError(
            f"material choice {group_id} retrieval surface conflict is inconsistent"
        )

    required = _unique_strings(
        policy["required_interpretations"],
        f"material choice {group_id} retrieval required interpretations",
    )
    required_identities = _unique_strings(
        policy.get("required_intrinsic_material_identities", []),
        f"material choice {group_id} retrieval required intrinsic identities",
    )
    required_coating_templates = _unique_strings(
        policy.get("required_coating_physics_templates", []),
        f"material choice {group_id} retrieval required coating templates",
    )
    available_raw = _object(
        policy["available_interpretation_counts"],
        f"material choice {group_id} retrieval available interpretations",
    )
    selected_raw = _object(
        policy["selected_material_ids_by_interpretation"],
        f"material choice {group_id} retrieval selected interpretations",
    )
    if set(available_raw) != set(_SURFACE_INTERPRETATIONS) or set(selected_raw) != set(
        _SURFACE_INTERPRETATIONS
    ):
        raise ConfidenceGateError(
            f"material choice {group_id} retrieval interpretation keys are invalid"
        )
    available = {
        name: _nonnegative_integer(
            available_raw[name],
            f"material choice {group_id} retrieval available {name}",
        )
        for name in _SURFACE_INTERPRETATIONS
    }
    selected = {
        name: _unique_strings(
            selected_raw[name],
            f"material choice {group_id} retrieval selected {name}",
        )
        for name in _SURFACE_INTERPRETATIONS
    }
    available_coating_raw = policy.get(
        "available_coating_physics_template_counts",
        {template: 0 for template in _COATING_PHYSICS_TEMPLATES},
    )
    selected_coating_raw = policy.get(
        "selected_material_ids_by_coating_physics_template",
        {template: [] for template in _COATING_PHYSICS_TEMPLATES},
    )
    available_coating_object = _object(
        available_coating_raw,
        f"material choice {group_id} retrieval available coating templates",
    )
    selected_coating_object = _object(
        selected_coating_raw,
        f"material choice {group_id} retrieval selected coating templates",
    )
    if (
        set(available_coating_object) != set(_COATING_PHYSICS_TEMPLATES)
        or set(selected_coating_object) != set(_COATING_PHYSICS_TEMPLATES)
    ):
        raise ConfidenceGateError(
            f"material choice {group_id} retrieval coating template keys are invalid"
        )
    available_coating_templates = {
        template: _nonnegative_integer(
            available_coating_object[template],
            f"material choice {group_id} retrieval available {template}",
        )
        for template in _COATING_PHYSICS_TEMPLATES
    }
    selected_coating_templates = {
        template: _unique_strings(
            selected_coating_object[template],
            f"material choice {group_id} retrieval selected {template}",
        )
        for template in _COATING_PHYSICS_TEMPLATES
    }
    selected_flat = [
        item for name in _SURFACE_INTERPRETATIONS for item in selected[name]
    ]
    if len(set(selected_flat)) != len(selected_flat):
        raise ConfidenceGateError(
            f"material choice {group_id} retrieval selected interpretations overlap"
        )
    complete = _boolean(
        policy["complete_required_coverage"],
        f"material choice {group_id} retrieval complete surface coverage",
    )
    if active and mode == "balanced_dark_metal_surface_interpretations":
        if (
            not family_reliable
            or not dark_multiview_color
            or required != list(_SURFACE_INTERPRETATIONS)
            or required_identities
            or required_coating_templates
            or not all(available.values())
            or not all(selected.values())
            or not complete
        ):
            raise ConfidenceGateError(
                f"material choice {group_id} balanced surface policy is inconsistent"
            )
    elif active and mode == "balanced_intrinsic_metal_identities":
        if (
            required
            or not required_identities
            or required_coating_templates
            or not complete
        ):
            raise ConfidenceGateError(
                f"material choice {group_id} intrinsic identity policy is inconsistent"
            )
    elif active and mode == "balanced_confirmed_applied_coating_physics":
        if (
            required
            or required_identities
            or required_coating_templates != list(_COATING_PHYSICS_TEMPLATES)
            or semantic_surface_class != "coating"
            or semantic_numeric_conflict
            or metallicity_class != "dielectric"
            or not all(available_coating_templates.values())
            or not all(selected_coating_templates.values())
            or not complete
        ):
            raise ConfidenceGateError(
                f"material choice {group_id} coating physics policy is inconsistent"
            )
    elif (
        mode != "score_ranked"
        or required
        or required_identities
        or required_coating_templates
        or not complete
    ):
        raise ConfidenceGateError(
            f"material choice {group_id} inactive surface policy is inconsistent"
        )
    return {
        "mode": mode,
        "active": active,
        "family_reliable": family_reliable,
        "semantic_surface_class": semantic_surface_class,
        "semantic_numeric_conflict": semantic_numeric_conflict,
        "multi_view_albedo_reliable": multi_view_albedo_reliable,
        "albedo_median": albedo,
        "albedo_luminance": albedo_luminance,
        "dark_multiview_color": dark_multiview_color,
        "metallic_reliable": metallic_reliable,
        "metallicity_class": metallicity_class,
        "roughness_reliable": roughness_reliable,
        "observed_roughness": observed_roughness,
        "roughness_class": roughness_class,
        "required_interpretations": required,
        "required_intrinsic_material_identities": required_identities,
        "required_coating_physics_templates": required_coating_templates,
        "available_interpretation_counts": available,
        "selected_material_ids_by_interpretation": selected,
        "available_coating_physics_template_counts": (
            available_coating_templates
        ),
        "selected_material_ids_by_coating_physics_template": (
            selected_coating_templates
        ),
        "complete_required_coverage": complete,
    }


def _retrieval_audit(
    record: Mapping[str, Any],
    *,
    group_id: str,
    chosen_material_id: str,
) -> dict[str, Any] | None:
    raw_audit = record.get("retrieval_audit")
    chosen_rank = record.get("chosen_retrieval_rank")
    matches_top = record.get("model_choice_matches_retrieval_top")
    supplied = [raw_audit is not None, chosen_rank is not None, matches_top is not None]
    if not any(supplied):
        return None
    if (
        raw_audit is None
        or "chosen_retrieval_rank" not in record
        or matches_top is None
    ):
        raise ConfidenceGateError(
            f"material choice {group_id} retrieval fields must be supplied together"
        )
    audit = _object(raw_audit, f"material choice {group_id}.retrieval_audit")
    visual_fallback_context: dict[str, Any] | None = None
    visual_wrapper_fields = {
        "strategy",
        "group_id",
        "pool_count",
        "eligible_pool_count",
        "full_catalog_indexed",
        "final_authority",
        "fallback_audit",
        "limit",
        "top_score",
        "runner_up_score",
        "score_margin",
        "normalized_margin",
        "margin_available",
        "ranking",
        "fixed_library_defaults_required",
    }
    is_visual_retrieval = audit.get("strategy") in _VISUAL_RETRIEVAL_STRATEGIES
    is_base_bank_retrieval = (
        audit.get("strategy") == _BASE_BANK_RETRIEVAL_STRATEGY
    )
    if is_visual_retrieval:
        if set(audit) != visual_wrapper_fields:
            raise ConfidenceGateError(
                f"material choice {group_id} visual retrieval fields are invalid; "
                f"unexpected={sorted(set(audit) - visual_wrapper_fields)}, "
                f"missing={sorted(visual_wrapper_fields - set(audit))}"
            )
        if audit.get("group_id") != group_id:
            raise ConfidenceGateError(
                f"material choice {group_id} visual retrieval group_id is inconsistent"
            )
        if audit.get("full_catalog_indexed") is not True:
            raise ConfidenceGateError(
                f"material choice {group_id} visual retrieval must index the full catalog"
            )
        if audit.get("final_authority") != _VISUAL_RETRIEVAL_FINAL_AUTHORITY:
            raise ConfidenceGateError(
                f"material choice {group_id} visual retrieval final authority is unsupported"
            )
        if audit.get("fixed_library_defaults_required") is not True:
            raise ConfidenceGateError(
                f"material choice {group_id} visual retrieval must preserve fixed MDL defaults"
            )
        fallback = _object(
            audit.get("fallback_audit"),
            f"material choice {group_id}.retrieval_audit.fallback_audit",
        )
        fallback_ranking = _array(
            fallback.get("ranking"),
            f"material choice {group_id} fallback retrieval ranking",
        )
        if not fallback_ranking or not isinstance(fallback_ranking[0], Mapping):
            raise ConfidenceGateError(
                f"material choice {group_id} fallback retrieval ranking cannot be empty"
            )
        fallback_top_material = _string(
            fallback_ranking[0].get("material_id"),
            f"material choice {group_id} fallback retrieval top material_id",
        )
        fallback_record: dict[str, Any] = {
            "retrieval_audit": fallback,
            "chosen_retrieval_rank": 1,
            "model_choice_matches_retrieval_top": True,
        }
        for context_field in ("selection_group", "semantic_reliability"):
            if context_field in record:
                fallback_record[context_field] = record[context_field]
        visual_fallback_context = _retrieval_audit(
            fallback_record,
            group_id=group_id,
            chosen_material_id=fallback_top_material,
        )
        if visual_fallback_context is None:
            raise ConfidenceGateError(
                f"material choice {group_id} fallback retrieval audit is unavailable"
            )
        if (
            audit.get("pool_count") != visual_fallback_context["pool_count"]
            or audit.get("eligible_pool_count")
            != visual_fallback_context["pool_count"]
            or audit.get("limit") != visual_fallback_context["limit"]
        ):
            raise ConfidenceGateError(
                f"material choice {group_id} visual retrieval catalog bounds "
                "do not match the validated fallback snapshot"
            )
        # The visual ranking is authoritative for candidate order and margin.
        # Family/semantic evidence remains a separately validated fallback
        # contract; it must not be flattened into the visual ranking because
        # the two rankings intentionally contain different material sets.
        audit = {
            key: value
            for key, value in audit.items()
            if key
            not in {
                "group_id",
                "full_catalog_indexed",
                "final_authority",
                "fallback_audit",
                "fixed_library_defaults_required",
            }
        }
        audit["family_pool_available"] = fallback["family_pool_available"]
        audit["family_pool_used"] = fallback["family_pool_used"]
    base_fields = {
        "strategy",
        "pool_count",
        "eligible_pool_count",
        "family_pool_available",
        "family_pool_used",
        "limit",
        "top_score",
        "runner_up_score",
        "score_margin",
        "normalized_margin",
        "margin_available",
        "ranking",
    }
    semantic_fields = {
        "semantic_reliability",
        "finish_evidence_used",
        "description_evidence_used",
        "intrinsic_surface_ambiguity",
    }
    coating_fields = {
        "pre_duplicate_alias_dedup_count",
        "duplicate_alias_dedup_count",
        "paint_pool_available",
        "paint_pool_used",
        "mvinverse_surface_class",
        "observed_metallic",
        "applied_coating_confirmed",
        "applied_coating_plausible",
    }
    surface_fields = {"surface_interpretation_policy"}
    tunable_equivalence_fields = {"mvinverse_tunable_equivalence_dedup_count"}
    fixed_default_fields = {"fixed_library_defaults_required"}
    thumbnail_default_fields = {"thumbnail_default_evidence_count"}
    fixed_effect_fields = {"unobserved_fixed_effect_policy"}
    niche_domain_fields = {"niche_domain_policy"}
    actual_fields = set(audit)
    valid_field_sets = (
        base_fields,
        base_fields | semantic_fields,
        base_fields | semantic_fields | coating_fields,
        base_fields | semantic_fields | coating_fields | surface_fields,
        base_fields
        | semantic_fields
        | coating_fields
        | surface_fields
        | tunable_equivalence_fields,
        # Mutable v7 retrieval still applies the catalog-wide niche-domain
        # exclusion policy.  It does not carry the later immutable-default,
        # thumbnail or fixed-effect fields, because the selected reviewed MDL
        # colour input remains writable.
        base_fields
        | semantic_fields
        | coating_fields
        | surface_fields
        | tunable_equivalence_fields
        | niche_domain_fields,
        base_fields
        | semantic_fields
        | coating_fields
        | surface_fields
        | tunable_equivalence_fields
        | fixed_default_fields,
        base_fields
        | semantic_fields
        | coating_fields
        | surface_fields
        | tunable_equivalence_fields
        | fixed_default_fields
        | thumbnail_default_fields,
        base_fields
        | semantic_fields
        | coating_fields
        | surface_fields
        | tunable_equivalence_fields
        | fixed_default_fields
        | thumbnail_default_fields
        | fixed_effect_fields,
        base_fields
        | semantic_fields
        | coating_fields
        | surface_fields
        | tunable_equivalence_fields
        | fixed_default_fields
        | thumbnail_default_fields
        | fixed_effect_fields
        | niche_domain_fields,
    )
    if actual_fields not in valid_field_sets:
        raise ConfidenceGateError(
            f"material choice {group_id}.retrieval_audit fields are invalid; "
            f"unexpected={sorted(actual_fields - base_fields - semantic_fields - coating_fields - surface_fields - tunable_equivalence_fields - fixed_default_fields - thumbnail_default_fields - fixed_effect_fields - niche_domain_fields)}, "
            f"missing={sorted(base_fields - actual_fields)}, "
            f"incomplete_semantic={sorted(semantic_fields - actual_fields)}"
        )
    has_semantic_context = semantic_fields <= actual_fields
    has_coating_context = coating_fields <= actual_fields
    has_surface_context = surface_fields <= actual_fields
    has_tunable_equivalence_context = tunable_equivalence_fields <= actual_fields
    has_fixed_default_context = fixed_default_fields <= actual_fields
    has_thumbnail_default_context = thumbnail_default_fields <= actual_fields
    has_fixed_effect_context = fixed_effect_fields <= actual_fields
    has_niche_domain_context = niche_domain_fields <= actual_fields
    strategy = _string(
        audit["strategy"], f"material choice {group_id} retrieval strategy"
    )
    expected_strategy = (
        _SEMANTIC_RETRIEVAL_STRATEGY
        if (
            has_fixed_effect_context
            and isinstance(audit.get("surface_interpretation_policy"), Mapping)
            and "required_coating_physics_templates"
            in audit["surface_interpretation_policy"]
        )
        else _SEMANTIC_RETRIEVAL_STRATEGY_V11
        if (
            has_fixed_effect_context
            and isinstance(audit.get("surface_interpretation_policy"), Mapping)
            and "required_intrinsic_material_identities"
            in audit["surface_interpretation_policy"]
        )
        else _SEMANTIC_RETRIEVAL_STRATEGY_V10
        if has_fixed_effect_context
        else _SEMANTIC_RETRIEVAL_STRATEGY_V9
        if has_thumbnail_default_context
        else _SEMANTIC_RETRIEVAL_STRATEGY_V8
        if has_fixed_default_context
        else _SEMANTIC_RETRIEVAL_STRATEGY_V7
        if has_surface_context
        else _SEMANTIC_RETRIEVAL_STRATEGY_V6
        if has_coating_context
        else _SEMANTIC_RETRIEVAL_STRATEGY_V5
        if has_semantic_context
        else strategy
    )
    if has_semantic_context and strategy != expected_strategy:
        raise ConfidenceGateError(
            f"material choice {group_id} retrieval strategy/semantic fields "
            "are inconsistent"
        )
    fixed_library_defaults_required = False
    if has_fixed_default_context:
        fixed_library_defaults_required = _boolean(
            audit["fixed_library_defaults_required"],
            f"material choice {group_id} retrieval fixed_library_defaults_required",
        )
        if not fixed_library_defaults_required:
            raise ConfidenceGateError(
                f"material choice {group_id} immutable retrieval must require "
                "fixed defaults"
            )
    thumbnail_default_evidence_count: int | None = None
    if has_thumbnail_default_context:
        if not has_fixed_default_context:
            raise ConfidenceGateError(
                f"material choice {group_id} thumbnail evidence requires fixed defaults"
            )
        thumbnail_default_evidence_count = _nonnegative_integer(
            audit["thumbnail_default_evidence_count"],
            f"material choice {group_id} retrieval thumbnail_default_evidence_count",
        )
    unobserved_fixed_effect_policy: str | None = None
    if has_fixed_effect_context:
        if not has_thumbnail_default_context:
            raise ConfidenceGateError(
                f"material choice {group_id} fixed-effect policy requires "
                "thumbnail default evidence"
            )
        unobserved_fixed_effect_policy = _string(
            audit["unobserved_fixed_effect_policy"],
            f"material choice {group_id} retrieval unobserved_fixed_effect_policy",
        )
        if (
            unobserved_fixed_effect_policy
            != "positive_reliable_semantics_required/v1"
        ):
            raise ConfidenceGateError(
                f"material choice {group_id} fixed-effect policy is unsupported"
            )
    niche_domain_policy: dict[str, Any] | None = None
    if has_niche_domain_context:
        if not has_fixed_effect_context and not (
            has_semantic_context
            and has_coating_context
            and has_surface_context
            and has_tunable_equivalence_context
            and strategy == _SEMANTIC_RETRIEVAL_STRATEGY_V7
        ):
            raise ConfidenceGateError(
                f"material choice {group_id} niche-domain policy requires either "
                "the mutable v7 surface contract or the immutable fixed-effect "
                "contract"
            )
        raw_niche_policy = _object(
            audit["niche_domain_policy"],
            f"material choice {group_id} retrieval niche_domain_policy",
        )
        niche_domain_policy = {
            "mode": _string(
                raw_niche_policy.get("mode"),
                f"material choice {group_id} niche-domain policy mode",
            ),
            "domains": _unique_strings(
                raw_niche_policy.get("domains"),
                f"material choice {group_id} niche-domain policy domains",
            ),
        }
        if (
            set(raw_niche_policy) != {"mode", "domains"}
            or niche_domain_policy
            != {
                "mode": "positive_reference_semantics_required",
                "domains": [
                    "automotive_finish",
                    "electronics_surface",
                ],
            }
        ):
            raise ConfidenceGateError(
                f"material choice {group_id} niche-domain policy is unsupported"
            )
    semantic_context = (
        _semantic_retrieval_context(record, audit, group_id=group_id)
        if has_semantic_context
        else None
    )
    if (
        not has_semantic_context
        and visual_fallback_context is None
        and (
        "selection_group" in record or "semantic_reliability" in record
        )
    ):
        raise ConfidenceGateError(
            f"material choice {group_id} legacy retrieval cannot carry "
            "semantic selection context"
        )
    pool_count = _nonnegative_integer(
        audit["pool_count"], f"material choice {group_id} retrieval pool_count"
    )
    eligible_count = _nonnegative_integer(
        audit["eligible_pool_count"],
        f"material choice {group_id} retrieval eligible_pool_count",
    )
    if eligible_count > pool_count:
        raise ConfidenceGateError(
            f"material choice {group_id} eligible_pool_count exceeds pool_count"
        )
    if is_visual_retrieval and eligible_count != pool_count:
        raise ConfidenceGateError(
            f"material choice {group_id} visual retrieval must index every "
            "catalog-eligible material"
        )
    if (
        thumbnail_default_evidence_count is not None
        and thumbnail_default_evidence_count > eligible_count
    ):
        raise ConfidenceGateError(
            f"material choice {group_id} thumbnail evidence count exceeds "
            "eligible_pool_count"
        )
    surface_context = (
        _surface_interpretation_context(
            audit["surface_interpretation_policy"],
            group_id=group_id,
        )
        if has_surface_context
        else None
    )
    coating_context: dict[str, Any] | None = None
    if has_coating_context:
        pre_dedup_count = _nonnegative_integer(
            audit["pre_duplicate_alias_dedup_count"],
            f"material choice {group_id} retrieval pre_duplicate_alias_dedup_count",
        )
        duplicate_alias_count = _nonnegative_integer(
            audit["duplicate_alias_dedup_count"],
            f"material choice {group_id} retrieval duplicate_alias_dedup_count",
        )
        if (
            not eligible_count <= pre_dedup_count <= pool_count
            or duplicate_alias_count != pre_dedup_count - eligible_count
        ):
            raise ConfidenceGateError(
                f"material choice {group_id} retrieval duplicate-alias counts "
                "are inconsistent"
            )
        paint_pool_available = _boolean(
            audit["paint_pool_available"],
            f"material choice {group_id} retrieval paint_pool_available",
        )
        paint_pool_used = _boolean(
            audit["paint_pool_used"],
            f"material choice {group_id} retrieval paint_pool_used",
        )
        if paint_pool_used and not paint_pool_available:
            raise ConfidenceGateError(
                f"material choice {group_id} retrieval used an unavailable paint pool"
            )
        surface_class = audit["mvinverse_surface_class"]
        if surface_class is not None and surface_class not in {
            "dielectric",
            "conductive",
            "conflict",
            "metal",
            "unknown",
        }:
            raise ConfidenceGateError(
                f"material choice {group_id} retrieval mvinverse_surface_class "
                "is invalid"
            )
        observed_raw = audit["observed_metallic"]
        observed_metallic = (
            None
            if observed_raw is None
            else _confidence(
                observed_raw,
                f"material choice {group_id} retrieval observed_metallic",
            )
        )
        coating_confirmed = _boolean(
            audit["applied_coating_confirmed"],
            f"material choice {group_id} retrieval applied_coating_confirmed",
        )
        coating_plausible = _boolean(
            audit["applied_coating_plausible"],
            f"material choice {group_id} retrieval applied_coating_plausible",
        )
        if coating_confirmed and coating_plausible:
            raise ConfidenceGateError(
                f"material choice {group_id} retrieval coating decisions conflict"
            )
        assert semantic_context is not None
        selection_group = _object(
            record.get("selection_group"),
            f"material choice {group_id}.selection_group",
        )
        canonical_finish = semantic_context["semantic_reliability"]["finish_hint"][
            "canonical_value"
        ]
        if surface_context is None:
            if coating_confirmed and not (
                surface_class == "dielectric"
                and paint_pool_used
                and semantic_context["finish_evidence_used"]
                and selection_group["family_hint"] == "metal"
                and selection_group["finish_hint"] in {"painted", "coated"}
            ):
                raise ConfidenceGateError(
                    f"material choice {group_id} retrieval confirmed coating is inconsistent"
                )
            if coating_plausible and not (
                surface_class == "dielectric"
                and observed_metallic is not None
                and observed_metallic <= 0.3
                and selection_group["family_hint"] == "metal"
                and canonical_finish in {"painted", "coated"}
                and not coating_confirmed
            ):
                raise ConfidenceGateError(
                    f"material choice {group_id} retrieval plausible coating is inconsistent"
                )
        else:
            surface_metallicity = surface_context["metallicity_class"]
            surface_semantic = surface_context["semantic_surface_class"]
            if surface_context["metallic_reliable"] and observed_metallic is None:
                raise ConfidenceGateError(
                    f"material choice {group_id} retrieval metallic reliability is inconsistent"
                )
            if surface_context["metallic_reliable"] and observed_metallic is not None:
                expected_metallicity = (
                    "dielectric"
                    if observed_metallic <= 0.35
                    else "conductive"
                    if observed_metallic >= 0.65
                    else "ambiguous"
                )
                if surface_metallicity != expected_metallicity:
                    raise ConfidenceGateError(
                        f"material choice {group_id} retrieval metallicity class is inconsistent"
                    )
            expected_confirmed = (
                selection_group["family_hint"] == "metal"
                and surface_semantic == "coating"
                and surface_metallicity == "dielectric"
                and not surface_context["semantic_numeric_conflict"]
            )
            expected_plausible = (
                selection_group["family_hint"] == "metal"
                and not expected_confirmed
                and surface_metallicity != "unknown"
                and (
                    surface_semantic == "coating" or surface_metallicity == "dielectric"
                )
            )
            if (
                coating_confirmed != expected_confirmed
                or coating_plausible != expected_plausible
                or paint_pool_used
            ):
                raise ConfidenceGateError(
                    f"material choice {group_id} retrieval v7 coating decision is inconsistent"
                )
        coating_context = {
            "pre_duplicate_alias_dedup_count": pre_dedup_count,
            "duplicate_alias_dedup_count": duplicate_alias_count,
            "paint_pool_available": paint_pool_available,
            "paint_pool_used": paint_pool_used,
            "mvinverse_surface_class": surface_class,
            "observed_metallic": observed_metallic,
            "applied_coating_confirmed": coating_confirmed,
            "applied_coating_plausible": coating_plausible,
        }
        if has_tunable_equivalence_context:
            tunable_equivalence_dedup_count = _nonnegative_integer(
                audit["mvinverse_tunable_equivalence_dedup_count"],
                "material choice "
                f"{group_id} retrieval mvinverse_tunable_equivalence_dedup_count",
            )
            if tunable_equivalence_dedup_count > eligible_count:
                raise ConfidenceGateError(
                    f"material choice {group_id} retrieval tunable-equivalence "
                    "dedup count exceeds eligible_pool_count"
                )
            coating_context["mvinverse_tunable_equivalence_dedup_count"] = (
                tunable_equivalence_dedup_count
            )
            if (
                fixed_library_defaults_required
                and tunable_equivalence_dedup_count != 0
            ):
                raise ConfidenceGateError(
                    f"material choice {group_id} immutable retrieval cannot "
                    "deduplicate tunable MDL exports"
                )
    family_pool_available = audit["family_pool_available"]
    if not isinstance(family_pool_available, bool):
        raise ConfidenceGateError(
            f"material choice {group_id} retrieval family_pool_available "
            "must be boolean"
        )
    family_pool_used = audit["family_pool_used"]
    if not isinstance(family_pool_used, bool):
        raise ConfidenceGateError(
            f"material choice {group_id} retrieval family_pool_used must be boolean"
        )
    limit = audit["limit"]
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ConfidenceGateError(
            f"material choice {group_id} retrieval limit must be a positive integer"
        )
    if (
        surface_context is not None
        and surface_context["active"]
        and limit < len(_SURFACE_INTERPRETATIONS)
    ):
        raise ConfidenceGateError(
            f"material choice {group_id} balanced surface policy exceeds its limit"
        )
    top_score = _finite_number(
        audit["top_score"], f"material choice {group_id} retrieval top_score"
    )
    margin_available = audit["margin_available"]
    if not isinstance(margin_available, bool):
        raise ConfidenceGateError(
            f"material choice {group_id} retrieval margin_available must be boolean"
        )
    runner_up_raw = audit["runner_up_score"]
    score_margin_raw = audit["score_margin"]
    normalized_raw = audit["normalized_margin"]
    if margin_available:
        runner_up_score = _finite_number(
            runner_up_raw,
            f"material choice {group_id} retrieval runner_up_score",
        )
        score_margin = _finite_number(
            score_margin_raw,
            f"material choice {group_id} retrieval score_margin",
        )
        normalized_margin = _confidence(
            normalized_raw,
            f"material choice {group_id} retrieval normalized_margin",
        )
        expected_score_margin = top_score - runner_up_score
        expected_normalized = expected_score_margin / (
            max(abs(top_score), 1e-12)
            if is_visual_retrieval
            else max(abs(top_score), 1.0)
        )
        if not math.isclose(
            score_margin, expected_score_margin, rel_tol=1e-9, abs_tol=1e-9
        ):
            raise ConfidenceGateError(
                f"material choice {group_id} retrieval score_margin is inconsistent"
            )
        if not math.isclose(
            normalized_margin, expected_normalized, rel_tol=1e-9, abs_tol=1e-9
        ):
            raise ConfidenceGateError(
                f"material choice {group_id} retrieval normalized_margin is inconsistent"
            )
    else:
        if any(
            value is not None
            for value in (runner_up_raw, score_margin_raw, normalized_raw)
        ):
            raise ConfidenceGateError(
                f"material choice {group_id} unavailable retrieval margin must use null fields"
            )
        runner_up_score = None
        score_margin = None
        normalized_margin = None

    raw_ranking = _array(
        audit["ranking"], f"material choice {group_id} retrieval ranking"
    )
    if not raw_ranking:
        raise ConfidenceGateError(
            f"material choice {group_id} retrieval ranking cannot be empty"
        )
    if len(raw_ranking) > limit or len(raw_ranking) > eligible_count:
        raise ConfidenceGateError(
            f"material choice {group_id} retrieval ranking exceeds its declared bounds"
        )
    ranking: list[dict[str, Any]] = []
    allowed_matches = {
        "family",
        "color",
        "finish",
        "optical_mode",
        "description_tokens",
        "mvinverse_color",
        "mvinverse_roughness",
        "mvinverse_metallic",
        "confirmed_applied_coating",
        "plausible_applied_coating",
        "coating_surface_family",
        "semantic_surface_interpretation",
        "mvinverse_metallicity_class",
        "multiview_albedo_color",
        "mvinverse_roughness_class",
        "mvinverse_tunable_template",
        "siglip2_catalog_wide_visual",
        "dinov2_masked_dense_texture",
        "siglip2_base_bank_rig_visual",
        "dinov2_base_bank_surface_texture",
        "masked_color_appearance",
        "mvinverse_authored_pbr_prior",
    }
    for index, raw_ranked in enumerate(raw_ranking):
        ranked = _object(
            raw_ranked, f"material choice {group_id} retrieval ranking[{index}]"
        )
        expected_ranked_fields = {"rank", "material_id", "score", "matched_fields"}
        if is_visual_retrieval:
            expected_ranked_fields |= {
                "siglip2_rank",
                "siglip2_score",
                "dino_rank",
                "dino_score",
            }
        if is_base_bank_retrieval:
            expected_ranked_fields |= {
                "color_rank",
                "color_score",
                "mvinverse_rank",
                "mvinverse_score",
            }
        if set(ranked) != expected_ranked_fields:
            raise ConfidenceGateError(
                f"material choice {group_id} retrieval ranking[{index}] has invalid fields"
            )
        rank = ranked["rank"]
        if isinstance(rank, bool) or not isinstance(rank, int):
            raise ConfidenceGateError(
                f"material choice {group_id} retrieval rank must be an integer"
            )
        matched_fields = _unique_strings(
            ranked["matched_fields"],
            f"material choice {group_id} retrieval ranking[{index}].matched_fields",
        )
        unexpected_matches = set(matched_fields) - allowed_matches
        if unexpected_matches:
            raise ConfidenceGateError(
                f"material choice {group_id} retrieval ranking[{index}] has unknown "
                f"matched_fields: {sorted(unexpected_matches)}"
            )
        parsed_ranked = {
            "rank": rank,
            "material_id": _string(
                ranked["material_id"],
                f"material choice {group_id} retrieval ranking[{index}].material_id",
            ),
            "score": _finite_number(
                ranked["score"],
                f"material choice {group_id} retrieval ranking[{index}].score",
            ),
            "matched_fields": matched_fields,
        }
        if is_visual_retrieval:
            siglip_rank = ranked["siglip2_rank"]
            if (
                isinstance(siglip_rank, bool)
                or not isinstance(siglip_rank, int)
                or siglip_rank < 1
            ):
                raise ConfidenceGateError(
                    f"material choice {group_id} retrieval ranking[{index}] "
                    "siglip2_rank must be a positive integer"
                )
            siglip_score = _finite_number(
                ranked["siglip2_score"],
                f"material choice {group_id} retrieval ranking[{index}].siglip2_score",
            )
            dino_rank_raw = ranked["dino_rank"]
            dino_score_raw = ranked["dino_score"]
            if (dino_rank_raw is None) != (dino_score_raw is None):
                raise ConfidenceGateError(
                    f"material choice {group_id} retrieval ranking[{index}] "
                    "DINO rank and score must be supplied together"
                )
            if dino_rank_raw is None:
                dino_rank = None
                dino_score = None
            else:
                if (
                    isinstance(dino_rank_raw, bool)
                    or not isinstance(dino_rank_raw, int)
                    or dino_rank_raw < 1
                ):
                    raise ConfidenceGateError(
                        f"material choice {group_id} retrieval ranking[{index}] "
                        "dino_rank must be a positive integer"
                    )
                dino_rank = dino_rank_raw
                dino_score = _finite_number(
                    dino_score_raw,
                    f"material choice {group_id} retrieval ranking[{index}].dino_score",
                )
            expected_visual_matches = (
                ["siglip2_base_bank_rig_visual"]
                if is_base_bank_retrieval
                else ["siglip2_catalog_wide_visual"]
            )
            if dino_rank is not None:
                expected_visual_matches.append(
                    "dinov2_base_bank_surface_texture"
                    if is_base_bank_retrieval
                    else "dinov2_masked_dense_texture"
                )
            color_rrf = 0.0
            mvinverse_rrf = 0.0
            if is_base_bank_retrieval:
                color_rank = ranked["color_rank"]
                if (
                    isinstance(color_rank, bool)
                    or not isinstance(color_rank, int)
                    or color_rank < 1
                ):
                    raise ConfidenceGateError(
                        f"material choice {group_id} retrieval ranking[{index}] "
                        "color_rank must be a positive integer"
                    )
                color_score = _finite_number(
                    ranked["color_score"],
                    f"material choice {group_id} retrieval ranking[{index}].color_score",
                )
                expected_visual_matches.append("masked_color_appearance")
                color_rrf = 0.8 / (60 + color_rank)
                mvinverse_rank_raw = ranked["mvinverse_rank"]
                mvinverse_score_raw = ranked["mvinverse_score"]
                if (mvinverse_rank_raw is None) != (
                    mvinverse_score_raw is None
                ):
                    raise ConfidenceGateError(
                        f"material choice {group_id} retrieval ranking[{index}] "
                        "MVInverse rank and score must be supplied together"
                    )
                if mvinverse_rank_raw is None:
                    mvinverse_rank = None
                    mvinverse_score = None
                else:
                    if (
                        isinstance(mvinverse_rank_raw, bool)
                        or not isinstance(mvinverse_rank_raw, int)
                        or mvinverse_rank_raw < 1
                    ):
                        raise ConfidenceGateError(
                            f"material choice {group_id} retrieval ranking[{index}] "
                            "mvinverse_rank must be a positive integer"
                        )
                    mvinverse_rank = mvinverse_rank_raw
                    mvinverse_score = _finite_number(
                        mvinverse_score_raw,
                        f"material choice {group_id} retrieval "
                        f"ranking[{index}].mvinverse_score",
                    )
                    expected_visual_matches.append(
                        "mvinverse_authored_pbr_prior"
                    )
                    mvinverse_rrf = 0.2 / (60 + mvinverse_rank)
            if matched_fields != expected_visual_matches:
                raise ConfidenceGateError(
                    f"material choice {group_id} retrieval ranking[{index}] "
                    "visual matched_fields are inconsistent"
                )
            expected_rrf_score = round(
                1.0 / (60 + siglip_rank)
                + (1.2 / (60 + dino_rank) if dino_rank is not None else 0.0)
                + color_rrf
                + mvinverse_rrf,
                10,
            )
            if not math.isclose(
                parsed_ranked["score"],
                expected_rrf_score,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ConfidenceGateError(
                    f"material choice {group_id} retrieval ranking[{index}] "
                    "RRF score is inconsistent"
                )
            parsed_ranked.update(
                {
                    "siglip2_rank": siglip_rank,
                    "siglip2_score": siglip_score,
                    "dino_rank": dino_rank,
                    "dino_score": dino_score,
                }
            )
            if is_base_bank_retrieval:
                parsed_ranked.update(
                    {
                        "color_rank": color_rank,
                        "color_score": color_score,
                        "mvinverse_rank": mvinverse_rank,
                        "mvinverse_score": mvinverse_score,
                    }
                )
        ranking.append(parsed_ranked)
    if [item["rank"] for item in ranking] != list(range(1, len(ranking) + 1)):
        raise ConfidenceGateError(
            f"material choice {group_id} retrieval ranks must be unique, ordered, and contiguous"
        )
    if len({item["material_id"] for item in ranking}) != len(ranking):
        raise ConfidenceGateError(
            f"material choice {group_id} retrieval ranking has duplicate material_id values"
        )
    if is_visual_retrieval:
        siglip_ranks = [item["siglip2_rank"] for item in ranking]
        dino_ranks = [
            item["dino_rank"]
            for item in ranking
            if item["dino_rank"] is not None
        ]
        color_ranks = (
            [item["color_rank"] for item in ranking]
            if is_base_bank_retrieval
            else []
        )
        mvinverse_ranks = (
            [
                item["mvinverse_rank"]
                for item in ranking
                if item["mvinverse_rank"] is not None
            ]
            if is_base_bank_retrieval
            else []
        )
        if (
            len(siglip_ranks) != len(set(siglip_ranks))
            or len(dino_ranks) != len(set(dino_ranks))
            or len(color_ranks) != len(set(color_ranks))
            or len(mvinverse_ranks) != len(set(mvinverse_ranks))
            or any(rank > pool_count for rank in siglip_ranks)
            or any(rank > pool_count for rank in dino_ranks)
            or any(rank > pool_count for rank in color_ranks)
            or any(rank > pool_count for rank in mvinverse_ranks)
        ):
            raise ConfidenceGateError(
                f"material choice {group_id} visual retrieval source ranks "
                "must be unique and within the indexed catalog"
            )
    if any(
        ranking[index]["score"] < ranking[index + 1]["score"]
        for index in range(len(ranking) - 1)
    ):
        raise ConfidenceGateError(
            f"material choice {group_id} retrieval ranking scores must be descending"
        )
    if not math.isclose(ranking[0]["score"], top_score, rel_tol=1e-9, abs_tol=1e-9):
        raise ConfidenceGateError(
            f"material choice {group_id} retrieval top_score does not match ranking"
        )
    if margin_available:
        if len(ranking) < 2 or not math.isclose(
            ranking[1]["score"], runner_up_score, rel_tol=1e-9, abs_tol=1e-9
        ):
            raise ConfidenceGateError(
                f"material choice {group_id} retrieval runner_up_score does not match ranking"
            )
    elif len(ranking) > 1:
        raise ConfidenceGateError(
            f"material choice {group_id} retrieval margin is unavailable despite a runner-up"
        )
    if semantic_context is not None:
        if not semantic_context["finish_evidence_used"] and any(
            "finish" in item["matched_fields"] for item in ranking
        ):
            raise ConfidenceGateError(
                f"material choice {group_id} retrieval ranking used disabled "
                "finish evidence"
            )
    if coating_context is not None:
        for item in ranking:
            matched = set(item["matched_fields"])
            if (
                "confirmed_applied_coating" in matched
                and not coating_context["applied_coating_confirmed"]
            ):
                raise ConfidenceGateError(
                    f"material choice {group_id} retrieval ranking has an "
                    "unauthorized confirmed coating match"
                )
            if (
                "plausible_applied_coating" in matched
                and not coating_context["applied_coating_plausible"]
            ):
                raise ConfidenceGateError(
                    f"material choice {group_id} retrieval ranking has an "
                    "unauthorized plausible coating match"
                )
        if not semantic_context["description_evidence_used"] and any(
            "description_tokens" in item["matched_fields"] for item in ranking
        ):
            raise ConfidenceGateError(
                f"material choice {group_id} retrieval ranking used disabled "
                "description evidence"
            )
        if fixed_library_defaults_required and any(
            "mvinverse_tunable_template" in item["matched_fields"]
            for item in ranking
        ):
            raise ConfidenceGateError(
                f"material choice {group_id} immutable retrieval cannot use "
                "tunable-template evidence"
            )
    if surface_context is not None:
        ranked_ids = {item["material_id"] for item in ranking}
        selected_by_interpretation = surface_context[
            "selected_material_ids_by_interpretation"
        ]
        selected_ids = {
            material_id
            for interpretation in _SURFACE_INTERPRETATIONS
            for material_id in selected_by_interpretation[interpretation]
        }
        available_counts = surface_context["available_interpretation_counts"]
        if sum(available_counts.values()) > eligible_count or any(
            len(selected_by_interpretation[name]) > available_counts[name]
            for name in _SURFACE_INTERPRETATIONS
        ):
            raise ConfidenceGateError(
                f"material choice {group_id} retrieval surface counts are inconsistent"
            )
        if not selected_ids <= ranked_ids or (
            surface_context["active"] and selected_ids != ranked_ids
        ):
            raise ConfidenceGateError(
                f"material choice {group_id} retrieval surface selection/ranking mismatch"
            )
        for item in ranking:
            matched = set(item["matched_fields"])
            if (
                "mvinverse_metallicity_class" in matched
                and surface_context["metallicity_class"] == "unknown"
            ):
                raise ConfidenceGateError(
                    f"material choice {group_id} ranking used unavailable metallicity"
                )
            if (
                "multiview_albedo_color" in matched
                and not surface_context["dark_multiview_color"]
            ):
                raise ConfidenceGateError(
                    f"material choice {group_id} ranking used unavailable dark albedo"
                )
            if (
                "mvinverse_roughness_class" in matched
                and surface_context["roughness_class"] == "unknown"
            ):
                raise ConfidenceGateError(
                    f"material choice {group_id} ranking used unavailable roughness"
                )
    ranked_choice = next(
        (item for item in ranking if item["material_id"] == chosen_material_id), None
    )
    if chosen_rank is not None:
        if (
            isinstance(chosen_rank, bool)
            or not isinstance(chosen_rank, int)
            or not 1 <= chosen_rank <= len(ranking)
        ):
            raise ConfidenceGateError(
                f"material choice {group_id}.chosen_retrieval_rank is invalid"
            )
        if ranking[chosen_rank - 1]["material_id"] != chosen_material_id:
            raise ConfidenceGateError(
                f"material choice {group_id} chosen material/rank are inconsistent"
            )
    elif ranked_choice is not None:
        raise ConfidenceGateError(
            f"material choice {group_id} chosen_retrieval_rank is null for a ranked material"
        )
    if not isinstance(matches_top, bool):
        raise ConfidenceGateError(
            f"material choice {group_id}.model_choice_matches_retrieval_top must be boolean"
        )
    actual_matches_top = ranking[0]["material_id"] == chosen_material_id
    if matches_top != actual_matches_top:
        raise ConfidenceGateError(
            f"material choice {group_id} retrieval-top agreement flag is inconsistent"
        )
    usable_margin = (
        normalized_margin
        if margin_available and matches_top and chosen_rank == 1
        else None
    )
    result = {
        "strategy": strategy,
        "pool_count": pool_count,
        "eligible_pool_count": eligible_count,
        "family_pool_used": family_pool_used,
        "limit": limit,
        "top_score": top_score,
        "runner_up_score": runner_up_score,
        "score_margin": score_margin,
        "normalized_margin": normalized_margin,
        "margin_available": margin_available,
        "ranking": ranking,
        "chosen_retrieval_rank": chosen_rank,
        "model_choice_matches_retrieval_top": matches_top,
        "usable_candidate_margin": usable_margin,
    }
    if semantic_context is not None:
        result.update(semantic_context)
    if coating_context is not None:
        result.update(coating_context)
    if surface_context is not None:
        result["surface_interpretation_policy"] = surface_context
    if fixed_library_defaults_required:
        result["fixed_library_defaults_required"] = True
    if thumbnail_default_evidence_count is not None:
        result["thumbnail_default_evidence_count"] = (
            thumbnail_default_evidence_count
        )
    if unobserved_fixed_effect_policy is not None:
        result["unobserved_fixed_effect_policy"] = unobserved_fixed_effect_policy
    if niche_domain_policy is not None:
        result["niche_domain_policy"] = niche_domain_policy
    if visual_fallback_context is not None:
        result.update(
            {
                "group_id": group_id,
                "full_catalog_indexed": True,
                "final_authority": _VISUAL_RETRIEVAL_FINAL_AUTHORITY,
                "fixed_library_defaults_required": True,
                "fallback_audit": visual_fallback_context,
            }
        )
    return result


def _material_choices(
    audit: Mapping[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    if audit is None:
        return {}
    result: dict[str, dict[str, Any]] = {}
    for raw_group_id, raw in audit.items():
        group_id = _string(raw_group_id, "material choice group_id")
        record = _object(raw, f"material_choice_audit[{group_id}]")
        forward = _object(record.get("forward"), f"material choice {group_id}.forward")
        reverse = _object(record.get("reverse"), f"material choice {group_id}.reverse")
        forward_material = _string(
            forward.get("material_id"),
            f"material choice {group_id}.forward.material_id",
        )
        reverse_material = _string(
            reverse.get("material_id"),
            f"material choice {group_id}.reverse.material_id",
        )
        confirmed = record.get("confirmed")
        if not isinstance(confirmed, bool):
            raise ConfidenceGateError(
                f"material choice {group_id}.confirmed must be boolean"
            )
        confirmation_basis = record.get("confirmation_basis")
        confirmed_material_raw = record.get("confirmed_material_id")
        has_resolution_contract = (
            "confirmation_basis" in record or "confirmed_material_id" in record
        )
        if has_resolution_contract and not (
            "confirmation_basis" in record and "confirmed_material_id" in record
        ):
            raise ConfidenceGateError(
                f"material choice {group_id} resolution fields must be supplied together"
            )
        resolved_material = forward_material
        if has_resolution_contract:
            confirmation_basis = _string(
                confirmation_basis,
                f"material choice {group_id}.confirmation_basis",
            )
            if confirmed_material_raw is not None:
                resolved_material = _string(
                    confirmed_material_raw,
                    f"material choice {group_id}.confirmed_material_id",
                )
            if confirmation_basis == "exact_forward_reverse_agreement":
                valid_resolution = (
                    confirmed
                    and forward_material == reverse_material == resolved_material
                )
            elif confirmation_basis == "nvidia_base_duplicate_paint_alias_agreement":
                first_match = _BASE_PAINT_EQUIVALENCE_RE.fullmatch(forward_material)
                second_match = _BASE_PAINT_EQUIVALENCE_RE.fullmatch(reverse_material)
                valid_resolution = (
                    confirmed
                    and first_match is not None
                    and second_match is not None
                    and first_match.group(1) == second_match.group(1)
                    and resolved_material
                    == f"mdl:Miscellaneous/{first_match.group(1)}.mdl#"
                    f"{first_match.group(1)}"
                )
            elif confirmation_basis == "mvinverse_resolved_base_paint_finish":
                valid_resolution = (
                    confirmed
                    and _BASE_PAINT_PRIMARY_RE.fullmatch(forward_material) is not None
                    and _BASE_PAINT_PRIMARY_RE.fullmatch(reverse_material) is not None
                    and _BASE_PAINT_PRIMARY_RE.fullmatch(resolved_material) is not None
                )
            elif (
                confirmation_basis
                == "immutable_intrinsic_metal_class_agreement"
            ):
                retrieval_audit = record.get("retrieval_audit")
                surface_policy = (
                    retrieval_audit.get("surface_interpretation_policy")
                    if isinstance(retrieval_audit, Mapping)
                    else None
                )
                selected_by_interpretation = (
                    surface_policy.get("selected_material_ids_by_interpretation")
                    if isinstance(surface_policy, Mapping)
                    else None
                )
                bare_metal_ids = (
                    selected_by_interpretation.get("bare_metal")
                    if isinstance(selected_by_interpretation, Mapping)
                    else None
                )
                ranking = (
                    retrieval_audit.get("ranking")
                    if isinstance(retrieval_audit, Mapping)
                    else None
                )
                ranking_by_id = (
                    {
                        item.get("material_id"): item
                        for item in ranking
                        if isinstance(item, Mapping)
                        and isinstance(item.get("material_id"), str)
                    }
                    if isinstance(ranking, list)
                    else {}
                )
                first_rank = ranking_by_id.get(forward_material)
                second_rank = ranking_by_id.get(reverse_material)
                first_score = (
                    first_rank.get("score")
                    if isinstance(first_rank, Mapping)
                    else None
                )
                second_score = (
                    second_rank.get("score")
                    if isinstance(second_rank, Mapping)
                    else None
                )
                expected_winner = (
                    forward_material
                    if isinstance(first_score, (int, float))
                    and not isinstance(first_score, bool)
                    and isinstance(second_score, (int, float))
                    and not isinstance(second_score, bool)
                    and float(first_score) > float(second_score)
                    else reverse_material
                )
                valid_resolution = (
                    confirmed
                    and forward_material != reverse_material
                    and isinstance(retrieval_audit, Mapping)
                    and retrieval_audit.get("fixed_library_defaults_required")
                    is True
                    and isinstance(surface_policy, Mapping)
                    and surface_policy.get("mode")
                    == "balanced_intrinsic_metal_identities"
                    and isinstance(bare_metal_ids, list)
                    and forward_material in bare_metal_ids
                    and reverse_material in bare_metal_ids
                    and isinstance(first_score, (int, float))
                    and not isinstance(first_score, bool)
                    and math.isfinite(float(first_score))
                    and isinstance(second_score, (int, float))
                    and not isinstance(second_score, bool)
                    and math.isfinite(float(second_score))
                    and float(first_score) != float(second_score)
                    and resolved_material == expected_winner
                )
            elif (
                confirmation_basis
                == "immutable_applied_paint_appearance_agreement"
            ):
                retrieval_audit = record.get("retrieval_audit")
                surface_policy = (
                    retrieval_audit.get("surface_interpretation_policy")
                    if isinstance(retrieval_audit, Mapping)
                    else None
                )
                selected_by_interpretation = (
                    surface_policy.get("selected_material_ids_by_interpretation")
                    if isinstance(surface_policy, Mapping)
                    else None
                )
                applied_paint_ids = (
                    selected_by_interpretation.get("applied_paint")
                    if isinstance(selected_by_interpretation, Mapping)
                    else None
                )
                ranking = (
                    retrieval_audit.get("ranking")
                    if isinstance(retrieval_audit, Mapping)
                    else None
                )
                ranking_by_id = (
                    {
                        item.get("material_id"): item
                        for item in ranking
                        if isinstance(item, Mapping)
                        and isinstance(item.get("material_id"), str)
                    }
                    if isinstance(ranking, list)
                    else {}
                )
                first_rank = ranking_by_id.get(forward_material)
                second_rank = ranking_by_id.get(reverse_material)
                first_score = (
                    first_rank.get("score")
                    if isinstance(first_rank, Mapping)
                    else None
                )
                second_score = (
                    second_rank.get("score")
                    if isinstance(second_rank, Mapping)
                    else None
                )
                expected_winner = (
                    forward_material
                    if isinstance(first_score, (int, float))
                    and not isinstance(first_score, bool)
                    and isinstance(second_score, (int, float))
                    and not isinstance(second_score, bool)
                    and float(first_score) > float(second_score)
                    else reverse_material
                )
                physics_resolution = record.get("physics_consistency_resolution")
                valid_resolution = (
                    confirmed
                    and forward_material != reverse_material
                    and isinstance(retrieval_audit, Mapping)
                    and retrieval_audit.get("fixed_library_defaults_required")
                    is True
                    and retrieval_audit.get("applied_coating_confirmed") is True
                    and isinstance(surface_policy, Mapping)
                    and surface_policy.get("mode")
                    == "balanced_confirmed_applied_coating_physics"
                    and surface_policy.get("metallicity_class") == "dielectric"
                    and isinstance(applied_paint_ids, list)
                    and forward_material in applied_paint_ids
                    and reverse_material in applied_paint_ids
                    and isinstance(first_rank, Mapping)
                    and "color" in (first_rank.get("matched_fields") or [])
                    and isinstance(second_rank, Mapping)
                    and "color" in (second_rank.get("matched_fields") or [])
                    and isinstance(first_score, (int, float))
                    and not isinstance(first_score, bool)
                    and math.isfinite(float(first_score))
                    and isinstance(second_score, (int, float))
                    and not isinstance(second_score, bool)
                    and math.isfinite(float(second_score))
                    and float(first_score) != float(second_score)
                    and resolved_material == expected_winner
                    and isinstance(physics_resolution, Mapping)
                    and physics_resolution.get("applied") is False
                    and physics_resolution.get("mode")
                    == "immutable_selected_mdl_preserved"
                    and physics_resolution.get("original_material_id")
                    == resolved_material
                    and physics_resolution.get("resolved_material_id")
                    == resolved_material
                    and physics_resolution.get(
                        "selected_mdl_parameters_mutable"
                    )
                    is False
                )
            elif (
                confirmation_basis
                == "immutable_confirmed_applied_coating_physics"
            ):
                physics_resolution = record.get("physics_consistency_resolution")
                retrieval_audit = record.get("retrieval_audit")
                surface_policy = (
                    retrieval_audit.get("surface_interpretation_policy")
                    if isinstance(retrieval_audit, Mapping)
                    else None
                )
                by_template = (
                    surface_policy.get(
                        "selected_material_ids_by_coating_physics_template"
                    )
                    if isinstance(surface_policy, Mapping)
                    else None
                )
                engineering_ids = (
                    by_template.get("painted_engineering_metal")
                    if isinstance(by_template, Mapping)
                    else None
                )
                valid_resolution = (
                    confirmed
                    and isinstance(physics_resolution, Mapping)
                    and physics_resolution.get("applied") is True
                    and physics_resolution.get("mode")
                    == "confirmed_painted_metal_requires_engineering_paint"
                    and physics_resolution.get("resolved_material_id")
                    == resolved_material
                    and physics_resolution.get("required_template")
                    == "painted_engineering_metal"
                    and physics_resolution.get("semantic_surface_class")
                    == "coating"
                    and physics_resolution.get("mvinverse_metallicity_class")
                    == "dielectric"
                    and physics_resolution.get(
                        "selected_mdl_parameters_mutable"
                    )
                    is False
                    and isinstance(retrieval_audit, Mapping)
                    and retrieval_audit.get("fixed_library_defaults_required")
                    is True
                    and retrieval_audit.get("applied_coating_confirmed") is True
                    and isinstance(surface_policy, Mapping)
                    and surface_policy.get("mode")
                    == "balanced_confirmed_applied_coating_physics"
                    and surface_policy.get("semantic_surface_class") == "coating"
                    and surface_policy.get("semantic_numeric_conflict") is False
                    and surface_policy.get("metallicity_class") == "dielectric"
                    and engineering_ids == [resolved_material]
                )
            elif confirmation_basis == "sam3_mask_unavailable_fail_closed":
                physics_resolution = record.get(
                    "physics_consistency_resolution"
                )
                independent_choices = record.get(
                    "independent_view_choices",
                    [],
                )
                valid_resolution = (
                    not confirmed
                    and confirmed_material_raw is None
                    and forward_material == reverse_material == resolved_material
                    and _choice_confidence(
                        forward,
                        f"material choice {group_id}.forward",
                    )
                    == 0.0
                    and _choice_confidence(
                        reverse,
                        f"material choice {group_id}.reverse",
                    )
                    == 0.0
                    and record.get("selection_confidence") == 0.0
                    and independent_choices == []
                    and isinstance(physics_resolution, Mapping)
                    and physics_resolution.get("applied") is False
                    and physics_resolution.get("mode")
                    == "immutable_selected_mdl_preserved"
                    and physics_resolution.get("original_material_id")
                    == resolved_material
                    and physics_resolution.get("resolved_material_id")
                    == resolved_material
                    and physics_resolution.get(
                        "selected_mdl_parameters_mutable"
                    )
                    is False
                )
            elif confirmation_basis == "forward_reverse_disagreement":
                valid_resolution = (
                    not confirmed
                    and confirmed_material_raw is None
                    and forward_material != reverse_material
                )
            else:
                valid_resolution = False
            if not valid_resolution:
                raise ConfidenceGateError(
                    f"material choice {group_id} resolution contract is inconsistent"
                )
        retrieval = _retrieval_audit(
            record,
            group_id=group_id,
            chosen_material_id=resolved_material,
        )
        margins = []
        if (
            retrieval is not None
            and confirmed is True
            and retrieval["model_choice_matches_retrieval_top"] is True
            and retrieval["usable_candidate_margin"] is not None
        ):
            margins.append(retrieval["usable_candidate_margin"])
        explicit_margin = record.get("candidate_margin")
        if explicit_margin is not None:
            margins.append(
                _confidence(
                    explicit_margin,
                    f"material choice {group_id}.candidate_margin",
                )
            )
        forward_confidence = _choice_confidence(
            forward, f"material choice {group_id}.forward"
        )
        reverse_confidence = _choice_confidence(
            reverse, f"material choice {group_id}.reverse"
        )
        raw_selection_confidence = record.get("selection_confidence")
        if raw_selection_confidence is None:
            # Backward-compatible readers retain the old behavior for sealed
            # artifacts created before deterministic confidence derivation.
            selection_confidence = min(
                forward_confidence, reverse_confidence
            )
        else:
            selection_confidence = _confidence(
                raw_selection_confidence,
                f"material choice {group_id}.selection_confidence",
            )
            derivation = _object(
                record.get("confidence_derivation"),
                f"material choice {group_id}.confidence_derivation",
            )
            if (
                derivation.get("schema_version")
                != MATERIAL_SELECTION_CONFIDENCE_SCHEMA_VERSION
                or derivation.get("reported_confidence_is_authoritative")
                is not False
                or derivation.get("derived_confidence")
                != selection_confidence
                or derivation.get("confirmation_basis") != confirmation_basis
                or derivation.get("reported_forward_confidence")
                != forward_confidence
                or derivation.get("reported_reverse_confidence")
                != reverse_confidence
            ):
                raise ConfidenceGateError(
                    f"material choice {group_id} derived-confidence contract "
                    "is inconsistent"
                )
        result[group_id] = {
            "forward_material_id": forward_material,
            "reverse_material_id": reverse_material,
            "resolved_material_id": resolved_material,
            "confirmation_basis": confirmation_basis,
            "forward_confidence": forward_confidence,
            "reverse_confidence": reverse_confidence,
            "selection_confidence": selection_confidence,
            "confirmed": confirmed,
            "retrieval_audit": retrieval,
            "candidate_margin": min(margins) if margins else None,
        }
    return result


def _view_evidence(
    document: Mapping[str, Any] | None, registry_ids: set[str]
) -> dict[str, list[dict[str, Any]]]:
    if document is None:
        return {}
    if document.get("schema_version") != VIEW_EVIDENCE_SCHEMA_VERSION:
        raise ConfidenceGateError(
            f"view evidence schema_version must be {VIEW_EVIDENCE_SCHEMA_VERSION!r}"
        )
    records = _array(document.get("predictions"), "view_evidence.predictions")
    result: dict[str, list[dict[str, Any]]] = {}
    seen: set[tuple[str, str]] = set()
    for index, raw in enumerate(records):
        prediction = _object(raw, f"view_evidence.predictions[{index}]")
        part_id = _part_id(
            prediction.get("part_id"), f"view_evidence.predictions[{index}].part_id"
        )
        if part_id not in registry_ids:
            raise ConfidenceGateError(
                f"view evidence contains unknown part_id: {part_id}"
            )
        view_id = _string(prediction.get("view_id"), f"view evidence {part_id}.view_id")
        if not _reference_view_id(view_id):
            raise ConfidenceGateError(
                f"view evidence {part_id}/{view_id} is not an independent reference view"
            )
        key = (part_id, view_id)
        if key in seen:
            raise ConfidenceGateError(
                f"duplicate view evidence prediction for {part_id}/{view_id}"
            )
        seen.add(key)
        margin = prediction.get("candidate_margin")
        result.setdefault(part_id, []).append(
            {
                "view_id": view_id,
                "material_id": _string(
                    prediction.get("material_id"),
                    f"view evidence {part_id}/{view_id}.material_id",
                ),
                "confidence": _confidence(
                    prediction.get("confidence"),
                    f"view evidence {part_id}/{view_id}.confidence",
                ),
                "candidate_margin": (
                    _confidence(
                        margin, f"view evidence {part_id}/{view_id}.candidate_margin"
                    )
                    if margin is not None
                    else None
                ),
            }
        )
    return result


def _visibility(part: Mapping[str, Any], policy: GatePolicy) -> dict[str, Any]:
    renders = part["renders"]
    pixels = [render["visible_pixels"] for render in renders]
    raw_visible_views = [
        render["view_id"]
        for render in renders
        if render["visible_pixels"] >= policy.visible_view_pixel_floor
    ]
    raw_max = max(pixels, default=0)
    isolated = part.get("isolated_evidence")
    isolated_eligible = (
        isinstance(isolated, Mapping)
        and isolated.get("schema_version") == ISOLATED_EVIDENCE_SCHEMA_VERSION
        and isolated.get("material_neutralized") is True
        and isolated.get("background_removed") is True
        and isinstance(isolated.get("source_max_visible_pixels"), int)
        and not isinstance(isolated.get("source_max_visible_pixels"), bool)
        and int(isolated["source_max_visible_pixels"])
        >= policy.isolated_source_visible_pixels
        and isinstance(isolated.get("normalized_max_visible_pixels"), int)
        and not isinstance(isolated.get("normalized_max_visible_pixels"), bool)
        and int(isolated["normalized_max_visible_pixels"])
        >= policy.review_visible_pixels
    )
    if isolated_eligible:
        isolated_source_by_view = isolated["source_visible_pixels_by_view"]
        visible_views = sorted(
            view_id
            for view_id, value in isolated_source_by_view.items()
            if value >= policy.isolated_source_visible_pixels
        )
        effective_max = int(isolated["normalized_max_visible_pixels"])
        evidence_mode = "isolated_mask_multiview"
    else:
        visible_views = raw_visible_views
        effective_max = raw_max
        evidence_mode = "source_projection"
    return {
        "render_count": len(renders),
        "max_visible_pixels": effective_max,
        "source_max_visible_pixels": raw_max,
        "visible_view_count": len(visible_views),
        "visible_views": visible_views,
        "evidence_mode": evidence_mode,
        "isolated_evidence_sha256": (
            isolated.get("sha256") if isolated_eligible else None
        ),
    }


def _append_reason(
    reasons: list[dict[str, str]], code: str, level: str, detail: str
) -> None:
    if level not in _DECISION_ORDER:
        raise AssertionError(f"unsupported reason level: {level}")
    if code not in {item["code"] for item in reasons}:
        reasons.append({"code": code, "level": level, "detail": detail})


def _decision(reasons: Sequence[Mapping[str, str]]) -> str:
    return max(
        (reason["level"] for reason in reasons),
        key=lambda level: _DECISION_ORDER[level],
        default="auto",
    )


def _geometry_risk_parts(
    value: Mapping[str, Any] | str | Path | None,
    *,
    registry_ids: set[str],
    rendered_registry: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Strictly validate optional geometry evidence and index it by part ID."""

    if value is None:
        return {}
    if isinstance(value, Mapping):
        document = value
    elif isinstance(value, (str, Path)):
        try:
            path = Path(value).expanduser().resolve(strict=True)
        except OSError as exc:
            raise ConfidenceGateError(
                f"geometry_risk_report does not resolve to a file: {value}"
            ) from exc
        if not path.is_file():
            raise ConfidenceGateError(f"geometry_risk_report is not a file: {path}")
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ConfidenceGateError(
                f"unable to read geometry_risk_report: {path}: {exc}"
            ) from exc
        if not isinstance(document, Mapping):
            raise ConfidenceGateError("geometry_risk_report must be a JSON object")
    else:
        raise ConfidenceGateError(
            "geometry_risk_report must be an object, JSON path, or null"
        )

    try:
        canonical = validate_geometry_risk(document)
    except GeometryRiskError as exc:
        raise ConfidenceGateError(
            f"geometry_risk_report validation failed: {exc}"
        ) from exc

    parts = {part["part_id"]: part for part in canonical["parts"]}
    geometry_ids = set(parts)
    if geometry_ids != registry_ids:
        raise ConfidenceGateError(
            "geometry_risk_report does not exactly cover rendered registry; "
            f"missing={sorted(registry_ids - geometry_ids)}, "
            f"unexpected={sorted(geometry_ids - registry_ids)}"
        )

    registry_asset_digest = rendered_registry.get("asset_sha256")
    if registry_asset_digest is not None:
        if not isinstance(registry_asset_digest, str):
            raise ConfidenceGateError(
                "rendered_registry.asset_sha256 must be a string when geometry risk is supplied"
            )
        if canonical["asset_sha256"] != registry_asset_digest.strip().lower():
            raise ConfidenceGateError(
                "geometry_risk_report asset_sha256 does not match rendered registry"
            )
    return parts


def evaluate_confidence_gate(
    staged_result: Mapping[str, Any],
    rendered_registry: Mapping[str, Any],
    *,
    review_plan: Mapping[str, Any] | None = None,
    batches: Sequence[Mapping[str, Any]] | None = None,
    material_choice_audit: Mapping[str, Any] | None = None,
    view_evidence: Mapping[str, Any] | None = None,
    geometry_risk_report: Mapping[str, Any] | str | Path | None = None,
    independent_validation_audit: Mapping[str, Any] | None = None,
    policy: GatePolicy | None = None,
) -> dict[str, Any]:
    """Evaluate all registered parts and return a deterministic audit report."""

    active_policy = policy or GatePolicy()
    active_policy.validate()
    rendered_registry_document = _object(rendered_registry, "rendered_registry")
    registry = _registry_parts(rendered_registry_document)
    registry_ids = set(registry)
    assignments, unknown = _staged_parts(
        _object(staged_result, "staged_result"), registry_ids
    )
    mapping_by_part = _batch_mappings(batches, registry_ids)
    independent_validations = _independent_mapping_validations(
        independent_validation_audit,
        registry_ids=registry_ids,
        mappings=mapping_by_part,
        minimum_confidence=active_policy.independently_validated_auto_confidence,
        maximum_conflict_confidence=active_policy.review_model_confidence,
    )
    review_by_part = _review_assignments(review_plan, registry_ids)
    choice_by_group = _material_choices(material_choice_audit)
    votes_by_part = _view_evidence(view_evidence, registry_ids)
    geometry_by_part = _geometry_risk_parts(
        geometry_risk_report,
        registry_ids=registry_ids,
        rendered_registry=rendered_registry_document,
    )

    decisions: list[dict[str, Any]] = []
    auto_assignments: list[dict[str, Any]] = []
    for part_id in sorted(registry):
        visibility = _visibility(registry[part_id], active_policy)
        reasons: list[dict[str, str]] = []
        assignment = assignments.get(part_id)
        review_assignment = (
            review_by_part.get(part_id) if review_by_part is not None else None
        )
        geometry_risk = geometry_by_part.get(part_id)
        geometry_multi_material = bool(
            geometry_risk and geometry_risk["risk"]["multi_material_risk"] is True
        )
        multi_material = (
            bool(review_assignment and review_assignment["face_subset_count"] > 0)
            or unknown.get(part_id) == "multi_material_mesh"
            or geometry_multi_material
        )

        if assignment is None:
            unknown_reason = unknown[part_id]
            _append_reason(
                reasons,
                "NO_MODEL_ASSIGNMENT",
                "preserve",
                f"staged result classified the part unknown: {unknown_reason}",
            )
            if unknown_reason == "multi_material_mesh":
                _append_reason(
                    reasons,
                    "MULTI_MATERIAL_RISK",
                    "preserve",
                    "a uniform automatic material would erase multiple appearances",
                )
            if geometry_multi_material:
                _append_reason(
                    reasons,
                    "GEOMETRY_MULTI_MATERIAL_RISK",
                    "preserve",
                    "topology risk evidence indicates that a uniform material may erase multiple appearances; "
                    f"geometry reasons={geometry_risk['reason_codes']}",
                )
            if visibility["max_visible_pixels"] < active_policy.review_visible_pixels:
                _append_reason(
                    reasons,
                    "CAD_EVIDENCE_BELOW_REVIEW",
                    "preserve",
                    "CAD visibility is below the minimum review evidence threshold",
                )
            decision = "preserve"
            decisions.append(
                {
                    "part_id": part_id,
                    "decision": decision,
                    "material_id": None,
                    "model": {
                        "status": "unknown",
                        "confidence": None,
                        "unknown_reason_code": unknown_reason,
                        "evidence_views": [],
                        "independent_reference_count": 0,
                    },
                    "cad_visibility": visibility,
                    "mapping": mapping_by_part.get(part_id),
                    "threshold_profile": "strict",
                    "active_auto_thresholds": {
                        "model_confidence": active_policy.auto_model_confidence,
                        "mapping_confidence": active_policy.auto_mapping_confidence,
                    },
                    "material_choice": None,
                    "candidate_margin": None,
                    "multi_material_risk": multi_material,
                    "geometry_risk": geometry_risk,
                    "review_plan": review_assignment,
                    "reason_codes": [reason["code"] for reason in reasons],
                    "reasons": reasons,
                }
            )
            continue

        material_id = assignment["material_id"]
        model_confidence = assignment["confidence"]
        independent_validation = independent_validations.get(part_id)
        model_auto_confidence = (
            active_policy.independently_validated_auto_confidence
            if independent_validation is not None
            else active_policy.auto_model_confidence
        )
        mapping_auto_confidence = (
            active_policy.independently_validated_auto_confidence
            if independent_validation is not None
            else active_policy.auto_mapping_confidence
        )
        if assignment["status"] != "auto":
            _append_reason(
                reasons,
                "MODEL_STATUS_REQUIRES_REVIEW",
                "review",
                f"staged model status is {assignment['status']!r}",
            )
        if model_confidence < active_policy.review_model_confidence:
            _append_reason(
                reasons,
                "MODEL_CONFIDENCE_BELOW_REVIEW",
                "preserve",
                "model confidence is below the minimum review threshold",
            )
        elif model_confidence < model_auto_confidence:
            _append_reason(
                reasons,
                "MODEL_CONFIDENCE_BELOW_AUTO",
                "review",
                "model confidence is below the automatic threshold",
            )

        staged_reference_views = sorted(
            {
                view_id
                for view_id in assignment["evidence_views"]
                if _reference_view_id(view_id)
            }
        )
        votes = votes_by_part.get(part_id, [])
        vote_margins: list[float] = []
        if view_evidence is not None:
            qualified_votes = [
                vote
                for vote in votes
                if vote["confidence"] >= active_policy.review_model_confidence
            ]
            conflicting = [
                vote for vote in qualified_votes if vote["material_id"] != material_id
            ]
            if conflicting:
                _append_reason(
                    reasons,
                    "CROSS_VIEW_MATERIAL_CONFLICT",
                    "preserve",
                    "an independent reference view predicts a different material",
                )
            agreeing = [
                vote
                for vote in qualified_votes
                if vote["material_id"] == material_id
                and vote["confidence"] >= model_auto_confidence
                and (
                    independent_validation is None
                    or vote["view_id"] in independent_validation["supporting_view_ids"]
                )
            ]
            independent_views = sorted({vote["view_id"] for vote in agreeing})
            vote_margins = [
                vote["candidate_margin"]
                for vote in agreeing
                if vote["candidate_margin"] is not None
            ]
            reference_source = "independent_view_predictions"
        else:
            independent_views = []
            reference_source = "staged_assignment_only"
            if staged_reference_views:
                _append_reason(
                    reasons,
                    "INDEPENDENT_VIEW_PREDICTIONS_UNAVAILABLE",
                    "review",
                    "staged reference citations do not replace independent per-view material predictions",
                )
        if not independent_views and staged_reference_views:
            _append_reason(
                reasons,
                "INSUFFICIENT_INDEPENDENT_REFERENCES",
                "review",
                "automatic application requires at least two agreeing independent per-view predictions",
            )
        elif not independent_views:
            _append_reason(
                reasons,
                "NO_INDEPENDENT_REFERENCE_EVIDENCE",
                "preserve",
                "no independent user reference proves this material assignment",
            )
        elif len(independent_views) < active_policy.minimum_independent_references:
            _append_reason(
                reasons,
                "INSUFFICIENT_INDEPENDENT_REFERENCES",
                "review",
                "automatic application requires at least two agreeing independent per-view predictions",
            )

        if visibility["max_visible_pixels"] < active_policy.review_visible_pixels:
            _append_reason(
                reasons,
                "CAD_EVIDENCE_BELOW_REVIEW",
                "preserve",
                "CAD visibility is below the minimum review evidence threshold",
            )
        else:
            if visibility["max_visible_pixels"] < active_policy.auto_visible_pixels:
                _append_reason(
                    reasons,
                    "CAD_PIXELS_BELOW_AUTO",
                    "review",
                    "best CAD render has too few visible pixels for automation",
                )
            required_visible_views = (
                active_policy.isolated_auto_visible_view_count
                if visibility["evidence_mode"] == "isolated_mask_multiview"
                else active_policy.auto_visible_view_count
            )
            if visibility["visible_view_count"] < required_visible_views:
                _append_reason(
                    reasons,
                    "CAD_VIEW_COUNT_BELOW_AUTO",
                    "review",
                    "too few CAD views show the part above the visibility floor",
                )

        mapping = mapping_by_part.get(part_id)
        group_id: str | None = None
        if mapping is None:
            _append_reason(
                reasons,
                "MAPPING_AUDIT_UNAVAILABLE",
                "review",
                "no part-to-palette batch mapping was supplied",
            )
        else:
            group_id = mapping["group_id"]
            if (
                mapping["status"] == "unknown"
                or mapping["confidence"] < active_policy.review_mapping_confidence
            ):
                _append_reason(
                    reasons,
                    "MAPPING_BELOW_REVIEW",
                    "preserve",
                    "part-to-palette localization is not reviewable",
                )
            elif (
                mapping["status"] != "matched"
                or mapping["confidence"] < mapping_auto_confidence
            ):
                _append_reason(
                    reasons,
                    "MAPPING_BELOW_AUTO",
                    "review",
                    "part-to-palette localization is below the automatic threshold",
                )

        choice = choice_by_group.get(group_id) if group_id is not None else None
        if material_choice_audit is None:
            _append_reason(
                reasons,
                "MATERIAL_CHOICE_AUDIT_UNAVAILABLE",
                "review",
                "candidate-order consistency was not supplied",
            )
        elif group_id is None:
            _append_reason(
                reasons,
                "MATERIAL_CHOICE_GROUP_UNAVAILABLE",
                "review",
                "candidate audit cannot be joined without a palette-group mapping",
            )
        elif choice is None:
            _append_reason(
                reasons,
                "MATERIAL_CHOICE_AUDIT_MISSING_GROUP",
                "preserve",
                "the mapped palette group has no material-choice audit",
            )
        else:
            choice_min_confidence = choice["selection_confidence"]
            if not choice["confirmed"]:
                _append_reason(
                    reasons,
                    "MATERIAL_ORDER_DISAGREEMENT",
                    "review",
                    "forward and reverse candidate ordering did not confirm one material",
                )
            if material_id != choice["resolved_material_id"]:
                _append_reason(
                    reasons,
                    "MATERIAL_CHOICE_INTEGRITY_MISMATCH",
                    "preserve",
                    "staged material differs from both audited material choices",
                )
            if choice_min_confidence < active_policy.review_material_choice_confidence:
                _append_reason(
                    reasons,
                    "MATERIAL_CHOICE_BELOW_REVIEW",
                    "preserve",
                    "material-choice confidence is below the review threshold",
                )
            elif choice_min_confidence < active_policy.auto_material_choice_confidence:
                _append_reason(
                    reasons,
                    "MATERIAL_CHOICE_BELOW_AUTO",
                    "review",
                    "material-choice confidence is below the automatic threshold",
                )
            retrieval = choice["retrieval_audit"]
            if retrieval is not None and (
                retrieval["model_choice_matches_retrieval_top"] is not True
                or retrieval["chosen_retrieval_rank"] != 1
            ):
                _append_reason(
                    reasons,
                    "RETRIEVAL_TOP_DISAGREEMENT",
                    "review",
                    "the model choice is not the deterministic retrieval top candidate",
                )

        choice_margin = choice["candidate_margin"] if choice is not None else None
        margins = (
            [
                margin
                for margin in (
                    choice_margin,
                    assignment["candidate_margin"],
                    *vote_margins,
                )
                if margin is not None
            ]
            if choice_margin is not None
            else []
        )
        candidate_margin = min(margins) if margins else None
        if candidate_margin is None:
            _append_reason(
                reasons,
                "CANDIDATE_MARGIN_UNAVAILABLE",
                "review",
                "no trustworthy deterministic retrieval runner-up margin was persisted; other margins cannot substitute for it",
            )
        elif candidate_margin < active_policy.minimum_candidate_margin:
            _append_reason(
                reasons,
                "CANDIDATE_MARGIN_BELOW_AUTO",
                "review",
                "top candidate is not separated enough from the runner-up",
            )

        if multi_material:
            _append_reason(
                reasons,
                "MULTI_MATERIAL_RISK",
                "preserve",
                "the part needs face-level material handling, not a uniform auto binding",
            )
        if geometry_multi_material:
            _append_reason(
                reasons,
                "GEOMETRY_MULTI_MATERIAL_RISK",
                "preserve",
                "topology risk evidence indicates that a uniform material may erase multiple appearances; "
                f"geometry reasons={geometry_risk['reason_codes']}",
            )
        if review_by_part is not None:
            if review_assignment is None:
                _append_reason(
                    reasons,
                    "HUMAN_REVIEW_PRESERVES_EXISTING",
                    "preserve",
                    "the approved plan intentionally has no assignment for this part",
                )
            elif review_assignment["material_id"] != material_id:
                _append_reason(
                    reasons,
                    "HUMAN_REVIEW_DIFFERS_FROM_MODEL",
                    "review",
                    "the approved human material differs from the staged model choice",
                )

        final_decision = _decision(reasons)
        if final_decision == "auto":
            source = dict(assignment["source_assignment"])
            source["status"] = "auto"
            auto_assignments.append(source)
        decisions.append(
            {
                "part_id": part_id,
                "decision": final_decision,
                "material_id": material_id,
                "model": {
                    "status": assignment["status"],
                    "confidence": model_confidence,
                    "unknown_reason_code": None,
                    "evidence_views": assignment["evidence_views"],
                    "staged_reference_views": staged_reference_views,
                    "independent_reference_count": len(independent_views),
                    "independent_reference_views": independent_views,
                    "reference_evidence_source": reference_source,
                },
                "cad_visibility": visibility,
                "mapping": mapping,
                "threshold_profile": (
                    "independently_validated"
                    if independent_validation is not None
                    else "strict"
                ),
                "active_auto_thresholds": {
                    "model_confidence": model_auto_confidence,
                    "mapping_confidence": mapping_auto_confidence,
                },
                "independent_validation": independent_validation,
                "material_choice": choice,
                "candidate_margin": candidate_margin,
                "multi_material_risk": multi_material,
                "geometry_risk": geometry_risk,
                "review_plan": review_assignment,
                "reason_codes": [reason["code"] for reason in reasons],
                "reasons": reasons,
            }
        )

    counts = {
        decision: sum(item["decision"] == decision for item in decisions)
        for decision in ("auto", "review", "preserve")
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "policy": asdict(active_policy),
        "summary": {
            "part_count": len(decisions),
            **{f"{decision}_count": count for decision, count in counts.items()},
            "fail_closed": True,
            "legacy_flow_modified": False,
        },
        "decisions": decisions,
        "auto_material_plan": {
            "schema_version": MATERIAL_PLAN_SCHEMA_VERSION,
            "assignments": auto_assignments,
        },
    }


def _read_object(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve(strict=True)
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ConfidenceGateError(f"expected a JSON object: {resolved}")
    return value


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve(strict=True).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_object(path: str | Path, document: Mapping[str, Any]) -> Path:
    resolved = Path(path).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved.with_name(f".{resolved.name}.tmp")
    temporary.write_text(
        json.dumps(dict(document), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(resolved)
    return resolved


def _load_batches(directory: str | Path | None) -> list[dict[str, Any]] | None:
    if directory is None:
        return None
    resolved = Path(directory).expanduser().resolve(strict=True)
    if not resolved.is_dir():
        raise ConfidenceGateError(f"batches path is not a directory: {resolved}")
    paths = sorted(resolved.glob("*.json"))
    if not paths:
        raise ConfidenceGateError(f"batches directory has no JSON files: {resolved}")
    return [_read_object(path) for path in paths]


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply a strict, deterministic confidence gate to staged Qwen output"
    )
    parser.add_argument("--staged-result", required=True)
    parser.add_argument("--rendered-registry", required=True)
    parser.add_argument("--review-plan")
    parser.add_argument("--batches-dir")
    parser.add_argument("--material-choice-audit")
    parser.add_argument("--view-evidence")
    parser.add_argument(
        "--independent-validation-audit",
        help="Validated qwen-spatial-mapping-gate/v1 audit",
    )
    parser.add_argument(
        "--geometry-risk",
        help="Validated qwen-geometry-uniform-material-risk/v1 report",
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    staged = _read_object(args.staged_result)
    registry = _read_object(args.rendered_registry)
    review = _read_object(args.review_plan) if args.review_plan else None
    material_audit = (
        _read_object(args.material_choice_audit) if args.material_choice_audit else None
    )
    view_evidence = _read_object(args.view_evidence) if args.view_evidence else None
    independent_validation = (
        _read_object(args.independent_validation_audit)
        if args.independent_validation_audit
        else None
    )
    geometry_risk = _read_object(args.geometry_risk) if args.geometry_risk else None
    if (
        geometry_risk is not None
        and geometry_risk.get("rendered_registry_sha256") is not None
        and geometry_risk["rendered_registry_sha256"] != _sha256(args.rendered_registry)
    ):
        raise ConfidenceGateError(
            "geometry risk report is not bound to the supplied rendered registry file"
        )
    report = evaluate_confidence_gate(
        staged,
        registry,
        review_plan=review,
        batches=_load_batches(args.batches_dir),
        material_choice_audit=material_audit,
        view_evidence=view_evidence,
        geometry_risk_report=geometry_risk,
        independent_validation_audit=independent_validation,
    )
    input_paths = {
        "staged_result": args.staged_result,
        "rendered_registry": args.rendered_registry,
        "review_plan": args.review_plan,
        "material_choice_audit": args.material_choice_audit,
        "view_evidence": args.view_evidence,
        "independent_validation_audit": args.independent_validation_audit,
        "geometry_risk": args.geometry_risk,
    }
    report["inputs"] = {
        name: {
            "path": str(Path(path).expanduser().resolve()),
            "sha256": _sha256(path),
        }
        for name, path in input_paths.items()
        if path is not None
    }
    if args.batches_dir:
        batch_paths = sorted(
            Path(args.batches_dir).expanduser().resolve(strict=True).glob("*.json")
        )
        report["inputs"]["batches"] = [
            {"path": str(path), "sha256": _sha256(path)} for path in batch_paths
        ]
    output = _write_object(args.output, report)
    print(
        json.dumps(
            {"output": str(output), "summary": report["summary"]},
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ConfidenceGateError",
    "GatePolicy",
    "SCHEMA_VERSION",
    "VIEW_EVIDENCE_SCHEMA_VERSION",
    "evaluate_confidence_gate",
    "main",
]
