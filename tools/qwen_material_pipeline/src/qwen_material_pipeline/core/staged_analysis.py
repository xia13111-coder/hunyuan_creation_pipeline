"""Strict, model-independent orchestration for staged material analysis.

This module deliberately does not load Qwen, inspect USD stages, or resolve MDL
paths.  It validates the small JSON contracts used between three independent
stages:

1. extract visible appearance groups from one user reference image;
2. map small, exactly specified batches of CAD parts to those groups; and
3. merge reviewed group-to-material selections into the existing material-plan
   schema.

Unknown parts remain visible in the returned audit data, but are never emitted
as material assignments.  This avoids inventing a placeholder material for a
surface that the single reference image does not show.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any


PALETTE_SCHEMA_VERSION = "qwen-material-palette/v1"
BATCH_SCHEMA_VERSION = "qwen-part-palette-map/v1"
GROUP_MATERIAL_SCHEMA_VERSION = "qwen-palette-material/v1"
STAGED_RESULT_SCHEMA_VERSION = "qwen-staged-material-result/v1"
MATERIAL_PLAN_SCHEMA_VERSION = "1.0"

GEOMETRY_VIEW_PREFIXES = (
    "cad_",
    "part_ids_",
    "part_contact_",
    "part_highlight_",
    "batch_parts_",
)
# A single Qwen extraction is prompted to return at most 12 groups.  This
# validator is also used for the deterministic union of several independently
# validated views, so its contract must not retain the old single-view limit.
# Four or more legitimate views can easily exceed 16 canonical appearances.
MAX_PALETTE_GROUPS = 9999
MAX_GROUP_BOXES = 4
AUTO_THRESHOLD = 0.85
REVIEW_THRESHOLD = 0.60
# The existing material-plan contract requires review confidence to be < 0.85.
# An otherwise high-confidence but unconfirmed group choice is capped rather
# than promoted to auto.
REVIEW_CONFIDENCE_CAP = 0.849999
UNKNOWN_CONFIDENCE_CAP = 0.599999
MIN_MATCHED_VISIBLE_PIXELS = 256

FAMILY_HINTS = frozenset(
    {"metal", "plastic", "rubber", "glass", "fabric", "ceramic", "other", "unknown"}
)
BASE_COLORS = frozenset(
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
FINISH_HINTS = frozenset(
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

MAPPING_STATUSES = frozenset({"matched", "review", "unknown"})
MATCHED_REASON_CODES = frozenset({"shape_and_location", "direct_visual_match"})
REVIEW_REASON_CODES = frozenset(
    {
        "shape_and_location",
        "direct_visual_match",
        "partial_visibility",
        "ambiguous",
        "too_small",
    }
)
UNKNOWN_REASON_CODES = frozenset(
    {
        "occluded",
        "not_in_reference",
        "ambiguous",
        "too_small",
        "partial_visibility",
        "no_cad_render",
        "multi_material_mesh",
    }
)
_GROUP_ID_RE = re.compile(r"G[0-9]{2,4}")
_BATCH_ID_RE = re.compile(r"B[0-9]{2,4}")


class StagedAnalysisError(ValueError):
    """Raised when staged analysis data is malformed, incomplete, or unsafe."""


class MaterialCollapseError(StagedAnalysisError):
    """Raised when material inference collapses a diverse reference.

    The stage metadata lets the subprocess boundary classify this deterministic
    quality failure and suppress an identical fresh-process retry.
    """

    stage_name = "material_collapse_gate"
    reason = "material_collapse_detected"

    def __init__(self, diagnostic: Mapping[str, Any]) -> None:
        self.diagnostic = dict(diagnostic)
        reasons = self.diagnostic.get("reasons")
        rendered_reasons = (
            "; ".join(str(item) for item in reasons)
            if isinstance(reasons, Sequence)
            and not isinstance(reasons, (str, bytes))
            else "invalid collapse diagnostic"
        )
        super().__init__("Material collapse detected: " + rendered_reasons)


def _require_exact_fields(
    value: Mapping[str, Any], expected: set[str], label: str
) -> None:
    actual = set(value)
    if actual != expected:
        unexpected = sorted(actual - expected)
        missing = sorted(expected - actual)
        raise StagedAnalysisError(
            f"{label} fields are invalid; unexpected={unexpected}, missing={missing}"
        )


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise StagedAnalysisError(f"{label} must be an object")
    return value


def _require_nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise StagedAnalysisError(f"{label} must be a non-empty string")
    return value.strip()


def _require_confidence(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0.0 <= value <= 1.0
    ):
        raise StagedAnalysisError(f"{label} must be a finite number from 0 to 1")
    return float(value)


def _require_id_set(values: Iterable[str], label: str) -> set[str]:
    if isinstance(values, (str, bytes)):
        raise StagedAnalysisError(f"{label} must be an iterable of IDs")
    identifiers: set[str] = set()
    count = 0
    for value in values:
        count += 1
        identifier = _require_nonempty_string(value, label)
        if identifier in identifiers:
            raise StagedAnalysisError(f"{label} contains duplicate ID: {identifier}")
        identifiers.add(identifier)
    if count == 0:
        raise StagedAnalysisError(f"{label} cannot be empty")
    return identifiers


def _validate_box(value: Any, label: str) -> list[int]:
    if not isinstance(value, list) or len(value) != 4:
        raise StagedAnalysisError(
            f"{label} must be [x0, y0, x1, y1] in normalized 0..1000 coordinates"
        )
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise StagedAnalysisError(f"{label} coordinates must be integers")
    x0, y0, x1, y1 = value
    if not all(0 <= item <= 1000 for item in value):
        raise StagedAnalysisError(f"{label} coordinates must be from 0 to 1000")
    if x0 >= x1 or y0 >= y1:
        raise StagedAnalysisError(f"{label} must have positive width and height")
    return [x0, y0, x1, y1]


def _coerce_model_confidence(value: Any, label: str) -> tuple[float, bool]:
    """Accept common JSON-number formatting without accepting ambiguous scales.

    The strict persisted schema still uses a JSON number.  This helper is only
    used at the model boundary, where small models sometimes emit ``"0.85"``
    or an explicit percentage such as ``"85%"``.  A bare ``85`` remains
    invalid because silently guessing whether it is a fraction or percentage
    would change the model's semantic claim.
    """

    if isinstance(value, str):
        text = value.strip()
        percentage = text.endswith("%")
        number_text = text[:-1].strip() if percentage else text
        try:
            numeric = float(number_text)
        except ValueError as exc:
            raise StagedAnalysisError(
                f"{label} must be a finite number from 0 to 1"
            ) from exc
        if percentage:
            numeric /= 100.0
        return _require_confidence(numeric, label), True
    return _require_confidence(value, label), False


def _coerce_model_box_index(value: Any, label: str) -> tuple[int, bool]:
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip()), True
    if isinstance(value, bool) or not isinstance(value, int):
        raise StagedAnalysisError(f"{label} must be a zero-based integer")
    return value, False


def _quarantined_mapping(part_id: str, *, too_small: bool) -> dict[str, Any]:
    return {
        "part_id": part_id,
        "group_id": None,
        "mapping_confidence": 0.0,
        "evidence_view_id": None,
        "evidence_box_index": None,
        "status": "unknown",
        "reason_code": "too_small" if too_small else "ambiguous",
    }


def normalize_part_palette_batch(
    batch: Mapping[str, Any],
    *,
    target_part_ids: Iterable[str],
    palette: Mapping[str, Any],
    expected_batch_id: str | None = None,
    visible_pixels_by_part: Mapping[str, int] | None = None,
    minimum_matched_visible_pixels: int = MIN_MATCHED_VISIBLE_PIXELS,
    quarantine_invalid_rows: bool = False,
    audit_events: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Normalize compact or legacy model output into the strict batch schema.

    Only mechanically derived fields are repaired: ``schema_version``,
    ``batch_id``, ``status``, and ``evidence_view_id``.  The model must still
    provide a real palette ``group_id`` and a valid ``evidence_box_index`` for
    every non-unknown mapping.  Invalid citations are never guessed.  Callers
    may opt into fail-closed, per-row quarantine after a bounded retry.
    """

    target_parts = _require_id_set(target_part_ids, "target_part_ids")
    canonical_palette = validate_palette(palette)
    groups = {group["group_id"]: group for group in canonical_palette["groups"]}
    document = _require_mapping(batch, "batch")
    raw_mappings = document.get("mappings")
    if not isinstance(raw_mappings, list):
        raise StagedAnalysisError("batch.mappings must be an array")

    if expected_batch_id is None:
        batch_id = _require_nonempty_string(document.get("batch_id"), "batch.batch_id")
    else:
        batch_id = _require_nonempty_string(expected_batch_id, "expected_batch_id")
    if not _BATCH_ID_RE.fullmatch(batch_id):
        raise StagedAnalysisError("batch.batch_id must use B followed by 2..4 digits")

    if (
        isinstance(minimum_matched_visible_pixels, bool)
        or not isinstance(minimum_matched_visible_pixels, int)
        or minimum_matched_visible_pixels < 1
    ):
        raise StagedAnalysisError(
            "minimum_matched_visible_pixels must be a positive integer"
        )
    raw_pixels = visible_pixels_by_part or {}
    if not isinstance(raw_pixels, Mapping):
        raise StagedAnalysisError("visible_pixels_by_part must be an object")
    pixels_by_part: dict[str, int] = {}
    for part_id in target_parts:
        pixels = raw_pixels.get(part_id, minimum_matched_visible_pixels)
        if isinstance(pixels, bool) or not isinstance(pixels, int) or pixels < 0:
            raise StagedAnalysisError(
                f"visible_pixels_by_part[{part_id}] must be a non-negative integer"
            )
        pixels_by_part[part_id] = pixels

    rows_by_part: dict[str, list[tuple[int, Mapping[str, Any]]]] = {
        part_id: [] for part_id in target_parts
    }
    unexpected_rows: list[int] = []
    for index, raw_mapping in enumerate(raw_mappings):
        if not isinstance(raw_mapping, Mapping):
            unexpected_rows.append(index)
            continue
        raw_part_id = raw_mapping.get("part_id")
        part_id = raw_part_id.strip() if isinstance(raw_part_id, str) else raw_part_id
        if part_id not in rows_by_part:
            unexpected_rows.append(index)
            continue
        rows_by_part[part_id].append((index, raw_mapping))

    events = audit_events if audit_events is not None else []
    if unexpected_rows:
        if not quarantine_invalid_rows:
            raise StagedAnalysisError(
                "batch contains non-object or unexpected mapping rows at indices: "
                + ", ".join(str(index) for index in unexpected_rows)
            )
        events.append(
            {
                "action": "ignored_unexpected_rows",
                "row_indices": unexpected_rows,
            }
        )

    def normalize_row(
        raw_mapping: Mapping[str, Any], *, part_id: str, row_index: int
    ) -> dict[str, Any]:
        required_fields = {
            "part_id",
            "group_id",
            "mapping_confidence",
            "evidence_box_index",
            "reason_code",
        }
        missing = sorted(required_fields - set(raw_mapping))
        if missing:
            raise StagedAnalysisError(
                f"batch.mappings[{row_index}] is missing semantic fields: {missing}"
            )

        changes: list[str] = []
        extra_fields = sorted(
            set(raw_mapping) - required_fields - {"status", "evidence_view_id"}
        )
        if extra_fields:
            changes.append("dropped_extra_fields:" + ",".join(extra_fields))

        confidence, confidence_was_coerced = _coerce_model_confidence(
            raw_mapping["mapping_confidence"],
            f"batch.mappings[{row_index}].mapping_confidence",
        )
        if confidence_was_coerced:
            changes.append("normalized_confidence_format")

        reason_code = _require_nonempty_string(
            raw_mapping["reason_code"],
            f"batch.mappings[{row_index}].reason_code",
        )
        all_reason_codes = (
            MATCHED_REASON_CODES | REVIEW_REASON_CODES | UNKNOWN_REASON_CODES
        )
        if reason_code not in all_reason_codes:
            raise StagedAnalysisError(
                f"batch.mappings[{row_index}].reason_code is unknown: {reason_code!r}"
            )

        raw_group_id = raw_mapping["group_id"]
        if raw_group_id is None:
            group_id = None
        elif isinstance(raw_group_id, str):
            group_id = raw_group_id.strip()
            if group_id not in groups:
                raise StagedAnalysisError(
                    f"batch.mappings[{row_index}] contains unknown group_id: "
                    f"{raw_group_id!r}"
                )
        else:
            raise StagedAnalysisError(
                f"batch.mappings[{row_index}].group_id must be a string or null"
            )

        evidence_box_index: int | None
        if group_id is None:
            evidence_box_index = None
            if raw_mapping["evidence_box_index"] is not None:
                changes.append("cleared_evidence_without_group")
        else:
            evidence_box_index, index_was_coerced = _coerce_model_box_index(
                raw_mapping["evidence_box_index"],
                f"batch.mappings[{row_index}].evidence_box_index",
            )
            if index_was_coerced:
                changes.append("normalized_evidence_box_index_format")
            if not 0 <= evidence_box_index < len(groups[group_id]["boxes"]):
                raise StagedAnalysisError(
                    f"batch.mappings[{row_index}].evidence_box_index is invalid "
                    f"for group {group_id}"
                )

        has_citation = group_id is not None
        unknown_only_reasons = UNKNOWN_REASON_CODES - REVIEW_REASON_CODES
        if not has_citation:
            if reason_code not in UNKNOWN_REASON_CODES:
                raise StagedAnalysisError(
                    f"batch.mappings[{row_index}] has no palette citation but "
                    f"reason_code {reason_code!r} is not an unknown reason"
                )
            status = "unknown"
        elif reason_code in unknown_only_reasons:
            # The semantic reason explicitly denies usable evidence.  Discarding
            # a contradictory citation is fail-closed; inventing one is not.
            status = "unknown"
            group_id = None
            evidence_box_index = None
            changes.append("cleared_citation_for_unknown_reason")
        elif confidence < REVIEW_THRESHOLD:
            if reason_code not in UNKNOWN_REASON_CODES:
                raise StagedAnalysisError(
                    f"batch.mappings[{row_index}] has confidence below "
                    f"{REVIEW_THRESHOLD} but no valid unknown reason"
                )
            status = "unknown"
            group_id = None
            evidence_box_index = None
            changes.append("cleared_low_confidence_citation")
        elif reason_code in REVIEW_REASON_CODES - MATCHED_REASON_CODES:
            status = "review"
        elif confidence >= AUTO_THRESHOLD:
            status = "matched"
        else:
            status = "review"

        # Legacy full-schema responses remain accepted, but an explicitly more
        # conservative legacy status acts as a safety cap.  Normalization may
        # downgrade a model row; it never promotes a row the model marked for
        # review/unknown into an automatic match.
        provided_status = raw_mapping.get("status")
        if provided_status == "review" and status == "matched":
            status = "review"
            changes.append("preserved_legacy_review_cap")
        elif provided_status == "unknown" and status != "unknown":
            if reason_code not in UNKNOWN_REASON_CODES:
                raise StagedAnalysisError(
                    f"batch.mappings[{row_index}] legacy unknown status conflicts "
                    f"with reason_code {reason_code!r}"
                )
            status = "unknown"
            group_id = None
            evidence_box_index = None
            changes.append("preserved_legacy_unknown_cap")

        if (
            status == "matched"
            and pixels_by_part[part_id] < minimum_matched_visible_pixels
        ):
            status = "review"
            reason_code = "too_small"
            changes.append("downgraded_sub_threshold_match")

        if status == "review" and confidence >= AUTO_THRESHOLD:
            confidence = REVIEW_CONFIDENCE_CAP
            changes.append("capped_review_confidence")
        elif status == "unknown" and confidence >= REVIEW_THRESHOLD:
            confidence = UNKNOWN_CONFIDENCE_CAP
            changes.append("capped_unknown_confidence")

        evidence_view_id = (
            canonical_palette["source_view_id"] if status != "unknown" else None
        )
        if provided_status is not None and provided_status != status:
            changes.append("derived_status")
        provided_evidence_view = raw_mapping.get("evidence_view_id")
        if (
            "evidence_view_id" in raw_mapping
            and provided_evidence_view != evidence_view_id
        ):
            changes.append("derived_evidence_view_id")

        normalized = {
            "part_id": part_id,
            "group_id": group_id,
            "mapping_confidence": confidence,
            "evidence_view_id": evidence_view_id,
            "evidence_box_index": evidence_box_index,
            "status": status,
            "reason_code": reason_code,
        }
        if changes:
            events.append(
                {
                    "action": "normalized",
                    "part_id": part_id,
                    "changes": changes,
                }
            )
        return normalized

    normalized_mappings: list[dict[str, Any]] = []
    for part_id in sorted(target_parts):
        candidates = rows_by_part[part_id]
        try:
            if len(candidates) != 1:
                raise StagedAnalysisError(
                    f"batch must contain exactly one row for {part_id}; "
                    f"found {len(candidates)}"
                )
            row_index, raw_mapping = candidates[0]
            normalized_mappings.append(
                normalize_row(raw_mapping, part_id=part_id, row_index=row_index)
            )
        except StagedAnalysisError as exc:
            if not quarantine_invalid_rows:
                raise
            normalized_mappings.append(
                _quarantined_mapping(
                    part_id,
                    too_small=(
                        pixels_by_part[part_id] < minimum_matched_visible_pixels
                    ),
                )
            )
            events.append(
                {
                    "action": "quarantined",
                    "part_id": part_id,
                    "error": str(exc),
                }
            )

    canonical = {
        "schema_version": BATCH_SCHEMA_VERSION,
        "batch_id": batch_id,
        "mappings": normalized_mappings,
    }
    return validate_part_palette_batch(
        canonical,
        target_part_ids=target_parts,
        palette=canonical_palette,
    )


def validate_palette(
    palette: Mapping[str, Any],
    *,
    allowed_reference_view_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Validate visible appearance groups extracted from one reference image."""

    document = _require_mapping(palette, "palette")
    _require_exact_fields(
        document,
        {"schema_version", "source_view_id", "groups"},
        "palette",
    )
    if document["schema_version"] != PALETTE_SCHEMA_VERSION:
        raise StagedAnalysisError(
            f"Unsupported palette schema_version: {document['schema_version']!r}"
        )
    source_view_id = _require_nonempty_string(
        document["source_view_id"], "palette.source_view_id"
    )
    if source_view_id.startswith(GEOMETRY_VIEW_PREFIXES):
        raise StagedAnalysisError(
            "palette.source_view_id must identify a user reference, not a CAD view"
        )
    if allowed_reference_view_ids is not None:
        allowed = _require_id_set(
            allowed_reference_view_ids, "allowed_reference_view_ids"
        )
        if source_view_id not in allowed:
            raise StagedAnalysisError(
                f"palette contains unknown source_view_id: {source_view_id!r}"
            )

    groups = document["groups"]
    if not isinstance(groups, list) or not 1 <= len(groups) <= MAX_PALETTE_GROUPS:
        raise StagedAnalysisError(
            f"palette.groups must contain 1..{MAX_PALETTE_GROUPS} objects"
        )
    expected_group_fields = {
        "group_id",
        "family_hint",
        "base_color",
        "finish_hint",
        "visual_description",
        "boxes",
        "confidence",
    }
    seen_groups: set[str] = set()
    validated_groups: list[dict[str, Any]] = []
    for index, raw_group in enumerate(groups):
        group = _require_mapping(raw_group, f"palette.groups[{index}]")
        _require_exact_fields(group, expected_group_fields, f"palette.groups[{index}]")
        group_id = _require_nonempty_string(
            group["group_id"], f"palette.groups[{index}].group_id"
        )
        if not _GROUP_ID_RE.fullmatch(group_id):
            raise StagedAnalysisError(
                f"palette.groups[{index}].group_id must use G followed by 2..4 digits"
            )
        if group_id in seen_groups:
            raise StagedAnalysisError(f"Duplicate palette group_id: {group_id}")
        seen_groups.add(group_id)

        family_hint = group["family_hint"]
        if family_hint not in FAMILY_HINTS:
            raise StagedAnalysisError(
                f"palette.groups[{index}].family_hint must be one of "
                f"{sorted(FAMILY_HINTS)}"
            )
        base_color = group["base_color"]
        if base_color not in BASE_COLORS:
            raise StagedAnalysisError(
                f"palette.groups[{index}].base_color must be one of "
                f"{sorted(BASE_COLORS)}"
            )
        finish_hint = group["finish_hint"]
        if finish_hint not in FINISH_HINTS:
            raise StagedAnalysisError(
                f"palette.groups[{index}].finish_hint must be one of "
                f"{sorted(FINISH_HINTS)}"
            )
        description = _require_nonempty_string(
            group["visual_description"],
            f"palette.groups[{index}].visual_description",
        )
        boxes = group["boxes"]
        if not isinstance(boxes, list) or not 1 <= len(boxes) <= MAX_GROUP_BOXES:
            raise StagedAnalysisError(
                f"palette.groups[{index}].boxes must contain 1..{MAX_GROUP_BOXES} boxes"
            )
        validated_boxes = [
            _validate_box(box, f"palette.groups[{index}].boxes[{box_index}]")
            for box_index, box in enumerate(boxes)
        ]
        for box_index, (x0, y0, x1, y1) in enumerate(validated_boxes):
            if (x1 - x0) * (y1 - y0) >= 850_000:
                raise StagedAnalysisError(
                    f"palette.groups[{index}].boxes[{box_index}] is an invalid "
                    "whole-image or near-whole-image citation; each box must "
                    "cover less than 85 percent of the normalized image area"
                )
        confidence = _require_confidence(
            group["confidence"], f"palette.groups[{index}].confidence"
        )
        validated_groups.append(
            {
                "group_id": group_id,
                "family_hint": family_hint,
                "base_color": base_color,
                "finish_hint": finish_hint,
                "visual_description": description,
                "boxes": validated_boxes,
                "confidence": confidence,
            }
        )

    return {
        "schema_version": PALETTE_SCHEMA_VERSION,
        "source_view_id": source_view_id,
        "groups": validated_groups,
    }


def validate_part_palette_batch(
    batch: Mapping[str, Any],
    *,
    target_part_ids: Iterable[str],
    palette: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one exact part batch against a previously validated palette."""

    target_parts = _require_id_set(target_part_ids, "target_part_ids")
    canonical_palette = validate_palette(palette)
    groups = {group["group_id"]: group for group in canonical_palette["groups"]}
    document = _require_mapping(batch, "batch")
    _require_exact_fields(document, {"schema_version", "batch_id", "mappings"}, "batch")
    if document["schema_version"] != BATCH_SCHEMA_VERSION:
        raise StagedAnalysisError(
            f"Unsupported batch schema_version: {document['schema_version']!r}"
        )
    batch_id = _require_nonempty_string(document["batch_id"], "batch.batch_id")
    if not _BATCH_ID_RE.fullmatch(batch_id):
        raise StagedAnalysisError("batch.batch_id must use B followed by 2..4 digits")
    mappings = document["mappings"]
    if not isinstance(mappings, list):
        raise StagedAnalysisError("batch.mappings must be an array")

    expected_mapping_fields = {
        "part_id",
        "group_id",
        "mapping_confidence",
        "evidence_view_id",
        "evidence_box_index",
        "status",
        "reason_code",
    }
    seen_parts: set[str] = set()
    validated_mappings: list[dict[str, Any]] = []
    for index, raw_mapping in enumerate(mappings):
        mapping = _require_mapping(raw_mapping, f"batch.mappings[{index}]")
        _require_exact_fields(
            mapping, expected_mapping_fields, f"batch.mappings[{index}]"
        )
        part_id = _require_nonempty_string(
            mapping["part_id"], f"batch.mappings[{index}].part_id"
        )
        if part_id in seen_parts:
            raise StagedAnalysisError(f"Duplicate mapping for part_id: {part_id}")
        seen_parts.add(part_id)
        confidence = _require_confidence(
            mapping["mapping_confidence"],
            f"batch.mappings[{index}].mapping_confidence",
        )
        status = mapping["status"]
        if status not in MAPPING_STATUSES:
            raise StagedAnalysisError(
                f"batch.mappings[{index}].status must be one of "
                f"{sorted(MAPPING_STATUSES)}"
            )
        reason_code = mapping["reason_code"]
        if not isinstance(reason_code, str):
            raise StagedAnalysisError(
                f"batch.mappings[{index}].reason_code must be a string"
            )

        if status == "matched":
            allowed_reasons = MATCHED_REASON_CODES
            if confidence < AUTO_THRESHOLD:
                raise StagedAnalysisError(
                    f"batch.mappings[{index}].status matched requires confidence "
                    f">= {AUTO_THRESHOLD}"
                )
        elif status == "review":
            allowed_reasons = REVIEW_REASON_CODES
            if not REVIEW_THRESHOLD <= confidence < AUTO_THRESHOLD:
                raise StagedAnalysisError(
                    f"batch.mappings[{index}].status review requires "
                    f"{REVIEW_THRESHOLD} <= confidence < {AUTO_THRESHOLD}"
                )
        else:
            allowed_reasons = UNKNOWN_REASON_CODES
            if confidence >= REVIEW_THRESHOLD:
                raise StagedAnalysisError(
                    f"batch.mappings[{index}].status unknown requires confidence "
                    f"< {REVIEW_THRESHOLD}"
                )
        if reason_code not in allowed_reasons:
            raise StagedAnalysisError(
                f"batch.mappings[{index}].reason_code {reason_code!r} is invalid "
                f"for status {status}"
            )

        group_id = mapping["group_id"]
        evidence_view_id = mapping["evidence_view_id"]
        evidence_box_index = mapping["evidence_box_index"]
        if status == "unknown":
            if any(
                value is not None
                for value in (group_id, evidence_view_id, evidence_box_index)
            ):
                raise StagedAnalysisError(
                    f"batch.mappings[{index}] unknown mapping must use null group "
                    "and evidence fields"
                )
        else:
            if not isinstance(group_id, str) or group_id not in groups:
                raise StagedAnalysisError(
                    f"batch.mappings[{index}] contains unknown group_id: {group_id!r}"
                )
            if evidence_view_id != canonical_palette["source_view_id"]:
                raise StagedAnalysisError(
                    f"batch.mappings[{index}].evidence_view_id must equal the "
                    "palette source reference"
                )
            if (
                isinstance(evidence_box_index, bool)
                or not isinstance(evidence_box_index, int)
                or not 0 <= evidence_box_index < len(groups[group_id]["boxes"])
            ):
                raise StagedAnalysisError(
                    f"batch.mappings[{index}].evidence_box_index is invalid for "
                    f"group {group_id}"
                )

        validated_mappings.append(
            {
                "part_id": part_id,
                "group_id": group_id,
                "mapping_confidence": confidence,
                "evidence_view_id": evidence_view_id,
                "evidence_box_index": evidence_box_index,
                "status": status,
                "reason_code": reason_code,
            }
        )

    if seen_parts != target_parts:
        missing = sorted(target_parts - seen_parts)
        unexpected = sorted(seen_parts - target_parts)
        raise StagedAnalysisError(
            f"batch {batch_id} does not exactly cover its target parts; "
            f"missing={missing}, unexpected={unexpected}"
        )
    return {
        "schema_version": BATCH_SCHEMA_VERSION,
        "batch_id": batch_id,
        "mappings": validated_mappings,
    }


def validate_group_materials(
    selections: Mapping[str, Any],
    *,
    palette: Mapping[str, Any],
    allowed_material_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Validate the small, independently confirmed MDL choice per palette group."""

    canonical_palette = validate_palette(palette)
    group_ids = {group["group_id"] for group in canonical_palette["groups"]}
    allowed_materials = (
        _require_id_set(allowed_material_ids, "allowed_material_ids")
        if allowed_material_ids is not None
        else None
    )
    document = _require_mapping(selections, "material selections")
    _require_exact_fields(
        document, {"schema_version", "selections"}, "material selections"
    )
    if document["schema_version"] != GROUP_MATERIAL_SCHEMA_VERSION:
        raise StagedAnalysisError(
            "Unsupported material-selection schema_version: "
            f"{document['schema_version']!r}"
        )
    raw_selections = document["selections"]
    if not isinstance(raw_selections, list):
        raise StagedAnalysisError("material selections must be an array")

    seen_groups: set[str] = set()
    validated: list[dict[str, Any]] = []
    expected_fields = {"group_id", "material_id", "confidence", "confirmed"}
    for index, raw_selection in enumerate(raw_selections):
        selection = _require_mapping(raw_selection, f"material selections[{index}]")
        _require_exact_fields(
            selection, expected_fields, f"material selections[{index}]"
        )
        group_id = _require_nonempty_string(
            selection["group_id"], f"material selections[{index}].group_id"
        )
        if group_id not in group_ids:
            raise StagedAnalysisError(
                f"material selections[{index}] contains unknown group_id: {group_id}"
            )
        if group_id in seen_groups:
            raise StagedAnalysisError(
                f"Duplicate material selection for group_id: {group_id}"
            )
        seen_groups.add(group_id)
        material_id = _require_nonempty_string(
            selection["material_id"],
            f"material selections[{index}].material_id",
        )
        if allowed_materials is not None and material_id not in allowed_materials:
            raise StagedAnalysisError(
                f"material selections[{index}] contains unknown material_id: "
                f"{material_id!r}"
            )
        confidence = _require_confidence(
            selection["confidence"],
            f"material selections[{index}].confidence",
        )
        confirmed = selection["confirmed"]
        if not isinstance(confirmed, bool):
            raise StagedAnalysisError(
                f"material selections[{index}].confirmed must be a boolean"
            )
        validated.append(
            {
                "group_id": group_id,
                "material_id": material_id,
                "confidence": confidence,
                "confirmed": confirmed,
            }
        )
    return {
        "schema_version": GROUP_MATERIAL_SCHEMA_VERSION,
        "selections": validated,
    }


def detect_material_collapse(
    *,
    palette: Mapping[str, Any],
    mappings: Sequence[Mapping[str, Any]],
    material_selections: Mapping[str, Any],
    dominance_threshold: float = 0.80,
    minimum_assignments: int = 8,
    minimum_eligible_retention_share: float = 0.20,
    pre_filter_palette_group_count: int | None = None,
    orientation_confidence: float = 1.0,
    total_part_count: int | None = None,
) -> dict[str, Any]:
    """Detect dominant-group, material, and evidence-filter collapse patterns.

    ``pre_filter_palette_group_count`` is optional so older callers retain the
    previous behavior.  When supplied, it makes loss of a multi-group model
    palette during deterministic evidence filtering visible to the audit.
    """

    if (
        isinstance(dominance_threshold, bool)
        or not isinstance(dominance_threshold, (int, float))
        or not 0.0 < dominance_threshold <= 1.0
    ):
        raise StagedAnalysisError("dominance_threshold must be in (0, 1]")
    if (
        isinstance(minimum_assignments, bool)
        or not isinstance(minimum_assignments, int)
        or minimum_assignments < 2
    ):
        raise StagedAnalysisError("minimum_assignments must be an integer >= 2")
    if (
        isinstance(minimum_eligible_retention_share, bool)
        or not isinstance(minimum_eligible_retention_share, (int, float))
        or not math.isfinite(minimum_eligible_retention_share)
        or not 0.0 < minimum_eligible_retention_share <= 1.0
    ):
        raise StagedAnalysisError(
            "minimum_eligible_retention_share must be in (0, 1]"
        )
    normalized_minimum_eligible_retention_share = float(
        minimum_eligible_retention_share
    )

    canonical_palette = validate_palette(palette)
    orientation = _require_confidence(
        orientation_confidence, "orientation_confidence"
    )
    filtered_palette_group_count = len(canonical_palette["groups"])
    if pre_filter_palette_group_count is None:
        original_palette_group_count = filtered_palette_group_count
        palette_filter_context_supplied = False
    else:
        if (
            isinstance(pre_filter_palette_group_count, bool)
            or not isinstance(pre_filter_palette_group_count, int)
            or pre_filter_palette_group_count < filtered_palette_group_count
        ):
            raise StagedAnalysisError(
                "pre_filter_palette_group_count must be an integer greater than "
                "or equal to the filtered palette group count"
            )
        original_palette_group_count = pre_filter_palette_group_count
        palette_filter_context_supplied = True
    canonical_selections = validate_group_materials(
        material_selections, palette=canonical_palette
    )
    group_records = {group["group_id"]: group for group in canonical_palette["groups"]}
    selection_by_group = {
        selection["group_id"]: selection
        for selection in canonical_selections["selections"]
    }

    usable_mappings: list[Mapping[str, Any]] = []
    for index, mapping in enumerate(mappings):
        record = _require_mapping(mapping, f"mappings[{index}]")
        status = record.get("status")
        group_id = record.get("group_id")
        if status == "unknown":
            continue
        if status not in {"matched", "review"} or group_id not in group_records:
            raise StagedAnalysisError(
                f"mappings[{index}] is not a validated part-to-palette mapping"
            )
        usable_mappings.append(record)

    if total_part_count is None:
        normalized_total_part_count = len(mappings)
    elif (
        isinstance(total_part_count, bool)
        or not isinstance(total_part_count, int)
        or total_part_count < len(mappings)
    ):
        raise StagedAnalysisError(
            "total_part_count must be an integer greater than or equal to "
            "the number of supplied mappings"
        )
    else:
        normalized_total_part_count = total_part_count

    # A concentration is dangerous only when it can actually enter the
    # material plan.  Counting review rows whose palette/material confidence
    # will immediately reduce them to ``low_combined_confidence`` produced a
    # false hard failure before the real merge gate had run.
    eligible_mappings = [
        mapping
        for mapping in usable_mappings
        if mapping["group_id"] in selection_by_group
        and min(
            orientation,
            group_records[mapping["group_id"]]["confidence"],
            float(mapping["mapping_confidence"]),
            selection_by_group[mapping["group_id"]]["confidence"],
        )
        >= REVIEW_THRESHOLD
    ]
    raw_group_counts = Counter(
        mapping["group_id"] for mapping in usable_mappings
    )
    raw_material_counts = Counter(
        selection_by_group[mapping["group_id"]]["material_id"]
        for mapping in usable_mappings
        if mapping["group_id"] in selection_by_group
    )
    raw_mapped_count = len(usable_mappings)
    raw_selected_count = sum(raw_material_counts.values())
    raw_dominant_group_share = (
        max(raw_group_counts.values()) / raw_mapped_count
        if raw_mapped_count
        else 0.0
    )
    raw_dominant_material_share = (
        max(raw_material_counts.values()) / raw_selected_count
        if raw_selected_count
        else 0.0
    )

    group_counts = Counter(mapping["group_id"] for mapping in eligible_mappings)
    material_counts = Counter(
        selection_by_group[mapping["group_id"]]["material_id"]
        for mapping in eligible_mappings
        if mapping["group_id"] in selection_by_group
    )
    mapped_count = len(eligible_mappings)
    selected_count = sum(material_counts.values())
    dominant_group_share = (
        max(group_counts.values()) / mapped_count if mapped_count else 0.0
    )
    dominant_material_share = (
        max(material_counts.values()) / selected_count if selected_count else 0.0
    )
    absolute_dominant_group_share = (
        max(group_counts.values()) / normalized_total_part_count
        if group_counts and normalized_total_part_count
        else 0.0
    )
    eligible_coverage_share = (
        mapped_count / normalized_total_part_count
        if normalized_total_part_count
        else 0.0
    )
    raw_to_eligible_retention_share = (
        mapped_count / raw_mapped_count if raw_mapped_count else 0.0
    )
    reliable_groups = [
        group
        for group in canonical_palette["groups"]
        if group["confidence"] >= REVIEW_THRESHOLD
    ]
    authorable_material_group_count = sum(
        selection["confidence"] >= REVIEW_THRESHOLD
        for selection in canonical_selections["selections"]
    )
    used_group_ids = set(group_counts)
    used_colors = {group_records[group_id]["base_color"] for group_id in used_group_ids}
    used_materials = {
        selection_by_group[group_id]["material_id"]
        for group_id in used_group_ids
        if group_id in selection_by_group
    }

    reasons: list[str] = []
    recovery_reasons: list[str] = []
    palette_filter_collapse = (
        palette_filter_context_supplied
        and original_palette_group_count > 1
        and filtered_palette_group_count == 1
    )
    if palette_filter_collapse:
        reasons.append(
            "palette evidence filtering reduced multiple model groups to one group"
        )
    if (
        len(used_colors) >= 2
        and len(used_materials) == 1
        and all(group_id in selection_by_group for group_id in used_group_ids)
    ):
        reasons.append("different palette colors resolved to one material_id")
    if (
        mapped_count >= minimum_assignments
        and len(reliable_groups) >= 3
        and dominant_group_share >= dominance_threshold
        and dominant_material_share >= dominance_threshold
        and absolute_dominant_group_share >= dominance_threshold
    ):
        reasons.append(
            "one palette group and material dominate a visibly diverse reference"
        )
    evidence_starvation = (
        len(reliable_groups) >= 3
        and raw_mapped_count >= minimum_assignments
        and authorable_material_group_count >= 1
        and mapped_count < minimum_assignments
        and raw_to_eligible_retention_share
        < normalized_minimum_eligible_retention_share
    )
    if evidence_starvation:
        # This is not itself a semantic/material collapse.  It is a sealed
        # signal that the high-confidence 2D-to-CAD join needs the downstream
        # spatial recovery lanes.  Treating it as ``detected`` here would stop
        # the staged workflow before those geometry-backed lanes can run.
        recovery_reasons.append(
            "authoring evidence starved after confidence filtering"
        )

    return {
        "detected": bool(reasons),
        "reasons": reasons,
        "recovery_required": evidence_starvation,
        "recovery_reasons": recovery_reasons,
        "mapped_assignment_count": mapped_count,
        "selected_assignment_count": selected_count,
        "raw_mapped_assignment_count": raw_mapped_count,
        "raw_selected_assignment_count": raw_selected_count,
        "total_part_count": normalized_total_part_count,
        "eligible_coverage_share": eligible_coverage_share,
        "raw_to_eligible_retention_share": raw_to_eligible_retention_share,
        "minimum_eligible_retention_share": (
            normalized_minimum_eligible_retention_share
        ),
        "minimum_assignments_threshold": minimum_assignments,
        "authorable_material_group_count": authorable_material_group_count,
        "evidence_starvation": evidence_starvation,
        "palette_filter_context_supplied": palette_filter_context_supplied,
        "pre_filter_palette_group_count": original_palette_group_count,
        "filtered_palette_group_count": filtered_palette_group_count,
        "palette_filter_collapse": palette_filter_collapse,
        "reliable_palette_group_count": len(reliable_groups),
        "used_group_count": len(used_group_ids),
        "used_material_count": len(used_materials),
        "dominant_group_share": dominant_group_share,
        "dominant_material_share": dominant_material_share,
        "absolute_dominant_group_share": absolute_dominant_group_share,
        "raw_dominant_group_share": raw_dominant_group_share,
        "raw_dominant_material_share": raw_dominant_material_share,
    }


def merge_staged_results(
    *,
    palette: Mapping[str, Any],
    batches: Sequence[Mapping[str, Any]],
    batch_targets: Mapping[str, Iterable[str]],
    material_selections: Mapping[str, Any],
    all_part_ids: Iterable[str],
    forced_unknown_parts: Mapping[str, str] | None = None,
    allowed_material_ids: Iterable[str] | None = None,
    orientation_confidence: float = 1.0,
    dominance_threshold: float = 0.80,
    minimum_collapse_assignments: int = 8,
    minimum_eligible_retention_share: float = 0.20,
    pre_filter_palette_group_count: int | None = None,
) -> dict[str, Any]:
    """Validate and merge disjoint batches into a non-applying unknown audit.

    Every registered part must be either present in exactly one declared batch
    target or explicitly listed in ``forced_unknown_parts``.  Batch-level
    unknown mappings and forced unknowns are omitted from ``material_plan``.
    """

    canonical_palette = validate_palette(palette)
    orientation = _require_confidence(orientation_confidence, "orientation_confidence")
    all_parts = _require_id_set(all_part_ids, "all_part_ids")
    if not isinstance(batch_targets, Mapping):
        raise StagedAnalysisError("batch_targets must be an object")

    normalized_targets: dict[str, set[str]] = {}
    targeted_parts: set[str] = set()
    for raw_batch_id, raw_targets in batch_targets.items():
        batch_id = _require_nonempty_string(raw_batch_id, "batch_targets batch_id")
        if not _BATCH_ID_RE.fullmatch(batch_id):
            raise StagedAnalysisError(
                f"batch target ID must use B followed by 2..4 digits: {batch_id}"
            )
        targets = _require_id_set(raw_targets, f"batch_targets[{batch_id}]")
        overlap = targeted_parts & targets
        if overlap:
            raise StagedAnalysisError(
                f"Part IDs occur in multiple batch targets: {sorted(overlap)}"
            )
        targeted_parts.update(targets)
        normalized_targets[batch_id] = targets

    forced = forced_unknown_parts or {}
    if not isinstance(forced, Mapping):
        raise StagedAnalysisError("forced_unknown_parts must be an object")
    normalized_forced: dict[str, str] = {}
    for raw_part_id, reason_code in forced.items():
        part_id = _require_nonempty_string(raw_part_id, "forced_unknown_parts part_id")
        if reason_code not in UNKNOWN_REASON_CODES:
            raise StagedAnalysisError(
                f"forced_unknown_parts[{part_id}] has invalid reason_code: "
                f"{reason_code!r}"
            )
        normalized_forced[part_id] = reason_code
    forced_parts = set(normalized_forced)
    overlap = targeted_parts & forced_parts
    if overlap:
        raise StagedAnalysisError(
            f"Parts cannot be both targeted and forced unknown: {sorted(overlap)}"
        )
    covered_parts = targeted_parts | forced_parts
    if covered_parts != all_parts:
        missing = sorted(all_parts - covered_parts)
        unexpected = sorted(covered_parts - all_parts)
        raise StagedAnalysisError(
            "Batch targets and forced unknowns do not exactly cover all parts; "
            f"missing={missing}, unexpected={unexpected}"
        )

    if isinstance(batches, (str, bytes)) or not isinstance(batches, Sequence):
        raise StagedAnalysisError("batches must be an array")
    canonical_batches: list[dict[str, Any]] = []
    seen_batch_ids: set[str] = set()
    for index, raw_batch in enumerate(batches):
        document = _require_mapping(raw_batch, f"batches[{index}]")
        batch_id = document.get("batch_id")
        if not isinstance(batch_id, str) or batch_id not in normalized_targets:
            raise StagedAnalysisError(
                f"batches[{index}] has undeclared batch_id: {batch_id!r}"
            )
        if batch_id in seen_batch_ids:
            raise StagedAnalysisError(f"Duplicate batch result: {batch_id}")
        seen_batch_ids.add(batch_id)
        canonical_batches.append(
            validate_part_palette_batch(
                document,
                target_part_ids=normalized_targets[batch_id],
                palette=canonical_palette,
            )
        )
    if seen_batch_ids != set(normalized_targets):
        missing_batches = sorted(set(normalized_targets) - seen_batch_ids)
        raise StagedAnalysisError(f"Missing batch results: {missing_batches}")

    canonical_selections = validate_group_materials(
        material_selections,
        palette=canonical_palette,
        allowed_material_ids=allowed_material_ids,
    )
    selection_by_group = {
        selection["group_id"]: selection
        for selection in canonical_selections["selections"]
    }
    group_by_id = {group["group_id"]: group for group in canonical_palette["groups"]}
    mappings = [mapping for batch in canonical_batches for mapping in batch["mappings"]]
    collapse = detect_material_collapse(
        palette=canonical_palette,
        mappings=mappings,
        material_selections=canonical_selections,
        dominance_threshold=dominance_threshold,
        minimum_assignments=minimum_collapse_assignments,
        minimum_eligible_retention_share=minimum_eligible_retention_share,
        pre_filter_palette_group_count=pre_filter_palette_group_count,
        orientation_confidence=orientation,
        total_part_count=len(all_parts),
    )
    if collapse["detected"]:
        raise MaterialCollapseError(collapse)

    assignments: list[dict[str, Any]] = []
    unknown_by_part: dict[str, str] = dict(normalized_forced)
    for mapping in mappings:
        part_id = mapping["part_id"]
        if mapping["status"] == "unknown":
            unknown_by_part[part_id] = mapping["reason_code"]
            continue
        group_id = mapping["group_id"]
        selection = selection_by_group.get(group_id)
        if selection is None:
            unknown_by_part[part_id] = "missing_material_selection"
            continue
        group = group_by_id[group_id]
        combined_confidence = min(
            orientation,
            group["confidence"],
            mapping["mapping_confidence"],
            selection["confidence"],
        )
        if combined_confidence < REVIEW_THRESHOLD:
            unknown_by_part[part_id] = "low_combined_confidence"
            continue
        can_auto = (
            combined_confidence >= AUTO_THRESHOLD
            and mapping["status"] == "matched"
            and selection["confirmed"]
        )
        if can_auto:
            status = "auto"
            final_confidence = combined_confidence
        else:
            status = "review"
            final_confidence = min(combined_confidence, REVIEW_CONFIDENCE_CAP)
        assignments.append(
            {
                "part_id": part_id,
                "material_id": selection["material_id"],
                "semantic": group["visual_description"],
                "confidence": final_confidence,
                "evidence_views": [canonical_palette["source_view_id"]],
                "status": status,
            }
        )

    assignments.sort(key=lambda assignment: assignment["part_id"])
    unknown_parts = [
        {"part_id": part_id, "reason_code": unknown_by_part[part_id]}
        for part_id in sorted(unknown_by_part)
    ]
    if {assignment["part_id"] for assignment in assignments} & set(unknown_by_part):
        raise AssertionError("A part cannot be both assigned and unknown")
    if {assignment["part_id"] for assignment in assignments} | set(
        unknown_by_part
    ) != all_parts:
        raise AssertionError("Merged result lost registered parts")

    auto_count = sum(assignment["status"] == "auto" for assignment in assignments)
    review_count = len(assignments) - auto_count
    return {
        "schema_version": STAGED_RESULT_SCHEMA_VERSION,
        "material_plan": {
            "schema_version": MATERIAL_PLAN_SCHEMA_VERSION,
            "assignments": assignments,
        },
        "unknown_parts": unknown_parts,
        "audit": {
            "source_view_id": canonical_palette["source_view_id"],
            "orientation_confidence": orientation,
            "palette_group_count": len(canonical_palette["groups"]),
            "batch_count": len(canonical_batches),
            "part_count": len(all_parts),
            "auto_count": auto_count,
            "review_count": review_count,
            "unknown_count": len(unknown_parts),
            "collapse_check": collapse,
        },
    }


__all__ = [
    "BATCH_SCHEMA_VERSION",
    "GROUP_MATERIAL_SCHEMA_VERSION",
    "PALETTE_SCHEMA_VERSION",
    "STAGED_RESULT_SCHEMA_VERSION",
    "MaterialCollapseError",
    "StagedAnalysisError",
    "UNKNOWN_REASON_CODES",
    "detect_material_collapse",
    "merge_staged_results",
    "normalize_part_palette_batch",
    "validate_group_materials",
    "validate_palette",
    "validate_part_palette_batch",
]
