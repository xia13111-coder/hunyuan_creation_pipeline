"""Annotate restored material plans with conservative canonical visual groups.

Older reviewed/restored plans can contain useful visual semantics and exact
NVIDIA MDL identities without the newer ``provenance.canonical_group_id``.
This module recovers only high-confidence appearance-group links from those
tokens.  Assignment groups live directly in their provenance; face-subset
groups live in the schema-safe assignment-level
``provenance.face_subset_canonical_group_ids`` map.  It never changes a
material identity, parameter, or face-subset authoring field, never uses a
part-ID allowlist, and leaves ambiguous or weak entries untouched.

Exact colour/family agreement is the normal path.  A small, explicit colour
neighbour graph (for example brown/copper to orange) is accepted only at a
low-confidence tier.  Existing valid canonical-group provenance is
authoritative and is never overwritten by token inference.

``policy_fallback`` material names are deliberately excluded from inference.
A fallback such as ``Steel_Stainless`` describes what the policy filled in; it
is not independent evidence that the corresponding CAD part occupies the
silver region in the reference photographs.  Reversing that fallback into a
canonical group would let one neutral default label hundreds of unrelated
parts and contaminate the later exact-MDL render tournament.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


PLAN_SCHEMA_VERSION = "1.0"
AUDIT_SCHEMA_VERSION = "qwen-visual-group-plan-annotation/v1"
ANNOTATION_METHOD = "semantic_material_token_consensus/v1"
SPATIAL_ANNOTATION_METHOD = "trusted_multiview_spatial_projection/v1"
SEMANTIC_SPATIAL_ANNOTATION_METHOD = (
    "corroborated_multiview_semantic_projection/v1"
)
SINGLE_VIEW_SPATIAL_ANNOTATION_METHOD = (
    "high_confidence_single_view_spatial_projection/v1"
)
FOREGROUND_SINGLE_VIEW_SPATIAL_ANNOTATION_METHOD = (
    "sam3_foreground_single_view_projection/v1"
)
MULTIVIEW_DIRECT_SPATIAL_ANNOTATION_METHOD = (
    "corroborated_multiview_direct_mask_projection/v1"
)
PALETTE_BOUND_SMALL_PART_SPATIAL_ANNOTATION_METHOD = (
    "multiview_palette_bound_small_part_projection/v1"
)
SOURCE_APPEARANCE_COHORT_METHOD = (
    "trusted_spatial_anchor_source_appearance_cohort/v1"
)
SOURCE_APPEARANCE_COHORT_SCHEMA_VERSION = (
    "qwen-source-appearance-cohort-propagation/v1"
)
SOURCE_APPEARANCE_COHORT_CONTRACT_SCHEMA_VERSION = (
    "qwen-source-appearance-cohort-contract/v1"
)
SOURCE_APPEARANCE_COHORT_MIN_MEMBER_COUNT = 2
SOURCE_APPEARANCE_COHORT_MAX_MEMBER_COUNT = 128
SOURCE_APPEARANCE_COHORT_SMALL_REGISTRY_ALLOWANCE = 6
SOURCE_APPEARANCE_COHORT_MAX_REGISTRY_FRACTION = 0.20
SOURCE_APPEARANCE_COHORT_MIN_SIGNATURE_PART_SHARE = 0.90
SOURCE_APPEARANCE_COHORT_MIN_SIGNATURE_FACE_SHARE = 0.90
SOURCE_APPEARANCE_RARE_PAIR_SIZE = 2
REPEATED_SUBSET_VISUAL_COHORT_METHOD = (
    "trusted_multiview_repeated_subset_visual_cohort/v1"
)
REPEATED_SUBSET_VISUAL_COHORT_SCHEMA_VERSION = (
    "qwen-repeated-subset-visual-cohort/v1"
)
REPEATED_SUBSET_VISUAL_COHORT_MAX_MEMBER_COUNT = 16
POLICY_FALLBACK_STATUS = "policy_fallback"
MINIMUM_CANONICAL_CONFIDENCE = 0.60
MINIMUM_EXACT_CONFIDENCE = 0.82
MINIMUM_NEIGHBOR_CONFIDENCE = 0.65
MINIMUM_SELECTION_MARGIN = 0.08
MAXIMUM_NEIGHBOR_CONFIDENCE = 0.74
MAXIMUM_SEMANTIC_RECOVERY_VISIBLE_PIXELS = 1024
MINIMUM_SEMANTIC_RECOVERY_VISIBLE_PIXELS = 32
MINIMUM_SINGLE_VIEW_SPATIAL_PIXELS = 384
MINIMUM_SINGLE_VIEW_COLOR_SHARE = 0.75
MINIMUM_SINGLE_VIEW_COLOR_MARGIN = 0.50
MINIMUM_SINGLE_VIEW_BBOX_COLOR_MARGIN = 0.70
MINIMUM_DIRECT_MULTIVIEW_AGREEMENT_VIEWS = 3
MINIMUM_DIRECT_MULTIVIEW_STRONG_VIEWS = 2
MINIMUM_DIRECT_MULTIVIEW_PIXELS = 128
MINIMUM_DIRECT_MULTIVIEW_COLOR_SHARE = 0.60
MINIMUM_DIRECT_MULTIVIEW_COLOR_MARGIN = 0.35
MINIMUM_DIRECT_MULTIVIEW_SEMANTIC_CONFLICT_CONFIDENCE = 0.85
MINIMUM_SPATIAL_FOREGROUND_OVERLAP = 0.50
MINIMUM_SPATIAL_FOREGROUND_PIXELS = 128
MINIMUM_FOREGROUND_SINGLE_VIEW_PIXELS = 12
MINIMUM_FOREGROUND_SINGLE_VIEW_COLOR_SHARE = 0.70
MINIMUM_FOREGROUND_SINGLE_VIEW_COLOR_MARGIN = 0.35

EXIT_SUCCESS = 0
EXIT_INPUT_ERROR = 2
EXIT_REQUIRE_UNAMBIGUOUS_FAILED = 3

_COLOR_ALIASES: dict[str, tuple[str, float, str]] = {
    "black": ("black", 1.0, "explicit"),
    "charcoal": ("black", 0.75, "appearance_inference"),
    "blued": ("black", 0.75, "appearance_inference"),
    "white": ("white", 1.0, "explicit"),
    "silver": ("silver", 1.0, "explicit"),
    "gray": ("silver", 0.90, "normalized_achromatic"),
    "grey": ("silver", 0.90, "normalized_achromatic"),
    "stainless": ("silver", 0.78, "material_appearance_inference"),
    "galvanized": ("silver", 0.75, "material_appearance_inference"),
    "chrome": ("silver", 0.75, "material_appearance_inference"),
    "zinc": ("silver", 0.72, "material_appearance_inference"),
    "green": ("green", 1.0, "explicit"),
    "mint": ("green", 0.75, "appearance_inference"),
    "lime": ("green", 0.75, "appearance_inference"),
    "blue": ("blue", 1.0, "explicit"),
    "cyan": ("blue", 0.80, "normalized_neighbour"),
    "orange": ("orange", 1.0, "explicit"),
    "rust": ("orange", 0.70, "appearance_inference"),
    "rusted": ("orange", 0.70, "appearance_inference"),
    "brown": ("brown", 1.0, "explicit"),
    "tan": ("brown", 0.75, "appearance_inference"),
    "copper": ("copper", 1.0, "explicit"),
    "bronze": ("bronze", 1.0, "explicit"),
    "red": ("red", 1.0, "explicit"),
    "yellow": ("yellow", 1.0, "explicit"),
}

_CJK_COLOR_ALIASES: dict[str, tuple[str, float, str]] = {
    "黑": ("black", 1.0, "explicit"),
    "白": ("white", 1.0, "explicit"),
    "银": ("silver", 1.0, "explicit"),
    "灰": ("silver", 0.90, "normalized_achromatic"),
    "绿": ("green", 1.0, "explicit"),
    "蓝": ("blue", 1.0, "explicit"),
    "青": ("blue", 0.80, "normalized_neighbour"),
    "橙": ("orange", 1.0, "explicit"),
    "棕": ("brown", 1.0, "explicit"),
    "铜": ("copper", 1.0, "explicit"),
    "红": ("red", 1.0, "explicit"),
    "黄": ("yellow", 1.0, "explicit"),
}

_FAMILY_ALIASES: dict[str, str] = {
    "metal": "metal",
    "metals": "metal",
    "steel": "metal",
    "stainless": "metal",
    "aluminum": "metal",
    "aluminium": "metal",
    "anodized": "metal",
    "iron": "metal",
    "copper": "metal",
    "brass": "metal",
    "bronze": "metal",
    "zinc": "metal",
    "chrome": "metal",
    "nickel": "metal",
    "titanium": "metal",
    "plastic": "plastic",
    "plastics": "plastic",
    "polymer": "plastic",
    "polypropylene": "plastic",
    "polyethylene": "plastic",
    "polycarbonate": "plastic",
    "abs": "plastic",
    "nylon": "plastic",
    "pvc": "plastic",
    "acrylic": "plastic",
    "resin": "plastic",
    "rubber": "rubber",
    "silicone": "rubber",
    "latex": "rubber",
    "caoutchouc": "rubber",
    "elastomer": "rubber",
    "glass": "glass",
    "ceramic": "ceramic",
    "fabric": "fabric",
    "textile": "fabric",
    "wood": "wood",
}

_CJK_FAMILY_ALIASES: dict[str, str] = {
    "金属": "metal",
    "钢": "metal",
    "铝": "metal",
    "铁": "metal",
    "铜": "metal",
    "塑料": "plastic",
    "塑胶": "plastic",
    "橡胶": "rubber",
    "硅胶": "rubber",
    "玻璃": "glass",
    "陶瓷": "ceramic",
    "织物": "fabric",
    "木": "wood",
}

# These relations are deliberately sparse and symmetric.  They represent
# nearby visible colour names, not semantic material-category equivalence.
_LOW_CONFIDENCE_COLOR_NEIGHBORS = {
    frozenset(("brown", "orange")),
    frozenset(("copper", "orange")),
    frozenset(("bronze", "orange")),
}

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_CAMEL_BOUNDARY_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


class VisualGroupAnnotationError(ValueError):
    """Raised when annotation inputs violate the deterministic trust boundary."""


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise VisualGroupAnnotationError(f"{label} must be an object")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise VisualGroupAnnotationError(f"{label} must be an array")
    return value


def _text(value: Any, label: str, *, required: bool = True) -> str:
    if value is None and not required:
        return ""
    if not isinstance(value, str) or (required and not value.strip()):
        qualifier = "a non-empty string" if required else "a string"
        raise VisualGroupAnnotationError(f"{label} must be {qualifier}")
    return value.strip()


def _confidence(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise VisualGroupAnnotationError(f"{label} must be a finite number from 0 to 1")
    return float(value)


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tokens(value: str) -> set[str]:
    separated = _CAMEL_BOUNDARY_RE.sub(" ", value)
    return set(_TOKEN_RE.findall(separated.casefold()))


def _source_signals(source: str, value: str) -> dict[str, Any]:
    tokens = _tokens(value)
    colors: list[dict[str, Any]] = []
    for token in sorted(tokens):
        alias = _COLOR_ALIASES.get(token)
        if alias is None:
            continue
        color, strength, basis = alias
        colors.append(
            {
                "source": source,
                "token": token,
                "color": color,
                "strength": strength,
                "basis": basis,
            }
        )
    for token, alias in _CJK_COLOR_ALIASES.items():
        if token not in value:
            continue
        color, strength, basis = alias
        colors.append(
            {
                "source": source,
                "token": token,
                "color": color,
                "strength": strength,
                "basis": basis,
            }
        )

    families = {
        family for token in tokens if (family := _FAMILY_ALIASES.get(token)) is not None
    }
    for token, family in _CJK_FAMILY_ALIASES.items():
        if token in value:
            families.add(family)
    # Some MDL catalogs place rubber below a generic Plastics folder.  Rubber
    # is the more specific visual family and must not become an artificial
    # plastic/rubber conflict.
    if "rubber" in families:
        families.discard("plastic")
    return {
        "source": source,
        "raw": value,
        "tokens": sorted(tokens),
        "colors": colors,
        "families": sorted(families),
    }


def _canonical_color(value: str) -> str:
    signals = _source_signals("canonical_base_color", value)["colors"]
    colors = {signal["color"] for signal in signals}
    if len(colors) != 1:
        raise VisualGroupAnnotationError(
            f"canonical base_color {value!r} must resolve to exactly one colour"
        )
    return next(iter(colors))


def _canonical_family(value: str) -> str:
    if value.casefold().strip() == "other":
        # Qwen uses ``other`` when the visual region is trusted but its
        # physical family is unresolved.  Keep it as an explicit wildcard
        # handled by a lower-confidence gate; do not inject ``other`` into
        # assignment token extraction because MDL paths commonly contain a
        # generic ``Other/`` catalog folder.
        return "other"
    signals = _source_signals("canonical_family", value)["families"]
    if len(signals) != 1:
        raise VisualGroupAnnotationError(
            f"canonical family_hint {value!r} must resolve to exactly one family"
        )
    return str(signals[0])


def _canonical_groups(
    palette_fusion: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    canonical_palette = _mapping(
        palette_fusion.get("canonical_palette"),
        "palette_fusion.canonical_palette",
    )
    groups: dict[str, dict[str, Any]] = {}
    for index, raw_group in enumerate(
        _sequence(canonical_palette.get("groups"), "canonical_palette.groups")
    ):
        group = _mapping(raw_group, f"canonical_palette.groups[{index}]")
        group_id = _text(
            group.get("group_id"),
            f"canonical_palette.groups[{index}].group_id",
        )
        if group_id in groups:
            raise VisualGroupAnnotationError(
                f"palette fusion repeats canonical group {group_id}"
            )
        base_color = _text(
            group.get("base_color"),
            f"canonical group {group_id}.base_color",
        )
        family_hint = _text(
            group.get("family_hint"),
            f"canonical group {group_id}.family_hint",
        )
        canonical_confidence = _confidence(
            group.get("confidence"),
            f"canonical group {group_id}.confidence",
        )
        raw_source_view_ids = group.get("source_view_ids", [])
        if (
            isinstance(raw_source_view_ids, (str, bytes))
            or not isinstance(raw_source_view_ids, Sequence)
            or any(
                not isinstance(view_id, str) or not view_id
                for view_id in raw_source_view_ids
            )
            or len(set(raw_source_view_ids)) != len(raw_source_view_ids)
        ):
            raise VisualGroupAnnotationError(
                f"canonical group {group_id}.source_view_ids is invalid"
            )
        groups[group_id] = {
            "group_id": group_id,
            "base_color": base_color,
            "normalized_color": _canonical_color(base_color),
            "family_hint": family_hint,
            "normalized_family": _canonical_family(family_hint),
            "canonical_confidence": canonical_confidence,
            "source_view_ids": sorted(raw_source_view_ids),
            "eligible": canonical_confidence >= MINIMUM_CANONICAL_CONFIDENCE,
        }
    if not groups:
        raise VisualGroupAnnotationError("palette fusion has no canonical groups")
    return groups


def _combined_signals(
    *,
    semantic: str,
    material_id: str,
) -> dict[str, Any]:
    source_records = [
        _source_signals("semantic", semantic),
        _source_signals("material_id", material_id),
    ]
    colors = {
        signal["color"]
        for source_record in source_records
        for signal in source_record["colors"]
    }
    families_by_source = [
        set(source_record["families"])
        for source_record in source_records
        if source_record["families"]
    ]
    families = set().union(*families_by_source) if families_by_source else set()
    reason_codes: list[str] = []
    if not colors:
        reason_codes.append("NO_COLOR_TOKEN")
    elif len(colors) > 1:
        reason_codes.append("CONFLICTING_COLOR_TOKENS")
    if not families:
        reason_codes.append("NO_MATERIAL_FAMILY_TOKEN")
    elif len(families) > 1:
        reason_codes.append("CONFLICTING_MATERIAL_FAMILY_TOKENS")
    return {
        "sources": source_records,
        "normalized_colors": sorted(colors),
        "normalized_families": sorted(families),
        "reason_codes": reason_codes,
    }


def _candidate_matches(
    signals: Mapping[str, Any],
    canonical_groups: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    colors = signals["normalized_colors"]
    families = signals["normalized_families"]
    if len(colors) != 1 or len(families) != 1:
        return []
    observed_color = str(colors[0])
    observed_family = str(families[0])
    color_signal_records = [
        signal
        for source in signals["sources"]
        for signal in source["colors"]
        if signal["color"] == observed_color
    ]
    exact_color_strength = max(
        (float(signal["strength"]) for signal in color_signal_records),
        default=0.0,
    )
    color_source_count = len({str(signal["source"]) for signal in color_signal_records})
    family_source_count = sum(
        observed_family in source["families"] for source in signals["sources"]
    )

    candidates: list[dict[str, Any]] = []
    for group_id, group in sorted(canonical_groups.items()):
        canonical_family = str(group["normalized_family"])
        family_wildcard = canonical_family == "other"
        if not group["eligible"] or (
            not family_wildcard and canonical_family != observed_family
        ):
            continue
        canonical_color = str(group["normalized_color"])
        if canonical_color == observed_color:
            match_kind = (
                "LOW_CONFIDENCE_FAMILY_WILDCARD" if family_wildcard else "EXACT"
            )
            color_strength = exact_color_strength
            minimum_confidence = (
                MINIMUM_NEIGHBOR_CONFIDENCE
                if family_wildcard
                else MINIMUM_EXACT_CONFIDENCE
            )
        elif (
            frozenset((canonical_color, observed_color))
            in _LOW_CONFIDENCE_COLOR_NEIGHBORS
        ):
            if family_wildcard:
                # A colour neighbour plus an unresolved family is two
                # independent assumptions, so it cannot clear automation.
                continue
            match_kind = "LOW_CONFIDENCE_NEIGHBOR"
            color_strength = 0.55
            minimum_confidence = MINIMUM_NEIGHBOR_CONFIDENCE
        else:
            continue
        family_strength = 0.55 if family_wildcard else 1.0
        confidence = (
            0.15
            + 0.55 * color_strength
            + 0.25 * family_strength
            + 0.05 * float(group["canonical_confidence"])
        )
        if color_source_count >= 2:
            confidence += 0.015
        if family_source_count >= 2:
            confidence += 0.015
        confidence = min(0.99, confidence)
        if match_kind in {
            "LOW_CONFIDENCE_NEIGHBOR",
            "LOW_CONFIDENCE_FAMILY_WILDCARD",
        }:
            confidence = min(MAXIMUM_NEIGHBOR_CONFIDENCE, confidence)
        candidates.append(
            {
                "group_id": group_id,
                "canonical_base_color": group["base_color"],
                "canonical_family": group["family_hint"],
                "observed_color": observed_color,
                "observed_family": observed_family,
                "match_kind": match_kind,
                "confidence": confidence,
                "minimum_confidence": minimum_confidence,
                "eligible": confidence >= minimum_confidence,
                "canonical_confidence": group["canonical_confidence"],
                "color_source_count": color_source_count,
                "family_source_count": family_source_count,
                "canonical_family_wildcard": family_wildcard,
            }
        )
    return sorted(
        candidates,
        key=lambda candidate: (
            -float(candidate["confidence"]),
            str(candidate["group_id"]),
        ),
    )


def _existing_group_id(
    entity: Mapping[str, Any],
    *,
    entity_label: str,
    canonical_groups: Mapping[str, Mapping[str, Any]],
) -> str | None:
    provenance = entity.get("provenance")
    if provenance is None:
        return None
    if not isinstance(provenance, Mapping):
        raise VisualGroupAnnotationError(f"{entity_label}.provenance must be an object")
    existing = provenance.get("canonical_group_id")
    raw_corroboration = provenance.get("source_visual_corroboration")
    nested_group_id = (
        raw_corroboration.get("canonical_group_id")
        if isinstance(raw_corroboration, Mapping)
        else None
    )
    if (
        nested_group_id is not None
        and (
            not isinstance(nested_group_id, str)
            or not nested_group_id
            or existing is not None
            and existing != nested_group_id
        )
    ):
        raise VisualGroupAnnotationError(
            f"{entity_label} has inconsistent source-corroboration group lineage"
        )
    if (
        provenance.get("tier") == "corroborated_source_visual_nvidia_mdl"
        and nested_group_id is not None
        and existing is None
    ):
        raise VisualGroupAnnotationError(
            f"{entity_label} source-corroborated NVIDIA MDL lacks top-level "
            "canonical_group_id"
        )
    if existing is None:
        return None
    if not isinstance(existing, str) or not existing:
        raise VisualGroupAnnotationError(
            f"{entity_label}.provenance.canonical_group_id must be a non-empty string"
        )
    if existing not in canonical_groups:
        raise VisualGroupAnnotationError(
            f"{entity_label} references unknown canonical group {existing}"
        )
    return existing


def _annotate_entity(
    entity: dict[str, Any],
    *,
    entity_kind: str,
    entity_label: str,
    part_id: str,
    subset_name: str | None,
    canonical_groups: Mapping[str, Mapping[str, Any]],
    existing_group_id_override: str | None = None,
    write_entity_provenance: bool = True,
    token_inference_allowed: bool = True,
) -> dict[str, Any]:
    semantic = _text(
        entity.get("semantic"),
        f"{entity_label}.semantic",
        required=False,
    )
    material_id = _text(entity.get("material_id"), f"{entity_label}.material_id")
    signals = _combined_signals(semantic=semantic, material_id=material_id)
    existing_group_id = (
        _existing_group_id(
            entity,
            entity_label=entity_label,
            canonical_groups=canonical_groups,
        )
        if write_entity_provenance
        else existing_group_id_override
    )
    base_record: dict[str, Any] = {
        "entity_kind": entity_kind,
        "entity_label": entity_label,
        "part_id": part_id,
        "subset_name": subset_name,
        "semantic": semantic,
        "material_id": material_id,
        "signals": signals,
    }
    if existing_group_id is not None:
        existing = canonical_groups[existing_group_id]
        token_match = signals["normalized_colors"] == [
            existing["normalized_color"]
        ] and signals["normalized_families"] == [existing["normalized_family"]]
        return {
            **base_record,
            "outcome": "PRESERVED_EXISTING",
            "selected_group_id": existing_group_id,
            "annotation_confidence": None,
            "confidence_tier": "EXISTING_TRUSTED_PROVENANCE",
            "reason_codes": [
                "EXISTING_TRUSTED_CANONICAL_GROUP_PRESERVED",
                *(
                    []
                    if token_match
                    else ["TOKEN_INFERENCE_NOT_ALLOWED_TO_OVERRIDE_EXISTING_GROUP"]
                ),
            ],
            "candidates": [],
        }

    if not token_inference_allowed:
        return {
            **base_record,
            "outcome": "UNRESOLVED",
            "selected_group_id": None,
            "annotation_confidence": None,
            "confidence_tier": None,
            "reason_codes": [
                "POLICY_FALLBACK_MATERIAL_TOKENS_ARE_NOT_REFERENCE_EVIDENCE"
            ],
            "candidates": [],
        }

    if signals["reason_codes"]:
        outcome = (
            "AMBIGUOUS"
            if any(code.startswith("CONFLICTING_") for code in signals["reason_codes"])
            else "UNRESOLVED"
        )
        return {
            **base_record,
            "outcome": outcome,
            "selected_group_id": None,
            "annotation_confidence": None,
            "confidence_tier": None,
            "reason_codes": list(signals["reason_codes"]),
            "candidates": [],
        }

    candidates = _candidate_matches(signals, canonical_groups)
    eligible = [candidate for candidate in candidates if candidate["eligible"]]
    if not eligible:
        return {
            **base_record,
            "outcome": "UNRESOLVED",
            "selected_group_id": None,
            "annotation_confidence": None,
            "confidence_tier": None,
            "reason_codes": ["NO_CANDIDATE_CLEARED_CONFIDENCE_GATE"],
            "candidates": candidates,
        }
    winner = eligible[0]
    runner_up = eligible[1] if len(eligible) > 1 else None
    margin = (
        float(winner["confidence"]) - float(runner_up["confidence"])
        if runner_up is not None
        else None
    )
    if (
        runner_up is not None
        and margin is not None
        and margin < MINIMUM_SELECTION_MARGIN
    ):
        return {
            **base_record,
            "outcome": "AMBIGUOUS",
            "selected_group_id": None,
            "annotation_confidence": None,
            "confidence_tier": None,
            "selection_margin": margin,
            "reason_codes": ["CANONICAL_GROUP_CANDIDATES_AMBIGUOUS"],
            "candidates": candidates,
        }

    if write_entity_provenance:
        provenance = entity.get("provenance")
        mutable_provenance = dict(provenance) if isinstance(provenance, Mapping) else {}
        mutable_provenance["canonical_group_id"] = winner["group_id"]
        mutable_provenance["canonical_group_annotation"] = {
            "method": ANNOTATION_METHOD,
            "confidence": winner["confidence"],
            "confidence_tier": winner["match_kind"],
            "observed_color": winner["observed_color"],
            "observed_family": winner["observed_family"],
            "canonical_base_color": winner["canonical_base_color"],
            "canonical_family": winner["canonical_family"],
            "selection_margin": margin,
            "material_identity_unchanged": True,
            "parameters_unchanged": True,
        }
        entity["provenance"] = mutable_provenance
    return {
        **base_record,
        "outcome": "ANNOTATED",
        "selected_group_id": winner["group_id"],
        "annotation_confidence": winner["confidence"],
        "confidence_tier": winner["match_kind"],
        "selection_margin": margin,
        "reason_codes": [
            {
                "LOW_CONFIDENCE_NEIGHBOR": ("LOW_CONFIDENCE_COLOR_NEIGHBOR_ACCEPTED"),
                "LOW_CONFIDENCE_FAMILY_WILDCARD": (
                    "LOW_CONFIDENCE_CANONICAL_FAMILY_WILDCARD_ACCEPTED"
                ),
                "EXACT": "EXACT_COLOR_AND_FAMILY_TOKEN_MATCH",
            }[str(winner["match_kind"])]
        ],
        "candidates": candidates,
    }


def _assignment_face_subset_group_ids(
    assignment: Mapping[str, Any],
    *,
    assignment_label: str,
    canonical_groups: Mapping[str, Mapping[str, Any]],
) -> dict[str, str]:
    provenance = assignment.get("provenance")
    if provenance is None:
        return {}
    if not isinstance(provenance, Mapping):
        raise VisualGroupAnnotationError(
            f"{assignment_label}.provenance must be an object"
        )
    raw = provenance.get("face_subset_canonical_group_ids")
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise VisualGroupAnnotationError(
            f"{assignment_label}.provenance.face_subset_canonical_group_ids "
            "must be an object"
        )
    output: dict[str, str] = {}
    for subset_name, group_id in raw.items():
        if not isinstance(subset_name, str) or not subset_name:
            raise VisualGroupAnnotationError(
                f"{assignment_label} has an invalid face-subset group-map key"
            )
        if (
            not isinstance(group_id, str)
            or not group_id
            or group_id not in canonical_groups
        ):
            raise VisualGroupAnnotationError(
                f"{assignment_label} face subset {subset_name!r} references "
                f"unknown canonical group {group_id!r}"
            )
        output[subset_name] = group_id
    return output


def _migrate_generated_face_subset_provenance(
    subset: dict[str, Any],
    *,
    subset_label: str,
    canonical_groups: Mapping[str, Mapping[str, Any]],
) -> str | None:
    """Remove only this module's legacy, apply-incompatible subset metadata."""

    raw = subset.get("provenance")
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise VisualGroupAnnotationError(f"{subset_label}.provenance must be an object")
    unexpected = set(raw) - {
        "canonical_group_id",
        "canonical_group_annotation",
    }
    annotation = raw.get("canonical_group_annotation")
    generated_by_this_module = (
        isinstance(annotation, Mapping)
        and annotation.get("method") == ANNOTATION_METHOD
    )
    if unexpected or not generated_by_this_module:
        raise VisualGroupAnnotationError(
            f"{subset_label}.provenance is not a migratable generated "
            "visual-group annotation"
        )
    group_id = raw.get("canonical_group_id")
    if (
        not isinstance(group_id, str)
        or not group_id
        or group_id not in canonical_groups
    ):
        raise VisualGroupAnnotationError(
            f"{subset_label}.provenance references unknown canonical group {group_id!r}"
        )
    # ``normalize_face_subsets`` intentionally rejects provenance and every
    # other non-authoring field.  The group ID is migrated to the assignment
    # provenance map before this key is removed.
    subset.pop("provenance")
    return group_id


def _finite_ratio(value: Any) -> float | None:
    """Return one finite unit-interval number without trusting booleans."""

    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        return None
    return float(value)


def _spatial_visual_group_aliases(
    *,
    canonical_groups: Mapping[str, Mapping[str, Any]],
    minimum_supporting_views: int,
) -> dict[str, str]:
    """Join one-view colour hypotheses to one unique multiview authority.

    Palette fusion deliberately keeps a model-described singleton separate
    from independently recovered pixel evidence because they remain different
    material hypotheses.  That distinction must not turn identical visual
    colours into a false 2D/3D localization conflict, however.  When exactly
    one eligible canonical group of a colour is supported by multiple source
    photographs, spatial localization may use that group as the representative
    for same-colour singleton observations.  Material selection still sees the
    original palette hypotheses and remains immutable.
    """

    groups_by_color: dict[str, list[tuple[str, Mapping[str, Any]]]] = {}
    for group_id, group in canonical_groups.items():
        normalized_color = group.get("normalized_color")
        if not isinstance(normalized_color, str) or not normalized_color:
            continue
        groups_by_color.setdefault(normalized_color, []).append((group_id, group))

    aliases: dict[str, str] = {}
    for groups in groups_by_color.values():
        authorities = [
            group_id
            for group_id, group in groups
            if group.get("eligible") is True
            and isinstance(group.get("source_view_ids"), list)
            and len(group["source_view_ids"]) >= minimum_supporting_views
        ]
        if len(authorities) != 1:
            continue
        authority = authorities[0]
        for group_id, group in groups:
            if (
                group_id != authority
                and group.get("eligible") is True
                and isinstance(group.get("source_view_ids"), list)
                and len(group["source_view_ids"]) < minimum_supporting_views
            ):
                aliases[group_id] = authority
    return dict(sorted(aliases.items()))


def _alias_group_score_rows(
    rows: Any,
    *,
    aliases: Mapping[str, str],
) -> Any:
    """Return score rows with same-colour aliases pooled exactly once."""

    if not isinstance(rows, list):
        return copy.deepcopy(rows)
    merged: dict[str, dict[str, Any]] = {}
    passthrough: list[Any] = []
    order: list[str] = []
    for raw_row in rows:
        if not isinstance(raw_row, Mapping):
            passthrough.append(copy.deepcopy(raw_row))
            continue
        raw_group_id = raw_row.get("canonical_group_id")
        if not isinstance(raw_group_id, str):
            passthrough.append(copy.deepcopy(dict(raw_row)))
            continue
        group_id = aliases.get(raw_group_id, raw_group_id)
        row = copy.deepcopy(dict(raw_row))
        row["canonical_group_id"] = group_id
        existing = merged.get(group_id)
        if existing is None:
            merged[group_id] = row
            order.append(group_id)
            continue
        current_share = _finite_ratio(existing.get("color_share"))
        added_share = _finite_ratio(row.get("color_share"))
        if current_share is not None and added_share is not None:
            existing["color_share"] = min(1.0, current_share + added_share)
    return [*(merged[group_id] for group_id in order), *passthrough]


def _alias_spatial_value(value: Any, *, aliases: Mapping[str, str]) -> Any:
    """Normalize canonical-group IDs in one copied spatial-evidence value."""

    if isinstance(value, list):
        return [
            _alias_spatial_value(item, aliases=aliases)
            for item in value
        ]
    if not isinstance(value, Mapping):
        return copy.deepcopy(value)
    output: dict[str, Any] = {}
    for key, raw_value in value.items():
        if (
            key in {
                "canonical_group_id",
                "bbox_canonical_group_id",
                "diagnostic_canonical_group_id",
                "refined_canonical_group_id",
            }
            and isinstance(raw_value, str)
        ):
            output[key] = aliases.get(raw_value, raw_value)
        elif key in {
            "alternative_canonical_group_ids",
            "semantic_conflicting_group_ids",
        } and isinstance(raw_value, list):
            output[key] = sorted(
                {
                    aliases.get(group_id, group_id)
                    for group_id in raw_value
                    if isinstance(group_id, str)
                }
            )
        elif key in {"group_scores", "bbox_group_scores"}:
            output[key] = _alias_group_score_rows(raw_value, aliases=aliases)
        elif key == "resolved_support_counts" and isinstance(raw_value, Mapping):
            counts: dict[str, int] = {}
            for raw_group_id, raw_count in raw_value.items():
                group_id = aliases.get(str(raw_group_id), str(raw_group_id))
                if isinstance(raw_count, int) and not isinstance(raw_count, bool):
                    counts[group_id] = counts.get(group_id, 0) + raw_count
            output[key] = counts
        else:
            output[key] = _alias_spatial_value(raw_value, aliases=aliases)

    for score_key, margin_key in (
        ("group_scores", "color_margin"),
        ("bbox_group_scores", "bbox_color_margin"),
    ):
        scores = output.get(score_key)
        if not isinstance(scores, list):
            continue
        shares = sorted(
            (
                float(row["color_share"])
                for row in scores
                if isinstance(row, Mapping)
                and _finite_ratio(row.get("color_share")) is not None
            ),
            reverse=True,
        )
        if shares:
            output[margin_key] = shares[0] - (shares[1] if len(shares) > 1 else 0.0)
    return output


def _trusted_foreground_projection(
    observation: Mapping[str, Any],
) -> tuple[int | None, float | None]:
    """Return SAM3-foreground support for one projected part mask.

    Raw reference RGB alone cannot distinguish a genuinely black component
    from the neutral black background used outside the user-confirmed SAM3
    mask.  Spatial annotation therefore consumes the foreground-intersection
    audit already sealed into ``canonical_palette_diagnostic`` and refuses to
    turn background pixels into material lineage.
    """

    diagnostic = observation.get("canonical_palette_diagnostic")
    if not isinstance(diagnostic, Mapping):
        return None, None
    direct_sample = diagnostic.get("direct_sample")
    if not isinstance(direct_sample, Mapping):
        return None, None
    foreground_pixels = direct_sample.get("sampled_foreground_pixels")
    foreground_overlap = _finite_ratio(
        direct_sample.get("foreground_overlap_ratio")
    )
    if (
        isinstance(foreground_pixels, bool)
        or not isinstance(foreground_pixels, int)
        or foreground_pixels < 0
    ):
        foreground_pixels = None
    return foreground_pixels, foreground_overlap


def _high_confidence_single_view_projection(
    *,
    observations: Sequence[Any],
    canonical_groups: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Validate one large, uniquely stable spatial projection.

    This is deliberately not a generic single-view fallback.  It is available
    only when exactly one resolved observation is registration- and
    perturbation-stable, no independently stable conflict exists, and the
    projected pixels, colour purity, bounding-box vote, and every fixed
    perturbation all independently identify the same canonical group.
    """

    stable_resolved = [
        observation
        for observation in observations
        if isinstance(observation, Mapping)
        and observation.get("classification") == "resolved"
        and observation.get("registration_label_stable") is True
        and observation.get("perturbation_label_stable") is True
        and isinstance(observation.get("canonical_group_id"), str)
        and observation.get("canonical_group_id") in canonical_groups
        and isinstance(observation.get("reference_view_id"), str)
        and bool(observation.get("reference_view_id"))
    ]
    if len(stable_resolved) != 1:
        return None, None

    observation = stable_resolved[0]
    group_id = str(observation["canonical_group_id"])
    view_id = str(observation["reference_view_id"])
    reason_codes: list[str] = []

    stable_conflicts = [
        candidate
        for candidate in observations
        if isinstance(candidate, Mapping)
        and candidate.get("classification") == "conflict"
        and candidate.get("registration_label_stable") is True
        and candidate.get("perturbation_label_stable") is True
        and candidate.get("canonical_group_id") != group_id
    ]
    if stable_conflicts:
        reason_codes.append("SINGLE_VIEW_STABLE_SPATIAL_CONFLICT_PRESENT")

    projected_pixels = observation.get("projected_part_pixels")
    sampled_pixels = observation.get("sampled_reference_pixels")
    if (
        isinstance(projected_pixels, bool)
        or not isinstance(projected_pixels, int)
        or projected_pixels < MINIMUM_SINGLE_VIEW_SPATIAL_PIXELS
    ):
        reason_codes.append("SINGLE_VIEW_PROJECTED_PIXELS_BELOW_MINIMUM")
    if (
        isinstance(sampled_pixels, bool)
        or not isinstance(sampled_pixels, int)
        or sampled_pixels < MINIMUM_SINGLE_VIEW_SPATIAL_PIXELS
    ):
        reason_codes.append("SINGLE_VIEW_SAMPLED_PIXELS_BELOW_MINIMUM")
    foreground_pixels, foreground_overlap = _trusted_foreground_projection(
        observation
    )
    if (
        foreground_pixels is None
        or foreground_pixels < MINIMUM_SPATIAL_FOREGROUND_PIXELS
    ):
        reason_codes.append("SINGLE_VIEW_FOREGROUND_PIXELS_BELOW_MINIMUM")
    if (
        foreground_overlap is None
        or foreground_overlap < MINIMUM_SPATIAL_FOREGROUND_OVERLAP
    ):
        reason_codes.append("SINGLE_VIEW_FOREGROUND_OVERLAP_BELOW_MINIMUM")

    raw_group_scores = observation.get("group_scores")
    valid_group_scores: list[tuple[float, str]] = []
    if isinstance(raw_group_scores, list):
        for raw_score in raw_group_scores:
            if not isinstance(raw_score, Mapping):
                continue
            score_group_id = raw_score.get("canonical_group_id")
            color_share = _finite_ratio(raw_score.get("color_share"))
            if (
                isinstance(score_group_id, str)
                and score_group_id in canonical_groups
                and color_share is not None
            ):
                valid_group_scores.append((color_share, score_group_id))
    winning_color_share: float | None = None
    winning_group_id: str | None = None
    if valid_group_scores:
        winning_color_share, winning_group_id = max(
            valid_group_scores,
            key=lambda item: (item[0], item[1]),
        )
    if winning_group_id != group_id:
        reason_codes.append("SINGLE_VIEW_WINNING_COLOR_GROUP_MISMATCH")
    if (
        winning_color_share is None
        or winning_color_share < MINIMUM_SINGLE_VIEW_COLOR_SHARE
    ):
        reason_codes.append("SINGLE_VIEW_WINNING_COLOR_SHARE_BELOW_MINIMUM")

    color_margin = _finite_ratio(observation.get("color_margin"))
    if (
        color_margin is None
        or color_margin < MINIMUM_SINGLE_VIEW_COLOR_MARGIN
    ):
        reason_codes.append("SINGLE_VIEW_COLOR_MARGIN_BELOW_MINIMUM")

    bbox_group_id = observation.get("bbox_canonical_group_id")
    if bbox_group_id != group_id:
        reason_codes.append("SINGLE_VIEW_BBOX_CANONICAL_GROUP_MISMATCH")
    bbox_color_margin = _finite_ratio(observation.get("bbox_color_margin"))
    if (
        bbox_color_margin is None
        or bbox_color_margin < MINIMUM_SINGLE_VIEW_BBOX_COLOR_MARGIN
    ):
        reason_codes.append("SINGLE_VIEW_BBOX_COLOR_MARGIN_BELOW_MINIMUM")

    raw_perturbations = observation.get("projection_perturbations")
    perturbation_count = 0
    perturbation_group_ids: list[str | None] = []
    perturbation_diagnostic_group_ids: list[str | None] = []
    if isinstance(raw_perturbations, list):
        perturbation_count = len(raw_perturbations)
        for perturbation in raw_perturbations:
            if not isinstance(perturbation, Mapping):
                perturbation_group_ids.append(None)
                perturbation_diagnostic_group_ids.append(None)
                continue
            raw_group_id = perturbation.get("canonical_group_id")
            perturbation_group_ids.append(
                str(raw_group_id) if isinstance(raw_group_id, str) else None
            )
            raw_diagnostic_group_id = perturbation.get(
                "diagnostic_canonical_group_id"
            )
            perturbation_diagnostic_group_ids.append(
                str(raw_diagnostic_group_id)
                if isinstance(raw_diagnostic_group_id, str)
                else None
            )
    if perturbation_count == 0:
        reason_codes.append("SINGLE_VIEW_PROJECTION_PERTURBATIONS_MISSING")
    elif any(
        perturbation_group_id != group_id
        for perturbation_group_id in perturbation_group_ids
    ) or any(
        diagnostic_group_id not in {None, group_id}
        for diagnostic_group_id in perturbation_diagnostic_group_ids
    ):
        reason_codes.append("SINGLE_VIEW_PERTURBATION_GROUP_CONFLICT")

    diagnostics = {
        "reference_view_id": view_id,
        "canonical_group_id": group_id,
        "projected_part_pixels": projected_pixels,
        "sampled_reference_pixels": sampled_pixels,
        "sampled_foreground_pixels": foreground_pixels,
        "foreground_overlap_ratio": foreground_overlap,
        "minimum_foreground_pixels": MINIMUM_SPATIAL_FOREGROUND_PIXELS,
        "minimum_foreground_overlap": MINIMUM_SPATIAL_FOREGROUND_OVERLAP,
        "winning_color_share": winning_color_share,
        "color_margin": color_margin,
        "bbox_canonical_group_id": bbox_group_id,
        "bbox_color_margin": bbox_color_margin,
        "perturbation_count": perturbation_count,
        "perturbation_group_ids": perturbation_group_ids,
        "perturbation_diagnostic_group_ids": (
            perturbation_diagnostic_group_ids
        ),
        "stable_conflict_view_ids": sorted(
            str(conflict.get("reference_view_id"))
            for conflict in stable_conflicts
            if isinstance(conflict.get("reference_view_id"), str)
            and conflict.get("reference_view_id")
        ),
        "reason_codes": reason_codes,
    }
    if reason_codes:
        return None, diagnostics

    assert winning_color_share is not None
    assert color_margin is not None
    assert bbox_color_margin is not None
    return {
        **diagnostics,
        "effective_confidence": min(
            winning_color_share,
            color_margin,
            bbox_color_margin,
        ),
    }, None


def _corroborated_multiview_direct_projection(
    *,
    observations: Sequence[Any],
    semantic_votes: Any,
    canonical_groups: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Recover a thin-part seed from sealed direct-mask evidence.

    Sparse or elongated parts can have a reliable projected mask while their
    axis-aligned bounding box contains mostly neighbouring geometry.  The
    normal spatial classifier correctly rejects that single view.  This lane
    accepts only a stronger cross-view contract: three independent direct
    winners must agree, two must clear fixed pixel/share/margin floors, and at
    least one strong view must also be registration- and perturbation-stable.
    Strong direct, semantic, or stable-resolved conflicts veto the result.

    Bounding-box disagreement is retained as diagnostics, not used as a veto:
    it is precisely the failure mode this multiview direct-mask lane is meant
    to handle.  The caller changes lineage only; material identity and
    parameters remain untouched for the rendered exact-MDL tournament.
    """

    direct_votes: list[dict[str, Any]] = []
    seen_view_ids: set[str] = set()
    duplicate_view_ids: set[str] = set()
    accepted_evidence_modes = {
        "source_projection",
        "isolated_mask_multiview_diagnostic",
    }
    for observation in observations:
        if (
            not isinstance(observation, Mapping)
            or observation.get("classification")
            not in {"resolved", "conflict", "insufficient_visibility"}
            or observation.get("evidence_mode") not in accepted_evidence_modes
        ):
            continue
        view_id = observation.get("reference_view_id")
        if not isinstance(view_id, str) or not view_id:
            continue
        if view_id in seen_view_ids:
            duplicate_view_ids.add(view_id)
            continue
        seen_view_ids.add(view_id)

        ranked_scores: list[tuple[float, str]] = []
        raw_scores = observation.get("group_scores")
        if isinstance(raw_scores, list):
            for raw_score in raw_scores:
                if not isinstance(raw_score, Mapping):
                    continue
                group_id = raw_score.get("canonical_group_id")
                color_share = _finite_ratio(raw_score.get("color_share"))
                if (
                    isinstance(group_id, str)
                    and group_id in canonical_groups
                    and color_share is not None
                ):
                    ranked_scores.append((color_share, group_id))
        if not ranked_scores:
            continue
        ranked_scores.sort(key=lambda item: (-item[0], item[1]))
        winning_share, winning_group_id = ranked_scores[0]
        runner_up_share = ranked_scores[1][0] if len(ranked_scores) > 1 else 0.0
        if winning_share <= 0.0 or math.isclose(
            winning_share,
            runner_up_share,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            continue

        projected_pixels = observation.get("projected_part_pixels")
        sampled_pixels = observation.get("sampled_reference_pixels")
        foreground_pixels, foreground_overlap = _trusted_foreground_projection(
            observation
        )
        color_margin = _finite_ratio(observation.get("color_margin"))
        foreground_aligned = bool(
            foreground_overlap is not None
            and foreground_overlap >= MINIMUM_SPATIAL_FOREGROUND_OVERLAP
        )
        foreground_trusted = bool(
            foreground_aligned
            and foreground_pixels is not None
            and foreground_pixels >= MINIMUM_SPATIAL_FOREGROUND_PIXELS
        )
        strong = (
            not isinstance(projected_pixels, bool)
            and isinstance(projected_pixels, int)
            and projected_pixels >= MINIMUM_DIRECT_MULTIVIEW_PIXELS
            and not isinstance(sampled_pixels, bool)
            and isinstance(sampled_pixels, int)
            and sampled_pixels >= MINIMUM_DIRECT_MULTIVIEW_PIXELS
            and winning_share >= MINIMUM_DIRECT_MULTIVIEW_COLOR_SHARE
            and color_margin is not None
            and color_margin >= MINIMUM_DIRECT_MULTIVIEW_COLOR_MARGIN
            and foreground_trusted
        )
        bbox_group_id = observation.get("bbox_canonical_group_id")
        direct_votes.append(
            {
                "reference_view_id": view_id,
                "canonical_group_id": winning_group_id,
                "winning_color_share": winning_share,
                "runner_up_color_share": runner_up_share,
                "color_margin": color_margin,
                "projected_part_pixels": projected_pixels,
                "sampled_reference_pixels": sampled_pixels,
                "sampled_foreground_pixels": foreground_pixels,
                "foreground_overlap_ratio": foreground_overlap,
                "foreground_aligned": foreground_aligned,
                "foreground_trusted": foreground_trusted,
                "strong": strong,
                "registration_and_perturbation_stable": (
                    observation.get("registration_label_stable") is True
                    and observation.get("perturbation_label_stable") is True
                ),
                "bbox_canonical_group_id": (
                    str(bbox_group_id)
                    if isinstance(bbox_group_id, str)
                    else None
                ),
            }
        )

    agreement_counts: dict[str, int] = {}
    for vote in direct_votes:
        if vote["foreground_aligned"] is not True:
            continue
        group_id = str(vote["canonical_group_id"])
        agreement_counts[group_id] = agreement_counts.get(group_id, 0) + 1
    eligible_group_ids = sorted(
        group_id
        for group_id, count in agreement_counts.items()
        if count >= MINIMUM_DIRECT_MULTIVIEW_AGREEMENT_VIEWS
    )
    if len(eligible_group_ids) != 1:
        return None, None

    group_id = eligible_group_ids[0]
    agreement_votes = [
        vote
        for vote in direct_votes
        if vote["canonical_group_id"] == group_id
        and vote["foreground_aligned"] is True
    ]
    strong_votes = [vote for vote in agreement_votes if vote["strong"] is True]
    strong_direct_conflicts = [
        vote
        for vote in direct_votes
        if vote["canonical_group_id"] != group_id and vote["strong"] is True
    ]
    stable_resolved_conflicts = [
        observation
        for observation in observations
        if (
            isinstance(observation, Mapping)
            and observation.get("classification") == "resolved"
            and observation.get("registration_label_stable") is True
            and observation.get("perturbation_label_stable") is True
            and isinstance(observation.get("canonical_group_id"), str)
            and observation["canonical_group_id"] in canonical_groups
            and observation["canonical_group_id"] != group_id
            and isinstance(observation.get("reference_view_id"), str)
        )
    ]
    strong_semantic_conflicts: list[Mapping[str, Any]] = []
    if isinstance(semantic_votes, list):
        for vote in semantic_votes:
            if not isinstance(vote, Mapping):
                continue
            vote_group_id = vote.get("canonical_group_id")
            confidence = _finite_ratio(vote.get("effective_confidence"))
            if (
                isinstance(vote_group_id, str)
                and vote_group_id in canonical_groups
                and vote_group_id != group_id
                and confidence is not None
                and confidence
                >= MINIMUM_DIRECT_MULTIVIEW_SEMANTIC_CONFLICT_CONFIDENCE
                and vote.get("pixel_gate_accepted") is True
                and vote.get("unique_canonical_join") is True
            ):
                strong_semantic_conflicts.append(vote)

    reason_codes: list[str] = []
    if duplicate_view_ids:
        reason_codes.append("DIRECT_MULTIVIEW_DUPLICATE_REFERENCE_VIEW")
    if len(strong_votes) < MINIMUM_DIRECT_MULTIVIEW_STRONG_VIEWS:
        reason_codes.append("DIRECT_MULTIVIEW_STRONG_SUPPORT_BELOW_MINIMUM")
    if not any(
        vote["registration_and_perturbation_stable"] is True
        for vote in strong_votes
    ):
        reason_codes.append("DIRECT_MULTIVIEW_LACKS_STABLE_STRONG_SUPPORT")
    if strong_direct_conflicts:
        reason_codes.append("DIRECT_MULTIVIEW_STRONG_GROUP_CONFLICT")
    if stable_resolved_conflicts:
        reason_codes.append("DIRECT_MULTIVIEW_STABLE_RESOLVED_GROUP_CONFLICT")
    if strong_semantic_conflicts:
        unanimous_stable_direct = bool(
            len(agreement_votes) >= MINIMUM_DIRECT_MULTIVIEW_AGREEMENT_VIEWS
            and len(strong_votes) == len(agreement_votes)
            and all(
                vote["registration_and_perturbation_stable"] is True
                and vote["bbox_canonical_group_id"] in {None, group_id}
                for vote in strong_votes
            )
        )
        if not unanimous_stable_direct:
            reason_codes.append(
                "DIRECT_MULTIVIEW_STRONG_SEMANTIC_GROUP_CONFLICT"
            )

    bbox_conflict_view_ids = sorted(
        str(vote["reference_view_id"])
        for vote in agreement_votes
        if vote["bbox_canonical_group_id"] not in {None, group_id}
    )
    diagnostics = {
        "canonical_group_id": group_id,
        "agreement_view_ids": sorted(
            str(vote["reference_view_id"]) for vote in agreement_votes
        ),
        "agreement_view_count": len(agreement_votes),
        "strong_supporting_view_ids": sorted(
            str(vote["reference_view_id"]) for vote in strong_votes
        ),
        "strong_supporting_view_count": len(strong_votes),
        "stable_strong_supporting_view_ids": sorted(
            str(vote["reference_view_id"])
            for vote in strong_votes
            if vote["registration_and_perturbation_stable"] is True
        ),
        "bbox_conflict_view_ids": bbox_conflict_view_ids,
        "bbox_conflict_is_diagnostic_only": True,
        "strong_direct_conflict_view_ids": sorted(
            str(vote["reference_view_id"]) for vote in strong_direct_conflicts
        ),
        "stable_resolved_conflict_view_ids": sorted(
            str(observation["reference_view_id"])
            for observation in stable_resolved_conflicts
        ),
        "strong_semantic_conflict_view_ids": sorted(
            str(vote["view_id"])
            for vote in strong_semantic_conflicts
            if isinstance(vote.get("view_id"), str)
        ),
        "semantic_conflict_overridden_by_unanimous_stable_direct": bool(
            strong_semantic_conflicts
            and "DIRECT_MULTIVIEW_STRONG_SEMANTIC_GROUP_CONFLICT"
            not in reason_codes
        ),
        "duplicate_reference_view_ids": sorted(duplicate_view_ids),
        "direct_votes": copy.deepcopy(direct_votes),
        "minimum_foreground_pixels": MINIMUM_SPATIAL_FOREGROUND_PIXELS,
        "minimum_foreground_overlap": MINIMUM_SPATIAL_FOREGROUND_OVERLAP,
        "reason_codes": reason_codes,
    }
    if reason_codes:
        return None, diagnostics

    return {
        **diagnostics,
        "effective_confidence": min(
            float(vote["winning_color_share"]) for vote in strong_votes
        ),
    }, None


def _sam3_foreground_single_view_projection(
    *,
    observations: Sequence[Any],
    canonical_groups: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Select one part group from any independently usable reference view.

    A part is not required to appear in multiple photographs.  Every view is
    evaluated independently against the user-confirmed SAM3 foreground; views
    where the part is absent simply cast no vote.  One sufficiently pure,
    perturbation-stable foreground vote is enough.  If two usable views vote
    for different groups, the lane abstains instead of choosing by count.
    """

    votes: list[dict[str, Any]] = []
    for observation in observations:
        if not isinstance(observation, Mapping):
            continue
        view_id = observation.get("reference_view_id")
        if not isinstance(view_id, str) or not view_id:
            continue
        diagnostic = observation.get("canonical_palette_diagnostic")
        direct_sample = (
            diagnostic.get("direct_sample")
            if isinstance(diagnostic, Mapping)
            else None
        )
        if not isinstance(direct_sample, Mapping):
            continue
        foreground_pixels = direct_sample.get("sampled_foreground_pixels")
        foreground_overlap = _finite_ratio(
            direct_sample.get("foreground_overlap_ratio")
        )
        if (
            isinstance(foreground_pixels, bool)
            or not isinstance(foreground_pixels, int)
            or foreground_pixels < MINIMUM_FOREGROUND_SINGLE_VIEW_PIXELS
            or foreground_overlap is None
            or foreground_overlap < MINIMUM_SPATIAL_FOREGROUND_OVERLAP
            or observation.get("perturbation_label_stable") is not True
        ):
            continue

        ranked_scores: list[tuple[float, str]] = []
        raw_scores = direct_sample.get("group_scores")
        if isinstance(raw_scores, list):
            for raw_score in raw_scores:
                if not isinstance(raw_score, Mapping):
                    continue
                group_id = raw_score.get("canonical_group_id")
                color_share = _finite_ratio(raw_score.get("color_share"))
                if (
                    isinstance(group_id, str)
                    and group_id in canonical_groups
                    and color_share is not None
                ):
                    ranked_scores.append((color_share, group_id))
        if not ranked_scores:
            continue
        ranked_scores.sort(key=lambda item: (-item[0], item[1]))
        winning_share, group_id = ranked_scores[0]
        runner_up_share = (
            ranked_scores[1][0] if len(ranked_scores) > 1 else 0.0
        )
        color_margin = winning_share - runner_up_share
        if (
            winning_share < MINIMUM_FOREGROUND_SINGLE_VIEW_COLOR_SHARE
            or color_margin < MINIMUM_FOREGROUND_SINGLE_VIEW_COLOR_MARGIN
        ):
            continue
        votes.append(
            {
                "reference_view_id": view_id,
                "canonical_group_id": group_id,
                "sampled_foreground_pixels": foreground_pixels,
                "foreground_overlap_ratio": foreground_overlap,
                "winning_color_share": winning_share,
                "runner_up_color_share": runner_up_share,
                "color_margin": color_margin,
                "registration_label_stable": (
                    observation.get("registration_label_stable") is True
                ),
                "perturbation_label_stable": True,
            }
        )

    group_ids = sorted(
        {str(vote["canonical_group_id"]) for vote in votes}
    )
    if not votes:
        return None, None
    if len(group_ids) != 1:
        return None, {
            "reason_codes": [
                "SAM3_FOREGROUND_SINGLE_VIEW_CANONICAL_GROUP_CONFLICT"
            ],
            "candidate_group_ids": group_ids,
            "votes": copy.deepcopy(votes),
        }

    selected = max(
        votes,
        key=lambda vote: (
            float(vote["winning_color_share"]),
            int(vote["sampled_foreground_pixels"]),
            str(vote["reference_view_id"]),
        ),
    )
    return {
        "canonical_group_id": group_ids[0],
        "reference_view_id": str(selected["reference_view_id"]),
        "supporting_view_ids": sorted(
            str(vote["reference_view_id"]) for vote in votes
        ),
        "votes": copy.deepcopy(votes),
        "effective_confidence": min(
            float(selected["foreground_overlap_ratio"]),
            float(selected["winning_color_share"]),
        ),
        "minimum_foreground_pixels": (
            MINIMUM_FOREGROUND_SINGLE_VIEW_PIXELS
        ),
        "minimum_foreground_overlap": MINIMUM_SPATIAL_FOREGROUND_OVERLAP,
        "minimum_winning_color_share": (
            MINIMUM_FOREGROUND_SINGLE_VIEW_COLOR_SHARE
        ),
        "minimum_color_margin": (
            MINIMUM_FOREGROUND_SINGLE_VIEW_COLOR_MARGIN
        ),
    }, None


def _apply_trusted_spatial_annotations(
    *,
    assignments: Sequence[Any],
    records: list[dict[str, Any]],
    canonical_groups: Mapping[str, Mapping[str, Any]],
    spatial_mapping_report: Mapping[str, Any],
) -> dict[str, Any]:
    """Annotate unresolved whole-part entities from sealed multiview evidence.

    This lane is intentionally limited to whole parts without authored face
    subsets.  It requires one unique canonical group in at least the spatial
    report's configured number of trusted, registration-stable and
    perturbation-stable reference views.  One large part visible in only one
    aligned view gets a separate strict lane: colour purity, main and bounding
    box margins, and every fixed perturbation must all independently agree.
    Thin parts get one bounded recovery lane when fixed pixel perturbations are
    larger than the projected feature: two independent, pixel-gated semantic
    projections must agree, at least one spatial color projection must
    corroborate that same group, and no stable projection may disagree.  All
    lanes change lineage only; the current material and all parameters remain
    byte-for-byte unchanged so the exact MDL render tournament can evaluate
    candidate identities on a real target.
    """

    if spatial_mapping_report.get("schema_version") != (
        "qwen-spatial-mapping-audit/v1"
    ):
        raise VisualGroupAnnotationError(
            "spatial mapping report has an unsupported schema_version"
        )
    integrity = spatial_mapping_report.get("integrity")
    if not isinstance(integrity, Mapping) or not isinstance(
        integrity.get("report_sha256"), str
    ):
        raise VisualGroupAnnotationError("spatial mapping report lacks integrity")
    unsigned_report = copy.deepcopy(dict(spatial_mapping_report))
    unsigned_report.pop("integrity", None)
    if _canonical_sha256(unsigned_report) != integrity["report_sha256"]:
        raise VisualGroupAnnotationError(
            "spatial mapping report SHA256 integrity check failed"
        )
    policy = spatial_mapping_report.get("policy")
    minimum_support = (
        policy.get("minimum_spatial_support_views")
        if isinstance(policy, Mapping)
        else None
    )
    if (
        isinstance(minimum_support, bool)
        or not isinstance(minimum_support, int)
        or minimum_support < 2
    ):
        raise VisualGroupAnnotationError(
            "spatial mapping report has an invalid support-view minimum"
        )
    visual_group_aliases = _spatial_visual_group_aliases(
        canonical_groups=canonical_groups,
        minimum_supporting_views=minimum_support,
    )
    raw_parts = spatial_mapping_report.get("parts")
    if not isinstance(raw_parts, list):
        raise VisualGroupAnnotationError("spatial mapping report parts are invalid")

    assignments_by_part: dict[str, dict[str, Any]] = {}
    for assignment in assignments:
        if not isinstance(assignment, dict):
            continue
        part_id = assignment.get("part_id")
        if not isinstance(part_id, str) or not part_id:
            continue
        assignments_by_part[part_id] = assignment
    assignment_records = {
        str(record["part_id"]): record
        for record in records
        if record.get("entity_kind") == "assignment"
        and isinstance(record.get("part_id"), str)
    }

    annotated: list[dict[str, Any]] = []
    single_view_rejections: list[dict[str, Any]] = []
    foreground_single_view_rejections: list[dict[str, Any]] = []
    direct_multiview_rejections: list[dict[str, Any]] = []
    palette_bound_small_part_rejections: list[dict[str, Any]] = []
    seen_parts: set[str] = set()
    report_sha256 = str(integrity["report_sha256"])
    for raw_part in raw_parts:
        if not isinstance(raw_part, Mapping):
            raise VisualGroupAnnotationError(
                "spatial mapping report contains an invalid part"
            )
        part_id = raw_part.get("part_id")
        if (
            not isinstance(part_id, str)
            or not part_id
            or part_id in seen_parts
        ):
            raise VisualGroupAnnotationError(
                "spatial mapping report contains duplicate or invalid part IDs"
            )
        seen_parts.add(part_id)
        assignment = assignments_by_part.get(part_id)
        record = assignment_records.get(part_id)
        if assignment is None or record is None:
            continue
        if assignment.get("face_subsets"):
            continue
        provenance = assignment.get("provenance")
        if isinstance(provenance, Mapping) and provenance.get(
            "canonical_group_id"
        ) is not None:
            continue

        normalized_part = _alias_spatial_value(
            raw_part,
            aliases=visual_group_aliases,
        )
        if not isinstance(normalized_part, Mapping):
            raise VisualGroupAnnotationError(
                f"spatial mapping part {part_id} could not be normalized"
            )

        observations = normalized_part.get("observations")
        if not isinstance(observations, list):
            raise VisualGroupAnnotationError(
                f"spatial mapping part {part_id} observations are invalid"
            )
        stable_views_by_group: dict[str, set[str]] = {}
        for observation in observations:
            if (
                not isinstance(observation, Mapping)
                or observation.get("classification") != "resolved"
                or observation.get("registration_label_stable") is not True
                or observation.get("perturbation_label_stable") is not True
            ):
                continue
            group_id = observation.get("canonical_group_id")
            view_id = observation.get("reference_view_id")
            if (
                isinstance(group_id, str)
                and group_id in canonical_groups
                and isinstance(view_id, str)
                and view_id
            ):
                stable_views_by_group.setdefault(group_id, set()).add(view_id)
        eligible_groups = {
            group_id: sorted(view_ids)
            for group_id, view_ids in stable_views_by_group.items()
            if len(view_ids) >= minimum_support
        }
        single_view_projection, single_view_rejection = (
            _high_confidence_single_view_projection(
                observations=observations,
                canonical_groups=canonical_groups,
            )
        )
        if single_view_rejection is not None:
            single_view_rejections.append(
                {"part_id": part_id, **single_view_rejection}
            )
        (
            direct_multiview_projection,
            direct_multiview_rejection,
        ) = _corroborated_multiview_direct_projection(
            observations=observations,
            semantic_votes=normalized_part.get("semantic_votes"),
            canonical_groups=canonical_groups,
        )
        if direct_multiview_rejection is not None:
            direct_multiview_rejections.append(
                {"part_id": part_id, **direct_multiview_rejection}
            )
        (
            foreground_single_view_projection,
            foreground_single_view_rejection,
        ) = _sam3_foreground_single_view_projection(
            observations=observations,
            canonical_groups=canonical_groups,
        )
        if foreground_single_view_rejection is not None:
            foreground_single_view_rejections.append(
                {"part_id": part_id, **foreground_single_view_rejection}
            )
        palette_bound_candidates: dict[str, list[Mapping[str, Any]]] = {}
        raw_palette_bound_diagnostics = 0
        for observation in observations:
            if not isinstance(observation, Mapping):
                continue
            diagnostic = observation.get("small_part_diagnostic")
            if (
                not isinstance(diagnostic, Mapping)
                or diagnostic.get("status") != "resolved"
            ):
                continue
            raw_palette_bound_diagnostics += 1
            group_id = diagnostic.get("canonical_group_id")
            group = (
                canonical_groups.get(group_id)
                if isinstance(group_id, str)
                else None
            )
            source_view_ids = (
                group.get("source_view_ids")
                if isinstance(group, Mapping)
                else None
            )
            diagnostic_reasons = diagnostic.get("reason_codes")
            alternative_group_ids = diagnostic.get(
                "alternative_canonical_group_ids"
            )
            resolved_samples = diagnostic.get("resolved_sample_count")
            target_samples = diagnostic.get("target_sample_count")
            consensus_ratio = diagnostic.get("consensus_ratio")
            reference_view_id = observation.get("reference_view_id")
            if (
                group is None
                or group.get("eligible") is not True
                or not isinstance(source_view_ids, list)
                or len(source_view_ids) < minimum_support
                or not isinstance(reference_view_id, str)
                or reference_view_id not in source_view_ids
                or observation.get("reason_code")
                not in {
                    "part_visible_pixels_below_floor",
                    "projected_part_pixels_below_floor",
                }
                or diagnostic.get("registration_label_stable") is not True
                or diagnostic.get("bbox_canonical_group_id") != group_id
                or diagnostic_reasons != []
                or alternative_group_ids != []
                or isinstance(resolved_samples, bool)
                or not isinstance(resolved_samples, int)
                or isinstance(target_samples, bool)
                or not isinstance(target_samples, int)
                or target_samples < 3
                or resolved_samples != target_samples
                or isinstance(consensus_ratio, bool)
                or not isinstance(consensus_ratio, (int, float))
                or float(consensus_ratio) < 0.75
            ):
                continue
            palette_bound_candidates.setdefault(str(group_id), []).append(
                observation
            )
        palette_bound_projection: dict[str, Any] | None = None
        palette_bound_rejection: dict[str, Any] | None = None
        if raw_palette_bound_diagnostics:
            if len(palette_bound_candidates) != 1:
                palette_bound_rejection = {
                    "reason_codes": [
                        (
                            "SMALL_PART_DIAGNOSTIC_NO_MULTIVIEW_PALETTE_BOUND_CANDIDATE"
                            if not palette_bound_candidates
                            else "SMALL_PART_DIAGNOSTIC_CANONICAL_GROUP_CONFLICT"
                        )
                    ],
                    "candidate_group_ids": sorted(palette_bound_candidates),
                }
            else:
                diagnostic_group_id, diagnostic_observations = next(
                    iter(palette_bound_candidates.items())
                )
                stable_resolved_conflicts = sorted(
                    {
                        str(observation["reference_view_id"])
                        for observation in observations
                        if isinstance(observation, Mapping)
                        and observation.get("classification") == "resolved"
                        and isinstance(observation.get("canonical_group_id"), str)
                        and observation["canonical_group_id"]
                        != diagnostic_group_id
                        and isinstance(observation.get("reference_view_id"), str)
                    }
                )
                strong_semantic_conflicts = sorted(
                    {
                        str(vote["view_id"])
                        for vote in (
                            normalized_part.get("semantic_votes")
                            if isinstance(
                                normalized_part.get("semantic_votes"), list
                            )
                            else []
                        )
                        if isinstance(vote, Mapping)
                        and isinstance(vote.get("canonical_group_id"), str)
                        and vote["canonical_group_id"] != diagnostic_group_id
                        and isinstance(vote.get("effective_confidence"), (int, float))
                        and not isinstance(
                            vote.get("effective_confidence"), bool
                        )
                        and float(vote["effective_confidence"])
                        >= MINIMUM_DIRECT_MULTIVIEW_SEMANTIC_CONFLICT_CONFIDENCE
                        and isinstance(vote.get("view_id"), str)
                    }
                )
                if stable_resolved_conflicts or strong_semantic_conflicts:
                    palette_bound_rejection = {
                        "reason_codes": [
                            *(
                                [
                                    "SMALL_PART_DIAGNOSTIC_STABLE_SPATIAL_GROUP_CONFLICT"
                                ]
                                if stable_resolved_conflicts
                                else []
                            ),
                            *(
                                [
                                    "SMALL_PART_DIAGNOSTIC_STRONG_SEMANTIC_GROUP_CONFLICT"
                                ]
                                if strong_semantic_conflicts
                                else []
                            ),
                        ],
                        "candidate_group_ids": [diagnostic_group_id],
                        "stable_resolved_conflict_view_ids": (
                            stable_resolved_conflicts
                        ),
                        "strong_semantic_conflict_view_ids": (
                            strong_semantic_conflicts
                        ),
                    }
                else:
                    group = canonical_groups[diagnostic_group_id]
                    palette_bound_projection = {
                        "canonical_group_id": diagnostic_group_id,
                        "spatial_projection_view_ids": sorted(
                            {
                                str(observation["reference_view_id"])
                                for observation in diagnostic_observations
                            }
                        ),
                        "canonical_palette_source_view_ids": list(
                            group["source_view_ids"]
                        ),
                        "minimum_canonical_palette_source_view_count": (
                            minimum_support
                        ),
                        "effective_confidence": min(
                            float(group["canonical_confidence"]),
                            *(
                                float(
                                    observation["small_part_diagnostic"][
                                        "consensus_ratio"
                                    ]
                                )
                                for observation in diagnostic_observations
                            ),
                        ),
                        "diagnostics": [
                            copy.deepcopy(
                                observation["small_part_diagnostic"]
                            )
                            for observation in diagnostic_observations
                        ],
                    }
        if palette_bound_rejection is not None:
            palette_bound_small_part_rejections.append(
                {"part_id": part_id, **palette_bound_rejection}
            )
        annotation_method = SPATIAL_ANNOTATION_METHOD
        spatial_corroborating_view_ids: list[str] = []
        single_view_projection_details: dict[str, Any] | None = None
        foreground_single_view_projection_details: dict[str, Any] | None = None
        direct_multiview_projection_details: dict[str, Any] | None = None
        palette_bound_projection_details: dict[str, Any] | None = None
        effective_confidence = 1.0
        if single_view_projection is not None:
            group_id = str(single_view_projection["canonical_group_id"])
            raw_counts = normalized_part.get("resolved_support_counts")
            if (
                not isinstance(raw_counts, Mapping)
                or raw_counts.get(group_id) != 1
            ):
                raise VisualGroupAnnotationError(
                    f"spatial mapping part {part_id} single-view support "
                    "counts are inconsistent"
                )
            supporting_view_ids = [
                str(single_view_projection["reference_view_id"])
            ]
            conflicting_view_ids = []
            effective_confidence = float(
                single_view_projection["effective_confidence"]
            )
            semantic_votes = normalized_part.get("semantic_votes")
            semantic_conflicting_group_ids = sorted(
                {
                    str(vote["canonical_group_id"])
                    for vote in semantic_votes
                    if isinstance(semantic_votes, list)
                    and isinstance(vote, Mapping)
                    and isinstance(vote.get("canonical_group_id"), str)
                    and vote["canonical_group_id"] in canonical_groups
                    and vote["canonical_group_id"] != group_id
                }
            ) if isinstance(semantic_votes, list) else []
            single_view_projection_details = {
                **single_view_projection,
                "semantic_conflicting_group_ids": (
                    semantic_conflicting_group_ids
                ),
                "semantic_vote_conflict_overridden": bool(
                    semantic_conflicting_group_ids
                ),
            }
            annotation_method = SINGLE_VIEW_SPATIAL_ANNOTATION_METHOD
        elif len(eligible_groups) == 1:
            group_id, supporting_view_ids = next(iter(eligible_groups.items()))
            conflicting_view_ids = sorted(
                {
                    view_id
                    for other_group_id, view_ids in stable_views_by_group.items()
                    if other_group_id != group_id
                    for view_id in view_ids
                }
            )
            if conflicting_view_ids:
                continue
            raw_counts = normalized_part.get("resolved_support_counts")
            if (
                not isinstance(raw_counts, Mapping)
                or raw_counts.get(group_id) != len(supporting_view_ids)
            ):
                raise VisualGroupAnnotationError(
                    f"spatial mapping part {part_id} support counts are inconsistent"
                )
        elif direct_multiview_projection is not None:
            group_id = str(
                direct_multiview_projection["canonical_group_id"]
            )
            supporting_view_ids = [
                str(view_id)
                for view_id in direct_multiview_projection[
                    "strong_supporting_view_ids"
                ]
            ]
            spatial_corroborating_view_ids = [
                str(view_id)
                for view_id in direct_multiview_projection["agreement_view_ids"]
            ]
            conflicting_view_ids = []
            effective_confidence = float(
                direct_multiview_projection["effective_confidence"]
            )
            direct_multiview_projection_details = (
                direct_multiview_projection
            )
            annotation_method = (
                MULTIVIEW_DIRECT_SPATIAL_ANNOTATION_METHOD
            )
        elif foreground_single_view_projection is not None:
            group_id = str(
                foreground_single_view_projection["canonical_group_id"]
            )
            supporting_view_ids = [
                str(view_id)
                for view_id in foreground_single_view_projection[
                    "supporting_view_ids"
                ]
            ]
            conflicting_view_ids = []
            effective_confidence = float(
                foreground_single_view_projection["effective_confidence"]
            )
            foreground_single_view_projection_details = (
                foreground_single_view_projection
            )
            annotation_method = (
                FOREGROUND_SINGLE_VIEW_SPATIAL_ANNOTATION_METHOD
            )
        elif palette_bound_projection is not None:
            group_id = str(
                palette_bound_projection["canonical_group_id"]
            )
            supporting_view_ids = list(
                palette_bound_projection["spatial_projection_view_ids"]
            )
            conflicting_view_ids = []
            effective_confidence = float(
                palette_bound_projection["effective_confidence"]
            )
            palette_bound_projection_details = palette_bound_projection
            annotation_method = (
                PALETTE_BOUND_SMALL_PART_SPATIAL_ANNOTATION_METHOD
            )
        else:
            semantic_votes = normalized_part.get("semantic_votes")
            if not isinstance(semantic_votes, list):
                continue
            semantic_by_group: dict[str, list[Mapping[str, Any]]] = {}
            for vote in semantic_votes:
                if not isinstance(vote, Mapping):
                    continue
                group_id = vote.get("canonical_group_id")
                view_id = vote.get("view_id")
                reference_sha256 = vote.get("reference_sha256")
                content_cluster_id = vote.get("content_cluster_id")
                confidence = vote.get("effective_confidence")
                visible_pixels = vote.get("cad_part_visible_pixels")
                evidence_mode = vote.get("cad_part_evidence_mode")
                isolated_sha256 = vote.get("isolated_evidence_sha256")
                independently_projected = vote.get("alignment_trusted") is True or (
                    evidence_mode == "isolated_mask_multiview"
                    and isinstance(isolated_sha256, str)
                    and len(isolated_sha256) == 64
                )
                if (
                    not isinstance(group_id, str)
                    or group_id not in canonical_groups
                    or not isinstance(view_id, str)
                    or not view_id
                    or not isinstance(reference_sha256, str)
                    or len(reference_sha256) != 64
                    or not isinstance(content_cluster_id, str)
                    or not content_cluster_id
                    or isinstance(confidence, bool)
                    or not isinstance(confidence, (int, float))
                    or float(confidence) < MINIMUM_CANONICAL_CONFIDENCE
                    or isinstance(visible_pixels, bool)
                    or not isinstance(visible_pixels, int)
                    or not (
                        MINIMUM_SEMANTIC_RECOVERY_VISIBLE_PIXELS
                        <= visible_pixels
                        <= MAXIMUM_SEMANTIC_RECOVERY_VISIBLE_PIXELS
                    )
                    or vote.get("status") not in {"auto", "review"}
                    or vote.get("pixel_gate_accepted") is not True
                    or vote.get("unique_canonical_join") is not True
                    or not independently_projected
                ):
                    continue
                semantic_by_group.setdefault(group_id, []).append(vote)

            semantic_candidates: dict[str, list[Mapping[str, Any]]] = {}
            for candidate_group_id, votes in semantic_by_group.items():
                view_ids = {str(vote["view_id"]) for vote in votes}
                reference_hashes = {
                    str(vote["reference_sha256"]) for vote in votes
                }
                content_clusters = {
                    str(vote["content_cluster_id"]) for vote in votes
                }
                if min(
                    len(view_ids), len(reference_hashes), len(content_clusters)
                ) >= minimum_support:
                    semantic_candidates[candidate_group_id] = votes
            if len(semantic_candidates) != 1:
                continue
            group_id, winning_votes = next(iter(semantic_candidates.items()))
            if any(
                other_group_id != group_id and other_votes
                for other_group_id, other_votes in semantic_by_group.items()
            ):
                continue
            if any(
                other_group_id != group_id and view_ids
                for other_group_id, view_ids in stable_views_by_group.items()
            ):
                continue

            spatial_corroborating_view_ids = sorted(
                {
                    str(observation["reference_view_id"])
                    for observation in observations
                    if isinstance(observation, Mapping)
                    and observation.get("canonical_group_id") == group_id
                    and observation.get("classification")
                    in {"conflict", "insufficient_visibility", "resolved"}
                    and isinstance(observation.get("projected_part_pixels"), int)
                    and not isinstance(
                        observation.get("projected_part_pixels"), bool
                    )
                    and int(observation["projected_part_pixels"])
                    >= MINIMUM_SEMANTIC_RECOVERY_VISIBLE_PIXELS
                    and isinstance(observation.get("reference_view_id"), str)
                    and observation["reference_view_id"]
                }
            )
            if not spatial_corroborating_view_ids:
                continue
            supporting_view_ids = sorted(
                {str(vote["view_id"]) for vote in winning_votes}
            )
            effective_confidence = min(
                float(vote["effective_confidence"]) for vote in winning_votes
            )
            conflicting_view_ids = []
            annotation_method = SEMANTIC_SPATIAL_ANNOTATION_METHOD

        mutable_provenance = (
            dict(provenance) if isinstance(provenance, Mapping) else {}
        )
        spatial_annotation = {
            "method": annotation_method,
            "spatial_report_sha256": report_sha256,
            "supporting_view_ids": supporting_view_ids,
            "supporting_view_count": len(supporting_view_ids),
            "minimum_supporting_view_count": (
                1
                if annotation_method == SINGLE_VIEW_SPATIAL_ANNOTATION_METHOD
                else minimum_support
            ),
            "conflicting_view_ids": [],
            "unique_canonical_group": True,
            "whole_part_without_face_subsets": True,
            "material_identity_unchanged": True,
            "parameters_unchanged": True,
        }
        if annotation_method == SEMANTIC_SPATIAL_ANNOTATION_METHOD:
            spatial_annotation.update(
                {
                    "spatial_corroborating_view_ids": (
                        spatial_corroborating_view_ids
                    ),
                    "independent_reference_hashes_required": True,
                    "independent_content_clusters_required": True,
                    "pixel_gate_required": True,
                    "minimum_effective_confidence": (
                        MINIMUM_CANONICAL_CONFIDENCE
                    ),
                    "effective_confidence": effective_confidence,
                    "thin_part_visible_pixel_range": [
                        MINIMUM_SEMANTIC_RECOVERY_VISIBLE_PIXELS,
                        MAXIMUM_SEMANTIC_RECOVERY_VISIBLE_PIXELS,
                    ],
                }
            )
        elif annotation_method == MULTIVIEW_DIRECT_SPATIAL_ANNOTATION_METHOD:
            assert direct_multiview_projection_details is not None
            spatial_annotation.update(
                {
                    "direct_agreement_view_ids": (
                        spatial_corroborating_view_ids
                    ),
                    "minimum_direct_agreement_view_count": (
                        MINIMUM_DIRECT_MULTIVIEW_AGREEMENT_VIEWS
                    ),
                    "minimum_strong_supporting_view_count": (
                        MINIMUM_DIRECT_MULTIVIEW_STRONG_VIEWS
                    ),
                    "minimum_projected_part_pixels": (
                        MINIMUM_DIRECT_MULTIVIEW_PIXELS
                    ),
                    "minimum_sampled_reference_pixels": (
                        MINIMUM_DIRECT_MULTIVIEW_PIXELS
                    ),
                    "minimum_winning_color_share": (
                        MINIMUM_DIRECT_MULTIVIEW_COLOR_SHARE
                    ),
                    "minimum_color_margin": (
                        MINIMUM_DIRECT_MULTIVIEW_COLOR_MARGIN
                    ),
                    "minimum_semantic_conflict_confidence": (
                        MINIMUM_DIRECT_MULTIVIEW_SEMANTIC_CONFLICT_CONFIDENCE
                    ),
                    "at_least_one_strong_stable_view_required": True,
                    "strong_direct_conflict_veto": True,
                    "strong_semantic_conflict_veto": True,
                    "stable_resolved_conflict_veto": True,
                    "bbox_conflict_is_diagnostic_only": True,
                    "effective_confidence": effective_confidence,
                    "projection_evidence": copy.deepcopy(
                        direct_multiview_projection_details
                    ),
                }
            )
        elif (
            annotation_method
            == PALETTE_BOUND_SMALL_PART_SPATIAL_ANNOTATION_METHOD
        ):
            assert palette_bound_projection_details is not None
            spatial_annotation.update(
                {
                    "minimum_supporting_view_count": 1,
                    "canonical_palette_source_view_ids": (
                        palette_bound_projection_details[
                            "canonical_palette_source_view_ids"
                        ]
                    ),
                    "minimum_canonical_palette_source_view_count": (
                        minimum_support
                    ),
                    "small_part_diagnostic_status_required": "resolved",
                    "visibility_failure_reason_required": True,
                    "registration_label_stable_required": True,
                    "bbox_canonical_group_match_required": True,
                    "all_diagnostic_samples_resolved_required": True,
                    "strong_semantic_conflict_veto": True,
                    "stable_resolved_conflict_veto": True,
                    "effective_confidence": effective_confidence,
                    "projection_evidence": copy.deepcopy(
                        palette_bound_projection_details
                    ),
                }
            )
        elif annotation_method == SINGLE_VIEW_SPATIAL_ANNOTATION_METHOD:
            assert single_view_projection_details is not None
            spatial_annotation.update(
                {
                    "registration_label_stable_required": True,
                    "perturbation_label_stable_required": True,
                    "all_perturbation_groups_must_match": True,
                    "minimum_projected_part_pixels": (
                        MINIMUM_SINGLE_VIEW_SPATIAL_PIXELS
                    ),
                    "minimum_sampled_reference_pixels": (
                        MINIMUM_SINGLE_VIEW_SPATIAL_PIXELS
                    ),
                    "minimum_winning_color_share": (
                        MINIMUM_SINGLE_VIEW_COLOR_SHARE
                    ),
                    "minimum_color_margin": (
                        MINIMUM_SINGLE_VIEW_COLOR_MARGIN
                    ),
                    "minimum_bbox_color_margin": (
                        MINIMUM_SINGLE_VIEW_BBOX_COLOR_MARGIN
                    ),
                    "effective_confidence": effective_confidence,
                    "projection_evidence": copy.deepcopy(
                        single_view_projection_details
                    ),
                }
            )
        elif (
            annotation_method
            == FOREGROUND_SINGLE_VIEW_SPATIAL_ANNOTATION_METHOD
        ):
            assert foreground_single_view_projection_details is not None
            spatial_annotation.update(
                {
                    "minimum_supporting_view_count": 1,
                    "sam3_foreground_required": True,
                    "unseen_views_cast_no_vote": True,
                    "visible_conflicting_view_veto": True,
                    "perturbation_label_stable_required": True,
                    "minimum_sampled_foreground_pixels": (
                        MINIMUM_FOREGROUND_SINGLE_VIEW_PIXELS
                    ),
                    "minimum_foreground_overlap": (
                        MINIMUM_SPATIAL_FOREGROUND_OVERLAP
                    ),
                    "minimum_winning_color_share": (
                        MINIMUM_FOREGROUND_SINGLE_VIEW_COLOR_SHARE
                    ),
                    "minimum_color_margin": (
                        MINIMUM_FOREGROUND_SINGLE_VIEW_COLOR_MARGIN
                    ),
                    "effective_confidence": effective_confidence,
                    "projection_evidence": copy.deepcopy(
                        foreground_single_view_projection_details
                    ),
                }
            )
        mutable_provenance["canonical_group_id"] = group_id
        mutable_provenance["canonical_group_annotation"] = spatial_annotation
        assignment["provenance"] = mutable_provenance
        record.update(
            {
                "outcome": "ANNOTATED",
                "selected_group_id": group_id,
                "annotation_confidence": effective_confidence,
                "confidence_tier": {
                    SPATIAL_ANNOTATION_METHOD: (
                        "TRUSTED_MULTIVIEW_SPATIAL_PROJECTION"
                    ),
                    SEMANTIC_SPATIAL_ANNOTATION_METHOD: (
                        "CORROBORATED_MULTIVIEW_SEMANTIC_PROJECTION"
                    ),
                    MULTIVIEW_DIRECT_SPATIAL_ANNOTATION_METHOD: (
                        "CORROBORATED_MULTIVIEW_DIRECT_MASK_PROJECTION"
                    ),
                    PALETTE_BOUND_SMALL_PART_SPATIAL_ANNOTATION_METHOD: (
                        "MULTIVIEW_PALETTE_BOUND_SMALL_PART_PROJECTION"
                    ),
                    SINGLE_VIEW_SPATIAL_ANNOTATION_METHOD: (
                        "HIGH_CONFIDENCE_SINGLE_VIEW_SPATIAL_PROJECTION"
                    ),
                    FOREGROUND_SINGLE_VIEW_SPATIAL_ANNOTATION_METHOD: (
                        "SAM3_FOREGROUND_SINGLE_VIEW_PROJECTION"
                    ),
                }[annotation_method],
                "reason_codes": [
                    {
                        SPATIAL_ANNOTATION_METHOD: (
                            "UNIQUE_TRUSTED_MULTIVIEW_SPATIAL_PROJECTION_ACCEPTED"
                        ),
                        SEMANTIC_SPATIAL_ANNOTATION_METHOD: (
                            "CORROBORATED_MULTIVIEW_SEMANTIC_PROJECTION_ACCEPTED"
                        ),
                        MULTIVIEW_DIRECT_SPATIAL_ANNOTATION_METHOD: (
                            "CORROBORATED_MULTIVIEW_DIRECT_MASK_PROJECTION_ACCEPTED"
                        ),
                        PALETTE_BOUND_SMALL_PART_SPATIAL_ANNOTATION_METHOD: (
                            "MULTIVIEW_PALETTE_BOUND_SMALL_PART_PROJECTION_ACCEPTED"
                        ),
                        SINGLE_VIEW_SPATIAL_ANNOTATION_METHOD: (
                            "HIGH_CONFIDENCE_SINGLE_VIEW_SPATIAL_PROJECTION_ACCEPTED"
                        ),
                        FOREGROUND_SINGLE_VIEW_SPATIAL_ANNOTATION_METHOD: (
                            "SAM3_FOREGROUND_SINGLE_VIEW_PROJECTION_ACCEPTED"
                        ),
                    }[annotation_method]
                ],
                "candidates": [],
                "spatial_annotation": spatial_annotation,
            }
        )
        annotated.append(
            {
                "part_id": part_id,
                "canonical_group_id": group_id,
                "supporting_view_ids": supporting_view_ids,
                "method": annotation_method,
            }
        )
    semantic_recovery_count = sum(
        item["method"] == SEMANTIC_SPATIAL_ANNOTATION_METHOD
        for item in annotated
    )
    direct_multiview_recovery_count = sum(
        item["method"] == MULTIVIEW_DIRECT_SPATIAL_ANNOTATION_METHOD
        for item in annotated
    )
    single_view_recovery_count = sum(
        item["method"] == SINGLE_VIEW_SPATIAL_ANNOTATION_METHOD
        for item in annotated
    )
    foreground_single_view_recovery_count = sum(
        item["method"] == FOREGROUND_SINGLE_VIEW_SPATIAL_ANNOTATION_METHOD
        for item in annotated
    )
    palette_bound_small_part_recovery_count = sum(
        item["method"]
        == PALETTE_BOUND_SMALL_PART_SPATIAL_ANNOTATION_METHOD
        for item in annotated
    )
    return {
        "enabled": True,
        "method": SPATIAL_ANNOTATION_METHOD,
        "spatial_report_sha256": report_sha256,
        "minimum_supporting_view_count": minimum_support,
        "visual_equivalence_policy": (
            "unique_multiview_authority_for_same_normalized_color/v1"
        ),
        "visual_group_aliases": visual_group_aliases,
        "annotated_part_count": len(annotated),
        "semantic_recovery_annotated_part_count": semantic_recovery_count,
        "direct_multiview_recovery_annotated_part_count": (
            direct_multiview_recovery_count
        ),
        "single_view_recovery_annotated_part_count": (
            single_view_recovery_count
        ),
        "foreground_single_view_recovery_annotated_part_count": (
            foreground_single_view_recovery_count
        ),
        "palette_bound_small_part_recovery_annotated_part_count": (
            palette_bound_small_part_recovery_count
        ),
        "direct_multiview_rejections": sorted(
            direct_multiview_rejections,
            key=lambda item: str(item["part_id"]),
        ),
        "single_view_rejections": sorted(
            single_view_rejections,
            key=lambda item: str(item["part_id"]),
        ),
        "foreground_single_view_rejections": sorted(
            foreground_single_view_rejections,
            key=lambda item: str(item["part_id"]),
        ),
        "palette_bound_small_part_rejections": sorted(
            palette_bound_small_part_rejections,
            key=lambda item: str(item["part_id"]),
        ),
        "annotations": sorted(annotated, key=lambda item: str(item["part_id"])),
        "whole_parts_with_face_subsets_excluded": True,
        "material_identity_unchanged": True,
        "parameters_unchanged": True,
    }


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _explicit_whole_part_source_layout(part: Mapping[str, Any]) -> bool:
    raw_subsets = part.get("existing_material_bind_face_subsets")
    return (
        isinstance(raw_subsets, Sequence)
        and not isinstance(raw_subsets, (str, bytes))
        and len(raw_subsets) == 0
    )


def _legacy_geometry_proxy(part: Mapping[str, Any]) -> dict[str, Any] | None:
    point_count = part.get("point_count")
    face_count = part.get("face_count")
    world_bbox = part.get("world_bbox")
    if (
        isinstance(point_count, bool)
        or not isinstance(point_count, int)
        or point_count < 1
        or isinstance(face_count, bool)
        or not isinstance(face_count, int)
        or face_count < 1
        or not isinstance(world_bbox, Sequence)
        or isinstance(world_bbox, (str, bytes))
        or len(world_bbox) != 2
        or any(
            not isinstance(corner, Sequence)
            or isinstance(corner, (str, bytes))
            or len(corner) != 3
            for corner in world_bbox
        )
    ):
        return None
    coordinates = [value for corner in world_bbox for value in corner]
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in coordinates
    ):
        return None
    sorted_extents = sorted(
        round(
            abs(float(world_bbox[1][axis]) - float(world_bbox[0][axis])),
            9,
        )
        for axis in range(3)
    )
    if any(extent <= 0.0 for extent in sorted_extents):
        return None
    return {
        "point_count": point_count,
        "face_count": face_count,
        "sorted_bbox_extents": sorted_extents,
    }


def _source_appearance_layout_signature(
    part: Mapping[str, Any],
    *,
    strong_hashes_required: bool,
) -> str | None:
    """Build the dominant-assembly key without constraining geometry."""

    if not _explicit_whole_part_source_layout(part):
        return None
    strong_hash_fields = (
        "source_appearance_sha256",
        "source_subset_layout_sha256",
    )
    if all(_valid_sha256(part.get(field)) for field in strong_hash_fields):
        return _canonical_sha256(
            {
                "schema_version": (
                    "qwen-dominant-assembly-source-appearance-key/v2"
                ),
                **{field: part[field] for field in strong_hash_fields},
                "whole_part_no_source_subsets": True,
            }
        )
    # Never let a partially upgraded registry join a strong cohort through a
    # weaker key.  Old registries, where no part has the v2 fields, retain a
    # conservative compatibility path.
    if strong_hashes_required or any(
        part.get(field) is not None for field in strong_hash_fields
    ):
        return None
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
        return _canonical_sha256(
            {
                "schema_version": (
                    "qwen-dominant-assembly-source-appearance-key/v1"
                ),
                "stable_source_appearance": stable_properties,
                "source_subset_layout": "explicit_whole_part_no_subsets",
            }
        )
    except (TypeError, ValueError):
        return None


def _repeated_source_identity_signature(
    part: Mapping[str, Any],
    *,
    strong_hashes_required: bool,
) -> str | None:
    """Build the rare/repeated sibling key including exact geometry."""

    if not _explicit_whole_part_source_layout(part):
        return None
    strong_hash_fields = (
        "geometry_content_sha256",
        "source_appearance_sha256",
        "source_subset_layout_sha256",
    )
    if all(_valid_sha256(part.get(field)) for field in strong_hash_fields):
        return _canonical_sha256(
            {
                "schema_version": "qwen-repeated-source-identity-key/v2",
                **{field: part[field] for field in strong_hash_fields},
                "whole_part_no_source_subsets": True,
            }
        )
    if strong_hashes_required or any(
        part.get(field) is not None for field in strong_hash_fields
    ):
        return None
    appearance_signature = _source_appearance_layout_signature(
        part,
        strong_hashes_required=False,
    )
    geometry_proxy = _legacy_geometry_proxy(part)
    if appearance_signature is None or geometry_proxy is None:
        return None
    return _canonical_sha256(
        {
            "schema_version": "qwen-repeated-source-identity-key/v1",
            "source_appearance_layout_signature_sha256": (
                appearance_signature
            ),
            "geometry_proxy": geometry_proxy,
        }
    )


def _assembly_ancestor_paths(parent_path: str) -> list[str]:
    components = [component for component in parent_path.split("/") if component]
    return [
        "/" + "/".join(components[:end])
        for end in range(1, len(components) + 1)
    ]


def _direct_assembly_path(parent_path: str) -> str:
    if "/" not in parent_path[1:]:
        return "/"
    return parent_path.rsplit("/", 1)[0]


def _validated_source_appearance_registry(
    *,
    part_registry: Mapping[str, Any],
    assignment_part_ids: set[str],
) -> dict[str, Mapping[str, Any]]:
    if part_registry.get("schema_version") != "qwen-material-parts/v1":
        raise VisualGroupAnnotationError(
            "source-appearance cohort registry has an unsupported schema_version"
        )
    raw_parts = _sequence(
        part_registry.get("parts"),
        "source-appearance cohort registry.parts",
    )
    registry_by_part: dict[str, Mapping[str, Any]] = {}
    for index, raw_part in enumerate(raw_parts):
        part = _mapping(
            raw_part,
            f"source-appearance cohort registry.parts[{index}]",
        )
        part_id = _text(
            part.get("part_id"),
            f"source-appearance cohort registry.parts[{index}].part_id",
        )
        if part_id in registry_by_part:
            raise VisualGroupAnnotationError(
                f"source-appearance cohort registry repeats part_id {part_id}"
            )
        registry_by_part[part_id] = part
    declared_count = part_registry.get("part_count")
    if (
        isinstance(declared_count, bool)
        or not isinstance(declared_count, int)
        or declared_count != len(registry_by_part)
    ):
        raise VisualGroupAnnotationError(
            "source-appearance cohort registry part_count is inconsistent"
        )
    registry_part_ids = set(registry_by_part)
    if registry_part_ids != assignment_part_ids:
        raise VisualGroupAnnotationError(
            "source-appearance cohort registry and material plan must have "
            "identical part IDs; missing_from_registry="
            f"{sorted(assignment_part_ids - registry_part_ids)!r}, "
            "missing_from_plan="
            f"{sorted(registry_part_ids - assignment_part_ids)!r}"
        )
    return registry_by_part


def _trusted_spatial_groups_by_part(
    spatial_mapping_report: Mapping[str, Any],
) -> dict[str, set[str]]:
    groups_by_part: dict[str, set[str]] = {}
    for raw_part in _sequence(
        spatial_mapping_report.get("parts"),
        "spatial mapping report.parts",
    ):
        if not isinstance(raw_part, Mapping):
            continue
        part_id = raw_part.get("part_id")
        if not isinstance(part_id, str) or not part_id:
            continue
        groups: set[str] = set()
        for observation in raw_part.get("observations", []):
            if (
                isinstance(observation, Mapping)
                and observation.get("classification") == "resolved"
                and observation.get("registration_label_stable") is True
                and observation.get("perturbation_label_stable") is True
                and isinstance(observation.get("canonical_group_id"), str)
                and observation["canonical_group_id"]
            ):
                groups.add(str(observation["canonical_group_id"]))
        groups_by_part[part_id] = groups
    return groups_by_part


def _apply_repeated_subset_visual_cohort_annotations(
    *,
    assignments: Sequence[Any],
    records: list[dict[str, Any]],
    canonical_groups: Mapping[str, Mapping[str, Any]],
    part_registry: Mapping[str, Any] | None,
    spatial_mapping_report: Mapping[str, Any] | None,
    annotation_input_plan_sha256: str,
) -> dict[str, Any]:
    """Bind exact repeated face-subset layouts to one proven visual group.

    A CAD materialBind subset is an authoring boundary, not appearance
    evidence.  This lane therefore never uses subset names or source colours
    to select a group.  It accepts either one parent resolved to the same group
    in two stable reference views, or an exact geometry/subset-layout cohort
    with two independently localized parents spanning two reference views.
    The parent and every topology-complete subset are then evaluated together
    by the immutable render tournament.  Material identities and parameters
    are unchanged here.
    """

    empty: dict[str, Any] = {
        "schema_version": REPEATED_SUBSET_VISUAL_COHORT_SCHEMA_VERSION,
        "enabled": False,
        "method": REPEATED_SUBSET_VISUAL_COHORT_METHOD,
        "annotation_input_plan_sha256": annotation_input_plan_sha256,
        "registry_sha256": None,
        "spatial_report_sha256": None,
        "cohort_count": 0,
        "annotated_part_count": 0,
        "annotated_face_subset_count": 0,
        "contracts": [],
        "rejected_candidates": [],
        "material_identity_unchanged": True,
        "parameters_unchanged": True,
        "exact_cover": True,
    }
    if part_registry is None or spatial_mapping_report is None:
        empty["disabled_reason"] = (
            "PART_REGISTRY_UNAVAILABLE"
            if part_registry is None
            else "SPATIAL_MAPPING_REPORT_UNAVAILABLE"
        )
        return empty

    assignments_by_part = {
        str(assignment["part_id"]): assignment
        for assignment in assignments
        if isinstance(assignment, dict)
        and isinstance(assignment.get("part_id"), str)
    }
    registry_by_part = _validated_source_appearance_registry(
        part_registry=part_registry,
        assignment_part_ids=set(assignments_by_part),
    )
    records_by_key = {
        (
            str(record["part_id"]),
            (
                str(record["subset_name"])
                if isinstance(record.get("subset_name"), str)
                else ""
            ),
        ): record
        for record in records
        if isinstance(record.get("part_id"), str)
    }
    integrity = spatial_mapping_report.get("integrity")
    report_sha256 = (
        integrity.get("report_sha256")
        if isinstance(integrity, Mapping)
        else None
    )
    if not _valid_sha256(report_sha256):
        raise VisualGroupAnnotationError(
            "repeated subset visual cohort lacks a valid spatial report hash"
        )

    stable_group_views: dict[str, dict[str, set[str]]] = {}
    for raw_part in _sequence(
        spatial_mapping_report.get("parts"),
        "spatial mapping report.parts",
    ):
        if not isinstance(raw_part, Mapping):
            continue
        part_id = raw_part.get("part_id")
        if not isinstance(part_id, str) or part_id not in assignments_by_part:
            continue
        for observation in raw_part.get("observations", []):
            if (
                not isinstance(observation, Mapping)
                or observation.get("classification") != "resolved"
                or observation.get("registration_label_stable") is not True
                or observation.get("perturbation_label_stable") is not True
            ):
                continue
            group_id = observation.get("canonical_group_id")
            view_id = observation.get("reference_view_id")
            if (
                isinstance(group_id, str)
                and group_id in canonical_groups
                and isinstance(view_id, str)
                and view_id
            ):
                stable_group_views.setdefault(part_id, {}).setdefault(
                    group_id, set()
                ).add(view_id)

    eligible_parts: dict[str, dict[str, Any]] = {}
    rejected: list[dict[str, Any]] = []
    for part_id, registry_part in registry_by_part.items():
        raw_source_subsets = registry_part.get(
            "existing_material_bind_face_subsets"
        )
        assignment = assignments_by_part[part_id]
        raw_plan_subsets = assignment.get("face_subsets")
        if (
            not isinstance(raw_source_subsets, list)
            or not raw_source_subsets
            or not isinstance(raw_plan_subsets, list)
            or not raw_plan_subsets
        ):
            continue
        geometry_hash = registry_part.get("geometry_content_sha256")
        subset_layout_hash = registry_part.get("source_subset_layout_sha256")
        if not _valid_sha256(geometry_hash) or not _valid_sha256(
            subset_layout_hash
        ):
            rejected.append(
                {
                    "part_id": part_id,
                    "reason_codes": ["STRONG_GEOMETRY_OR_SUBSET_LAYOUT_HASH_MISSING"],
                }
            )
            continue
        source_names = [
            subset.get("subset_name")
            for subset in raw_source_subsets
            if isinstance(subset, Mapping)
        ]
        plan_names = [
            subset.get("subset_name")
            for subset in raw_plan_subsets
            if isinstance(subset, Mapping)
        ]
        source_face_count = sum(
            len(subset.get("face_indices", []))
            for subset in raw_source_subsets
            if isinstance(subset, Mapping)
            and isinstance(subset.get("face_indices"), list)
        )
        if (
            len(source_names) != len(raw_source_subsets)
            or len(plan_names) != len(raw_plan_subsets)
            or source_names != plan_names
            or source_face_count != registry_part.get("face_count")
        ):
            rejected.append(
                {
                    "part_id": part_id,
                    "reason_codes": ["FACE_SUBSET_TOPOLOGY_CONTRACT_INCOMPLETE"],
                }
            )
            continue
        eligible_parts[part_id] = {
            "signature": _canonical_sha256(
                {
                    "schema_version": (
                        "qwen-repeated-subset-visual-identity/v1"
                    ),
                    "geometry_content_sha256": geometry_hash,
                    "source_subset_layout_sha256": subset_layout_hash,
                }
            ),
            "subset_names": source_names,
        }

    candidates: list[dict[str, Any]] = []
    for part_id in sorted(eligible_parts):
        groups = stable_group_views.get(part_id, {})
        if len(groups) != 1:
            continue
        group_id, view_ids = next(iter(groups.items()))
        group = canonical_groups[group_id]
        if (
            len(view_ids) >= 2
            and len(group.get("source_view_ids", [])) >= 2
        ):
            candidates.append(
                {
                    "candidate_kind": "direct_multiview_subset_owner",
                    "canonical_group_id": group_id,
                    "member_part_ids": [part_id],
                    "anchor_part_ids": [part_id],
                    "supporting_view_ids": sorted(view_ids),
                    "signature": eligible_parts[part_id]["signature"],
                }
            )

    parts_by_signature: dict[str, list[str]] = {}
    for part_id, evidence in eligible_parts.items():
        parts_by_signature.setdefault(str(evidence["signature"]), []).append(
            part_id
        )
    for signature, raw_member_ids in sorted(parts_by_signature.items()):
        member_ids = sorted(raw_member_ids)
        if not 2 <= len(member_ids) <= REPEATED_SUBSET_VISUAL_COHORT_MAX_MEMBER_COUNT:
            continue
        anchored = [
            part_id
            for part_id in member_ids
            if len(stable_group_views.get(part_id, {})) == 1
        ]
        if len(anchored) < 2:
            continue
        anchor_group_ids = {
            next(iter(stable_group_views[part_id]))
            for part_id in anchored
        }
        if len(anchor_group_ids) != 1:
            rejected.append(
                {
                    "signature": signature,
                    "member_part_ids": member_ids,
                    "reason_codes": ["REPEATED_SUBSET_ANCHOR_GROUP_CONFLICT"],
                }
            )
            continue
        group_id = next(iter(anchor_group_ids))
        supporting_view_ids = sorted(
            {
                view_id
                for part_id in anchored
                for view_ids in stable_group_views[part_id].values()
                for view_id in view_ids
            }
        )
        if (
            len(supporting_view_ids) < 2
            or len(canonical_groups[group_id].get("source_view_ids", [])) < 2
        ):
            continue
        candidates.append(
            {
                "candidate_kind": "exact_repeated_geometry_subset_layout",
                "canonical_group_id": group_id,
                "member_part_ids": member_ids,
                "anchor_part_ids": anchored,
                "supporting_view_ids": supporting_view_ids,
                "signature": signature,
            }
        )

    accepted: list[dict[str, Any]] = []
    claimed: set[str] = set()
    for candidate in sorted(
        candidates,
        key=lambda item: (
            0
            if item["candidate_kind"] == "direct_multiview_subset_owner"
            else 1,
            str(item["canonical_group_id"]),
            tuple(item["member_part_ids"]),
        ),
    ):
        member_ids = set(candidate["member_part_ids"])
        if member_ids & claimed:
            continue
        group_id = str(candidate["canonical_group_id"])
        conflicts: list[str] = []
        for part_id in sorted(member_ids):
            assignment = assignments_by_part[part_id]
            provenance = assignment.get("provenance")
            existing_parent = (
                provenance.get("canonical_group_id")
                if isinstance(provenance, Mapping)
                else None
            )
            subset_map = (
                provenance.get("face_subset_canonical_group_ids", {})
                if isinstance(provenance, Mapping)
                else {}
            )
            if isinstance(existing_parent, str) and existing_parent != group_id:
                conflicts.append(part_id)
                continue
            if isinstance(subset_map, Mapping) and any(
                isinstance(value, str) and value != group_id
                for value in subset_map.values()
            ):
                conflicts.append(part_id)
                continue
            for subset_name in eligible_parts[part_id]["subset_names"]:
                record = records_by_key.get((part_id, str(subset_name)))
                selected = (
                    record.get("selected_group_id")
                    if isinstance(record, Mapping)
                    else None
                )
                if isinstance(selected, str) and selected != group_id:
                    conflicts.append(part_id)
                    break
        if conflicts:
            rejected.append(
                {
                    **copy.deepcopy(candidate),
                    "reason_codes": ["AUTHORITATIVE_SUBSET_GROUP_CONFLICT"],
                    "conflicting_part_ids": sorted(set(conflicts)),
                }
            )
            continue
        claimed.update(member_ids)
        accepted.append(candidate)

    contracts: list[dict[str, Any]] = []
    annotated_part_ids: set[str] = set()
    annotated_subset_keys: set[tuple[str, str]] = set()
    for candidate in accepted:
        group_id = str(candidate["canonical_group_id"])
        supporting_view_ids = list(candidate["supporting_view_ids"])
        contract = {
            "schema_version": REPEATED_SUBSET_VISUAL_COHORT_SCHEMA_VERSION,
            **copy.deepcopy(candidate),
            "registry_sha256": _canonical_sha256(part_registry),
            "spatial_report_sha256": report_sha256,
            "annotation_input_plan_sha256": annotation_input_plan_sha256,
            "material_identity_unchanged": True,
            "parameters_unchanged": True,
            "exact_cover": True,
        }
        contract["contract_sha256"] = _canonical_sha256(contract)
        contracts.append(contract)
        for part_id in candidate["member_part_ids"]:
            assignment = assignments_by_part[part_id]
            provenance = assignment.get("provenance")
            mutable_provenance = (
                dict(provenance) if isinstance(provenance, Mapping) else {}
            )
            mutable_provenance["canonical_group_id"] = group_id
            mutable_provenance["face_subset_canonical_group_ids"] = {
                str(subset_name): group_id
                for subset_name in eligible_parts[part_id]["subset_names"]
            }
            lineage = {
                "method": REPEATED_SUBSET_VISUAL_COHORT_METHOD,
                "contract_sha256": contract["contract_sha256"],
                "candidate_kind": candidate["candidate_kind"],
                "canonical_group_id": group_id,
                "anchor_part_ids": list(candidate["anchor_part_ids"]),
                "supporting_view_ids": supporting_view_ids,
                "topology_complete_face_subsets": True,
                "material_identity_unchanged": True,
                "parameters_unchanged": True,
            }
            mutable_provenance["repeated_subset_visual_cohort"] = lineage
            assignment["provenance"] = mutable_provenance
            spatial_annotation = {
                **copy.deepcopy(lineage),
                "supporting_view_count": len(supporting_view_ids),
                "minimum_supporting_view_count": 2,
                "conflicting_view_ids": [],
                "unique_canonical_group": True,
            }
            for subset_name in [
                "",
                *[
                    str(value)
                    for value in eligible_parts[part_id]["subset_names"]
                ],
            ]:
                record = records_by_key.get((part_id, subset_name))
                if record is None:
                    raise VisualGroupAnnotationError(
                        "repeated subset visual cohort is missing an entity "
                        f"record: {part_id}:{subset_name}"
                    )
                if record.get("selected_group_id") == group_id:
                    continue
                record.update(
                    {
                        "outcome": "ANNOTATED",
                        "selected_group_id": group_id,
                        "annotation_confidence": 1.0,
                        "confidence_tier": (
                            "TRUSTED_MULTIVIEW_REPEATED_SUBSET_VISUAL_COHORT"
                        ),
                        "reason_codes": [
                            "MULTIVIEW_SUBSET_VISUAL_COHORT_ACCEPTED"
                        ],
                        "candidates": [],
                        **(
                            {
                                "spatial_annotation": copy.deepcopy(
                                    spatial_annotation
                                )
                            }
                            if part_id in candidate["anchor_part_ids"]
                            else {
                                "repeated_subset_visual_cohort": copy.deepcopy(
                                    lineage
                                )
                            }
                        ),
                    }
                )
                if subset_name:
                    annotated_subset_keys.add((part_id, subset_name))
                else:
                    annotated_part_ids.add(part_id)

    return {
        **empty,
        "enabled": True,
        "registry_sha256": _canonical_sha256(part_registry),
        "spatial_report_sha256": report_sha256,
        "cohort_count": len(contracts),
        "annotated_part_count": len(annotated_part_ids),
        "annotated_face_subset_count": len(annotated_subset_keys),
        "contracts": contracts,
        "rejected_candidates": rejected,
    }


def _apply_source_appearance_cohort_annotations(
    *,
    assignments: Sequence[Any],
    records: list[dict[str, Any]],
    part_registry: Mapping[str, Any] | None,
    spatial_mapping_report: Mapping[str, Any] | None,
    spatial_annotation_audit: Mapping[str, Any],
    annotation_input_plan_sha256: str,
) -> dict[str, Any]:
    """Propagate trusted spatial lineage through bounded authored cohorts.

    The source USD hierarchy and stable authored appearance select members;
    they never choose a canonical group.  A canonical group can enter this
    lane only through a spatial annotation accepted immediately beforehand.
    Existing authoritative group conflicts fail the candidate closed.  Older
    provisional hypotheses are retained as audit evidence, but cannot create
    an anchor or override the stronger spatial/structural contract.
    """

    empty_audit: dict[str, Any] = {
        "schema_version": SOURCE_APPEARANCE_COHORT_SCHEMA_VERSION,
        "enabled": False,
        "method": SOURCE_APPEARANCE_COHORT_METHOD,
        "registry_sha256": None,
        "spatial_report_sha256": spatial_annotation_audit.get(
            "spatial_report_sha256"
        ),
        "annotation_input_plan_sha256": annotation_input_plan_sha256,
        "cohort_count": 0,
        "expected_member_count": 0,
        "propagated_member_count": 0,
        "contracts": [],
        "rejected_candidates": [],
        "coverage_blockers": [],
        "exact_cover": True,
        "material_identity_unchanged": True,
        "parameters_unchanged": True,
    }
    if part_registry is None or spatial_mapping_report is None:
        empty_audit["disabled_reason"] = (
            "PART_REGISTRY_UNAVAILABLE"
            if part_registry is None
            else "SPATIAL_MAPPING_REPORT_UNAVAILABLE"
        )
        return empty_audit

    assignments_by_part = {
        str(assignment["part_id"]): assignment
        for assignment in assignments
        if isinstance(assignment, dict)
        and isinstance(assignment.get("part_id"), str)
    }
    records_by_part = {
        str(record["part_id"]): record
        for record in records
        if record.get("entity_kind") == "assignment"
        and isinstance(record.get("part_id"), str)
    }
    registry_by_part = _validated_source_appearance_registry(
        part_registry=part_registry,
        assignment_part_ids=set(assignments_by_part),
    )
    registry_sha256 = _canonical_sha256(part_registry)
    spatial_report_sha256 = spatial_annotation_audit.get("spatial_report_sha256")
    trusted_spatial_groups = _trusted_spatial_groups_by_part(
        spatial_mapping_report
    )
    dominant_strong_hash_fields = (
        "source_appearance_sha256",
        "source_subset_layout_sha256",
    )
    repeated_strong_hash_fields = (
        "geometry_content_sha256",
        *dominant_strong_hash_fields,
    )
    dominant_strong_hashes_required = any(
        all(
            _valid_sha256(part.get(field))
            for field in dominant_strong_hash_fields
        )
        for part in registry_by_part.values()
    )
    repeated_strong_hashes_required = any(
        all(
            _valid_sha256(part.get(field))
            for field in repeated_strong_hash_fields
        )
        for part in registry_by_part.values()
    )
    dominant_strong_hash_part_ids = sorted(
        part_id
        for part_id, part in registry_by_part.items()
        if all(
            _valid_sha256(part.get(field))
            for field in dominant_strong_hash_fields
        )
    )
    repeated_strong_hash_part_ids = sorted(
        part_id
        for part_id, part in registry_by_part.items()
        if all(
            _valid_sha256(part.get(field))
            for field in repeated_strong_hash_fields
        )
    )
    source_appearance_layout_signatures = {
        part_id: _source_appearance_layout_signature(
            part,
            strong_hashes_required=dominant_strong_hashes_required,
        )
        for part_id, part in registry_by_part.items()
    }
    repeated_source_identity_signatures = {
        part_id: _repeated_source_identity_signature(
            part,
            strong_hashes_required=repeated_strong_hashes_required,
        )
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

    anchors: list[dict[str, str]] = []
    for annotation in spatial_annotation_audit.get("annotations", []):
        if not isinstance(annotation, Mapping):
            continue
        part_id = annotation.get("part_id")
        group_id = annotation.get("canonical_group_id")
        if (
            isinstance(part_id, str)
            and part_id in assignments_by_part
            and isinstance(group_id, str)
            and group_id
            and (
                source_appearance_layout_signatures.get(part_id) is not None
                or repeated_source_identity_signatures.get(part_id) is not None
            )
            and part_id in parent_paths
            and part_id in face_counts
        ):
            anchors.append(
                {
                    "part_id": part_id,
                    "canonical_group_id": group_id,
                    **(
                        {
                            "source_appearance_layout_signature_sha256": str(
                                source_appearance_layout_signatures[part_id]
                            )
                        }
                        if source_appearance_layout_signatures.get(part_id)
                        is not None
                        else {}
                    ),
                    **(
                        {
                            "repeated_source_identity_signature_sha256": str(
                                repeated_source_identity_signatures[part_id]
                            )
                        }
                        if repeated_source_identity_signatures.get(part_id)
                        is not None
                        else {}
                    ),
                }
            )

    registry_count = len(registry_by_part)
    maximum_cohort_size = max(
        SOURCE_APPEARANCE_COHORT_SMALL_REGISTRY_ALLOWANCE,
        min(
            SOURCE_APPEARANCE_COHORT_MAX_MEMBER_COUNT,
            math.floor(
                SOURCE_APPEARANCE_COHORT_MAX_REGISTRY_FRACTION
                * registry_count
            ),
        ),
    )
    repeated_signature_part_ids: dict[str, list[str]] = {}
    for part_id, signature in repeated_source_identity_signatures.items():
        if signature is not None:
            repeated_signature_part_ids.setdefault(signature, []).append(part_id)
    appearance_signature_part_ids: dict[str, list[str]] = {}
    for part_id, signature in source_appearance_layout_signatures.items():
        if signature is not None:
            appearance_signature_part_ids.setdefault(signature, []).append(
                part_id
            )

    def member_hard_conflicts(part_id: str, group_id: str) -> list[str]:
        reasons: list[str] = []
        assignment = assignments_by_part[part_id]
        provenance = assignment.get("provenance")
        existing_group_id = (
            provenance.get("canonical_group_id")
            if isinstance(provenance, Mapping)
            else None
        )
        if isinstance(existing_group_id, str) and existing_group_id != group_id:
            reasons.append("AUTHORITATIVE_CANONICAL_GROUP_CONFLICT")
        alternative_spatial_groups = sorted(
            trusted_spatial_groups.get(part_id, set()) - {group_id}
        )
        if alternative_spatial_groups:
            reasons.append("TRUSTED_STABLE_SPATIAL_GROUP_CONFLICT")
        if assignment.get("face_subsets"):
            reasons.append("FACE_SUBSETS_FORBID_WHOLE_PART_PROPAGATION")
        if assignment.get("apply_action") == "source_visual_preserve":
            reasons.append("SOURCE_VISUAL_PRESERVE_FORBIDS_PROPAGATION")
        return reasons

    def advisory_conflicts(part_id: str, group_id: str) -> list[str]:
        provenance = assignments_by_part[part_id].get("provenance")
        provisional_group_id = (
            provenance.get("provisional_canonical_group_id")
            if isinstance(provenance, Mapping)
            else None
        )
        if (
            isinstance(provisional_group_id, str)
            and provisional_group_id
            and provisional_group_id != group_id
        ):
            return ["NON_AUTHORITATIVE_PROVISIONAL_GROUP_DISAGREES"]
        return []

    raw_candidates: dict[
        tuple[str, str, str, str, tuple[str, ...]], dict[str, Any]
    ] = {}
    rejected_candidates: list[dict[str, Any]] = []

    def add_candidate(
        *,
        candidate_kind: str,
        signature_kind: str,
        group_id: str,
        signature: str,
        assembly_path: str,
        member_part_ids: list[str],
        subtree_part_ids: list[str],
        anchor_part_id: str,
        signature_part_share: float | None,
        signature_face_share: float | None,
    ) -> None:
        member_part_ids = sorted(set(member_part_ids))
        hard_conflicts = {
            part_id: member_hard_conflicts(part_id, group_id)
            for part_id in member_part_ids
        }
        hard_conflicts = {
            part_id: reasons
            for part_id, reasons in hard_conflicts.items()
            if reasons
        }
        if hard_conflicts:
            rejected_candidates.append(
                {
                    "candidate_kind": candidate_kind,
                    "cohort_signature_kind": signature_kind,
                    "canonical_group_id": group_id,
                    "assembly_path": assembly_path,
                    "source_appearance_cohort_signature_sha256": signature,
                    "member_part_ids": member_part_ids,
                    "reason_codes": ["AUTHORITATIVE_MEMBER_CONFLICT"],
                    "member_conflicts": hard_conflicts,
                }
            )
            return
        if not any(
            not isinstance(
                assignments_by_part[part_id].get("provenance"), Mapping
            )
            or assignments_by_part[part_id]["provenance"].get(
                "canonical_group_id"
            )
            is None
            for part_id in member_part_ids
        ):
            return
        key = (
            candidate_kind,
            group_id,
            signature,
            assembly_path,
            tuple(member_part_ids),
        )
        candidate = raw_candidates.get(key)
        if candidate is None:
            candidate = {
                "candidate_kind": candidate_kind,
                "cohort_signature_kind": signature_kind,
                "canonical_group_id": group_id,
                "assembly_path": assembly_path,
                "source_appearance_cohort_signature_sha256": signature,
                "anchor_part_ids": [],
                "expected_member_part_ids": member_part_ids,
                "subtree_part_ids": sorted(subtree_part_ids),
                "signature_part_share": signature_part_share,
                "signature_face_share": signature_face_share,
            }
            raw_candidates[key] = candidate
        candidate["anchor_part_ids"] = sorted(
            set(candidate["anchor_part_ids"]) | {anchor_part_id}
        )

    # Dominant authored subtrees are safe even when the same PreviewSurface
    # signature is common elsewhere in a large CAD assembly.
    for anchor in anchors:
        anchor_part_id = anchor["part_id"]
        group_id = anchor["canonical_group_id"]
        signature = anchor.get("source_appearance_layout_signature_sha256")
        if not isinstance(signature, str):
            continue
        qualifying: list[tuple[str, list[str], list[str], float, float]] = []
        for assembly_path in _assembly_ancestor_paths(
            parent_paths[anchor_part_id]
        ):
            subtree_part_ids = sorted(
                part_id
                for part_id, parent_path in parent_paths.items()
                if parent_path == assembly_path
                or parent_path.startswith(f"{assembly_path}/")
            )
            member_part_ids = sorted(
                part_id
                for part_id in subtree_part_ids
                if source_appearance_layout_signatures.get(part_id) == signature
            )
            if (
                not SOURCE_APPEARANCE_COHORT_MIN_MEMBER_COUNT
                <= len(member_part_ids)
                <= maximum_cohort_size
                or len(subtree_part_ids) > maximum_cohort_size
                or any(part_id not in face_counts for part_id in subtree_part_ids)
            ):
                continue
            subtree_face_count = sum(
                face_counts[part_id] for part_id in subtree_part_ids
            )
            signature_face_count = sum(
                face_counts[part_id] for part_id in member_part_ids
            )
            part_share = len(member_part_ids) / len(subtree_part_ids)
            face_share = (
                signature_face_count / subtree_face_count
                if subtree_face_count
                else 0.0
            )
            if (
                part_share
                < SOURCE_APPEARANCE_COHORT_MIN_SIGNATURE_PART_SHARE
                or face_share
                < SOURCE_APPEARANCE_COHORT_MIN_SIGNATURE_FACE_SHARE
            ):
                continue
            qualifying.append(
                (
                    assembly_path,
                    member_part_ids,
                    subtree_part_ids,
                    part_share,
                    face_share,
                )
            )
        if qualifying:
            # The deepest valid authored subtree is the narrowest contract.
            (
                assembly_path,
                member_part_ids,
                subtree_part_ids,
                part_share,
                face_share,
            ) = max(qualifying, key=lambda item: item[0].count("/"))
            add_candidate(
                candidate_kind="dominant_assembly",
                signature_kind="source_appearance_plus_subset_layout",
                group_id=group_id,
                signature=signature,
                assembly_path=assembly_path,
                member_part_ids=member_part_ids,
                subtree_part_ids=subtree_part_ids,
                anchor_part_id=anchor_part_id,
                signature_part_share=part_share,
                signature_face_share=face_share,
            )

    # A globally unique two-member signature may span two sibling branches.
    # This bounded lane recovers the second part without allowing a common
    # neutral source material to spread across the whole asset.
    anchors_by_signature: dict[str, list[dict[str, str]]] = {}
    for anchor in anchors:
        signature = anchor.get("repeated_source_identity_signature_sha256")
        if isinstance(signature, str):
            anchors_by_signature.setdefault(signature, []).append(anchor)
    for signature, raw_part_ids in sorted(repeated_signature_part_ids.items()):
        part_ids = sorted(raw_part_ids)
        if len(part_ids) != SOURCE_APPEARANCE_RARE_PAIR_SIZE:
            continue
        if any(part_id not in parent_paths or part_id not in face_counts for part_id in part_ids):
            continue
        assembly_paths = {
            _direct_assembly_path(parent_paths[part_id]) for part_id in part_ids
        }
        if len(assembly_paths) != 1 or len(
            {parent_paths[part_id] for part_id in part_ids}
        ) != SOURCE_APPEARANCE_RARE_PAIR_SIZE:
            continue
        signature_anchors = anchors_by_signature.get(signature, [])
        anchor_group_ids = {
            anchor["canonical_group_id"] for anchor in signature_anchors
        }
        if len(anchor_group_ids) != 1:
            continue
        group_id = next(iter(anchor_group_ids))
        assembly_path = next(iter(assembly_paths))
        subtree_part_ids = sorted(
            part_id
            for part_id, parent_path in parent_paths.items()
            if parent_path == assembly_path
            or parent_path.startswith(f"{assembly_path}/")
        )
        for anchor in signature_anchors:
            add_candidate(
                candidate_kind="rare_source_appearance_pair",
                signature_kind="geometry_plus_appearance_plus_subset_layout",
                group_id=group_id,
                signature=signature,
                assembly_path=assembly_path,
                member_part_ids=part_ids,
                subtree_part_ids=subtree_part_ids,
                anchor_part_id=anchor["part_id"],
                signature_part_share=None,
                signature_face_share=None,
            )

    # Geometry can legitimately differ between a two-piece visual assembly
    # (for example, two segments of one copper conduit).  When an exact
    # appearance/subset-layout signature occurs only twice in the entire
    # registry, both members are sibling branches of one direct assembly, and
    # at least one trusted spatial anchor selects one unique group, propagate
    # membership without pretending the geometry is identical.
    anchors_by_appearance_signature: dict[str, list[dict[str, str]]] = {}
    for anchor in anchors:
        signature = anchor.get(
            "source_appearance_layout_signature_sha256"
        )
        if isinstance(signature, str):
            anchors_by_appearance_signature.setdefault(signature, []).append(
                anchor
            )
    for signature, raw_part_ids in sorted(
        appearance_signature_part_ids.items()
    ):
        part_ids = sorted(raw_part_ids)
        if len(part_ids) != SOURCE_APPEARANCE_RARE_PAIR_SIZE:
            continue
        if any(
            part_id not in parent_paths or part_id not in face_counts
            for part_id in part_ids
        ):
            continue
        assembly_paths = {
            _direct_assembly_path(parent_paths[part_id])
            for part_id in part_ids
        }
        if (
            len(assembly_paths) != 1
            or len({parent_paths[part_id] for part_id in part_ids})
            != SOURCE_APPEARANCE_RARE_PAIR_SIZE
        ):
            continue
        signature_anchors = anchors_by_appearance_signature.get(
            signature, []
        )
        anchor_group_ids = {
            anchor["canonical_group_id"] for anchor in signature_anchors
        }
        if len(anchor_group_ids) != 1:
            continue
        group_id = next(iter(anchor_group_ids))
        assembly_path = next(iter(assembly_paths))
        subtree_part_ids = sorted(
            part_id
            for part_id, parent_path in parent_paths.items()
            if parent_path == assembly_path
            or parent_path.startswith(f"{assembly_path}/")
        )
        for anchor in signature_anchors:
            add_candidate(
                candidate_kind="rare_source_appearance_layout_pair",
                signature_kind="source_appearance_plus_subset_layout",
                group_id=group_id,
                signature=signature,
                assembly_path=assembly_path,
                member_part_ids=part_ids,
                subtree_part_ids=subtree_part_ids,
                anchor_part_id=anchor["part_id"],
                signature_part_share=None,
                signature_face_share=None,
            )

    accepted_candidates: list[dict[str, Any]] = []
    claimed_part_ids: set[str] = set()
    for candidate in sorted(
        raw_candidates.values(),
        key=lambda item: (
            -str(item["assembly_path"]).count("/"),
            0 if item["candidate_kind"] == "dominant_assembly" else 1,
            str(item["canonical_group_id"]),
            str(item["assembly_path"]),
        ),
    ):
        member_part_ids = set(candidate["expected_member_part_ids"])
        overlap = sorted(member_part_ids & claimed_part_ids)
        if overlap:
            rejected_candidates.append(
                {
                    **copy.deepcopy(candidate),
                    "reason_codes": ["OVERLAPPING_ACCEPTED_COHORT"],
                    "overlapping_part_ids": overlap,
                }
            )
            continue
        claimed_part_ids.update(member_part_ids)
        accepted_candidates.append(candidate)

    contracts: list[dict[str, Any]] = []
    propagated_member_ids: set[str] = set()
    for candidate in accepted_candidates:
        group_id = str(candidate["canonical_group_id"])
        expected_member_part_ids = list(candidate["expected_member_part_ids"])
        anchor_part_ids = sorted(set(candidate["anchor_part_ids"]))
        newly_propagated = sorted(
            part_id
            for part_id in expected_member_part_ids
            if not isinstance(
                assignments_by_part[part_id].get("provenance"), Mapping
            )
            or assignments_by_part[part_id]["provenance"].get(
                "canonical_group_id"
            )
            is None
        )
        identity_payload = {
            "schema_version": SOURCE_APPEARANCE_COHORT_CONTRACT_SCHEMA_VERSION,
            "method": SOURCE_APPEARANCE_COHORT_METHOD,
            "candidate_kind": candidate["candidate_kind"],
            "cohort_signature_kind": candidate["cohort_signature_kind"],
            "canonical_group_id": group_id,
            "assembly_path": candidate["assembly_path"],
            "source_appearance_cohort_signature_sha256": candidate[
                "source_appearance_cohort_signature_sha256"
            ],
            "anchor_part_ids": anchor_part_ids,
            "expected_member_part_ids": expected_member_part_ids,
            "registry_sha256": registry_sha256,
            "spatial_report_sha256": spatial_report_sha256,
            "annotation_input_plan_sha256": annotation_input_plan_sha256,
        }
        cohort_id = _canonical_sha256(identity_payload)
        advisory = {
            part_id: advisory_conflicts(part_id, group_id)
            for part_id in expected_member_part_ids
        }
        advisory = {
            part_id: reason_codes
            for part_id, reason_codes in advisory.items()
            if reason_codes
        }
        contract: dict[str, Any] = {
            **identity_payload,
            "cohort_id": cohort_id,
            "subtree_part_ids": list(candidate["subtree_part_ids"]),
            "propagated_member_part_ids": newly_propagated,
            "advisory_conflicts": advisory,
            "signature_dominance": {
                "part_share": candidate["signature_part_share"],
                "face_share": candidate["signature_face_share"],
            },
            "exact_cover": True,
            "material_identity_unchanged": True,
            "parameters_unchanged": True,
        }
        contract["contract_sha256"] = _canonical_sha256(contract)
        contracts.append(contract)
        contract_sha256 = str(contract["contract_sha256"])
        for part_id in expected_member_part_ids:
            assignment = assignments_by_part[part_id]
            provenance = assignment.get("provenance")
            mutable_provenance = (
                dict(provenance) if isinstance(provenance, Mapping) else {}
            )
            existing_group_id = mutable_provenance.get("canonical_group_id")
            role = (
                "anchor"
                if part_id in anchor_part_ids
                else "existing_member"
                if existing_group_id == group_id
                else "propagated_member"
            )
            lineage: dict[str, Any] = {
                "schema_version": SOURCE_APPEARANCE_COHORT_CONTRACT_SCHEMA_VERSION,
                "method": SOURCE_APPEARANCE_COHORT_METHOD,
                "cohort_id": cohort_id,
                "contract_sha256": contract_sha256,
                "canonical_group_id": group_id,
                "member_role": role,
                "anchor_part_ids": anchor_part_ids,
                "expected_member_part_ids": expected_member_part_ids,
                "propagated_member_part_ids": newly_propagated,
                "registry_sha256": registry_sha256,
                "spatial_report_sha256": spatial_report_sha256,
                "annotation_input_plan_sha256": annotation_input_plan_sha256,
                "exact_cover": True,
                "material_identity_unchanged": True,
                "parameters_unchanged": True,
            }
            if part_id in advisory:
                lineage["advisory_conflicts"] = advisory[part_id]
                lineage["superseded_provisional_group_id"] = (
                    mutable_provenance.get("provisional_canonical_group_id")
                )
            mutable_provenance["canonical_group_id"] = group_id
            mutable_provenance["source_appearance_cohort"] = lineage
            if role == "propagated_member":
                mutable_provenance["canonical_group_annotation"] = {
                    **lineage,
                    "whole_part_without_face_subsets": True,
                    "source_appearance_only_selects_membership": True,
                    "canonical_group_authority": "trusted_spatial_anchor",
                }
                record = records_by_part[part_id]
                record.update(
                    {
                        "outcome": "ANNOTATED",
                        "selected_group_id": group_id,
                        "annotation_confidence": 1.0,
                        "confidence_tier": (
                            "TRUSTED_SPATIAL_ANCHOR_SOURCE_APPEARANCE_COHORT"
                        ),
                        "reason_codes": [
                            "SOURCE_APPEARANCE_COHORT_CONTRACT_ACCEPTED"
                        ],
                        "candidates": [],
                        "source_appearance_cohort": copy.deepcopy(lineage),
                    }
                )
                propagated_member_ids.add(part_id)
            assignment["provenance"] = mutable_provenance

    coverage_blockers: list[dict[str, Any]] = []
    for contract in contracts:
        group_id = str(contract["canonical_group_id"])
        missing_part_ids = sorted(
            part_id
            for part_id in contract["expected_member_part_ids"]
            if assignments_by_part[part_id].get("provenance", {}).get(
                "canonical_group_id"
            )
            != group_id
        )
        if missing_part_ids:
            coverage_blockers.append(
                {
                    "cohort_id": contract["cohort_id"],
                    "canonical_group_id": group_id,
                    "reason": "SOURCE_APPEARANCE_COHORT_PLAN_COVERAGE_INCOMPLETE",
                    "missing_part_ids": missing_part_ids,
                }
            )
    if coverage_blockers:
        raise VisualGroupAnnotationError(
            "source-appearance cohort propagation failed exact coverage: "
            f"{coverage_blockers!r}"
        )

    return {
        **empty_audit,
        "enabled": True,
        "registry_sha256": registry_sha256,
        "spatial_report_sha256": spatial_report_sha256,
        "thresholds": {
            "minimum_member_count": SOURCE_APPEARANCE_COHORT_MIN_MEMBER_COUNT,
            "maximum_member_count": maximum_cohort_size,
            "maximum_absolute_member_count": (
                SOURCE_APPEARANCE_COHORT_MAX_MEMBER_COUNT
            ),
            "maximum_registry_fraction": (
                SOURCE_APPEARANCE_COHORT_MAX_REGISTRY_FRACTION
            ),
            "small_registry_allowance": (
                SOURCE_APPEARANCE_COHORT_SMALL_REGISTRY_ALLOWANCE
            ),
            "minimum_signature_part_share": (
                SOURCE_APPEARANCE_COHORT_MIN_SIGNATURE_PART_SHARE
            ),
            "minimum_signature_face_share": (
                SOURCE_APPEARANCE_COHORT_MIN_SIGNATURE_FACE_SHARE
            ),
            "rare_pair_global_member_count": (
                SOURCE_APPEARANCE_RARE_PAIR_SIZE
            ),
        },
        "cohort_identity": {
            "dominant_assembly_lane": {
                "mode": (
                    "path_free_appearance_plus_subset_layout"
                    if dominant_strong_hashes_required
                    else "legacy_stable_appearance_plus_no_subset_sentinel"
                ),
                "geometry_equality_required": False,
                "required_strong_hash_fields": list(
                    dominant_strong_hash_fields
                ),
                "strong_hash_part_count": len(
                    dominant_strong_hash_part_ids
                ),
                "strong_hash_part_ids": dominant_strong_hash_part_ids,
                "incomplete_strong_hash_part_ids": (
                    sorted(
                        set(registry_by_part)
                        - set(dominant_strong_hash_part_ids)
                    )
                    if dominant_strong_hashes_required
                    else []
                ),
                "whole_part_no_source_subsets_required": True,
            },
            "rare_repeated_sibling_lane": {
                "mode": (
                    "geometry_plus_appearance_plus_subset_layout"
                    if repeated_strong_hashes_required
                    else "legacy_geometry_proxy_plus_stable_appearance"
                ),
                "geometry_equality_required": True,
                "required_strong_hash_fields": list(
                    repeated_strong_hash_fields
                ),
                "strong_hash_part_count": len(
                    repeated_strong_hash_part_ids
                ),
                "strong_hash_part_ids": repeated_strong_hash_part_ids,
                "incomplete_strong_hash_part_ids": (
                    sorted(
                        set(registry_by_part)
                        - set(repeated_strong_hash_part_ids)
                    )
                    if repeated_strong_hashes_required
                    else []
                ),
                "whole_part_no_source_subsets_required": True,
            },
            "weak_and_strong_members_may_mix": False,
        },
        "cohort_count": len(contracts),
        "expected_member_count": len(
            {
                part_id
                for contract in contracts
                for part_id in contract["expected_member_part_ids"]
            }
        ),
        "propagated_member_count": len(propagated_member_ids),
        "contracts": sorted(
            contracts,
            key=lambda item: (
                str(item["canonical_group_id"]),
                str(item["assembly_path"]),
                str(item["cohort_id"]),
            ),
        ),
        "rejected_candidates": sorted(
            rejected_candidates,
            key=lambda item: (
                str(item.get("canonical_group_id", "")),
                str(item.get("assembly_path", "")),
                str(
                    item.get(
                        "source_appearance_cohort_signature_sha256",
                        "",
                    )
                ),
            ),
        ),
        "coverage_blockers": [],
        "exact_cover": True,
    }


def annotate_visual_groups(
    *,
    material_plan: Mapping[str, Any],
    palette_fusion: Mapping[str, Any],
    spatial_mapping_report: Mapping[str, Any] | None = None,
    part_registry: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return an annotated plan and a complete deterministic decision audit."""

    if material_plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise VisualGroupAnnotationError("material plan schema_version must be '1.0'")
    canonical_groups = _canonical_groups(palette_fusion)
    output = copy.deepcopy(dict(material_plan))
    assignments = _sequence(output.get("assignments"), "material_plan.assignments")
    seen_part_ids: set[str] = set()
    records: list[dict[str, Any]] = []
    for assignment_index, raw_assignment in enumerate(assignments):
        if not isinstance(raw_assignment, dict):
            raise VisualGroupAnnotationError(
                f"material_plan.assignments[{assignment_index}] must be an object"
            )
        part_id = _text(
            raw_assignment.get("part_id"),
            f"material_plan.assignments[{assignment_index}].part_id",
        )
        if part_id in seen_part_ids:
            raise VisualGroupAnnotationError(f"material plan repeats part_id {part_id}")
        seen_part_ids.add(part_id)
        assignment_label = f"assignment {part_id}"
        # A policy fallback exists precisely because visual localization was
        # unresolved.  Its output tokens cannot then become the missing
        # localization evidence.  Explicit canonical provenance still takes
        # precedence inside ``_annotate_entity`` and remains preserved.
        token_inference_allowed = (
            raw_assignment.get("status") != POLICY_FALLBACK_STATUS
        )
        records.append(
            _annotate_entity(
                raw_assignment,
                entity_kind="assignment",
                entity_label=assignment_label,
                part_id=part_id,
                subset_name=None,
                canonical_groups=canonical_groups,
                token_inference_allowed=token_inference_allowed,
            )
        )
        existing_subset_groups = _assignment_face_subset_group_ids(
            raw_assignment,
            assignment_label=assignment_label,
            canonical_groups=canonical_groups,
        )
        resolved_subset_groups: dict[str, str] = {}
        seen_subset_names: set[str] = set()
        raw_subsets = raw_assignment.get("face_subsets", [])
        for subset_index, raw_subset in enumerate(
            _sequence(raw_subsets, f"{assignment_label}.face_subsets")
        ):
            if not isinstance(raw_subset, dict):
                raise VisualGroupAnnotationError(
                    f"{assignment_label}.face_subsets[{subset_index}] must be an object"
                )
            subset_name = _text(
                raw_subset.get("subset_name"),
                f"{assignment_label}.face_subsets[{subset_index}].subset_name",
            )
            if subset_name in seen_subset_names:
                raise VisualGroupAnnotationError(
                    f"{assignment_label} repeats face subset {subset_name!r}"
                )
            seen_subset_names.add(subset_name)
            subset_label = f"{assignment_label}.face_subsets[{subset_index}]"
            migrated_group_id = _migrate_generated_face_subset_provenance(
                raw_subset,
                subset_label=subset_label,
                canonical_groups=canonical_groups,
            )
            mapped_group_id = existing_subset_groups.get(subset_name)
            if (
                mapped_group_id is not None
                and migrated_group_id is not None
                and mapped_group_id != migrated_group_id
            ):
                raise VisualGroupAnnotationError(
                    f"{assignment_label} face subset {subset_name!r} has "
                    "conflicting canonical-group provenance"
                )
            existing_group_id = mapped_group_id or migrated_group_id
            record = _annotate_entity(
                raw_subset,
                entity_kind="face_subset",
                entity_label=subset_label,
                part_id=part_id,
                subset_name=subset_name,
                canonical_groups=canonical_groups,
                existing_group_id_override=existing_group_id,
                write_entity_provenance=False,
                token_inference_allowed=token_inference_allowed,
            )
            records.append(record)
            selected_group_id = record["selected_group_id"]
            if isinstance(selected_group_id, str):
                resolved_subset_groups[subset_name] = selected_group_id
        unknown_mapped_subsets = sorted(set(existing_subset_groups) - seen_subset_names)
        if unknown_mapped_subsets:
            raise VisualGroupAnnotationError(
                f"{assignment_label}.provenance face-subset group map "
                f"references absent subsets: {unknown_mapped_subsets}"
            )
        if resolved_subset_groups:
            provenance = raw_assignment.get("provenance")
            mutable_provenance = (
                dict(provenance) if isinstance(provenance, Mapping) else {}
            )
            mutable_provenance["face_subset_canonical_group_ids"] = dict(
                sorted(resolved_subset_groups.items())
            )
            raw_assignment["provenance"] = mutable_provenance

    spatial_annotation_audit: dict[str, Any] = {
        "enabled": False,
        "method": SPATIAL_ANNOTATION_METHOD,
        "annotated_part_count": 0,
        "semantic_recovery_annotated_part_count": 0,
        "direct_multiview_recovery_annotated_part_count": 0,
        "single_view_recovery_annotated_part_count": 0,
        "direct_multiview_rejections": [],
        "single_view_rejections": [],
        "annotations": [],
    }
    if spatial_mapping_report is not None:
        spatial_annotation_audit = _apply_trusted_spatial_annotations(
            assignments=assignments,
            records=records,
            canonical_groups=canonical_groups,
            spatial_mapping_report=spatial_mapping_report,
        )
    source_appearance_cohort_audit = (
        _apply_source_appearance_cohort_annotations(
            assignments=assignments,
            records=records,
            part_registry=part_registry,
            spatial_mapping_report=spatial_mapping_report,
            spatial_annotation_audit=spatial_annotation_audit,
            annotation_input_plan_sha256=_canonical_sha256(material_plan),
        )
    )
    repeated_subset_visual_cohort_audit = (
        _apply_repeated_subset_visual_cohort_annotations(
            assignments=assignments,
            records=records,
            canonical_groups=canonical_groups,
            part_registry=part_registry,
            spatial_mapping_report=spatial_mapping_report,
            annotation_input_plan_sha256=_canonical_sha256(material_plan),
        )
    )

    counts: dict[str, int] = {}
    for record in records:
        key = f"{record['entity_kind']}_{record['outcome']}".casefold()
        counts[key] = counts.get(key, 0) + 1
    ambiguous_count = sum(1 for record in records if record["outcome"] == "AMBIGUOUS")
    unresolved_count = sum(1 for record in records if record["outcome"] == "UNRESOLVED")
    if ambiguous_count:
        status = "COMPLETED_FAIL_CLOSED_AMBIGUITY"
    elif unresolved_count:
        status = "COMPLETED_WITH_UNRESOLVED"
    else:
        status = "COMPLETED"

    plan_provenance = output.get("provenance")
    if plan_provenance is not None and not isinstance(plan_provenance, Mapping):
        raise VisualGroupAnnotationError("material_plan.provenance must be an object")
    mutable_plan_provenance = (
        dict(plan_provenance) if isinstance(plan_provenance, Mapping) else {}
    )
    mutable_plan_provenance["visual_group_annotation"] = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "method": ANNOTATION_METHOD,
        "source_plan_sha256": _canonical_sha256(material_plan),
        "palette_fusion_sha256": _canonical_sha256(palette_fusion),
        "existing_group_precedence": True,
        "face_subset_group_storage": (
            "assignment.provenance.face_subset_canonical_group_ids"
        ),
        "face_subset_schema_fields_unchanged": True,
        "material_identity_unchanged": True,
        "parameters_unchanged": True,
        "spatial_annotation": copy.deepcopy(spatial_annotation_audit),
        "source_appearance_cohort_propagation": copy.deepcopy(
            source_appearance_cohort_audit
        ),
        "repeated_subset_visual_cohort": copy.deepcopy(
            repeated_subset_visual_cohort_audit
        ),
        "counts": dict(sorted(counts.items())),
    }
    output["provenance"] = mutable_plan_provenance

    assignments_by_group: dict[str, list[str]] = {}
    face_subsets_by_group: dict[str, list[str]] = {}
    for record in records:
        group_id = record["selected_group_id"]
        if not isinstance(group_id, str):
            continue
        if record["entity_kind"] == "assignment":
            assignments_by_group.setdefault(group_id, []).append(record["part_id"])
        else:
            face_subsets_by_group.setdefault(group_id, []).append(
                f"{record['part_id']}:{record['subset_name']}"
            )
    audit = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "status": status,
        "source_plan_sha256": _canonical_sha256(material_plan),
        "annotated_plan_sha256": _canonical_sha256(output),
        "palette_fusion_sha256": _canonical_sha256(palette_fusion),
        "policy": {
            "method": ANNOTATION_METHOD,
            "minimum_canonical_confidence": MINIMUM_CANONICAL_CONFIDENCE,
            "minimum_exact_confidence": MINIMUM_EXACT_CONFIDENCE,
            "minimum_neighbor_confidence": MINIMUM_NEIGHBOR_CONFIDENCE,
            "maximum_neighbor_confidence": MAXIMUM_NEIGHBOR_CONFIDENCE,
            "minimum_selection_margin": MINIMUM_SELECTION_MARGIN,
            "low_confidence_color_neighbors": sorted(
                sorted(relation) for relation in _LOW_CONFIDENCE_COLOR_NEIGHBORS
            ),
            "existing_group_precedence": True,
            "face_subset_group_storage": (
                "assignment.provenance.face_subset_canonical_group_ids"
            ),
            "face_subset_schema_fields_unchanged": True,
            "material_identity_mutation_allowed": False,
            "parameter_mutation_allowed": False,
            "policy_fallback_token_inference_allowed": False,
            "trusted_multiview_spatial_projection_allowed": (
                spatial_mapping_report is not None
            ),
            "spatial_annotation_method": SPATIAL_ANNOTATION_METHOD,
            "semantic_spatial_annotation_method": (
                SEMANTIC_SPATIAL_ANNOTATION_METHOD
            ),
            "semantic_spatial_minimum_effective_confidence": (
                MINIMUM_CANONICAL_CONFIDENCE
            ),
            "semantic_spatial_visible_pixel_range": [
                MINIMUM_SEMANTIC_RECOVERY_VISIBLE_PIXELS,
                MAXIMUM_SEMANTIC_RECOVERY_VISIBLE_PIXELS,
            ],
            "direct_multiview_spatial_annotation_method": (
                MULTIVIEW_DIRECT_SPATIAL_ANNOTATION_METHOD
            ),
            "direct_multiview_minimum_agreement_views": (
                MINIMUM_DIRECT_MULTIVIEW_AGREEMENT_VIEWS
            ),
            "direct_multiview_minimum_strong_views": (
                MINIMUM_DIRECT_MULTIVIEW_STRONG_VIEWS
            ),
            "direct_multiview_minimum_projected_pixels": (
                MINIMUM_DIRECT_MULTIVIEW_PIXELS
            ),
            "direct_multiview_minimum_sampled_pixels": (
                MINIMUM_DIRECT_MULTIVIEW_PIXELS
            ),
            "direct_multiview_minimum_winning_color_share": (
                MINIMUM_DIRECT_MULTIVIEW_COLOR_SHARE
            ),
            "direct_multiview_minimum_color_margin": (
                MINIMUM_DIRECT_MULTIVIEW_COLOR_MARGIN
            ),
            "direct_multiview_minimum_semantic_conflict_confidence": (
                MINIMUM_DIRECT_MULTIVIEW_SEMANTIC_CONFLICT_CONFIDENCE
            ),
            "direct_multiview_minimum_sam3_foreground_pixels": (
                MINIMUM_SPATIAL_FOREGROUND_PIXELS
            ),
            "direct_multiview_minimum_sam3_foreground_overlap": (
                MINIMUM_SPATIAL_FOREGROUND_OVERLAP
            ),
            "direct_multiview_bbox_conflict_is_diagnostic_only": True,
            "single_view_spatial_annotation_method": (
                SINGLE_VIEW_SPATIAL_ANNOTATION_METHOD
            ),
            "single_view_spatial_minimum_projected_pixels": (
                MINIMUM_SINGLE_VIEW_SPATIAL_PIXELS
            ),
            "single_view_spatial_minimum_sampled_pixels": (
                MINIMUM_SINGLE_VIEW_SPATIAL_PIXELS
            ),
            "single_view_spatial_minimum_winning_color_share": (
                MINIMUM_SINGLE_VIEW_COLOR_SHARE
            ),
            "single_view_spatial_minimum_color_margin": (
                MINIMUM_SINGLE_VIEW_COLOR_MARGIN
            ),
            "single_view_spatial_minimum_bbox_color_margin": (
                MINIMUM_SINGLE_VIEW_BBOX_COLOR_MARGIN
            ),
            "single_view_spatial_minimum_sam3_foreground_pixels": (
                MINIMUM_SPATIAL_FOREGROUND_PIXELS
            ),
            "single_view_spatial_minimum_sam3_foreground_overlap": (
                MINIMUM_SPATIAL_FOREGROUND_OVERLAP
            ),
            "single_view_spatial_all_perturbation_groups_must_match": True,
            "single_view_spatial_semantic_conflict_override_allowed": (
                "only_after_all_high_confidence_spatial_gates_pass"
            ),
            "sam3_foreground_single_view_annotation_method": (
                FOREGROUND_SINGLE_VIEW_SPATIAL_ANNOTATION_METHOD
            ),
            "sam3_foreground_single_view_minimum_pixels": (
                MINIMUM_FOREGROUND_SINGLE_VIEW_PIXELS
            ),
            "sam3_foreground_single_view_minimum_overlap": (
                MINIMUM_SPATIAL_FOREGROUND_OVERLAP
            ),
            "sam3_foreground_single_view_minimum_winning_color_share": (
                MINIMUM_FOREGROUND_SINGLE_VIEW_COLOR_SHARE
            ),
            "sam3_foreground_single_view_minimum_color_margin": (
                MINIMUM_FOREGROUND_SINGLE_VIEW_COLOR_MARGIN
            ),
            "sam3_foreground_single_view_unseen_views_cast_no_vote": True,
            "sam3_foreground_single_view_visible_conflicts_veto": True,
            "source_appearance_cohort_method": (
                SOURCE_APPEARANCE_COHORT_METHOD
            ),
            "source_appearance_can_select_canonical_group": False,
            "source_appearance_requires_trusted_spatial_anchor": True,
            "source_appearance_authoritative_conflict_veto": True,
            "provisional_group_can_create_cohort_anchor": False,
            "repeated_subset_visual_cohort_method": (
                REPEATED_SUBSET_VISUAL_COHORT_METHOD
            ),
            "repeated_subset_visual_cohort_requires_exact_geometry_hash": True,
            "repeated_subset_visual_cohort_requires_exact_subset_layout_hash": (
                True
            ),
            "repeated_subset_visual_cohort_minimum_supporting_views": 2,
            "repeated_subset_visual_cohort_material_identity_mutation_allowed": (
                False
            ),
        },
        "canonical_groups": [
            dict(group) for _group_id, group in sorted(canonical_groups.items())
        ],
        "summary": {
            "assignment_count": len(assignments),
            "face_subset_count": sum(
                1 for record in records if record["entity_kind"] == "face_subset"
            ),
            "annotated_count": sum(
                1 for record in records if record["outcome"] == "ANNOTATED"
            ),
            "preserved_existing_count": sum(
                1 for record in records if record["outcome"] == "PRESERVED_EXISTING"
            ),
            "ambiguous_count": ambiguous_count,
            "unresolved_count": unresolved_count,
            "counts": dict(sorted(counts.items())),
            "assignment_part_ids_by_group": {
                group_id: sorted(part_ids)
                for group_id, part_ids in sorted(assignments_by_group.items())
            },
            "face_subsets_by_group": {
                group_id: sorted(subsets)
                for group_id, subsets in sorted(face_subsets_by_group.items())
            },
        },
        "records": records,
        "spatial_annotation": spatial_annotation_audit,
        "source_appearance_cohort_propagation": (
            source_appearance_cohort_audit
        ),
        "repeated_subset_visual_cohort": (
            repeated_subset_visual_cohort_audit
        ),
    }
    return output, audit


def _load_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VisualGroupAnnotationError(
            f"unable to read {label} {path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise VisualGroupAnnotationError(f"{label} must be a JSON object")
    return value


def _write_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Annotate a complete/restored material plan with conservative "
            "canonical visual-group provenance"
        )
    )
    parser.add_argument(
        "--plan",
        "--complete-plan",
        dest="plan",
        required=True,
        help="complete or restored material plan to annotate",
    )
    parser.add_argument("--palette-fusion", required=True)
    parser.add_argument(
        "--spatial-mapping-report",
        help="optional sealed spatial mapping report for trusted annotations",
    )
    parser.add_argument(
        "--part-registry",
        help=(
            "optional source/rendered part registry used for bounded "
            "source-appearance cohort propagation"
        ),
    )
    parser.add_argument("--output-plan", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument(
        "--require-unambiguous",
        action="store_true",
        help="return exit status 3 if any entity remains ambiguous",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        plan_path = Path(args.plan).expanduser().resolve(strict=True)
        fusion_path = Path(args.palette_fusion).expanduser().resolve(strict=True)
        output_path = Path(args.output_plan).expanduser().resolve()
        audit_path = Path(args.audit).expanduser().resolve()
        output, audit = annotate_visual_groups(
            material_plan=_load_object(plan_path, "material plan"),
            palette_fusion=_load_object(fusion_path, "palette fusion"),
            spatial_mapping_report=(
                _load_object(
                    Path(args.spatial_mapping_report).expanduser().resolve(
                        strict=True
                    ),
                    "spatial mapping report",
                )
                if args.spatial_mapping_report
                else None
            ),
            part_registry=(
                _load_object(
                    Path(args.part_registry).expanduser().resolve(strict=True),
                    "part registry",
                )
                if args.part_registry
                else None
            ),
        )
        audit["inputs"] = {
            "plan": str(plan_path),
            "plan_file_sha256": _sha256_file(plan_path),
            "palette_fusion": str(fusion_path),
            "palette_fusion_file_sha256": _sha256_file(fusion_path),
        }
        _write_atomic(output_path, output)
        _write_atomic(audit_path, audit)
        print(
            json.dumps(
                {
                    "status": audit["status"],
                    "output_plan": str(output_path),
                    "audit": str(audit_path),
                    **{
                        key: audit["summary"][key]
                        for key in (
                            "assignment_count",
                            "face_subset_count",
                            "annotated_count",
                            "preserved_existing_count",
                            "ambiguous_count",
                            "unresolved_count",
                        )
                    },
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        if args.require_unambiguous and audit["summary"]["ambiguous_count"]:
            return EXIT_REQUIRE_UNAMBIGUOUS_FAILED
        return EXIT_SUCCESS
    except (OSError, VisualGroupAnnotationError) as exc:
        print(
            json.dumps(
                {"status": "INPUT_ERROR", "error": str(exc)},
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return EXIT_INPUT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
