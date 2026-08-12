"""Shared physical-surface and immutable-catalog semantics.

The material pipeline has two independent responsibilities:

* ``substrate`` describes the bulk material fact that downstream physics,
  BOM, or ERP stages may consume (this module does not author physics);
* ``surface_treatment`` and ``optical_behavior`` describe the visible layer
  that an appearance MDL must reproduce.

Keeping those concepts separate prevents a dielectric paint layer from being
mistaken for plastic, or a green glass preset from being used as a visual
substitute for painted steel.  The helpers in this module are deliberately
dependency-free so catalog building, inference, validation, and tests all use
the same vocabulary.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import math
from typing import Any


PART_MATERIAL_SEMANTICS_SCHEMA_VERSION = "qwen-part-material-semantics/v1"
CATALOG_SURFACE_SEMANTICS_SCHEMA_VERSION = (
    "qwen-catalog-surface-semantics/v1"
)

SUBSTRATES = frozenset(
    {
        "metal",
        "polymer",
        "elastomer",
        "glass",
        "ceramic",
        "textile",
        "leather",
        "wood",
        "stone",
        "paper",
        "mineral",
        "liquid",
        "unknown",
    }
)
SURFACE_TREATMENTS = frozenset(
    {
        "bare",
        "paint",
        "powder_coat",
        "anodized",
        "plated",
        "galvanized",
        "oxidized",
        "conversion_coating",
        "rubber_overmold",
        "transparent_coating",
        "emissive",
        "unknown",
    }
)
OPTICAL_BEHAVIORS = frozenset(
    {"opaque", "translucent", "transparent", "emissive", "unknown"}
)
FINISH_CLASSES = frozenset(
    {
        "matte",
        "satin",
        "glossy",
        "polished",
        "brushed",
        "rough",
        "smooth",
        "textured",
        "weathered",
        "unknown",
    }
)
EVIDENCE_STATUSES = frozenset(
    {"observed", "ambiguous", "unobserved", "unknown"}
)
PHYSICAL_SOURCES = frozenset(
    {
        "step_bom",
        "erp",
        "part_standard",
        "cad_metadata",
        "vision_inference",
        "policy_fallback",
        "unknown",
    }
)
MATERIAL_SELECTION_STATUSES = frozenset(
    {
        "selected",
        "catalog_gap",
        "unknown",
        "unobserved",
        "safe_fallback_unverified",
    }
)

LEGACY_PHYSICAL_SURFACE_CLASSES = frozenset(
    {
        "bare_metal",
        "painted_metal",
        "plastic",
        "rubber",
        "glass",
        "ceramic",
        "textile",
        "wood",
        "stone",
        "paper",
        "emissive",
        "other_dielectric",
    }
)

_LEGACY_TO_SEMANTICS: dict[str, tuple[str, str, str]] = {
    "bare_metal": ("metal", "bare", "opaque"),
    "painted_metal": ("metal", "paint", "opaque"),
    "plastic": ("polymer", "bare", "opaque"),
    "rubber": ("elastomer", "bare", "opaque"),
    "glass": ("glass", "bare", "transparent"),
    "ceramic": ("ceramic", "bare", "opaque"),
    "textile": ("textile", "bare", "opaque"),
    "wood": ("wood", "bare", "opaque"),
    "stone": ("stone", "bare", "opaque"),
    "paper": ("paper", "bare", "opaque"),
    "emissive": ("unknown", "emissive", "emissive"),
    "other_dielectric": ("unknown", "unknown", "unknown"),
}


class MaterialSemanticsError(ValueError):
    """Raised when physical or catalog semantics are malformed."""


def _choice(value: Any, allowed: frozenset[str], label: str) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise MaterialSemanticsError(
            f"{label} must be one of {sorted(allowed)}, got {value!r}"
        )
    return value


def _confidence(value: Any) -> float | None:
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise MaterialSemanticsError("confidence must be a finite number in [0, 1]")
    return float(value)


def semantics_from_legacy_surface(
    physical_surface_class: str,
    *,
    confidence: float | None = None,
    physical_source: str = "vision_inference",
    evidence_status: str = "observed",
) -> dict[str, Any]:
    """Convert the v1 flat physical label into the hierarchical vocabulary."""

    if physical_surface_class not in LEGACY_PHYSICAL_SURFACE_CLASSES:
        raise MaterialSemanticsError(
            f"unsupported legacy physical surface class: {physical_surface_class!r}"
        )
    substrate, treatment, optical = _LEGACY_TO_SEMANTICS[physical_surface_class]
    if physical_surface_class == "other_dielectric":
        evidence_status = "unknown"
    return normalize_part_material_semantics(
        {
            "schema_version": PART_MATERIAL_SEMANTICS_SCHEMA_VERSION,
            "substrate": substrate,
            "surface_treatment": treatment,
            "optical_behavior": optical,
            "finish": "unknown",
            "physical_source": physical_source,
            "evidence_status": evidence_status,
            "confidence": confidence,
            "legacy_physical_surface_class": physical_surface_class,
        }
    )


def normalize_part_material_semantics(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and canonicalize one Part-ID physical material description."""

    if not isinstance(value, Mapping):
        raise MaterialSemanticsError("part material semantics must be an object")
    schema = value.get("schema_version", PART_MATERIAL_SEMANTICS_SCHEMA_VERSION)
    if schema != PART_MATERIAL_SEMANTICS_SCHEMA_VERSION:
        raise MaterialSemanticsError(f"unsupported part semantics schema: {schema!r}")
    substrate = _choice(value.get("substrate"), SUBSTRATES, "substrate")
    treatment = _choice(
        value.get("surface_treatment"),
        SURFACE_TREATMENTS,
        "surface_treatment",
    )
    optical = _choice(
        value.get("optical_behavior"),
        OPTICAL_BEHAVIORS,
        "optical_behavior",
    )
    finish = _choice(value.get("finish", "unknown"), FINISH_CLASSES, "finish")
    physical_source = _choice(
        value.get("physical_source", "unknown"),
        PHYSICAL_SOURCES,
        "physical_source",
    )
    evidence_status = _choice(
        value.get("evidence_status", "unknown"),
        EVIDENCE_STATUSES,
        "evidence_status",
    )
    confidence = _confidence(value.get("confidence"))
    legacy = value.get("legacy_physical_surface_class")
    if legacy is not None and legacy not in LEGACY_PHYSICAL_SURFACE_CLASSES:
        raise MaterialSemanticsError(
            f"unsupported legacy physical surface class: {legacy!r}"
        )
    if (treatment == "emissive") != (optical == "emissive"):
        raise MaterialSemanticsError(
            "emissive surface_treatment and optical_behavior must agree"
        )
    if (
        substrate == "metal"
        and treatment
        in {
            "bare",
            "anodized",
            "plated",
            "galvanized",
            "oxidized",
            "conversion_coating",
        }
        and optical in {"transparent", "translucent"}
    ):
        raise MaterialSemanticsError(
            "an exposed metallic surface cannot be transparent or translucent"
        )
    if evidence_status == "observed":
        if treatment == "unknown" or optical == "unknown":
            raise MaterialSemanticsError(
                "observed evidence cannot contain an unknown visible surface"
            )
        if substrate == "unknown" and treatment != "emissive":
            raise MaterialSemanticsError(
                "observed evidence requires a substrate unless it is emissive"
            )
    return {
        "schema_version": PART_MATERIAL_SEMANTICS_SCHEMA_VERSION,
        "substrate": substrate,
        "surface_treatment": treatment,
        "optical_behavior": optical,
        "finish": finish,
        "physical_source": physical_source,
        "evidence_status": evidence_status,
        "confidence": confidence,
        "legacy_physical_surface_class": legacy,
    }


def legacy_surface_from_semantics(value: Mapping[str, Any]) -> str | None:
    """Return a compatibility label when the hierarchy has an exact v1 form."""

    semantics = normalize_part_material_semantics(value)
    key = (
        semantics["substrate"],
        semantics["surface_treatment"],
        semantics["optical_behavior"],
    )
    inverse = {triple: label for label, triple in _LEGACY_TO_SEMANTICS.items()}
    return inverse.get(key)


def _coarse_finish(tokens: set[str]) -> str:
    for candidate in (
        "matte",
        "satin",
        "glossy",
        "polished",
        "brushed",
        "rough",
        "smooth",
    ):
        if candidate in tokens:
            return candidate
    if tokens & {"textured", "pattern", "woven", "berber", "linen"}:
        return "textured"
    if tokens & {"rust", "rusted", "rusty", "worn", "weathered", "dirty"}:
        return "weathered"
    return "unknown"


def infer_catalog_surface_semantics(
    *,
    family: str,
    tokens: Iterable[str],
) -> dict[str, Any]:
    """Infer auditable surface compatibility from NVIDIA path/name metadata.

    NVIDIA Base does not publish a machine-readable substrate taxonomy.  This
    function therefore records the inference source and confidence instead of
    presenting filename heuristics as measured physical truth.
    """

    normalized_family = str(family or "").casefold()
    normalized_tokens = {str(token).casefold() for token in tokens}
    compatible_substrates: tuple[str, ...] = ("unknown",)
    treatment = "unknown"
    optical = "opaque"
    confidence = "low"

    if normalized_family == "paint":
        compatible_substrates = ("metal", "polymer", "wood")
        treatment = "paint"
        confidence = "high"
    elif normalized_family == "metal":
        compatible_substrates = ("metal",)
        confidence = "high"
        if "anodized" in normalized_tokens:
            treatment = "anodized"
        elif normalized_tokens & {"galvanized", "zinc"}:
            treatment = "galvanized"
        elif normalized_tokens & {"rust", "rusted", "rusty"}:
            treatment = "oxidized"
        elif normalized_tokens & {"blued", "blueing", "blackened"}:
            treatment = "conversion_coating"
        elif normalized_tokens & {"chrome", "chromed", "plated"}:
            treatment = "plated"
        elif normalized_tokens & {"paint", "painted", "coated", "coating"}:
            treatment = "paint"
        else:
            treatment = "bare"
    elif normalized_family == "plastic":
        confidence = "high"
        treatment = "bare"
        if "rubber" in normalized_tokens:
            compatible_substrates = ("elastomer",)
        elif "veneer" in normalized_tokens:
            compatible_substrates = ("wood",)
        else:
            compatible_substrates = ("polymer",)
        if normalized_tokens & {"clear", "transparent"}:
            optical = "transparent"
    elif normalized_family == "rubber":
        compatible_substrates = ("elastomer",)
        treatment = "bare"
        confidence = "high"
    elif normalized_family == "glass":
        compatible_substrates = ("glass",)
        treatment = "bare"
        confidence = "high"
        if "mirror" in normalized_tokens:
            optical = "opaque"
        elif normalized_tokens & {"frosted", "dull"}:
            optical = "translucent"
        else:
            optical = "transparent"
    elif normalized_family in {"carpet", "fabric", "textile", "textiles"}:
        compatible_substrates = ("textile",)
        treatment = "bare"
        confidence = "high"
    elif normalized_family == "leather":
        compatible_substrates = ("leather",)
        treatment = "bare"
        confidence = "high"
    elif normalized_family == "wood":
        compatible_substrates = ("wood",)
        treatment = "bare"
        confidence = "high"
    elif normalized_family in {"stone", "masonry", "plaster", "concrete"}:
        compatible_substrates = (
            ("ceramic",)
            if normalized_tokens & {"ceramic", "porcelain", "terracotta"}
            else ("stone", "mineral")
        )
        treatment = "bare"
        confidence = "medium"
    elif normalized_family in {"paper", "wall_board"}:
        compatible_substrates = (
            ("wood",)
            if normalized_tokens & {"mdf", "fiberboard"}
            else ("paper",)
        )
        treatment = "bare"
        confidence = "medium"
    elif normalized_family in {"liquid", "liquids"} or "water" in normalized_tokens:
        compatible_substrates = ("liquid",)
        treatment = "bare"
        optical = "transparent"
        confidence = "medium"
    elif normalized_family in {"emissive", "emissives"}:
        compatible_substrates = ("unknown",)
        treatment = "emissive"
        optical = "emissive"
        confidence = "high"

    return normalize_catalog_surface_semantics(
        {
            "schema_version": CATALOG_SURFACE_SEMANTICS_SCHEMA_VERSION,
            "compatible_substrates": list(compatible_substrates),
            "surface_treatment": treatment,
            "optical_behavior": optical,
            "finish": _coarse_finish(normalized_tokens),
            "inference_source": "nvidia_path_name_and_authored_defaults/v1",
            "confidence": confidence,
        }
    )


def normalize_catalog_surface_semantics(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one catalog material's immutable visible-surface metadata."""

    if not isinstance(value, Mapping):
        raise MaterialSemanticsError("catalog surface semantics must be an object")
    schema = value.get("schema_version")
    if schema != CATALOG_SURFACE_SEMANTICS_SCHEMA_VERSION:
        raise MaterialSemanticsError(
            f"unsupported catalog surface semantics schema: {schema!r}"
        )
    raw_substrates = value.get("compatible_substrates")
    if (
        not isinstance(raw_substrates, list)
        or not raw_substrates
        or any(item not in SUBSTRATES for item in raw_substrates)
    ):
        raise MaterialSemanticsError(
            "compatible_substrates must be a non-empty list of known substrates"
        )
    source = value.get("inference_source")
    if not isinstance(source, str) or not source:
        raise MaterialSemanticsError("catalog inference_source must be non-empty")
    confidence = value.get("confidence")
    if confidence not in {"high", "medium", "low"}:
        raise MaterialSemanticsError(
            "catalog semantic confidence must be high, medium, or low"
        )
    treatment = _choice(
        value.get("surface_treatment"),
        SURFACE_TREATMENTS,
        "catalog surface_treatment",
    )
    optical = _choice(
        value.get("optical_behavior"),
        OPTICAL_BEHAVIORS,
        "catalog optical_behavior",
    )
    if (treatment == "emissive") != (optical == "emissive"):
        raise MaterialSemanticsError(
            "catalog emissive surface_treatment and optical_behavior must agree"
        )
    if (
        "metal" in raw_substrates
        and len(raw_substrates) == 1
        and treatment
        in {
            "bare",
            "anodized",
            "plated",
            "galvanized",
            "oxidized",
            "conversion_coating",
        }
        and optical in {"transparent", "translucent"}
    ):
        raise MaterialSemanticsError(
            "catalog exposed metallic surfaces cannot be transparent"
        )
    return {
        "schema_version": CATALOG_SURFACE_SEMANTICS_SCHEMA_VERSION,
        "compatible_substrates": sorted(set(raw_substrates)),
        "surface_treatment": treatment,
        "optical_behavior": optical,
        "finish": _choice(
            value.get("finish", "unknown"), FINISH_CLASSES, "catalog finish"
        ),
        "inference_source": source,
        "confidence": confidence,
    }


def catalog_matches_part_semantics(
    catalog_semantics: Mapping[str, Any],
    part_semantics: Mapping[str, Any],
) -> bool:
    """Return whether an immutable catalog surface satisfies a Part-ID layer."""

    candidate = normalize_catalog_surface_semantics(catalog_semantics)
    target = normalize_part_material_semantics(part_semantics)
    if target["evidence_status"] in {"ambiguous", "unobserved", "unknown"}:
        return False
    if target["surface_treatment"] == "emissive":
        return (
            candidate["surface_treatment"] == "emissive"
            and candidate["optical_behavior"] == "emissive"
        )
    if target["substrate"] == "unknown" or target["surface_treatment"] == "unknown":
        return False
    if target["substrate"] not in candidate["compatible_substrates"]:
        return False
    if candidate["surface_treatment"] != target["surface_treatment"]:
        return False
    target_optical = target["optical_behavior"]
    if (
        target_optical != "unknown"
        and candidate["optical_behavior"] != target_optical
    ):
        return False
    return True


def validate_selection_status(value: Any) -> str:
    """Validate the final closed/open-set material selection outcome."""

    return _choice(value, MATERIAL_SELECTION_STATUSES, "selection_status")
