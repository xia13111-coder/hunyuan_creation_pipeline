"""Bounded MVInverse parameter adapters for selected NVIDIA MDL materials.

Material retrieval and parameter authoring are separate trust decisions:

* Qwen plus deterministic similarity chooses an exact whitelisted MDL export.
* MVInverse may tune that export only when its multi-view evidence says
  ``auto_parameter_eligible``.
* Only explicitly reviewed MDL interfaces below can receive authored inputs.

This preserves the selected library material and changes only its exposed
colour/roughness/metallic controls.  Texture, wear, normal, coating, and other
library characteristics remain at the selected preset defaults.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .parameters import srgb_to_linear


STEEL_PAINTED_MODULE_PREFIX = "mdl:vMaterials_2/Metal/Steel_Painted.mdl#"
POLYCARBONATE_OPAQUE_MODULE_PREFIX = (
    "mdl:vMaterials_2/Plastic/Polycarbonate_Opaque.mdl#"
)
THICK_PLASTIC_MODULE_PREFIX = (
    "mdl:vMaterials_2/Plastic/Plastic_Thick_Translucent.mdl#"
)
PLASTIC_ABS = "mdl:Plastics/Plastic_ABS.mdl#Plastic_ABS"
BASE_PAINT_MODULE_PREFIX = "mdl:Miscellaneous/Paint_"
BASE_METAL_MODULE_PREFIX = "mdl:Metals/"
_BASE_TUNABLE_DIELECTRICS = frozenset(
    {
        PLASTIC_ABS,
        "mdl:Plastics/Plastic.mdl#Plastic",
        "mdl:Plastics/Rubber_Smooth.mdl#Rubber_Smooth",
        "mdl:Plastics/Rubber_Textured.mdl#Rubber_Textured",
        "mdl:Plastics/Veneer_OU_Walnut.mdl#Veneer_OU_Walnut",
        "mdl:Plastics/Veneer_UX_Walnut_Cherry.mdl#Veneer_UX_Walnut_Cherry",
        "mdl:Plastics/Veneer_Z5_Maple.mdl#Veneer_Z5_Maple",
        "mdl:Plastics/Vinyl.mdl#Vinyl",
    }
)
_BASE_UNTEXTURED_TUNABLE_DIELECTRICS = frozenset(
    {
        PLASTIC_ABS,
        "mdl:Plastics/Plastic.mdl#Plastic",
        "mdl:Plastics/Vinyl.mdl#Vinyl",
    }
)


@dataclass(frozen=True)
class MaterialTuningProfile:
    profile_id: str
    material_prefix: str
    surface_class: str
    color_parameters: tuple[str, ...]
    roughness_parameter: str
    metallic_parameter: str | None = None
    normalize_chromatic_diffuse_tint: bool = False


_PROFILES = (
    MaterialTuningProfile(
        profile_id="nvidia_base_paint_omnipbr",
        material_prefix=BASE_PAINT_MODULE_PREFIX,
        surface_class="dielectric",
        color_parameters=("diffuse_tint",),
        roughness_parameter="reflection_roughness_constant",
        normalize_chromatic_diffuse_tint=True,
    ),
    MaterialTuningProfile(
        profile_id="nvidia_base_metal_omnipbr",
        material_prefix=BASE_METAL_MODULE_PREFIX,
        surface_class="metal",
        color_parameters=("diffuse_tint",),
        roughness_parameter="reflection_roughness_constant",
        metallic_parameter="metallic_constant",
        normalize_chromatic_diffuse_tint=True,
    ),
    MaterialTuningProfile(
        profile_id="nvidia_steel_painted",
        material_prefix=STEEL_PAINTED_MODULE_PREFIX,
        surface_class="dielectric",
        color_parameters=("paint_color",),
        roughness_parameter="paint_roughness",
    ),
    MaterialTuningProfile(
        profile_id="nvidia_polycarbonate_opaque",
        material_prefix=POLYCARBONATE_OPAQUE_MODULE_PREFIX,
        surface_class="dielectric",
        color_parameters=("diffuse_tint",),
        roughness_parameter="roughness",
    ),
    MaterialTuningProfile(
        profile_id="nvidia_thick_plastic",
        material_prefix=THICK_PLASTIC_MODULE_PREFIX,
        surface_class="dielectric",
        color_parameters=("diffuse_color", "transmissive_color"),
        roughness_parameter="reflection_roughness",
    ),
)

_BASE_TEXTURED_DIELECTRIC_PROFILE = MaterialTuningProfile(
    profile_id="nvidia_base_dielectric_omnipbr",
    material_prefix="",
    surface_class="dielectric",
    color_parameters=("diffuse_tint",),
    roughness_parameter="reflection_roughness_constant",
    metallic_parameter="metallic_constant",
    normalize_chromatic_diffuse_tint=True,
)

_BASE_UNTEXTURED_DIELECTRIC_PROFILE = MaterialTuningProfile(
    profile_id="nvidia_base_untextured_dielectric_omnipbr",
    material_prefix="",
    surface_class="dielectric",
    color_parameters=("diffuse_tint",),
    roughness_parameter="reflection_roughness_constant",
    metallic_parameter="metallic_constant",
)


def tuning_profile_for_material(
    material_id: str,
) -> MaterialTuningProfile | None:
    if not isinstance(material_id, str):
        return None
    # Catalog IDs are relative to the configured root.  Scanning
    # ``NVIDIA/Materials`` yields ``mdl:Base/...`` while scanning its Base
    # child yields the historical ``mdl:...`` IDs.  Both identify the same
    # reviewed interfaces and must receive the same bounded parameter policy.
    normalized = (
        "mdl:" + material_id[len("mdl:Base/") :]
        if material_id.startswith("mdl:Base/")
        else material_id
    )
    if normalized in _BASE_UNTEXTURED_TUNABLE_DIELECTRICS:
        return _BASE_UNTEXTURED_DIELECTRIC_PROFILE
    if normalized in _BASE_TUNABLE_DIELECTRICS:
        return _BASE_TEXTURED_DIELECTRIC_PROFILE
    for profile in _PROFILES:
        if normalized.startswith(profile.material_prefix):
            return profile
    return None


def parameter_policy_for_material(
    material_id: str,
) -> dict[str, tuple[str, float | None, float | None]]:
    """Return the only shader inputs the pipeline may author for a material."""

    profile = tuning_profile_for_material(material_id)
    if profile is None:
        return {}
    policy: dict[str, tuple[str, float | None, float | None]] = {
        name: ("color3f_linear", 0.0, 1.0)
        for name in profile.color_parameters
    }
    policy[profile.roughness_parameter] = ("float", 0.0, 1.0)
    if profile.metallic_parameter is not None:
        policy[profile.metallic_parameter] = ("float", 0.0, 1.0)
    return policy


def _unit(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise ValueError(f"{label} must be a finite number from 0 to 1")
    return float(value)


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def color_parameters_for_target_srgb(
    profile: MaterialTuningProfile,
    color_srgb: Sequence[float],
) -> tuple[dict[str, list[float]], dict[str, Any]]:
    """Translate a target colour into the selected MDL input semantics.

    A reviewed profile states whether ``diffuse_tint`` multiplies an active
    base-colour texture or is the material's absolute colour.  Textured
    profiles preserve hue and saturation by normalizing the strongest linear
    channel to one.  Untextured OmniPBR materials and direct-colour interfaces
    retain the absolute linear target; otherwise dark green/orange/blue
    evidence becomes an over-bright tint.  Neutral targets always retain their
    absolute value so black, grey and white evidence is not collapsed.
    """

    validated_srgb = [
        _unit(value, f"target color_srgb[{index}]")
        for index, value in enumerate(color_srgb)
    ]
    color_linear = srgb_to_linear(validated_srgb)
    maximum_srgb = max(validated_srgb)
    saturation = (
        (maximum_srgb - min(validated_srgb)) / maximum_srgb
        if maximum_srgb > 1e-8
        else 0.0
    )
    authored_color = list(color_linear)
    semantics = "absolute_linear_color"
    if (
        profile.color_parameters == ("diffuse_tint",)
        and profile.normalize_chromatic_diffuse_tint
        and saturation >= 0.15
    ):
        peak = max(color_linear)
        if peak > 1e-8:
            authored_color = [
                min(1.0, max(0.0, float(value) / float(peak)))
                for value in color_linear
            ]
            semantics = "chromatic_texture_multiplier_normalized"
    parameters = {
        name: list(authored_color) for name in profile.color_parameters
    }
    return parameters, {
        "base_color_srgb": validated_srgb,
        "base_color_linear": color_linear,
        "authored_color_linear": authored_color,
        "color_parameter_semantics": semantics,
        "target_saturation": saturation,
    }


def tune_selected_material_from_mvinverse(
    evidence_group: Mapping[str, Any],
    *,
    group_id: str,
    material_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a minimal parameter delta for one already-selected MDL export."""

    profile = tuning_profile_for_material(material_id)
    if profile is None:
        raise ValueError(f"selected MDL material has no tuning profile: {material_id}")
    if evidence_group.get("group_id") != group_id:
        raise ValueError(f"MVInverse evidence group does not match {group_id}")
    suggestion = evidence_group.get("suggestion")
    if not isinstance(suggestion, Mapping):
        raise ValueError(f"MVInverse group {group_id} has no suggestion")
    if (
        suggestion.get("decision") != "auto"
        or suggestion.get("auto_parameter_eligible") is not True
    ):
        raise ValueError(
            f"MVInverse group {group_id} is not eligible for automatic parameters"
        )
    if evidence_group.get("surface_class") != profile.surface_class:
        raise ValueError(
            f"MVInverse group {group_id} surface class is incompatible with "
            f"{profile.profile_id}"
        )

    raw_color = suggestion.get("base_color_srgb")
    if (
        not isinstance(raw_color, Sequence)
        or isinstance(raw_color, (str, bytes))
        or len(raw_color) != 3
    ):
        raise ValueError(
            f"MVInverse group {group_id} base_color_srgb must have three channels"
        )
    color_srgb = [
        _unit(value, f"MVInverse group {group_id}.base_color_srgb[{index}]")
        for index, value in enumerate(raw_color)
    ]
    color_parameters, color_audit = color_parameters_for_target_srgb(
        profile,
        color_srgb,
    )
    color_linear = color_audit["base_color_linear"]
    authored_metallic = _unit(
        suggestion.get("metallic"), f"MVInverse group {group_id}.metallic"
    )
    roughness = _unit(
        suggestion.get("roughness"), f"MVInverse group {group_id}.roughness"
    )
    if profile.surface_class == "dielectric" and authored_metallic != 0.0:
        raise ValueError(
            f"dielectric material {material_id} cannot author nonzero metallic"
        )

    raw_views = evidence_group.get("contributing_view_ids")
    if not isinstance(raw_views, Sequence) or isinstance(raw_views, (str, bytes)):
        raise ValueError(
            f"MVInverse group {group_id}.contributing_view_ids must be an array"
        )
    contributing_view_ids = [
        _text(value, f"MVInverse group {group_id}.contributing_view_ids")
        for value in raw_views
    ]
    if len(contributing_view_ids) < 2 or len(set(contributing_view_ids)) != len(
        contributing_view_ids
    ):
        raise ValueError(
            f"MVInverse group {group_id} needs distinct multi-view evidence"
        )

    metallic_stats = evidence_group.get("metallic")
    if not isinstance(metallic_stats, Mapping):
        raise ValueError(f"MVInverse group {group_id} has no metallic statistics")
    observed_metallic = _unit(
        metallic_stats.get("median"),
        f"MVInverse group {group_id}.metallic.median",
    )

    parameters: dict[str, Any] = dict(color_parameters)
    parameters[profile.roughness_parameter] = roughness
    if profile.metallic_parameter is not None:
        parameters[profile.metallic_parameter] = authored_metallic

    audit = {
        "group_id": group_id,
        "material_id": material_id,
        "tuning_profile_id": profile.profile_id,
        "parameterization_mode": "full_mvinverse_pbr",
        "contributing_view_ids": sorted(contributing_view_ids),
        "base_color_srgb": color_srgb,
        "base_color_linear": color_linear,
        "authored_color_linear": color_audit["authored_color_linear"],
        "color_parameter_semantics": color_audit["color_parameter_semantics"],
        "observed_metallic": observed_metallic,
        "authored_metallic": authored_metallic,
        "roughness": roughness,
        "authored_parameter_names": sorted(parameters),
        "parameters": parameters,
    }
    return parameters, audit


def tune_selected_material_color_from_mvinverse(
    evidence_group: Mapping[str, Any],
    *,
    group_id: str,
    material_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Author colour only from one MVInverse view corroborated elsewhere.

    The caller must independently establish repeated cross-view palette-colour
    agreement.  This adapter accepts only the narrow MVInverse
    ``insufficient_distinct_views`` outcome and deliberately leaves the
    selected NVIDIA preset's roughness and metallic controls untouched.
    """

    profile = tuning_profile_for_material(material_id)
    if profile is None:
        raise ValueError(f"selected MDL material has no tuning profile: {material_id}")
    if evidence_group.get("group_id") != group_id:
        raise ValueError(f"MVInverse evidence group does not match {group_id}")
    if evidence_group.get("surface_class") != profile.surface_class:
        raise ValueError(
            f"MVInverse group {group_id} surface class is incompatible with "
            f"{profile.profile_id}"
        )
    suggestion = evidence_group.get("suggestion")
    if not isinstance(suggestion, Mapping):
        raise ValueError(f"MVInverse group {group_id} has no suggestion")
    reason_codes = suggestion.get("reason_codes")
    if (
        suggestion.get("decision") != "preserve"
        or suggestion.get("auto_parameter_eligible") is not False
        or not isinstance(reason_codes, Sequence)
        or isinstance(reason_codes, (str, bytes))
        or set(reason_codes) != {"insufficient_distinct_views"}
    ):
        raise ValueError(
            f"MVInverse group {group_id} is not eligible for corroborated color"
        )
    raw_views = evidence_group.get("contributing_view_ids")
    if (
        not isinstance(raw_views, Sequence)
        or isinstance(raw_views, (str, bytes))
        or len(raw_views) != 1
    ):
        raise ValueError(
            f"MVInverse group {group_id} needs exactly one contributing color view"
        )
    contributing_view_ids = [
        _text(value, f"MVInverse group {group_id}.contributing_view_ids")
        for value in raw_views
    ]
    albedo = evidence_group.get("albedo")
    raw_color = albedo.get("median") if isinstance(albedo, Mapping) else None
    if (
        not isinstance(raw_color, Sequence)
        or isinstance(raw_color, (str, bytes))
        or len(raw_color) != 3
    ):
        raise ValueError(f"MVInverse group {group_id} has no valid albedo median")
    color_srgb = [
        _unit(value, f"MVInverse group {group_id}.albedo.median[{index}]")
        for index, value in enumerate(raw_color)
    ]
    parameters, color_audit = color_parameters_for_target_srgb(
        profile,
        color_srgb,
    )
    color_linear = color_audit["base_color_linear"]
    metallic_stats = evidence_group.get("metallic")
    roughness_stats = evidence_group.get("roughness")
    observed_metallic = (
        _unit(
            metallic_stats.get("median"),
            f"MVInverse group {group_id}.metallic.median",
        )
        if isinstance(metallic_stats, Mapping)
        else None
    )
    observed_roughness = (
        _unit(
            roughness_stats.get("median"),
            f"MVInverse group {group_id}.roughness.median",
        )
        if isinstance(roughness_stats, Mapping)
        else None
    )
    audit = {
        "group_id": group_id,
        "material_id": material_id,
        "tuning_profile_id": profile.profile_id,
        "parameterization_mode": "multiview_palette_corroborated_color_only",
        "contributing_view_ids": contributing_view_ids,
        "base_color_srgb": color_srgb,
        "base_color_linear": color_linear,
        "authored_color_linear": color_audit["authored_color_linear"],
        "color_parameter_semantics": color_audit["color_parameter_semantics"],
        "observed_metallic": observed_metallic,
        "authored_metallic": None,
        "roughness": observed_roughness,
        "authored_parameter_names": sorted(parameters),
        "parameters": parameters,
    }
    return parameters, audit


__all__ = [
    "BASE_METAL_MODULE_PREFIX",
    "BASE_PAINT_MODULE_PREFIX",
    "PLASTIC_ABS",
    "POLYCARBONATE_OPAQUE_MODULE_PREFIX",
    "STEEL_PAINTED_MODULE_PREFIX",
    "THICK_PLASTIC_MODULE_PREFIX",
    "MaterialTuningProfile",
    "color_parameters_for_target_srgb",
    "parameter_policy_for_material",
    "tune_selected_material_color_from_mvinverse",
    "tune_selected_material_from_mvinverse",
    "tuning_profile_for_material",
]
