"""Deterministic NVIDIA MDL appearance profiles and MVInverse similarity.

Qwen remains the final bounded classifier, but it should only see candidates
that are plausible according to the library source and the inverse-rendering
evidence.  This module extracts a small, auditable PBR profile from exported
MDL presets without evaluating MDL:

* the first recognized surface-colour parameter;
* the first recognized roughness parameter; and
* the first recognized metallic parameter.

The parser is deliberately conservative.  Unknown material interfaces simply
produce no numeric profile and continue to be ranked by catalog semantics.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?f?"
_EXPORT_RE_TEMPLATE = r"\bexport\s+material\s+{identifier}\s*\("
_NEXT_EXPORT_RE = re.compile(r"\bexport\s+material\s+[A-Za-z_][A-Za-z0-9_]*\s*\(")
_COLOR_PARAMETER_PRIORITY = (
    "paint_color",
    "diffuse_tint",
    "diffuse_color",
    "diffuse_color_constant",
    "base_color",
    "albedo",
    "tint",
    "color_1",
)
_ROUGHNESS_PARAMETER_PRIORITY = (
    "paint_roughness",
    "reflection_roughness_constant",
    "reflection_roughness",
    "roughness",
    "anodization_roughness",
)
_METALLIC_PARAMETER_PRIORITY = (
    "metallic_constant",
    "metallic",
    "metalness",
)


def _unit(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) and 0.0 <= number <= 1.0 else None


def _linear_channel_to_srgb(channel: float) -> float:
    value = max(0.0, min(1.0, float(channel)))
    if value <= 0.0031308:
        return 12.92 * value
    return 1.055 * (value ** (1.0 / 2.4)) - 0.055


def _srgb_channel_to_linear(channel: float) -> float:
    value = max(0.0, min(1.0, float(channel)))
    if value <= 0.04045:
        return value / 12.92
    return ((value + 0.055) / 1.055) ** 2.4


def _oklab(rgb: Sequence[float]) -> tuple[float, float, float]:
    """Convert an sRGB triplet to OKLab for perceptual default-color ranking."""

    red, green, blue = (_srgb_channel_to_linear(value) for value in rgb)
    light = 0.4122214708 * red + 0.5363325363 * green + 0.0514459929 * blue
    medium = 0.2119034982 * red + 0.6806995451 * green + 0.1073969566 * blue
    short = 0.0883024619 * red + 0.2817188376 * green + 0.6299787005 * blue
    light_root = max(0.0, light) ** (1.0 / 3.0)
    medium_root = max(0.0, medium) ** (1.0 / 3.0)
    short_root = max(0.0, short) ** (1.0 / 3.0)
    return (
        0.2104542553 * light_root
        + 0.7936177850 * medium_root
        - 0.0040720468 * short_root,
        1.9779984951 * light_root
        - 2.4285922050 * medium_root
        + 0.4505937099 * short_root,
        0.0259040371 * light_root
        + 0.7827717662 * medium_root
        - 0.8086757660 * short_root,
    )


def _gaussian_similarity(distance: float, *, sigma: float) -> float:
    return math.exp(-0.5 * (max(0.0, distance) / sigma) ** 2)


def _hsv(rgb: Sequence[float]) -> tuple[float, float, float]:
    red, green, blue = (float(value) for value in rgb)
    maximum = max(red, green, blue)
    minimum = min(red, green, blue)
    delta = maximum - minimum
    saturation = 0.0 if maximum <= 0.0 else delta / maximum
    if delta <= 0.0:
        hue = 0.0
    elif maximum == red:
        hue = ((green - blue) / delta) % 6.0
    elif maximum == green:
        hue = (blue - red) / delta + 2.0
    else:
        hue = (red - green) / delta + 4.0
    return hue / 6.0, saturation, maximum


def _perceptual_color_similarity(
    observed: Sequence[float],
    candidate: Sequence[float],
) -> float:
    observed_oklab = _oklab(observed)
    candidate_oklab = _oklab(candidate)
    oklab_distance = math.sqrt(
        sum(
            (left - right) ** 2
            for left, right in zip(
                observed_oklab,
                candidate_oklab,
                strict=True,
            )
        )
    )
    oklab_similarity = _gaussian_similarity(oklab_distance, sigma=0.12)
    observed_hue, observed_saturation, _ = _hsv(observed)
    candidate_hue, candidate_saturation, _ = _hsv(candidate)
    if observed_saturation < 0.18 or candidate_saturation < 0.12:
        return oklab_similarity
    hue_distance = abs(observed_hue - candidate_hue)
    hue_distance = min(hue_distance, 1.0 - hue_distance)
    hue_similarity = _gaussian_similarity(hue_distance, sigma=0.075)
    saturation_similarity = _gaussian_similarity(
        abs(observed_saturation - candidate_saturation),
        sigma=0.25,
    )
    return (
        0.45 * oklab_similarity
        + 0.45 * hue_similarity
        + 0.10 * saturation_similarity
    )


def _float_token(value: str) -> float | None:
    try:
        number = float(value.rstrip("fF"))
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _export_block(text: str, sub_identifier: str) -> str | None:
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", sub_identifier) is None:
        return None
    match = re.search(
        _EXPORT_RE_TEMPLATE.format(identifier=re.escape(sub_identifier)),
        text,
    )
    if match is None:
        return None
    next_match = _NEXT_EXPORT_RE.search(text, match.end())
    return text[match.start() : next_match.start() if next_match else len(text)]


def _named_color(block: str, name: str) -> list[float] | None:
    match = re.search(
        rf"\b{re.escape(name)}\b\s*(?:=|:)\s*color\s*\(\s*"
        rf"({_NUMBER})(?:\s*,\s*({_NUMBER})\s*,\s*({_NUMBER}))?\s*\)",
        block,
    )
    if match is None:
        return None
    first = _float_token(match.group(1))
    second = _float_token(match.group(2)) if match.group(2) is not None else first
    third = _float_token(match.group(3)) if match.group(3) is not None else first
    if first is None or second is None or third is None:
        return None
    values = [first, second, third]
    if any(value < 0.0 or value > 1.0 for value in values):
        return None
    return values


def _named_scalar(block: str, name: str) -> float | None:
    match = re.search(
        rf"\b{re.escape(name)}\b\s*(?:=|:)\s*({_NUMBER})\b",
        block,
    )
    return _float_token(match.group(1)) if match is not None else None


def _named_bool(block: str, name: str) -> bool | None:
    match = re.search(
        rf"\b{re.escape(name)}\b\s*(?:=|:)\s*(true|false)\b",
        block,
        re.IGNORECASE,
    )
    return match.group(1).casefold() == "true" if match is not None else None


def _named_texture_has_path(block: str, name: str) -> bool:
    return (
        re.search(
            rf"\b{re.escape(name)}\b\s*(?:=|:)\s*"
            r'texture_2d\s*\(\s*"[^"]+"',
            block,
        )
        is not None
    )


def extract_mdl_appearance_profile(
    mdl_file: Path,
    sub_identifier: str,
) -> dict[str, Any] | None:
    """Extract a conservative numeric appearance profile for one MDL export."""

    try:
        text = mdl_file.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    block = _export_block(text, sub_identifier)
    if block is None:
        return None

    color_name: str | None = None
    color_linear: list[float] | None = None
    for name in _COLOR_PARAMETER_PRIORITY:
        color_linear = _named_color(block, name)
        if color_linear is not None:
            color_name = name
            break

    roughness_name: str | None = None
    roughness: float | None = None
    for name in _ROUGHNESS_PARAMETER_PRIORITY:
        roughness = _unit(_named_scalar(block, name))
        if roughness is not None:
            roughness_name = name
            break

    metallic_name: str | None = None
    metallic: float | None = None
    for name in _METALLIC_PARAMETER_PRIORITY:
        metallic = _unit(_named_scalar(block, name))
        if metallic is not None:
            metallic_name = name
            break

    color_texture_driven = any(
        _named_texture_has_path(block, name)
        for name in ("albedo_texture", "base_color_texture", "diffuse_texture")
    )
    orm_texture_driven = _named_bool(
        block, "enable_ORM_texture"
    ) is True and _named_texture_has_path(block, "ORM_texture")
    roughness_texture_driven = orm_texture_driven or any(
        _named_texture_has_path(block, name)
        for name in ("reflectionroughness_texture", "roughness_texture")
    )
    metallic_texture_driven = orm_texture_driven or _named_texture_has_path(
        block, "metallic_texture"
    )
    texture_driven_channels = [
        channel
        for channel, driven in (
            ("albedo", color_texture_driven),
            ("roughness", roughness_texture_driven),
            ("metallic", metallic_texture_driven),
        )
        if driven
    ]
    if color_texture_driven:
        color_linear = None
    if roughness_texture_driven:
        roughness = None
    if metallic_texture_driven:
        metallic = None

    if (
        color_linear is None
        and roughness is None
        and metallic is None
        and not texture_driven_channels
    ):
        return None
    return {
        "source": "parsed_export_defaults",
        "color_parameter": color_name,
        "base_color_linear": color_linear,
        "base_color_srgb": (
            [_linear_channel_to_srgb(channel) for channel in color_linear]
            if color_linear is not None
            else None
        ),
        "roughness_parameter": roughness_name,
        "roughness": roughness,
        "metallic_parameter": metallic_name,
        "metallic": metallic,
        "texture_driven_channels": texture_driven_channels,
    }


def extract_thumbnail_appearance_profile(
    thumbnail_file: Path,
) -> dict[str, Any] | None:
    """Estimate the immutable preview's central material color.

    NVIDIA material thumbnails use a centered preview object with studio
    lighting.  A conservative central disk avoids the environment background;
    chromatic pixels are preferred when they form a meaningful share so
    highlights do not desaturate a colored material.  This is catalog evidence,
    not reference-image evidence, and never authorizes shader edits.
    """

    try:
        from PIL import Image

        with Image.open(thumbnail_file) as source:
            image = source.convert("RGB")
            if min(image.size) < 64:
                return None
            image.thumbnail((64, 64), Image.Resampling.BILINEAR)
            width, height = image.size
            center_x = (width - 1) / 2.0
            center_y = (height - 1) / 2.0
            radius = min(width, height) * 0.38
            valid: list[tuple[float, float, float]] = []
            chromatic: list[tuple[float, float, float]] = []
            for y in range(height):
                for x in range(width):
                    if (
                        (x - center_x) ** 2 + (y - center_y) ** 2
                        > radius**2
                    ):
                        continue
                    rgb = tuple(
                        channel / 255.0 for channel in image.getpixel((x, y))
                    )
                    _hue, saturation, value = _hsv(rgb)
                    if not 0.05 <= value <= 0.90:
                        continue
                    valid.append(rgb)
                    if saturation >= 0.18:
                        chromatic.append(rgb)
    except (ImportError, OSError, ValueError):
        return None
    if len(valid) < 512:
        return None
    samples = (
        chromatic
        if len(chromatic) >= 512 and len(chromatic) / len(valid) >= 0.20
        else valid
    )
    ordered_channels = [
        sorted(sample[channel] for sample in samples) for channel in range(3)
    ]
    middle = len(samples) // 2
    medians = [
        (
            values[middle]
            if len(values) % 2
            else (values[middle - 1] + values[middle]) / 2.0
        )
        for values in ordered_channels
    ]
    return {
        "source": "nvidia_thumbnail_central_disk_median/v1",
        "base_color_srgb": medians,
        "sample_count": len(samples),
        "chromatic_sample_preferred": samples is chromatic,
    }


def mvinverse_similarity_terms(
    profile: Mapping[str, Any] | None,
    evidence_group: Mapping[str, Any] | None,
    *,
    fixed_defaults_required: bool = False,
    thumbnail_profile: Mapping[str, Any] | None = None,
) -> tuple[float, list[str]]:
    """Return a bounded score bonus from verified-looking MVInverse statistics.

    This helper does not authorize parameter writing.  It only ranks already
    whitelisted candidates.  In the normal mode, one accepted view contributes
    a modest similarity signal because a later policy may tune exposed inputs.
    When ``fixed_defaults_required`` is true, the selected NVIDIA export must
    render with its library defaults forever, so verified multi-view evidence
    receives enough weight to make the immutable default appearance decisive.
    """

    if not isinstance(fixed_defaults_required, bool):
        raise ValueError("fixed_defaults_required must be a boolean")
    if not isinstance(evidence_group, Mapping):
        return 0.0, []
    if not isinstance(profile, Mapping):
        profile = {}
    score = 0.0
    matched: list[str] = []

    albedo = evidence_group.get("albedo")
    candidate_colors = [profile.get("base_color_srgb")]
    if fixed_defaults_required and isinstance(thumbnail_profile, Mapping):
        candidate_colors.append(thumbnail_profile.get("base_color_srgb"))
    if isinstance(albedo, Mapping):
        observed = albedo.get("median")
        if (
            isinstance(observed, Sequence)
            and not isinstance(observed, (str, bytes))
            and len(observed) == 3
        ):
            observed_values = [_unit(value) for value in observed]
            candidate_value_sets = [
                [_unit(value) for value in candidate_color]
                for candidate_color in candidate_colors
                if isinstance(candidate_color, Sequence)
                and not isinstance(candidate_color, (str, bytes))
                and len(candidate_color) == 3
            ]
            candidate_value_sets = [
                values for values in candidate_value_sets if None not in values
            ]
            if None not in observed_values and candidate_value_sets:
                if fixed_defaults_required:
                    similarity = max(
                        _perceptual_color_similarity(
                            [float(value) for value in observed_values],
                            [float(value) for value in candidate_values],
                        )
                        for candidate_values in candidate_value_sets
                    )
                    score += 700.0 * similarity
                else:
                    candidate_values = candidate_value_sets[0]
                    distance = math.sqrt(
                        sum(
                            (float(left) - float(right)) ** 2
                            for left, right in zip(
                                observed_values, candidate_values, strict=True
                            )
                        )
                        / 3.0
                    )
                    score += 60.0 * max(0.0, 1.0 - distance)
                matched.append("mvinverse_color")

    roughness = evidence_group.get("roughness")
    candidate_roughness = _unit(profile.get("roughness"))
    if isinstance(roughness, Mapping) and candidate_roughness is not None:
        observed_roughness = _unit(roughness.get("median"))
        if observed_roughness is not None:
            difference = abs(observed_roughness - candidate_roughness)
            score += (
                75.0 * _gaussian_similarity(difference, sigma=0.25)
                if fixed_defaults_required
                else 30.0 * (1.0 - difference)
            )
            matched.append("mvinverse_roughness")

    metallic = evidence_group.get("metallic")
    candidate_metallic = _unit(profile.get("metallic"))
    if isinstance(metallic, Mapping) and candidate_metallic is not None:
        observed_metallic = _unit(metallic.get("median"))
        if observed_metallic is not None:
            difference = abs(observed_metallic - candidate_metallic)
            score += (
                100.0 * _gaussian_similarity(difference, sigma=0.30)
                if fixed_defaults_required
                else 30.0 * (1.0 - difference)
            )
            matched.append("mvinverse_metallic")

    return score, matched


__all__ = [
    "extract_mdl_appearance_profile",
    "extract_thumbnail_appearance_profile",
    "mvinverse_similarity_terms",
]
