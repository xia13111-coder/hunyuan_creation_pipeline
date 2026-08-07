"""Deterministically recover visual regions omitted by a VLM palette.

Qwen is good at describing the dominant enclosure but can omit narrow copper
tubes, valve caps, status buttons, small white control modules, and dark
workpiece surfaces photographed against a black background.  This module does
not guess a final material.  It finds connected, non-background chromatic or
high-value neutral regions and adds only missing appearance evidence at review
confidence.  Dark-region recovery is enabled only inside an explicit
foreground mask, so the image background cannot become material evidence.
The normal pixel gate, multiview mapping, material selector, and final render
QA remain authoritative.
"""

from __future__ import annotations

import colorsys
import hashlib
import math
import statistics
from collections import deque
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from ..core.staged_analysis import (
    MAX_PALETTE_GROUPS,
    PALETTE_SCHEMA_VERSION,
    REVIEW_THRESHOLD,
    validate_palette,
)
from .color_semantics import (
    evidence_color_labels,
    fusion_color_label,
    pixel_color_label,
)


ACCENT_AUGMENTATION_SCHEMA_VERSION = "qwen-palette-accent-augmentation/v1"

# These are all chromatic families emitted by ``pixel_color_label`` after
# folding illumination-adjacent aliases (brown -> orange, cyan -> blue).  The
# recovery lane must not be tied to colors seen in one fixture.
_ACCENT_COLORS = ("red", "orange", "yellow", "green", "blue", "pink")
_LIGHT_NEUTRAL_COLORS = ("white",)
_MASKED_DARK_COLORS = ("black",)
_MAX_COMPONENTS_PER_GROUP = 4
_MAX_ANALYSIS_SIDE = 512
_MASK_THRESHOLD = 127
_MASKED_RUST_MINIMUM_SATURATION = 0.14
_MASKED_RUST_MAXIMUM_SATURATION = 0.36
_MASKED_RUST_MINIMUM_VALUE = 0.16
_MASKED_RUST_MAXIMUM_VALUE = 0.48
_MASKED_RUST_MAXIMUM_HUE_DEGREES = 20.0
_MASKED_RUST_MINIMUM_RED_CHANNEL_MARGIN = 4


def _background_rgb(image: Image.Image) -> tuple[int, int, int]:
    width, height = image.size
    stride = max(1, min(width, height) // 128)
    pixels = image.load()
    samples: list[tuple[int, int, int]] = []
    for x in range(0, width, stride):
        samples.extend((pixels[x, 0], pixels[x, height - 1]))
    for y in range(0, height, stride):
        samples.extend((pixels[0, y], pixels[width - 1, y]))
    return tuple(
        int(round(statistics.median(sample[channel] for sample in samples)))
        for channel in range(3)
    )


def _distance(
    first: tuple[int, int, int],
    second: tuple[int, int, int],
) -> float:
    return math.sqrt(sum((first[index] - second[index]) ** 2 for index in range(3)))


def _is_accent_pixel(pixel: tuple[int, int, int], color: str) -> bool:
    if pixel_color_label(*pixel) not in evidence_color_labels(color):
        return False
    _hue, saturation, value = colorsys.rgb_to_hsv(
        pixel[0] / 255.0,
        pixel[1] / 255.0,
        pixel[2] / 255.0,
    )
    return saturation >= 0.30 and value >= 0.18


def _is_light_neutral_pixel(pixel: tuple[int, int, int], color: str) -> bool:
    """Return true only for bright, low-saturation neutral surface pixels."""

    if color != "white":
        return False
    _hue, saturation, value = colorsys.rgb_to_hsv(
        pixel[0] / 255.0,
        pixel[1] / 255.0,
        pixel[2] / 255.0,
    )
    return (
        pixel_color_label(*pixel) in evidence_color_labels("white")
        and saturation < 0.14
        and value >= 0.68
    )


def _is_masked_dark_pixel(pixel: tuple[int, int, int], color: str) -> bool:
    """Return true for an achromatic dark surface inside a trusted mask."""

    if color != "black":
        return False
    _hue, saturation, value = colorsys.rgb_to_hsv(
        pixel[0] / 255.0,
        pixel[1] / 255.0,
        pixel[2] / 255.0,
    )
    return pixel_color_label(*pixel) == "black" and (
        value <= 0.38 or saturation <= 0.24
    )


def _is_masked_rust_pixel(pixel: tuple[int, int, int], color: str) -> bool:
    """Recover dark red-brown coating that ordinary accent gates omit.

    Industrial rust and dark brown paint can be both too desaturated for the
    regular chromatic gate and slightly too bright to be classified as black.
    This narrow lane is available only inside an explicit foreground mask.
    It intentionally joins the existing orange/brown evidence family; it does
    not infer a physical material.
    """

    if color != "orange":
        return False
    hue, saturation, value = colorsys.rgb_to_hsv(
        pixel[0] / 255.0,
        pixel[1] / 255.0,
        pixel[2] / 255.0,
    )
    hue_degrees = hue * 360.0
    return (
        _MASKED_RUST_MINIMUM_SATURATION
        <= saturation
        <= _MASKED_RUST_MAXIMUM_SATURATION
        and _MASKED_RUST_MINIMUM_VALUE <= value <= _MASKED_RUST_MAXIMUM_VALUE
        and (
            hue_degrees < _MASKED_RUST_MAXIMUM_HUE_DEGREES
            or hue_degrees >= 345.0
        )
        and pixel[0] >= pixel[1] + _MASKED_RUST_MINIMUM_RED_CHANNEL_MARGIN
        and pixel[0] >= pixel[2] + _MASKED_RUST_MINIMUM_RED_CHANNEL_MARGIN
    )


def _mask_plane(image: Image.Image) -> tuple[Image.Image, str]:
    """Return the useful foreground plane from an explicit mask image."""

    if "A" in image.getbands():
        alpha = image.getchannel("A")
        if alpha.getextrema() != (255, 255):
            return alpha, "alpha"
    return image.convert("L"), "luminance"


def _components(
    flags: list[bool],
    *,
    width: int,
    height: int,
) -> list[dict[str, int]]:
    visited = bytearray(len(flags))
    output: list[dict[str, int]] = []
    for start, enabled in enumerate(flags):
        if not enabled or visited[start]:
            continue
        visited[start] = 1
        queue: deque[int] = deque([start])
        count = 0
        minimum_x = width
        minimum_y = height
        maximum_x = -1
        maximum_y = -1
        touches_border = False
        while queue:
            index = queue.popleft()
            x = index % width
            y = index // width
            count += 1
            minimum_x = min(minimum_x, x)
            minimum_y = min(minimum_y, y)
            maximum_x = max(maximum_x, x)
            maximum_y = max(maximum_y, y)
            touches_border = touches_border or (
                x <= 1 or y <= 1 or x + 2 >= width or y + 2 >= height
            )
            for neighbor_y in range(max(0, y - 1), min(height, y + 2)):
                row = neighbor_y * width
                for neighbor_x in range(max(0, x - 1), min(width, x + 2)):
                    neighbor = row + neighbor_x
                    if flags[neighbor] and not visited[neighbor]:
                        visited[neighbor] = 1
                        queue.append(neighbor)
        output.append(
            {
                "pixel_count": count,
                "minimum_x": minimum_x,
                "minimum_y": minimum_y,
                "maximum_x": maximum_x,
                "maximum_y": maximum_y,
                "touches_border": int(touches_border),
            }
        )
    return output


def _component_box(
    component: Mapping[str, int],
    *,
    width: int,
    height: int,
) -> list[int]:
    padding = 2
    left = max(0, int(component["minimum_x"]) - padding)
    top = max(0, int(component["minimum_y"]) - padding)
    right = min(width, int(component["maximum_x"]) + padding + 1)
    bottom = min(height, int(component["maximum_y"]) + padding + 1)
    return [
        max(0, min(999, int(math.floor(left * 1000 / width)))),
        max(0, min(999, int(math.floor(top * 1000 / height)))),
        max(1, min(1000, int(math.ceil(right * 1000 / width)))),
        max(1, min(1000, int(math.ceil(bottom * 1000 / height)))),
    ]


def _normalized_box_overlap_fraction(
    left: list[int],
    right: list[int],
) -> float:
    """Return intersection over the smaller normalized box area."""

    intersection_width = max(0, min(left[2], right[2]) - max(left[0], right[0]))
    intersection_height = max(0, min(left[3], right[3]) - max(left[1], right[1]))
    intersection = intersection_width * intersection_height
    left_area = max(1, (left[2] - left[0]) * (left[3] - left[1]))
    right_area = max(1, (right[2] - right[0]) * (right[3] - right[1]))
    return intersection / min(left_area, right_area)


def _group_semantics(color: str) -> tuple[str, str, str]:
    if color == "white":
        return (
            "other",
            "other",
            "connected light neutral surface region detected from pixels; "
            "physical material unresolved",
        )
    if color == "black":
        return (
            "other",
            "other",
            "connected dark surface region detected inside the trusted "
            "foreground mask; physical material unresolved",
        )
    return (
        "other",
        "other",
        f"connected {color} chromatic region detected from pixels; "
        "physical material unresolved",
    )


def augment_palette_with_detected_accents(
    palette: Mapping[str, Any],
    image_path: str | Path,
    *,
    mask_path: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Add connected missing visual groups without choosing an MDL.

    Chromatic and light-neutral recovery remains image-only.  Black recovery
    requires ``mask_path`` and samples only pixels inside that explicit
    foreground mask.
    """

    canonical = validate_palette(palette)
    resolved = Path(image_path).expanduser().resolve(strict=True)
    with Image.open(resolved) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
    resolved_mask: Path | None = None
    foreground_mask: Image.Image | None = None
    mask_channel: str | None = None
    if mask_path is not None:
        resolved_mask = Path(mask_path).expanduser().resolve(strict=True)
        with Image.open(resolved_mask) as opened_mask:
            foreground_mask, mask_channel = _mask_plane(
                ImageOps.exif_transpose(opened_mask)
            )
            foreground_mask = foreground_mask.copy()
        if foreground_mask.size != image.size:
            raise ValueError(
                "foreground mask size does not match the reference image: "
                f"{foreground_mask.size} != {image.size}"
            )
    scale = min(1.0, _MAX_ANALYSIS_SIDE / max(image.size))
    if scale < 1.0:
        resized_size = (
            max(1, int(round(image.width * scale))),
            max(1, int(round(image.height * scale))),
        )
        image = image.resize(resized_size, Image.Resampling.BILINEAR)
        if foreground_mask is not None:
            foreground_mask = foreground_mask.resize(
                resized_size,
                Image.Resampling.NEAREST,
            )
    width, height = image.size
    pixels = list(
        image.get_flattened_data()
        if hasattr(image, "get_flattened_data")
        else image.getdata()
    )
    foreground_values = (
        list(
            foreground_mask.get_flattened_data()
            if hasattr(foreground_mask, "get_flattened_data")
            else foreground_mask.getdata()
        )
        if foreground_mask is not None
        else None
    )
    background = _background_rgb(image)
    existing_colors = {
        fusion_color_label(str(group["base_color"]))
        for group in canonical["groups"]
    }
    minimum_pixels = max(12, int(round(width * height * 0.00004)))
    maximum_box_area = width * height * 0.25
    additions: list[dict[str, Any]] = []
    component_audits: list[dict[str, Any]] = []
    next_group_number = max(
        int(str(group["group_id"])[1:]) for group in canonical["groups"]
    ) + 1
    available_group_slots = MAX_PALETTE_GROUPS - len(canonical["groups"])

    for color in (
        *_ACCENT_COLORS,
        *_LIGHT_NEUTRAL_COLORS,
        *_MASKED_DARK_COLORS,
    ):
        masked_rust_completion = bool(
            color == "orange" and foreground_values is not None
        )
        if color in existing_colors and not masked_rust_completion:
            component_audits.append(
                {
                    "base_color": color,
                    "decision": "already_present",
                    "accepted_components": [],
                }
            )
            continue
        if color in _MASKED_DARK_COLORS and foreground_values is None:
            component_audits.append(
                {
                    "base_color": color,
                    "decision": "foreground_mask_required",
                    "accepted_components": [],
                }
            )
            continue
        if color in _MASKED_DARK_COLORS:
            assert foreground_values is not None
            flags = [
                mask_value > _MASK_THRESHOLD
                and _is_masked_dark_pixel(pixel, color)
                for pixel, mask_value in zip(pixels, foreground_values)
            ]
        else:
            flags: list[bool] = []
            for index, pixel in enumerate(pixels):
                regular_accent = (
                    _is_accent_pixel(pixel, color)
                    if color in _ACCENT_COLORS
                    else _is_light_neutral_pixel(pixel, color)
                )
                masked_rust = bool(
                    masked_rust_completion
                    and foreground_values is not None
                    and foreground_values[index] > _MASK_THRESHOLD
                    and _is_masked_rust_pixel(pixel, color)
                )
                flags.append(
                    (regular_accent or masked_rust)
                    and _distance(pixel, background) >= 28.0
                )
        accepted: list[dict[str, int]] = []
        rejected_counts: dict[str, int] = {}
        for component in _components(flags, width=width, height=height):
            box_width = component["maximum_x"] - component["minimum_x"] + 1
            box_height = component["maximum_y"] - component["minimum_y"] + 1
            box_area = box_width * box_height
            occupancy = component["pixel_count"] / max(1, box_area)
            reason: str | None = None
            if component["touches_border"]:
                reason = "touches_image_border"
            elif component["pixel_count"] < minimum_pixels:
                reason = "too_few_pixels"
            elif min(box_width, box_height) < 2:
                reason = "subpixel_line"
            elif box_area > maximum_box_area:
                reason = "not_a_small_accent"
            elif occupancy < (
                0.20
                if color in _LIGHT_NEUTRAL_COLORS
                else (0.05 if color in _MASKED_DARK_COLORS else 0.02)
            ):
                reason = "line_or_sparse_overlay"
            if reason is not None:
                rejected_counts[reason] = rejected_counts.get(reason, 0) + 1
                continue
            accepted.append(component)
        accepted.sort(
            key=lambda item: (
                -item["pixel_count"],
                item["minimum_y"],
                item["minimum_x"],
            )
        )
        if color in existing_colors:
            existing_boxes = [
                list(box)
                for group in canonical["groups"]
                if fusion_color_label(str(group["base_color"])) == color
                for box in group["boxes"]
            ]
            accepted = [
                component
                for component in accepted
                if all(
                    _normalized_box_overlap_fraction(
                        _component_box(component, width=width, height=height),
                        existing_box,
                    )
                    < 0.25
                    for existing_box in existing_boxes
                )
            ]
        accepted = accepted[:_MAX_COMPONENTS_PER_GROUP]
        boxes = [
            _component_box(component, width=width, height=height)
            for component in accepted
        ]
        capacity_exhausted = bool(boxes) and len(additions) >= available_group_slots
        if boxes and not capacity_exhausted:
            family, finish, description = _group_semantics(color)
            additions.append(
                {
                    "group_id": f"G{next_group_number:02d}",
                    "family_hint": family,
                    "base_color": color,
                    "finish_hint": finish,
                    "visual_description": description,
                    "boxes": boxes,
                    "confidence": REVIEW_THRESHOLD,
                }
            )
            next_group_number += 1
        component_audits.append(
                {
                    "base_color": color,
                    "decision": (
                        "capacity_exhausted"
                        if capacity_exhausted
                        else (
                            "added"
                            if boxes
                            else (
                                "already_present_no_uncovered_component"
                                if color in existing_colors
                                else "not_detected"
                            )
                        )
                    ),
                "accepted_components": [
                    {
                        **component,
                        "box": box,
                    }
                    for component, box in zip(accepted, boxes)
                ],
                "rejected_component_counts": dict(sorted(rejected_counts.items())),
            }
        )

    augmented = validate_palette(
        {
            "schema_version": PALETTE_SCHEMA_VERSION,
            "source_view_id": canonical["source_view_id"],
            "groups": [*canonical["groups"], *additions],
        }
    )
    return augmented, {
        "schema_version": ACCENT_AUGMENTATION_SCHEMA_VERSION,
        "source_view_id": canonical["source_view_id"],
        "image": str(resolved),
        "image_sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
        "mask": str(resolved_mask) if resolved_mask is not None else None,
        "mask_sha256": (
            hashlib.sha256(resolved_mask.read_bytes()).hexdigest()
            if resolved_mask is not None
            else None
        ),
        "mask_channel": mask_channel,
        "masked_dark_recovery_enabled": foreground_values is not None,
        "masked_low_saturation_rust_recovery_enabled": (
            foreground_values is not None
        ),
        "analysis_size": [width, height],
        "estimated_background_rgb": list(background),
        "minimum_component_pixels": minimum_pixels,
        "added_group_ids": [group["group_id"] for group in additions],
        "components": component_audits,
    }


__all__ = [
    "ACCENT_AUGMENTATION_SCHEMA_VERSION",
    "augment_palette_with_detected_accents",
]
