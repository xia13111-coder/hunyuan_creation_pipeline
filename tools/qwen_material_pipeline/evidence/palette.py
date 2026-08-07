"""Deterministic pixel checks for Qwen palette evidence boxes.

The checks are deliberately conservative.  They do not try to segment an
object perfectly; they only reject citations that are mostly background or
whose visible pixels contradict the claimed base color.
"""

from __future__ import annotations

import math
import statistics
from collections import Counter, deque
from pathlib import Path
from typing import Any, Mapping

from PIL import Image, ImageOps

from qwen_material_pipeline.core.staged_analysis import (
    REVIEW_CONFIDENCE_CAP,
    REVIEW_THRESHOLD,
    StagedAnalysisError,
    validate_palette,
)
from qwen_material_pipeline.evidence.color_semantics import (
    evidence_color_labels,
    pixel_color_label,
)


_MASK_THRESHOLD = 127
_MINIMUM_BLACK_STRUCTURE_COVERAGE = 0.02
_MINIMUM_MIXED_ACHROMATIC_MATCH = 0.25
_MINIMUM_MATCHING_PIXELS = 128
_MINIMUM_MATCHING_STRUCTURE_COVERAGE = 0.02
_MINIMUM_MIXED_CHROMATIC_MATCH = 0.04
_MINIMUM_CHROMATIC_MATCHING_PIXELS = 48
_CHROMATIC_COLORS = frozenset(
    {"red", "orange", "brown", "yellow", "green", "cyan", "blue", "pink"}
)


def _background_color(image: Image.Image) -> tuple[int, int, int]:
    width, height = image.size
    stride = max(1, min(width, height) // 128)
    pixels = image.load()
    samples: list[tuple[int, int, int]] = []
    for x in range(0, width, stride):
        samples.append(pixels[x, 0])
        samples.append(pixels[x, height - 1])
    for y in range(0, height, stride):
        samples.append(pixels[0, y])
        samples.append(pixels[width - 1, y])
    return tuple(
        int(round(statistics.median(sample[channel] for sample in samples)))
        for channel in range(3)
    )


def _mask_plane(image: Image.Image) -> tuple[Image.Image, str]:
    """Return the useful foreground plane from an explicit mask image."""

    if "A" in image.getbands():
        alpha = image.getchannel("A")
        # A regular RGBA black/white segmentation mask is commonly fully
        # opaque.  In that case its luminance, not its alpha, carries the mask.
        if alpha.getextrema() != (255, 255):
            return alpha, "alpha"
    return image.convert("L"), "luminance"


def _largest_component_size(flags: list[bool], width: int, height: int) -> int:
    """Return the largest 4-connected region in a small sampled crop."""

    visited = bytearray(len(flags))
    largest = 0
    for start, enabled in enumerate(flags):
        if not enabled or visited[start]:
            continue
        visited[start] = 1
        queue: deque[int] = deque([start])
        size = 0
        while queue:
            index = queue.popleft()
            size += 1
            x = index % width
            if x and flags[index - 1] and not visited[index - 1]:
                visited[index - 1] = 1
                queue.append(index - 1)
            if x + 1 < width and flags[index + 1] and not visited[index + 1]:
                visited[index + 1] = 1
                queue.append(index + 1)
            if index >= width and flags[index - width] and not visited[index - width]:
                visited[index - width] = 1
                queue.append(index - width)
            below = index + width
            if below < len(flags) and flags[below] and not visited[below]:
                visited[below] = 1
                queue.append(below)
        largest = max(largest, size)
    return largest


def _box_metrics(
    image: Image.Image,
    box: list[int],
    *,
    foreground_mask: Image.Image | None,
    background: tuple[int, int, int],
    expected_color: str,
    background_distance: float,
) -> dict[str, Any]:
    width, height = image.size
    left = max(0, int(math.floor(box[0] * width / 1000)))
    top = max(0, int(math.floor(box[1] * height / 1000)))
    right = min(width, int(math.ceil(box[2] * width / 1000)))
    bottom = min(height, int(math.ceil(box[3] * height / 1000)))
    crop = image.crop((left, top, right, bottom))
    mask_crop = (
        foreground_mask.crop((left, top, right, bottom))
        if foreground_mask is not None
        else None
    )
    scale = min(1.0, 160.0 / max(crop.size))
    if scale < 1.0:
        resized_size = (
            max(1, int(round(crop.width * scale))),
            max(1, int(round(crop.height * scale))),
        )
        crop = crop.resize(
            resized_size,
            Image.Resampling.BILINEAR,
        )
        if mask_crop is not None:
            mask_crop = mask_crop.resize(resized_size, Image.Resampling.NEAREST)
    pixel_data = list(
        crop.get_flattened_data()
        if hasattr(crop, "get_flattened_data")
        else crop.getdata()
    )
    distances = [
        math.sqrt(
            sum((pixel[channel] - background[channel]) ** 2 for channel in range(3))
        )
        for pixel in pixel_data
    ]
    total = max(1, crop.width * crop.height)

    # A black object photographed against a black background can remain below
    # the normal background-distance threshold.  For black evidence only, use
    # a smaller distance to find subtly lit object pixels.  Exact background
    # pixels are still excluded, and acceptance later requires that candidate
    # black pixels form a meaningful connected structure.
    dark_background = max(background) < 80
    black_distance = min(
        background_distance,
        max(6.0, background_distance * 0.25),
    )
    if mask_crop is not None:
        mask_data = list(
            mask_crop.get_flattened_data()
            if hasattr(mask_crop, "get_flattened_data")
            else mask_crop.getdata()
        )
        foreground_flags = [value > _MASK_THRESHOLD for value in mask_data]
        foreground_method = "mask"
        effective_background_distance: float | None = None
    elif expected_color == "black" and dark_background:
        foreground_flags = [distance >= black_distance for distance in distances]
        foreground_method = "dark_structure"
        effective_background_distance = black_distance
    else:
        foreground_flags = [distance >= background_distance for distance in distances]
        foreground_method = "color_distance"
        effective_background_distance = background_distance

    foreground = [
        pixel for pixel, enabled in zip(pixel_data, foreground_flags) if enabled
    ]
    foreground_coverage = len(foreground) / total
    accepted_colors = evidence_color_labels(expected_color)
    foreground_color_counts = Counter(
        pixel_color_label(*pixel) for pixel in foreground
    )
    if expected_color in {"other", "unknown", "clear"}:
        color_match = None
        matching_pixel_data: list[tuple[int, int, int]] = []
        matching_pixel_count: int | None = None
        representative_srgb: list[int] | None = None
    else:
        matching_pixel_data = [
            pixel
            for pixel in foreground
            if pixel_color_label(*pixel) in accepted_colors
        ]
        matching_pixel_count = len(matching_pixel_data)
        color_match = matching_pixel_count / len(foreground) if foreground else 0.0
        representative_srgb = (
            [
                int(
                    round(
                        statistics.median(
                            pixel[channel] for pixel in matching_pixel_data
                        )
                    )
                )
                for channel in range(3)
            ]
            if matching_pixel_data
            else None
        )

    black_structure_flags = [
        enabled and pixel_color_label(*pixel) == "black"
        for pixel, enabled in zip(pixel_data, foreground_flags)
    ]
    black_structure_pixels = sum(black_structure_flags)
    largest_black_structure_pixels = _largest_component_size(
        black_structure_flags, crop.width, crop.height
    )
    largest_black_structure_coverage = largest_black_structure_pixels / total
    black_structure_supported = (
        largest_black_structure_coverage >= _MINIMUM_BLACK_STRUCTURE_COVERAGE
    )
    matching_structure_flags = [
        enabled and pixel_color_label(*pixel) in accepted_colors
        for pixel, enabled in zip(pixel_data, foreground_flags)
    ]
    largest_matching_structure_pixels = _largest_component_size(
        matching_structure_flags, crop.width, crop.height
    )
    largest_matching_structure_coverage = largest_matching_structure_pixels / total
    # Qwen boxes are evidence citations, not segmentation masks.  A tight box
    # around a small neutral component can still contain a neighboring painted
    # panel.  Accept that mixed citation only when the claimed neutral color has
    # both substantial pixels and a connected surface; chromatic groups retain
    # the stricter global ratio.
    mixed_achromatic_supported = (
        expected_color in {"black", "white", "gray", "silver"}
        and color_match is not None
        and color_match >= _MINIMUM_MIXED_ACHROMATIC_MATCH
        and matching_pixel_count is not None
        and matching_pixel_count >= _MINIMUM_MATCHING_PIXELS
        and largest_matching_structure_coverage >= _MINIMUM_MATCHING_STRUCTURE_COVERAGE
    )
    # Small accent materials (valve caps, copper tubes, status buttons) often
    # occupy only a narrow connected region inside an otherwise useful Qwen
    # citation.  Preserve such evidence for review when the matching chromatic
    # pixels form a real component; a few scattered lighting pixels are still
    # rejected.
    mixed_chromatic_supported = (
        expected_color in _CHROMATIC_COLORS
        and color_match is not None
        and color_match >= _MINIMUM_MIXED_CHROMATIC_MATCH
        and matching_pixel_count is not None
        and matching_pixel_count >= _MINIMUM_CHROMATIC_MATCHING_PIXELS
        and largest_matching_structure_coverage
        >= _MINIMUM_MATCHING_STRUCTURE_COVERAGE
    )
    return {
        "foreground_coverage": foreground_coverage,
        "color_match": color_match,
        "sampled_pixels": total,
        "foreground_pixels": len(foreground),
        # Keep the original spelling for consumers that started reading the
        # richer audit during development; the explicit name is authoritative.
        "matching_pixels": matching_pixel_count,
        "matching_pixel_count": matching_pixel_count,
        "representative_srgb": representative_srgb,
        "foreground_method": foreground_method,
        "effective_background_distance": effective_background_distance,
        "accepted_color_labels": sorted(accepted_colors),
        "foreground_color_counts": dict(sorted(foreground_color_counts.items())),
        "black_structure_required": expected_color == "black",
        "black_structure_supported": black_structure_supported,
        "black_structure_pixels": black_structure_pixels,
        "largest_black_structure_pixels": largest_black_structure_pixels,
        "largest_black_structure_coverage": largest_black_structure_coverage,
        "largest_matching_structure_pixels": largest_matching_structure_pixels,
        "largest_matching_structure_coverage": largest_matching_structure_coverage,
        "mixed_achromatic_supported": mixed_achromatic_supported,
        "mixed_chromatic_supported": mixed_chromatic_supported,
    }


def filter_palette_by_image_evidence(
    palette: Mapping[str, Any],
    image_path: str | Path,
    *,
    minimum_foreground_coverage: float = 0.10,
    minimum_color_match: float = 0.55,
    background_distance: float = 28.0,
    mask_path: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a palette containing only boxes supported by reference pixels.

    ``mask_path`` is optional and backwards compatible.  Its non-zero alpha
    (when meaningful) or light luminance pixels define foreground.  When no
    explicit mask is supplied, a non-opaque alpha channel in the reference
    image is used automatically before falling back to color-distance checks.
    """

    canonical = validate_palette(palette)
    resolved = Path(image_path).expanduser().resolve(strict=True)
    with Image.open(resolved) as opened:
        transposed = ImageOps.exif_transpose(opened)
        embedded_mask: Image.Image | None = None
        if "A" in transposed.getbands():
            alpha = transposed.getchannel("A")
            if alpha.getextrema() != (255, 255):
                embedded_mask = alpha.copy()
        image = transposed.convert("RGB")

    resolved_mask: Path | None = None
    mask_source: str | None = None
    mask_channel: str | None = None
    foreground_mask = embedded_mask
    if embedded_mask is not None:
        mask_source = "image_alpha"
        mask_channel = "alpha"
    if mask_path is not None:
        resolved_mask = Path(mask_path).expanduser().resolve(strict=True)
        with Image.open(resolved_mask) as opened_mask:
            transposed_mask = ImageOps.exif_transpose(opened_mask)
            foreground_mask, mask_channel = _mask_plane(transposed_mask)
            foreground_mask = foreground_mask.copy()
        mask_source = "explicit"
    if foreground_mask is not None and foreground_mask.size != image.size:
        raise StagedAnalysisError(
            "reference foreground mask size does not match the reference image: "
            f"{foreground_mask.size} != {image.size}"
        )

    background = _background_color(image)
    accepted_groups: list[dict[str, Any]] = []
    audit_groups: list[dict[str, Any]] = []
    for group in canonical["groups"]:
        accepted_boxes: list[list[int]] = []
        box_audits: list[dict[str, Any]] = []
        best_match = 0.0
        for index, box in enumerate(group["boxes"]):
            metrics = _box_metrics(
                image,
                box,
                foreground_mask=foreground_mask,
                background=background,
                expected_color=group["base_color"],
                background_distance=background_distance,
            )
            color_match = metrics["color_match"]
            rejection_reasons: list[str] = []
            if metrics["foreground_coverage"] < minimum_foreground_coverage:
                rejection_reasons.append("insufficient_foreground")
            if (
                color_match is not None
                and color_match < minimum_color_match
                and not metrics["mixed_achromatic_supported"]
                and not metrics["mixed_chromatic_supported"]
            ):
                rejection_reasons.append("insufficient_color_match")
            if (
                group["base_color"] == "black"
                and not metrics["black_structure_supported"]
            ):
                rejection_reasons.append("missing_black_structure")
            accepted = not rejection_reasons
            if accepted:
                accepted_boxes.append(box)
                evidence_confidence = 1.0 if color_match is None else color_match
                if metrics["mixed_achromatic_supported"]:
                    # The group is real but the Qwen citation is not a clean
                    # segmentation mask.  Keep it eligible for human review,
                    # never promote it to auto solely from this mixed box.
                    evidence_confidence = max(evidence_confidence, REVIEW_THRESHOLD)
                if metrics["mixed_chromatic_supported"]:
                    # Connected accent evidence verifies that the appearance is
                    # present, but its broad citation is not a segmentation
                    # mask.  Keep it at review confidence until an independent
                    # view or spatial registration confirms the affected part.
                    evidence_confidence = max(evidence_confidence, REVIEW_THRESHOLD)
                best_match = max(best_match, evidence_confidence)
            box_audits.append(
                {
                    "box_index": index,
                    "box": box,
                    "accepted": accepted,
                    "rejection_reasons": rejection_reasons,
                    **metrics,
                }
            )
        accepted = bool(accepted_boxes)
        audit_groups.append(
            {
                "group_id": group["group_id"],
                "base_color": group["base_color"],
                "accepted": accepted,
                "boxes": box_audits,
            }
        )
        if accepted:
            record = dict(group)
            record["boxes"] = accepted_boxes
            record["confidence"] = min(record["confidence"], best_match)
            accepted_groups.append(record)
    if not accepted_groups:
        raise StagedAnalysisError(
            "all Qwen palette groups failed reference-image evidence checks"
        )
    # If a visibly diverse model palette collapses to one pixel-supported
    # group, keep it available for review but never let that lone group create
    # automatic whole-asset assignments.
    if len(canonical["groups"]) >= 3 and len(accepted_groups) == 1:
        accepted_groups[0]["confidence"] = min(
            accepted_groups[0]["confidence"], REVIEW_CONFIDENCE_CAP
        )
    filtered = validate_palette(
        {
            "schema_version": canonical["schema_version"],
            "source_view_id": canonical["source_view_id"],
            "groups": accepted_groups,
        }
    )
    audit = {
        "image": str(resolved),
        "mask": str(resolved_mask) if resolved_mask is not None else None,
        "mask_source": mask_source,
        "mask_channel": mask_channel,
        "estimated_background_rgb": list(background),
        "minimum_foreground_coverage": minimum_foreground_coverage,
        "minimum_color_match": minimum_color_match,
        "background_distance": background_distance,
        "minimum_black_structure_coverage": _MINIMUM_BLACK_STRUCTURE_COVERAGE,
        "accepted_group_ids": [group["group_id"] for group in accepted_groups],
        "rejected_group_ids": [
            group["group_id"]
            for group in canonical["groups"]
            if group["group_id"] not in {item["group_id"] for item in accepted_groups}
        ],
        "groups": audit_groups,
    }
    return filtered, audit


__all__ = ["filter_palette_by_image_evidence"]
