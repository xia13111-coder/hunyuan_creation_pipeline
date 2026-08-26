#!/usr/bin/env python3
"""Conservative reference-photo versus RTX-render comparison.

This module intentionally separates *view alignment* from *material colour*.
Silhouette and edge metrics are used only as an eligibility gate.  A mapped
pair that does not clear that gate is reported as ``UNSCORABLE`` and never as
a material failure.  Every in-scope source reference view must nevertheless
receive a confident, one-to-one fixed-render match before the aggregate result
can pass; missing or ambiguous pose coverage therefore fails closed as
``INSUFFICIENT_EVIDENCE``.  The default scope is the whole asset.  An optional
canonical-group scope keeps whole-object pose alignment but measures material
appearance only inside trusted reference ROIs and requested render part-ID
masks.

The implementation uses Pillow and deterministic image statistics only; it
does not download or load a learned vision model.
"""

from __future__ import annotations

import argparse
import colorsys
import hashlib
import json
import math
import statistics
import sys
from collections import Counter, deque
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from PIL import (
    Image,
    ImageChops,
    ImageDraw,
    ImageFilter,
    ImageOps,
    UnidentifiedImageError,
)


SCHEMA_VERSION = "qwen-reference-render-comparison/v1"
NORMALIZED_MASK_SIZE = 128
MAX_ANALYSIS_IMAGE_SIDE = 512
MIN_TRUSTED_EVIDENCE_PIXELS = 64
MIN_TEXTURE_ANALYSIS_PIXELS = 64
TEXTURE_DETAIL_RADII = (2.0, 6.0)
TEXTURE_INTERIOR_KERNELS = (9, 5, 3, 1)
TEXTURE_HISTOGRAM_BIN_COUNT = 32
TEXTURE_LUMA_FLOOR_8BIT = 24.0
EXIT_SUCCESS = 0
EXIT_INPUT_ERROR = 2
EXIT_REQUIRE_PASS_FAILED = 3

DEFAULT_THRESHOLDS: dict[str, float | int] = {
    "minimum_reference_foreground_coverage": 0.015,
    "maximum_reference_foreground_coverage": 0.80,
    "minimum_render_foreground_coverage": 0.015,
    "maximum_render_foreground_coverage": 0.80,
    "minimum_explicit_alignment_score": 0.40,
    "minimum_auto_alignment_score": 0.48,
    "minimum_auto_match_margin": 0.055,
    "strong_alignment_score": 0.55,
    "pass_color_score": 0.62,
    "fail_color_score": 0.34,
    "aggregate_fail_color_score": 0.40,
    # A correct hue with a visibly wrong exposure must not be called PASS.
    # Neutral-light tournament renders make this a stable appearance gate;
    # lower values remain REVIEW rather than being treated as proof of failure.
    "minimum_median_value_similarity_for_pass": 0.85,
    "minimum_median_saturation_similarity_for_pass": 0.70,
    # Real multiview photographs commonly carry a view-dependent exposure
    # that cannot be reproduced by a fixed neutral-light tournament render.
    # A value-only REVIEW may therefore be promoted only by a complete,
    # tightly bounded multiview cohort proof.  Single-view evidence and large
    # value errors remain REVIEW.
    "minimum_photometric_cohort_views": 3,
    "maximum_photometric_value_offset": 0.30,
    "minimum_reference_value_span_for_photometric_cohort": 0.08,
    "maximum_render_value_span_for_photometric_cohort": 0.08,
    "minimum_photometric_cohort_category_similarity": 0.70,
    "minimum_photometric_cohort_hue_similarity": 0.90,
    "minimum_photometric_cohort_appearance_score": 0.70,
    "minimum_evidence_group_recall": 0.50,
    "minimum_evidence_macro_recall": 0.55,
    "minimum_reliable_group_evidence_pixels": 128,
    "low_evidence_recall_tolerance_ratio": 0.90,
    "minimum_low_evidence_observed_render_share": 0.001,
    "minimum_dominant_reference_share": 0.25,
    "minimum_dominant_share_margin": 0.10,
    "minimum_dominant_mass_recall": 0.80,
    "minimum_dominant_absolute_deficit": 0.08,
    "minimum_dominant_silhouette_iou": 0.75,
    "minimum_unreferenced_render_chromatic_share": 0.05,
    "maximum_reference_share_for_unreferenced_chromatic": 0.01,
    "minimum_unreferenced_render_chromatic_excess": 0.05,
    "minimum_comparable_views": 2,
    "minimum_failing_views_for_aggregate_fail": 2,
    # A scoped material group may legitimately cover far less than the
    # whole-object 1.5% foreground floor.  Keep the local trust boundary in
    # absolute pixels instead; this is also the minimum needed by the texture
    # statistic below.
    "minimum_target_foreground_pixels": MIN_TEXTURE_ANALYSIS_PIXELS,
}

_COLOR_BINS = (
    "black",
    "achromatic_dark",
    "achromatic_mid",
    "achromatic_light",
    "red",
    "orange_brown",
    "yellow",
    "green",
    "cyan_blue",
    "purple",
)
_CHROMATIC_COLOR_BINS = (
    "red",
    "orange_brown",
    "yellow",
    "green",
    "cyan_blue",
    "purple",
)


class ComparisonInputError(ValueError):
    """Raised when comparison inputs do not satisfy the trust boundary."""


def _image_data(image: Image.Image) -> Iterable[Any]:
    if hasattr(image, "get_flattened_data"):
        return image.get_flattened_data()
    return image.getdata()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ComparisonInputError(
            f"Unable to read {label} JSON {path}: {exc}"
        ) from exc
    if not isinstance(document, dict):
        raise ComparisonInputError(f"{label} must be a JSON object: {path}")
    return document


def _resolve_file(value: Any, owner: Path, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ComparisonInputError(f"{label} must be a non-empty path string")
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = owner.parent / candidate
    try:
        return candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ComparisonInputError(f"{label} does not exist: {candidate}") from exc


def _open_rgb(path: Path, label: str) -> Image.Image:
    try:
        with Image.open(path) as image:
            return ImageOps.exif_transpose(image).convert("RGB")
    except (OSError, UnidentifiedImageError) as exc:
        raise ComparisonInputError(f"Unable to decode {label} {path}: {exc}") from exc


def _open_mask(path: Path, expected_size: tuple[int, int], label: str) -> Image.Image:
    try:
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image)
            if image.size != expected_size:
                raise ComparisonInputError(
                    f"{label} size {image.size} does not match image size {expected_size}"
                )
            if "A" in image.getbands() and image.getchannel("A").getextrema() != (
                255,
                255,
            ):
                plane = image.getchannel("A")
            else:
                plane = image.convert("L")
            return plane.point(lambda value: 255 if value > 127 else 0, mode="L")
    except ComparisonInputError:
        raise
    except (OSError, UnidentifiedImageError) as exc:
        raise ComparisonInputError(f"Unable to decode {label} {path}: {exc}") from exc


def _resize_pair(
    image: Image.Image,
    mask: Image.Image | None = None,
    max_side: int = MAX_ANALYSIS_IMAGE_SIDE,
) -> tuple[Image.Image, Image.Image | None]:
    if max(image.size) <= max_side:
        return image, mask
    scale = max_side / max(image.size)
    size = (
        max(1, int(round(image.width * scale))),
        max(1, int(round(image.height * scale))),
    )
    resized_image = image.resize(size, Image.Resampling.BILINEAR)
    resized_mask = (
        mask.resize(size, Image.Resampling.NEAREST) if mask is not None else None
    )
    return resized_image, resized_mask


def _border_samples(image: Image.Image) -> list[tuple[int, int, int]]:
    pixels = image.load()
    width, height = image.size
    step = max(1, min(width, height) // 256)
    samples: list[tuple[int, int, int]] = []
    for x in range(0, width, step):
        samples.append(pixels[x, 0])
        samples.append(pixels[x, height - 1])
    for y in range(0, height, step):
        samples.append(pixels[0, y])
        samples.append(pixels[width - 1, y])
    return samples


def _rgb_distance(left: Sequence[int], right: Sequence[int]) -> float:
    return math.sqrt(
        sum((int(left[index]) - int(right[index])) ** 2 for index in range(3))
    )


def _connected_foreground(mask: Image.Image) -> tuple[Image.Image, dict[str, Any]]:
    """Remove small/border-connected background artefacts conservatively."""

    width, height = mask.size
    flags = bytearray(1 if value else 0 for value in _image_data(mask))
    visited = bytearray(len(flags))
    components: list[tuple[list[int], bool]] = []
    for start, enabled in enumerate(flags):
        if not enabled or visited[start]:
            continue
        visited[start] = 1
        queue: deque[int] = deque([start])
        indices: list[int] = []
        touches_border = False
        while queue:
            index = queue.popleft()
            indices.append(index)
            x = index % width
            y = index // width
            if x == 0 or y == 0 or x + 1 == width or y + 1 == height:
                touches_border = True
            if x and flags[index - 1] and not visited[index - 1]:
                visited[index - 1] = 1
                queue.append(index - 1)
            if x + 1 < width and flags[index + 1] and not visited[index + 1]:
                visited[index + 1] = 1
                queue.append(index + 1)
            if y and flags[index - width] and not visited[index - width]:
                visited[index - width] = 1
                queue.append(index - width)
            below = index + width
            if y + 1 < height and flags[below] and not visited[below]:
                visited[below] = 1
                queue.append(below)
        components.append((indices, touches_border))

    if not components:
        return Image.new("L", mask.size, 0), {
            "component_count": 0,
            "kept_component_count": 0,
            "largest_component_pixels": 0,
        }
    largest = max(len(indices) for indices, _ in components)
    minimum = max(12, int(math.ceil(largest * 0.002)))
    kept: list[list[int]] = []
    for indices, touches_border in components:
        if len(indices) < minimum:
            continue
        if touches_border and len(indices) < largest * 0.50:
            continue
        kept.append(indices)
    if not kept:
        kept = [max(components, key=lambda item: len(item[0]))[0]]

    output = bytearray(width * height)
    for indices in kept:
        for index in indices:
            output[index] = 255
    return Image.frombytes("L", mask.size, bytes(output)), {
        "component_count": len(components),
        "kept_component_count": len(kept),
        "largest_component_pixels": largest,
        "minimum_kept_component_pixels": minimum,
    }


def _infer_reference_mask(image: Image.Image) -> tuple[Image.Image, dict[str, Any]]:
    image, _ = _resize_pair(image)
    border = _border_samples(image)
    background = tuple(
        int(round(statistics.median(pixel[channel] for pixel in border)))
        for channel in range(3)
    )
    border_distances = sorted(_rgb_distance(pixel, background) for pixel in border)
    percentile_index = min(
        len(border_distances) - 1,
        max(0, int(math.floor(len(border_distances) * 0.95))),
    )
    threshold = min(45.0, max(12.0, border_distances[percentile_index] + 6.0))
    values = bytearray(
        255 if _rgb_distance(pixel, background) >= threshold else 0
        for pixel in _image_data(image)
    )
    raw = Image.frombytes("L", image.size, bytes(values))
    # Remove thin CAD-viewer axes/grid overlays before closing tiny capture
    # gaps.  JPEG colour fringes make nominal one-pixel guide lines several
    # pixels wide; if they touch the object, connected-component filtering
    # alone cannot remove them and the inferred bbox spans the whole frame.
    # The opening intentionally drops features too thin to be dependable
    # silhouette evidence; material pixels are still measured from the
    # original image after registration.
    opened = raw.filter(ImageFilter.MinFilter(5)).filter(ImageFilter.MaxFilter(5))
    closed = opened.filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.MinFilter(3))
    filtered, component_stats = _connected_foreground(closed)
    return filtered, {
        "method": "border_background_connected_components",
        "estimated_background_rgb": list(background),
        "background_distance_threshold": round(threshold, 6),
        "thin_overlay_opening_size": 5,
        **component_stats,
    }


def _mask_metrics(mask: Image.Image) -> dict[str, Any]:
    count = mask.histogram()[255]
    total = mask.width * mask.height
    bbox = mask.getbbox()
    if not bbox or count <= 0:
        return {
            "pixel_count": 0,
            "coverage": 0.0,
            "bbox": None,
            "bbox_aspect_ratio": None,
            "bbox_fill_ratio": None,
        }
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    return {
        "pixel_count": count,
        "coverage": count / total,
        "bbox": list(bbox),
        "bbox_aspect_ratio": width / max(1, height),
        "bbox_fill_ratio": count / max(1, width * height),
    }


def _part_color(part_id: str) -> tuple[int, int, int]:
    """Mirror render_part_views' stable part-ID colour encoding."""

    number = int(part_id[1:]) if part_id[1:].isdigit() else sum(map(ord, part_id))
    hue = (number * 0.618033988749895) % 1.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.72, 0.95)
    return int(red * 255), int(green * 255), int(blue * 255)


def _render_mask(
    part_ids_image: Image.Image,
    visible_parts: list[dict[str, Any]],
    registered_part_ids: set[str],
) -> tuple[Image.Image, dict[str, Any]]:
    part_ids: list[str] = []
    seen_part_ids: set[str] = set()
    declared_pixels = 0
    for item in visible_parts:
        if not isinstance(item, dict) or not isinstance(item.get("part_id"), str):
            raise ComparisonInputError(
                "render_set visible_parts entries require part_id"
            )
        part_id = item["part_id"]
        if part_id not in registered_part_ids:
            raise ComparisonInputError(
                f"render_set visible part is absent from registry parts: {part_id}"
            )
        if part_id in seen_part_ids:
            raise ComparisonInputError(
                f"render_set visible_parts contains duplicate part ID: {part_id}"
            )
        seen_part_ids.add(part_id)
        pixels = item.get("pixels")
        if isinstance(pixels, bool) or not isinstance(pixels, int) or pixels < 0:
            raise ComparisonInputError(
                "render_set visible_parts pixels must be non-negative integers"
            )
        part_ids.append(part_id)
        declared_pixels += pixels
    if not part_ids or declared_pixels <= 0:
        raise ComparisonInputError("render_set view has no visible registered parts")
    allowed = {_part_color(part_id) for part_id in part_ids}
    values = bytearray(
        255 if pixel in allowed else 0 for pixel in _image_data(part_ids_image)
    )
    mask = Image.frombytes("L", part_ids_image.size, bytes(values))
    actual_pixels = mask.histogram()[255]
    retained_ratio = actual_pixels / declared_pixels
    if retained_ratio < 0.70 or retained_ratio > 1.01:
        raise ComparisonInputError(
            "part-ID image is inconsistent with registry visible pixel counts "
            f"(retained ratio {retained_ratio:.3f})"
        )
    return mask, {
        "declared_visible_pixels": declared_pixels,
        "decoded_visible_pixels": actual_pixels,
        "decoded_to_declared_ratio": retained_ratio,
        "part_id_count": len(part_ids),
    }


def _render_target_mask(
    part_ids_image: Image.Image,
    visible_parts: list[dict[str, Any]],
    target_part_ids: set[str],
) -> tuple[Image.Image, dict[str, Any]]:
    """Decode only requested part IDs while preserving invisible-view evidence."""

    declared_by_part: dict[str, int] = {}
    for item in visible_parts:
        if not isinstance(item, Mapping):
            continue
        part_id = item.get("part_id")
        pixels = item.get("pixels")
        if (
            isinstance(part_id, str)
            and part_id in target_part_ids
            and isinstance(pixels, int)
            and not isinstance(pixels, bool)
        ):
            declared_by_part[part_id] = pixels
    allowed = {_part_color(part_id) for part_id in target_part_ids}
    values = bytearray(
        255 if pixel in allowed else 0 for pixel in _image_data(part_ids_image)
    )
    mask = Image.frombytes("L", part_ids_image.size, bytes(values))
    actual_pixels = _mask_count(mask)
    declared_pixels = sum(declared_by_part.values())
    if declared_pixels <= 0:
        if actual_pixels:
            raise ComparisonInputError(
                "part-ID image contains target parts absent from visible_parts"
            )
        retained_ratio: float | None = None
    else:
        retained_ratio = actual_pixels / declared_pixels
        if retained_ratio < 0.70 or retained_ratio > 1.01:
            raise ComparisonInputError(
                "target part-ID image is inconsistent with registry visible pixel "
                f"counts (retained ratio {retained_ratio:.3f})"
            )
    return mask, {
        "method": "requested_part_id_color_decode",
        "requested_part_ids": sorted(target_part_ids),
        "declared_visible_target_part_ids": sorted(
            part_id for part_id, pixels in declared_by_part.items() if pixels > 0
        ),
        "declared_visible_pixels": declared_pixels,
        "decoded_visible_pixels": actual_pixels,
        "decoded_to_declared_ratio": retained_ratio,
        "target_visible": actual_pixels > 0,
    }


def _trusted_evidence(view: dict[str, Any], manifest_path: Path) -> dict[str, Any]:
    reasons: list[str] = []
    if view.get("palette_status") != "usable":
        reasons.append("palette_status_not_usable")
    artifacts = view.get("palette_artifacts")
    if not isinstance(artifacts, dict):
        reasons.append("missing_palette_artifacts")
        return {"usable": False, "reasons": reasons, "samples": [], "sample_count": 0}
    audit_value = artifacts.get("normalized_evidence_audit") or artifacts.get(
        "evidence_audit"
    )
    if not audit_value:
        reasons.append("missing_evidence_audit")
        return {"usable": False, "reasons": reasons, "samples": [], "sample_count": 0}
    try:
        audit_path = _resolve_file(audit_value, manifest_path, "palette evidence audit")
        audit = _load_json_object(audit_path, "palette evidence audit")
    except ComparisonInputError as exc:
        reasons.append(f"invalid_evidence_audit:{exc}")
        return {"usable": False, "reasons": reasons, "samples": [], "sample_count": 0}

    samples: list[dict[str, Any]] = []
    groups = audit.get("groups")
    if not isinstance(groups, list):
        reasons.append("evidence_audit_groups_missing")
        groups = []
    for group in groups:
        if not isinstance(group, dict) or group.get("accepted") is not True:
            continue
        base_color = group.get("base_color")
        if not isinstance(base_color, str) or not base_color:
            continue
        boxes = group.get("boxes")
        if not isinstance(boxes, list):
            continue
        for box in boxes:
            if not isinstance(box, dict) or box.get("accepted") is not True:
                continue
            representative = box.get("representative_srgb")
            matching_pixels = box.get(
                "matching_pixel_count", box.get("matching_pixels")
            )
            normalized_box = box.get("box")
            box_is_valid = (
                isinstance(normalized_box, list)
                and len(normalized_box) == 4
                and all(
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and 0 <= float(value) <= 1000
                    for value in normalized_box
                )
                and float(normalized_box[0]) < float(normalized_box[2])
                and float(normalized_box[1]) < float(normalized_box[3])
            )
            if (
                not isinstance(representative, list)
                or len(representative) != 3
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or not 0 <= value <= 255
                    for value in representative
                )
                or isinstance(matching_pixels, bool)
                or not isinstance(matching_pixels, int)
                or matching_pixels < MIN_TRUSTED_EVIDENCE_PIXELS
            ):
                continue
            samples.append(
                {
                    "group_id": group.get("group_id"),
                    "base_color": base_color.lower(),
                    "representative_srgb": representative,
                    "weight_pixels": matching_pixels,
                    # Qwen palette boxes use normalized 0..1000 image
                    # coordinates.  Global comparison never consumes this
                    # optional field; scoped comparison requires it unless an
                    # explicit per-group mask is present.
                    "box_normalized_1000": (
                        [float(value) for value in normalized_box]
                        if box_is_valid
                        else None
                    ),
                }
            )
    if not samples:
        reasons.append("no_trusted_accepted_evidence_boxes")
    return {
        "usable": not reasons and bool(samples),
        "reasons": reasons,
        "audit": str(audit_path),
        "audit_sha256": _sha256(audit_path),
        "samples": samples,
        "sample_count": len(samples),
        "total_weight_pixels": sum(sample["weight_pixels"] for sample in samples),
    }


def _load_target_group_scope(
    palette_fusion_path: Path,
    target_group_id: str,
    *,
    target_reference_view_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Resolve one canonical palette group to its per-view local group IDs."""

    fusion = _load_json_object(palette_fusion_path, "palette fusion")
    canonical_palette = fusion.get("canonical_palette")
    raw_groups = (
        canonical_palette.get("groups")
        if isinstance(canonical_palette, Mapping)
        else None
    )
    if not isinstance(raw_groups, list):
        raise ComparisonInputError(
            "palette fusion requires canonical_palette.groups for target-group scoring"
        )
    matches = [
        group
        for group in raw_groups
        if isinstance(group, Mapping) and group.get("group_id") == target_group_id
    ]
    if len(matches) != 1:
        raise ComparisonInputError(
            f"target group {target_group_id!r} must occur exactly once in palette fusion"
        )
    sources = matches[0].get("sources")
    if not isinstance(sources, list) or not sources:
        raise ComparisonInputError(
            f"target group {target_group_id!r} has no reference-view sources"
        )
    local_group_ids_by_view: dict[str, str] = {}
    for source in sources:
        if not isinstance(source, Mapping):
            raise ComparisonInputError(
                f"target group {target_group_id!r} has an invalid source entry"
            )
        view_id = source.get("view_id")
        local_group_id = source.get("local_group_id")
        if (
            not isinstance(view_id, str)
            or not view_id
            or not isinstance(local_group_id, str)
            or not local_group_id
        ):
            raise ComparisonInputError(
                f"target group {target_group_id!r} sources require view_id and local_group_id"
            )
        previous = local_group_ids_by_view.get(view_id)
        if previous is not None and previous != local_group_id:
            raise ComparisonInputError(
                f"target group {target_group_id!r} repeats reference view {view_id!r}"
            )
        local_group_ids_by_view[view_id] = local_group_id

    raw_maps = fusion.get("view_group_id_maps")
    if isinstance(raw_maps, Mapping):
        for view_id, local_group_id in local_group_ids_by_view.items():
            view_map = raw_maps.get(view_id)
            mapped = (
                view_map.get(local_group_id) if isinstance(view_map, Mapping) else None
            )
            if mapped is not None and mapped != target_group_id:
                raise ComparisonInputError(
                    "palette fusion source and view_group_id_maps disagree for "
                    f"{view_id}:{local_group_id}"
                )
    canonical_local_group_ids_by_view = dict(sorted(local_group_ids_by_view.items()))
    canonical_reference_view_ids = sorted(canonical_local_group_ids_by_view)
    if target_reference_view_ids is None:
        scoring_reference_view_ids = canonical_reference_view_ids
        reference_scope_mode = "all_canonical_sources"
    else:
        if isinstance(target_reference_view_ids, (str, bytes)):
            raise ComparisonInputError(
                "target_reference_view_ids must be an iterable of view IDs"
            )
        raw_scoring_view_ids = list(target_reference_view_ids)
        if (
            len(raw_scoring_view_ids) < 2
            or any(
                not isinstance(view_id, str) or not view_id
                for view_id in raw_scoring_view_ids
            )
            or len(set(raw_scoring_view_ids)) != len(raw_scoring_view_ids)
        ):
            raise ComparisonInputError(
                "target_reference_view_ids must contain at least two unique "
                "non-empty view IDs"
            )
        unknown_scoring_views = sorted(
            set(raw_scoring_view_ids) - set(canonical_reference_view_ids)
        )
        if unknown_scoring_views:
            raise ComparisonInputError(
                "target reference scoring scope contains non-canonical views: "
                + ", ".join(unknown_scoring_views)
            )
        scoring_reference_view_ids = sorted(raw_scoring_view_ids)
        reference_scope_mode = "explicit_canonical_source_subset"
    scoring_local_group_ids_by_view = {
        view_id: canonical_local_group_ids_by_view[view_id]
        for view_id in scoring_reference_view_ids
    }
    excluded_reference_sources = [
        {
            "reference_view_id": view_id,
            "local_group_id": canonical_local_group_ids_by_view[view_id],
            "reason": "EXCLUDED_BY_TRUSTED_SCORING_SCOPE",
        }
        for view_id in canonical_reference_view_ids
        if view_id not in scoring_local_group_ids_by_view
    ]
    return {
        "target_group_id": target_group_id,
        "palette_fusion": str(palette_fusion_path),
        "palette_fusion_sha256": _sha256(palette_fusion_path),
        "local_group_ids_by_view": scoring_local_group_ids_by_view,
        "reference_view_ids": scoring_reference_view_ids,
        "canonical_local_group_ids_by_view": canonical_local_group_ids_by_view,
        "canonical_reference_view_ids": canonical_reference_view_ids,
        "reference_scope_mode": reference_scope_mode,
        "excluded_reference_sources": excluded_reference_sources,
    }


def _scoped_reference_evidence(
    evidence: Mapping[str, Any],
    *,
    local_group_id: str,
) -> dict[str, Any]:
    samples = [
        dict(sample)
        for sample in evidence.get("samples", [])
        if isinstance(sample, Mapping) and sample.get("group_id") == local_group_id
    ]
    if not evidence.get("usable") or not samples:
        raise ComparisonInputError(
            f"target reference group {local_group_id!r} lacks trusted accepted evidence"
        )
    return {
        **dict(evidence),
        "samples": samples,
        "sample_count": len(samples),
        "total_weight_pixels": sum(int(sample["weight_pixels"]) for sample in samples),
        "target_local_group_id": local_group_id,
        "target_group_filter_applied": True,
    }


def _reference_group_mask(
    *,
    image: Image.Image,
    global_mask: Image.Image,
    evidence: Mapping[str, Any],
    explicit_mask: Image.Image | None,
    explicit_mask_path: Path | None,
) -> tuple[Image.Image, dict[str, Any]]:
    """Build a local reference mask from an explicit mask or trusted Qwen ROIs."""

    if explicit_mask is not None:
        resized = explicit_mask.resize(image.size, Image.Resampling.NEAREST)
        mask = ImageChops.multiply(global_mask, resized)
        return mask, {
            "method": "explicit_target_group_mask_intersected_foreground",
            "mask": str(explicit_mask_path),
            "mask_sha256": (
                _sha256(explicit_mask_path) if explicit_mask_path is not None else None
            ),
            "target_local_group_id": evidence["target_local_group_id"],
        }

    samples = evidence.get("samples")
    assert isinstance(samples, list)
    normalized_boxes: list[list[float]] = []
    families: set[str] = set()
    for sample in samples:
        if not isinstance(sample, Mapping):
            continue
        raw_box = sample.get("box_normalized_1000")
        if (
            not isinstance(raw_box, list)
            or len(raw_box) != 4
            or not all(
                isinstance(value, (int, float)) and not isinstance(value, bool)
                for value in raw_box
            )
        ):
            raise ComparisonInputError(
                "target reference evidence needs normalized 0..1000 boxes "
                "when no explicit target_group_masks entry is provided"
            )
        normalized_boxes.append([float(value) for value in raw_box])
        base_color = sample.get("base_color")
        if isinstance(base_color, str):
            families.update(_evidence_family_bins(base_color))
    if not normalized_boxes:
        raise ComparisonInputError("target reference group has no usable ROI boxes")

    roi = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(roi)
    pixel_boxes: list[list[int]] = []
    for left, top, right, bottom in normalized_boxes:
        pixel_box = [
            max(0, int(math.floor(left * image.width / 1000.0))),
            max(0, int(math.floor(top * image.height / 1000.0))),
            min(image.width, int(math.ceil(right * image.width / 1000.0))),
            min(image.height, int(math.ceil(bottom * image.height / 1000.0))),
        ]
        if pixel_box[0] >= pixel_box[2] or pixel_box[1] >= pixel_box[3]:
            raise ComparisonInputError("target reference ROI collapses after scaling")
        # Pillow rectangle endpoints are inclusive.  Subtract one from the
        # exclusive right/bottom coordinates used everywhere else.
        draw.rectangle(
            (
                pixel_box[0],
                pixel_box[1],
                pixel_box[2] - 1,
                pixel_box[3] - 1,
            ),
            fill=255,
        )
        pixel_boxes.append(pixel_box)
    roi = ImageChops.multiply(roi, global_mask)

    # Tight Qwen boxes can still contain neighbouring parts.  Restrict the
    # derived mask to the trusted coarse colour family.  An unknown family
    # intentionally falls back to the ROI instead of inventing a category.
    if families:
        roi_pixels = roi.load()
        image_pixels = image.load()
        values = bytearray(image.width * image.height)
        for y in range(image.height):
            for x in range(image.width):
                if roi_pixels[x, y] and _color_bin(*image_pixels[x, y])[0] in families:
                    values[y * image.width + x] = 255
        mask = Image.frombytes("L", image.size, bytes(values))
        method = "trusted_group_roi_and_color_family_intersected_foreground"
    else:
        mask = roi
        method = "trusted_group_roi_intersected_foreground"
    return mask, {
        "method": method,
        "target_local_group_id": evidence["target_local_group_id"],
        "normalized_boxes_1000": normalized_boxes,
        "pixel_boxes": pixel_boxes,
        "trusted_color_bins": sorted(families),
    }


def _load_reference_views(
    manifest_path: Path,
    *,
    target_local_group_ids_by_view: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    manifest = _load_json_object(manifest_path, "reference manifest")
    source_views = manifest.get("source_views")
    if not isinstance(source_views, list) or not source_views:
        raise ComparisonInputError("reference manifest requires non-empty source_views")
    seen: set[str] = set()
    records: list[dict[str, Any]] = []
    for raw_view in source_views:
        if not isinstance(raw_view, dict):
            raise ComparisonInputError("reference source_views entries must be objects")
        view_id = raw_view.get("id")
        if not isinstance(view_id, str) or not view_id or view_id in seen:
            raise ComparisonInputError(
                "reference view IDs must be unique non-empty strings"
            )
        seen.add(view_id)
        if (
            target_local_group_ids_by_view is not None
            and view_id not in target_local_group_ids_by_view
        ):
            continue
        image_path = _resolve_file(
            raw_view.get("image"), manifest_path, f"reference image {view_id}"
        )
        image = _open_rgb(image_path, f"reference image {view_id}")
        original_size = image.size
        local_group_id = (
            target_local_group_ids_by_view.get(view_id)
            if target_local_group_ids_by_view is not None
            else None
        )
        explicit_target_mask: Image.Image | None = None
        explicit_target_mask_path: Path | None = None
        if local_group_id is not None:
            raw_group_masks = raw_view.get("target_group_masks")
            if raw_group_masks is not None and not isinstance(raw_group_masks, Mapping):
                raise ComparisonInputError(
                    f"reference view {view_id} target_group_masks must be an object"
                )
            raw_target_mask = (
                raw_group_masks.get(local_group_id)
                if isinstance(raw_group_masks, Mapping)
                else None
            )
            if raw_target_mask is not None:
                explicit_target_mask_path = _resolve_file(
                    raw_target_mask,
                    manifest_path,
                    f"reference target group mask {view_id}:{local_group_id}",
                )
                explicit_target_mask = _open_mask(
                    explicit_target_mask_path,
                    original_size,
                    f"reference target group mask {view_id}:{local_group_id}",
                )
        if raw_view.get("palette_mask"):
            mask_path = _resolve_file(
                raw_view["palette_mask"], manifest_path, f"reference mask {view_id}"
            )
            mask = _open_mask(mask_path, image.size, f"reference mask {view_id}")
            image, resized_mask = _resize_pair(image, mask)
            assert resized_mask is not None
            mask = resized_mask
            mask_audit = {
                "method": "explicit_palette_mask",
                "mask": str(mask_path),
                "mask_sha256": _sha256(mask_path),
            }
        else:
            mask, mask_audit = _infer_reference_mask(image)
            image, _ = _resize_pair(image)
        alignment_mask = mask
        evidence = _trusted_evidence(raw_view, manifest_path)
        if local_group_id is not None:
            evidence = _scoped_reference_evidence(
                evidence,
                local_group_id=local_group_id,
            )
            mask, target_mask_audit = _reference_group_mask(
                image=image,
                global_mask=alignment_mask,
                evidence=evidence,
                explicit_mask=explicit_target_mask,
                explicit_mask_path=explicit_target_mask_path,
            )
        else:
            target_mask_audit = None
        records.append(
            {
                "view_id": view_id,
                "image_path": image_path,
                "image": image,
                "mask": mask,
                "mask_audit": mask_audit,
                "mask_metrics": _mask_metrics(mask),
                "alignment_mask": alignment_mask,
                "alignment_mask_metrics": _mask_metrics(alignment_mask),
                "target_mask_audit": target_mask_audit,
                "evidence": evidence,
                "image_sha256": _sha256(image_path),
            }
        )
    if target_local_group_ids_by_view is not None:
        unknown_target_views = sorted(set(target_local_group_ids_by_view) - seen)
        if unknown_target_views:
            raise ComparisonInputError(
                "target group references views absent from reference manifest: "
                + ", ".join(unknown_target_views)
            )
        if not records:
            raise ComparisonInputError(
                "target group has no reference views in the reference manifest"
            )
    return records


def _load_render_views(
    registry_path: Path,
    *,
    target_part_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    registry = _load_json_object(registry_path, "rendered registry")
    parts = registry.get("parts")
    render_set = registry.get("render_set")
    if not isinstance(parts, list) or not isinstance(render_set, dict):
        raise ComparisonInputError("rendered registry requires parts and render_set")
    registered_part_ids: set[str] = set()
    for part in parts:
        if not isinstance(part, dict) or not isinstance(part.get("part_id"), str):
            raise ComparisonInputError("registry parts entries require string part_id")
        part_id = part["part_id"]
        if not part_id or part_id in registered_part_ids:
            raise ComparisonInputError(
                "registry part IDs must be unique non-empty strings"
            )
        registered_part_ids.add(part_id)
    if not registered_part_ids:
        raise ComparisonInputError("rendered registry parts cannot be empty")
    if target_part_ids is not None:
        unknown_target_parts = sorted(target_part_ids - registered_part_ids)
        if unknown_target_parts:
            raise ComparisonInputError(
                "target part IDs are absent from rendered registry: "
                + ", ".join(unknown_target_parts)
            )
    views = render_set.get("views")
    if not isinstance(views, list) or not views:
        raise ComparisonInputError(
            "rendered registry render_set requires non-empty views"
        )
    seen: set[str] = set()
    records: list[dict[str, Any]] = []
    for raw_view in views:
        if not isinstance(raw_view, dict):
            raise ComparisonInputError("render_set views entries must be objects")
        view_id = raw_view.get("view_id")
        if not isinstance(view_id, str) or not view_id or view_id in seen:
            raise ComparisonInputError(
                "render view IDs must be unique non-empty strings"
            )
        seen.add(view_id)
        rgb_path = _resolve_file(
            raw_view.get("rgb"), registry_path, f"render RGB {view_id}"
        )
        part_ids_path = _resolve_file(
            raw_view.get("part_ids"), registry_path, f"part-ID image {view_id}"
        )
        rgb = _open_rgb(rgb_path, f"render RGB {view_id}")
        part_ids_image = _open_rgb(part_ids_path, f"part-ID image {view_id}")
        if rgb.size != part_ids_image.size:
            raise ComparisonInputError(
                f"render RGB and part-ID size mismatch for {view_id}: {rgb.size} vs {part_ids_image.size}"
            )
        visible_parts = raw_view.get("visible_parts")
        if not isinstance(visible_parts, list):
            raise ComparisonInputError(f"render view {view_id} requires visible_parts")
        alignment_mask, alignment_mask_audit = _render_mask(
            part_ids_image, visible_parts, registered_part_ids
        )
        if target_part_ids is not None:
            mask, target_mask_audit = _render_target_mask(
                part_ids_image,
                visible_parts,
                target_part_ids,
            )
        else:
            mask = alignment_mask
            target_mask_audit = None
        rgb, resized_mask = _resize_pair(rgb, mask)
        assert resized_mask is not None
        resized_alignment_mask = alignment_mask.resize(
            rgb.size,
            Image.Resampling.NEAREST,
        )
        records.append(
            {
                "view_id": view_id,
                "image_path": rgb_path,
                "part_ids_path": part_ids_path,
                "image": rgb,
                "mask": resized_mask,
                "mask_audit": alignment_mask_audit,
                "mask_metrics": _mask_metrics(resized_mask),
                "alignment_mask": resized_alignment_mask,
                "alignment_mask_metrics": _mask_metrics(resized_alignment_mask),
                "target_mask_audit": target_mask_audit,
                "image_sha256": _sha256(rgb_path),
                "part_ids_sha256": _sha256(part_ids_path),
            }
        )
    return records


def _normalized_mask(
    mask: Image.Image, size: int = NORMALIZED_MASK_SIZE
) -> Image.Image:
    bbox = mask.getbbox()
    if not bbox:
        return Image.new("L", (size, size), 0)
    crop = mask.crop(bbox)
    inner = size - 16
    scale = min(inner / crop.width, inner / crop.height)
    resized = crop.resize(
        (
            max(1, int(round(crop.width * scale))),
            max(1, int(round(crop.height * scale))),
        ),
        Image.Resampling.NEAREST,
    )
    canvas = Image.new("L", (size, size), 0)
    canvas.paste(resized, ((size - resized.width) // 2, (size - resized.height) // 2))
    return canvas.point(lambda value: 255 if value else 0, mode="L")


def _view_alignment_mask(view: Mapping[str, Any]) -> Image.Image:
    """Return the whole-object mask even when material scoring is local."""

    candidate = view.get("alignment_mask", view.get("mask"))
    if not isinstance(candidate, Image.Image):
        raise ComparisonInputError("view is missing a valid alignment mask")
    return candidate


def _mask_count(mask: Image.Image) -> int:
    return mask.histogram()[255]


def _intersection_count(left: Image.Image, right: Image.Image) -> int:
    return _mask_count(
        ImageChops.logical_and(left.convert("1"), right.convert("1")).convert("L")
    )


def _profile_similarity(left: Image.Image, right: Image.Image) -> float:
    def profiles(mask: Image.Image) -> tuple[list[float], list[float]]:
        pixels = mask.load()
        rows = [
            sum(1 for x in range(mask.width) if pixels[x, y]) / mask.width
            for y in range(mask.height)
        ]
        columns = [
            sum(1 for y in range(mask.height) if pixels[x, y]) / mask.height
            for x in range(mask.width)
        ]
        return rows, columns

    left_rows, left_columns = profiles(left)
    right_rows, right_columns = profiles(right)
    row_score = 1.0 - sum(abs(a - b) for a, b in zip(left_rows, right_rows)) / len(
        left_rows
    )
    column_score = 1.0 - sum(
        abs(a - b) for a, b in zip(left_columns, right_columns)
    ) / len(left_columns)
    return max(0.0, min(1.0, (row_score + column_score) * 0.5))


def _alignment_metrics(
    reference_mask: Image.Image, render_mask: Image.Image
) -> dict[str, float]:
    left = _normalized_mask(reference_mask)
    right = _normalized_mask(render_mask)
    intersection = _intersection_count(left, right)
    union = _mask_count(
        ImageChops.logical_or(left.convert("1"), right.convert("1")).convert("L")
    )
    silhouette_iou = intersection / union if union else 0.0

    left_edge = ImageChops.subtract(left, left.filter(ImageFilter.MinFilter(3)))
    right_edge = ImageChops.subtract(right, right.filter(ImageFilter.MinFilter(3)))
    left_edge_count = _mask_count(left_edge)
    right_edge_count = _mask_count(right_edge)
    left_near_right = _intersection_count(
        left_edge, right_edge.filter(ImageFilter.MaxFilter(7))
    )
    right_near_left = _intersection_count(
        right_edge, left_edge.filter(ImageFilter.MaxFilter(7))
    )
    precision = left_near_right / left_edge_count if left_edge_count else 0.0
    recall = right_near_left / right_edge_count if right_edge_count else 0.0
    edge_f1 = (
        2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    )

    left_metrics = _mask_metrics(reference_mask)
    right_metrics = _mask_metrics(render_mask)
    left_aspect = float(left_metrics["bbox_aspect_ratio"] or 1e-6)
    right_aspect = float(right_metrics["bbox_aspect_ratio"] or 1e-6)
    aspect_similarity = math.exp(-abs(math.log(left_aspect / right_aspect)))
    profile_similarity = _profile_similarity(left, right)
    score = (
        0.30 * silhouette_iou
        + 0.30 * edge_f1
        + 0.25 * profile_similarity
        + 0.15 * aspect_similarity
    )
    return {
        "score": score,
        "difference_score": 1.0 - score,
        "silhouette_iou": silhouette_iou,
        "edge_f1_tolerance_3px": edge_f1,
        "profile_similarity": profile_similarity,
        "bbox_aspect_similarity": aspect_similarity,
    }


def _color_bin(red: int, green: int, blue: int) -> tuple[str, float, float, float]:
    hue, saturation, value = colorsys.rgb_to_hsv(
        red / 255.0, green / 255.0, blue / 255.0
    )
    degrees = hue * 360.0
    if value < 0.16:
        label = "black"
    elif saturation < 0.14:
        if value < 0.38:
            label = "achromatic_dark"
        elif value < 0.78:
            label = "achromatic_mid"
        else:
            label = "achromatic_light"
    elif degrees < 20.0 or degrees >= 345.0:
        label = "red"
    elif degrees < 55.0:
        label = "orange_brown"
    elif degrees < 75.0:
        label = "yellow"
    elif degrees < 170.0:
        label = "green"
    elif degrees < 260.0:
        label = "cyan_blue"
    else:
        label = "purple"
    return label, hue, saturation, value


def _color_distribution(image: Image.Image, mask: Image.Image) -> dict[str, Any]:
    image, resized_mask = _resize_pair(image, mask)
    assert resized_mask is not None
    image_pixels = image.load()
    mask_pixels = resized_mask.load()
    foreground = _mask_count(resized_mask)
    step = max(1, int(math.sqrt(max(1, foreground) / 100_000)))
    bins: Counter[str] = Counter()
    hue_bins = [0] * 12
    saturations: list[float] = []
    values: list[float] = []
    chromatic = 0
    sampled = 0
    for y in range(0, image.height, step):
        for x in range(0, image.width, step):
            if not mask_pixels[x, y]:
                continue
            label, hue, saturation, value = _color_bin(*image_pixels[x, y])
            bins[label] += 1
            saturations.append(saturation)
            values.append(value)
            sampled += 1
            if saturation >= 0.18 and value >= 0.12:
                hue_bins[min(11, int(hue * 12.0))] += 1
                chromatic += 1
    if not sampled:
        return {
            "sampled_pixels": 0,
            "category_distribution": {label: 0.0 for label in _COLOR_BINS},
            "hue_distribution_12": [0.0] * 12,
            "chromatic_coverage": 0.0,
            "median_saturation": None,
            "median_value": None,
            "dominant_categories": [],
        }
    category_distribution = {label: bins[label] / sampled for label in _COLOR_BINS}
    hue_distribution = [value / chromatic if chromatic else 0.0 for value in hue_bins]
    dominant = [
        {"category": label, "share": share}
        for label, share in sorted(
            category_distribution.items(), key=lambda item: (-item[1], item[0])
        )[:4]
        if share > 0
    ]
    return {
        "sampled_pixels": sampled,
        "sample_step": step,
        "category_distribution": category_distribution,
        "hue_distribution_12": hue_distribution,
        "chromatic_coverage": chromatic / sampled,
        "median_saturation": statistics.median(saturations),
        "median_value": statistics.median(values),
        "dominant_categories": dominant,
    }


def _histogram_intersection(left: Iterable[float], right: Iterable[float]) -> float:
    return sum(min(a, b) for a, b in zip(left, right))


def _evidence_family_bins(base_color: str) -> set[str]:
    normalized = base_color.lower()
    if normalized in {"white", "gray", "grey", "silver", "clear"}:
        return {"achromatic_dark", "achromatic_mid", "achromatic_light"}
    if normalized == "black":
        return {"black", "achromatic_dark"}
    if normalized in {"brown", "orange", "copper", "bronze"}:
        return {"orange_brown", "red"}
    if normalized in {"blue", "cyan"}:
        return {"cyan_blue"}
    if normalized in {"pink", "purple", "magenta"}:
        return {"purple", "red"}
    if normalized in _COLOR_BINS:
        return {normalized}
    return set()


def _evidence_recall(
    evidence: dict[str, Any], render_distribution: dict[str, Any]
) -> float | None:
    samples = evidence.get("samples")
    if not isinstance(samples, list) or not samples:
        return None
    shares = render_distribution["category_distribution"]
    numerator = 0.0
    denominator = 0.0
    for sample in samples:
        family = _evidence_family_bins(sample["base_color"])
        if not family:
            continue
        weight = float(sample["weight_pixels"])
        presence = sum(shares.get(label, 0.0) for label in family)
        numerator += weight * min(1.0, presence / 0.01)
        denominator += weight
    return numerator / denominator if denominator else None


def _evidence_group_recall(
    evidence: dict[str, Any],
    reference_distribution: dict[str, Any],
    render_distribution: dict[str, Any],
) -> dict[str, Any] | None:
    """Measure each accepted palette group instead of letting large groups win.

    The legacy weighted recall can hide a missing copper tube or blue valve cap
    behind a large green enclosure.  This companion metric gives every
    independently accepted group one vote while scaling its required render
    presence by the observed reference color share.  A fixed one-percent cap
    lets a narrow correctly coloured edge falsely stand in for a dominant
    enclosure, so dominant and accent groups use the same relative-recall
    contract.
    """

    samples = evidence.get("samples")
    if not isinstance(samples, list) or not samples:
        return None
    grouped: dict[str, list[dict[str, Any]]] = {}
    for index, sample in enumerate(samples):
        group_id = sample.get("group_id")
        key = (
            str(group_id)
            if isinstance(group_id, str) and group_id
            else f"{sample.get('base_color', 'unknown')}:{index}"
        )
        grouped.setdefault(key, []).append(sample)
    reference_foreground_pixels = max(
        1.0,
        float(reference_distribution.get("sampled_pixels") or 0.0),
    )
    reference_shares = reference_distribution["category_distribution"]
    render_shares = render_distribution["category_distribution"]
    records: list[dict[str, Any]] = []
    for group_id, group_samples in sorted(grouped.items()):
        family: set[str] = set()
        weight = 0.0
        colors: set[str] = set()
        for sample in group_samples:
            color = str(sample["base_color"])
            colors.add(color)
            family.update(_evidence_family_bins(color))
            weight += float(sample["weight_pixels"])
        if not family:
            continue
        reference_color_share = sum(
            reference_shares.get(label, 0.0) for label in family
        )
        # Evidence-box weights are pixel footprints in the reference image.
        # Dividing by the sum of accepted evidence weights makes a small part
        # look artificially large whenever the evidence boxes cover only a
        # fraction of the object.  Normalize by the actual reference
        # foreground instead.  The independent dominant-mass gate below still
        # checks whole-object colour coverage for large material families.
        reference_evidence_share = min(
            1.0,
            weight / reference_foreground_pixels,
        )
        # Whole-image hue bins can include lighting, antialiasing, or another
        # group of the same coarse colour.  The localized accepted evidence
        # footprint is the conservative upper bound for this particular
        # group.
        reference_group_share = min(
            reference_color_share,
            reference_evidence_share,
        )
        observed_share = sum(render_shares.get(label, 0.0) for label in family)
        # Permit a twofold view/lighting difference while still requiring the
        # render to carry a meaningful fraction of a dominant appearance.
        required_share = max(0.001, min(0.95, reference_group_share * 0.50))
        recall = min(1.0, observed_share / required_share)
        records.append(
            {
                "group_id": group_id,
                "base_colors": sorted(colors),
                "render_color_bins": sorted(family),
                "reference_evidence_weight": int(round(weight)),
                "reference_evidence_share": reference_evidence_share,
                "reference_foreground_pixels": int(round(reference_foreground_pixels)),
                "reference_group_share_basis": (
                    "trusted_evidence_footprint_capped_color_share"
                ),
                "reference_color_share": reference_color_share,
                "reference_group_share": reference_group_share,
                "required_render_share": required_share,
                "observed_render_share": observed_share,
                "recall": recall,
            }
        )
    if not records:
        return None
    return {
        "group_count": len(records),
        "macro_recall": sum(record["recall"] for record in records) / len(records),
        "minimum_group_recall": min(record["recall"] for record in records),
        "groups": records,
    }


def _annotate_group_delivery_presence(
    group_recall: Mapping[str, Any] | None,
    *,
    thresholds: Mapping[str, float | int],
) -> bool:
    """Mark delivery-relevant missing groups and return whether any are missing.

    Raw recall remains untouched.  A tiny accepted palette region is subject to
    whole-image bin quantization, so a nonzero render observation that is within
    ten percent of the regular recall boundary is recorded as a bounded
    low-evidence presence instead of turning an otherwise matching view into a
    hard material failure.  Zero observations, larger regions, and deficits
    outside that narrow band remain failures.
    """

    if not isinstance(group_recall, Mapping):
        return False
    raw_groups = group_recall.get("groups")
    if not isinstance(raw_groups, list):
        return False
    minimum_recall = float(thresholds["minimum_evidence_group_recall"])
    minimum_pixels = int(thresholds["minimum_reliable_group_evidence_pixels"])
    tolerance_ratio = float(thresholds["low_evidence_recall_tolerance_ratio"])
    minimum_observed = float(thresholds["minimum_low_evidence_observed_render_share"])
    missing = False
    for raw_group in raw_groups:
        if not isinstance(raw_group, dict):
            continue
        recall = float(raw_group["recall"])
        evidence_pixels = int(raw_group["reference_evidence_weight"])
        observed_share = float(raw_group["observed_render_share"])
        if recall >= minimum_recall:
            status = "PRESENT"
        elif (
            evidence_pixels < minimum_pixels
            and recall >= minimum_recall * tolerance_ratio
            and observed_share >= minimum_observed
        ):
            status = "LOW_EVIDENCE_NEAR_THRESHOLD_PRESENT"
        else:
            status = "MISSING"
            missing = True
        raw_group["delivery_presence_status"] = status
    return missing


def _dominant_evidence_mass(
    evidence: dict[str, Any],
    reference_distribution: dict[str, Any],
    render_distribution: dict[str, Any],
    *,
    alignment: Mapping[str, Any],
    thresholds: Mapping[str, float | int],
    reference_target_mask_audit: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Measure dominant trusted colour-family mass without assigning a part.

    Palette groups that map to the same coarse render bins are pooled so one
    achromatic pixel population cannot independently satisfy both a white and
    a silver group.  This is a delivery gate only: local group IDs are emitted
    for a later, independently localized repair join, but this metric never
    selects a part or material by itself.
    """

    raw_samples = evidence.get("samples")
    if not isinstance(raw_samples, list):
        raw_samples = []
    grouped: dict[tuple[str, ...], dict[str, set[str]]] = {}
    for sample in raw_samples:
        if not isinstance(sample, Mapping):
            continue
        group_id = sample.get("group_id")
        base_color = sample.get("base_color")
        if (
            not isinstance(group_id, str)
            or not group_id
            or not isinstance(base_color, str)
            or not base_color
        ):
            continue
        family = tuple(sorted(_evidence_family_bins(base_color)))
        if not family:
            continue
        record = grouped.setdefault(
            family,
            {"local_group_ids": set(), "base_colors": set()},
        )
        record["local_group_ids"].add(group_id)
        record["base_colors"].add(base_color.lower())

    reference_shares = reference_distribution["category_distribution"]
    render_shares = render_distribution["category_distribution"]
    family_shares = {
        family: sum(float(reference_shares.get(label, 0.0)) for label in family)
        for family in grouped
    }
    minimum_reference_share = float(thresholds["minimum_dominant_reference_share"])
    minimum_share_margin = float(thresholds["minimum_dominant_share_margin"])
    minimum_mass_recall = float(thresholds["minimum_dominant_mass_recall"])
    minimum_absolute_deficit = float(thresholds["minimum_dominant_absolute_deficit"])
    minimum_silhouette_iou = float(thresholds["minimum_dominant_silhouette_iou"])
    strong_alignment = float(thresholds["strong_alignment_score"])
    alignment_score = float(alignment["score"])
    silhouette_iou = float(alignment["silhouette_iou"])

    reference_mask_color_censored = (
        isinstance(reference_target_mask_audit, Mapping)
        and reference_target_mask_audit.get("method")
        == "trusted_group_roi_and_color_family_intersected_foreground"
    )
    records: list[dict[str, Any]] = []
    for family in sorted(grouped):
        reference_share = family_shares[family]
        runner_up_share = max(
            (
                share
                for other_family, share in family_shares.items()
                if other_family != family
            ),
            default=0.0,
        )
        reference_share_margin = reference_share - runner_up_share
        observed_share = sum(float(render_shares.get(label, 0.0)) for label in family)
        deficit_share = max(0.0, reference_share - observed_share)
        mass_recall = (
            min(1.0, observed_share / reference_share) if reference_share > 0.0 else 1.0
        )
        eligibility_reasons: list[str] = []
        if reference_share < minimum_reference_share:
            eligibility_reasons.append("REFERENCE_SHARE_BELOW_DOMINANT_FLOOR")
        if reference_share_margin < minimum_share_margin:
            eligibility_reasons.append("REFERENCE_DOMINANCE_MARGIN_BELOW_FLOOR")
        if alignment_score < strong_alignment:
            eligibility_reasons.append("ALIGNMENT_NOT_STRONG")
        if silhouette_iou < minimum_silhouette_iou:
            eligibility_reasons.append("SILHOUETTE_IOU_BELOW_DOMINANT_FLOOR")
        raw_eligible = not eligibility_reasons
        raw_hard_failure = (
            raw_eligible
            and mass_recall < minimum_mass_recall
            and deficit_share >= minimum_absolute_deficit
        )
        if reference_mask_color_censored:
            eligibility_reasons.append("REFERENCE_MASK_COLOR_CENSORED")
        eligible = not eligibility_reasons
        hard_failure = raw_hard_failure and not reference_mask_color_censored
        records.append(
            {
                "family_key": "|".join(family),
                "local_group_ids": sorted(grouped[family]["local_group_ids"]),
                "base_colors": sorted(grouped[family]["base_colors"]),
                "render_color_bins": list(family),
                "reference_share": reference_share,
                "runner_up_reference_share": runner_up_share,
                "reference_share_margin": reference_share_margin,
                "observed_render_share": observed_share,
                "deficit_share": deficit_share,
                "mass_recall": mass_recall,
                "raw_eligible": raw_eligible,
                "raw_status": (
                    "FAIL"
                    if raw_hard_failure
                    else "PASS"
                    if raw_eligible
                    else "NOT_APPLICABLE"
                ),
                "eligible": eligible,
                "status": (
                    "FAIL" if hard_failure else "PASS" if eligible else "NOT_APPLICABLE"
                ),
                "reason_codes": (
                    ["DOMINANT_FAMILY_MASS_DEFICIT"]
                    if hard_failure
                    else eligibility_reasons
                ),
            }
        )
    eligible_count = sum(1 for record in records if record["eligible"])
    failed_count = sum(1 for record in records if record["status"] == "FAIL")
    return {
        "status": (
            "FAIL" if failed_count else "PASS" if eligible_count else "NOT_APPLICABLE"
        ),
        "eligible_family_count": eligible_count,
        "failed_family_count": failed_count,
        "reference_mask_color_censored": reference_mask_color_censored,
        "decision_applicability": (
            "NOT_APPLICABLE_REFERENCE_MASK_COLOR_CENSORED"
            if reference_mask_color_censored
            else "APPLICABLE"
        ),
        "families": records,
    }


def _unreferenced_render_chromatic_mass(
    reference_distribution: Mapping[str, Any],
    render_distribution: Mapping[str, Any],
    *,
    alignment: Mapping[str, Any],
    thresholds: Mapping[str, float | int],
    evidence: Mapping[str, Any] | None = None,
    reference_target_mask_audit: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Detect large render-only chromatic masses in a strongly aligned view."""

    reference_categories = reference_distribution["category_distribution"]
    render_categories = render_distribution["category_distribution"]
    scoped_comparison = isinstance(reference_target_mask_audit, Mapping)
    reference_mask_color_censored = (
        scoped_comparison
        and reference_target_mask_audit.get("method")
        == "trusted_group_roi_and_color_family_intersected_foreground"
    )
    trusted_compatible_bins: set[str] = set()
    raw_samples = evidence.get("samples") if isinstance(evidence, Mapping) else None
    if scoped_comparison and isinstance(raw_samples, list):
        for sample in raw_samples:
            if not isinstance(sample, Mapping):
                continue
            base_color = sample.get("base_color")
            if isinstance(base_color, str) and base_color:
                trusted_compatible_bins.update(_evidence_family_bins(base_color))
    records: list[dict[str, Any]] = []
    strong_alignment = float(alignment["score"]) >= float(
        thresholds["strong_alignment_score"]
    )
    for color_bin in _CHROMATIC_COLOR_BINS:
        reference_share = float(reference_categories[color_bin])
        render_share = float(render_categories[color_bin])
        excess_share = max(0.0, render_share - reference_share)
        raw_failure = (
            strong_alignment
            and render_share
            >= float(thresholds["minimum_unreferenced_render_chromatic_share"])
            and reference_share
            <= float(thresholds["maximum_reference_share_for_unreferenced_chromatic"])
            and excess_share
            >= float(thresholds["minimum_unreferenced_render_chromatic_excess"])
        )
        trusted_family_compatible = color_bin in trusted_compatible_bins
        effective_failure = (
            raw_failure
            and not reference_mask_color_censored
            and not trusted_family_compatible
        )
        if reference_mask_color_censored:
            status = "NOT_APPLICABLE"
            reason_codes = ["REFERENCE_MASK_COLOR_CENSORED"]
        elif trusted_family_compatible:
            status = "PASS"
            reason_codes = ["TRUSTED_COMPATIBLE_COLOR_FAMILY"]
        else:
            status = "FAIL" if effective_failure else "PASS"
            reason_codes = []
        records.append(
            {
                "color_bin": color_bin,
                "reference_share": reference_share,
                "render_share": render_share,
                "raw_excess_share": excess_share,
                "effective_excess_share": (
                    0.0
                    if reference_mask_color_censored or trusted_family_compatible
                    else excess_share
                ),
                # Retained for schema compatibility.  Downstream decision code
                # consumes effective_excess_share.
                "excess_share": (
                    0.0
                    if reference_mask_color_censored or trusted_family_compatible
                    else excess_share
                ),
                "trusted_family_compatible": trusted_family_compatible,
                "raw_status": "FAIL" if raw_failure else "PASS",
                "status": status,
                "reason_codes": reason_codes,
            }
        )
    failures = [record["color_bin"] for record in records if record["status"] == "FAIL"]
    return {
        "status": (
            "NOT_APPLICABLE"
            if reference_mask_color_censored
            else "FAIL"
            if failures
            else "PASS"
        ),
        "strong_alignment": strong_alignment,
        "failed_color_bins": failures,
        "reference_mask_color_censored": reference_mask_color_censored,
        "trusted_compatible_color_bins": sorted(trusted_compatible_bins),
        "decision_applicability": (
            "NOT_APPLICABLE_REFERENCE_MASK_COLOR_CENSORED"
            if reference_mask_color_censored
            else "APPLICABLE"
        ),
        "bins": records,
    }


def _color_metrics(
    reference: dict[str, Any],
    render: dict[str, Any],
    *,
    alignment: Mapping[str, Any],
    thresholds: Mapping[str, float | int],
) -> dict[str, Any]:
    reference_distribution = _color_distribution(reference["image"], reference["mask"])
    render_distribution = _color_distribution(render["image"], render["mask"])
    category_similarity = _histogram_intersection(
        reference_distribution["category_distribution"].values(),
        render_distribution["category_distribution"].values(),
    )
    hue_similarity: float | None
    if (
        reference_distribution["chromatic_coverage"] >= 0.02
        and render_distribution["chromatic_coverage"] >= 0.02
    ):
        hue_similarity = _histogram_intersection(
            reference_distribution["hue_distribution_12"],
            render_distribution["hue_distribution_12"],
        )
    else:
        hue_similarity = None
    reference_saturation = reference_distribution["median_saturation"]
    render_saturation = render_distribution["median_saturation"]
    saturation_similarity = (
        max(0.0, 1.0 - abs(reference_saturation - render_saturation))
        if reference_saturation is not None and render_saturation is not None
        else None
    )
    evidence_recall = _evidence_recall(reference["evidence"], render_distribution)
    evidence_group_recall = _evidence_group_recall(
        reference["evidence"],
        reference_distribution,
        render_distribution,
    )
    dominant_mass = _dominant_evidence_mass(
        reference["evidence"],
        reference_distribution,
        render_distribution,
        alignment=alignment,
        thresholds=thresholds,
        reference_target_mask_audit=reference.get("target_mask_audit"),
    )
    unreferenced_chromatic_mass = _unreferenced_render_chromatic_mass(
        reference_distribution,
        render_distribution,
        alignment=alignment,
        thresholds=thresholds,
        evidence=reference["evidence"],
        reference_target_mask_audit=reference.get("target_mask_audit"),
    )

    weighted: list[tuple[float, float]] = [(category_similarity, 0.45)]
    if hue_similarity is not None:
        weighted.append((hue_similarity, 0.25))
    if evidence_recall is not None:
        weighted.append((evidence_recall, 0.12))
    if evidence_group_recall is not None:
        weighted.append((evidence_group_recall["macro_recall"], 0.08))
    if saturation_similarity is not None:
        weighted.append((saturation_similarity, 0.10))
    reference_value = reference_distribution["median_value"]
    render_value = render_distribution["median_value"]
    value_similarity = (
        max(0.0, 1.0 - abs(reference_value - render_value))
        if reference_value is not None and render_value is not None
        else None
    )
    if value_similarity is not None:
        # Hue/category agreement alone can rank an over-bright fluorescent
        # coating above the darker fixed MDL seen in the photographs.  Value
        # is measured only inside the foreground mask and is therefore a
        # bounded appearance term, not a material-parameter mutation.
        weighted.append((value_similarity, 0.10))
    score = sum(value * weight for value, weight in weighted) / sum(
        weight for _, weight in weighted
    )
    reference_mask_color_censored = (
        isinstance(reference.get("target_mask_audit"), Mapping)
        and reference["target_mask_audit"].get("method")
        == "trusted_group_roi_and_color_family_intersected_foreground"
    )
    return {
        "score": score,
        "difference_score": 1.0 - score,
        "category_histogram_intersection": category_similarity,
        "chromatic_hue_histogram_intersection": hue_similarity,
        "trusted_evidence_color_recall": evidence_recall,
        "trusted_evidence_group_recall": evidence_group_recall,
        "trusted_evidence_dominant_mass": dominant_mass,
        "unreferenced_render_chromatic_mass": unreferenced_chromatic_mass,
        "median_saturation_similarity": saturation_similarity,
        "median_value_similarity": value_similarity,
        "median_value_absolute_difference": (
            abs(reference_value - render_value)
            if reference_value is not None and render_value is not None
            else None
        ),
        "distribution_threshold_applicability": (
            "SCORE_ONLY_REFERENCE_MASK_COLOR_CENSORED"
            if reference_mask_color_censored
            else "STATUS_AND_SCORE"
        ),
        "reference_distribution": reference_distribution,
        "render_distribution": render_distribution,
    }


def _resolve_multiview_photometric_cohort(
    view_reports: Sequence[dict[str, Any]],
    *,
    thresholds: Mapping[str, float | int],
) -> dict[str, Any]:
    """Resolve value-only reviews when a full multiview exposure proof exists.

    Absolute HSV value is deliberately retained in the material score and in
    the regular per-view gate.  This resolver is narrower: every source view
    must already be comparable, every non-passing view must be blocked only by
    foreground value, colour families and texture must independently pass,
    and the source photographs must exhibit an exposure span that is absent
    from the neutral-light renders.  This prevents a single over-bright
    material from being excused as lighting while allowing an unattended
    multiview workflow to distinguish source exposure from material identity.
    """

    audit: dict[str, Any] = {
        "schema_version": "qwen-multiview-photometric-cohort/v1",
        "status": "NOT_APPLICABLE",
        "promoted_reference_view_ids": [],
        "reason_codes": [],
    }
    comparable = [
        item
        for item in view_reports
        if item.get("status") in {"PASS", "REVIEW", "FAIL"}
    ]
    review_items = [item for item in comparable if item.get("status") == "REVIEW"]
    required_views = int(thresholds["minimum_photometric_cohort_views"])
    if not review_items:
        audit["reason_codes"].append("NO_VALUE_ONLY_REVIEWS")
        return audit
    if len(comparable) != len(view_reports):
        audit["status"] = "REJECTED"
        audit["reason_codes"].append("INCOMPLETE_REFERENCE_VIEW_COVERAGE")
        return audit
    if len(comparable) < required_views:
        audit["status"] = "REJECTED"
        audit["reason_codes"].append("INSUFFICIENT_MULTIVIEW_COHORT")
        return audit
    if any(item.get("status") == "FAIL" for item in comparable):
        audit["status"] = "REJECTED"
        audit["reason_codes"].append("COHORT_CONTAINS_FAILED_VIEW")
        return audit
    value_review_reason = "foreground_value_similarity_below_pass_threshold"
    if any(set(item.get("reasons") or []) != {value_review_reason} for item in review_items):
        audit["status"] = "REJECTED"
        audit["reason_codes"].append("REVIEW_IS_NOT_VALUE_ONLY")
        return audit

    reference_values: list[float] = []
    render_values: list[float] = []
    category_similarities: list[float] = []
    hue_similarities: list[float] = []
    saturation_similarities: list[float] = []
    appearance_scores: list[float] = []
    group_macro_recalls: list[float] = []
    group_minimum_recalls: list[float] = []
    for item in comparable:
        color = item.get("material_color")
        texture = item.get("material_texture")
        alignment = item.get("alignment")
        if not isinstance(color, Mapping) or not isinstance(texture, Mapping):
            audit["status"] = "REJECTED"
            audit["reason_codes"].append("MISSING_COLOR_OR_TEXTURE_EVIDENCE")
            return audit
        reference_distribution = color.get("reference_distribution")
        render_distribution = color.get("render_distribution")
        group_recall = color.get("trusted_evidence_group_recall")
        appearance = item.get("material_appearance_score")
        hue_similarity = color.get("chromatic_hue_histogram_intersection")
        required_numbers = (
            reference_distribution.get("median_value")
            if isinstance(reference_distribution, Mapping)
            else None,
            render_distribution.get("median_value")
            if isinstance(render_distribution, Mapping)
            else None,
            color.get("category_histogram_intersection"),
            hue_similarity,
            color.get("median_saturation_similarity"),
            appearance,
            group_recall.get("macro_recall")
            if isinstance(group_recall, Mapping)
            else None,
            group_recall.get("minimum_group_recall")
            if isinstance(group_recall, Mapping)
            else None,
            alignment.get("score") if isinstance(alignment, Mapping) else None,
        )
        if any(
            not isinstance(value, (int, float)) or isinstance(value, bool)
            for value in required_numbers
        ):
            audit["status"] = "REJECTED"
            audit["reason_codes"].append("INCOMPLETE_PHOTOMETRIC_COHORT_EVIDENCE")
            return audit
        if texture.get("status") != "PASS":
            audit["status"] = "REJECTED"
            audit["reason_codes"].append("TEXTURE_EVIDENCE_DID_NOT_PASS")
            return audit
        reference_value, render_value = required_numbers[:2]
        reference_values.append(float(reference_value))
        render_values.append(float(render_value))
        category_similarities.append(float(required_numbers[2]))
        hue_similarities.append(float(required_numbers[3]))
        saturation_similarities.append(float(required_numbers[4]))
        appearance_scores.append(float(required_numbers[5]))
        group_macro_recalls.append(float(required_numbers[6]))
        group_minimum_recalls.append(float(required_numbers[7]))
        if float(required_numbers[8]) < float(thresholds["strong_alignment_score"]):
            audit["status"] = "REJECTED"
            audit["reason_codes"].append("ALIGNMENT_IS_NOT_STRONG_IN_EVERY_VIEW")
            return audit

    offsets = [
        render_value - reference_value
        for reference_value, render_value in zip(reference_values, render_values)
    ]
    reference_span = max(reference_values) - min(reference_values)
    render_span = max(render_values) - min(render_values)
    maximum_absolute_offset = max(abs(value) for value in offsets)
    direction_consistent = all(value >= 0.0 for value in offsets) or all(
        value <= 0.0 for value in offsets
    )
    audit["metrics"] = {
        "cohort_view_count": len(comparable),
        "reference_median_values": reference_values,
        "render_median_values": render_values,
        "render_minus_reference_value_offsets": offsets,
        "reference_value_span": reference_span,
        "render_value_span": render_span,
        "maximum_absolute_value_offset": maximum_absolute_offset,
        "offset_direction_consistent": direction_consistent,
        "minimum_category_histogram_intersection": min(category_similarities),
        "minimum_chromatic_hue_histogram_intersection": min(hue_similarities),
        "minimum_saturation_similarity": min(saturation_similarities),
        "minimum_material_appearance_score": min(appearance_scores),
        "minimum_evidence_macro_recall": min(group_macro_recalls),
        "minimum_evidence_group_recall": min(group_minimum_recalls),
    }

    checks = (
        (
            direction_consistent,
            "PHOTOMETRIC_OFFSET_DIRECTION_IS_INCONSISTENT",
        ),
        (
            maximum_absolute_offset
            <= float(thresholds["maximum_photometric_value_offset"]),
            "PHOTOMETRIC_VALUE_OFFSET_EXCEEDS_BOUND",
        ),
        (
            reference_span
            >= float(
                thresholds["minimum_reference_value_span_for_photometric_cohort"]
            ),
            "SOURCE_VIEWS_DO_NOT_PROVE_EXPOSURE_VARIATION",
        ),
        (
            render_span
            <= float(thresholds["maximum_render_value_span_for_photometric_cohort"]),
            "RENDER_VALUE_IS_NOT_STABLE_ACROSS_COHORT",
        ),
        (
            min(category_similarities)
            >= float(thresholds["minimum_photometric_cohort_category_similarity"]),
            "COLOR_CATEGORY_SIMILARITY_BELOW_COHORT_BOUND",
        ),
        (
            min(hue_similarities)
            >= float(thresholds["minimum_photometric_cohort_hue_similarity"]),
            "CHROMATIC_HUE_SIMILARITY_BELOW_COHORT_BOUND",
        ),
        (
            min(saturation_similarities)
            >= float(thresholds["minimum_median_saturation_similarity_for_pass"]),
            "SATURATION_SIMILARITY_BELOW_COHORT_BOUND",
        ),
        (
            min(appearance_scores)
            >= float(thresholds["minimum_photometric_cohort_appearance_score"]),
            "APPEARANCE_SIMILARITY_BELOW_COHORT_BOUND",
        ),
        (
            min(group_macro_recalls)
            >= float(thresholds["minimum_evidence_macro_recall"]),
            "EVIDENCE_MACRO_RECALL_BELOW_COHORT_BOUND",
        ),
        (
            min(group_minimum_recalls)
            >= float(thresholds["minimum_evidence_group_recall"]),
            "EVIDENCE_GROUP_RECALL_BELOW_COHORT_BOUND",
        ),
    )
    audit["reason_codes"] = [reason for passed, reason in checks if not passed]
    if audit["reason_codes"]:
        audit["status"] = "REJECTED"
        return audit

    audit["status"] = "PASS"
    audit["resolution"] = (
        "VALUE_DIFFERENCE_EXPLAINED_BY_BOUNDED_MULTIVIEW_SOURCE_EXPOSURE"
    )
    for item in review_items:
        item["status"] = "PASS"
        item["reasons"] = []
        item["photometric_value_resolution"] = {
            "status": "PASS",
            "cohort_schema_version": audit["schema_version"],
            "resolution": audit["resolution"],
        }
        audit["promoted_reference_view_ids"].append(item["reference_view_id"])
    return audit


def _texture_detail_histogram(
    image: Image.Image,
    mask: Image.Image,
    *,
    radius: float,
    erosion_kernel: int,
) -> tuple[list[float] | None, int]:
    """Measure normalized local luminance detail inside a stable surface mask."""

    interior = (
        mask.filter(ImageFilter.MinFilter(erosion_kernel))
        if erosion_kernel > 1
        else mask
    )
    sample_count = _mask_count(interior)
    if sample_count < MIN_TEXTURE_ANALYSIS_PIXELS:
        return None, sample_count

    gray = ImageOps.grayscale(image)
    lowpass = gray.filter(ImageFilter.GaussianBlur(radius))
    histogram = [0] * TEXTURE_HISTOGRAM_BIN_COUNT
    retained = 0
    for luminance, local_luminance, enabled in zip(
        _image_data(gray),
        _image_data(lowpass),
        _image_data(interior),
    ):
        if not enabled:
            continue
        normalized_residual = min(
            1.0,
            abs(float(luminance) - float(local_luminance))
            / max(TEXTURE_LUMA_FLOOR_8BIT, float(luminance)),
        )
        bin_id = min(
            TEXTURE_HISTOGRAM_BIN_COUNT - 1,
            int(normalized_residual * TEXTURE_HISTOGRAM_BIN_COUNT),
        )
        histogram[bin_id] += 1
        retained += 1
    if retained < MIN_TEXTURE_ANALYSIS_PIXELS:
        return None, retained
    return [count / retained for count in histogram], retained


def _surface_texture_metrics(
    reference: Mapping[str, Any],
    render: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare visible surface detail without consulting material categories.

    Foreground boundaries are eroded so silhouette and small geometric edges do
    not masquerade as material texture.  Two local-detail scales are compared
    as luminance-normalized histograms, which makes the statistic insensitive
    to the absolute exposure while still detecting procedural grain or fabric
    fibres on an otherwise smooth reference surface.
    """

    selected_kernel: int | None = None
    scale_records: list[dict[str, Any]] = []
    for erosion_kernel in TEXTURE_INTERIOR_KERNELS:
        candidate_records: list[dict[str, Any]] = []
        valid = True
        for radius in TEXTURE_DETAIL_RADII:
            reference_histogram, reference_pixels = _texture_detail_histogram(
                reference["image"],
                reference["mask"],
                radius=radius,
                erosion_kernel=erosion_kernel,
            )
            render_histogram, render_pixels = _texture_detail_histogram(
                render["image"],
                render["mask"],
                radius=radius,
                erosion_kernel=erosion_kernel,
            )
            if reference_histogram is None or render_histogram is None:
                valid = False
                break
            score = _histogram_intersection(
                reference_histogram,
                render_histogram,
            )
            candidate_records.append(
                {
                    "radius_px": radius,
                    "score": score,
                    "reference_sample_count": reference_pixels,
                    "render_sample_count": render_pixels,
                    "reference_histogram": reference_histogram,
                    "render_histogram": render_histogram,
                }
            )
        if valid:
            selected_kernel = erosion_kernel
            scale_records = candidate_records
            break

    if selected_kernel is None:
        return {
            "status": "UNSCORABLE",
            "score": None,
            "difference_score": None,
            "method": "masked_multiscale_luma_residual_histogram/v1",
            "reason_codes": ["INSUFFICIENT_INTERIOR_TEXTURE_PIXELS"],
            "scales": [],
        }

    score = sum(record["score"] for record in scale_records) / len(scale_records)
    return {
        "status": "PASS",
        "score": score,
        "difference_score": 1.0 - score,
        "method": "masked_multiscale_luma_residual_histogram/v1",
        "reason_codes": [],
        "radii_px": list(TEXTURE_DETAIL_RADII),
        "interior_erosion_kernel": selected_kernel,
        "histogram_bin_count": TEXTURE_HISTOGRAM_BIN_COUNT,
        "luma_floor_8bit": TEXTURE_LUMA_FLOOR_8BIT,
        "scales": scale_records,
    }


def _validate_foreground(
    metrics: Mapping[str, Any],
    *,
    reference: bool,
    thresholds: Mapping[str, float | int],
) -> list[str]:
    coverage = float(metrics.get("coverage") or 0.0)
    prefix = "reference" if reference else "render"
    minimum = float(thresholds[f"minimum_{prefix}_foreground_coverage"])
    maximum = float(thresholds[f"maximum_{prefix}_foreground_coverage"])
    reasons: list[str] = []
    if coverage < minimum:
        reasons.append(f"{prefix}_foreground_too_small")
    if coverage > maximum:
        reasons.append(f"{prefix}_foreground_too_large")
    if metrics.get("bbox") is None:
        reasons.append(f"{prefix}_foreground_missing")
    return reasons


def _validate_target_foreground(
    metrics: Mapping[str, Any],
    *,
    reference: bool,
    thresholds: Mapping[str, float | int],
) -> list[str]:
    """Validate a local group by evidence pixels, not whole-frame coverage."""

    prefix = "reference_target" if reference else "render_target"
    pixel_count = int(metrics.get("pixel_count") or 0)
    reasons: list[str] = []
    if pixel_count < int(thresholds["minimum_target_foreground_pixels"]):
        reasons.append(f"{prefix}_foreground_too_small")
    if metrics.get("bbox") is None:
        reasons.append(f"{prefix}_foreground_missing")
    maximum_key = (
        "maximum_reference_foreground_coverage"
        if reference
        else "maximum_render_foreground_coverage"
    )
    if float(metrics.get("coverage") or 0.0) > float(thresholds[maximum_key]):
        reasons.append(f"{prefix}_foreground_too_large")
    return reasons


def _auto_mapping(
    references: list[dict[str, Any]],
    renders: list[dict[str, Any]],
    thresholds: Mapping[str, float | int],
    *,
    locked_mapping: Mapping[str, str] | None = None,
) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    locked = dict(locked_mapping or {})
    audits = _explicit_mapping_audit(references, renders, locked)
    mapping = dict(locked)
    locked_render_ids = set(locked.values())
    unlocked_references = [
        reference for reference in references if reference["view_id"] not in locked
    ]
    available_renders = [
        render for render in renders if render["view_id"] not in locked_render_ids
    ]
    auto_mode = "auto_completion" if locked else "auto"

    for reference_id in locked:
        audits[reference_id]["mode"] = "explicit_locked"
        audits[reference_id]["locked"] = True

    if not unlocked_references:
        return mapping, audits

    render_by_id = {render["view_id"]: render for render in available_renders}
    render_ids = sorted(render_by_id)
    candidates_by_reference: list[list[dict[str, Any]]] = []
    score_tables: list[dict[str, float]] = []
    for reference in unlocked_references:
        candidates = sorted(
            (
                {
                    "render_view_id": render_id,
                    **_alignment_metrics(
                        _view_alignment_mask(reference),
                        _view_alignment_mask(render_by_id[render_id]),
                    ),
                }
                for render_id in render_ids
            ),
            key=lambda item: (-float(item["score"]), item["render_view_id"]),
        )
        candidates_by_reference.append(candidates)
        score_tables.append(
            {
                str(candidate["render_view_id"]): float(candidate["score"])
                for candidate in candidates
            }
        )

    def best_global_assignment(
        forbidden: tuple[int, str] | None = None,
    ) -> tuple[float, tuple[str | None, ...]]:
        """Return a deterministic maximum-weight injective assignment."""

        states: dict[int, tuple[float, tuple[str | None, ...]]] = {0: (0.0, ())}
        for reference_index, scores in enumerate(score_tables):
            next_states: dict[int, tuple[float, tuple[str | None, ...]]] = {}

            def retain(
                used_mask: int,
                score: float,
                assignment: tuple[str | None, ...],
            ) -> None:
                current = next_states.get(used_mask)
                if current is None:
                    next_states[used_mask] = (score, assignment)
                    return
                current_score, current_assignment = current
                candidate_key = (
                    round(score, 12),
                    sum(value is not None for value in assignment),
                    tuple(value or "~" for value in assignment),
                )
                current_key = (
                    round(current_score, 12),
                    sum(value is not None for value in current_assignment),
                    tuple(value or "~" for value in current_assignment),
                )
                if candidate_key[:2] > current_key[:2] or (
                    candidate_key[:2] == current_key[:2]
                    and candidate_key[2] < current_key[2]
                ):
                    next_states[used_mask] = (score, assignment)

            for used_mask, (score, assignment) in states.items():
                retain(used_mask, score, (*assignment, None))
                for render_index, render_id in enumerate(render_ids):
                    if used_mask & (1 << render_index):
                        continue
                    if forbidden == (reference_index, render_id):
                        continue
                    retain(
                        used_mask | (1 << render_index),
                        score + scores[render_id],
                        (*assignment, render_id),
                    )
            states = next_states

        best_score = max(round(item[0], 12) for item in states.values())
        score_tied = [
            item for item in states.values() if round(item[0], 12) == best_score
        ]
        best_count = max(
            sum(value is not None for value in item[1]) for item in score_tied
        )
        finalists = [
            item
            for item in score_tied
            if sum(value is not None for value in item[1]) == best_count
        ]
        return min(
            finalists,
            key=lambda item: tuple(value or "~" for value in item[1]),
        )

    global_score, assignment = best_global_assignment()
    minimum_score = float(thresholds["minimum_auto_alignment_score"])
    minimum_margin = float(thresholds["minimum_auto_match_margin"])
    for reference_index, reference in enumerate(unlocked_references):
        reference_id = reference["view_id"]
        candidates = candidates_by_reference[reference_index]
        assigned_render_id = assignment[reference_index]
        assigned = (
            next(
                candidate
                for candidate in candidates
                if candidate["render_view_id"] == assigned_render_id
            )
            if assigned_render_id is not None
            else None
        )
        alternatives = [
            candidate
            for candidate in candidates
            if candidate["render_view_id"] != assigned_render_id
        ]
        runner_up = alternatives[0] if alternatives else None
        local_margin = (
            float(assigned["score"]) - float(runner_up["score"])
            if assigned is not None and runner_up is not None
            else None
        )
        alternative_global_score: float | None = None
        global_margin: float | None = None
        reasons: list[str] = []
        if assigned_render_id is None or assigned is None:
            reasons.append("global_one_to_one_assignment_unmatched")
            if not render_ids:
                reasons.append("no_unused_render_view_available")
        else:
            alternative_global_score, _ = best_global_assignment(
                (reference_index, assigned_render_id)
            )
            global_margin = global_score - alternative_global_score
            if float(assigned["score"]) < minimum_score:
                reasons.append("best_auto_alignment_below_threshold")
            if global_margin < minimum_margin:
                reasons.extend(
                    [
                        "auto_match_margin_too_small",
                        "global_assignment_margin_too_small",
                    ]
                )

        selected_render_id = None if reasons else assigned_render_id
        audits[reference_id] = {
            "mode": auto_mode,
            "locked": False,
            "selected_render_view_id": selected_render_id,
            "proposed_render_view_id": assigned_render_id,
            "best_score": float(assigned["score"]) if assigned is not None else None,
            "runner_up_score": (
                float(runner_up["score"]) if runner_up is not None else None
            ),
            # ``margin`` remains the stable public field, but now represents
            # confidence in the complete injective assignment rather than a
            # collision-prone per-reference greedy choice.
            "margin": global_margin,
            "global_assignment_margin": global_margin,
            "local_candidate_margin": local_margin,
            "global_assignment_score": global_score,
            "alternative_global_assignment_score": alternative_global_score,
            "global_one_to_one_assignment": True,
            "reasons": reasons,
            "candidates": candidates,
        }
        if selected_render_id is not None:
            mapping[reference_id] = selected_render_id
    return mapping, audits


def _explicit_mapping_audit(
    references: list[dict[str, Any]],
    renders: list[dict[str, Any]],
    mapping: Mapping[str, str],
) -> dict[str, dict[str, Any]]:
    reference_ids = {item["view_id"] for item in references}
    render_by_id = {item["view_id"]: item for item in renders}
    unknown_references = sorted(set(mapping) - reference_ids)
    unknown_renders = sorted(set(mapping.values()) - set(render_by_id))
    if unknown_references:
        raise ComparisonInputError(
            f"view mapping has unknown reference IDs: {unknown_references}"
        )
    if unknown_renders:
        raise ComparisonInputError(
            f"view mapping has unknown render IDs: {unknown_renders}"
        )
    if len(set(mapping.values())) != len(mapping):
        raise ComparisonInputError("explicit view mapping must be one-to-one")
    reference_by_id = {item["view_id"]: item for item in references}
    audits: dict[str, dict[str, Any]] = {}
    for reference_id in reference_ids:
        render_id = mapping.get(reference_id)
        metrics = (
            _alignment_metrics(
                _view_alignment_mask(reference_by_id[reference_id]),
                _view_alignment_mask(render_by_id[render_id]),
            )
            if render_id
            else None
        )
        audits[reference_id] = {
            "mode": "explicit",
            "selected_render_view_id": render_id,
            "reasons": [] if render_id else ["reference_view_not_explicitly_mapped"],
            "alignment_preview": metrics,
        }
    return audits


def load_view_mapping(path: str | Path) -> dict[str, str]:
    mapping_path = Path(path).expanduser().resolve(strict=True)
    document = _load_json_object(mapping_path, "view mapping")
    raw: Any = document.get("mapping", document.get("mappings", document))
    mapping: dict[str, str] = {}
    if isinstance(raw, dict):
        for reference_id, render_id in raw.items():
            if reference_id in {"schema_version", "notes"}:
                continue
            if not isinstance(reference_id, str) or not isinstance(render_id, str):
                raise ComparisonInputError("view mapping object values must be strings")
            mapping[reference_id] = render_id
    elif isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                raise ComparisonInputError("view mapping list entries must be objects")
            reference_id = item.get("reference_view_id")
            render_id = item.get("render_view_id")
            if not isinstance(reference_id, str) or not isinstance(render_id, str):
                raise ComparisonInputError(
                    "view mapping entries require string reference_view_id and render_view_id"
                )
            if reference_id in mapping:
                raise ComparisonInputError(
                    f"duplicate reference mapping: {reference_id}"
                )
            mapping[reference_id] = render_id
    else:
        raise ComparisonInputError("view mapping must contain an object or list")
    return mapping


def compare_reference_renders(
    reference_manifest: str | Path,
    rendered_registry: str | Path,
    *,
    view_mapping: Mapping[str, str] | None = None,
    minimum_comparable_views: int | None = None,
    target_part_ids: Iterable[str] | None = None,
    target_entities: Iterable[Mapping[str, Any]] | None = None,
    target_group_id: str | None = None,
    palette_fusion: str | Path | None = None,
    target_reference_view_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    manifest_path = Path(reference_manifest).expanduser().resolve(strict=True)
    registry_path = Path(rendered_registry).expanduser().resolve(strict=True)
    scope_arguments_present = (
        target_part_ids is not None,
        target_group_id is not None,
        palette_fusion is not None,
    )
    if any(scope_arguments_present) and not all(scope_arguments_present):
        raise ComparisonInputError(
            "local comparison requires target_part_ids, target_group_id, "
            "and palette_fusion together"
        )
    if target_entities is not None and not all(scope_arguments_present):
        raise ComparisonInputError(
            "target_entities are valid only for a complete local comparison scope"
        )
    if target_reference_view_ids is not None and not all(scope_arguments_present):
        raise ComparisonInputError(
            "target_reference_view_ids are valid only for a complete local "
            "comparison scope"
        )
    scoped_target_parts: set[str] | None = None
    target_scope: dict[str, Any] | None = None
    if all(scope_arguments_present):
        if isinstance(target_part_ids, (str, bytes)):
            raise ComparisonInputError(
                "target_part_ids must be an iterable of unique part ID strings"
            )
        assert target_part_ids is not None
        raw_target_parts = list(target_part_ids)
        if (
            not raw_target_parts
            or any(
                not isinstance(part_id, str) or not part_id
                for part_id in raw_target_parts
            )
            or len(set(raw_target_parts)) != len(raw_target_parts)
        ):
            raise ComparisonInputError(
                "target_part_ids must contain unique non-empty strings"
            )
        if not isinstance(target_group_id, str) or not target_group_id:
            raise ComparisonInputError("target_group_id must be a non-empty string")
        assert palette_fusion is not None
        palette_fusion_path = Path(palette_fusion).expanduser().resolve(strict=True)
        scoped_target_parts = set(raw_target_parts)
        target_scope = _load_target_group_scope(
            palette_fusion_path,
            target_group_id,
            target_reference_view_ids=(
                list(target_reference_view_ids)
                if target_reference_view_ids is not None
                else None
            ),
        )
        target_scope["target_part_ids"] = sorted(scoped_target_parts)
        if target_entities is None:
            normalized_target_entities = [
                {
                    "entity_kind": "assignment",
                    "part_id": part_id,
                }
                for part_id in sorted(scoped_target_parts)
            ]
        else:
            if isinstance(target_entities, (str, bytes)):
                raise ComparisonInputError(
                    "target_entities must be an iterable of objects"
                )
            normalized_target_entities = []
            seen_target_entities: set[tuple[str, str]] = set()
            for index, raw_entity in enumerate(target_entities):
                if not isinstance(raw_entity, Mapping):
                    raise ComparisonInputError(
                        f"target_entities[{index}] must be an object"
                    )
                part_id = raw_entity.get("part_id")
                subset_name = raw_entity.get("subset_name")
                raw_kind = raw_entity.get("entity_kind")
                if (
                    not isinstance(part_id, str)
                    or not part_id
                    or part_id not in scoped_target_parts
                ):
                    raise ComparisonInputError(
                        f"target_entities[{index}] has an invalid part_id"
                    )
                if subset_name is None:
                    entity_kind = "assignment"
                    entity = {
                        "entity_kind": entity_kind,
                        "part_id": part_id,
                    }
                    entity_key = (part_id, "")
                else:
                    if not isinstance(subset_name, str) or not subset_name:
                        raise ComparisonInputError(
                            f"target_entities[{index}] has an invalid subset_name"
                        )
                    entity_kind = "face_subset"
                    entity = {
                        "entity_kind": entity_kind,
                        "part_id": part_id,
                        "subset_name": subset_name,
                    }
                    entity_key = (part_id, subset_name)
                if raw_kind is not None and raw_kind != entity_kind:
                    raise ComparisonInputError(
                        f"target_entities[{index}] has inconsistent entity_kind"
                    )
                if entity_key in seen_target_entities:
                    raise ComparisonInputError(
                        "target_entities must not contain duplicates"
                    )
                seen_target_entities.add(entity_key)
                normalized_target_entities.append(entity)
            if (
                not normalized_target_entities
                or {str(entity["part_id"]) for entity in normalized_target_entities}
                != scoped_target_parts
            ):
                raise ComparisonInputError(
                    "target_entities must exactly cover target_part_ids"
                )
            normalized_target_entities.sort(
                key=lambda entity: (
                    str(entity["part_id"]),
                    str(entity.get("subset_name", "")),
                )
            )
        target_scope["target_entities"] = normalized_target_entities
        face_subset_scope = any(
            entity["entity_kind"] == "face_subset"
            for entity in normalized_target_entities
        )
        target_scope["render_mask_granularity"] = (
            "containing_part_proxy" if face_subset_scope else "exact_part_id"
        )
        target_scope["face_subset_render_mask_exact"] = not face_subset_scope
        target_scope["mode"] = "canonical_group_local"

    thresholds = dict(DEFAULT_THRESHOLDS)
    if minimum_comparable_views is not None:
        if (
            isinstance(minimum_comparable_views, bool)
            or not isinstance(minimum_comparable_views, int)
            or minimum_comparable_views < 1
        ):
            raise ComparisonInputError(
                "minimum_comparable_views must be a positive integer"
            )
        thresholds["minimum_comparable_views"] = minimum_comparable_views

    references = _load_reference_views(
        manifest_path,
        target_local_group_ids_by_view=(
            target_scope["local_group_ids_by_view"] if target_scope else None
        ),
    )
    renders = _load_render_views(
        registry_path,
        target_part_ids=scoped_target_parts,
    )
    render_by_id = {item["view_id"]: item for item in renders}
    seeded_mapping: dict[str, str] = {}
    if view_mapping is not None and target_scope is not None:
        manifest = _load_json_object(manifest_path, "reference manifest")
        raw_source_views = manifest.get("source_views")
        assert isinstance(raw_source_views, list)
        all_manifest_reference_ids = {
            raw_view.get("id")
            for raw_view in raw_source_views
            if isinstance(raw_view, Mapping) and isinstance(raw_view.get("id"), str)
        }
        unknown_mapping_references = sorted(
            set(view_mapping) - all_manifest_reference_ids
        )
        if unknown_mapping_references:
            raise ComparisonInputError(
                f"view mapping has unknown reference IDs: {unknown_mapping_references}"
            )
        scoped_reference_ids = {reference["view_id"] for reference in references}
        view_mapping = {
            reference_id: render_id
            for reference_id, render_id in view_mapping.items()
            if reference_id in scoped_reference_ids
        }
    if view_mapping is None:
        selected_mapping, mapping_audit = _auto_mapping(references, renders, thresholds)
        mapping_mode = "auto"
    else:
        for reference_id, render_id in view_mapping.items():
            if not isinstance(reference_id, str) or not isinstance(render_id, str):
                raise ComparisonInputError(
                    "view mapping keys and values must be strings"
                )
            if not reference_id or not render_id:
                raise ComparisonInputError(
                    "view mapping keys and values cannot be empty"
                )
            seeded_mapping[reference_id] = render_id
        if len(seeded_mapping) == len(references):
            selected_mapping = dict(seeded_mapping)
            mapping_audit = _explicit_mapping_audit(
                references, renders, selected_mapping
            )
            mapping_mode = "explicit"
        else:
            selected_mapping, mapping_audit = _auto_mapping(
                references,
                renders,
                thresholds,
                locked_mapping=seeded_mapping,
            )
            mapping_mode = (
                "explicit_seeded_auto_completion" if seeded_mapping else "auto"
            )

    view_reports: list[dict[str, Any]] = []
    for reference in references:
        reference_id = reference["view_id"]
        render_id = selected_mapping.get(reference_id)
        render = render_by_id.get(render_id) if render_id else None
        if target_scope is not None:
            reasons = _validate_target_foreground(
                reference["mask_metrics"],
                reference=True,
                thresholds=thresholds,
            )
        else:
            reasons = _validate_foreground(
                reference["mask_metrics"], reference=True, thresholds=thresholds
            )
        reasons.extend(reference["evidence"]["reasons"])
        if render is None:
            reasons.extend(mapping_audit[reference_id]["reasons"])
        else:
            if target_scope is not None:
                reasons.extend(
                    _validate_target_foreground(
                        render["mask_metrics"],
                        reference=False,
                        thresholds=thresholds,
                    )
                )
            else:
                reasons.extend(
                    _validate_foreground(
                        render["mask_metrics"],
                        reference=False,
                        thresholds=thresholds,
                    )
                )

        alignment = (
            _alignment_metrics(
                _view_alignment_mask(reference),
                _view_alignment_mask(render),
            )
            if render
            else None
        )
        alignment_pass = alignment is not None and alignment["score"] >= float(
            thresholds["minimum_explicit_alignment_score"]
        )
        if alignment is not None and not alignment_pass:
            reasons.append("view_alignment_below_material_scoring_threshold")

        color: dict[str, Any] | None = None
        texture: dict[str, Any] | None = None
        appearance_score: float | None = None
        status = "UNSCORABLE"
        if render is not None and not reasons and alignment_pass:
            color = _color_metrics(
                reference,
                render,
                alignment=alignment,
                thresholds=thresholds,
            )
            texture = _surface_texture_metrics(reference, render)
            texture_score = texture.get("score")
            if isinstance(texture_score, (int, float)) and not isinstance(
                texture_score, bool
            ):
                appearance_score = math.sqrt(
                    max(0.0, float(color["score"])) * max(0.0, float(texture_score))
                )
            score = color["score"]
            group_recall = color.get("trusted_evidence_group_recall")
            dominant_mass = color.get("trusted_evidence_dominant_mass")
            dominant_mass_failure = (
                isinstance(dominant_mass, Mapping)
                and dominant_mass.get("status") == "FAIL"
            )
            unreferenced_chromatic_mass = color.get(
                "unreferenced_render_chromatic_mass"
            )
            unreferenced_chromatic_failure = (
                isinstance(unreferenced_chromatic_mass, Mapping)
                and unreferenced_chromatic_mass.get("status") == "FAIL"
            )
            missing_salient_group = _annotate_group_delivery_presence(
                group_recall,
                thresholds=thresholds,
            )
            weak_macro_recall = isinstance(group_recall, Mapping) and float(
                group_recall["macro_recall"]
            ) < float(thresholds["minimum_evidence_macro_recall"])
            hard_failure = False
            if dominant_mass_failure:
                status = "FAIL"
                reasons.append("trusted_dominant_family_mass_deficit")
                hard_failure = True
            if unreferenced_chromatic_failure:
                status = "FAIL"
                reasons.append("unreferenced_render_chromatic_mass")
                hard_failure = True
            if missing_salient_group and alignment["score"] >= float(
                thresholds["strong_alignment_score"]
            ):
                status = "FAIL"
                reasons.append("trusted_palette_group_missing_from_render")
                hard_failure = True
            if hard_failure:
                pass
            elif weak_macro_recall:
                status = "REVIEW"
                reasons.append("trusted_palette_macro_recall_below_threshold")
            elif (
                color.get("distribution_threshold_applicability")
                != "SCORE_ONLY_REFERENCE_MASK_COLOR_CENSORED"
                and color["median_value_similarity"]
                < float(
                thresholds["minimum_median_value_similarity_for_pass"]
                )
            ):
                status = "REVIEW"
                reasons.append("foreground_value_similarity_below_pass_threshold")
            elif (
                color.get("distribution_threshold_applicability")
                != "SCORE_ONLY_REFERENCE_MASK_COLOR_CENSORED"
                and color["median_saturation_similarity"]
                < float(
                thresholds["minimum_median_saturation_similarity_for_pass"]
                )
            ):
                status = "REVIEW"
                reasons.append("foreground_saturation_similarity_below_pass_threshold")
            elif score >= float(thresholds["pass_color_score"]):
                status = "PASS"
            elif score < float(thresholds["fail_color_score"]) and alignment[
                "score"
            ] >= float(thresholds["strong_alignment_score"]):
                status = "FAIL"
                reasons.append("aligned_view_color_score_below_fail_threshold")
            else:
                status = "REVIEW"
                reasons.append("color_score_requires_human_review")

        view_reports.append(
            {
                "reference_view_id": reference_id,
                "render_view_id": render_id,
                "status": status,
                "reasons": list(dict.fromkeys(reasons)),
                "mapping": mapping_audit[reference_id],
                "reference": {
                    "image": str(reference["image_path"]),
                    "image_sha256": reference["image_sha256"],
                    "foreground": reference["mask_metrics"],
                    "foreground_audit": reference["mask_audit"],
                    "alignment_foreground": reference["alignment_mask_metrics"],
                    "target_foreground_audit": reference["target_mask_audit"],
                    "trusted_evidence": reference["evidence"],
                },
                "render": (
                    {
                        "image": str(render["image_path"]),
                        "image_sha256": render["image_sha256"],
                        "part_ids": str(render["part_ids_path"]),
                        "part_ids_sha256": render["part_ids_sha256"],
                        "foreground": render["mask_metrics"],
                        "foreground_audit": render["mask_audit"],
                        "alignment_foreground": render["alignment_mask_metrics"],
                        "target_foreground_audit": render["target_mask_audit"],
                    }
                    if render
                    else None
                ),
                "alignment": alignment,
                "material_color": color,
                "material_texture": texture,
                "material_appearance_score": appearance_score,
            }
        )

    photometric_cohort_resolution = _resolve_multiview_photometric_cohort(
        view_reports,
        thresholds=thresholds,
    )
    comparable = [
        item for item in view_reports if item["status"] in {"PASS", "REVIEW", "FAIL"}
    ]
    failures = [item for item in comparable if item["status"] == "FAIL"]
    passes = [item for item in comparable if item["status"] == "PASS"]
    reviews = [item for item in comparable if item["status"] == "REVIEW"]
    unscorable = [item for item in view_reports if item["status"] == "UNSCORABLE"]
    unmapped_reference_ids = [
        item["reference_view_id"]
        for item in view_reports
        if item["render_view_id"] is None
    ]
    unscorable_reference_ids = [item["reference_view_id"] for item in unscorable]
    aggregate_score = (
        sum(
            item["material_color"]["score"] * item["alignment"]["score"]
            for item in comparable
        )
        / sum(item["alignment"]["score"] for item in comparable)
        if comparable
        else None
    )
    texture_comparable = [
        item
        for item in comparable
        if (
            isinstance(item.get("material_texture"), Mapping)
            and isinstance(item["material_texture"].get("score"), (int, float))
            and not isinstance(item["material_texture"].get("score"), bool)
        )
    ]
    aggregate_texture_score = (
        sum(
            float(item["material_texture"]["score"]) * item["alignment"]["score"]
            for item in texture_comparable
        )
        / sum(item["alignment"]["score"] for item in texture_comparable)
        if texture_comparable
        else None
    )
    aggregate_appearance_score = (
        math.sqrt(aggregate_score * aggregate_texture_score)
        if aggregate_score is not None and aggregate_texture_score is not None
        else None
    )
    aggregate_reasons: list[str] = []
    salient_group_failures = [
        item
        for item in failures
        if "trusted_palette_group_missing_from_render" in item["reasons"]
    ]
    dominant_mass_failures = [
        item
        for item in failures
        if "trusted_dominant_family_mass_deficit" in item["reasons"]
    ]
    unreferenced_chromatic_failures = [
        item
        for item in failures
        if "unreferenced_render_chromatic_mass" in item["reasons"]
    ]
    if dominant_mass_failures:
        aggregate_status = "FAIL"
        aggregate_reasons.append(
            "single_strong_view_confirms_dominant_family_mass_deficit"
        )
        if len(dominant_mass_failures) >= 2:
            aggregate_reasons.append("multiple_aligned_views_confirm_color_mismatch")
    elif unreferenced_chromatic_failures:
        aggregate_status = "FAIL"
        aggregate_reasons.append(
            "single_strong_view_confirms_unreferenced_render_chromatic_mass"
        )
        if len(unreferenced_chromatic_failures) >= 2:
            aggregate_reasons.append("multiple_aligned_views_confirm_color_mismatch")
    elif len(comparable) < int(thresholds["minimum_comparable_views"]):
        aggregate_status = "INSUFFICIENT_EVIDENCE"
        aggregate_reasons.append("fewer_than_minimum_comparable_aligned_views")
        if unscorable:
            aggregate_reasons.append("reference_view_coverage_failed_closed")
        if unmapped_reference_ids:
            aggregate_reasons.append(
                "not_all_reference_views_have_confident_one_to_one_mapping"
            )
        if len(unscorable) > len(unmapped_reference_ids):
            aggregate_reasons.append("not_all_reference_views_are_comparable")
    elif len(salient_group_failures) >= 2:
        aggregate_status = "FAIL"
        aggregate_reasons.extend(
            [
                "multiple_aligned_views_confirm_color_mismatch",
                "multiple_views_confirm_missing_trusted_palette_groups",
            ]
        )
    elif (
        len(failures) >= int(thresholds["minimum_failing_views_for_aggregate_fail"])
        and aggregate_score is not None
        and aggregate_score < float(thresholds["aggregate_fail_color_score"])
    ):
        aggregate_status = "FAIL"
        aggregate_reasons.append("multiple_aligned_views_confirm_color_mismatch")
    elif unscorable:
        aggregate_status = "INSUFFICIENT_EVIDENCE"
        aggregate_reasons.append("reference_view_coverage_failed_closed")
        if unmapped_reference_ids:
            aggregate_reasons.append(
                "not_all_reference_views_have_confident_one_to_one_mapping"
            )
        if len(unscorable) > len(unmapped_reference_ids):
            aggregate_reasons.append("not_all_reference_views_are_comparable")
    elif (
        aggregate_score is not None
        and aggregate_score >= float(thresholds["pass_color_score"])
        and not failures
        and not reviews
    ):
        aggregate_status = "PASS"
    else:
        aggregate_status = "REVIEW"
        aggregate_reasons.append("aggregate_color_evidence_is_not_conclusive")

    return {
        "schema_version": SCHEMA_VERSION,
        "inputs": {
            "reference_manifest": str(manifest_path),
            "reference_manifest_sha256": _sha256(manifest_path),
            "rendered_registry": str(registry_path),
            "rendered_registry_sha256": _sha256(registry_path),
            "mapping_mode": mapping_mode,
            "seeded_view_mapping": seeded_mapping,
            "selected_view_mapping": selected_mapping,
            "comparison_scope": (
                target_scope if target_scope is not None else {"mode": "whole_asset"}
            ),
        },
        "thresholds": thresholds,
        "photometric_cohort_resolution": photometric_cohort_resolution,
        "aggregate": {
            "status": aggregate_status,
            "material_match_conclusion": (
                aggregate_status
                if aggregate_status in {"PASS", "FAIL"}
                else "NOT_CONCLUSIVE"
            ),
            "material_color_score": aggregate_score,
            "material_color_difference_score": (
                1.0 - aggregate_score if aggregate_score is not None else None
            ),
            "material_texture_score": aggregate_texture_score,
            "material_texture_difference_score": (
                1.0 - aggregate_texture_score
                if aggregate_texture_score is not None
                else None
            ),
            "material_appearance_score": aggregate_appearance_score,
            "material_appearance_difference_score": (
                1.0 - aggregate_appearance_score
                if aggregate_appearance_score is not None
                else None
            ),
            "texture_comparable_view_count": len(texture_comparable),
            "texture_unscorable_view_count": len(comparable) - len(texture_comparable),
            "reference_view_count": len(references),
            "render_view_count": len(renders),
            "comparable_view_count": len(comparable),
            "passed_view_count": len(passes),
            "review_view_count": len(reviews),
            "failed_view_count": len(failures),
            "unscorable_view_count": len(view_reports) - len(comparable),
            "reference_view_coverage_status": (
                "PASS" if not unscorable else "FAIL_CLOSED"
            ),
            "unmapped_reference_view_ids": unmapped_reference_ids,
            "unscorable_reference_view_ids": unscorable_reference_ids,
            "reasons": aggregate_reasons,
        },
        "views": view_reports,
        "limitations": [
            "This is a deterministic colour-and-surface-detail check, not a learned perceptual or semantic judgement.",
            "Silhouette and edge scores are alignment gates only; unaligned pairs cannot fail material quality, but any uncovered reference view prevents aggregate PASS.",
            (
                "Scoped comparison uses whole-object masks for pose alignment, "
                "requested part-ID masks for render statistics, and trusted "
                "per-view group ROIs or explicit group masks for reference statistics."
                if target_scope is not None
                else "Whole-asset comparison scores every decoded visible registered part."
            ),
            "Foreground inference cannot recover a perfectly black object on a perfectly black reference background without an explicit mask.",
            "Per-group macro recall detects missing trusted colour families; multiscale masked luminance residuals compare visible texture, but neither proves exact per-part identity or every roughness, metallic, and lighting effect.",
            "Dominant-family mass is a delivery gate only; it never identifies a part or authorizes a material mutation without independent spatial evidence.",
            "A PASS requires every source reference view to have a confident one-to-one render mapping and comparable trusted palette evidence.",
        ],
    }


def _parse_cli_mapping(values: Sequence[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ComparisonInputError(
                f"--map must use reference=render syntax: {value}"
            )
        reference_id, render_id = value.split("=", 1)
        reference_id = reference_id.strip()
        render_id = render_id.strip()
        if not reference_id or not render_id:
            raise ComparisonInputError(
                f"--map must use non-empty reference=render IDs: {value}"
            )
        if reference_id in mapping:
            raise ComparisonInputError(f"duplicate --map reference ID: {reference_id}")
        mapping[reference_id] = render_id
    return mapping


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare trusted reference photos with RTX renders while failing closed "
            "on view misalignment or insufficient evidence"
        )
    )
    parser.add_argument("--reference-manifest", required=True)
    parser.add_argument("--rendered-registry", required=True)
    parser.add_argument("--output", required=True)
    mapping_group = parser.add_mutually_exclusive_group()
    mapping_group.add_argument(
        "--view-map", help="JSON file containing explicit reference-to-render mapping"
    )
    mapping_group.add_argument(
        "--map",
        action="append",
        default=[],
        metavar="REFERENCE=RENDER",
        help="explicit one-to-one mapping; repeat for each comparable reference view",
    )
    parser.add_argument("--minimum-comparable-views", type=int)
    parser.add_argument(
        "--target-part-id",
        action="append",
        default=[],
        help=(
            "score only this rendered registry part; repeat for every part in "
            "the target canonical group"
        ),
    )
    parser.add_argument(
        "--target-group-id",
        help="canonical palette group to score in the reference views",
    )
    parser.add_argument(
        "--target-reference-view-id",
        action="append",
        default=[],
        help=(
            "trusted canonical source view to include in group-local scoring; "
            "repeat for every source in the sealed scoring subset"
        ),
    )
    parser.add_argument(
        "--target-entity-json",
        action="append",
        default=[],
        help=(
            "canonical JSON object identifying one assignment or face-subset "
            "material target; repeat for every target entity"
        ),
    )
    parser.add_argument(
        "--palette-fusion",
        help=(
            "palette_fusion.json used to resolve the canonical target group "
            "to trusted per-view local evidence"
        ),
    )
    parser.add_argument(
        "--require-pass",
        action="store_true",
        help="return exit status 3 unless the aggregate status is PASS",
    )
    return parser.parse_args(argv)


def _write_report_atomic(output: Path, report: Mapping[str, Any]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.view_map:
            mapping: Mapping[str, str] | None = load_view_mapping(args.view_map)
        elif args.map:
            mapping = _parse_cli_mapping(args.map)
        else:
            mapping = None
        target_entities: list[Mapping[str, Any]] | None = None
        if args.target_entity_json:
            target_entities = []
            for index, raw_entity in enumerate(args.target_entity_json):
                try:
                    entity = json.loads(raw_entity)
                except json.JSONDecodeError as exc:
                    raise ComparisonInputError(
                        f"--target-entity-json[{index}] is invalid JSON: {exc}"
                    ) from exc
                if not isinstance(entity, Mapping):
                    raise ComparisonInputError(
                        f"--target-entity-json[{index}] must be an object"
                    )
                target_entities.append(entity)
        report = compare_reference_renders(
            args.reference_manifest,
            args.rendered_registry,
            view_mapping=mapping,
            minimum_comparable_views=args.minimum_comparable_views,
            target_part_ids=(args.target_part_id or None),
            target_entities=target_entities,
            target_group_id=args.target_group_id,
            palette_fusion=args.palette_fusion,
            target_reference_view_ids=(args.target_reference_view_id or None),
        )
        output = Path(args.output).expanduser().resolve()
        _write_report_atomic(output, report)
        print(
            json.dumps(
                {"output": str(output), **report["aggregate"]},
                ensure_ascii=False,
                indent=2,
            )
        )
        if args.require_pass and report["aggregate"]["status"] != "PASS":
            return EXIT_REQUIRE_PASS_FAILED
        return EXIT_SUCCESS
    except (ComparisonInputError, OSError) as exc:
        print(
            json.dumps(
                {"status": "INPUT_ERROR", "error": str(exc)}, ensure_ascii=False
            ),
            file=sys.stderr,
        )
        return EXIT_INPUT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
