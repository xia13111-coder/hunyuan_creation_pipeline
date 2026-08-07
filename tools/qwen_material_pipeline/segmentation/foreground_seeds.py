"""Deterministic, model-independent foreground box proposals for SAM3.

The foreground stage intentionally runs before any VLM.  These proposals use
only border-background contrast and image morphology; SAM3 remains responsible
for the final pixel mask.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


SEED_POLICY_SCHEMA_VERSION = "sam3-automatic-foreground-seeds/v1"


class ForegroundSeedError(ValueError):
    """Raised when an image has no defensible automatic foreground proposal."""


def _normalized_box(
    *,
    x: int,
    y: int,
    width: int,
    height: int,
    image_width: int,
    image_height: int,
) -> list[int]:
    padding = max(3, int(round(0.035 * max(width, height))))
    left = max(0, x - padding)
    top = max(0, y - padding)
    right = min(image_width, x + width + padding)
    bottom = min(image_height, y + height + padding)
    return [
        max(0, min(999, int(round(left * 1000 / image_width)))),
        max(0, min(999, int(round(top * 1000 / image_height)))),
        max(1, min(1000, int(round(right * 1000 / image_width)))),
        max(1, min(1000, int(round(bottom * 1000 / image_height)))),
    ]


def build_automatic_foreground_seeds(image_path: Path) -> dict[str, Any]:
    """Return tight SAM3 seed boxes without consulting a language model.

    A robust border median estimates the background colour.  Otsu thresholding
    finds border-distinct pixels, a morphological opening removes thin viewport
    grids, and significant connected components become independent SAM3 boxes.
    The function fails closed when the image provides no defensible component.
    """

    import cv2
    import numpy as np
    from PIL import Image, ImageOps

    resolved = image_path.expanduser().resolve(strict=True)
    with Image.open(resolved) as opened:
        rgb = np.asarray(ImageOps.exif_transpose(opened).convert("RGB"))
    image_height, image_width = rgb.shape[:2]
    if image_width < 8 or image_height < 8:
        raise ForegroundSeedError(
            f"reference image is too small for automatic foreground seeds: {resolved}"
        )

    border_width = max(2, min(image_width, image_height) // 50)
    border = np.concatenate(
        (
            rgb[:border_width].reshape(-1, 3),
            rgb[-border_width:].reshape(-1, 3),
            rgb[:, :border_width].reshape(-1, 3),
            rgb[:, -border_width:].reshape(-1, 3),
        ),
        axis=0,
    )
    background_rgb = np.median(border, axis=0)
    distance = np.linalg.norm(
        rgb.astype(np.float32) - background_rgb.astype(np.float32),
        axis=2,
    )
    distance_u8 = np.clip(distance, 0, 255).astype(np.uint8)
    filtered = cv2.medianBlur(distance_u8, 3)
    otsu_threshold, _unused = cv2.threshold(
        filtered,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    # The lower bound suppresses JPEG noise and faint viewport grids.  The
    # upper bound prevents an extreme highlight from hiding ordinary surfaces.
    threshold = max(18.0, min(96.0, float(otsu_threshold)))
    binary = np.where(filtered >= threshold, 255, 0).astype(np.uint8)
    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        np.ones((3, 3), dtype=np.uint8),
    )
    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        np.ones((7, 7), dtype=np.uint8),
    )

    _count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )
    minimum_pixels = max(64, int(round(image_width * image_height * 0.00025)))
    components = sorted(
        (
            tuple(int(value) for value in row)
            for row in stats[1:]
            if int(row[cv2.CC_STAT_AREA]) >= minimum_pixels
        ),
        key=lambda row: row[cv2.CC_STAT_AREA],
        reverse=True,
    )
    if not components:
        raise ForegroundSeedError(
            "automatic foreground seed extraction found no border-distinct "
            f"component: {resolved}"
        )

    largest_pixels = components[0][cv2.CC_STAT_AREA]
    significant_pixels = max(minimum_pixels, int(round(largest_pixels * 0.008)))
    boxes: list[list[int]] = []
    component_audit: list[dict[str, Any]] = []
    for row in components:
        x = row[cv2.CC_STAT_LEFT]
        y = row[cv2.CC_STAT_TOP]
        width = row[cv2.CC_STAT_WIDTH]
        height = row[cv2.CC_STAT_HEIGHT]
        pixels = row[cv2.CC_STAT_AREA]
        if pixels < significant_pixels:
            continue
        # A surviving near-full-width one-pixel axis is still a viewport line,
        # not a manufactured component.
        if (
            width > 0.85 * image_width
            and height < 0.04 * image_height
        ) or (
            height > 0.85 * image_height
            and width < 0.04 * image_width
        ):
            continue
        box = _normalized_box(
            x=x,
            y=y,
            width=width,
            height=height,
            image_width=image_width,
            image_height=image_height,
        )
        if box[0] >= box[2] or box[1] >= box[3] or box in boxes:
            continue
        boxes.append(box)
        component_audit.append(
            {
                "box": box,
                "component_pixels": pixels,
                "component_image_fraction": round(
                    pixels / (image_width * image_height),
                    8,
                ),
            }
        )
        if len(boxes) == 12:
            break
    if not boxes:
        raise ForegroundSeedError(
            "automatic foreground seed extraction found only thin background "
            f"structures: {resolved}"
        )

    return {
        "schema_version": SEED_POLICY_SCHEMA_VERSION,
        "image": str(resolved),
        "image_size": [image_width, image_height],
        "background_rgb": [int(round(value)) for value in background_rgb],
        "distance_threshold": round(threshold, 4),
        "minimum_component_pixels": minimum_pixels,
        "significant_component_pixels": significant_pixels,
        "boxes": boxes,
        "components": component_audit,
    }
