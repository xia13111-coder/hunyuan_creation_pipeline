#!/usr/bin/env python3
"""Build automatic SAM3 local-refinement boxes for projected CAD Part IDs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


SCHEMA_VERSION = "qwen-sam3-region-request/v1"
MINIMUM_PROJECTED_MASK_PIXELS = 6


def _read(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {resolved}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _box(mask_path: Path) -> tuple[list[int], dict[str, Any]]:
    with Image.open(mask_path) as opened:
        mask = np.asarray(opened.convert("L"), dtype=np.uint8) >= 128
    ys, xs = np.where(mask)
    if len(xs) < MINIMUM_PROJECTED_MASK_PIXELS:
        raise ValueError(
            "projected Part-ID mask has insufficient pixels "
            f"(<{MINIMUM_PROJECTED_MASK_PIXELS}): {mask_path}"
        )
    height, width = mask.shape
    object_width = int(xs.max() - xs.min() + 1)
    object_height = int(ys.max() - ys.min() + 1)
    # The whole-asset camera is the location prior, not a pixel-perfect part
    # segmentation. A wider local search box tolerates residual camera error
    # and photographed movable accessories; the CAD seed still ranks SAM3
    # candidates and the downstream bounded registration rejects bad fits.
    padding_fraction = 0.35
    padding = max(
        8,
        int(round(padding_fraction * max(object_width, object_height))),
    )
    left = max(0, int(xs.min()) - padding)
    top = max(0, int(ys.min()) - padding)
    right = min(width, int(xs.max()) + padding + 1)
    bottom = min(height, int(ys.max()) + padding + 1)

    def normalized(value: int, extent: int, *, upper: bool) -> int:
        raw = int(math.ceil(value * 1000 / extent)) if upper else int(
            math.floor(value * 1000 / extent)
        )
        return max(1 if upper else 0, min(1000 if upper else 999, raw))

    return [
        normalized(left, width, upper=False),
        normalized(top, height, upper=False),
        normalized(right, width, upper=True),
        normalized(bottom, height, upper=True),
    ], {
        "mask_size": [width, height],
        "projected_mask_pixels": int(mask.sum()),
        "projected_bbox_pixels": [left, top, right, bottom],
        "local_search_padding_fraction": padding_fraction,
    }


def _cad_seed_points(
    mask_path: Path,
) -> tuple[dict[str, list[list[int]]], dict[str, Any]]:
    """Derive deterministic instance prompts from a projected CAD mask."""

    with Image.open(mask_path) as opened:
        mask = np.asarray(opened.convert("L"), dtype=np.uint8) >= 128
    height, width = mask.shape
    if int(mask.sum()) < MINIMUM_PROJECTED_MASK_PIXELS:
        raise ValueError(
            "projected Part-ID mask has insufficient pixels "
            f"(<{MINIMUM_PROJECTED_MASK_PIXELS}): {mask_path}"
        )
    binary = mask.astype(np.uint8)
    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    components = sorted(
        range(1, component_count),
        key=lambda label: (-int(stats[label, cv2.CC_STAT_AREA]), label),
    )
    positive_pixels: list[tuple[int, int]] = []
    for label in components[:4]:
        component_distance = np.where(labels == label, distance, -1.0)
        y, x = np.unravel_index(
            int(np.argmax(component_distance)), component_distance.shape
        )
        positive_pixels.append((int(x), int(y)))

    ys, xs = np.where(mask)
    diagonal = math.hypot(
        int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)
    )
    spacing = max(4.0, 0.22 * diagonal)
    ranked = np.dstack(
        np.unravel_index(np.argsort(distance.ravel())[::-1], distance.shape)
    )[0]
    for y, x in ranked:
        if distance[y, x] <= 0:
            break
        candidate = (int(x), int(y))
        if all(
            math.hypot(candidate[0] - px, candidate[1] - py) >= spacing
            for px, py in positive_pixels
        ):
            positive_pixels.append(candidate)
        if len(positive_pixels) >= 4:
            break

    ring_radius = max(5, int(round(0.08 * diagonal)))
    inner = cv2.dilate(
        binary,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )
    outer = cv2.dilate(
        binary,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * ring_radius + 1, 2 * ring_radius + 1),
        ),
        iterations=1,
    )
    ring_y, ring_x = np.where((outer > 0) & (inner == 0))
    negative_pixels: list[tuple[int, int]] = []
    if len(ring_x):
        center_x = float(xs.mean())
        center_y = float(ys.mean())
        for angle in np.linspace(0.0, 2.0 * math.pi, 8, endpoint=False):
            target_x = center_x + math.cos(float(angle)) * (
                0.5 * diagonal + ring_radius
            )
            target_y = center_y + math.sin(float(angle)) * (
                0.5 * diagonal + ring_radius
            )
            index = int(
                np.argmin(
                    (ring_x - target_x) ** 2 + (ring_y - target_y) ** 2
                )
            )
            candidate = (int(ring_x[index]), int(ring_y[index]))
            if candidate not in negative_pixels:
                negative_pixels.append(candidate)

    def grid(point: tuple[int, int]) -> list[int]:
        x, y = point
        return [
            int(round(x * 1000 / max(1, width - 1))),
            int(round(y * 1000 / max(1, height - 1))),
        ]

    return {
        "positive_points": [grid(point) for point in positive_pixels],
        "negative_points": [grid(point) for point in negative_pixels],
    }, {
        "cad_seed_positive_point_count": len(positive_pixels),
        "cad_seed_negative_point_count": len(negative_pixels),
        "cad_seed_negative_ring_radius_pixels": ring_radius,
    }


def build_request(
    evidence_path: Path,
    *,
    part_ids: set[str] | None = None,
) -> dict[str, Any]:
    evidence_path = evidence_path.expanduser().resolve(strict=True)
    evidence = _read(evidence_path)
    source_by_view: dict[str, dict[str, Any]] = {}
    regions: list[dict[str, Any]] = []
    for part in evidence.get("parts", []):
        if not isinstance(part, dict) or part.get("status") != "observed":
            continue
        part_id = part.get("part_id")
        if (
            not isinstance(part_id, str)
            or (part_ids is not None and part_id not in part_ids)
        ):
            continue
        observations = [
            row
            for row in part.get("observations", [])
            if isinstance(row, dict)
        ]
        if not observations:
            raise ValueError(f"{part_id} does not have a usable observation")
        for observation in observations:
            view_id = str(observation["view_id"])
            image = (
                Path(str(observation["image"]))
                .expanduser()
                .resolve(strict=True)
            )
            mask = (
                Path(str(observation["mask"]))
                .expanduser()
                .resolve(strict=True)
            )
            box, audit = _box(mask)
            source_by_view.setdefault(
                view_id,
                {
                    "id": view_id,
                    "image": str(image),
                    "image_sha256": _sha256(image),
                },
            )
            regions.append(
                {
                    "view_id": view_id,
                    "group_id": part_id,
                    "prompt": (
                        "one individual rigid component or contiguous surface "
                        "of the industrial machine overlapping the CAD seed "
                        "inside this box; segment only that local component "
                        "and exclude adjacent components"
                    ),
                    "boxes": [box],
                    "cad_projection_seed": {
                        "path": str(mask),
                        "sha256": _sha256(mask),
                        **audit,
                    },
                }
            )
    if not regions:
        raise ValueError("no observed Part IDs matched the request")
    return {
        "schema_version": SCHEMA_VERSION,
        "source_views": [
            source_by_view[view_id] for view_id in sorted(source_by_view)
        ],
        "regions": sorted(
            regions, key=lambda row: (row["view_id"], row["group_id"])
        ),
        "prompt_authority": (
            "all_visible_cad_part_id_projection_boxes_inside_"
            "human_sam3_foreground"
        ),
        "part_id_evidence": {
            "path": str(evidence_path),
            "sha256": _sha256(evidence_path),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--part-id",
        action="append",
        default=[],
        help="Optional repeatable Part-ID filter used for diagnostics",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    document = build_request(
        args.evidence,
        part_ids=set(args.part_id) if args.part_id else None,
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "source_view_count": len(document["source_views"]),
                "region_count": len(document["regions"]),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
