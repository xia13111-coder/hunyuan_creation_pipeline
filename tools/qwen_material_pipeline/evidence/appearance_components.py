#!/usr/bin/env python3
"""Derive conservative photo-supported appearance components for CAD Part IDs.

This stage sits between rigid camera registration and material retrieval.  It
does *not* select, bind, tune, or mutate a material.  Its only output is a
traceable constraint proposal: which independently-addressable CAD Part IDs
are sufficiently likely to share one visible coating in the reference photos.

The implementation deliberately avoids the two failure modes that made the
older palette and source-material grouping paths unsuitable for visual CAD
matching:

* a palette label or a source CAD material is never membership authority;
* no per-Part-ID registration, crop warp, or photo segmentation is performed.

Each part inherits the one rigid whole-asset image transform selected by camera
registration.  Its renderer-authored Part-ID interior is intersected with the
human-confirmed SAM3 foreground, then robust reference colour is sampled.  A
component needs same-view colour agreement, spatial proximity, sufficient
visible support, and no high-confidence conflicting observation.  Parts that
are hidden, tiny, ambiguous, or merely similar in a source assembly remain
independent.
"""

from __future__ import annotations

import argparse
import colorsys
import hashlib
import json
import math
import os
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np


SCHEMA_VERSION = "qwen-part-id-appearance-components/v1"
MINIMUM_TRUSTED_PIXELS = 64
MINIMUM_FOREGROUND_OVERLAP = 0.75
MINIMUM_COMPONENT_TRUSTED_PIXELS = 512
MINIMUM_COMPONENT_PARTS = 2
MAXIMUM_COMPONENT_PARTS = 96
MAXIMUM_SAME_VIEW_DELTA_E = 32.0
CONFLICT_DELTA_E = 48.0
MINIMUM_LINK_SUPPORT = 0.60
USABLE_MINIMUM_IOU = 0.92
USABLE_MAXIMUM_BOUNDARY_P95_PX = 10.0
DOWNWEIGHTED_MINIMUM_IOU = 0.88
DOWNWEIGHTED_MAXIMUM_BOUNDARY_P95_PX = 15.0


class AppearanceComponentError(ValueError):
    """Raised when the evidence boundary for component construction is invalid."""


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AppearanceComponentError(f"unable to read {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AppearanceComponentError(f"{label} must be a JSON object")
    return value


def _resolve_file(value: Any, *, owner: Path, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise AppearanceComponentError(f"{label} must be a non-empty file path")
    path = Path(os.path.expandvars(value)).expanduser()
    if not path.is_absolute():
        path = owner.parent / path
    try:
        path = path.resolve(strict=True)
    except OSError as exc:
        raise AppearanceComponentError(f"{label} does not exist: {path}") from exc
    if not path.is_file():
        raise AppearanceComponentError(f"{label} is not a file: {path}")
    return path


def _part_color(part_id: str) -> tuple[int, int, int]:
    """Return the stable RGB colour authored by the Part-ID renderer."""

    suffix = part_id[1:] if part_id.startswith("P") else ""
    number = int(suffix) if suffix.isdigit() else sum(map(ord, part_id))
    hue = (number * 0.618033988749895) % 1.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.72, 0.95)
    return int(red * 255), int(green * 255), int(blue * 255)


def _packed_bgr(red: int, green: int, blue: int) -> int:
    return int(blue) | (int(green) << 8) | (int(red) << 16)


def _component_colour_family(rgb: Sequence[float]) -> str:
    red, green, blue = (float(value) for value in rgb)
    hue, saturation, value = colorsys.rgb_to_hsv(red, green, blue)
    if value < 0.16:
        return "dark_neutral"
    if saturation < 0.12:
        return "light_neutral" if value >= 0.62 else "neutral"
    if hue < 0.055 or hue >= 0.96:
        return "red"
    if hue < 0.20:
        return "warm"
    if hue < 0.49:
        return "green"
    if hue < 0.75:
        return "cyan_blue"
    return "purple_pink"


def _robust_colour(image_bgr: np.ndarray, mask: np.ndarray) -> dict[str, Any] | None:
    pixels = image_bgr[mask > 0]
    if len(pixels) < MINIMUM_TRUSTED_PIXELS:
        return None
    # Fixed deterministic sub-sampling keeps large enclosures from dominating
    # CPU or the robust medoid while preserving every thin component in full.
    if len(pixels) > 4096:
        indices = np.linspace(0, len(pixels) - 1, 4096, dtype=np.int64)
        pixels = pixels[indices]
    rgb = pixels[:, ::-1].astype(np.float32) / 255.0
    lab = cv2.cvtColor(rgb.reshape(1, -1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3)
    coordinate_median = np.median(lab, axis=0)
    medoid = lab[int(np.argmin(np.linalg.norm(lab - coordinate_median, axis=1)))]
    distances = np.linalg.norm(lab - medoid, axis=1)
    inliers = distances <= 20.0
    if not np.any(inliers):
        inliers[int(np.argmin(distances))] = True
    median_rgb = np.median(rgb[inliers], axis=0)
    median_lab = np.median(lab[inliers], axis=0)
    return {
        "median_rgb": [round(float(value), 8) for value in median_rgb],
        "median_lab": [round(float(value), 8) for value in median_lab],
        "inlier_fraction": round(float(np.mean(inliers)), 8),
        "median_delta_e": round(float(np.median(distances)), 8),
        "appearance_family": _component_colour_family(median_rgb),
    }


def _bbox(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.where(mask > 0)
    if not len(xs):
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def _bbox_gap(first: Sequence[int], second: Sequence[int]) -> float:
    left_a, top_a, right_a, bottom_a = (int(value) for value in first)
    left_b, top_b, right_b, bottom_b = (int(value) for value in second)
    dx = max(0, left_a - right_b, left_b - right_a)
    dy = max(0, top_a - bottom_b, top_b - bottom_a)
    return float(math.hypot(dx, dy))


def _proximity_limit(first: Sequence[int], second: Sequence[int]) -> float:
    sizes = [
        max(1, int(first[2]) - int(first[0])),
        max(1, int(first[3]) - int(first[1])),
        max(1, int(second[2]) - int(second[0])),
        max(1, int(second[3]) - int(second[1])),
    ]
    return float(max(10.0, min(36.0, 0.06 * max(sizes))))


def _as_affine(value: Any, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (2, 3) or not np.isfinite(array).all():
        raise AppearanceComponentError(f"{label} must be a finite 2x3 affine matrix")
    return array


def _camera_acceptance(camera_report: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    raw_views = camera_report.get("views")
    if not isinstance(raw_views, list) or not raw_views:
        raise AppearanceComponentError("camera report has no views")
    output: dict[str, dict[str, Any]] = {}
    for raw in raw_views:
        if not isinstance(raw, Mapping):
            raise AppearanceComponentError("camera report has an invalid view")
        view_id = raw.get("reference_view_id")
        final = raw.get("final")
        if not isinstance(view_id, str) or not view_id or not isinstance(final, Mapping):
            raise AppearanceComponentError("camera report view lacks final evidence")
        iou = final.get("projection_iou")
        boundary = final.get("boundary_p95_px")
        similarity = final.get("whole_asset_similarity")
        if (
            isinstance(iou, bool)
            or not isinstance(iou, (int, float))
            or isinstance(boundary, bool)
            or not isinstance(boundary, (int, float))
            or not isinstance(similarity, Mapping)
        ):
            raise AppearanceComponentError(f"camera report metrics are invalid for {view_id}")
        if raw.get("complete_alignment_passed") is True:
            tier, weight = "strict", 1.0
        elif float(iou) >= USABLE_MINIMUM_IOU and float(boundary) <= USABLE_MAXIMUM_BOUNDARY_P95_PX:
            tier, weight = "usable_box_correspondence", 0.80
        elif (
            float(iou) >= DOWNWEIGHTED_MINIMUM_IOU
            and float(boundary) <= DOWNWEIGHTED_MAXIMUM_BOUNDARY_P95_PX
        ):
            tier, weight = "downweighted_box_correspondence", 0.55
        else:
            tier, weight = "rejected", 0.0
        output[view_id] = {
            "tier": tier,
            "evidence_weight": weight,
            "projection_iou": round(float(iou), 8),
            "boundary_p95_px": round(float(boundary), 8),
            "bbox_affine": _as_affine(
                similarity.get("bbox_affine"), f"camera report affine {view_id}"
            ),
        }
    return output


def _manifest_references(manifest: Mapping[str, Any], owner: Path) -> dict[str, dict[str, Any]]:
    raw_views = manifest.get("source_views")
    if not isinstance(raw_views, list) or not raw_views:
        raise AppearanceComponentError("reference manifest has no source_views")
    output: dict[str, dict[str, Any]] = {}
    for raw in raw_views:
        if not isinstance(raw, Mapping):
            raise AppearanceComponentError("reference manifest contains an invalid view")
        view_id = raw.get("id")
        if not isinstance(view_id, str) or not view_id or view_id in output:
            raise AppearanceComponentError("reference manifest view IDs are invalid")
        confirmed = raw.get("confirmed_mask")
        mask_value = raw.get("palette_mask")
        if mask_value is None and isinstance(confirmed, Mapping):
            mask_value = confirmed.get("path")
        image_path = _resolve_file(raw.get("image"), owner=owner, label=f"reference image {view_id}")
        mask_path = _resolve_file(mask_value, owner=owner, label=f"SAM3 foreground {view_id}")
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        foreground = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None or foreground is None or image.shape[:2] != foreground.shape:
            raise AppearanceComponentError(f"reference image/mask dimensions differ for {view_id}")
        if not np.any(foreground > 0):
            raise AppearanceComponentError(f"SAM3 foreground is empty for {view_id}")
        output[view_id] = {
            "image_path": image_path,
            "mask_path": mask_path,
            "image": image,
            "foreground": foreground > 0,
        }
    return output


def _registry_views(registry: Mapping[str, Any], owner: Path) -> dict[str, dict[str, Any]]:
    render_set = registry.get("render_set")
    raw_views = render_set.get("views") if isinstance(render_set, Mapping) else None
    if not isinstance(raw_views, list) or not raw_views:
        raise AppearanceComponentError("rendered registry has no render views")
    output: dict[str, dict[str, Any]] = {}
    for raw in raw_views:
        if not isinstance(raw, Mapping):
            raise AppearanceComponentError("rendered registry contains an invalid view")
        view_id = raw.get("view_id")
        if not isinstance(view_id, str) or not view_id:
            raise AppearanceComponentError("rendered registry view lacks view_id")
        calibration = raw.get("camera_calibration")
        reference_id = (
            calibration.get("reference_view_id")
            if isinstance(calibration, Mapping)
            and isinstance(calibration.get("reference_view_id"), str)
            else view_id
        )
        if reference_id in output:
            raise AppearanceComponentError(
                f"rendered registry has duplicate camera reference {reference_id}"
            )
        ids_path = _resolve_file(
            raw.get("part_ids_raw") or raw.get("part_ids"),
            owner=owner,
            label=f"Part-ID render {view_id}",
        )
        ids = cv2.imread(str(ids_path), cv2.IMREAD_COLOR)
        if ids is None:
            raise AppearanceComponentError(f"unable to read Part-ID render {view_id}")
        output[reference_id] = {"view_id": view_id, "ids_path": ids_path, "ids": ids}
    return output


def _part_ids(registry: Mapping[str, Any]) -> list[str]:
    raw_parts = registry.get("parts")
    if not isinstance(raw_parts, list) or not raw_parts:
        raise AppearanceComponentError("rendered registry has no parts")
    output: list[str] = []
    for raw in raw_parts:
        part_id = raw.get("part_id") if isinstance(raw, Mapping) else None
        if not isinstance(part_id, str) or not part_id or part_id in output:
            raise AppearanceComponentError("rendered registry Part IDs are invalid")
        output.append(part_id)
    return output


def _label_image(ids: np.ndarray, part_ids: Sequence[str]) -> tuple[np.ndarray, dict[int, str]]:
    packed = (
        ids[:, :, 0].astype(np.uint32)
        | (ids[:, :, 1].astype(np.uint32) << 8)
        | (ids[:, :, 2].astype(np.uint32) << 16)
    )
    color_to_part = {
        _packed_bgr(*_part_color(part_id)): part_id for part_id in part_ids
    }
    values, inverse = np.unique(packed, return_inverse=True)
    value_to_label = np.zeros(len(values), dtype=np.int32)
    label_to_part: dict[int, str] = {}
    for index, value in enumerate(values.tolist()):
        part_id = color_to_part.get(int(value))
        if part_id is None:
            continue
        label = part_ids.index(part_id) + 1
        value_to_label[index] = label
        label_to_part[label] = part_id
    return value_to_label[inverse].reshape(packed.shape), label_to_part


def _scaled_affine(
    affine: np.ndarray,
    *,
    current_shape: tuple[int, int],
    scored_shape: tuple[int, int],
) -> np.ndarray:
    current_height, current_width = current_shape
    scored_height, scored_width = scored_shape
    if min(current_height, current_width, scored_height, scored_width) <= 0:
        raise AppearanceComponentError("Part-ID render dimensions must be positive")
    scaling = np.asarray(
        [
            [float(scored_width) / float(current_width), 0.0, 0.0],
            [0.0, float(scored_height) / float(current_height), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    homogeneous = np.vstack((affine, np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32)))
    return (homogeneous @ scaling)[:2, :]


def _observation_weight(*, pixels: int, alignment_weight: float) -> float:
    return float(alignment_weight * min(1.0, math.log2(max(2, pixels)) / 12.0))


def _normalised_component_id(member_part_ids: Sequence[str]) -> str:
    return "AC_" + _canonical_sha256(
        {"schema_version": SCHEMA_VERSION, "member_part_ids": sorted(member_part_ids)}
    )[:12]


def _weighted_median(rows: Sequence[tuple[float, float]]) -> float:
    ordered = sorted(rows, key=lambda item: item[0])
    threshold = sum(weight for _value, weight in ordered) * 0.5
    total = 0.0
    for value, weight in ordered:
        total += weight
        if total >= threshold:
            return value
    return ordered[-1][0]


def build_appearance_components(
    *,
    rendered_registry: Mapping[str, Any] | str | Path,
    reference_manifest: Mapping[str, Any] | str | Path,
    camera_report: Mapping[str, Any] | str | Path,
) -> dict[str, Any]:
    """Return immutable appearance-component constraints for CAD Part IDs."""

    def read_input(value: Mapping[str, Any] | str | Path, label: str) -> tuple[dict[str, Any], Path | None]:
        if isinstance(value, Mapping):
            return dict(value), None
        path = Path(value).expanduser().resolve(strict=True)
        return _read_object(path, label), path

    registry, registry_path = read_input(rendered_registry, "rendered registry")
    manifest, manifest_path = read_input(reference_manifest, "reference manifest")
    report, report_path = read_input(camera_report, "camera report")
    if registry_path is None or manifest_path is None or report_path is None:
        raise AppearanceComponentError("file-backed inputs are required for image evidence")
    part_ids = _part_ids(registry)
    references = _manifest_references(manifest, manifest_path)
    render_views = _registry_views(registry, registry_path)
    acceptance = _camera_acceptance(report)
    if set(render_views) != set(references):
        raise AppearanceComponentError(
            "rendered camera views and reference views must have exact coverage"
        )
    if set(acceptance) != set(references):
        raise AppearanceComponentError(
            "camera report and reference views must have exact coverage"
        )

    scored_registry_value = report.get("final_rendered_registry")
    if not isinstance(scored_registry_value, str) or not scored_registry_value:
        raise AppearanceComponentError("camera report lacks final_rendered_registry")
    scored_registry_path = _resolve_file(
        scored_registry_value, owner=report_path, label="camera scored registry"
    )
    scored_registry = _read_object(scored_registry_path, "camera scored registry")
    scored_views = _registry_views(scored_registry, scored_registry_path)
    if set(scored_views) != set(references):
        raise AppearanceComponentError("camera scored registry has incomplete view coverage")

    observations: dict[str, list[dict[str, Any]]] = {part_id: [] for part_id in part_ids}
    rejected_view_ids: list[str] = []
    for reference_id in sorted(references):
        camera = acceptance[reference_id]
        if camera["tier"] == "rejected":
            rejected_view_ids.append(reference_id)
            continue
        current = render_views[reference_id]
        reference = references[reference_id]
        scored = scored_views[reference_id]
        labels, label_to_part = _label_image(current["ids"], part_ids)
        affine = _scaled_affine(
            camera["bbox_affine"],
            current_shape=current["ids"].shape[:2],
            scored_shape=scored["ids"].shape[:2],
        )
        warped_labels = cv2.warpAffine(
            labels,
            affine,
            (reference["image"].shape[1], reference["image"].shape[0]),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        for label, part_id in sorted(label_to_part.items()):
            projected = warped_labels == label
            projected_pixels = int(np.count_nonzero(projected))
            if projected_pixels < MINIMUM_TRUSTED_PIXELS:
                continue
            projected_u8 = projected.astype(np.uint8)
            interior = cv2.erode(
                projected_u8,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                iterations=1,
            )
            sampling = interior if int(np.count_nonzero(interior)) >= MINIMUM_TRUSTED_PIXELS else projected_u8
            sampling_pixels = int(np.count_nonzero(sampling))
            trusted = (sampling > 0) & reference["foreground"]
            trusted_pixels = int(np.count_nonzero(trusted))
            foreground_overlap = trusted_pixels / max(1, sampling_pixels)
            if (
                trusted_pixels < MINIMUM_TRUSTED_PIXELS
                or foreground_overlap < MINIMUM_FOREGROUND_OVERLAP
            ):
                continue
            appearance = _robust_colour(reference["image"], trusted.astype(np.uint8))
            bounds = _bbox(trusted.astype(np.uint8))
            if appearance is None or bounds is None:
                continue
            observations[part_id].append(
                {
                    "reference_view_id": reference_id,
                    "render_view_id": current["view_id"],
                    "trusted_pixels": trusted_pixels,
                    "projected_pixels": projected_pixels,
                    "foreground_overlap": round(foreground_overlap, 8),
                    "bbox": bounds,
                    "evidence_weight": camera["evidence_weight"],
                    "camera_alignment_tier": camera["tier"],
                    "appearance": appearance,
                }
            )

    # Candidate links are constructed only between co-visible parts.  The
    # reference image may contain a valid part in one view only; a missing
    # observation is therefore neutral rather than a contradictory vote.
    by_view: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for part_id, rows in observations.items():
        for row in rows:
            by_view[str(row["reference_view_id"])].append((part_id, row))
    candidate_links: dict[tuple[str, str], dict[str, Any]] = {}
    conflicts: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for view_id, rows in sorted(by_view.items()):
        for first_index, (first_id, first) in enumerate(rows):
            first_appearance = first["appearance"]
            for second_id, second in rows[first_index + 1 :]:
                key = tuple(sorted((first_id, second_id)))
                second_appearance = second["appearance"]
                first_lab = np.asarray(first_appearance["median_lab"], dtype=np.float32)
                second_lab = np.asarray(second_appearance["median_lab"], dtype=np.float32)
                delta_e = float(np.linalg.norm(first_lab - second_lab))
                same_family = (
                    first_appearance["appearance_family"]
                    == second_appearance["appearance_family"]
                )
                shared_weight = min(
                    float(first["evidence_weight"]), float(second["evidence_weight"])
                )
                if (not same_family or delta_e > CONFLICT_DELTA_E) and shared_weight >= 0.80:
                    conflicts[key].append(
                        {
                            "reference_view_id": view_id,
                            "delta_e": round(delta_e, 8),
                            "first_family": first_appearance["appearance_family"],
                            "second_family": second_appearance["appearance_family"],
                        }
                    )
                    continue
                gap = _bbox_gap(first["bbox"], second["bbox"])
                limit = _proximity_limit(first["bbox"], second["bbox"])
                if not same_family or delta_e > MAXIMUM_SAME_VIEW_DELTA_E or gap > limit:
                    continue
                support = min(
                    _observation_weight(
                        pixels=int(first["trusted_pixels"]),
                        alignment_weight=float(first["evidence_weight"]),
                    ),
                    _observation_weight(
                        pixels=int(second["trusted_pixels"]),
                        alignment_weight=float(second["evidence_weight"]),
                    ),
                )
                record = candidate_links.setdefault(
                    key,
                    {"support": 0.0, "evidence": []},
                )
                record["support"] += support
                record["evidence"].append(
                    {
                        "reference_view_id": view_id,
                        "appearance_family": first_appearance["appearance_family"],
                        "delta_e": round(delta_e, 8),
                        "bbox_gap_px": round(gap, 8),
                        "maximum_allowed_gap_px": round(limit, 8),
                        "support": round(support, 8),
                    }
                )

    accepted_links: list[dict[str, Any]] = []
    adjacency: dict[str, set[str]] = {part_id: set() for part_id in part_ids}
    for key, record in sorted(candidate_links.items()):
        if conflicts.get(key):
            continue
        # A usable camera with a component-sized observation contributes at
        # least .60 after the bounded pixel-quality factor.  A downweighted
        # view stays below this floor even when its projected area is large,
        # so it cannot manufacture an appearance link by itself.
        if float(record["support"]) < MINIMUM_LINK_SUPPORT:
            continue
        first_id, second_id = key
        adjacency[first_id].add(second_id)
        adjacency[second_id].add(first_id)
        accepted_links.append(
            {
                "part_ids": [first_id, second_id],
                "support": round(float(record["support"]), 8),
                "evidence": sorted(record["evidence"], key=lambda item: item["reference_view_id"]),
            }
        )

    components: list[dict[str, Any]] = []
    assigned: set[str] = set()
    for seed in part_ids:
        if seed in assigned or not adjacency[seed]:
            continue
        stack = [seed]
        members: set[str] = set()
        while stack:
            part_id = stack.pop()
            if part_id in members:
                continue
            members.add(part_id)
            stack.extend(sorted(adjacency[part_id] - members, reverse=True))
        assigned.update(members)
        member_ids = sorted(members)
        member_observations = [
            (part_id, row)
            for part_id in member_ids
            for row in observations[part_id]
        ]
        total_pixels = sum(int(row["trusted_pixels"]) for _part_id, row in member_observations)
        if (
            len(member_ids) < MINIMUM_COMPONENT_PARTS
            or len(member_ids) > MAXIMUM_COMPONENT_PARTS
            or total_pixels < MINIMUM_COMPONENT_TRUSTED_PIXELS
        ):
            continue
        family_weights: dict[str, float] = defaultdict(float)
        for _part_id, row in member_observations:
            family_weights[str(row["appearance"]["appearance_family"])] += _observation_weight(
                pixels=int(row["trusted_pixels"]), alignment_weight=float(row["evidence_weight"])
            )
        family = max(family_weights, key=lambda key: (family_weights[key], key))
        matching_rows = [
            row
            for _part_id, row in member_observations
            if row["appearance"]["appearance_family"] == family
        ]
        canonical_rgb = [
            round(
                _weighted_median(
                    [
                        (float(row["appearance"]["median_rgb"][channel]), _observation_weight(pixels=int(row["trusted_pixels"]), alignment_weight=float(row["evidence_weight"])))
                        for row in matching_rows
                    ]
                ),
                8,
            )
            for channel in range(3)
        ]
        support_views = sorted({str(row["reference_view_id"]) for row in matching_rows})
        anchor_id = max(
            member_ids,
            key=lambda part_id: (
                max(
                    (_observation_weight(pixels=int(row["trusted_pixels"]), alignment_weight=float(row["evidence_weight"])) for row in observations[part_id]),
                    default=0.0,
                ),
                part_id,
            ),
        )
        component_id = _normalised_component_id(member_ids)
        components.append(
            {
                "component_id": component_id,
                "member_part_ids": member_ids,
                "anchor_part_id": anchor_id,
                "appearance_family": family,
                "canonical_reference_rgb": canonical_rgb,
                "supporting_view_ids": support_views,
                "total_trusted_pixels": total_pixels,
                "membership_authority": (
                    "same_view_rigid_part_id_projection_plus_sam3_foreground_"
                    "colour_and_proximity"
                ),
                "material_identity_assigned": False,
                "mdl_parameter_mutation_allowed": False,
                "independent_part_id_bindings_required": True,
            }
        )

    component_by_part = {
        part_id: component["component_id"]
        for component in components
        for part_id in component["member_part_ids"]
    }
    memberships = []
    for part_id in part_ids:
        rows = sorted(observations[part_id], key=lambda item: item["reference_view_id"])
        memberships.append(
            {
                "part_id": part_id,
                "component_id": component_by_part.get(part_id),
                "observation_count": len(rows),
                "status": (
                    "component_member"
                    if part_id in component_by_part
                    else "observed_independent"
                    if rows
                    else "unobserved_independent"
                ),
                "observations": rows,
            }
        )
    unsigned = {
        "schema_version": SCHEMA_VERSION,
        "status": "COMPLETED",
        "assignment_unit": "part_id",
        "inputs": {
            "rendered_registry": str(registry_path),
            "rendered_registry_sha256": _sha256_file(registry_path),
            "reference_manifest": str(manifest_path),
            "reference_manifest_sha256": _sha256_file(manifest_path),
            "camera_report": str(report_path),
            "camera_report_sha256": _sha256_file(report_path),
            "camera_scored_registry": str(scored_registry_path),
            "camera_scored_registry_sha256": _sha256_file(scored_registry_path),
        },
        "contract": {
            "whole_asset_camera_transform_only": True,
            "per_part_geometric_warp_applied": False,
            "sam3_role": "human_confirmed_whole_workpiece_foreground_only",
            "photo_part_segmentation_applied": False,
            "material_identity_mutated": False,
            "mdl_parameter_mutated": False,
            "palette_group_membership_used": False,
            "source_cad_material_membership_used": False,
            "unseen_views_cast_vote": False,
        },
        "policy": {
            "minimum_trusted_pixels": MINIMUM_TRUSTED_PIXELS,
            "minimum_foreground_overlap": MINIMUM_FOREGROUND_OVERLAP,
            "minimum_component_trusted_pixels": MINIMUM_COMPONENT_TRUSTED_PIXELS,
            "maximum_same_view_delta_e": MAXIMUM_SAME_VIEW_DELTA_E,
            "conflict_delta_e": CONFLICT_DELTA_E,
            "minimum_link_support": MINIMUM_LINK_SUPPORT,
            "same_view_spatial_proximity_required": True,
            "downweighted_view_cannot_create_link_alone": True,
        },
        "camera_alignment": {
            view_id: {
                key: value
                for key, value in row.items()
                if key != "bbox_affine"
            }
            for view_id, row in sorted(acceptance.items())
        },
        "rejected_camera_view_ids": rejected_view_ids,
        "components": sorted(components, key=lambda item: item["component_id"]),
        "accepted_links": accepted_links,
        "rejected_conflicting_links": [
            {"part_ids": list(key), "conflicts": value}
            for key, value in sorted(conflicts.items())
            if value
        ],
        "part_memberships": memberships,
        "summary": {
            "part_count": len(part_ids),
            "observed_part_count": sum(bool(observations[part_id]) for part_id in part_ids),
            "component_count": len(components),
            "component_member_count": len(component_by_part),
            "independent_observed_part_count": sum(
                bool(observations[part_id]) and part_id not in component_by_part
                for part_id in part_ids
            ),
            "unobserved_independent_part_count": sum(
                not observations[part_id] for part_id in part_ids
            ),
            "accepted_link_count": len(accepted_links),
            "conflicting_link_count": sum(bool(value) for value in conflicts.values()),
            "exact_part_id_cover": len(memberships) == len(part_ids),
        },
    }
    return {**unsigned, "integrity": {"document_sha256": _canonical_sha256(unsigned)}}


def _write_object(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rendered-registry", required=True, type=Path)
    parser.add_argument("--reference-manifest", required=True, type=Path)
    parser.add_argument("--camera-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        document = build_appearance_components(
            rendered_registry=args.rendered_registry,
            reference_manifest=args.reference_manifest,
            camera_report=args.camera_report,
        )
        output = args.output.expanduser().resolve()
        _write_object(output, document)
        print(json.dumps({"output": str(output), **document["summary"]}, ensure_ascii=False, indent=2))
        return 0
    except (OSError, AppearanceComponentError) as exc:
        print(json.dumps({"status": "INPUT_ERROR", "error": str(exc)}, ensure_ascii=False), flush=True)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
