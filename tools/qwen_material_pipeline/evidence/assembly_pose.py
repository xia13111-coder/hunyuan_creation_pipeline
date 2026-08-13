#!/usr/bin/env python3
"""Repair coherent local assembly displacement with bounded rigid motion.

This stage deliberately runs *after* global camera calibration.  It discovers
multi-Part residuals that share one authored Xform subtree, estimates a
camera-plane translation from the sealed residual image, and confirms a small
set of translations with real CAD renders.  No Part ID or asset name is
special-cased and no per-Part image warp is permitted.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from qwen_material_pipeline.evidence.camera_calibration import (
    _part_color,
    _robust_part_consensus,
    _part_residual_attribution,
    _read_object,
    _reference_image,
    _reference_masks,
    _score_candidates,
    _write_residual_audit,
)


REPORT_SCHEMA_VERSION = "qwen-assembly-pose-optimization/v1"
OVERRIDE_SCHEMA_VERSION = "qwen-assembly-pose-overrides/v1"
ASSEMBLY_CLASSIFICATION = "assembly_state_or_geometry_mismatch"


def _write_object(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _side_row(
    report: Mapping[str, Any], reference_view_id: str
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    for raw in report.get("views", []):
        if (
            isinstance(raw, Mapping)
            and raw.get("reference_view_id") == reference_view_id
            and isinstance(raw.get("final"), Mapping)
        ):
            return raw, raw["final"]
    raise ValueError(f"Camera report has no final view {reference_view_id!r}")


def discover_assembly_candidate(
    *, report: Mapping[str, Any], reference_view_id: str
) -> dict[str, Any]:
    """Choose the strongest coherent multi-Part residual without asset labels."""

    _, score = _side_row(report, reference_view_id)
    candidates = [
        dict(raw)
        for raw in score.get("assembly_residual_clusters", [])
        if isinstance(raw, Mapping)
        and raw.get("classification") == ASSEMBLY_CLASSIFICATION
        and isinstance(raw.get("assembly_subtree"), str)
        and str(raw["assembly_subtree"]).startswith("/")
        and int(raw.get("part_count", 0)) >= 2
        and float(raw.get("residual_direction_coherence", 0.0)) >= 0.7
        and float(raw.get("minimum_inside_reference_ratio", 0.0)) <= 0.1
    ]
    if not candidates:
        raise ValueError(
            f"No coherent assembly residual exists for {reference_view_id!r}"
        )
    candidates.sort(
        key=lambda raw: (
            -int(raw.get("projected_pixels", 0)),
            -float(raw.get("median_residual_px", 0.0)),
            str(raw["assembly_subtree"]),
        )
    )
    return candidates[0]


def _ids_image(
    registry: Mapping[str, Any], registry_path: Path, view_id: str
) -> np.ndarray:
    for raw in registry.get("render_set", {}).get("views", []):
        if isinstance(raw, Mapping) and raw.get("view_id") == view_id:
            value = raw.get("part_ids_raw") or raw.get("part_ids")
            source = Path(str(value)).expanduser()
            if not source.is_absolute():
                source = registry_path.parent / source
            image = cv2.imread(str(source.resolve(strict=True)), cv2.IMREAD_COLOR)
            if image is None:
                break
            return image
    raise ValueError(f"Rendered registry has no Part-ID image for {view_id!r}")


def _member_part_ids(
    registry: Mapping[str, Any], assembly_subtree: str
) -> list[str]:
    prefix = assembly_subtree.rstrip("/") + "/"
    members = sorted(
        str(raw["part_id"])
        for raw in registry.get("parts", [])
        if isinstance(raw, Mapping)
        and isinstance(raw.get("part_id"), str)
        and isinstance(raw.get("prim_path"), str)
        and str(raw["prim_path"]).startswith(prefix)
    )
    if len(members) < 2:
        raise ValueError("Assembly subtree does not contain multiple registered Parts")
    return members


def _candidate_assembly_subtrees(
    *, registry: Mapping[str, Any], detected_subtree: str
) -> list[tuple[str, list[str]]]:
    """Return the detected subtree and at most one bounded parent assembly.

    Small sibling parts can fall below the robust residual pixel threshold even
    though they belong to the same authored mechanism.  Testing one parent is
    therefore useful, but walking to the asset root would turn a local repair
    back into a global object transform.  The deterministic member/ratio caps
    preserve that boundary without using asset names.
    """

    part_count = sum(
        isinstance(raw, Mapping) and isinstance(raw.get("part_id"), str)
        for raw in registry.get("parts", [])
    )
    detected_members = _member_part_ids(registry, detected_subtree)
    output = [(detected_subtree, detected_members)]
    if "/" not in detected_subtree.rstrip("/"):
        return output
    parent = detected_subtree.rstrip("/").rsplit("/", 1)[0]
    if not parent:
        return output
    parent_members = _member_part_ids(registry, parent)
    if (
        len(parent_members) > len(detected_members)
        and len(parent_members) <= 128
        and len(parent_members) <= max(8, math.ceil(0.25 * part_count))
    ):
        output.append((parent, parent_members))
    return output


def _part_mask(ids: np.ndarray, part_ids: Sequence[str]) -> np.ndarray:
    output = np.zeros(ids.shape[:2], dtype=np.uint8)
    for part_id in part_ids:
        red, green, blue = _part_color(part_id)
        output[np.all(ids == np.asarray((blue, green, red), np.uint8), axis=2)] = 1
    return output


def _foreground_mask(ids: np.ndarray, registry: Mapping[str, Any]) -> np.ndarray:
    part_ids = [
        str(raw["part_id"])
        for raw in registry.get("parts", [])
        if isinstance(raw, Mapping) and isinstance(raw.get("part_id"), str)
    ]
    return _part_mask(ids, part_ids)


def _largest_matching_residual_component(
    *, reference_only: np.ndarray, cluster: np.ndarray
) -> np.ndarray:
    cluster_y, cluster_x = np.nonzero(cluster > 0)
    cluster_area = len(cluster_x)
    if cluster_area < 20:
        raise ValueError("Assembly cluster is not sufficiently visible")
    cluster_centroid = np.asarray((cluster_x.mean(), cluster_y.mean()))
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        (reference_only > 0).astype(np.uint8), 8
    )
    diagonal = max(1.0, math.hypot(*reference_only.shape))
    ranked: list[tuple[float, int]] = []
    for component_id in range(1, count):
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        ratio = area / cluster_area
        if area < 20 or not 0.2 <= ratio <= 5.0:
            continue
        distance = float(np.linalg.norm(centroids[component_id] - cluster_centroid))
        if distance > 0.3 * diagonal:
            continue
        ranked.append(
            (distance / diagonal + 0.25 * abs(math.log(max(1e-9, ratio))), component_id)
        )
    if not ranked:
        raise ValueError("No compatible reference-only component matches the assembly")
    component_id = min(ranked)[1]
    return (labels == component_id).astype(np.uint8)


def _camera_basis(spec: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    direction = np.asarray(spec.get("analysis_direction"), dtype=np.float64)
    up = np.asarray(spec.get("analysis_up_axis"), dtype=np.float64)
    if direction.shape != (3,) or up.shape != (3,):
        raise ValueError(
            "Assembly pose estimation requires camera direction and up axis"
        )
    forward = -direction / np.linalg.norm(direction)
    up = up / np.linalg.norm(up)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    return right, up


def _world_bbox_points(
    registry: Mapping[str, Any], member_part_ids: Sequence[str]
) -> np.ndarray:
    selected = set(member_part_ids)
    points: list[np.ndarray] = []
    for raw in registry.get("parts", []):
        if not isinstance(raw, Mapping) or raw.get("part_id") not in selected:
            continue
        bounds = np.asarray(raw.get("world_bbox"), dtype=np.float64)
        if bounds.shape != (2, 3) or not np.isfinite(bounds).all():
            raise ValueError(f"Part {raw.get('part_id')} has no finite world bbox")
        for bits in range(8):
            points.append(
                np.asarray(
                    [bounds[(bits >> axis) & 1, axis] for axis in range(3)],
                    dtype=np.float64,
                )
            )
    if not points:
        raise ValueError("Assembly has no world bounding boxes")
    return np.asarray(points)


def estimate_world_translation(
    *,
    registered_cluster: np.ndarray,
    raw_cluster: np.ndarray,
    target_component: np.ndarray,
    affine: np.ndarray,
    camera_spec: Mapping[str, Any],
    registry: Mapping[str, Any],
    member_part_ids: Sequence[str],
) -> tuple[list[float], list[float], list[float]]:
    """Lift a 2-D component displacement into the camera image plane."""

    source_y, source_x = np.nonzero(registered_cluster > 0)
    target_y, target_x = np.nonzero(target_component > 0)
    raw_y, raw_x = np.nonzero(raw_cluster > 0)
    if not len(source_x) or not len(target_x) or not len(raw_x):
        raise ValueError("Assembly translation requires non-empty masks")
    scale_x = float(np.linalg.norm(affine[:, 0]))
    scale_y = float(np.linalg.norm(affine[:, 1]))
    if scale_x <= 1e-6 or scale_y <= 1e-6:
        raise ValueError("Camera affine scale is degenerate")
    raw_dx = (float(target_x.mean()) - float(source_x.mean())) / scale_x
    raw_dy = (float(target_y.mean()) - float(source_y.mean())) / scale_y
    right, up = _camera_basis(camera_spec)
    points = _world_bbox_points(registry, member_part_ids)
    world_width = float(np.ptp(points @ right))
    world_height = float(np.ptp(points @ up))
    world_per_x = world_width / max(1, int(np.ptp(raw_x)) + 1)
    world_per_y = world_height / max(1, int(np.ptp(raw_y)) + 1)
    horizontal = right * raw_dx * world_per_x
    vertical = -up * raw_dy * world_per_y
    translation = horizontal + vertical
    return (
        [float(value) for value in translation],
        [float(value) for value in horizontal],
        [float(value) for value in vertical],
    )


def candidate_translations(
    horizontal_component: Sequence[float], vertical_component: Sequence[float]
) -> list[list[float]]:
    """Return a bounded 3x3 image-plane neighborhood around one proposal."""

    horizontal_value = np.asarray(horizontal_component, dtype=np.float64)
    vertical_value = np.asarray(vertical_component, dtype=np.float64)
    if (
        horizontal_value.shape != (3,)
        or vertical_value.shape != (3,)
        or not np.isfinite(horizontal_value).all()
        or not np.isfinite(vertical_value).all()
    ):
        raise ValueError("Assembly camera-plane components must be finite vectors")
    output: list[list[float]] = []
    for horizontal in (0.5, 1.0, 1.5):
        for vertical in (0.75, 1.0, 1.25):
            proposal = horizontal_value * horizontal + vertical_value * vertical
            output.append([float(item) for item in proposal])
    unique: dict[tuple[float, ...], list[float]] = {}
    for raw in output:
        unique.setdefault(tuple(round(value, 9) for value in raw), raw)
    return list(unique.values())


def _mask_metrics(target: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    target_bool = target > 0
    candidate_bool = candidate > 0
    intersection = int(np.count_nonzero(target_bool & candidate_bool))
    union = int(np.count_nonzero(target_bool | candidate_bool))
    target_pixels = int(np.count_nonzero(target_bool))
    candidate_pixels = int(np.count_nonzero(candidate_bool))
    target_edge = cv2.morphologyEx(
        target_bool.astype(np.uint8),
        cv2.MORPH_GRADIENT,
        np.ones((3, 3), np.uint8),
    )
    candidate_edge = cv2.morphologyEx(
        candidate_bool.astype(np.uint8),
        cv2.MORPH_GRADIENT,
        np.ones((3, 3), np.uint8),
    )
    target_distance = cv2.distanceTransform(1 - target_edge, cv2.DIST_L2, 3)
    candidate_distance = cv2.distanceTransform(1 - candidate_edge, cv2.DIST_L2, 3)
    distances = np.concatenate(
        (target_distance[candidate_edge > 0], candidate_distance[target_edge > 0])
    )
    target_y, target_x = np.nonzero(target_bool)
    candidate_y, candidate_x = np.nonzero(candidate_bool)
    centroid = math.hypot(
        float(target_x.mean() - candidate_x.mean()),
        float(target_y.mean() - candidate_y.mean()),
    )
    iou = intersection / max(1, union)
    precision = intersection / max(1, candidate_pixels)
    recall = intersection / max(1, target_pixels)
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    diagonal = max(1.0, math.hypot(*target.shape))
    boundary_p95 = float(np.percentile(distances, 95)) if len(distances) else diagonal
    objective = 0.55 * iou + 0.25 * f1 + 0.20 * math.exp(
        -boundary_p95 / max(3.0, 0.03 * diagonal)
    )
    return {
        "iou": round(iou, 8),
        "precision": round(precision, 8),
        "recall": round(recall, 8),
        "f1": round(f1, 8),
        "boundary_p95_px": round(boundary_p95, 8),
        "centroid_error_px": round(centroid, 8),
        "objective": round(objective, 8),
    }


def _regional_support_bounds(
    *,
    registered_cluster: np.ndarray,
    target_component: np.ndarray,
) -> list[int]:
    """Build a deterministic neighborhood around the displaced mechanism.

    Scoring only the largest residual component can prefer a leaf Xform while
    leaving smaller siblings at the old pose.  The support rectangle includes
    both the current and requested locations plus 25% padding, so every nearby
    reference-only and CAD-only pixel votes on the assembly hierarchy.
    """

    combined = (registered_cluster > 0) | (target_component > 0)
    rows, columns = np.nonzero(combined)
    if not len(columns):
        raise ValueError("Regional assembly support cannot be empty")
    extent = max(
        int(columns.max() - columns.min() + 1),
        int(rows.max() - rows.min() + 1),
    )
    padding = max(12, int(round(0.25 * extent)))
    height, width = combined.shape
    return [
        max(0, int(columns.min()) - padding),
        max(0, int(rows.min()) - padding),
        min(width, int(columns.max()) + 1 + padding),
        min(height, int(rows.max()) + 1 + padding),
    ]


def _regional_overlap_metrics(
    *,
    reference_mask: np.ndarray,
    registered_foreground: np.ndarray,
    support_bounds_xyxy: Sequence[int],
) -> dict[str, float | int]:
    if len(support_bounds_xyxy) != 4:
        raise ValueError("Regional support must be [left, top, right, bottom]")
    left, top, right, bottom = (int(value) for value in support_bounds_xyxy)
    if not (0 <= left < right <= reference_mask.shape[1]):
        raise ValueError("Regional horizontal support is outside the image")
    if not (0 <= top < bottom <= reference_mask.shape[0]):
        raise ValueError("Regional vertical support is outside the image")
    reference = reference_mask[top:bottom, left:right] > 0
    rendered = registered_foreground[top:bottom, left:right] > 0
    intersection = int(np.count_nonzero(reference & rendered))
    union = int(np.count_nonzero(reference | rendered))
    reference_only = int(np.count_nonzero(reference & ~rendered))
    cad_only = int(np.count_nonzero(rendered & ~reference))
    return {
        "iou": round(intersection / max(1, union), 8),
        "intersection_pixels": intersection,
        "union_pixels": union,
        "reference_only_pixels": reference_only,
        "cad_only_pixels": cad_only,
        "mismatch_pixels": reference_only + cad_only,
    }


def _registered_foreground(
    *,
    registry_path: Path,
    view_id: str,
    affine: Sequence[Sequence[float]],
    target_shape: tuple[int, int],
) -> np.ndarray:
    registry = _read_object(registry_path)
    ids = _ids_image(registry, registry_path, view_id)
    return cv2.warpAffine(
        _foreground_mask(ids, registry),
        np.asarray(affine, dtype=np.float32),
        (target_shape[1], target_shape[0]),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def _select_pose_winner(
    *,
    candidates: Sequence[Mapping[str, Any]],
    baseline_local: Mapping[str, Any],
    baseline_regional: Mapping[str, Any],
    baseline_projection_iou: float,
    baseline_fixed_by_subtree: Mapping[str, float],
) -> Mapping[str, Any]:
    admitted = [
        raw
        for raw in candidates
        if float(raw["local"]["objective"])
        >= float(baseline_local["objective"]) + 0.15
        and float(raw["local"]["centroid_error_px"])
        <= 0.6 * float(baseline_local["centroid_error_px"])
        and float(raw["regional"]["iou"])
        >= float(baseline_regional["iou"]) + 0.15
        and float(raw["global"]["projection_iou"])
        >= baseline_projection_iou - 0.003
        and float(raw["global"]["fixed_geometry_consensus_residual_px"])
        <= baseline_fixed_by_subtree[str(raw["assembly_subtree"])] + 0.5
    ]
    if not admitted:
        raise RuntimeError("No assembly pose candidate passed local and global gates")
    # The regional residual is the hierarchy authority.  A leaf can still be
    # selected, but not merely because it fits one large component while its
    # sibling geometry remains visibly displaced nearby.
    return max(
        admitted,
        key=lambda raw: (
            float(raw["regional"]["iou"]),
            float(raw["global"]["projection_iou"]),
            float(raw["local"]["objective"]),
            -len(raw["subtree_member_part_ids"]),
            -float(raw["translation_norm"]),
        ),
    )


def discover_nested_residual_xform(
    *, residual_audit: Mapping[str, Any], parent_subtree: str
) -> dict[str, Any]:
    """Select one isolated child Xform that remains wrong after parent repair."""

    prefix = parent_subtree.rstrip("/") + "/"
    candidates: list[dict[str, Any]] = []
    for raw in residual_audit.get("cad_part_residual_attribution", []):
        if not isinstance(raw, Mapping):
            continue
        prim_path = raw.get("prim_path")
        if (
            not isinstance(prim_path, str)
            or not prim_path.startswith(prefix)
            or not prim_path.endswith("/Mesh")
            or int(raw.get("projected_pixels", 0)) < 32
            or float(raw.get("inside_reference_ratio", 1.0)) > 0.6
        ):
            continue
        item = dict(raw)
        item["assembly_subtree"] = prim_path.rsplit("/", 1)[0]
        candidates.append(item)
    if not candidates:
        raise ValueError("No isolated nested Xform residual passed discovery gates")
    candidates.sort(
        key=lambda raw: (
            -int(raw.get("cad_only_pixels", 0)),
            -int(raw.get("projected_pixels", 0)),
            str(raw["prim_path"]),
        )
    )
    return candidates[0]


def search_nested_screen_translation(
    *,
    reference_mask: np.ndarray,
    registered_foreground: np.ndarray,
    registered_part: np.ndarray,
    support_bounds_xyxy: Sequence[int],
    maximum_shift_px: int | None = None,
) -> dict[str, Any]:
    """Find a rigid 2-D target for a thin nested part in the local residual.

    This is only a proposal generator.  The chosen displacement must later be
    lifted through render-measured world responses and confirmed by a new CAD
    render; the image is never warped into a deliverable.
    """

    if reference_mask.shape != registered_foreground.shape or (
        reference_mask.shape != registered_part.shape
    ):
        raise ValueError("Nested residual masks must share one image shape")
    if int(np.count_nonzero(registered_part)) < 8:
        raise ValueError("Nested residual part is not sufficiently visible")
    left, top, right, bottom = (int(value) for value in support_bounds_xyxy)
    height, width = reference_mask.shape
    diagonal = math.hypot(right - left, bottom - top)
    limit = (
        max(8, int(round(0.35 * diagonal)))
        if maximum_shift_px is None
        else int(maximum_shift_px)
    )
    if not 1 <= limit <= max(height, width):
        raise ValueError("Nested residual screen search has an invalid bound")
    fixed = (registered_foreground > 0) & ~(registered_part > 0)
    reference = reference_mask > 0
    local_reference = reference[top:bottom, left:right]
    baseline_centroid = np.asarray(
        tuple(reversed(np.mean(np.argwhere(registered_part > 0), axis=0))),
        dtype=np.float64,
    )
    ranked: list[tuple[tuple[float, ...], dict[str, Any]]] = []
    for dy in range(-limit, limit + 1):
        for dx in range(-limit, limit + 1):
            shifted = cv2.warpAffine(
                (registered_part > 0).astype(np.uint8),
                np.asarray(((1.0, 0.0, dx), (0.0, 1.0, dy)), dtype=np.float32),
                (width, height),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            ) > 0
            candidate = fixed | shifted
            local_candidate = candidate[top:bottom, left:right]
            mismatch = int(np.count_nonzero(local_reference ^ local_candidate))
            overlap = int(np.count_nonzero(reference & shifted))
            part_rows = np.argwhere(shifted)
            centroid = np.asarray(
                tuple(reversed(np.mean(part_rows, axis=0))), dtype=np.float64
            )
            shift = centroid - baseline_centroid
            value = {
                "screen_translation_px": [float(shift[0]), float(shift[1])],
                "requested_integer_translation_px": [dx, dy],
                "regional_mismatch_pixels": mismatch,
                "part_reference_overlap_pixels": overlap,
            }
            ranked.append(
                (
                    (
                        float(mismatch),
                        float(-overlap),
                        float(math.hypot(dx, dy)),
                        float(dy),
                        float(dx),
                    ),
                    value,
                )
            )
    return min(ranked, key=lambda item: item[0])[1]


def solve_world_translation_from_render_responses(
    *,
    target_screen_translation_px: Sequence[float],
    probe_world_translations: Sequence[Sequence[float]],
    measured_screen_translations_px: Sequence[Sequence[float]],
) -> dict[str, Any]:
    """Invert a 2x2 render-measured image/world translation Jacobian."""

    target = np.asarray(target_screen_translation_px, dtype=np.float64)
    world = np.asarray(probe_world_translations, dtype=np.float64)
    measured = np.asarray(measured_screen_translations_px, dtype=np.float64)
    if target.shape != (2,) or world.shape != (2, 3) or measured.shape != (2, 2):
        raise ValueError("Nested translation response inputs have invalid shapes")
    if not np.isfinite(target).all() or not np.isfinite(world).all() or (
        not np.isfinite(measured).all()
    ):
        raise ValueError("Nested translation response inputs must be finite")
    response = measured.T
    determinant = float(np.linalg.det(response))
    if abs(determinant) < 1e-6:
        raise ValueError("Nested translation render responses are degenerate")
    coefficients = np.linalg.solve(response, target)
    translation = coefficients @ world
    return {
        "probe_coefficients": [float(value) for value in coefficients],
        "world_translation": [float(value) for value in translation],
        "world_translation_norm": float(np.linalg.norm(translation)),
        "response_determinant": determinant,
    }


def _mask_centroid(mask: np.ndarray) -> np.ndarray:
    rows, columns = np.nonzero(mask > 0)
    if not len(columns):
        raise ValueError("Rendered nested assembly part is not visible")
    return np.asarray((float(columns.mean()), float(rows.mean())), dtype=np.float64)


def _nested_override_document(
    *,
    parent_subtree: str,
    parent_translation: Sequence[float],
    child_subtree: str,
    child_translation: Sequence[float],
) -> dict[str, Any]:
    return {
        "schema_version": OVERRIDE_SCHEMA_VERSION,
        "overrides": [
            {
                "prim_path": parent_subtree,
                "world_translation": [float(value) for value in parent_translation],
            },
            {
                "prim_path": child_subtree,
                "world_translation": [float(value) for value in child_translation],
            },
        ],
    }


def _project_cluster(
    *,
    registry_path: Path,
    view_id: str,
    part_ids: Sequence[str],
    affine: Sequence[Sequence[float]],
    target_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    registry = _read_object(registry_path)
    ids = _ids_image(registry, registry_path, view_id)
    raw = _part_mask(ids, part_ids)
    matrix = np.asarray(affine, dtype=np.float32)
    projected = cv2.warpAffine(
        raw,
        matrix,
        (target_shape[1], target_shape[0]),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return raw, projected


def _fixed_geometry_consensus(
    *,
    registry_path: Path,
    view_id: str,
    excluded_part_ids: Sequence[str],
    affine: Sequence[Sequence[float]],
    reference_image: np.ndarray,
    reference_mask: np.ndarray,
) -> dict[str, Any]:
    """Score only geometry outside the assembly being optimized.

    The moving assembly must not veto its own correction.  Conversely, every
    fixed visible Part remains an independent guard against a camera/frame
    regression or an accidentally over-broad subtree edit.
    """

    registry = _read_object(registry_path)
    excluded = set(excluded_part_ids)
    fixed_parts = [
        raw
        for raw in registry.get("parts", [])
        if isinstance(raw, Mapping) and raw.get("part_id") not in excluded
    ]
    ids = _ids_image(registry, registry_path, view_id)
    return _robust_part_consensus(
        ids=ids,
        parts=fixed_parts,
        affine=np.asarray(affine, dtype=np.float32),
        reference_image=reference_image,
        reference_mask=reference_mask,
    )


def _render_candidate(
    *,
    python_sh: Path,
    source_registry: Path,
    final_view_specs: Path,
    override_path: Path,
    output_dir: Path,
    resolution: int,
    rt_subframes: int,
    repository_root: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=False)
    log_path = output_dir / "render.log"
    command = [
        str(python_sh),
        "-m",
        "qwen_material_pipeline",
        "usd",
        "render",
        "--registry",
        str(source_registry),
        "--output-dir",
        str(output_dir),
        "--resolution",
        str(resolution),
        "--view-specs",
        str(final_view_specs),
        "--assembly-pose-overrides",
        str(override_path),
        "--rt-subframes",
        str(rt_subframes),
        "--lighting-profile",
        "material-neutral",
        "--analysis-up-axis",
        "z",
        "--analysis-front-axis=-y",
    ]
    environment = dict(os.environ)
    tools_path = str(repository_root / "tools")
    environment["PYTHONPATH"] = (
        tools_path
        if not environment.get("PYTHONPATH")
        else tools_path + os.pathsep + environment["PYTHONPATH"]
    )
    with log_path.open("w", encoding="utf-8") as stream:
        result = subprocess.run(
            command,
            cwd=repository_root,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"Assembly candidate render failed ({result.returncode}): {log_path}"
        )
    registry_path = output_dir / "part_registry.rendered.json"
    if not registry_path.is_file():
        raise RuntimeError(
            "Assembly candidate renderer produced no sealed registry; "
            f"python.sh may have normalized an internal failure: {log_path}"
        )
    return registry_path.resolve(strict=True)


def _reused_candidate_registry(
    *,
    reuse_report: Mapping[str, Any],
    camera_report_path: Path,
    reference_view: str,
    candidate_id: str,
    assembly_subtree: str,
    world_translation: Sequence[float],
    source_registry: Mapping[str, Any],
    final_view_specs_path: Path,
    resolution: int,
    rt_subframes: int,
) -> Path:
    """Admit one prior real render only when every render input is identical."""

    if reuse_report.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise ValueError("Reusable assembly report has an unsupported schema")
    if reuse_report.get("status") != "PASS":
        raise ValueError("Reusable assembly report is not sealed PASS")
    if Path(str(reuse_report.get("camera_report"))).resolve(strict=True) != camera_report_path:
        raise ValueError("Reusable candidates belong to another camera report")
    if reuse_report.get("reference_view_id") != reference_view:
        raise ValueError("Reusable candidates belong to another reference view")
    matches = [
        raw
        for raw in reuse_report.get("candidates", [])
        if isinstance(raw, Mapping) and raw.get("candidate_id") == candidate_id
    ]
    if len(matches) != 1:
        raise ValueError(f"Reusable report does not exactly bind {candidate_id}")
    candidate = matches[0]
    if candidate.get("assembly_subtree") != assembly_subtree:
        raise ValueError(f"Reusable subtree changed for {candidate_id}")
    reused_translation = np.asarray(candidate.get("world_translation"), dtype=np.float64)
    expected_translation = np.asarray(world_translation, dtype=np.float64)
    if (
        reused_translation.shape != (3,)
        or expected_translation.shape != (3,)
        or not np.array_equal(reused_translation, expected_translation)
    ):
        raise ValueError(f"Reusable translation changed for {candidate_id}")
    override_path = Path(str(candidate.get("override"))).resolve(strict=True)
    expected_override = {
        "schema_version": OVERRIDE_SCHEMA_VERSION,
        "overrides": [
            {
                "prim_path": assembly_subtree,
                "world_translation": [float(value) for value in world_translation],
            }
        ],
    }
    if _read_object(override_path) != expected_override:
        raise ValueError(f"Reusable override payload changed for {candidate_id}")
    registry_path = Path(str(candidate.get("rendered_registry"))).resolve(strict=True)
    registry = _read_object(registry_path)
    render_set = registry.get("render_set")
    if not isinstance(render_set, Mapping):
        raise ValueError(f"Reusable registry is not sealed for {candidate_id}")
    source_asset = Path(str(source_registry.get("asset_usd"))).resolve(strict=True)
    if Path(str(render_set.get("asset_usd"))).resolve(strict=True) != source_asset:
        raise ValueError(f"Reusable source asset changed for {candidate_id}")
    if Path(str(render_set.get("custom_view_specs"))).resolve(strict=True) != final_view_specs_path:
        raise ValueError(f"Reusable camera specs changed for {candidate_id}")
    if Path(str(render_set.get("assembly_pose_overrides"))).resolve(strict=True) != override_path:
        raise ValueError(f"Reusable registry/override binding changed for {candidate_id}")
    if render_set.get("assembly_pose_override_count") != 1:
        raise ValueError(f"Reusable registry has the wrong override count for {candidate_id}")
    if render_set.get("resolution") != [resolution, resolution]:
        raise ValueError(f"Reusable resolution changed for {candidate_id}")
    if render_set.get("rt_subframes") != rt_subframes:
        raise ValueError(f"Reusable RT subframes changed for {candidate_id}")
    # This also resolves and decodes the actual Part-ID image consumed below.
    _ids_image(registry, registry_path, reference_view)
    return registry_path


def optimize(args: argparse.Namespace) -> dict[str, Any]:
    camera_report_path = args.camera_report.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Assembly optimization output already exists: {output}")
    output.mkdir(parents=True)
    report = _read_object(camera_report_path)
    reuse_report = (
        _read_object(args.reuse_candidates_from.resolve(strict=True))
        if args.reuse_candidates_from is not None
        else None
    )
    _, baseline_score = _side_row(report, args.reference_view)
    cluster = discover_assembly_candidate(
        report=report, reference_view_id=args.reference_view
    )
    source_registry_path = Path(str(report["source_registry"])).resolve(strict=True)
    source_registry = _read_object(source_registry_path)
    final_registry_path = Path(str(report["final_rendered_registry"])).resolve(
        strict=True
    )
    final_registry = _read_object(final_registry_path)
    final_view_specs_path = Path(str(report["final_view_specs"])).resolve(strict=True)
    final_view_specs = _read_object(final_view_specs_path)
    camera_specs = [
        raw
        for raw in final_view_specs.get("views", [])
        if isinstance(raw, Mapping) and raw.get("view_id") == args.reference_view
    ]
    if len(camera_specs) != 1:
        raise ValueError("Final camera specs do not exactly cover the requested view")
    manifest_path = Path(str(report["reference_manifest"])).resolve(strict=True)
    reference_mask, reference_row = _reference_masks(manifest_path)[args.reference_view]
    reference_image = _reference_image(
        reference_row, manifest_path, reference_mask.shape
    )
    anchor_part_ids = _member_part_ids(
        source_registry, str(cluster["assembly_subtree"])
    )
    subtree_candidates = _candidate_assembly_subtrees(
        registry=source_registry,
        detected_subtree=str(cluster["assembly_subtree"]),
    )
    ids = _ids_image(final_registry, final_registry_path, args.reference_view)
    raw_cluster = _part_mask(ids, anchor_part_ids)
    matrix = np.asarray(
        baseline_score["whole_asset_similarity"]["bbox_affine"], dtype=np.float32
    )
    registered_cluster = cv2.warpAffine(
        raw_cluster,
        matrix,
        (reference_mask.shape[1], reference_mask.shape[0]),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    foreground = cv2.warpAffine(
        _foreground_mask(ids, final_registry),
        matrix,
        (reference_mask.shape[1], reference_mask.shape[0]),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    reference_only = ((reference_mask > 0) & (foreground == 0)).astype(np.uint8)
    target_component = _largest_matching_residual_component(
        reference_only=reference_only,
        cluster=registered_cluster,
    )
    seed, horizontal_seed, vertical_seed = estimate_world_translation(
        registered_cluster=registered_cluster,
        raw_cluster=raw_cluster,
        target_component=target_component,
        affine=matrix,
        camera_spec=camera_specs[0],
        registry=source_registry,
        member_part_ids=anchor_part_ids,
    )
    baseline_local = _mask_metrics(target_component, registered_cluster)
    regional_support = _regional_support_bounds(
        registered_cluster=registered_cluster,
        target_component=target_component,
    )
    baseline_regional = _regional_overlap_metrics(
        reference_mask=reference_mask,
        registered_foreground=foreground,
        support_bounds_xyxy=regional_support,
    )
    baseline_fixed_by_subtree: dict[str, float] = {}
    for subtree, subtree_part_ids in subtree_candidates:
        consensus = _fixed_geometry_consensus(
            registry_path=final_registry_path,
            view_id=args.reference_view,
            excluded_part_ids=subtree_part_ids,
            affine=baseline_score["whole_asset_similarity"]["bbox_affine"],
            reference_image=reference_image,
            reference_mask=reference_mask,
        )
        baseline_fixed_by_subtree[subtree] = float(
            consensus.get("rigid_consensus_residual_px", 0.0)
        )
    cv2.imwrite(str(output / "target_component.png"), target_component * 255)
    cv2.imwrite(str(output / "baseline_cluster.png"), registered_cluster * 255)
    candidates: list[dict[str, Any]] = []
    translations = candidate_translations(horizontal_seed, vertical_seed)
    total_candidates = len(subtree_candidates) * len(translations)
    candidate_index = 0
    for subtree_index, (subtree, subtree_part_ids) in enumerate(
        subtree_candidates, start=1
    ):
        for translation_index, translation in enumerate(translations, start=1):
            candidate_index += 1
            candidate_dir = (
                output
                / "candidates"
                / f"subtree_{subtree_index:02d}"
                / f"candidate_{translation_index:02d}"
            )
            override_path = candidate_dir / "assembly_pose_overrides.json"
            _write_object(
                override_path,
                {
                    "schema_version": OVERRIDE_SCHEMA_VERSION,
                    "overrides": [
                        {
                            "prim_path": subtree,
                            "world_translation": translation,
                        }
                    ],
                },
            )
            if reuse_report is not None:
                registry_path = _reused_candidate_registry(
                    reuse_report=reuse_report,
                    camera_report_path=camera_report_path,
                    reference_view=args.reference_view,
                    candidate_id=(
                        f"subtree_{subtree_index:02d}_candidate_"
                        f"{translation_index:02d}"
                    ),
                    assembly_subtree=subtree,
                    world_translation=translation,
                    source_registry=source_registry,
                    final_view_specs_path=final_view_specs_path,
                    resolution=args.resolution,
                    rt_subframes=args.rt_subframes,
                )
            else:
                registry_path = _render_candidate(
                    python_sh=args.python_sh.resolve(strict=True),
                    source_registry=source_registry_path,
                    final_view_specs=final_view_specs_path,
                    override_path=override_path,
                    output_dir=candidate_dir / "renders",
                    resolution=args.resolution,
                    rt_subframes=args.rt_subframes,
                    repository_root=args.repository_root.resolve(strict=True),
                )
            score, _ = _score_candidates(
                reference_id=args.reference_view,
                reference_mask=reference_mask,
                reference_image=reference_image,
                registry_path=registry_path,
            )
            _, projected = _project_cluster(
                registry_path=registry_path,
                view_id=args.reference_view,
                part_ids=anchor_part_ids,
                affine=score["whole_asset_similarity"]["bbox_affine"],
                target_shape=reference_mask.shape,
            )
            local = _mask_metrics(target_component, projected)
            registered_foreground = _registered_foreground(
                registry_path=registry_path,
                view_id=args.reference_view,
                affine=score["whole_asset_similarity"]["bbox_affine"],
                target_shape=reference_mask.shape,
            )
            regional = _regional_overlap_metrics(
                reference_mask=reference_mask,
                registered_foreground=registered_foreground,
                support_bounds_xyxy=regional_support,
            )
            fixed_consensus = _fixed_geometry_consensus(
                registry_path=registry_path,
                view_id=args.reference_view,
                excluded_part_ids=subtree_part_ids,
                affine=score["whole_asset_similarity"]["bbox_affine"],
                reference_image=reference_image,
                reference_mask=reference_mask,
            )
            cv2.imwrite(
                str(candidate_dir / "registered_cluster.png"),
                projected * 255,
            )
            candidates.append(
                {
                    "candidate_id": (
                        f"subtree_{subtree_index:02d}_candidate_"
                        f"{translation_index:02d}"
                    ),
                    "assembly_subtree": subtree,
                    "subtree_member_part_ids": subtree_part_ids,
                    "world_translation": translation,
                    "translation_norm": float(np.linalg.norm(translation)),
                    "override": str(override_path),
                    "rendered_registry": str(registry_path),
                    "render_reused": reuse_report is not None,
                    "registered_cluster": str(
                        candidate_dir / "registered_cluster.png"
                    ),
                    "local": local,
                    "regional": regional,
                    "global": {
                        "projection_iou": score["projection_iou"],
                        "boundary_p95_px": score["boundary_p95_px"],
                        "structure_p75_px": score["structure_p75_px"],
                        "rigid_consensus_residual_px": score.get(
                            "rigid_consensus_residual_px"
                        ),
                        "fixed_geometry_consensus_residual_px": (
                            fixed_consensus.get("rigid_consensus_residual_px")
                        ),
                    },
                    "score": score,
                }
            )
            print(
                f"[ASSEMBLY] {candidate_index}/{total_candidates} "
                f"subtree_parts={len(subtree_part_ids)} "
                f"local_iou={local['iou']:.4f} "
                f"regional_iou={regional['iou']:.4f} "
                f"p95={local['boundary_p95_px']:.2f} "
                f"global_iou={score['projection_iou']:.4f}",
                flush=True,
            )
    winner = _select_pose_winner(
        candidates=candidates,
        baseline_local=baseline_local,
        baseline_regional=baseline_regional,
        baseline_projection_iou=float(baseline_score["projection_iou"]),
        baseline_fixed_by_subtree=baseline_fixed_by_subtree,
    )
    winning_override = output / "assembly_pose_overrides.json"
    shutil.copy2(winner["override"], winning_override)
    winning_registry = Path(winner["rendered_registry"])
    winning_score = winner["score"]
    residual = _write_residual_audit(
        reference_id=args.reference_view,
        reference_mask=reference_mask,
        foreground=_foreground_mask(
            _ids_image(
                _read_object(winning_registry),
                winning_registry,
                args.reference_view,
            ),
            _read_object(winning_registry),
        ),
        score=winning_score,
        part_residuals=_part_residual_attribution(
            registry_path=winning_registry,
            view_id=args.reference_view,
            reference_mask=reference_mask,
            score=winning_score,
        ),
        output_dir=output / "residual_audit",
    )
    nested_refinement: dict[str, Any] = {
        "status": "NOT_ATTEMPTED",
        "accepted": False,
    }
    if not args.disable_nested_refinement:
        try:
            nested = discover_nested_residual_xform(
                residual_audit=residual,
                parent_subtree=str(winner["assembly_subtree"]),
            )
            nested_part_id = str(nested["part_id"])
            nested_subtree = str(nested["assembly_subtree"])
            _, baseline_nested_part = _project_cluster(
                registry_path=winning_registry,
                view_id=args.reference_view,
                part_ids=[nested_part_id],
                affine=winning_score["whole_asset_similarity"]["bbox_affine"],
                target_shape=reference_mask.shape,
            )
            parent_registered_foreground = _registered_foreground(
                registry_path=winning_registry,
                view_id=args.reference_view,
                affine=winning_score["whole_asset_similarity"]["bbox_affine"],
                target_shape=reference_mask.shape,
            )
            screen_proposal = search_nested_screen_translation(
                reference_mask=reference_mask,
                registered_foreground=parent_registered_foreground,
                registered_part=baseline_nested_part,
                support_bounds_xyxy=regional_support,
            )
            all_part_ids = [
                str(raw["part_id"])
                for raw in source_registry.get("parts", [])
                if isinstance(raw, Mapping) and isinstance(raw.get("part_id"), str)
            ]
            asset_points = _world_bbox_points(source_registry, all_part_ids)
            asset_diagonal = float(np.linalg.norm(np.ptp(asset_points, axis=0)))
            probe_length = max(1e-4, 0.02 * asset_diagonal)
            camera_right, camera_up = _camera_basis(camera_specs[0])
            probe_world = [
                camera_right * probe_length,
                camera_up * probe_length,
            ]
            probe_registries: list[Path] = []
            measured_screen: list[list[float]] = []
            baseline_centroid = _mask_centroid(baseline_nested_part)
            for probe_name, probe_translation in zip(
                ("horizontal", "vertical"), probe_world, strict=True
            ):
                probe_dir = output / "nested_refinement" / f"probe_{probe_name}"
                probe_override = probe_dir / "assembly_pose_overrides.json"
                _write_object(
                    probe_override,
                    _nested_override_document(
                        parent_subtree=str(winner["assembly_subtree"]),
                        parent_translation=winner["world_translation"],
                        child_subtree=nested_subtree,
                        child_translation=probe_translation,
                    ),
                )
                probe_registry = _render_candidate(
                    python_sh=args.python_sh.resolve(strict=True),
                    source_registry=source_registry_path,
                    final_view_specs=final_view_specs_path,
                    override_path=probe_override,
                    output_dir=probe_dir / "renders",
                    resolution=args.resolution,
                    rt_subframes=args.rt_subframes,
                    repository_root=args.repository_root.resolve(strict=True),
                )
                probe_score, _ = _score_candidates(
                    reference_id=args.reference_view,
                    reference_mask=reference_mask,
                    reference_image=reference_image,
                    registry_path=probe_registry,
                )
                _, probe_part = _project_cluster(
                    registry_path=probe_registry,
                    view_id=args.reference_view,
                    part_ids=[nested_part_id],
                    affine=probe_score["whole_asset_similarity"]["bbox_affine"],
                    target_shape=reference_mask.shape,
                )
                measured = _mask_centroid(probe_part) - baseline_centroid
                probe_registries.append(probe_registry)
                measured_screen.append([float(value) for value in measured])
            response_solution = solve_world_translation_from_render_responses(
                target_screen_translation_px=screen_proposal[
                    "screen_translation_px"
                ],
                probe_world_translations=probe_world,
                measured_screen_translations_px=measured_screen,
            )
            if float(response_solution["world_translation_norm"]) > (
                0.05 * asset_diagonal
            ):
                raise ValueError(
                    "Nested residual correction exceeds the bounded asset scale"
                )
            final_nested_dir = output / "nested_refinement" / "final"
            final_nested_override = (
                final_nested_dir / "assembly_pose_overrides.json"
            )
            _write_object(
                final_nested_override,
                _nested_override_document(
                    parent_subtree=str(winner["assembly_subtree"]),
                    parent_translation=winner["world_translation"],
                    child_subtree=nested_subtree,
                    child_translation=response_solution["world_translation"],
                ),
            )
            nested_registry = _render_candidate(
                python_sh=args.python_sh.resolve(strict=True),
                source_registry=source_registry_path,
                final_view_specs=final_view_specs_path,
                override_path=final_nested_override,
                output_dir=final_nested_dir / "renders",
                resolution=args.resolution,
                rt_subframes=args.rt_subframes,
                repository_root=args.repository_root.resolve(strict=True),
            )
            nested_score, _ = _score_candidates(
                reference_id=args.reference_view,
                reference_mask=reference_mask,
                reference_image=reference_image,
                registry_path=nested_registry,
            )
            nested_registered_foreground = _registered_foreground(
                registry_path=nested_registry,
                view_id=args.reference_view,
                affine=nested_score["whole_asset_similarity"]["bbox_affine"],
                target_shape=reference_mask.shape,
            )
            nested_regional = _regional_overlap_metrics(
                reference_mask=reference_mask,
                registered_foreground=nested_registered_foreground,
                support_bounds_xyxy=regional_support,
            )
            nested_part_residuals = _part_residual_attribution(
                registry_path=nested_registry,
                view_id=args.reference_view,
                reference_mask=reference_mask,
                score=nested_score,
            )
            nested_part_rows = [
                raw
                for raw in nested_part_residuals
                if raw.get("part_id") == nested_part_id
            ]
            if len(nested_part_rows) != 1:
                raise ValueError("Nested residual final render lost its Part binding")
            nested_part_row = nested_part_rows[0]
            parent_fixed = _fixed_geometry_consensus(
                registry_path=winning_registry,
                view_id=args.reference_view,
                excluded_part_ids=[nested_part_id],
                affine=winning_score["whole_asset_similarity"]["bbox_affine"],
                reference_image=reference_image,
                reference_mask=reference_mask,
            )
            nested_fixed = _fixed_geometry_consensus(
                registry_path=nested_registry,
                view_id=args.reference_view,
                excluded_part_ids=[nested_part_id],
                affine=nested_score["whole_asset_similarity"]["bbox_affine"],
                reference_image=reference_image,
                reference_mask=reference_mask,
            )
            nested_residual = _write_residual_audit(
                reference_id=args.reference_view,
                reference_mask=reference_mask,
                foreground=_foreground_mask(
                    _ids_image(
                        _read_object(nested_registry),
                        nested_registry,
                        args.reference_view,
                    ),
                    _read_object(nested_registry),
                ),
                score=nested_score,
                part_residuals=nested_part_residuals,
                output_dir=output / "nested_refinement" / "residual_audit",
            )
            nested_gates = {
                "part_inside_reference_gain": (
                    float(nested_part_row["inside_reference_ratio"])
                    - float(nested["inside_reference_ratio"])
                ),
                "regional_iou_gain": (
                    float(nested_regional["iou"])
                    - float(winner["regional"]["iou"])
                ),
                "global_iou_regression": (
                    float(winner["global"]["projection_iou"])
                    - float(nested_score["projection_iou"])
                ),
                "fixed_geometry_consensus_regression_px": (
                    float(nested_fixed["rigid_consensus_residual_px"])
                    - float(parent_fixed["rigid_consensus_residual_px"])
                ),
                "mismatch_over_union_gain": (
                    float(residual["mismatch_over_union"])
                    - float(nested_residual["mismatch_over_union"])
                ),
            }
            nested_accepted = (
                nested_gates["part_inside_reference_gain"] >= 0.15
                and nested_gates["regional_iou_gain"] >= 0.01
                and nested_gates["global_iou_regression"] <= 0.003
                and nested_gates["fixed_geometry_consensus_regression_px"] <= 0.5
                and nested_gates["mismatch_over_union_gain"] > 0.0
            )
            nested_refinement = {
                "status": "PASS" if nested_accepted else "REJECTED",
                "accepted": nested_accepted,
                "part_id": nested_part_id,
                "assembly_subtree": nested_subtree,
                "baseline_residual": nested,
                "screen_proposal": screen_proposal,
                "probe_world_translations": [
                    [float(value) for value in raw] for raw in probe_world
                ],
                "measured_probe_screen_translations_px": measured_screen,
                "response_solution": response_solution,
                "probe_rendered_registries": [
                    str(path) for path in probe_registries
                ],
                "final_override": str(final_nested_override),
                "final_rendered_registry": str(nested_registry),
                "final_part_residual": nested_part_row,
                "final_regional": nested_regional,
                "final_global": {
                    "projection_iou": nested_score["projection_iou"],
                    "boundary_p95_px": nested_score["boundary_p95_px"],
                    "structure_p75_px": nested_score["structure_p75_px"],
                    "rigid_consensus_residual_px": nested_score.get(
                        "rigid_consensus_residual_px"
                    ),
                    "fixed_geometry_consensus_residual_px": nested_fixed.get(
                        "rigid_consensus_residual_px"
                    ),
                },
                "final_residual_audit": nested_residual,
                "gates": nested_gates,
            }
            if nested_accepted:
                shutil.copy2(final_nested_override, winning_override)
                winning_registry = nested_registry
                winning_score = nested_score
                residual = nested_residual
                winner["rendered_registry"] = str(nested_registry)
                winner["override"] = str(final_nested_override)
                winner["regional"] = nested_regional
                winner["global"] = nested_refinement["final_global"]
                _, final_anchor = _project_cluster(
                    registry_path=nested_registry,
                    view_id=args.reference_view,
                    part_ids=anchor_part_ids,
                    affine=nested_score["whole_asset_similarity"]["bbox_affine"],
                    target_shape=reference_mask.shape,
                )
                winner["local"] = _mask_metrics(target_component, final_anchor)
                winner["registered_cluster"] = str(
                    final_nested_dir / "registered_cluster.png"
                )
                cv2.imwrite(winner["registered_cluster"], final_anchor * 255)
        except (ValueError, RuntimeError) as error:
            nested_refinement = {
                "status": "SKIPPED_FAIL_CLOSED",
                "accepted": False,
                "reason": str(error),
            }
    public_candidates = []
    for raw in candidates:
        public = copy.deepcopy(raw)
        public.pop("score", None)
        public_candidates.append(public)
    result = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "PASS",
        "camera_report": str(camera_report_path),
        "reference_view_id": args.reference_view,
        "detected_assembly_subtree": cluster["assembly_subtree"],
        "detected_outlier_part_ids": cluster["part_ids"],
        "detected_subtree_member_part_ids": anchor_part_ids,
        "candidate_assembly_subtrees": [
            {
                "assembly_subtree": subtree,
                "subtree_member_part_ids": subtree_part_ids,
                "baseline_fixed_geometry_consensus_residual_px": (
                    baseline_fixed_by_subtree[subtree]
                ),
            }
            for subtree, subtree_part_ids in subtree_candidates
        ],
        "seed_world_translation": seed,
        "seed_camera_horizontal_world_translation": horizontal_seed,
        "seed_camera_vertical_world_translation": vertical_seed,
        "baseline": {
            "local": baseline_local,
            "regional": baseline_regional,
            "global": {
                "projection_iou": baseline_score["projection_iou"],
                "boundary_p95_px": baseline_score["boundary_p95_px"],
                "structure_p75_px": baseline_score["structure_p75_px"],
                "rigid_consensus_residual_px": baseline_score.get(
                    "rigid_consensus_residual_px"
                ),
                "fixed_geometry_consensus_residual_px": (
                    baseline_fixed_by_subtree[
                        str(cluster["assembly_subtree"])
                    ]
                ),
            },
            "rendered_registry": str(final_registry_path),
        },
        "candidates": public_candidates,
        "nested_refinement": nested_refinement,
        "winner": {
            "candidate_id": winner["candidate_id"],
            "assembly_subtree": winner["assembly_subtree"],
            "subtree_member_part_ids": winner["subtree_member_part_ids"],
            "world_translation": winner["world_translation"],
            "translation_norm": winner["translation_norm"],
            "local": winner["local"],
            "regional": winner["regional"],
            "global": winner["global"],
            "rendered_registry": winner["rendered_registry"],
            "registered_cluster": winner["registered_cluster"],
            "assembly_pose_overrides": str(winning_override),
            "residual_audit": residual,
        },
        "acceptance": {
            "minimum_local_objective_gain": 0.15,
            "minimum_regional_iou_gain": 0.15,
            "regional_support_bounds_xyxy": regional_support,
            "regional_support_padding_fraction": 0.25,
            "maximum_centroid_error_ratio": 0.6,
            "maximum_global_iou_regression": 0.003,
            "maximum_fixed_geometry_consensus_residual_regression_px": 0.5,
            "moving_assembly_excluded_from_nonregression_consensus": True,
            "maximum_tested_ancestor_levels": 1,
            "maximum_subtree_part_count": 128,
            "maximum_subtree_registry_fraction": 0.25,
            "per_part_warp_applied": False,
            "whole_subtree_rigid_translation_only": True,
            "nested_refinement_translation_only": True,
            "nested_minimum_part_inside_reference_gain": 0.15,
            "nested_minimum_regional_iou_gain": 0.01,
            "nested_maximum_asset_diagonal_translation_fraction": 0.05,
        },
    }
    _write_object(output / "assembly_pose_report.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-view", default="side")
    parser.add_argument(
        "--python-sh",
        type=Path,
        default=(
            Path(os.environ["ISAACSIM_PYTHON_SH"])
            if os.environ.get("ISAACSIM_PYTHON_SH")
            else None
        ),
        help="Isaac Sim Python launcher; defaults to ISAACSIM_PYTHON_SH",
    )
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--rt-subframes", type=int, default=4)
    parser.add_argument(
        "--reuse-candidates-from",
        type=Path,
        help=(
            "Re-score prior real renders only when the sealed report, camera, "
            "subtree, transform, source asset, resolution, and RT inputs match"
        ),
    )
    parser.add_argument(
        "--disable-nested-refinement",
        action="store_true",
        help="Stop after the parent assembly hierarchy optimization",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.python_sh is None:
        raise ValueError("--python-sh or ISAACSIM_PYTHON_SH is required")
    result = optimize(args)
    print(json.dumps(result["winner"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
