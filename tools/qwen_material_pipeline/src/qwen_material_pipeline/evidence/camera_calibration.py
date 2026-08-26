#!/usr/bin/env python3
"""Continuously calibrate whole-asset cameras against confirmed silhouettes.

The optimizer never edits the USD or any Part-ID transform.  It renders nearby
3-D camera hypotheses, fits one bounded 2-D similarity transform for the whole
assembly, and selects the camera that minimizes full-resolution silhouette and
boundary error.  A second, finer neighborhood is rendered around the winner.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from qwen_material_pipeline.core.paths import SOURCE_ROOT
from qwen_material_pipeline.evidence.spatial import (
    DEFAULT_POLICY,
    _part_color,
    _refine_projection,
)


SCHEMA_VERSION = "qwen-whole-asset-camera-calibration/v10"
VIEW_SPEC_SCHEMA_VERSION = "qwen-camera-view-specs/v1"
FINALIST_COUNT = 5
FAST_FINALIST_COUNT = 12
REFINEMENT_SEED_COUNT = 3
RENDER_RETRY_ATTEMPTS = 3
RENDER_RETRY_DELAY_SECONDS = 3.0
MAX_FOCAL_LENGTH_MM = 2000.0
CAMERA_OBJECTIVE_VERSION = "hierarchical_visible_part_alignment/v9"
CAMERA_SELECTION_POLICY_VERSION = (
    "alignment_gate_then_canonical_camera_signature_with_view_fallback/v2"
)
COMPLETE_ALIGNMENT_MINIMUM_IOU = 0.97
COMPLETE_ALIGNMENT_MAXIMUM_BOUNDARY_P95_PX = 3.0
FAST_SEARCH_MODES = ("auto", "disabled", "required")
FAST_VERIFY_MAXIMUM_ABSOLUTE_IOU_DELTA = 0.025
FAST_VERIFY_MAXIMUM_ABSOLUTE_SCORE_DELTA = 0.04
FAST_VERIFY_MAXIMUM_BOUNDARY_DELTA_DIAGONAL_RATIO = 0.012
FAST_VERIFY_OBJECTIVE_EDGE_RANK_FRACTION = 0.75
CAMERA_PHASES = (
    "coarse",
    "lens",
    "fine",
    "perspective",
    "component_pose",
    "perspective_recheck",
    "component_pose_recheck",
    "orthographic",
    "settle",
    "micro",
    "target",
    "lens_micro",
    "nano",
    "target_micro",
    "pico",
    "target_pico",
)


def _read_object(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {resolved}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(contiguous.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _write_object(path: Path, value: Mapping[str, Any]) -> Path:
    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return resolved


def _resolve_path(value: object, owner: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Invalid path in {owner}: {value!r}")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = owner.parent / path
    return path.resolve(strict=True)


def _deterministic_part_id_foreground(
    part_ids: np.ndarray,
    part_colors_bgr: Sequence[np.ndarray],
) -> np.ndarray:
    """Return geometry-only foreground from the stable Part-ID render.

    Camera calibration must be a function of CAD geometry and camera
    parameters only.  The general evidence helper also unions an RGB
    appearance mask so it can recover unlabelled geometry in legacy renders.
    RTX lighting noise makes that fallback unsuitable here: two renders from
    an identical camera can have identical Part-ID images but slightly
    different RGB pixels, which previously changed the fitted silhouette and
    even the selected camera.  Continuous calibration always renders a
    lossless stable-colour Part-ID image, so exact ID membership is both more
    complete and deterministic.
    """

    if part_ids.ndim != 3 or part_ids.shape[2] < 3:
        raise ValueError("Part-ID render must be a three-channel image")
    packed_ids = (
        part_ids[:, :, 0].astype(np.uint32)
        | (part_ids[:, :, 1].astype(np.uint32) << 8)
        | (part_ids[:, :, 2].astype(np.uint32) << 16)
    )
    packed_colors = np.asarray(
        [
            int(color[0])
            | (int(color[1]) << 8)
            | (int(color[2]) << 16)
            for color in part_colors_bgr
        ],
        dtype=np.uint32,
    )
    if not packed_colors.size:
        raise ValueError("Part registry contains no stable Part-ID colours")
    return np.isin(packed_ids, packed_colors).astype(np.uint8) * 255


def _normalize(vector: Sequence[float]) -> np.ndarray:
    output = np.asarray(vector, dtype=np.float64)
    length = float(np.linalg.norm(output))
    if output.shape != (3,) or not np.isfinite(output).all() or length <= 1e-12:
        raise ValueError(f"Invalid non-zero 3-D vector: {vector!r}")
    return output / length


def _angles(direction: Sequence[float]) -> tuple[float, float]:
    x, y, z = _normalize(direction)
    elevation = math.degrees(math.asin(float(np.clip(z, -1.0, 1.0))))
    azimuth = math.degrees(math.atan2(float(x), float(-y))) % 360.0
    return azimuth, elevation


def _direction(azimuth_degrees: float, elevation_degrees: float) -> list[float]:
    azimuth = math.radians(azimuth_degrees)
    elevation = math.radians(float(np.clip(elevation_degrees, -90.0, 90.0)))
    planar = math.cos(elevation)
    return [
        planar * math.sin(azimuth),
        -planar * math.cos(azimuth),
        math.sin(elevation),
    ]


def _camera_up(direction: Sequence[float]) -> list[float]:
    view = _normalize(direction)
    physical_up = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    projected = physical_up - float(np.dot(physical_up, view)) * view
    if float(np.linalg.norm(projected)) <= 1e-6:
        projected = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
        projected -= float(np.dot(projected, view)) * view
    return _normalize(projected).tolist()


def _candidate_specs(
    *,
    reference_id: str,
    seed: Mapping[str, Any],
    phase: str,
) -> dict[str, Any]:
    azimuth, elevation = _angles(seed["analysis_direction"])
    seed_distance = float(seed.get("distance_multiplier", 2.15))
    seed_focal = float(seed.get("focal_length_mm", 45.0))
    seed_target_u = float(seed.get("target_offset_u", 0.0))
    seed_target_v = float(seed.get("target_offset_v", 0.0))
    seed_projection_mode = str(seed.get("projection_mode", "perspective"))
    seed_orthographic_span = float(
        seed.get("orthographic_span_multiplier", 2.0)
    )
    projection_modes = (seed_projection_mode,)
    orthographic_span_multiplier = seed_orthographic_span
    target_u_values = (seed_target_u,)
    target_v_values = (seed_target_v,)
    distance_focal_pairs: tuple[tuple[float, float], ...] | None = None
    if phase == "coarse":
        # Replicator render products are expensive to register in bulk.  A
        # deterministic 3x3x3 lattice spans the same useful neighborhood as a
        # much denser Cartesian grid, then the second pass resolves the local
        # optimum.  Keeping each pass at 27 cameras avoids quadratic Kit
        # annotator startup growth on 24 GB workstations.
        angle_offsets = (-12.0, 0.0, 12.0)
        elevation_offsets = (-10.0, 0.0, 10.0)
        distances = (1.55, 2.15, 3.0)
        focal_lengths = (seed_focal,)
    elif phase == "lens":
        # Distance and focal length jointly control perspective.  Image-space
        # translation/scale are still fitted once for the entire rigid asset,
        # but they cannot remove the depth-dependent perspective change.
        angle_offsets = (0.0,)
        elevation_offsets = (0.0,)
        distances = sorted(
            {
                round(float(np.clip(seed_distance + value, 1.10, 100.0)), 4)
                for value in (-0.60, -0.30, 0.0, 0.30, 0.60)
            }
        )
        focal_lengths = sorted(
            {
                round(
                    float(np.clip(seed_focal * value, 12.0, MAX_FOCAL_LENGTH_MM)),
                    4,
                )
                for value in (0.80, 0.90, 1.0, 1.10, 1.25)
            }
        )
    elif phase == "fine":
        angle_offsets = (-3.0, 0.0, 3.0)
        elevation_offsets = (-3.0, 0.0, 3.0)
        distances = sorted(
            {
                round(float(np.clip(seed_distance + value, 1.10, 100.0)), 4)
                for value in (-0.30, 0.0, 0.30)
            }
        )
        focal_lengths = (seed_focal,)
    elif phase in {"perspective", "perspective_recheck"}:
        # Apparent image scale is resolved by the single whole-image
        # similarity, but depth-dependent parallax is controlled by physical
        # camera distance.  The former local +/-0.60 search could terminate on
        # its upper boundary for weak-perspective reference captures.  Sweep
        # a wide multiplicative range before local convergence.  The recheck
        # deliberately repeats this sweep after the broad component-aware pose
        # update: changing azimuth/elevation changes the assembly depth range,
        # so the perspective optimum found at the old pose is no longer valid.
        angle_offsets = (0.0,)
        elevation_offsets = (0.0,)
        perspective_distances = sorted(
            {
                round(
                    float(np.clip(seed_distance * factor, 1.10, 100.0)),
                    5,
                )
                for factor in (
                    0.60,
                    0.80,
                    1.0,
                    1.50,
                    2.0,
                    3.0,
                    4.0,
                    6.0,
                    8.0,
                    12.0,
                    16.0,
                )
            }
        )
        distance_focal_pairs = tuple(
            (
                distance,
                round(
                    float(
                        np.clip(
                            seed_focal * distance / seed_distance,
                            12.0,
                            MAX_FOCAL_LENGTH_MM,
                        )
                    ),
                    5,
                ),
            )
            for distance in perspective_distances
        )
        distances = ()
        focal_lengths = ()
        projection_modes = ("perspective",)
    elif phase == "orthographic":
        # Some CAD reference images are captured with a true orthographic
        # viewport.  An arbitrarily long perspective lens is only an
        # approximation and leaves depth-dependent Part-ID displacement.
        # Compare both camera models in the same bounded pose lattice.  The
        # perspective incumbent is deliberately retained: an orthographic
        # phase must be model selection, not an unconditional mode switch.
        angle_offsets = (-1.0, 0.0, 1.0)
        elevation_offsets = (-1.0, 0.0, 1.0)
        distances = (seed_distance,)
        focal_lengths = (seed_focal,)
        projection_modes = ("perspective", "orthographic")
    elif phase in {"component_pose", "component_pose_recheck"}:
        # Detached pedals and external boxes can be tens of pixels away while
        # the enclosure already overlaps.  Re-open a bounded rigid-pose
        # neighborhood before local convergence so the large shell cannot
        # trap the camera in the wrong basin.  After perspective is re-solved,
        # use a smaller second pose lattice to complete one alternating
        # pose-perspective-pose optimization cycle.
        if phase == "component_pose":
            angle_offsets = (-12.0, -6.0, 0.0, 6.0, 12.0)
            elevation_offsets = (-12.0, -6.0, 0.0, 6.0, 12.0)
        else:
            angle_offsets = (-6.0, -3.0, 0.0, 3.0, 6.0)
            elevation_offsets = (-6.0, -3.0, 0.0, 3.0, 6.0)
        distances = (seed_distance,)
        focal_lengths = (seed_focal,)
    elif phase == "settle":
        # A fine winner on the distance boundary is not a converged camera.
        # Expand perspective depth while reducing the angular step so tall
        # attachments are not forced into a wrong orthographic compromise.
        angle_offsets = (-1.5, 0.0, 1.5)
        elevation_offsets = (-1.5, 0.0, 1.5)
        distances = sorted(
            {
                round(float(np.clip(seed_distance + value, 1.10, 100.0)), 4)
                for value in (-0.60, 0.0, 0.60)
            }
        )
        focal_lengths = (seed_focal,)
    elif phase == "micro":
        # Resolve the last depth-dependent Part-ID displacement after the
        # coarse pose and lens searches.  This remains a bounded rigid-camera
        # search: no CAD or per-part transform is changed.
        angle_offsets = (-1.0, 0.0, 1.0)
        elevation_offsets = (-1.0, 0.0, 1.0)
        distances = sorted(
            {
                round(float(np.clip(seed_distance + value, 1.10, 100.0)), 4)
                for value in (-0.15, 0.0, 0.15)
            }
        )
        focal_lengths = (seed_focal,)
    elif phase == "target":
        # Complete the rigid camera extrinsics: the camera optical axis is not
        # generally constrained to pass through the CAD bounding-box center.
        # These offsets alter only the virtual look-at target, never the USD.
        angle_offsets = (0.0,)
        elevation_offsets = (0.0,)
        distances = (seed_distance,)
        focal_lengths = (seed_focal,)
        target_u_values = tuple(
            sorted(
                {
                    round(float(np.clip(seed_target_u + value, -0.5, 0.5)), 5)
                    for value in (-0.08, -0.04, 0.0, 0.04, 0.08)
                }
            )
        )
        target_v_values = tuple(
            sorted(
                {
                    round(float(np.clip(seed_target_v + value, -0.5, 0.5)), 5)
                    for value in (-0.08, -0.04, 0.0, 0.04, 0.08)
                }
            )
        )
    elif phase == "lens_micro":
        # Focal length and distance are coupled by apparent object size but
        # not by depth-dependent perspective.  Resolve that ratio after the
        # pose has settled; the whole-image similarity fit may absorb crop
        # scale, but it cannot hide the remaining parallax between near and
        # far Part IDs.
        angle_offsets = (0.0,)
        elevation_offsets = (0.0,)
        distances = sorted(
            {
                round(float(np.clip(seed_distance + value, 1.10, 100.0)), 5)
                for value in (-0.08, -0.04, 0.0, 0.04, 0.08)
            }
        )
        focal_lengths = sorted(
            {
                round(
                    float(np.clip(seed_focal * value, 12.0, MAX_FOCAL_LENGTH_MM)),
                    5,
                )
                for value in (0.98, 0.99, 1.0, 1.01, 1.02)
            }
        )
    elif phase == "nano":
        angle_offsets = (-0.25, 0.0, 0.25)
        elevation_offsets = (-0.25, 0.0, 0.25)
        distances = sorted(
            {
                round(float(np.clip(seed_distance + value, 1.10, 100.0)), 5)
                for value in (-0.04, 0.0, 0.04)
            }
        )
        focal_lengths = (seed_focal,)
    elif phase == "target_micro":
        angle_offsets = (0.0,)
        elevation_offsets = (0.0,)
        distances = (seed_distance,)
        focal_lengths = (seed_focal,)
        target_u_values = tuple(
            sorted(
                {
                    round(float(np.clip(seed_target_u + value, -0.5, 0.5)), 5)
                    for value in (-0.01, 0.0, 0.01)
                }
            )
        )
        target_v_values = tuple(
            sorted(
                {
                    round(float(np.clip(seed_target_v + value, -0.5, 0.5)), 5)
                    for value in (-0.01, 0.0, 0.01)
                }
            )
        )
    elif phase == "pico":
        # The production result is selected only after this sub-tenth-degree
        # stage.  Keeping it last is essential: a later coarse "settle" pass
        # previously overwrote the more accurate micro winner.
        angle_offsets = (-0.05, 0.0, 0.05)
        elevation_offsets = (-0.05, 0.0, 0.05)
        distances = sorted(
            {
                round(float(np.clip(seed_distance + value, 1.10, 100.0)), 5)
                for value in (-0.01, 0.0, 0.01)
            }
        )
        focal_lengths = (seed_focal,)
    elif phase == "target_pico":
        angle_offsets = (0.0,)
        elevation_offsets = (0.0,)
        distances = (seed_distance,)
        focal_lengths = (seed_focal,)
        target_u_values = tuple(
            sorted(
                {
                    round(float(np.clip(seed_target_u + value, -0.5, 0.5)), 5)
                    for value in (-0.003, 0.0, 0.003)
                }
            )
        )
        target_v_values = tuple(
            sorted(
                {
                    round(float(np.clip(seed_target_v + value, -0.5, 0.5)), 5)
                    for value in (-0.003, 0.0, 0.003)
                }
            )
        )
    else:
        raise ValueError(f"Unknown camera calibration phase: {phase}")
    camera_pairs = (
        distance_focal_pairs
        if distance_focal_pairs is not None
        else tuple(
            (distance, focal_length)
            for distance in distances
            for focal_length in focal_lengths
        )
    )
    views: list[dict[str, Any]] = []
    index = 0
    for azimuth_offset in angle_offsets:
        for elevation_offset in elevation_offsets:
            for distance, focal_length in camera_pairs:
                for projection_mode in projection_modes:
                    for target_u in target_u_values:
                        for target_v in target_v_values:
                            exact_seed = (
                                azimuth_offset == 0.0
                                and elevation_offset == 0.0
                                and math.isclose(
                                    distance, seed_distance, abs_tol=1e-6
                                )
                                and math.isclose(
                                    focal_length, seed_focal, abs_tol=1e-6
                                )
                                and math.isclose(
                                    target_u, seed_target_u, abs_tol=1e-6
                                )
                                and math.isclose(
                                    target_v, seed_target_v, abs_tol=1e-6
                                )
                                and projection_mode == seed_projection_mode
                            )
                            direction = (
                                _normalize(seed["analysis_direction"]).tolist()
                                if exact_seed
                                else _direction(
                                    azimuth + azimuth_offset,
                                    elevation + elevation_offset,
                                )
                            )
                            up_axis = (
                                _normalize(seed["analysis_up_axis"]).tolist()
                                if exact_seed
                                and seed.get("analysis_up_axis") is not None
                                else _camera_up(direction)
                            )
                            view_id = f"cal_{reference_id}_{phase}_{index:03d}"
                            views.append(
                                {
                                    "view_id": view_id,
                                    "analysis_direction": direction,
                                    "analysis_up_axis": up_axis,
                                    "focal_length_mm": focal_length,
                                    "distance_multiplier": distance,
                                    "target_offset_u": target_u,
                                    "target_offset_v": target_v,
                                    "projection_mode": projection_mode,
                                    "orthographic_span_multiplier": (
                                        orthographic_span_multiplier
                                    ),
                                    "calibration": {
                                        "reference_view_id": reference_id,
                                        "phase": phase,
                                        "azimuth_degrees": round(
                                            (azimuth + azimuth_offset) % 360.0,
                                            6,
                                        ),
                                        "elevation_degrees": round(
                                            float(
                                                np.clip(
                                                    elevation + elevation_offset,
                                                    -90.0,
                                                    90.0,
                                                )
                                            ),
                                            6,
                                        ),
                                        "distance_multiplier": distance,
                                        "focal_length_mm": focal_length,
                                        "target_offset_u": target_u,
                                        "target_offset_v": target_v,
                                        "projection_mode": projection_mode,
                                        "orthographic_span_multiplier": (
                                            orthographic_span_multiplier
                                        ),
                                    },
                                }
                            )
                            index += 1
    return {"schema_version": VIEW_SPEC_SCHEMA_VERSION, "views": views}


def _reference_masks(
    manifest_path: Path,
) -> dict[str, tuple[np.ndarray, dict[str, Any]]]:
    document = _read_object(manifest_path)
    output: dict[str, tuple[np.ndarray, dict[str, Any]]] = {}
    for raw in document.get("source_views", []):
        if not isinstance(raw, dict) or not isinstance(raw.get("id"), str):
            continue
        confirmed = raw.get("confirmed_mask")
        mask_value = raw.get("palette_mask")
        if mask_value is None and isinstance(confirmed, Mapping):
            mask_value = confirmed.get("path")
        mask_path = _resolve_path(mask_value, manifest_path)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None or not np.any(mask > 0):
            raise ValueError(f"Reference foreground mask is empty: {mask_path}")
        output[raw["id"]] = (mask, raw)
    return output


def _reference_image(
    raw: Mapping[str, Any],
    manifest_path: Path,
    expected_shape: tuple[int, int],
) -> np.ndarray:
    image = cv2.imread(
        str(_resolve_path(raw.get("image"), manifest_path)),
        cv2.IMREAD_COLOR,
    )
    if image is None or image.shape[:2] != expected_shape:
        raise ValueError(
            f"Reference image dimensions do not match foreground mask: "
            f"{raw.get('id')}"
        )
    return image


def _camera_signature(item: Mapping[str, Any]) -> tuple[Any, ...]:
    """Return a path- and generation-order-independent camera identity."""

    direction = item.get("analysis_direction")
    up_axis = item.get("analysis_up_axis")
    if not isinstance(direction, Sequence) or isinstance(direction, (str, bytes)):
        direction = ()
    if not isinstance(up_axis, Sequence) or isinstance(up_axis, (str, bytes)):
        up_axis = ()
    return (
        str(item.get("projection_mode", "perspective")),
        tuple(round(float(value), 10) for value in direction),
        tuple(round(float(value), 10) for value in up_axis),
        round(float(item.get("focal_length_mm", 45.0)), 8),
        round(float(item.get("distance_multiplier", 2.15)), 8),
        round(float(item.get("target_offset_u", 0.0)), 8),
        round(float(item.get("target_offset_v", 0.0)), 8),
        round(float(item.get("orthographic_span_multiplier", 2.0)), 8),
    )


def _registry_geometry_and_pose_contract(registry_path: Path) -> dict[str, Any]:
    """Fingerprint only camera-relevant CAD geometry and rendered Part IDs.

    Absolute output paths, RTX RGB images, materials and timestamps are
    intentionally excluded.  Camera calibration consumes stable CAD geometry,
    camera metadata and lossless Part-ID pixels; identical inputs in another
    run directory must therefore produce the same contract.
    """

    registry = _read_object(registry_path)
    raw_parts = registry.get("parts")
    render_set = registry.get("render_set")
    if not isinstance(raw_parts, list) or not isinstance(render_set, Mapping):
        raise ValueError("Camera source registry lacks parts or render_set")
    parts: list[dict[str, Any]] = []
    for raw in raw_parts:
        if not isinstance(raw, Mapping) or not isinstance(raw.get("part_id"), str):
            raise ValueError("Camera source registry has a malformed Part-ID row")
        parts.append(
            {
                "part_id": raw["part_id"],
                "prim_path": raw.get("prim_path"),
                "point_count": raw.get("point_count"),
                "face_count": raw.get("face_count"),
                "geometry_content_sha256": raw.get("geometry_content_sha256"),
                "world_bbox": raw.get("world_bbox"),
                "source_subset_layout_sha256": raw.get(
                    "source_subset_layout_sha256"
                ),
            }
        )
    parts.sort(key=lambda row: str(row["part_id"]))

    views: list[dict[str, Any]] = []
    raw_views = render_set.get("views")
    if not isinstance(raw_views, list) or not raw_views:
        raise ValueError("Camera source registry has no rendered pose bank")
    for raw in raw_views:
        if not isinstance(raw, Mapping) or not isinstance(raw.get("view_id"), str):
            raise ValueError("Camera source registry has a malformed view row")
        ids_path = _resolve_path(
            raw.get("part_ids_raw") or raw.get("part_ids"),
            registry_path,
        )
        ids = cv2.imread(str(ids_path), cv2.IMREAD_COLOR)
        if ids is None:
            raise ValueError(f"Unable to read source Part-ID render: {ids_path}")
        views.append(
            {
                "view_id": raw["view_id"],
                "analysis_direction": raw.get("analysis_direction"),
                "analysis_up_axis": raw.get("analysis_camera_up_axis")
                or raw.get("camera_up_axis"),
                "focal_length_mm": raw.get("focal_length_mm"),
                "distance_multiplier": raw.get("camera_distance_multiplier"),
                "target_offset_u": raw.get("camera_target_offset_u", 0.0),
                "target_offset_v": raw.get("camera_target_offset_v", 0.0),
                "projection_mode": raw.get("camera_projection_mode", "perspective"),
                "orthographic_span_multiplier": raw.get(
                    "camera_orthographic_span_multiplier", 2.0
                ),
                "part_ids_pixel_sha256": _array_sha256(ids),
            }
        )
    views.sort(key=lambda row: str(row["view_id"]))
    return {
        "schema_version": "qwen-camera-registry-input/v1",
        "registry_schema_version": registry.get("schema_version"),
        "part_count": len(parts),
        "parts_sha256": _canonical_sha256(parts),
        "render_contract": {
            "resolution": render_set.get("resolution"),
            "analysis_up_axis": render_set.get("analysis_up_axis"),
            "analysis_front_axis": render_set.get("analysis_front_axis"),
            "requested_view_tokens": render_set.get("requested_view_tokens"),
            "analysis_basis_world": render_set.get("analysis_basis_world"),
        },
        "views_sha256": _canonical_sha256(views),
        "view_count": len(views),
    }


def _reference_evidence_contract(
    references: Mapping[str, tuple[np.ndarray, Mapping[str, Any]]],
    reference_images: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    rows = []
    for reference_id in sorted(references):
        mask = references[reference_id][0]
        image = reference_images[reference_id]
        rows.append(
            {
                "reference_view_id": reference_id,
                "mask_pixel_sha256": _array_sha256(mask),
                "image_pixel_sha256": _array_sha256(image),
                "mask_shape": list(mask.shape),
                "image_shape": list(image.shape),
            }
        )
    return {
        "schema_version": "qwen-camera-reference-input/v1",
        "reference_count": len(rows),
        "references_sha256": _canonical_sha256(rows),
    }


def _camera_solution_contract(specs: Mapping[str, Any]) -> dict[str, Any]:
    raw_views = specs.get("views")
    if not isinstance(raw_views, list) or not raw_views:
        raise ValueError("Final camera specs have no views")
    views = []
    for raw in raw_views:
        if not isinstance(raw, Mapping) or not isinstance(raw.get("view_id"), str):
            raise ValueError("Final camera specs contain a malformed view")
        signature = _camera_signature(raw)
        views.append(
            {
                "reference_view_id": raw["view_id"],
                "projection_mode": signature[0],
                "analysis_direction": list(signature[1]),
                "analysis_up_axis": list(signature[2]),
                "focal_length_mm": signature[3],
                "distance_multiplier": signature[4],
                "target_offset_u": signature[5],
                "target_offset_v": signature[6],
                "orthographic_span_multiplier": signature[7],
            }
        )
    views.sort(key=lambda row: str(row["reference_view_id"]))
    return {
        "schema_version": "qwen-camera-solution/v1",
        "views": views,
    }


def _boundary_metrics(
    target: np.ndarray,
    registered: np.ndarray,
) -> dict[str, float]:
    target_binary = (target > 0).astype(np.uint8)
    registered_binary = (registered > 0).astype(np.uint8)
    target_edge = cv2.morphologyEx(
        target_binary, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)
    )
    registered_edge = cv2.morphologyEx(
        registered_binary, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)
    )
    target_distance = cv2.distanceTransform(1 - target_edge, cv2.DIST_L2, 3)
    registered_distance = cv2.distanceTransform(1 - registered_edge, cv2.DIST_L2, 3)
    distances = np.concatenate(
        [
            target_distance[registered_edge > 0],
            registered_distance[target_edge > 0],
        ]
    )
    if not distances.size:
        return {"boundary_mean_px": float("inf"), "boundary_p95_px": float("inf")}
    return {
        "boundary_mean_px": float(np.mean(distances)),
        "boundary_p95_px": float(np.percentile(distances, 95)),
    }


def _reference_structure_edges(
    image: np.ndarray,
    foreground: np.ndarray,
) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0.0)
    median = float(np.median(gray[foreground > 0]))
    lower = int(max(20.0, 0.55 * median))
    upper = int(max(lower + 20, min(255.0, 1.45 * median)))
    edges = cv2.Canny(gray, lower, upper)
    # Human-confirmed SAM masks can legitimately omit black CAD parts against
    # a black viewport.  Keep a bounded neighborhood around the confirmed
    # object so edges from adjacent dark rails/tubes remain available without
    # admitting the full viewport grid.
    support = cv2.dilate(
        (foreground > 0).astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)),
        iterations=1,
    )
    return np.where(support > 0, edges, 0).astype(np.uint8)


def _part_balanced_structure_metrics(
    *,
    ids: np.ndarray,
    parts: Sequence[Mapping[str, Any]],
    affine: np.ndarray,
    reference_image: np.ndarray,
    reference_mask: np.ndarray,
) -> dict[str, Any]:
    """Score internal CAD boundaries without letting the large shell dominate."""

    reference_edges = _reference_structure_edges(
        reference_image,
        reference_mask,
    )
    if not np.any(reference_edges > 0):
        return {
            "structure_part_count": 0,
            "structure_median_px": None,
            "structure_p75_px": None,
            "structure_score": 0.0,
            "structure_size_strata": {},
        }
    reference_distance = cv2.distanceTransform(
        (reference_edges == 0).astype(np.uint8),
        cv2.DIST_L2,
        3,
    )
    height, width = reference_mask.shape
    per_part: list[dict[str, float | int]] = []
    packed_ids = (
        ids[:, :, 0].astype(np.uint32)
        | (ids[:, :, 1].astype(np.uint32) << 8)
        | (ids[:, :, 2].astype(np.uint32) << 16)
    )
    visible_values, visible_counts = np.unique(
        packed_ids,
        return_counts=True,
    )
    count_by_value = {
        int(value): int(count) for value, count in zip(visible_values, visible_counts)
    }
    part_values: list[tuple[int, int]] = []
    for part in parts:
        if not isinstance(part, Mapping) or not isinstance(part.get("part_id"), str):
            continue
        red, green, blue = _part_color(str(part["part_id"]))
        packed = int(blue) | (int(green) << 8) | (int(red) << 16)
        if count_by_value.get(packed, 0) >= 24:
            part_values.append((packed, count_by_value[packed]))
    # Equal-capacity logarithmic size strata prevent the enclosure from
    # dominating the camera while still keeping subpixel fasteners out.
    strata: dict[str, list[tuple[int, int]]] = {
        "small": [],
        "medium": [],
        "large": [],
    }
    for packed, pixel_count in part_values:
        if pixel_count < 96:
            strata["small"].append((packed, pixel_count))
        elif pixel_count < 768:
            strata["medium"].append((packed, pixel_count))
        else:
            strata["large"].append((packed, pixel_count))
    selected_values: list[tuple[str, int, int]] = []
    for stratum, values in strata.items():
        values.sort(key=lambda item: (-item[1], item[0]))
        selected_values.extend(
            (stratum, packed, pixel_count) for packed, pixel_count in values[:16]
        )
    foreground_support = cv2.dilate(
        (reference_mask > 0).astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)),
        iterations=1,
    )
    for stratum, packed, pixel_count in selected_values:
        raw = (packed_ids == packed).astype(np.uint8)
        if pixel_count < 24:
            continue
        projected = cv2.warpAffine(
            raw,
            affine,
            (width, height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        projected_pixels = int(np.count_nonzero(projected))
        supported_pixels = int(
            np.count_nonzero(np.logical_and(projected > 0, foreground_support > 0))
        )
        # Parts far outside the human-confirmed object are usually invisible
        # black-on-black geometry.  They must not pull the camera toward an
        # edge that is not observable in the reference image.
        if projected_pixels <= 0 or supported_pixels / projected_pixels < 0.10:
            continue
        boundary = cv2.morphologyEx(
            projected,
            cv2.MORPH_GRADIENT,
            np.ones((3, 3), dtype=np.uint8),
        )
        distances = reference_distance[boundary > 0]
        if len(distances) < 8:
            continue
        per_part.append(
            {
                "stratum": stratum,
                "median_px": float(np.median(distances)),
            }
        )
    if not per_part:
        return {
            "structure_part_count": 0,
            "structure_median_px": None,
            "structure_p75_px": None,
            "structure_score": 0.0,
            "structure_size_strata": {},
        }
    distances_by_stratum = {
        stratum: [
            float(item["median_px"]) for item in per_part if item["stratum"] == stratum
        ]
        for stratum in strata
    }
    populated_strata = {
        stratum: values for stratum, values in distances_by_stratum.items() if values
    }
    flattened = [value for values in populated_strata.values() for value in values]
    median_px = float(np.median(flattened))
    p75_px = float(np.percentile(flattened, 75))
    diagonal = math.hypot(width, height)
    decay = max(3.0, 0.015 * diagonal)
    stratum_scores = {
        stratum: math.exp(-float(np.percentile(values, 75)) / decay)
        for stratum, values in populated_strata.items()
    }
    macro_score = float(np.mean(list(stratum_scores.values())))
    return {
        "structure_part_count": len(per_part),
        "structure_median_px": median_px,
        "structure_p75_px": p75_px,
        "structure_score": macro_score,
        "structure_size_strata": {
            stratum: {
                "part_count": len(distances_by_stratum[stratum]),
                "p75_px": (
                    float(np.percentile(distances_by_stratum[stratum], 75))
                    if distances_by_stratum[stratum]
                    else None
                ),
                "score": stratum_scores.get(stratum),
            }
            for stratum in strata
        },
    }


def _silhouette_coverage_metrics(
    reference_mask: np.ndarray,
    registered_mask: np.ndarray,
) -> dict[str, float]:
    """Score a possibly incomplete human mask without rewarding overgrowth."""

    target = reference_mask > 0
    rendered = registered_mask > 0
    intersection = int(np.count_nonzero(np.logical_and(target, rendered)))
    target_pixels = int(np.count_nonzero(target))
    rendered_pixels = int(np.count_nonzero(rendered))
    target_recall = intersection / max(1, target_pixels)
    rendered_precision = intersection / max(1, rendered_pixels)
    f_score = (
        2.0
        * target_recall
        * rendered_precision
        / max(1e-12, target_recall + rendered_precision)
    )
    return {
        "target_recall": float(target_recall),
        "rendered_precision": float(rendered_precision),
        "silhouette_f_score": float(f_score),
    }


def _component_balanced_reference_metrics(
    reference_mask: np.ndarray,
    registered_mask: np.ndarray,
) -> dict[str, Any]:
    """Give each detached reference component equal registration weight."""

    target = (reference_mask > 0).astype(np.uint8)
    # A one- or two-pixel cable can connect a detached pedal or control box to
    # the main enclosure in the human mask.  Break only those narrow bridges
    # for scoring; the confirmed source mask itself remains untouched.
    component_source = cv2.morphologyEx(
        target,
        cv2.MORPH_OPEN,
        np.ones((5, 5), dtype=np.uint8),
    )
    rendered = registered_mask > 0
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        component_source,
        connectivity=8,
    )
    recalls: list[float] = []
    for component_id in range(1, component_count):
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        if area < 12:
            continue
        component = labels == component_id
        overlap = int(np.count_nonzero(np.logical_and(component, rendered)))
        recalls.append(float(overlap / area))
    if not recalls:
        return {
            "reference_component_count": 0,
            "reference_component_macro_recall": 0.0,
            "reference_component_min_recall": 0.0,
        }
    return {
        "reference_component_count": len(recalls),
        "reference_component_macro_recall": float(np.mean(recalls)),
        "reference_component_min_recall": float(np.min(recalls)),
    }


def _spatial_balanced_reference_metrics(
    reference_mask: np.ndarray,
    registered_mask: np.ndarray,
) -> dict[str, Any]:
    """Prevent a large central shell from hiding local alignment failures.

    Connected-component balancing cannot isolate an attachment joined to the
    enclosure by a thin tube or cable.  Split the confirmed foreground's own
    bounding box into a fixed 3x3 spatial grid and give every populated cell
    equal weight.  This is semantic-free and applies to arbitrary assemblies;
    it never moves an individual CAD part.
    """

    target = reference_mask > 0
    rendered = registered_mask > 0
    ys, xs = np.nonzero(target)
    if not len(xs):
        return {
            "reference_spatial_cell_count": 0,
            "reference_spatial_macro_recall": 0.0,
            "reference_spatial_min_recall": 0.0,
            "reference_spatial_cells": [],
        }
    left, right = int(xs.min()), int(xs.max()) + 1
    top, bottom = int(ys.min()), int(ys.max()) + 1
    x_edges = np.rint(np.linspace(left, right, 4)).astype(int)
    y_edges = np.rint(np.linspace(top, bottom, 4)).astype(int)
    minimum_pixels = max(12, int(round(float(np.count_nonzero(target)) * 0.001)))
    cells: list[dict[str, Any]] = []
    for row in range(3):
        for column in range(3):
            x0, x1 = int(x_edges[column]), int(x_edges[column + 1])
            y0, y1 = int(y_edges[row]), int(y_edges[row + 1])
            cell_target = target[y0:y1, x0:x1]
            target_pixels = int(np.count_nonzero(cell_target))
            if target_pixels < minimum_pixels:
                continue
            overlap = int(
                np.count_nonzero(
                    np.logical_and(
                        cell_target,
                        rendered[y0:y1, x0:x1],
                    )
                )
            )
            cells.append(
                {
                    "row": row,
                    "column": column,
                    "target_pixels": target_pixels,
                    "recall": float(overlap / target_pixels),
                }
            )
    recalls = [float(cell["recall"]) for cell in cells]
    if not recalls:
        return {
            "reference_spatial_cell_count": 0,
            "reference_spatial_macro_recall": 0.0,
            "reference_spatial_min_recall": 0.0,
            "reference_spatial_cells": [],
        }
    return {
        "reference_spatial_cell_count": len(cells),
        "reference_spatial_macro_recall": float(np.mean(recalls)),
        "reference_spatial_min_recall": float(np.min(recalls)),
        "reference_spatial_cells": cells,
    }


def _residual_components(mask: np.ndarray) -> list[dict[str, Any]]:
    binary = (mask > 0).astype(np.uint8)
    component_count, _, stats, centroids = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )
    image_area = max(1, int(binary.size))
    output: list[dict[str, Any]] = []
    for component_id in range(1, component_count):
        x, y, width, height, area = (int(value) for value in stats[component_id])
        output.append(
            {
                "area_pixels": area,
                "area_ratio": round(area / image_area, 8),
                "bbox_xywh": [x, y, width, height],
                "centroid_xy": [
                    round(float(centroids[component_id][0]), 3),
                    round(float(centroids[component_id][1]), 3),
                ],
            }
        )
    output.sort(key=lambda item: -int(item["area_pixels"]))
    return output


def _write_residual_audit(
    *,
    reference_id: str,
    reference_mask: np.ndarray,
    foreground: np.ndarray,
    score: Mapping[str, Any],
    part_residuals: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    matrix = np.asarray(
        score["whole_asset_similarity"]["bbox_affine"],
        dtype=np.float32,
    )
    registered = cv2.warpAffine(
        foreground,
        matrix,
        (reference_mask.shape[1], reference_mask.shape[0]),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    target = reference_mask > 0
    cad = registered > 0
    target_only = np.logical_and(target, ~cad).astype(np.uint8) * 255
    cad_only = np.logical_and(cad, ~target).astype(np.uint8) * 255
    overlap = np.logical_and(target, cad)
    audit = np.zeros((*reference_mask.shape, 3), dtype=np.uint8)
    audit[overlap] = (80, 190, 80)
    audit[target_only > 0] = (30, 30, 235)
    audit[cad_only > 0] = (235, 160, 20)
    cv2.putText(
        audit,
        "green=overlap  red=reference-only  blue=CAD-only",
        (16, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    overlay_path = output_dir / f"{reference_id}_residual_overlay.png"
    target_only_path = output_dir / f"{reference_id}_reference_only.png"
    cad_only_path = output_dir / f"{reference_id}_cad_only.png"
    cv2.imwrite(str(overlay_path), audit)
    cv2.imwrite(str(target_only_path), target_only)
    cv2.imwrite(str(cad_only_path), cad_only)
    union_pixels = int(np.count_nonzero(np.logical_or(target, cad)))
    mismatch_pixels = int(np.count_nonzero(target_only) + np.count_nonzero(cad_only))
    return {
        "overlap_pixels": int(np.count_nonzero(overlap)),
        "reference_only_pixels": int(np.count_nonzero(target_only)),
        "cad_only_pixels": int(np.count_nonzero(cad_only)),
        "mismatch_pixels": mismatch_pixels,
        "mismatch_over_union": round(
            mismatch_pixels / max(1, union_pixels),
            8,
        ),
        "reference_only_components": _residual_components(target_only),
        "cad_only_components": _residual_components(cad_only),
        "cad_part_residual_attribution": list(part_residuals),
        "residual_overlay": str(overlay_path),
        "reference_only_mask": str(target_only_path),
        "cad_only_mask": str(cad_only_path),
    }


def _score_candidate_ids(
    *,
    reference_mask: np.ndarray,
    reference_image: np.ndarray,
    ids: np.ndarray,
    parts: Sequence[Mapping[str, Any]],
    colors: Sequence[np.ndarray],
    view: Mapping[str, Any],
) -> dict[str, Any]:
    """Score one stable Part-ID image with the authoritative camera objective."""

    calibration = view.get("camera_calibration")
    if not isinstance(calibration, Mapping):
        raise ValueError(f"Camera candidate {view.get('view_id')!r} has no calibration")
    if ids.ndim != 3 or ids.shape[2] < 3:
        raise ValueError(f"Camera candidate {view.get('view_id')!r} has invalid Part IDs")
    foreground = _deterministic_part_id_foreground(ids, colors)
    projection = _refine_projection(
        reference_mask,
        foreground,
        DEFAULT_POLICY,
    )
    matrix = np.asarray(projection["bbox_affine"], dtype=np.float32)
    registered = cv2.warpAffine(
        foreground,
        matrix,
        (reference_mask.shape[1], reference_mask.shape[0]),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    boundary = _boundary_metrics(reference_mask, registered)
    coverage = _silhouette_coverage_metrics(reference_mask, registered)
    component_coverage = _component_balanced_reference_metrics(
        reference_mask,
        registered,
    )
    spatial_coverage = _spatial_balanced_reference_metrics(
        reference_mask,
        registered,
    )
    structure = _part_balanced_structure_metrics(
        ids=ids,
        parts=parts,
        affine=matrix,
        reference_image=reference_image,
        reference_mask=reference_mask,
    )
    iou = float(projection["projection_iou"])
    diagonal = math.hypot(reference_mask.shape[1], reference_mask.shape[0])
    boundary_decay = max(3.0, 0.02 * diagonal)
    boundary_score = math.exp(-float(boundary["boundary_p95_px"]) / boundary_decay)
    # Hierarchical objective: the confirmed foreground first has to be
    # covered, then the outer boundary and equal-weight size strata select
    # the camera.  IoU remains bounded evidence but can no longer let one
    # large enclosure hide small-part displacement.
    score = (
        0.12 * iou
        + 0.12 * float(coverage["target_recall"])
        + 0.08 * float(coverage["rendered_precision"])
        + 0.13 * boundary_score
        + 0.18 * float(structure["structure_score"])
        + 0.10 * float(component_coverage["reference_component_macro_recall"])
        + 0.05 * float(component_coverage["reference_component_min_recall"])
        + 0.14 * float(spatial_coverage["reference_spatial_macro_recall"])
        + 0.08 * float(spatial_coverage["reference_spatial_min_recall"])
    )
    complete_alignment_candidate = (
        iou >= COMPLETE_ALIGNMENT_MINIMUM_IOU
        and float(boundary["boundary_p95_px"])
        <= COMPLETE_ALIGNMENT_MAXIMUM_BOUNDARY_P95_PX
    )
    return {
        "view_id": view["view_id"],
        "objective_version": CAMERA_OBJECTIVE_VERSION,
        "score": round(score, 8),
        "projection_iou": round(iou, 8),
        **{key: round(float(value), 8) for key, value in coverage.items()},
        **{
            key: (round(float(value), 8) if isinstance(value, float) else value)
            for key, value in component_coverage.items()
        },
        **{
            key: (round(float(value), 8) if isinstance(value, float) else value)
            for key, value in spatial_coverage.items()
        },
        **{key: round(value, 8) for key, value in boundary.items()},
        "boundary_score": round(boundary_score, 8),
        "complete_alignment_candidate": complete_alignment_candidate,
        **{
            key: (round(float(value), 8) if isinstance(value, float) else value)
            for key, value in structure.items()
        },
        "analysis_direction": view.get("analysis_direction"),
        "analysis_up_axis": (
            view.get("analysis_camera_up_axis") or view.get("camera_up_axis")
        ),
        "focal_length_mm": view.get("focal_length_mm"),
        "distance_multiplier": view.get("camera_distance_multiplier"),
        "target_offset_u": view.get("camera_target_offset_u", 0.0),
        "target_offset_v": view.get("camera_target_offset_v", 0.0),
        "projection_mode": view.get("camera_projection_mode", "perspective"),
        "orthographic_span_multiplier": view.get(
            "camera_orthographic_span_multiplier", 2.0
        ),
        "calibration": dict(calibration),
        "render_backend": view.get("render_backend", "isaac_rtx_part_id"),
        "whole_asset_similarity": projection,
    }


def _score_candidates(
    *,
    reference_id: str,
    reference_mask: np.ndarray,
    reference_image: np.ndarray,
    registry_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    registry = _read_object(registry_path)
    parts = registry.get("parts", [])
    colors = [
        np.asarray(
            (
                _part_color(str(part["part_id"]))[2],
                _part_color(str(part["part_id"]))[1],
                _part_color(str(part["part_id"]))[0],
            ),
            dtype=np.uint8,
        )
        for part in parts
        if isinstance(part, dict) and isinstance(part.get("part_id"), str)
    ]
    records: list[dict[str, Any]] = []
    for view in registry.get("render_set", {}).get("views", []):
        calibration = view.get("camera_calibration")
        if (
            not isinstance(calibration, dict)
            or calibration.get("reference_view_id") != reference_id
        ):
            continue
        ids_path = _resolve_path(
            view.get("part_ids_raw") or view.get("part_ids"),
            registry_path,
        )
        ids = cv2.imread(str(ids_path), cv2.IMREAD_COLOR)
        if ids is None:
            raise ValueError(f"Unable to read calibration render {view.get('view_id')}")
        records.append(
            _score_candidate_ids(
                reference_mask=reference_mask,
                reference_image=reference_image,
                ids=ids,
                parts=parts,
                colors=colors,
                view=view,
            )
        )
    if not records:
        raise ValueError(f"No calibration candidates found for {reference_id}")
    records.sort(key=_alignment_candidate_sort_key)
    return records[0], records


def _score_fast_candidates(
    *,
    reference_id: str,
    reference_mask: np.ndarray,
    reference_image: np.ndarray,
    specs: Mapping[str, Any],
    rasterizer: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Render and score a phase without starting Isaac.

    The adapter intentionally emits the same camera metadata keys as the USD
    renderer.  That keeps candidate ordering and the later full-resolution
    Isaac verification on one scoring implementation.
    """

    views = specs.get("views")
    if not isinstance(views, list) or not views:
        raise ValueError("Fast camera search requires non-empty candidate specs")
    records: list[dict[str, Any]] = []
    for raw in views:
        if not isinstance(raw, Mapping):
            raise ValueError("Fast camera candidate spec must be an object")
        calibration = raw.get("calibration")
        if (
            not isinstance(calibration, Mapping)
            or calibration.get("reference_view_id") != reference_id
        ):
            raise ValueError(
                f"Fast camera candidate belongs to another reference: "
                f"{raw.get('view_id')!r}"
            )
        ids = rasterizer.render_part_ids(raw)
        view = {
            "view_id": raw.get("view_id"),
            "analysis_direction": raw.get("analysis_direction"),
            "analysis_camera_up_axis": raw.get("analysis_up_axis"),
            "focal_length_mm": raw.get("focal_length_mm"),
            "camera_distance_multiplier": raw.get("distance_multiplier"),
            "camera_target_offset_u": raw.get("target_offset_u", 0.0),
            "camera_target_offset_v": raw.get("target_offset_v", 0.0),
            "camera_projection_mode": raw.get("projection_mode", "perspective"),
            "camera_orthographic_span_multiplier": raw.get(
                "orthographic_span_multiplier", 2.0
            ),
            "camera_calibration": dict(calibration),
            "render_backend": rasterizer.audit["backend"],
        }
        records.append(
            _score_candidate_ids(
                reference_mask=reference_mask,
                reference_image=reference_image,
                ids=ids,
                parts=rasterizer.parts,
                colors=rasterizer.part_colors_bgr,
                view=view,
            )
        )
    records.sort(key=_alignment_candidate_sort_key)
    return records[0], records


def _alignment_candidate_sort_key(item: Mapping[str, Any]) -> tuple[Any, ...]:
    """Rank by the published alignment gate before the secondary objective."""

    similarity = item.get("whole_asset_similarity")
    transform_status = (
        similarity.get("ecc_status") if isinstance(similarity, Mapping) else None
    )
    # At equal render resolution, a camera that needs an out-of-contract 2-D
    # scale/rotation/translation can never be a valid Part-ID projection.  It
    # must not beat a physically valid camera merely by reducing one boundary
    # statistic.  Legacy/synthetic score rows without this audit remain
    # orderable; production rows always carry it.
    invalid_transform = transform_status not in {None, "success"}
    iou = float(item["projection_iou"])
    boundary_p95 = float(item["boundary_p95_px"])
    iou_deficit = max(0.0, COMPLETE_ALIGNMENT_MINIMUM_IOU - iou) / max(
        1e-12,
        1.0 - COMPLETE_ALIGNMENT_MINIMUM_IOU,
    )
    boundary_deficit = max(
        0.0,
        boundary_p95 - COMPLETE_ALIGNMENT_MAXIMUM_BOUNDARY_P95_PX,
    ) / COMPLETE_ALIGNMENT_MAXIMUM_BOUNDARY_P95_PX
    complete = iou_deficit <= 0.0 and boundary_deficit <= 0.0
    residual_audit = (
        similarity.get("ecc_transform_audit")
        if isinstance(similarity, Mapping)
        else None
    )
    residual_key = (0.0, 0.0, 0.0)
    if isinstance(residual_audit, Mapping):
        minimum_scale = float(residual_audit.get("minimum_scale", 1.0))
        maximum_scale = float(residual_audit.get("maximum_scale", minimum_scale))
        residual_key = (
            round(abs(math.log(max(1e-12, 0.5 * (minimum_scale + maximum_scale)))), 10),
            round(abs(float(residual_audit.get("rotation_degrees", 0.0))), 10),
            round(abs(float(residual_audit.get("translation_ratio", 0.0))), 10),
        )
    return (
        1 if invalid_transform else 0,
        0 if complete else 1,
        max(iou_deficit, boundary_deficit),
        iou_deficit + boundary_deficit,
        -float(item["score"]),
        -iou,
        boundary_p95,
        # Only exact metric ties reach these final fields. Prefer the camera
        # that needs the least whole-image residual correction, then a
        # canonical camera signature. Candidate list order and generated view
        # IDs therefore cannot decide an otherwise equivalent 3-D solution.
        residual_key,
        _camera_signature(item),
        str(item["view_id"]),
    )


def _rank_seed_candidates(
    candidates: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Rank global pose seeds by the same gate used for final cameras."""

    return [
        dict(item)
        for item in sorted(candidates, key=_alignment_candidate_sort_key)
    ]


def _foreground_for_view(
    registry_path: Path,
    view_id: str,
) -> np.ndarray:
    registry = _read_object(registry_path)
    colors = [
        np.asarray(
            (
                _part_color(str(part["part_id"]))[2],
                _part_color(str(part["part_id"]))[1],
                _part_color(str(part["part_id"]))[0],
            ),
            dtype=np.uint8,
        )
        for part in registry.get("parts", [])
        if isinstance(part, dict) and isinstance(part.get("part_id"), str)
    ]
    for view in registry.get("render_set", {}).get("views", []):
        if view.get("view_id") != view_id:
            continue
        ids = cv2.imread(
            str(
                _resolve_path(
                    view.get("part_ids_raw") or view.get("part_ids"),
                    registry_path,
                )
            ),
            cv2.IMREAD_COLOR,
        )
        if ids is None:
            break
        return _deterministic_part_id_foreground(ids, colors)
    raise ValueError(f"Unable to find rendered view {view_id!r} in {registry_path}")


def _part_residual_attribution(
    *,
    registry_path: Path,
    view_id: str,
    reference_mask: np.ndarray,
    score: Mapping[str, Any],
) -> list[dict[str, Any]]:
    registry = _read_object(registry_path)
    ids: np.ndarray | None = None
    for view in registry.get("render_set", {}).get("views", []):
        if view.get("view_id") == view_id:
            ids = cv2.imread(
                str(
                    _resolve_path(
                        view.get("part_ids_raw") or view.get("part_ids"),
                        registry_path,
                    )
                ),
                cv2.IMREAD_COLOR,
            )
            break
    if ids is None:
        raise ValueError(f"Unable to read Part-ID render {view_id!r}")
    matrix = np.asarray(
        score["whole_asset_similarity"]["bbox_affine"],
        dtype=np.float32,
    )
    target = reference_mask > 0
    output: list[dict[str, Any]] = []
    for part in registry.get("parts", []):
        if not isinstance(part, dict) or not isinstance(part.get("part_id"), str):
            continue
        part_id = part["part_id"]
        red, green, blue = _part_color(part_id)
        raw = np.all(
            ids == np.asarray((blue, green, red), dtype=np.uint8),
            axis=2,
        ).astype(np.uint8)
        if not np.any(raw):
            continue
        projected = (
            cv2.warpAffine(
                raw,
                matrix,
                (reference_mask.shape[1], reference_mask.shape[0]),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            > 0
        )
        projected_pixels = int(np.count_nonzero(projected))
        if projected_pixels <= 0:
            continue
        overlap_pixels = int(np.count_nonzero(np.logical_and(projected, target)))
        cad_only_pixels = projected_pixels - overlap_pixels
        output.append(
            {
                "part_id": part_id,
                "prim_path": part.get("prim_path"),
                "projected_pixels": projected_pixels,
                "overlap_pixels": overlap_pixels,
                "cad_only_pixels": cad_only_pixels,
                "inside_reference_ratio": round(
                    overlap_pixels / projected_pixels,
                    8,
                ),
            }
        )
    output.sort(
        key=lambda item: (
            -int(item["cad_only_pixels"]),
            float(item["inside_reference_ratio"]),
            str(item["part_id"]),
        )
    )
    return output


def _spec_from_score(
    score: Mapping[str, Any],
    *,
    view_id: str,
    phase: str,
) -> dict[str, Any]:
    reference_id = str(score["calibration"]["reference_view_id"])
    return {
        "view_id": view_id,
        "analysis_direction": score["analysis_direction"],
        "analysis_up_axis": score["analysis_up_axis"],
        "focal_length_mm": score["focal_length_mm"],
        "distance_multiplier": score["distance_multiplier"],
        "target_offset_u": score.get("target_offset_u", 0.0),
        "target_offset_v": score.get("target_offset_v", 0.0),
        "projection_mode": score.get("projection_mode", "perspective"),
        "orthographic_span_multiplier": score.get(
            "orthographic_span_multiplier", 2.0
        ),
        "calibration": {
            **score["calibration"],
            "reference_view_id": reference_id,
            "phase": phase,
            "source_candidate_view_id": score["view_id"],
        },
    }


def _global_finalists(
    candidates: Sequence[Mapping[str, Any]],
    *,
    count: int,
) -> list[dict[str, Any]]:
    """Keep the best distinct cameras across every refinement phase."""

    ordered = sorted(candidates, key=_alignment_candidate_sort_key)
    output: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for raw in ordered:
        signature = _camera_signature(raw)
        if signature in seen:
            continue
        seen.add(signature)
        output.append(dict(raw))
        if len(output) >= count:
            break
    return output


def _run_render(
    *,
    isaac_python: Path,
    registry: Path,
    output_dir: Path,
    view_specs: Path,
    resolution: int,
    rt_subframes: int,
    analysis_up_axis: str,
    analysis_front_axis: str,
) -> Path:
    command = [
        str(isaac_python),
        "-m",
        "qwen_material_pipeline",
        "usd",
        "render",
        "--registry",
        str(registry),
        "--output-dir",
        str(output_dir),
        "--resolution",
        str(resolution),
        "--view-specs",
        str(view_specs),
        "--rt-subframes",
        str(rt_subframes),
        "--analysis-up-axis",
        analysis_up_axis,
        f"--analysis-front-axis={analysis_front_axis}",
        "--rgb-only",
    ]
    rendered = output_dir / "part_registry.rendered.json"
    environment = os.environ.copy()
    source_root = SOURCE_ROOT
    inherited_pythonpath = environment.get("PYTHONPATH")
    pythonpath_entries = (
        []
        if not inherited_pythonpath
        else [
            entry
            for entry in inherited_pythonpath.split(os.pathsep)
            if entry and Path(entry).expanduser().resolve() != source_root
        ]
    )
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(source_root), *pythonpath_entries)
    )
    last_error: BaseException | None = None
    for attempt in range(1, RENDER_RETRY_ATTEMPTS + 1):
        # Every camera phase launches a short-lived Isaac process.  Its Python
        # extension startup is occasionally transiently unavailable after a
        # prior process shuts down.  Do not let a partial render directory be
        # mistaken for evidence, and retry the exact same immutable command.
        if output_dir.exists():
            shutil.rmtree(output_dir)
        print(
            "[CAMERA] "
            + " ".join(command)
            + f" (attempt {attempt}/{RENDER_RETRY_ATTEMPTS})",
            flush=True,
        )
        try:
            subprocess.run(command, check=True, env=environment)
            return rendered.resolve(strict=True)
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            last_error = exc
            if attempt >= RENDER_RETRY_ATTEMPTS:
                break
            print(
                "[CAMERA] transient render startup failure; "
                f"retrying the identical candidate batch in "
                f"{RENDER_RETRY_DELAY_SECONDS:.0f}s "
                f"({type(exc).__name__}: {exc})",
                flush=True,
            )
            time.sleep(RENDER_RETRY_DELAY_SECONDS)
    raise RuntimeError(
        "Camera candidate render failed after "
        f"{RENDER_RETRY_ATTEMPTS} clean retries: {output_dir}"
    ) from last_error


def _completed_phase(
    *,
    specs_path: Path,
    scores_path: Path,
    expected_specs: Mapping[str, Any],
    reference_id: str,
    phase: str,
    expected_search_backend: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
    """Return a validated completed phase that can be resumed without rendering.

    A camera-search run can be interrupted between short-lived Isaac processes.
    Reusing only a phase whose exact generated candidate specs and scored view
    IDs match the current immutable search avoids both duplicate GPU work and
    accidental reuse of a candidate batch from a different pose/configuration.
    """

    if not specs_path.is_file() or not scores_path.is_file():
        return None
    try:
        stored_specs = _read_object(specs_path)
        stored_scores = _read_object(scores_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if stored_specs != dict(expected_specs):
        return None
    if (
        stored_scores.get("schema_version") != SCHEMA_VERSION
        or stored_scores.get("reference_view_id") != reference_id
        or stored_scores.get("phase") != phase
        or stored_scores.get("camera_objective_version")
        != CAMERA_OBJECTIVE_VERSION
        or stored_scores.get("camera_selection_policy_version")
        != CAMERA_SELECTION_POLICY_VERSION
        or (
            expected_search_backend is not None
            and stored_scores.get("search_backend") != expected_search_backend
        )
    ):
        return None
    winner = stored_scores.get("winner")
    candidates = stored_scores.get("candidates")
    if not isinstance(winner, dict) or not isinstance(candidates, list) or not candidates:
        return None
    expected_ids = {
        str(view.get("view_id"))
        for view in expected_specs.get("views", [])
        if isinstance(view, Mapping) and isinstance(view.get("view_id"), str)
    }
    candidate_ids = {
        str(candidate.get("view_id"))
        for candidate in candidates
        if isinstance(candidate, Mapping) and isinstance(candidate.get("view_id"), str)
    }
    winner_id = winner.get("view_id")
    if (
        not expected_ids
        or candidate_ids != expected_ids
        or not isinstance(winner_id, str)
        or winner_id not in candidate_ids
    ):
        return None
    if any(
        not isinstance(candidate, Mapping)
        or candidate.get("objective_version") != CAMERA_OBJECTIVE_VERSION
        for candidate in candidates
    ):
        return None
    return dict(winner), [dict(candidate) for candidate in candidates]


def _fast_verification_fallback_reasons(
    verification: Mapping[str, Any],
    *,
    exact_winner: Mapping[str, Any],
    expected_candidate_count: int,
    reference_shape: Sequence[int],
) -> list[str]:
    """Audit one view before accepting a fast-search camera.

    Absolute photo alignment is deliberately left to the existing downstream
    camera gate: a difficult reference can be equally difficult for both
    renderers.  This gate instead asks whether Kaolin and the authoritative
    full-resolution Isaac render agree on the same Top-K neighborhood.  A
    disagreement reruns only this reference view through the legacy Isaac
    search, so one unsafe view cannot invalidate or slow the other views.
    """

    reasons: list[str] = []
    candidates = verification.get("candidates")
    if not isinstance(candidates, list):
        return ["verification_candidates_missing"]
    if len(candidates) != expected_candidate_count:
        reasons.append("verification_candidate_coverage_incomplete")
    if not candidates:
        return [*reasons, "verification_candidates_empty"]

    if len(reference_shape) < 2:
        raise ValueError("Reference shape requires height and width")
    diagonal = math.hypot(float(reference_shape[0]), float(reference_shape[1]))
    if not math.isfinite(diagonal) or diagonal <= 0.0:
        raise ValueError("Reference shape has no finite image diagonal")

    objective_disagreement = False
    for row in candidates:
        if not isinstance(row, Mapping):
            reasons.append("verification_candidate_invalid")
            continue
        backend = str(row.get("candidate_search_backend", ""))
        if not backend.startswith("kaolin_cuda_part_id/"):
            reasons.append("verification_candidate_backend_mixed")
        try:
            iou_delta = abs(float(row["iou_delta"]))
            score_delta = abs(float(row["score_delta"]))
            boundary_ratio = abs(float(row["boundary_p95_delta_px"])) / diagonal
        except (KeyError, TypeError, ValueError):
            reasons.append("verification_candidate_metrics_missing")
            continue
        if not all(
            math.isfinite(value)
            for value in (iou_delta, score_delta, boundary_ratio)
        ):
            reasons.append("verification_candidate_metrics_non_finite")
            continue
        if iou_delta > FAST_VERIFY_MAXIMUM_ABSOLUTE_IOU_DELTA:
            reasons.append("fast_isaac_iou_disagreement")
        if score_delta > FAST_VERIFY_MAXIMUM_ABSOLUTE_SCORE_DELTA:
            objective_disagreement = True
        if boundary_ratio > FAST_VERIFY_MAXIMUM_BOUNDARY_DELTA_DIAGONAL_RATIO:
            reasons.append("fast_isaac_boundary_disagreement")

    similarity = exact_winner.get("whole_asset_similarity")
    if (
        not isinstance(similarity, Mapping)
        or similarity.get("ecc_status") != "success"
    ):
        reasons.append("authoritative_winner_transform_invalid")
    winner_rank = verification.get("isaac_winner_candidate_search_rank")
    if (
        not isinstance(winner_rank, int)
        or winner_rank < 1
        or winner_rank > expected_candidate_count
    ):
        reasons.append("authoritative_winner_not_bound_to_top_k")
    elif (
        objective_disagreement
        and winner_rank
        > max(
            1,
            math.ceil(
                expected_candidate_count
                * FAST_VERIFY_OBJECTIVE_EDGE_RANK_FRACTION
            ),
        )
    ):
        # Isaac has already corrected the final objective. A score mismatch is
        # unsafe only when its exact winner lands near the retained Top-K
        # boundary, where an omitted candidate could plausibly have won.
        reasons.append("fast_isaac_objective_disagreement_near_top_k_edge")
    return sorted(set(reasons))


def _seed_by_reference(
    spatial_mapping: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for alignment in spatial_mapping.get("view_alignments", []):
        if not isinstance(alignment, dict):
            continue
        reference_id = alignment.get("reference_view_id")
        pose = alignment.get("camera_pose")
        if (
            not isinstance(reference_id, str)
            or not isinstance(pose, dict)
            or pose.get("analysis_direction") is None
        ):
            continue
        output[reference_id] = {
            "analysis_direction": pose["analysis_direction"],
            "analysis_up_axis": pose.get("camera_up_axis"),
            "focal_length_mm": pose.get("focal_length_mm") or 45.0,
            "distance_multiplier": 2.15,
            "target_offset_u": 0.0,
            "target_offset_v": 0.0,
            "projection_mode": "perspective",
            "orthographic_span_multiplier": 2.0,
        }
    return output


def _seed_candidates_by_view_specs(
    document: Mapping[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    if document.get("schema_version") != VIEW_SPEC_SCHEMA_VERSION:
        raise ValueError("Initial camera view specs use an unsupported schema_version")
    raw_views = document.get("views")
    if not isinstance(raw_views, list) or not raw_views:
        raise ValueError("Initial camera view specs require non-empty views")
    output: dict[str, list[dict[str, Any]]] = {}
    for raw in raw_views:
        if not isinstance(raw, Mapping) or not isinstance(raw.get("view_id"), str):
            continue
        calibration = raw.get("calibration")
        calibrated_reference_id = (
            calibration.get("reference_view_id")
            if isinstance(calibration, Mapping)
            else None
        )
        reference_id = (
            str(calibrated_reference_id)
            if isinstance(calibrated_reference_id, str)
            and calibrated_reference_id
            else str(raw["view_id"])
        )
        direction = raw.get("analysis_direction")
        up_axis = raw.get("analysis_up_axis")
        if direction is None or up_axis is None:
            continue
        output.setdefault(reference_id, []).append({
            "analysis_direction": direction,
            "analysis_up_axis": up_axis,
            "focal_length_mm": float(raw.get("focal_length_mm", 45.0)),
            "distance_multiplier": float(raw.get("distance_multiplier", 2.15)),
            "target_offset_u": float(raw.get("target_offset_u", 0.0)),
            "target_offset_v": float(raw.get("target_offset_v", 0.0)),
            "projection_mode": str(raw.get("projection_mode", "perspective")),
            "orthographic_span_multiplier": float(
                raw.get("orthographic_span_multiplier", 2.0)
            ),
            "source_seed_view_id": str(raw["view_id"]),
        })
    if not output:
        raise ValueError("Initial camera view specs contain no usable views")
    return output


def _seed_by_view_specs(document: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Return the first seed per view for legacy callers.

    Production refinement consumes all candidates through
    ``_seed_candidates_by_view_specs``. Keeping this compatibility helper
    makes single-camera tools and older tests deterministic.
    """

    candidates = _seed_candidates_by_view_specs(document)
    return {reference_id: values[0] for reference_id, values in candidates.items()}


def _namespace_candidate_specs(
    specs: Mapping[str, Any],
    *,
    seed_index: int,
    source_seed_view_id: str,
) -> dict[str, Any]:
    """Give multi-start candidates globally unique, auditable view IDs."""

    output = copy.deepcopy(dict(specs))
    views = output.get("views")
    if not isinstance(views, list):
        raise ValueError("Camera candidate specs require views")
    for raw in views:
        if not isinstance(raw, dict) or not isinstance(raw.get("view_id"), str):
            raise ValueError("Camera candidate spec has no stable view ID")
        raw["view_id"] = f"seed_{seed_index:02d}_{raw['view_id']}"
        calibration = raw.get("calibration")
        if not isinstance(calibration, dict):
            raise ValueError("Camera candidate spec has no calibration")
        calibration["refinement_seed_index"] = seed_index
        calibration["refinement_source_seed_view_id"] = source_seed_view_id
    return output


def _rotate_about_axis(
    vector: Sequence[float],
    axis: Sequence[float],
    degrees: float,
) -> list[float]:
    value = _normalize(vector)
    unit_axis = _normalize(axis)
    angle = math.radians(degrees)
    rotated = (
        value * math.cos(angle)
        + np.cross(unit_axis, value) * math.sin(angle)
        + unit_axis * float(np.dot(unit_axis, value)) * (1.0 - math.cos(angle))
    )
    return _normalize(rotated).tolist()


def _seed_by_registry(
    *,
    registry_path: Path,
    references: Mapping[str, tuple[np.ndarray, Mapping[str, Any]]],
    reference_images: Mapping[str, np.ndarray],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Select a continuous-search seed without requiring an earlier Qwen run."""

    registry = _read_object(registry_path)
    parts = [
        part
        for part in registry.get("parts", [])
        if isinstance(part, Mapping) and isinstance(part.get("part_id"), str)
    ]
    colors = [
        np.asarray(
            (
                _part_color(str(part["part_id"]))[2],
                _part_color(str(part["part_id"]))[1],
                _part_color(str(part["part_id"]))[0],
            ),
            dtype=np.uint8,
        )
        for part in parts
    ]
    rendered_views = registry.get("render_set", {}).get("views", [])
    if not isinstance(rendered_views, list) or not rendered_views:
        raise ValueError("Source registry has no rendered camera views")
    seeds: dict[str, dict[str, Any]] = {}
    audit: dict[str, Any] = {}
    for reference_id, (reference_mask, _raw) in references.items():
        candidates: list[dict[str, Any]] = []
        for view in rendered_views:
            if not isinstance(view, Mapping):
                continue
            direction = view.get("analysis_direction")
            up_axis = view.get("analysis_camera_up_axis") or view.get("camera_up_axis")
            if direction is None or up_axis is None:
                continue
            ids = cv2.imread(
                str(
                    _resolve_path(
                        view.get("part_ids_raw") or view.get("part_ids"),
                        registry_path,
                    )
                ),
                cv2.IMREAD_COLOR,
            )
            if ids is None:
                continue
            foreground = _deterministic_part_id_foreground(ids, colors)
            for quarter_turns in range(4):
                rotated_foreground = np.rot90(foreground, quarter_turns).copy()
                rotated_ids = np.rot90(ids, quarter_turns).copy()
                projection = _refine_projection(
                    reference_mask,
                    rotated_foreground,
                    DEFAULT_POLICY,
                )
                affine = np.asarray(
                    projection["bbox_affine"],
                    dtype=np.float32,
                )
                registered = cv2.warpAffine(
                    rotated_foreground,
                    affine,
                    (reference_mask.shape[1], reference_mask.shape[0]),
                    flags=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )
                boundary = _boundary_metrics(reference_mask, registered)
                structure = _part_balanced_structure_metrics(
                    ids=rotated_ids,
                    parts=parts,
                    affine=affine,
                    reference_image=reference_images[reference_id],
                    reference_mask=reference_mask,
                )
                diagonal = math.hypot(reference_mask.shape[1], reference_mask.shape[0])
                boundary_score = math.exp(
                    -float(boundary["boundary_p95_px"]) / max(3.0, 0.02 * diagonal)
                )
                coverage = _silhouette_coverage_metrics(
                    reference_mask,
                    registered,
                )
                component_coverage = _component_balanced_reference_metrics(
                    reference_mask,
                    registered,
                )
                score = (
                    0.15 * float(projection["projection_iou"])
                    + 0.15 * float(coverage["target_recall"])
                    + 0.10 * float(coverage["rendered_precision"])
                    + 0.15 * boundary_score
                    + 0.20 * float(structure["structure_score"])
                    + 0.15
                    * float(component_coverage["reference_component_macro_recall"])
                    + 0.10 * float(component_coverage["reference_component_min_recall"])
                )
                candidates.append(
                    {
                        "view_id": str(view.get("view_id")),
                        "quarter_turns_ccw": quarter_turns,
                        "objective_version": CAMERA_OBJECTIVE_VERSION,
                        "score": round(score, 8),
                        "projection_iou": projection["projection_iou"],
                        **{
                            key: round(float(value), 8)
                            for key, value in coverage.items()
                        },
                        **{
                            key: (
                                round(float(value), 8)
                                if isinstance(value, float)
                                else value
                            )
                            for key, value in component_coverage.items()
                        },
                        "boundary_p95_px": round(float(boundary["boundary_p95_px"]), 8),
                        "whole_asset_similarity": projection,
                        **structure,
                        "analysis_direction": list(direction),
                        "analysis_up_axis": _rotate_about_axis(
                            up_axis,
                            direction,
                            90.0 * quarter_turns,
                        ),
                        "focal_length_mm": float(view.get("focal_length_mm") or 45.0),
                        "distance_multiplier": float(
                            view.get("camera_distance_multiplier") or 2.15
                        ),
                        "target_offset_u": float(
                            view.get("camera_target_offset_u") or 0.0
                        ),
                        "target_offset_v": float(
                            view.get("camera_target_offset_v") or 0.0
                        ),
                        "projection_mode": str(
                            view.get("camera_projection_mode") or "perspective"
                        ),
                        "orthographic_span_multiplier": float(
                            view.get("camera_orthographic_span_multiplier") or 2.0
                        ),
                    }
                )
        if not candidates:
            raise ValueError(f"No source camera seed candidate for {reference_id}")
        # Seed selection must use the same public geometry contract as every
        # rendered refinement phase.  The blended secondary objective is
        # intentionally useful after a camera is already in the correct pose
        # basin, but it is not a safe global pose selector: a large component
        # or a slightly different foreground mask can otherwise make a
        # structurally plausible, badly bounded pose outrank a camera with a
        # substantially better silhouette/boundary fit.  That single-seed
        # mistake cannot be recovered by the later bounded local search.
        candidates = _rank_seed_candidates(candidates)
        winner = candidates[0]
        seeds[reference_id] = {
            "analysis_direction": winner["analysis_direction"],
            "analysis_up_axis": winner["analysis_up_axis"],
            "focal_length_mm": winner["focal_length_mm"],
            "distance_multiplier": winner["distance_multiplier"],
            "target_offset_u": winner.get("target_offset_u", 0.0),
            "target_offset_v": winner.get("target_offset_v", 0.0),
            "projection_mode": winner.get("projection_mode", "perspective"),
            "orthographic_span_multiplier": winner.get(
                "orthographic_span_multiplier", 2.0
            ),
        }
        audit[reference_id] = {
            "winner": winner,
            "candidate_count": len(candidates),
            "top_candidates": candidates[:5],
            "selection_policy": (
                "published_alignment_gate_then_hierarchical_objective"
            ),
        }
    return seeds, audit


def _merge_registry(
    *,
    baseline_path: Path,
    calibrated_path: Path,
    output_path: Path,
    reference_ids: set[str],
    calibration_provenance: Mapping[str, Any] | None = None,
) -> Path:
    baseline = _read_object(baseline_path)
    calibrated = _read_object(calibrated_path)
    baseline_render_set = baseline.get("render_set")
    calibrated_render_set = calibrated.get("render_set")
    if not isinstance(baseline_render_set, dict) or not isinstance(
        calibrated_render_set, dict
    ):
        raise ValueError("Rendered registries require render_set objects")
    replacements = [
        view
        for view in calibrated_render_set.get("views", [])
        if view.get("view_id") in reference_ids
    ]
    if {view.get("view_id") for view in replacements} != reference_ids:
        raise ValueError("Final calibrated render is missing reference view IDs")
    baseline["render_set"] = {
        **baseline_render_set,
        # The dense source bank is a seed-search implementation detail.  Once
        # every reference has its own calibrated camera, downstream Part-ID
        # projection must not silently switch back to a nearby discrete view.
        "views": replacements,
        "requested_view_tokens": sorted(reference_ids),
        "expanded_view_count": len(replacements),
        "view_presets": [],
        "continuous_camera_calibration": True,
        "visibility_source": "calibrated_render_set_visible_parts",
        "isolated_evidence_scope": "source_pose_bank_geometry_only",
        "calibration_source_view_count": len(baseline_render_set.get("views", [])),
        **(
            {"camera_calibration_provenance": dict(calibration_provenance)}
            if calibration_provenance is not None
            else {}
        ),
    }
    return _write_object(output_path, baseline)


def _seal_full_resolution_winners(
    *,
    rendered_path: Path,
    winners: Mapping[str, Mapping[str, Any]],
    output_path: Path,
    rendered_paths_by_reference: Mapping[str, Path] | None = None,
) -> Path:
    """Seal already-scored full-resolution frames as the final render set.

    Re-rendering an identical camera is redundant and makes final evidence
    depend on RTX sampling.  The selected full-resolution finalist is the
    exact frame whose score justified the decision, so preserve that frame and
    expose it under the reference view ID expected by downstream Part-ID
    projection.
    """

    rendered = _read_object(rendered_path)
    render_set = rendered.get("render_set")
    if not isinstance(render_set, dict):
        raise ValueError("Full-resolution finalist registry has no render_set")
    registry_cache: dict[Path, dict[str, dict[str, Any]]] = {}

    def registry_views(path: Path) -> dict[str, dict[str, Any]]:
        resolved = path.expanduser().resolve(strict=True)
        cached = registry_cache.get(resolved)
        if cached is not None:
            return cached
        document = _read_object(resolved)
        source_render_set = document.get("render_set")
        if not isinstance(source_render_set, dict):
            raise ValueError(
                f"Full-resolution winner registry has no render_set: {resolved}"
            )
        by_view_id = {
            str(view.get("view_id")): view
            for view in source_render_set.get("views", [])
            if isinstance(view, dict) and isinstance(view.get("view_id"), str)
        }
        registry_cache[resolved] = by_view_id
        return by_view_id

    sealed_views: list[dict[str, Any]] = []
    source_views: dict[str, str] = {}
    source_registries: dict[str, str] = {}
    for reference_id, winner in winners.items():
        source_registry = (
            rendered_paths_by_reference.get(reference_id, rendered_path)
            if rendered_paths_by_reference is not None
            else rendered_path
        )
        by_view_id = registry_views(source_registry)
        source_view_id = winner.get("view_id")
        if not isinstance(source_view_id, str) or source_view_id not in by_view_id:
            raise ValueError(
                f"Winning finalist for {reference_id!r} is missing from rendered registry"
            )
        view = copy.deepcopy(by_view_id[source_view_id])
        calibration = view.get("camera_calibration")
        if not isinstance(calibration, dict):
            raise ValueError(
                f"Winning finalist {source_view_id!r} has no camera calibration"
            )
        if calibration.get("reference_view_id") != reference_id:
            raise ValueError(
                f"Winning finalist {source_view_id!r} belongs to a different reference"
            )
        view["view_id"] = reference_id
        view["sealed_source_view_id"] = source_view_id
        view["camera_calibration"] = {
            **calibration,
            "phase": "sealed_full_resolution_finalist",
            "sealed_source_view_id": source_view_id,
        }
        sealed_views.append(view)
        source_views[reference_id] = source_view_id
        source_registries[reference_id] = str(source_registry.resolve(strict=True))
    rendered["render_set"] = {
        **render_set,
        "views": sealed_views,
        "requested_view_tokens": list(winners),
        "expanded_view_count": len(sealed_views),
        "sealed_full_resolution_winners": True,
        "sealed_source_views": source_views,
        "sealed_source_registries": source_registries,
    }
    return _write_object(output_path, rendered)


def calibrate(
    *,
    registry: Path,
    reference_manifest: Path,
    spatial_mapping: Path | None,
    isaac_python: Path,
    output_dir: Path,
    reference_ids: Sequence[str] | None,
    search_resolution: int,
    final_resolution: int,
    rt_subframes: int,
    analysis_up_axis: str,
    analysis_front_axis: str,
    initial_view_specs: Path | None = None,
    search_phases: Sequence[str] | None = None,
    fast_search_mode: str = "disabled",
) -> dict[str, Any]:
    registry = registry.expanduser().resolve(strict=True)
    reference_manifest = reference_manifest.expanduser().resolve(strict=True)
    spatial_mapping = (
        spatial_mapping.expanduser().resolve(strict=True)
        if spatial_mapping is not None
        else None
    )
    initial_view_specs = (
        initial_view_specs.expanduser().resolve(strict=True)
        if initial_view_specs is not None
        else None
    )
    active_phases = tuple(search_phases or CAMERA_PHASES)
    if not active_phases:
        raise ValueError("Camera calibration requires at least one search phase")
    invalid_phases = [phase for phase in active_phases if phase not in CAMERA_PHASES]
    if invalid_phases:
        raise ValueError(
            "Unknown camera calibration search phases: " + ", ".join(invalid_phases)
        )
    if fast_search_mode not in FAST_SEARCH_MODES:
        raise ValueError(
            "Unknown fast camera search mode: "
            f"{fast_search_mode!r}; expected one of {', '.join(FAST_SEARCH_MODES)}"
        )
    isaac_python = isaac_python.expanduser().resolve(strict=True)
    destination = output_dir.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    references = _reference_masks(reference_manifest)
    reference_images = {
        reference_id: _reference_image(
            raw,
            reference_manifest,
            mask.shape,
        )
        for reference_id, (mask, raw) in references.items()
    }
    if initial_view_specs is not None:
        seed_candidates = _seed_candidates_by_view_specs(
            _read_object(initial_view_specs)
        )
        seeds = {
            reference_id: candidates[0]
            for reference_id, candidates in seed_candidates.items()
        }
        seed_audit = {
            "mode": "existing_continuous_camera_specs",
            "initial_view_specs": str(initial_view_specs),
            "candidate_count_by_reference": {
                reference_id: len(candidates)
                for reference_id, candidates in seed_candidates.items()
            },
        }
    elif spatial_mapping is not None:
        seeds = _seed_by_reference(_read_object(spatial_mapping))
        seed_candidates = {
            reference_id: [
                {**seed, "source_seed_view_id": reference_id}
            ]
            for reference_id, seed in seeds.items()
        }
        seed_audit: dict[str, Any] = {
            "mode": "existing_spatial_mapping",
            "spatial_mapping": str(spatial_mapping),
        }
    else:
        seeds, initial_candidates = _seed_by_registry(
            registry_path=registry,
            references=references,
            reference_images=reference_images,
        )
        seed_candidates = {
            reference_id: [
                {**seed, "source_seed_view_id": reference_id}
            ]
            for reference_id, seed in seeds.items()
        }
        seed_audit = {
            "mode": "source_render_continuous_seed",
            "references": initial_candidates,
        }
    requested = list(reference_ids or references)
    missing = sorted(set(requested) - set(references) | (set(requested) - set(seeds)))
    if missing:
        raise ValueError(
            "Missing reference masks or camera seeds: " + ", ".join(missing)
        )
    registry_input_contract = _registry_geometry_and_pose_contract(registry)
    reference_input_contract = _reference_evidence_contract(
        references,
        reference_images,
    )
    initial_specs_document = (
        _read_object(initial_view_specs) if initial_view_specs is not None else None
    )
    spatial_mapping_document = (
        _read_object(spatial_mapping) if spatial_mapping is not None else None
    )
    calibration_input_contract = {
        "schema_version": "qwen-camera-calibration-input/v1",
        "camera_objective_version": CAMERA_OBJECTIVE_VERSION,
        "camera_selection_policy_version": CAMERA_SELECTION_POLICY_VERSION,
        "registry_geometry_and_pose_sha256": _canonical_sha256(
            registry_input_contract
        ),
        "reference_evidence_sha256": _canonical_sha256(reference_input_contract),
        "initial_view_specs_sha256": (
            _canonical_sha256(initial_specs_document)
            if initial_specs_document is not None
            else None
        ),
        "spatial_mapping_sha256": (
            _canonical_sha256(spatial_mapping_document)
            if spatial_mapping_document is not None
            else None
        ),
        "reference_view_ids": sorted(requested),
        "search_phases": list(active_phases),
        "search_resolution": int(search_resolution),
        "final_resolution": int(final_resolution),
        "rt_subframes": int(rt_subframes),
        "analysis_up_axis": str(analysis_up_axis),
        "analysis_front_axis": str(analysis_front_axis),
        "fast_search_mode": str(fast_search_mode),
        "whole_asset_similarity_policy_sha256": _canonical_sha256(DEFAULT_POLICY),
    }
    calibration_input_fingerprint = _canonical_sha256(calibration_input_contract)

    fast_rasterizer = None
    fast_fallback_reason: str | None = None
    if fast_search_mode != "disabled":
        from qwen_material_pipeline.evidence.camera_fast_raster import (
            FastCameraRasterUnavailable,
            FastPartIdRasterizer,
        )

        try:
            fast_rasterizer = FastPartIdRasterizer(
                registry_path=registry,
                resolution=search_resolution,
                analysis_up_axis=analysis_up_axis,
                analysis_front_axis=analysis_front_axis,
            )
        except FastCameraRasterUnavailable as exc:
            if fast_search_mode == "required":
                raise
            fast_fallback_reason = str(exc)
            print(
                "[CAMERA] fast Part-ID raster unavailable; "
                f"falling back to Isaac candidate search ({exc})",
                flush=True,
            )
        else:
            print(
                "[CAMERA] fast Part-ID raster ready: "
                f"{fast_rasterizer.audit['triangle_count']} triangles, "
                f"{search_resolution}px, "
                f"initialized in "
                f"{fast_rasterizer.audit['initialization_seconds']:.3f}s",
                flush=True,
            )

    winners: dict[str, dict[str, Any]] = {}
    phases: dict[str, list[dict[str, Any]]] = {}
    finalists: dict[str, list[dict[str, Any]]] = {}
    phase_search_backend = (
        str(fast_rasterizer.audit["backend"])
        if fast_rasterizer is not None
        else "isaac_rtx_part_id"
    )
    for view_index, reference_id in enumerate(requested, start=1):
        print(
            f"[CAMERA] reference {view_index}/{len(requested)} {reference_id}",
            flush=True,
        )
        phase_records: list[dict[str, Any]] = []
        phase_candidate_pool: list[dict[str, Any]] = []
        reference_seeds = seed_candidates[reference_id]
        multi_start = len(reference_seeds) > 1
        for seed_index, initial_seed in enumerate(reference_seeds, start=1):
            seed = dict(initial_seed)
            source_seed_view_id = str(
                seed.get("source_seed_view_id", reference_id)
            )
            phase_dir = (
                destination / reference_id / f"seed_{seed_index:02d}"
                if multi_start
                else destination / reference_id
            )
            for phase in active_phases:
                specs = _candidate_specs(
                    reference_id=reference_id,
                    seed=seed,
                    phase=phase,
                )
                if multi_start:
                    specs = _namespace_candidate_specs(
                        specs,
                        seed_index=seed_index,
                        source_seed_view_id=source_seed_view_id,
                    )
                view_specs_path = phase_dir / f"{phase}_view_specs.json"
                scores_path = phase_dir / f"{phase}_scores.json"
                completed = _completed_phase(
                    specs_path=view_specs_path,
                    scores_path=scores_path,
                    expected_specs=specs,
                    reference_id=reference_id,
                    phase=phase,
                    expected_search_backend=phase_search_backend,
                )
                if completed is not None:
                    winner, candidates = completed
                    phase_backend = str(
                        winner.get(
                            "render_backend", "verified_legacy_checkpoint"
                        )
                    )
                    print(
                        f"[CAMERA] {reference_id}/seed_{seed_index:02d}/"
                        f"{phase} reusing verified completed candidate batch",
                        flush=True,
                    )
                else:
                    specs_path = _write_object(view_specs_path, specs)
                    if fast_rasterizer is not None:
                        winner, candidates = _score_fast_candidates(
                            reference_id=reference_id,
                            reference_mask=references[reference_id][0],
                            reference_image=reference_images[reference_id],
                            specs=specs,
                            rasterizer=fast_rasterizer,
                        )
                        phase_backend = str(fast_rasterizer.audit["backend"])
                    else:
                        rendered = _run_render(
                            isaac_python=isaac_python,
                            registry=registry,
                            output_dir=phase_dir / f"{phase}_renders",
                            view_specs=specs_path,
                            resolution=search_resolution,
                            rt_subframes=rt_subframes,
                            analysis_up_axis=analysis_up_axis,
                            analysis_front_axis=analysis_front_axis,
                        )
                        winner, candidates = _score_candidates(
                            reference_id=reference_id,
                            reference_mask=references[reference_id][0],
                            reference_image=reference_images[reference_id],
                            registry_path=rendered,
                        )
                        phase_backend = "isaac_rtx_part_id"
                    _write_object(
                        scores_path,
                        {
                            "schema_version": SCHEMA_VERSION,
                            "camera_objective_version": CAMERA_OBJECTIVE_VERSION,
                            "camera_selection_policy_version": (
                                CAMERA_SELECTION_POLICY_VERSION
                            ),
                            "reference_view_id": reference_id,
                            "phase": phase,
                            "refinement_seed_index": seed_index,
                            "refinement_source_seed_view_id": (
                                source_seed_view_id
                            ),
                            "search_backend": phase_backend,
                            "winner": winner,
                            "candidates": candidates,
                        },
                    )
                phase_records.append(
                    {
                        "phase": phase,
                        "refinement_seed_index": seed_index,
                        "refinement_source_seed_view_id": source_seed_view_id,
                        "winner": winner,
                        "candidate_count": len(candidates),
                        "search_backend": phase_backend,
                    }
                )
                phase_candidate_pool.extend(candidates[:FINALIST_COUNT])
                seed = {
                    "analysis_direction": winner["analysis_direction"],
                    "analysis_up_axis": winner["analysis_up_axis"],
                    "focal_length_mm": winner["focal_length_mm"],
                    "distance_multiplier": winner["distance_multiplier"],
                    "target_offset_u": winner.get("target_offset_u", 0.0),
                    "target_offset_v": winner.get("target_offset_v", 0.0),
                    "projection_mode": winner.get(
                        "projection_mode", "perspective"
                    ),
                    "orthographic_span_multiplier": winner.get(
                        "orthographic_span_multiplier", 2.0
                    ),
                }
                print(
                    f"[CAMERA] {reference_id}/seed_{seed_index:02d}/{phase} "
                    f"IoU={winner['projection_iou']:.4f} "
                    f"boundary_p95={winner['boundary_p95_px']:.2f}px",
                    flush=True,
                )
        finalists[reference_id] = _global_finalists(
            phase_candidate_pool,
            count=(
                FAST_FINALIST_COUNT
                if fast_rasterizer is not None
                else FINALIST_COUNT
            ),
        )
        winners[reference_id] = finalists[reference_id][0]
        phases[reference_id] = phase_records

    if fast_rasterizer is not None:
        # Isaac owns the authoritative full-resolution verification.  Drop
        # every search tensor and empty Torch's cache before that independent
        # process allocates RTX/Replicator resources on the same GPU.
        fast_rasterizer.release()

    finalist_specs = {
        "schema_version": VIEW_SPEC_SCHEMA_VERSION,
        "views": [
            _spec_from_score(
                score,
                view_id=f"rerank_{reference_id}_{rank:02d}",
                phase="full_resolution_rerank",
            )
            for reference_id in requested
            for rank, score in enumerate(finalists[reference_id], start=1)
        ],
    }
    finalist_specs_path = _write_object(
        destination / "full_resolution_finalists.json",
        finalist_specs,
    )
    finalist_rendered = _run_render(
        isaac_python=isaac_python,
        registry=registry,
        output_dir=destination / "full_resolution_finalists",
        view_specs=finalist_specs_path,
        resolution=final_resolution,
        rt_subframes=rt_subframes,
        analysis_up_axis=analysis_up_axis,
        analysis_front_axis=analysis_front_axis,
    )
    finalist_verification: dict[str, dict[str, Any]] = {}
    exact_finalist_winners: dict[str, dict[str, Any]] = {}
    exact_finalist_candidates: dict[str, list[dict[str, Any]]] = {}
    for reference_id in requested:
        winner, candidates = _score_candidates(
            reference_id=reference_id,
            reference_mask=references[reference_id][0],
            reference_image=reference_images[reference_id],
            registry_path=finalist_rendered,
        )
        fast_by_id = {
            str(item["view_id"]): item for item in finalists[reference_id]
        }
        verification_candidates: list[dict[str, Any]] = []
        for exact in candidates:
            calibration = exact.get("calibration")
            source_id = (
                calibration.get("source_candidate_view_id")
                if isinstance(calibration, Mapping)
                else None
            )
            predicted = fast_by_id.get(str(source_id))
            if predicted is None:
                raise ValueError(
                    "Full-resolution Isaac finalist is not bound to its "
                    f"candidate-search source: {exact.get('view_id')!r}"
                )
            verification_candidates.append(
                {
                    "isaac_view_id": exact["view_id"],
                    "source_candidate_view_id": source_id,
                    "candidate_search_rank": finalists[reference_id].index(predicted)
                    + 1,
                    "candidate_search_backend": predicted.get("render_backend"),
                    "candidate_search_iou": predicted["projection_iou"],
                    "isaac_iou": exact["projection_iou"],
                    "iou_delta": round(
                        float(exact["projection_iou"])
                        - float(predicted["projection_iou"]),
                        8,
                    ),
                    "candidate_search_score": predicted["score"],
                    "isaac_score": exact["score"],
                    "score_delta": round(
                        float(exact["score"]) - float(predicted["score"]),
                        8,
                    ),
                    "candidate_search_boundary_p95_px": predicted[
                        "boundary_p95_px"
                    ],
                    "isaac_boundary_p95_px": exact["boundary_p95_px"],
                    "boundary_p95_delta_px": round(
                        float(exact["boundary_p95_px"])
                        - float(predicted["boundary_p95_px"]),
                        8,
                    ),
                }
            )
        exact_winner_source = winner["calibration"].get(
            "source_candidate_view_id"
        )
        exact_winner_record = next(
            (
                row
                for row in verification_candidates
                if row["source_candidate_view_id"] == exact_winner_source
            ),
            None,
        )
        if exact_winner_record is None:
            raise ValueError(
                f"Unable to bind Isaac winner for {reference_id} to search evidence"
            )
        finalist_verification[reference_id] = {
            "candidate_count": len(verification_candidates),
            "authoritative_backend": "isaac_rtx_part_id",
            "isaac_winner_view_id": winner["view_id"],
            "isaac_winner_source_candidate_view_id": exact_winner_source,
            "isaac_winner_candidate_search_rank": exact_winner_record[
                "candidate_search_rank"
            ],
            "candidate_search_top1_preserved": (
                exact_winner_record["candidate_search_rank"] == 1
            ),
            "candidates": verification_candidates,
        }
        exact_finalist_winners[reference_id] = winner
        exact_finalist_candidates[reference_id] = candidates
        winners[reference_id] = winner
        phases[reference_id].append(
            {
                "phase": "full_resolution_rerank",
                "winner": winner,
                "candidate_count": len(candidates),
                "search_backend": "isaac_rtx_part_id",
                "candidate_search_verification": finalist_verification[
                    reference_id
                ],
            }
        )
        _write_object(
            destination / reference_id / "full_resolution_scores.json",
            {
                "schema_version": SCHEMA_VERSION,
                "camera_objective_version": CAMERA_OBJECTIVE_VERSION,
                "camera_selection_policy_version": (
                    CAMERA_SELECTION_POLICY_VERSION
                ),
                "reference_view_id": reference_id,
                "phase": "full_resolution_rerank",
                "winner": winner,
                "candidates": candidates,
            },
        )
        print(
            f"[CAMERA] {reference_id}/full_resolution_rerank "
            f"IoU={winner['projection_iou']:.4f} "
            f"boundary_p95={winner['boundary_p95_px']:.2f}px",
            flush=True,
        )

    winner_rendered_paths: dict[str, Path] = {
        reference_id: finalist_rendered for reference_id in requested
    }
    per_view_fallback: dict[str, dict[str, Any]] = {}
    fallback_refinement_views: dict[str, list[dict[str, Any]]] = {}
    if fast_rasterizer is not None:
        for reference_id in requested:
            verification = finalist_verification[reference_id]
            reasons = _fast_verification_fallback_reasons(
                verification,
                exact_winner=exact_finalist_winners[reference_id],
                expected_candidate_count=len(finalists[reference_id]),
                reference_shape=references[reference_id][0].shape,
            )
            verification["fast_search_accepted"] = not reasons
            verification["fallback_reasons"] = reasons
            if not reasons:
                print(
                    f"[CAMERA] {reference_id}/fast_search_gate accepted "
                    f"{len(finalists[reference_id])} Isaac-verified finalists",
                    flush=True,
                )
                continue
            if fast_search_mode == "required":
                raise RuntimeError(
                    "Fast camera search failed authoritative per-view "
                    f"verification for {reference_id}: {', '.join(reasons)}"
                )

            print(
                f"[CAMERA] {reference_id}/fast_search_gate unsafe "
                f"({', '.join(reasons)}); rerunning only this view with "
                "the legacy Isaac search",
                flush=True,
            )
            fallback_dir = destination / "per_view_legacy_fallback" / reference_id
            fallback_report = calibrate(
                registry=registry,
                reference_manifest=reference_manifest,
                spatial_mapping=spatial_mapping,
                isaac_python=isaac_python,
                output_dir=fallback_dir,
                reference_ids=(reference_id,),
                search_resolution=search_resolution,
                final_resolution=final_resolution,
                rt_subframes=rt_subframes,
                analysis_up_axis=analysis_up_axis,
                analysis_front_axis=analysis_front_axis,
                initial_view_specs=initial_view_specs,
                search_phases=active_phases,
                fast_search_mode="disabled",
            )
            fallback_views = fallback_report.get("views")
            if not isinstance(fallback_views, list) or len(fallback_views) != 1:
                raise RuntimeError(
                    f"Legacy camera fallback for {reference_id} returned an "
                    "invalid view report"
                )
            fallback_final = fallback_views[0].get("final")
            fallback_registry_value = fallback_report.get("final_rendered_registry")
            if not isinstance(fallback_final, dict) or not isinstance(
                fallback_registry_value, str
            ):
                raise RuntimeError(
                    f"Legacy camera fallback for {reference_id} lacks exact "
                    "full-resolution evidence"
                )
            fallback_registry = Path(fallback_registry_value).resolve(strict=True)
            fallback_refinement_value = fallback_report.get(
                "refinement_seed_view_specs"
            )
            if not isinstance(fallback_refinement_value, str):
                raise RuntimeError(
                    f"Legacy camera fallback for {reference_id} lacks "
                    "refinement seeds"
                )
            fallback_refinement_document = _read_object(
                Path(fallback_refinement_value).resolve(strict=True)
            )
            fallback_candidates = _seed_candidates_by_view_specs(
                fallback_refinement_document
            ).get(reference_id, [])
            if not fallback_candidates:
                raise RuntimeError(
                    f"Legacy camera fallback for {reference_id} produced no "
                    "usable refinement seed"
                )
            fallback_refinement_views[reference_id] = [
                copy.deepcopy(raw)
                for raw in fallback_refinement_document.get("views", [])
                if isinstance(raw, dict)
                and isinstance(raw.get("calibration"), Mapping)
                and raw["calibration"].get("reference_view_id")
                == reference_id
            ][:REFINEMENT_SEED_COUNT]
            winners[reference_id] = dict(fallback_final)
            winner_rendered_paths[reference_id] = fallback_registry
            per_view_fallback[reference_id] = {
                "reasons": reasons,
                "report": str(
                    (fallback_dir / "camera_calibration_report.json").resolve(
                        strict=True
                    )
                ),
                "selected_backend": "isaac_rtx_part_id",
                "selected_iou": fallback_final.get("projection_iou"),
                "selected_boundary_p95_px": fallback_final.get(
                    "boundary_p95_px"
                ),
            }
            phases[reference_id].append(
                {
                    "phase": "per_view_legacy_fallback",
                    "winner": dict(fallback_final),
                    "candidate_count": None,
                    "search_backend": "isaac_rtx_part_id",
                    "fallback_reasons": reasons,
                    "fallback_report": per_view_fallback[reference_id]["report"],
                }
            )

    refinement_seed_views: list[dict[str, Any]] = []
    for reference_id in requested:
        if reference_id in fallback_refinement_views:
            for rank, raw in enumerate(
                fallback_refinement_views[reference_id], start=1
            ):
                raw["view_id"] = f"refine_seed_{reference_id}_{rank:02d}"
                refinement_seed_views.append(raw)
            continue
        refinement_seed_views.extend(
            _spec_from_score(
                score,
                view_id=f"refine_seed_{reference_id}_{rank:02d}",
                phase="next_pass_refinement_seed",
            )
            for rank, score in enumerate(
                _global_finalists(
                    exact_finalist_candidates[reference_id],
                    count=REFINEMENT_SEED_COUNT,
                ),
                start=1,
            )
        )
    refinement_seed_specs = {
        "schema_version": VIEW_SPEC_SCHEMA_VERSION,
        "views": refinement_seed_views,
    }
    refinement_seed_specs_path = _write_object(
        destination / "refinement_seed_view_specs.json",
        refinement_seed_specs,
    )
    final_specs = {
        "schema_version": VIEW_SPEC_SCHEMA_VERSION,
        "views": [
            _spec_from_score(
                winners[reference_id],
                view_id=reference_id,
                phase="final",
            )
            for reference_id in requested
        ],
    }
    solution_contract = _camera_solution_contract(final_specs)
    solution_fingerprint = _canonical_sha256(solution_contract)
    final_specs_path = _write_object(destination / "final_view_specs.json", final_specs)
    final_rendered = _seal_full_resolution_winners(
        rendered_path=finalist_rendered,
        winners=winners,
        output_path=destination / "sealed_finalists" / "part_registry.rendered.json",
        rendered_paths_by_reference=winner_rendered_paths,
    )
    final_scores: dict[str, Any] = {}
    residual_audits: dict[str, Any] = {}
    for reference_id in requested:
        winner, candidates = _score_candidates(
            reference_id=reference_id,
            reference_mask=references[reference_id][0],
            reference_image=reference_images[reference_id],
            registry_path=final_rendered,
        )
        final_scores[reference_id] = winner
        residual_audits[reference_id] = _write_residual_audit(
            reference_id=reference_id,
            reference_mask=references[reference_id][0],
            foreground=_foreground_for_view(final_rendered, reference_id),
            score=winner,
            part_residuals=_part_residual_attribution(
                registry_path=final_rendered,
                view_id=reference_id,
                reference_mask=references[reference_id][0],
                score=winner,
            ),
            output_dir=destination / "residual_audit",
        )
    merged_registry = _merge_registry(
        baseline_path=registry,
        calibrated_path=final_rendered,
        output_path=destination / "part_registry.camera_calibrated.json",
        reference_ids=set(requested),
        calibration_provenance={
            "camera_objective_version": CAMERA_OBJECTIVE_VERSION,
            "camera_selection_policy_version": CAMERA_SELECTION_POLICY_VERSION,
            "calibration_input_fingerprint": calibration_input_fingerprint,
            "camera_solution_fingerprint": solution_fingerprint,
        },
    )
    fast_raster_audit: dict[str, Any] | None = None
    if fast_rasterizer is not None:
        fast_raster_audit = dict(fast_rasterizer.audit)
        current_process_candidate_count = int(
            fast_raster_audit.get("candidate_count", 0)
        )
        verified_phase_candidate_count = sum(
            int(record["candidate_count"])
            for reference_id in requested
            for record in phases[reference_id]
            if str(record.get("search_backend", "")).startswith(
                "kaolin_cuda_part_id/"
            )
            and isinstance(record.get("candidate_count"), int)
        )
        fast_raster_audit.update(
            {
                "current_process_candidate_count": (
                    current_process_candidate_count
                ),
                "current_process_raster_seconds": float(
                    fast_raster_audit.get("raster_seconds", 0.0)
                ),
                "verified_phase_candidate_count": (
                    verified_phase_candidate_count
                ),
                "phase_checkpoint_reuse_detected": (
                    verified_phase_candidate_count
                    > current_process_candidate_count
                ),
            }
        )
    report = {
        "schema_version": SCHEMA_VERSION,
        "source_registry": str(registry),
        "source_registry_sha256": _sha256_file(registry),
        "reference_manifest": str(reference_manifest),
        "reference_manifest_sha256": _sha256_file(reference_manifest),
        "source_spatial_mapping": (
            str(spatial_mapping) if spatial_mapping is not None else None
        ),
        "source_spatial_mapping_sha256": (
            _sha256_file(spatial_mapping) if spatial_mapping is not None else None
        ),
        "source_initial_view_specs": (
            str(initial_view_specs) if initial_view_specs is not None else None
        ),
        "source_initial_view_specs_sha256": (
            _sha256_file(initial_view_specs)
            if initial_view_specs is not None
            else None
        ),
        "calibration_input_contract": calibration_input_contract,
        "calibration_input_fingerprint": calibration_input_fingerprint,
        "camera_solution_contract": solution_contract,
        "camera_solution_fingerprint": solution_fingerprint,
        "search_phases": list(active_phases),
        "candidate_search": {
            "requested_mode": fast_search_mode,
            "selected_backend": (
                "per_view_hybrid"
                if per_view_fallback
                else (
                    fast_rasterizer.audit["backend"]
                    if fast_rasterizer is not None
                    else "isaac_rtx_part_id"
                )
            ),
            "fallback_reason": fast_fallback_reason,
            "per_view_selected_backend": {
                reference_id: (
                    "isaac_rtx_part_id"
                    if reference_id in per_view_fallback
                    else phase_search_backend
                )
                for reference_id in requested
            },
            "per_view_fallback": per_view_fallback,
            "fast_finalist_count": (
                FAST_FINALIST_COUNT if fast_rasterizer is not None else None
            ),
            "fast_raster_audit": (
                fast_raster_audit
            ),
            "authoritative_full_resolution_backend": "isaac_rtx_part_id",
            "full_resolution_verification": finalist_verification,
        },
        "seed_search": seed_audit,
        "whole_asset_only": True,
        "per_part_geometric_warp_applied": False,
        "camera_objective_version": CAMERA_OBJECTIVE_VERSION,
        "camera_selection_policy_version": CAMERA_SELECTION_POLICY_VERSION,
        "camera_intrinsics_optimized": [
            "projection_mode",
            "focal_length_mm",
            "orthographic_aperture",
        ],
        "camera_extrinsics_optimized": [
            "orbit_azimuth",
            "orbit_elevation",
            "camera_distance",
            "optical_axis_target_u",
            "optical_axis_target_v",
        ],
        "image_frame_residual_optimized": [
            "uniform_scale",
            "in_plane_rotation",
            "principal_point_xy",
            "crop_translation_xy",
        ],
        "part_balanced_size_strata": ["small", "medium", "large"],
        "search_resolution": search_resolution,
        "final_resolution": final_resolution,
        "views": [
            {
                "reference_view_id": reference_id,
                "phases": phases[reference_id],
                "final": final_scores[reference_id],
                "residual_audit": residual_audits[reference_id],
                "full_resolution_reproducibility": {
                    "selected_candidate_iou": winners[reference_id]["projection_iou"],
                    "sealed_final_iou": final_scores[reference_id]["projection_iou"],
                    "minimum_allowed_iou": round(
                        float(winners[reference_id]["projection_iou"]) - 0.01,
                        8,
                    ),
                    "passed": (
                        float(final_scores[reference_id]["projection_iou"])
                        >= float(winners[reference_id]["projection_iou"]) - 0.01
                    ),
                },
                "complete_alignment_target": {
                    "minimum_iou": COMPLETE_ALIGNMENT_MINIMUM_IOU,
                    "maximum_boundary_p95_px": (
                        COMPLETE_ALIGNMENT_MAXIMUM_BOUNDARY_P95_PX
                    ),
                },
                "complete_alignment_passed": (
                    float(final_scores[reference_id]["projection_iou"])
                    >= COMPLETE_ALIGNMENT_MINIMUM_IOU
                    and float(final_scores[reference_id]["boundary_p95_px"])
                    <= COMPLETE_ALIGNMENT_MAXIMUM_BOUNDARY_P95_PX
                    and float(final_scores[reference_id]["projection_iou"])
                    >= float(winners[reference_id]["projection_iou"]) - 0.01
                ),
            }
            for reference_id in requested
        ],
        "final_view_specs": str(final_specs_path),
        "final_view_specs_sha256": _sha256_file(final_specs_path),
        "refinement_seed_view_specs": str(refinement_seed_specs_path),
        "refinement_seed_view_specs_sha256": _sha256_file(
            refinement_seed_specs_path
        ),
        "refinement_seed_count_per_reference": REFINEMENT_SEED_COUNT,
        "refinement_seed_actual_count_by_reference": {
            reference_id: sum(
                1
                for raw in refinement_seed_views
                if isinstance(raw.get("calibration"), Mapping)
                and raw["calibration"].get("reference_view_id")
                == reference_id
            )
            for reference_id in requested
        },
        "full_resolution_finalists": str(finalist_specs_path),
        "full_resolution_finalist_renders": str(finalist_rendered),
        "final_rendered_registry": str(final_rendered),
        "final_render_mode": "sealed_scored_full_resolution_finalist",
        "merged_rendered_registry": str(merged_registry),
    }
    _write_object(destination / "camera_calibration_report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--spatial-mapping", type=Path)
    parser.add_argument(
        "--initial-view-specs",
        type=Path,
        help=(
            "optional qwen-camera-view-specs/v1 seed; useful for resuming a "
            "strictly rigid calibration without reusing any material result"
        ),
    )
    parser.add_argument(
        "--search-phases",
        help=(
            "optional comma-separated ordered subset of the production phases; "
            "defaults to the complete from-zero search"
        ),
    )
    parser.add_argument("--isaac-python", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--reference-ids",
        help="comma-separated subset; defaults to every reference view",
    )
    parser.add_argument("--search-resolution", type=int, default=256)
    parser.add_argument("--final-resolution", type=int, default=512)
    parser.add_argument("--rt-subframes", type=int, default=2)
    parser.add_argument("--analysis-up-axis", default="z")
    parser.add_argument("--analysis-front-axis", default="-y")
    parser.add_argument(
        "--fast-search",
        choices=FAST_SEARCH_MODES,
        default="auto",
        help=(
            "use one-load CUDA Part-ID rasterization for candidate search; "
            "Top-K cameras are always verified by Isaac at full resolution"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = calibrate(
        registry=args.registry,
        reference_manifest=args.reference_manifest,
        spatial_mapping=args.spatial_mapping,
        isaac_python=args.isaac_python,
        output_dir=args.output_dir,
        reference_ids=(
            [value.strip() for value in args.reference_ids.split(",") if value.strip()]
            if args.reference_ids
            else None
        ),
        search_resolution=args.search_resolution,
        final_resolution=args.final_resolution,
        rt_subframes=args.rt_subframes,
        analysis_up_axis=args.analysis_up_axis,
        analysis_front_axis=args.analysis_front_axis,
        initial_view_specs=args.initial_view_specs,
        search_phases=(
            [value.strip() for value in args.search_phases.split(",") if value.strip()]
            if args.search_phases
            else None
        ),
        fast_search_mode=args.fast_search,
    )
    print(
        json.dumps(
            {
                "report": str(
                    (args.output_dir / "camera_calibration_report.json").resolve()
                ),
                "view_count": len(report["views"]),
                "passed": sum(
                    item["complete_alignment_passed"] for item in report["views"]
                ),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
