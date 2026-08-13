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
import importlib
import itertools
import json
import math
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import cv2
import numpy as np

from qwen_material_pipeline.evidence.spatial import (
    DEFAULT_POLICY,
    _part_color,
    _refine_projection,
)


SCHEMA_VERSION = "qwen-whole-asset-camera-calibration/v9"
VIEW_SPEC_SCHEMA_VERSION = "qwen-camera-view-specs/v1"
FINALIST_COUNT = 5
RENDER_RETRY_ATTEMPTS = 3
RENDER_RETRY_DELAY_SECONDS = 3.0
# Real Isaac 5.0 runs on the 24 GiB workstation showed that the third new
# render batch in one session can enter a native-crash window.  Two completed
# batches still halves process startups while keeping the worker lifetime
# strictly below every observed failure point.
SUPERVISOR_RENDER_BATCH_LIMIT = 2
SUPERVISOR_ROTATION_SCHEMA_VERSION = "qwen-camera-session-rotation/v1"
SUPERVISOR_ROTATION_MARKER = ".camera_session_rotation.json"
MAX_FOCAL_LENGTH_MM = 2000.0
CAMERA_OBJECTIVE_VERSION = "observable_rigid_part_consensus/v15"
CHECKPOINT_COMPATIBLE_OBJECTIVE_VERSIONS = frozenset(
    {
        CAMERA_OBJECTIVE_VERSION,
    }
)
COMPLETE_ALIGNMENT_MINIMUM_IOU = 0.97
COMPLETE_ALIGNMENT_MAXIMUM_BOUNDARY_P95_PX = 3.0
PHASE_INCUMBENT_MAXIMUM_IOU_REGRESSION = 0.005
PHASE_INCUMBENT_MAXIMUM_BOUNDARY_REGRESSION_PX = 0.75
# A camera candidate is scored after only a small, camera-independent frame
# normalization.  Unlike the historical bbox fit, this correction cannot hide
# an incorrect focal length, distance, roll, or principal point.
FRAME_RESIDUAL_MINIMUM_SCALE = 0.985
FRAME_RESIDUAL_MAXIMUM_SCALE = 1.015
FRAME_RESIDUAL_MAXIMUM_ROTATION_DEGREES = 1.0
FRAME_RESIDUAL_MAXIMUM_TRANSLATION_RATIO = 0.02
FRAME_RESIDUAL_OPTIMIZATION_MAXIMUM_SIDE = 256
RIGID_CONSENSUS_MINIMUM_PART_PIXELS = 24
RIGID_CONSENSUS_MAXIMUM_PARTS_PER_STRATUM = 16
RIGID_CONSENSUS_MINIMUM_INLIER_PARTS = 3
RIGID_CONSENSUS_MINIMUM_INLIER_COVERAGE = 0.35
RIGID_CONSENSUS_HUBER_CUTOFF = 1.5
RIGID_CONSENSUS_OUTLIER_CUTOFF = 3.5
RIGID_CONSENSUS_MINIMUM_EDGE_SUPPORT = 0.25
RIGID_CONSENSUS_MAXIMUM_INSIDE_RATIO_FOR_SILHOUETTE_EVIDENCE = 0.85
CAMERA_PHASES = (
    "coarse",
    "lens",
    "fine",
    "roll",
    "principal_point",
    "radial_distortion",
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
    "frame_micro",
    "radial_distortion_micro",
    "pico",
    "target_pico",
)

RenderRunner = Callable[..., Path]


class _RenderBatchBudgetReached(RuntimeError):
    """Request a fresh Isaac process after a sealed camera-search phase."""

    def __init__(self, message: str, *, checkpoint: Path) -> None:
        super().__init__(message)
        self.checkpoint = checkpoint


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


def _roll_up_axis(
    direction: Sequence[float],
    up_axis: Sequence[float],
    roll_degrees: float,
) -> list[float]:
    """Rotate a camera up vector around its optical/orbit direction."""

    view = _normalize(direction)
    up = _normalize(up_axis)
    up = up - float(np.dot(up, view)) * view
    up = _normalize(up)
    angle = math.radians(float(roll_degrees))
    rotated = up * math.cos(angle) + np.cross(view, up) * math.sin(angle)
    return _normalize(rotated).tolist()


def _transport_up_axis(
    source_direction: Sequence[float],
    source_up: Sequence[float],
    target_direction: Sequence[float],
) -> list[float]:
    """Parallel-transport the incumbent roll into a nearby view direction."""

    source_view = _normalize(source_direction)
    target_view = _normalize(target_direction)
    up = _normalize(source_up)
    up = up - float(np.dot(up, source_view)) * source_view
    up = _normalize(up)
    transported = up - float(np.dot(up, target_view)) * target_view
    if float(np.linalg.norm(transported)) <= 1e-6:
        return _camera_up(target_view)
    return _normalize(transported).tolist()


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
    seed_roll = float(seed.get("roll_degrees", 0.0))
    seed_principal_u = float(seed.get("principal_point_u", 0.0))
    seed_principal_v = float(seed.get("principal_point_v", 0.0))
    seed_radial_k1 = float(seed.get("radial_distortion_k1", 0.0))
    seed_radial_k2 = float(seed.get("radial_distortion_k2", 0.0))
    seed_frame_anchor = seed.get("frame_anchor_affine")
    seed_orthographic_span = float(
        seed.get("orthographic_span_multiplier", 2.0)
    )
    projection_modes = (seed_projection_mode,)
    orthographic_span_multiplier = seed_orthographic_span
    target_u_values = (seed_target_u,)
    target_v_values = (seed_target_v,)
    roll_values = (seed_roll,)
    principal_u_values = (seed_principal_u,)
    principal_v_values = (seed_principal_v,)
    radial_k1_values = (seed_radial_k1,)
    radial_k2_values = (seed_radial_k2,)
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
    elif phase == "roll":
        angle_offsets = (0.0,)
        elevation_offsets = (0.0,)
        distances = (seed_distance,)
        focal_lengths = (seed_focal,)
        roll_values = tuple(
            sorted(
                {
                    round(float(np.clip(seed_roll + value, -15.0, 15.0)), 5)
                    for value in (-6.0, -3.0, 0.0, 3.0, 6.0)
                }
            )
        )
    elif phase == "principal_point":
        angle_offsets = (0.0,)
        elevation_offsets = (0.0,)
        distances = (seed_distance,)
        focal_lengths = (seed_focal,)
        principal_u_values = tuple(
            sorted(
                {
                    round(
                        float(np.clip(seed_principal_u + value, -0.20, 0.20)),
                        5,
                    )
                    for value in (-0.08, -0.04, 0.0, 0.04, 0.08)
                }
            )
        )
        principal_v_values = tuple(
            sorted(
                {
                    round(
                        float(np.clip(seed_principal_v + value, -0.20, 0.20)),
                        5,
                    )
                    for value in (-0.08, -0.04, 0.0, 0.04, 0.08)
                }
            )
        )
    elif phase == "radial_distortion":
        angle_offsets = (0.0,)
        elevation_offsets = (0.0,)
        distances = (seed_distance,)
        focal_lengths = (seed_focal,)
        radial_k1_values = tuple(
            sorted(
                {
                    round(
                        float(np.clip(seed_radial_k1 + value, -0.35, 0.35)),
                        5,
                    )
                    for value in (-0.16, -0.08, 0.0, 0.08, 0.16)
                }
            )
        )
        radial_k2_values = tuple(
            sorted(
                {
                    round(
                        float(np.clip(seed_radial_k2 + value, -0.20, 0.20)),
                        5,
                    )
                    for value in (-0.06, 0.0, 0.06)
                }
            )
        )
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
    elif phase == "frame_micro":
        angle_offsets = (0.0,)
        elevation_offsets = (0.0,)
        distances = (seed_distance,)
        focal_lengths = (seed_focal,)
        roll_values = tuple(
            sorted(
                {
                    round(float(np.clip(seed_roll + value, -15.0, 15.0)), 5)
                    for value in (-0.5, 0.0, 0.5)
                }
            )
        )
        principal_u_values = tuple(
            sorted(
                {
                    round(
                        float(np.clip(seed_principal_u + value, -0.20, 0.20)),
                        5,
                    )
                    for value in (-0.01, 0.0, 0.01)
                }
            )
        )
        principal_v_values = tuple(
            sorted(
                {
                    round(
                        float(np.clip(seed_principal_v + value, -0.20, 0.20)),
                        5,
                    )
                    for value in (-0.01, 0.0, 0.01)
                }
            )
        )
    elif phase == "radial_distortion_micro":
        angle_offsets = (0.0,)
        elevation_offsets = (0.0,)
        distances = (seed_distance,)
        focal_lengths = (seed_focal,)
        radial_k1_values = tuple(
            sorted(
                {
                    round(
                        float(np.clip(seed_radial_k1 + value, -0.35, 0.35)),
                        5,
                    )
                    for value in (-0.02, 0.0, 0.02)
                }
            )
        )
        radial_k2_values = tuple(
            sorted(
                {
                    round(
                        float(np.clip(seed_radial_k2 + value, -0.20, 0.20)),
                        5,
                    )
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
                            for (
                                roll,
                                principal_u,
                                principal_v,
                                radial_k1,
                                radial_k2,
                            ) in itertools.product(
                                roll_values,
                                principal_u_values,
                                principal_v_values,
                                radial_k1_values,
                                radial_k2_values,
                            ):
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
                                    and math.isclose(
                                        roll, seed_roll, abs_tol=1e-6
                                    )
                                    and math.isclose(
                                        principal_u,
                                        seed_principal_u,
                                        abs_tol=1e-6,
                                    )
                                    and math.isclose(
                                        principal_v,
                                        seed_principal_v,
                                        abs_tol=1e-6,
                                    )
                                    and math.isclose(
                                        radial_k1, seed_radial_k1, abs_tol=1e-6
                                    )
                                    and math.isclose(
                                        radial_k2, seed_radial_k2, abs_tol=1e-6
                                    )
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
                                    else (
                                        _transport_up_axis(
                                            seed["analysis_direction"],
                                            seed["analysis_up_axis"],
                                            direction,
                                        )
                                        if seed.get("analysis_up_axis") is not None
                                        else _camera_up(direction)
                                    )
                                )
                                up_axis = _roll_up_axis(
                                    direction,
                                    up_axis,
                                    0.0 if exact_seed else roll - seed_roll,
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
                                        "roll_degrees": roll,
                                        "principal_point_u": principal_u,
                                        "principal_point_v": principal_v,
                                        "radial_distortion_k1": radial_k1,
                                        "radial_distortion_k2": radial_k2,
                                        "projection_mode": projection_mode,
                                        "orthographic_span_multiplier": (
                                            orthographic_span_multiplier
                                        ),
                                        "calibration": {
                                            "reference_view_id": reference_id,
                                            "phase": phase,
                                            "frame_anchor": exact_seed,
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
                                            "roll_degrees": roll,
                                            "principal_point_u": principal_u,
                                            "principal_point_v": principal_v,
                                            "radial_distortion_k1": radial_k1,
                                            "radial_distortion_k2": radial_k2,
                                            "projection_mode": projection_mode,
                                            "orthographic_span_multiplier": (
                                                orthographic_span_multiplier
                                            ),
                                            **(
                                                {
                                                    "frame_anchor_affine": (
                                                        seed_frame_anchor
                                                    )
                                                }
                                                if seed_frame_anchor is not None
                                                else {}
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


def _compose_affines(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left3 = np.vstack((np.asarray(left, dtype=np.float64), (0.0, 0.0, 1.0)))
    right3 = np.vstack((np.asarray(right, dtype=np.float64), (0.0, 0.0, 1.0)))
    return (left3 @ right3)[:2].astype(np.float32)


def _constrained_frame_projection(
    reference_mask: np.ndarray,
    render_mask: np.ndarray,
    *,
    anchor_affine: np.ndarray,
) -> dict[str, Any]:
    """Fit only a small residual around one shared physical-camera frame.

    ``anchor_affine`` is bootstrapped once from the incumbent camera in the
    current render batch.  Every competing camera uses that same mapping.
    This prevents a fresh bbox fit from independently erasing the physical
    effect of focal length, distance, roll, or principal-point candidates.
    """

    height, width = reference_mask.shape
    output_size = (width, height)
    anchored = cv2.warpAffine(
        render_mask,
        np.asarray(anchor_affine, dtype=np.float32),
        output_size,
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    resize_factor = min(
        1.0,
        FRAME_RESIDUAL_OPTIMIZATION_MAXIMUM_SIDE / max(height, width),
    )
    small_size = (
        max(1, int(round(width * resize_factor))),
        max(1, int(round(height * resize_factor))),
    )
    small_reference = cv2.resize(
        reference_mask,
        small_size,
        interpolation=cv2.INTER_NEAREST,
    )
    small_anchored = cv2.resize(
        anchored,
        small_size,
        interpolation=cv2.INTER_NEAREST,
    )
    target = small_reference > 0
    target_edge = cv2.morphologyEx(
        target.astype(np.uint8),
        cv2.MORPH_GRADIENT,
        np.ones((3, 3), np.uint8),
    )
    target_distance = cv2.distanceTransform(
        1 - target_edge,
        cv2.DIST_L2,
        3,
    )
    center = (0.5 * (small_size[0] - 1), 0.5 * (small_size[1] - 1))

    def evaluate(values: Sequence[float]) -> tuple[float, dict[str, float], np.ndarray]:
        scale, rotation, tx_ratio, ty_ratio = (float(value) for value in values)
        residual = cv2.getRotationMatrix2D(center, rotation, scale).astype(np.float32)
        residual[0, 2] += tx_ratio * small_size[0]
        residual[1, 2] += ty_ratio * small_size[1]
        registered = cv2.warpAffine(
            small_anchored,
            residual,
            small_size,
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        selected = registered > 0
        intersection = int(np.count_nonzero(target & selected))
        union = int(np.count_nonzero(target | selected))
        iou = intersection / max(1, union)
        edge = cv2.morphologyEx(
            selected.astype(np.uint8),
            cv2.MORPH_GRADIENT,
            np.ones((3, 3), np.uint8),
        )
        if np.any(edge) and np.any(target_edge):
            selected_distance = cv2.distanceTransform(
                1 - edge,
                cv2.DIST_L2,
                3,
            )
            distances = np.concatenate(
                (target_distance[edge > 0], selected_distance[target_edge > 0])
            )
            boundary_mean = float(np.mean(distances))
            boundary_p95 = float(np.percentile(distances, 95))
        else:
            boundary_mean = boundary_p95 = float(max(small_size))
        diagonal = max(1.0, math.hypot(*small_size))
        residual_prior = (
            abs(math.log(max(scale, 1e-9))) / math.log(FRAME_RESIDUAL_MAXIMUM_SCALE)
            + abs(rotation) / FRAME_RESIDUAL_MAXIMUM_ROTATION_DEGREES
            + abs(tx_ratio) / FRAME_RESIDUAL_MAXIMUM_TRANSLATION_RATIO
            + abs(ty_ratio) / FRAME_RESIDUAL_MAXIMUM_TRANSLATION_RATIO
        ) / 4.0
        loss = (
            0.40 * (1.0 - iou)
            + 0.25 * min(1.0, boundary_mean / (0.025 * diagonal))
            + 0.25 * min(1.0, boundary_p95 / (0.06 * diagonal))
            + 0.10 * residual_prior
        )
        return (
            float(loss),
            {
                "iou": iou,
                "boundary_mean_px": boundary_mean / resize_factor,
                "boundary_p95_px": boundary_p95 / resize_factor,
                "residual_prior": residual_prior,
            },
            residual,
        )

    centroid_delta = np.asarray((0.0, 0.0), dtype=np.float64)
    anchored_moments = cv2.moments((small_anchored > 0).astype(np.uint8), True)
    target_moments = cv2.moments(target.astype(np.uint8), True)
    if anchored_moments["m00"] > 0.0 and target_moments["m00"] > 0.0:
        centroid_delta = np.asarray(
            (
                target_moments["m10"] / target_moments["m00"]
                - anchored_moments["m10"] / anchored_moments["m00"],
                target_moments["m01"] / target_moments["m00"]
                - anchored_moments["m01"] / anchored_moments["m00"],
            )
        )
    centroid_seed = (
        1.0,
        0.0,
        float(
            np.clip(
                centroid_delta[0] / max(1, small_size[0]),
                -FRAME_RESIDUAL_MAXIMUM_TRANSLATION_RATIO,
                FRAME_RESIDUAL_MAXIMUM_TRANSLATION_RATIO,
            )
        ),
        float(
            np.clip(
                centroid_delta[1] / max(1, small_size[1]),
                -FRAME_RESIDUAL_MAXIMUM_TRANSLATION_RATIO,
                FRAME_RESIDUAL_MAXIMUM_TRANSLATION_RATIO,
            )
        ),
    )
    seeds = [
        (1.0, 0.0, 0.0, 0.0),
        centroid_seed,
        (FRAME_RESIDUAL_MINIMUM_SCALE, 0.0, centroid_seed[2], centroid_seed[3]),
        (FRAME_RESIDUAL_MAXIMUM_SCALE, 0.0, centroid_seed[2], centroid_seed[3]),
    ]
    bounds = (
        (FRAME_RESIDUAL_MINIMUM_SCALE, FRAME_RESIDUAL_MAXIMUM_SCALE),
        (
            -FRAME_RESIDUAL_MAXIMUM_ROTATION_DEGREES,
            FRAME_RESIDUAL_MAXIMUM_ROTATION_DEGREES,
        ),
        (
            -FRAME_RESIDUAL_MAXIMUM_TRANSLATION_RATIO,
            FRAME_RESIDUAL_MAXIMUM_TRANSLATION_RATIO,
        ),
        (
            -FRAME_RESIDUAL_MAXIMUM_TRANSLATION_RATIO,
            FRAME_RESIDUAL_MAXIMUM_TRANSLATION_RATIO,
        ),
    )
    initial_steps = (0.015, 1.0, 0.02, 0.02)
    best: tuple[float, tuple[float, ...], dict[str, float], np.ndarray] | None = None
    evaluation_count = 0
    for seed in seeds:
        values = [float(value) for value in seed]
        loss, metrics, matrix = evaluate(values)
        evaluation_count += 1
        local = (loss, tuple(values), metrics, matrix)
        steps = list(initial_steps)
        for _level in range(4):
            changed = True
            while changed:
                changed = False
                for index, step in enumerate(steps):
                    for direction in (-1.0, 1.0):
                        candidate = list(local[1])
                        candidate[index] = float(
                            np.clip(
                                candidate[index] + direction * step,
                                bounds[index][0],
                                bounds[index][1],
                            )
                        )
                        candidate_loss, candidate_metrics, candidate_matrix = evaluate(
                            candidate
                        )
                        evaluation_count += 1
                        proposed = (
                            candidate_loss,
                            tuple(candidate),
                            candidate_metrics,
                            candidate_matrix,
                        )
                        if (proposed[0], proposed[1]) < (local[0], local[1]):
                            local = proposed
                            changed = True
            steps = [step * 0.5 for step in steps]
        if best is None or (local[0], local[1]) < (best[0], best[1]):
            best = local
    if best is None:
        raise RuntimeError("Constrained frame optimizer produced no candidate")
    scale, rotation, tx_ratio, ty_ratio = best[1]
    full_residual = cv2.getRotationMatrix2D(
        (0.5 * (width - 1), 0.5 * (height - 1)),
        rotation,
        scale,
    ).astype(np.float32)
    full_residual[0, 2] += tx_ratio * width
    full_residual[1, 2] += ty_ratio * height
    total = _compose_affines(full_residual, anchor_affine)
    registered = cv2.warpAffine(
        render_mask,
        total,
        output_size,
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    target_selected = reference_mask > 0
    registered_selected = registered > 0
    intersection = int(np.count_nonzero(target_selected & registered_selected))
    union = int(np.count_nonzero(target_selected | registered_selected))
    full_iou = intersection / max(1, union)
    return {
        "bbox_affine": [
            [round(float(value), 10) for value in row] for row in total.tolist()
        ],
        "ecc_warp": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        "global_similarity_affine": [
            [round(float(value), 10) for value in row] for row in total.tolist()
        ],
        "projection_iou_before": round(
            float(
                np.count_nonzero(target_selected & (anchored > 0))
                / max(1, np.count_nonzero(target_selected | (anchored > 0)))
            ),
            8,
        ),
        "projection_iou": round(float(full_iou), 8),
        "ecc_status": "constrained_multi_start_coordinate_descent",
        "ecc_correlation": 0.0,
        "ecc_transform_audit": {
            "registration_mode": "shared_anchor_plus_bounded_frame_residual",
            "anchor_affine": [
                [round(float(value), 10) for value in row]
                for row in np.asarray(anchor_affine).tolist()
            ],
            "residual_scale": round(scale, 8),
            "residual_rotation_degrees": round(rotation, 8),
            "residual_translation_ratio_xy": [
                round(tx_ratio, 8),
                round(ty_ratio, 8),
            ],
            "residual_prior": round(float(best[2]["residual_prior"]), 8),
            "optimizer_loss": round(float(best[0]), 8),
            "optimizer_evaluation_count": evaluation_count,
            "constraints_passed": True,
            "constraint_failures": [],
        },
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


def _assembly_subtree_path(prim_path: Any) -> str | None:
    """Return a stable assembly subtree without asset-specific names.

    The leaf mesh and its immediate parent identify a Part, not an assembly.
    Clustering at the grandparent groups siblings that share a local rigid or
    articulated transform while leaving unrelated branches independent.
    """

    if not isinstance(prim_path, str) or not prim_path.startswith("/"):
        return None
    segments = [segment for segment in prim_path.split("/") if segment]
    if len(segments) < 3:
        return "/" + "/".join(segments[:-1]) if len(segments) > 1 else None
    return "/" + "/".join(segments[:-2])


def _robust_part_consensus(
    *,
    ids: np.ndarray,
    parts: Sequence[Mapping[str, Any]],
    affine: np.ndarray,
    reference_image: np.ndarray,
    reference_mask: np.ndarray,
    fixed_anchor_part_ids: Sequence[str] = (),
) -> dict[str, Any]:
    """Estimate the rigid camera consensus and isolate assembly outliers.

    Each visible Part-ID contributes one observation built from its external
    boundary distance to the photographed object and the fraction projected
    inside the human foreground.  A median/MAD M-estimator finds the dominant
    rigid population without any asset names, hand-authored movable-part list,
    or per-Part transform.  Size strata and capped pixel weights prevent both
    one enclosure and a cloud of fasteners from becoming the sole authority.
    """

    if ids.ndim != 3 or ids.shape[2] < 3:
        raise ValueError("Rigid consensus requires a three-channel Part-ID image")
    reference_edges = _reference_structure_edges(reference_image, reference_mask)
    if not np.any(reference_edges > 0):
        return _empty_rigid_consensus("reference_has_no_edges")
    reference_distance, reference_labels = cv2.distanceTransformWithLabels(
        (reference_edges == 0).astype(np.uint8),
        cv2.DIST_L2,
        3,
        labelType=cv2.DIST_LABEL_PIXEL,
    )
    # DIST_LABEL_PIXEL numbers the zero-valued reference edge pixels in
    # row-major order.  Retaining those coordinates gives every observable
    # Part a signed 2-D residual vector, which is required to distinguish a
    # coherent local assembly displacement from unrelated silhouette noise.
    reference_edge_y, reference_edge_x = np.nonzero(reference_edges > 0)
    target = reference_mask > 0
    height, width = reference_mask.shape
    diagonal = max(1.0, math.hypot(width, height))
    support = cv2.dilate(
        target.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)),
        iterations=1,
    ) > 0
    packed_ids = (
        ids[:, :, 0].astype(np.uint32)
        | (ids[:, :, 1].astype(np.uint32) << 8)
        | (ids[:, :, 2].astype(np.uint32) << 16)
    )
    visible_values, visible_counts = np.unique(packed_ids, return_counts=True)
    count_by_value = {
        int(value): int(count) for value, count in zip(visible_values, visible_counts)
    }
    observations: list[dict[str, Any]] = []
    for part in parts:
        if not isinstance(part, Mapping) or not isinstance(part.get("part_id"), str):
            continue
        part_id = str(part["part_id"])
        red, green, blue = _part_color(part_id)
        packed = int(blue) | (int(green) << 8) | (int(red) << 16)
        raw_pixels = count_by_value.get(packed, 0)
        if raw_pixels < RIGID_CONSENSUS_MINIMUM_PART_PIXELS:
            continue
        projected = cv2.warpAffine(
            (packed_ids == packed).astype(np.uint8),
            np.asarray(affine, dtype=np.float32),
            (width, height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        ) > 0
        projected_pixels = int(np.count_nonzero(projected))
        if projected_pixels < RIGID_CONSENSUS_MINIMUM_PART_PIXELS:
            continue
        support_ratio = float(np.count_nonzero(projected & support)) / projected_pixels
        # Geometry with no photographed support is a strong assembly-state or
        # version-mismatch observation.  Keep it in the robust population: the
        # median/MAD estimator will reject it, while omitting it would hide the
        # most important detached-assembly evidence.
        overlap_pixels = int(np.count_nonzero(projected & target))
        inside_ratio = float(overlap_pixels / projected_pixels)
        boundary = cv2.morphologyEx(
            projected.astype(np.uint8),
            cv2.MORPH_GRADIENT,
            np.ones((3, 3), dtype=np.uint8),
        )
        distances = reference_distance[boundary > 0]
        if len(distances) < 8:
            continue
        if projected_pixels < 96:
            stratum = "small"
        elif projected_pixels < 768:
            stratum = "medium"
        else:
            stratum = "large"
        median_boundary = float(np.median(distances))
        p75_boundary = float(np.percentile(distances, 75))
        edge_support_px = max(3.0, 0.006 * diagonal)
        edge_support_ratio = float(np.count_nonzero(distances <= edge_support_px)) / len(
            distances
        )
        boundary_y, boundary_x = np.nonzero(boundary > 0)
        nearest_labels = reference_labels[boundary_y, boundary_x].astype(np.int64)
        valid_labels = (nearest_labels > 0) & (
            nearest_labels <= len(reference_edge_x)
        )
        if np.any(valid_labels):
            nearest_indexes = nearest_labels[valid_labels] - 1
            residual_dx = float(
                np.median(
                    reference_edge_x[nearest_indexes]
                    - boundary_x[valid_labels]
                )
            )
            residual_dy = float(
                np.median(
                    reference_edge_y[nearest_indexes]
                    - boundary_y[valid_labels]
                )
            )
        else:
            residual_dx = 0.0
            residual_dy = 0.0
        # A zero-overlap Part is a decisive outlier even when it lies close to
        # a different photographed edge.  The bounded inside penalty keeps the
        # residual in pixel units and avoids large Part area domination.
        residual_px = median_boundary + min(
            0.04 * diagonal,
            (1.0 - inside_ratio) * 0.04 * diagonal,
        )
        observations.append(
            {
                "part_id": part_id,
                "prim_path": part.get("prim_path"),
                "assembly_subtree": _assembly_subtree_path(part.get("prim_path")),
                "stratum": stratum,
                "projected_pixels": projected_pixels,
                "inside_reference_ratio": inside_ratio,
                "support_ratio": support_ratio,
                "edge_support_ratio": edge_support_ratio,
                "boundary_median_px": median_boundary,
                "boundary_p75_px": p75_boundary,
                "residual_vector_px": [residual_dx, residual_dy],
                "residual_px": residual_px,
            }
        )
    fixed_anchor = _fixed_rigid_anchor_metrics(
        observations=observations,
        expected_part_ids=fixed_anchor_part_ids,
        image_shape=reference_mask.shape,
    )
    if len(observations) < RIGID_CONSENSUS_MINIMUM_INLIER_PARTS:
        return {
            **_empty_rigid_consensus("insufficient_visible_parts", observations),
            **fixed_anchor,
        }

    # Cap each size stratum deterministically, then fit location/scale with a
    # median and MAD.  This is the robust IRLS seed; no asset-specific label is
    # allowed to affect membership.
    selected: list[dict[str, Any]] = []
    for stratum in ("small", "medium", "large"):
        values = [item for item in observations if item["stratum"] == stratum]
        values.sort(
            key=lambda item: (
                -min(int(item["projected_pixels"]), 768),
                str(item["part_id"]),
            )
        )
        by_support = sorted(
            values,
            key=lambda item: (
                -float(item["support_ratio"]),
                -min(int(item["projected_pixels"]), 768),
                str(item["part_id"]),
            ),
        )
        # Half of each bounded stratum preserves large silhouette authority;
        # the other half preserves well-observed fixed geometry.  This keeps
        # either a large articulated branch or many tiny fasteners from
        # crowding the consensus sample.
        selected_by_id: dict[str, dict[str, Any]] = {}
        half = RIGID_CONSENSUS_MAXIMUM_PARTS_PER_STRATUM // 2
        for item in values[:half] + by_support:
            selected_by_id.setdefault(str(item["part_id"]), item)
            if len(selected_by_id) >= RIGID_CONSENSUS_MAXIMUM_PARTS_PER_STRATUM:
                break
        selected.extend(selected_by_id.values())
    # A CAD Part wholly inside the foreground but lacking a photographic edge
    # is not observable enough to vote on camera pose.  It remains in the
    # audit as an indeterminate observation, but does not become an outlier.
    # Conversely, a Part that leaves the foreground is observable through the
    # object silhouette even when it has no internal texture edge.
    eligible = [
        item
        for item in selected
        if float(item["edge_support_ratio"])
        >= RIGID_CONSENSUS_MINIMUM_EDGE_SUPPORT
        or float(item["inside_reference_ratio"])
        < RIGID_CONSENSUS_MAXIMUM_INSIDE_RATIO_FOR_SILHOUETTE_EVIDENCE
    ]
    for item in selected:
        item["consensus_observable"] = item in eligible
        item["robust_weight"] = 0.0
        item["rigid_inlier"] = None
    if len(eligible) < RIGID_CONSENSUS_MINIMUM_INLIER_PARTS:
        return {
            **_empty_rigid_consensus(
                "insufficient_observable_parts",
                observations,
                selected=selected,
            ),
            **fixed_anchor,
        }
    residuals = np.asarray([float(item["residual_px"]) for item in eligible])
    center = float(np.median(residuals))
    mad = float(np.median(np.abs(residuals - center)))
    robust_sigma = max(0.75, 1.4826 * mad)
    inlier_cutoff = center + RIGID_CONSENSUS_OUTLIER_CUTOFF * robust_sigma
    huber_cutoff = max(1e-12, RIGID_CONSENSUS_HUBER_CUTOFF * robust_sigma)
    for item in eligible:
        deviation = max(0.0, float(item["residual_px"]) - center)
        item["robust_weight"] = float(
            1.0 if deviation <= huber_cutoff else huber_cutoff / deviation
        )
        item["rigid_inlier"] = bool(float(item["residual_px"]) <= inlier_cutoff)

    inliers = [item for item in eligible if item["rigid_inlier"]]
    outliers = [item for item in eligible if not item["rigid_inlier"]]
    indeterminate = [item for item in selected if not item["consensus_observable"]]
    total_pixels = sum(int(item["projected_pixels"]) for item in eligible)
    inlier_pixels = sum(int(item["projected_pixels"]) for item in inliers)
    inlier_coverage = float(inlier_pixels / max(1, total_pixels))
    if (
        len(inliers) < RIGID_CONSENSUS_MINIMUM_INLIER_PARTS
        or inlier_coverage < RIGID_CONSENSUS_MINIMUM_INLIER_COVERAGE
    ):
        return {
            **_empty_rigid_consensus(
                "insufficient_rigid_consensus",
                observations,
                center=center,
                sigma=robust_sigma,
                selected=selected,
            ),
            **fixed_anchor,
        }

    def balanced_value(key: str) -> float:
        stratum_values: list[float] = []
        for stratum in ("small", "medium", "large"):
            values = [item for item in inliers if item["stratum"] == stratum]
            if not values:
                continue
            weights = np.asarray(
                [
                    float(item["robust_weight"])
                    * math.sqrt(min(int(item["projected_pixels"]), 768))
                    for item in values
                ],
                dtype=np.float64,
            )
            observations_for_key = np.asarray(
                [float(item[key]) for item in values], dtype=np.float64
            )
            stratum_values.append(
                float(np.average(observations_for_key, weights=weights))
            )
        return float(np.mean(stratum_values)) if stratum_values else 0.0

    residual_px = balanced_value("residual_px")
    inside_ratio = balanced_value("inside_reference_ratio")
    decay = max(3.0, 0.015 * diagonal)
    consensus_score = float(math.exp(-residual_px / decay) * inside_ratio)
    clusters: dict[str, list[dict[str, Any]]] = {}
    for item in outliers:
        cluster = item.get("assembly_subtree") or f"part:{item['part_id']}"
        clusters.setdefault(str(cluster), []).append(item)
    assembly_clusters = []
    for cluster, items in clusters.items():
        residual_vectors = np.asarray(
            [item["residual_vector_px"] for item in items], dtype=np.float64
        )
        median_vector = np.median(residual_vectors, axis=0)
        median_vector_norm = float(np.linalg.norm(median_vector))
        if len(items) >= 2 and median_vector_norm > 1e-6:
            unit_vector = median_vector / median_vector_norm
            directions = residual_vectors / np.maximum(
                1e-6, np.linalg.norm(residual_vectors, axis=1, keepdims=True)
            )
            direction_coherence = float(
                np.mean(np.maximum(0.0, directions @ unit_vector))
            )
        else:
            direction_coherence = 0.0
        coherent_assembly = len(items) >= 2 and direction_coherence >= 0.7
        assembly_clusters.append(
            {
                "assembly_subtree": cluster,
                "part_ids": sorted(str(item["part_id"]) for item in items),
                "part_count": len(items),
                "projected_pixels": sum(
                    int(item["projected_pixels"]) for item in items
                ),
                "median_residual_px": float(
                    np.median([float(item["residual_px"]) for item in items])
                ),
                "minimum_inside_reference_ratio": min(
                    float(item["inside_reference_ratio"]) for item in items
                ),
                "median_residual_vector_px": [
                    float(median_vector[0]),
                    float(median_vector[1]),
                ],
                "residual_direction_coherence": direction_coherence,
                "classification": (
                    "assembly_state_or_geometry_mismatch"
                    if coherent_assembly
                    else "multi_part_residual_without_coherent_direction"
                    if len(items) >= 2
                    else "isolated_part_geometry_or_visibility_mismatch"
                ),
            }
        )
    assembly_clusters.sort(
        key=lambda item: (
            -int(item["projected_pixels"]),
            str(item["assembly_subtree"]),
        )
    )
    return {
        "rigid_consensus_valid": True,
        "rigid_consensus_reason": "robust_part_consensus",
        "rigid_consensus_score": consensus_score,
        "rigid_consensus_part_count": len(inliers),
        "rigid_consensus_candidate_part_count": len(eligible),
        "rigid_consensus_indeterminate_part_count": len(indeterminate),
        "rigid_consensus_outlier_part_count": len(outliers),
        "rigid_consensus_inlier_ratio": float(len(inliers) / len(eligible)),
        "rigid_consensus_pixel_coverage": inlier_coverage,
        "rigid_consensus_residual_px": residual_px,
        "rigid_consensus_inside_reference_ratio": inside_ratio,
        "rigid_consensus_center_px": center,
        "rigid_consensus_sigma_px": robust_sigma,
        "rigid_consensus_inlier_cutoff_px": inlier_cutoff,
        "rigid_consensus_inlier_part_ids": sorted(
            str(item["part_id"]) for item in inliers
        ),
        "rigid_consensus_outlier_part_ids": sorted(
            str(item["part_id"]) for item in outliers
        ),
        "rigid_consensus_indeterminate_part_ids": sorted(
            str(item["part_id"]) for item in indeterminate
        ),
        "rigid_consensus_size_strata": {
            stratum: {
                "candidate_part_count": sum(
                    item["stratum"] == stratum for item in eligible
                ),
                "inlier_part_count": sum(
                    item["stratum"] == stratum for item in inliers
                ),
            }
            for stratum in ("small", "medium", "large")
        },
        "rigid_consensus_part_residuals": sorted(
            [
                {
                    **item,
                    "inside_reference_ratio": round(
                        float(item["inside_reference_ratio"]), 8
                    ),
                    "support_ratio": round(float(item["support_ratio"]), 8),
                    "edge_support_ratio": round(
                        float(item["edge_support_ratio"]), 8
                    ),
                    "boundary_median_px": round(
                        float(item["boundary_median_px"]), 8
                    ),
                    "boundary_p75_px": round(
                        float(item["boundary_p75_px"]), 8
                    ),
                    "residual_vector_px": [
                        round(float(value), 8)
                        for value in item["residual_vector_px"]
                    ],
                    "residual_px": round(float(item["residual_px"]), 8),
                    "robust_weight": round(float(item["robust_weight"]), 8),
                }
                for item in selected
            ],
            key=lambda item: (-float(item["residual_px"]), str(item["part_id"])),
        ),
        "assembly_residual_clusters": assembly_clusters,
        **fixed_anchor,
    }


def _fixed_rigid_anchor_metrics(
    *,
    observations: Sequence[Mapping[str, Any]],
    expected_part_ids: Sequence[str],
    image_shape: Sequence[int],
) -> dict[str, Any]:
    """Score one sealed Part-ID set without candidate-wise reselection."""

    raw_ids = [str(value) for value in expected_part_ids]
    if not raw_ids:
        return {
            "fixed_anchor_enabled": False,
            "fixed_anchor_valid": False,
            "fixed_anchor_expected_part_ids": [],
            "fixed_anchor_observed_part_ids": [],
            "fixed_anchor_missing_part_ids": [],
            "fixed_anchor_expected_part_count": 0,
            "fixed_anchor_observed_part_count": 0,
            "fixed_anchor_coverage": 0.0,
            "fixed_anchor_residual_px": None,
            "fixed_anchor_inside_reference_ratio": 0.0,
            "fixed_anchor_score": 0.0,
            "fixed_anchor_size_strata": {},
        }
    if raw_ids != sorted(set(raw_ids)) or any(not value for value in raw_ids):
        raise ValueError("Fixed rigid anchor Part IDs must be sorted and unique")
    by_id = {
        str(item["part_id"]): item
        for item in observations
        if isinstance(item, Mapping) and isinstance(item.get("part_id"), str)
    }
    selected = [by_id[part_id] for part_id in raw_ids if part_id in by_id]
    missing = [part_id for part_id in raw_ids if part_id not in by_id]
    coverage = float(len(selected) / len(raw_ids))
    residuals: list[float] = []
    inside_ratios: list[float] = []
    strata_audit: dict[str, Any] = {}
    for stratum in ("small", "medium", "large"):
        items = [item for item in selected if item.get("stratum") == stratum]
        if not items:
            strata_audit[stratum] = {"part_count": 0}
            continue
        weights = np.asarray(
            [math.sqrt(min(int(item["projected_pixels"]), 768)) for item in items],
            dtype=np.float64,
        )
        residual = float(
            np.average(
                np.asarray([float(item["residual_px"]) for item in items]),
                weights=weights,
            )
        )
        inside = float(
            np.average(
                np.asarray(
                    [float(item["inside_reference_ratio"]) for item in items]
                ),
                weights=weights,
            )
        )
        residuals.append(residual)
        inside_ratios.append(inside)
        strata_audit[stratum] = {
            "part_count": len(items),
            "residual_px": round(residual, 8),
            "inside_reference_ratio": round(inside, 8),
        }
    residual_px = float(np.mean(residuals)) if residuals else None
    inside_ratio = float(np.mean(inside_ratios)) if inside_ratios else 0.0
    diagonal = max(1.0, math.hypot(int(image_shape[1]), int(image_shape[0])))
    decay = max(3.0, 0.015 * diagonal)
    score = (
        math.exp(-residual_px / decay) * inside_ratio * coverage
        if residual_px is not None
        else 0.0
    )
    minimum_observed = max(3, math.ceil(0.60 * len(raw_ids)))
    return {
        "fixed_anchor_enabled": True,
        "fixed_anchor_valid": len(raw_ids) >= 3 and len(selected) >= minimum_observed,
        "fixed_anchor_expected_part_ids": raw_ids,
        "fixed_anchor_observed_part_ids": sorted(
            str(item["part_id"]) for item in selected
        ),
        "fixed_anchor_missing_part_ids": missing,
        "fixed_anchor_expected_part_count": len(raw_ids),
        "fixed_anchor_observed_part_count": len(selected),
        "fixed_anchor_coverage": round(coverage, 8),
        "fixed_anchor_residual_px": (
            round(residual_px, 8) if residual_px is not None else None
        ),
        "fixed_anchor_inside_reference_ratio": round(inside_ratio, 8),
        "fixed_anchor_score": round(float(score), 8),
        "fixed_anchor_size_strata": strata_audit,
    }


def _classify_multiview_residuals(
    views: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Fuse per-view residuals into asset-independent Part diagnoses.

    A Part that is rejected in several independently calibrated views is
    unlikely to be a camera-only failure.  A one-view rejection remains
    ambiguous because self-occlusion, segmentation, and view-local geometry
    can produce the same 2-D pattern.
    """

    observations: dict[str, list[dict[str, Any]]] = {}
    for reference_id, score in views.items():
        if not isinstance(score, Mapping):
            continue
        inliers = {
            str(value) for value in score.get("rigid_consensus_inlier_part_ids", [])
        }
        for raw in score.get("rigid_consensus_part_residuals", []):
            if not isinstance(raw, Mapping) or not isinstance(raw.get("part_id"), str):
                continue
            observations.setdefault(str(raw["part_id"]), []).append(
                {
                    "reference_view_id": str(reference_id),
                    "consensus_observable": bool(
                        raw.get("consensus_observable", True)
                    ),
                    "rigid_inlier": (
                        str(raw["part_id"]) in inliers
                        if bool(raw.get("consensus_observable", True))
                        else None
                    ),
                    "residual_px": raw.get("residual_px"),
                    "inside_reference_ratio": raw.get("inside_reference_ratio"),
                    "assembly_subtree": raw.get("assembly_subtree"),
                }
            )
    part_diagnoses: list[dict[str, Any]] = []
    for part_id, items in observations.items():
        observable = [item for item in items if item["consensus_observable"]]
        rejected = [item for item in observable if item["rigid_inlier"] is False]
        visible_view_count = len(observable)
        indeterminate_view_count = len(items) - visible_view_count
        rejected_view_count = len(rejected)
        if rejected_view_count >= 2:
            classification = "persistent_assembly_or_geometry_mismatch"
        elif rejected_view_count == 1 and visible_view_count >= 2:
            classification = "view_local_occlusion_geometry_or_mask_mismatch"
        elif rejected_view_count == 1:
            classification = "single_view_unresolved"
        elif visible_view_count:
            classification = "rigid_consensus_inlier"
        else:
            classification = "insufficient_observable_edge_evidence"
        part_diagnoses.append(
            {
                "part_id": part_id,
                "assembly_subtree": next(
                    (
                        item["assembly_subtree"]
                        for item in items
                        if item.get("assembly_subtree") is not None
                    ),
                    None,
                ),
                "visible_view_count": visible_view_count,
                "indeterminate_view_count": indeterminate_view_count,
                "rejected_view_count": rejected_view_count,
                "classification": classification,
                "views": sorted(items, key=lambda item: item["reference_view_id"]),
            }
        )
    part_diagnoses.sort(
        key=lambda item: (
            -int(item["rejected_view_count"]),
            -int(item["visible_view_count"]),
            str(item["part_id"]),
        )
    )
    return {
        "schema_version": "qwen-camera-multiview-rigid-consensus/v1",
        "reference_view_count": len(views),
        "part_diagnoses": part_diagnoses,
        "persistent_mismatch_part_ids": [
            item["part_id"]
            for item in part_diagnoses
            if item["classification"]
            == "persistent_assembly_or_geometry_mismatch"
        ],
        "view_local_mismatch_part_ids": [
            item["part_id"]
            for item in part_diagnoses
            if item["classification"]
            == "view_local_occlusion_geometry_or_mask_mismatch"
        ],
    }


def _empty_rigid_consensus(
    reason: str,
    observations: Sequence[Mapping[str, Any]] = (),
    *,
    center: float | None = None,
    sigma: float | None = None,
    selected: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    return {
        "rigid_consensus_valid": False,
        "rigid_consensus_reason": reason,
        "rigid_consensus_score": 0.0,
        "rigid_consensus_part_count": 0,
        "rigid_consensus_candidate_part_count": len(selected or observations),
        "rigid_consensus_indeterminate_part_count": 0,
        "rigid_consensus_outlier_part_count": 0,
        "rigid_consensus_inlier_ratio": 0.0,
        "rigid_consensus_pixel_coverage": 0.0,
        "rigid_consensus_residual_px": None,
        "rigid_consensus_inside_reference_ratio": 0.0,
        "rigid_consensus_center_px": center,
        "rigid_consensus_sigma_px": sigma,
        "rigid_consensus_inlier_cutoff_px": None,
        "rigid_consensus_inlier_part_ids": [],
        "rigid_consensus_outlier_part_ids": [],
        "rigid_consensus_indeterminate_part_ids": [],
        "rigid_consensus_size_strata": {},
        "rigid_consensus_part_residuals": [],
        "assembly_residual_clusters": [],
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


def _score_candidates(
    *,
    reference_id: str,
    reference_mask: np.ndarray,
    reference_image: np.ndarray,
    registry_path: Path,
    fixed_anchor_part_ids: Sequence[str] = (),
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
    scored_views: list[tuple[Mapping[str, Any], np.ndarray, np.ndarray]] = []
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
        foreground = _deterministic_part_id_foreground(ids, colors)
        scored_views.append((view, ids, foreground))
    if not scored_views:
        raise ValueError(f"No calibration candidates found for {reference_id}")
    bound_anchor_values = [
        item[0].get("camera_calibration", {}).get("frame_anchor_affine")
        for item in scored_views
        if item[0].get("camera_calibration", {}).get("frame_anchor_affine")
        is not None
    ]
    if bound_anchor_values:
        if len(bound_anchor_values) != len(scored_views):
            raise ValueError(
                "Camera candidate batch mixes bound and unbound frame anchors"
            )
        canonical_bound_anchors = {
            json.dumps(value, sort_keys=True, separators=(",", ":"))
            for value in bound_anchor_values
        }
        if len(canonical_bound_anchors) != 1:
            raise ValueError("Camera candidate batch has inconsistent frame anchors")
        anchor_affine = np.asarray(bound_anchor_values[0], dtype=np.float32)
        if anchor_affine.shape != (2, 3) or not np.isfinite(anchor_affine).all():
            raise ValueError("Bound camera frame anchor must be a finite 2x3 affine")
    else:
        anchors = [
            item
            for item in scored_views
            if item[0].get("camera_calibration", {}).get("frame_anchor") is True
        ]
        if len(anchors) != 1:
            # Legacy/custom batches without an explicit anchor retain one
            # deterministic bootstrap: the smallest view ID.  The fit is shared
            # by every other camera and therefore cannot rank candidates by giving
            # each one an independent image-space transform.
            anchors = [
                min(scored_views, key=lambda item: str(item[0].get("view_id")))
            ]
        anchor_projection = _refine_projection(
            reference_mask,
            anchors[0][2],
            DEFAULT_POLICY,
        )
        anchor_affine = np.asarray(
            anchor_projection["bbox_affine"], dtype=np.float32
        )
    for view, ids, foreground in scored_views:
        calibration = view.get("camera_calibration")
        if not isinstance(calibration, dict):
            raise ValueError(
                f"Calibration candidate {view.get('view_id')!r} has no metadata"
            )
        projection = _constrained_frame_projection(
            reference_mask,
            foreground,
            anchor_affine=anchor_affine,
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
        rigid_consensus = _robust_part_consensus(
            ids=ids,
            parts=parts,
            affine=matrix,
            reference_image=reference_image,
            reference_mask=reference_mask,
            fixed_anchor_part_ids=fixed_anchor_part_ids,
        )
        iou = float(projection["projection_iou"])
        diagonal = math.hypot(reference_mask.shape[1], reference_mask.shape[0])
        boundary_decay = max(3.0, 0.02 * diagonal)
        boundary_score = math.exp(-float(boundary["boundary_p95_px"]) / boundary_decay)
        # Hierarchical objective: the confirmed foreground first has to be
        # covered, then the outer boundary and equal-weight size strata select
        # the camera.  IoU remains bounded evidence but can no longer let one
        # large enclosure hide small-part displacement.
        if fixed_anchor_part_ids:
            # This anchor set was sealed from independent multi-view baseline
            # evidence.  It is the pose authority; whole-object silhouette and
            # boundary remain non-regression evidence at final selection.
            score = (
                0.08 * iou
                + 0.08 * float(coverage["target_recall"])
                + 0.07 * float(coverage["rendered_precision"])
                + 0.10 * boundary_score
                + 0.08 * float(structure["structure_score"])
                + 0.07
                * float(component_coverage["reference_component_macro_recall"])
                + 0.03
                * float(component_coverage["reference_component_min_recall"])
                + 0.08
                * float(spatial_coverage["reference_spatial_macro_recall"])
                + 0.04 * float(spatial_coverage["reference_spatial_min_recall"])
                + 0.37 * float(rigid_consensus["fixed_anchor_score"])
            )
        elif bool(rigid_consensus["rigid_consensus_valid"]):
            # Global silhouette terms remain bounded evidence, but the robust
            # Part-ID consensus is the plurality authority for the physical
            # camera.  Outlier attachments are audited rather than allowed to
            # pull the camera away from the rigid population.
            score = (
                0.10 * iou
                + 0.10 * float(coverage["target_recall"])
                + 0.07 * float(coverage["rendered_precision"])
                + 0.10 * boundary_score
                + 0.12 * float(structure["structure_score"])
                + 0.08
                * float(component_coverage["reference_component_macro_recall"])
                + 0.03
                * float(component_coverage["reference_component_min_recall"])
                + 0.10
                * float(spatial_coverage["reference_spatial_macro_recall"])
                + 0.05 * float(spatial_coverage["reference_spatial_min_recall"])
                + 0.25 * float(rigid_consensus["rigid_consensus_score"])
            )
        else:
            # Sparse assets without enough visible Parts retain the Phase-1
            # objective instead of manufacturing a weak consensus.
            score = (
                0.12 * iou
                + 0.12 * float(coverage["target_recall"])
                + 0.08 * float(coverage["rendered_precision"])
                + 0.13 * boundary_score
                + 0.18 * float(structure["structure_score"])
                + 0.10
                * float(component_coverage["reference_component_macro_recall"])
                + 0.05
                * float(component_coverage["reference_component_min_recall"])
                + 0.14
                * float(spatial_coverage["reference_spatial_macro_recall"])
                + 0.08 * float(spatial_coverage["reference_spatial_min_recall"])
            )
        complete_alignment_candidate = (
            iou >= COMPLETE_ALIGNMENT_MINIMUM_IOU
            and float(boundary["boundary_p95_px"])
            <= COMPLETE_ALIGNMENT_MAXIMUM_BOUNDARY_P95_PX
        )
        records.append(
            {
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
                **{
                    key: (round(float(value), 8) if isinstance(value, float) else value)
                    for key, value in rigid_consensus.items()
                },
                "analysis_direction": view.get("analysis_direction"),
                "analysis_up_axis": (
                    view.get("analysis_camera_up_axis") or view.get("camera_up_axis")
                ),
                "focal_length_mm": view.get("focal_length_mm"),
                "distance_multiplier": view.get("camera_distance_multiplier"),
                "target_offset_u": view.get("camera_target_offset_u", 0.0),
                "target_offset_v": view.get("camera_target_offset_v", 0.0),
                "roll_degrees": view.get("camera_roll_degrees", 0.0),
                "principal_point_u": view.get("camera_principal_point_u", 0.0),
                "principal_point_v": view.get("camera_principal_point_v", 0.0),
                "radial_distortion_k1": view.get(
                    "camera_radial_distortion_k1", 0.0
                ),
                "radial_distortion_k2": view.get(
                    "camera_radial_distortion_k2", 0.0
                ),
                "projection_mode": view.get(
                    "camera_projection_mode", "perspective"
                ),
                "orthographic_span_multiplier": view.get(
                    "camera_orthographic_span_multiplier", 2.0
                ),
                "calibration": {
                    **calibration,
                    "frame_anchor_affine": anchor_affine.tolist(),
                },
                "whole_asset_similarity": projection,
            }
        )
    if not records:
        raise ValueError(f"No calibration candidates found for {reference_id}")
    records.sort(key=_alignment_candidate_sort_key)
    return _select_alignment_candidate(records), records


def _select_alignment_candidate(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Select within a trust region around the phase incumbent.

    Every generated phase has one exact seed marked as its frame anchor.  A
    higher composite score may only replace it when whole-object IoU and P95
    boundary remain locally non-regressive.  This prevents structure terms
    from walking the sequential physical search away from an already better
    silhouette.  Rerank/custom batches without one incumbent use the global
    complete-objective ordering directly.
    """

    if not records:
        raise ValueError("Camera alignment selection requires candidates")
    complete = [
        item for item in records if bool(item.get("complete_alignment_candidate"))
    ]
    if complete:
        return dict(min(complete, key=_alignment_candidate_sort_key))
    incumbents = [
        item
        for item in records
        if item.get("calibration", {}).get("frame_anchor") is True
    ]
    if len(incumbents) != 1:
        return dict(min(records, key=_alignment_candidate_sort_key))
    incumbent = incumbents[0]
    incumbent_has_consensus = bool(incumbent.get("rigid_consensus_valid"))
    minimum_iou = (
        float(incumbent["projection_iou"])
        - PHASE_INCUMBENT_MAXIMUM_IOU_REGRESSION
    )
    maximum_boundary = (
        float(incumbent["boundary_p95_px"])
        + PHASE_INCUMBENT_MAXIMUM_BOUNDARY_REGRESSION_PX
    )
    trusted = [
        item
        for item in records
        if float(item["projection_iou"]) >= minimum_iou
        and float(item["boundary_p95_px"]) <= maximum_boundary
        and (
            not incumbent_has_consensus
            or bool(item.get("rigid_consensus_valid"))
        )
    ]
    return dict(min(trusted, key=_alignment_candidate_sort_key))


def _alignment_candidate_sort_key(item: Mapping[str, Any]) -> tuple[Any, ...]:
    """Prefer actual gate passes, otherwise use the complete camera objective.

    The published 0.97 IoU / 3 px boundary gate is deliberately strict.  When
    no candidate passes it, sorting by normalized distance to that unreachable
    corner lets a modest boundary change dominate every other alignment cue.
    The composite objective already includes silhouette, boundary, structure,
    component, and spatial evidence, so it is authoritative for incomplete
    candidates.
    """

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
    return (
        0 if complete else 1,
        0 if bool(item.get("rigid_consensus_valid")) else 1,
        -float(item["score"]),
        -float(item.get("rigid_consensus_score", 0.0)),
        -iou,
        boundary_p95,
        str(item["view_id"]),
    )


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
    bind_frame_anchor: bool = False,
    mark_phase_incumbent: bool = False,
) -> dict[str, Any]:
    reference_id = str(score["calibration"]["reference_view_id"])
    calibration = {
        **score["calibration"],
        "reference_view_id": reference_id,
        "phase": phase,
        "frame_anchor": mark_phase_incumbent,
    }
    if bind_frame_anchor:
        audit = score.get("whole_asset_similarity", {}).get(
            "ecc_transform_audit", {}
        )
        anchor_affine = audit.get("anchor_affine")
        if not isinstance(anchor_affine, list):
            raise ValueError("Final camera score has no sealed frame anchor")
        calibration["frame_anchor_affine"] = anchor_affine
    return {
        "view_id": view_id,
        "analysis_direction": score["analysis_direction"],
        "analysis_up_axis": score["analysis_up_axis"],
        "focal_length_mm": score["focal_length_mm"],
        "distance_multiplier": score["distance_multiplier"],
        "target_offset_u": score.get("target_offset_u", 0.0),
        "target_offset_v": score.get("target_offset_v", 0.0),
        "roll_degrees": score.get("roll_degrees", 0.0),
        "principal_point_u": score.get("principal_point_u", 0.0),
        "principal_point_v": score.get("principal_point_v", 0.0),
        "radial_distortion_k1": score.get("radial_distortion_k1", 0.0),
        "radial_distortion_k2": score.get("radial_distortion_k2", 0.0),
        "projection_mode": score.get("projection_mode", "perspective"),
        "orthographic_span_multiplier": score.get(
            "orthographic_span_multiplier", 2.0
        ),
        "calibration": calibration,
    }


def _global_finalists(
    candidates: Sequence[Mapping[str, Any]],
    *,
    count: int,
    required: Sequence[Mapping[str, Any]] = (),
) -> list[dict[str, Any]]:
    """Keep the best distinct cameras across every refinement phase."""

    ordered = sorted(candidates, key=_alignment_candidate_sort_key)
    output: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for raw in (*required, *ordered):
        direction = tuple(round(float(value), 8) for value in raw["analysis_direction"])
        up_axis = tuple(round(float(value), 8) for value in raw["analysis_up_axis"])
        signature = (
            direction,
            up_axis,
            round(float(raw["focal_length_mm"]), 6),
            round(float(raw["distance_multiplier"]), 6),
            round(float(raw.get("target_offset_u", 0.0)), 6),
            round(float(raw.get("target_offset_v", 0.0)), 6),
            round(float(raw.get("roll_degrees", 0.0)), 6),
            round(float(raw.get("principal_point_u", 0.0)), 6),
            round(float(raw.get("principal_point_v", 0.0)), 6),
            round(float(raw.get("radial_distortion_k1", 0.0)), 6),
            round(float(raw.get("radial_distortion_k2", 0.0)), 6),
            str(raw.get("projection_mode", "perspective")),
            round(float(raw.get("orthographic_span_multiplier", 2.0)), 6),
        )
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
    render_runner: RenderRunner | None = None,
) -> Path:
    rendered = output_dir / "part_registry.rendered.json"
    if render_runner is not None:
        # An in-process failure may leave Kit or Replicator poisoned.  Clean
        # the exact batch before starting, make one attempt, and let the whole
        # Isaac child exit on any exception.  The supervisor can then retry in
        # a genuinely fresh process while completed phase checkpoints remain.
        if output_dir.exists():
            shutil.rmtree(output_dir)
        print(
            f"[CAMERA] in-process render {view_specs} -> {output_dir}",
            flush=True,
        )
        try:
            result = Path(
                render_runner(
                    registry=registry,
                    output_dir=output_dir,
                    view_specs=view_specs,
                    resolution=resolution,
                    rt_subframes=rt_subframes,
                    analysis_up_axis=analysis_up_axis,
                    analysis_front_axis=analysis_front_axis,
                )
            ).expanduser().resolve(strict=True)
        except Exception as exc:
            raise RuntimeError(
                "In-process camera render failed; refusing to reuse this "
                f"Isaac session: {output_dir}"
            ) from exc
        expected = rendered.resolve()
        if result != expected:
            raise RuntimeError(
                "In-process camera renderer returned an unexpected registry: "
                f"{result} (expected {expected})"
            )
        return result

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
    environment = os.environ.copy()
    tools_root = Path(__file__).resolve().parents[2]
    inherited_pythonpath = environment.get("PYTHONPATH")
    pythonpath_entries = (
        []
        if not inherited_pythonpath
        else [
            entry
            for entry in inherited_pythonpath.split(os.pathsep)
            if entry and Path(entry).expanduser().resolve() != tools_root
        ]
    )
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(tools_root), *pythonpath_entries)
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
    recorded_specs_sha256 = stored_scores.get("view_specs_sha256")
    if (
        recorded_specs_sha256 is not None
        and recorded_specs_sha256 != _sha256_file(specs_path)
    ):
        return None
    if (
        stored_scores.get("schema_version") != SCHEMA_VERSION
        or stored_scores.get("reference_view_id") != reference_id
        or stored_scores.get("phase") != phase
    ):
        return None
    winner = stored_scores.get("winner")
    candidates = stored_scores.get("candidates")
    if not isinstance(winner, dict) or not isinstance(candidates, list) or not candidates:
        return None
    if winner.get("objective_version") not in (
        CHECKPOINT_COMPATIBLE_OBJECTIVE_VERSIONS
    ) or any(
        not isinstance(candidate, Mapping)
        or candidate.get("objective_version")
        not in CHECKPOINT_COMPATIBLE_OBJECTIVE_VERSIONS
        for candidate in candidates
    ):
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
    upgraded_winner = {**winner, "objective_version": CAMERA_OBJECTIVE_VERSION}
    upgraded_candidates = [
        {**candidate, "objective_version": CAMERA_OBJECTIVE_VERSION}
        for candidate in candidates
    ]
    return upgraded_winner, upgraded_candidates


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
            "roll_degrees": 0.0,
            "principal_point_u": 0.0,
            "principal_point_v": 0.0,
            "radial_distortion_k1": 0.0,
            "radial_distortion_k2": 0.0,
            "projection_mode": "perspective",
            "orthographic_span_multiplier": 2.0,
        }
    return output


def _seed_by_view_specs(document: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    if document.get("schema_version") != VIEW_SPEC_SCHEMA_VERSION:
        raise ValueError("Initial camera view specs use an unsupported schema_version")
    raw_views = document.get("views")
    if not isinstance(raw_views, list) or not raw_views:
        raise ValueError("Initial camera view specs require non-empty views")
    output: dict[str, dict[str, Any]] = {}
    for raw in raw_views:
        if not isinstance(raw, Mapping) or not isinstance(raw.get("view_id"), str):
            continue
        reference_id = str(raw["view_id"])
        direction = raw.get("analysis_direction")
        up_axis = raw.get("analysis_up_axis")
        if direction is None or up_axis is None:
            continue
        output[reference_id] = {
            "analysis_direction": direction,
            "analysis_up_axis": up_axis,
            "focal_length_mm": float(raw.get("focal_length_mm", 45.0)),
            "distance_multiplier": float(raw.get("distance_multiplier", 2.15)),
            "target_offset_u": float(raw.get("target_offset_u", 0.0)),
            "target_offset_v": float(raw.get("target_offset_v", 0.0)),
            "roll_degrees": float(raw.get("roll_degrees", 0.0)),
            "principal_point_u": float(raw.get("principal_point_u", 0.0)),
            "principal_point_v": float(raw.get("principal_point_v", 0.0)),
            "radial_distortion_k1": float(raw.get("radial_distortion_k1", 0.0)),
            "radial_distortion_k2": float(raw.get("radial_distortion_k2", 0.0)),
            "projection_mode": str(raw.get("projection_mode", "perspective")),
            "orthographic_span_multiplier": float(
                raw.get("orthographic_span_multiplier", 2.0)
            ),
            **(
                {
                    "frame_anchor_affine": raw.get("calibration", {}).get(
                        "frame_anchor_affine"
                    )
                }
                if isinstance(raw.get("calibration"), Mapping)
                and raw.get("calibration", {}).get("frame_anchor_affine")
                is not None
                else {}
            ),
        }
    if not output:
        raise ValueError("Initial camera view specs contain no usable views")
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
                        "roll_degrees": float(view.get("camera_roll_degrees") or 0.0),
                        "principal_point_u": float(
                            view.get("camera_principal_point_u") or 0.0
                        ),
                        "principal_point_v": float(
                            view.get("camera_principal_point_v") or 0.0
                        ),
                        "radial_distortion_k1": float(
                            view.get("camera_radial_distortion_k1") or 0.0
                        ),
                        "radial_distortion_k2": float(
                            view.get("camera_radial_distortion_k2") or 0.0
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
        candidates.sort(
            key=lambda item: (
                -float(item["score"]),
                -float(item["projection_iou"]),
                float(item["boundary_p95_px"]),
                str(item["view_id"]),
                int(item["quarter_turns_ccw"]),
            )
        )
        winner = candidates[0]
        seeds[reference_id] = {
            "analysis_direction": winner["analysis_direction"],
            "analysis_up_axis": winner["analysis_up_axis"],
            "focal_length_mm": winner["focal_length_mm"],
            "distance_multiplier": winner["distance_multiplier"],
            "target_offset_u": winner.get("target_offset_u", 0.0),
            "target_offset_v": winner.get("target_offset_v", 0.0),
            "roll_degrees": winner.get("roll_degrees", 0.0),
            "principal_point_u": winner.get("principal_point_u", 0.0),
            "principal_point_v": winner.get("principal_point_v", 0.0),
            "radial_distortion_k1": winner.get("radial_distortion_k1", 0.0),
            "radial_distortion_k2": winner.get("radial_distortion_k2", 0.0),
            "projection_mode": winner.get("projection_mode", "perspective"),
            "orthographic_span_multiplier": winner.get(
                "orthographic_span_multiplier", 2.0
            ),
        }
        audit[reference_id] = {
            "winner": winner,
            "candidate_count": len(candidates),
            "top_candidates": candidates[:5],
        }
    return seeds, audit


def _merge_registry(
    *,
    baseline_path: Path,
    calibrated_path: Path,
    output_path: Path,
    reference_ids: set[str],
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
    }
    return _write_object(output_path, baseline)


def _seal_full_resolution_winners(
    *,
    rendered_path: Path,
    winners: Mapping[str, Mapping[str, Any]],
    output_path: Path,
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
    by_view_id = {
        str(view.get("view_id")): view
        for view in render_set.get("views", [])
        if isinstance(view, dict) and isinstance(view.get("view_id"), str)
    }
    sealed_views: list[dict[str, Any]] = []
    source_views: dict[str, str] = {}
    for reference_id, winner in winners.items():
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
        anchor_affine = winner.get("whole_asset_similarity", {}).get(
            "ecc_transform_audit", {}
        ).get("anchor_affine")
        if not isinstance(anchor_affine, list):
            raise ValueError(
                f"Winning finalist {source_view_id!r} has no sealed frame anchor"
            )
        view["view_id"] = reference_id
        view["sealed_source_view_id"] = source_view_id
        view["camera_calibration"] = {
            **calibration,
            "phase": "sealed_full_resolution_finalist",
            "sealed_source_view_id": source_view_id,
            "frame_anchor_affine": anchor_affine,
        }
        sealed_views.append(view)
        source_views[reference_id] = source_view_id
    rendered["render_set"] = {
        **render_set,
        "views": sealed_views,
        "requested_view_tokens": list(winners),
        "expanded_view_count": len(sealed_views),
        "sealed_full_resolution_winners": True,
        "sealed_source_views": source_views,
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
    render_runner: RenderRunner | None = None,
    max_new_render_batches: int | None = None,
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
    if max_new_render_batches is not None and (
        isinstance(max_new_render_batches, bool)
        or not isinstance(max_new_render_batches, int)
        or max_new_render_batches <= 0
    ):
        raise ValueError("max_new_render_batches must be a positive integer")
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
        seeds = _seed_by_view_specs(_read_object(initial_view_specs))
        seed_audit = {
            "mode": "existing_continuous_camera_specs",
            "initial_view_specs": str(initial_view_specs),
        }
    elif spatial_mapping is not None:
        seeds = _seed_by_reference(_read_object(spatial_mapping))
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

    winners: dict[str, dict[str, Any]] = {}
    phases: dict[str, list[dict[str, Any]]] = {}
    finalists: dict[str, list[dict[str, Any]]] = {}
    new_render_batches = 0
    for view_index, reference_id in enumerate(requested, start=1):
        print(
            f"[CAMERA] reference {view_index}/{len(requested)} {reference_id}",
            flush=True,
        )
        seed = seeds[reference_id]
        phase_records: list[dict[str, Any]] = []
        phase_candidate_pool: list[dict[str, Any]] = []
        for phase in active_phases:
            specs = _candidate_specs(
                reference_id=reference_id,
                seed=seed,
                phase=phase,
            )
            view_specs_path = destination / reference_id / f"{phase}_view_specs.json"
            scores_path = destination / reference_id / f"{phase}_scores.json"
            completed = _completed_phase(
                specs_path=view_specs_path,
                scores_path=scores_path,
                expected_specs=specs,
                reference_id=reference_id,
                phase=phase,
            )
            if completed is not None:
                winner, candidates = completed
                print(
                    f"[CAMERA] {reference_id}/{phase} reusing verified "
                    "completed candidate batch",
                    flush=True,
                )
            else:
                # Once a candidate specification changes, its previous score
                # can no longer be a completion checkpoint.  Remove it before
                # publishing the replacement specs so an interruption cannot
                # pair new specs with stale scores that happen to reuse the
                # same deterministic view IDs.
                scores_path.unlink(missing_ok=True)
                specs_path = _write_object(view_specs_path, specs)
                rendered = _run_render(
                    isaac_python=isaac_python,
                    registry=registry,
                    output_dir=destination / reference_id / f"{phase}_renders",
                    view_specs=specs_path,
                    resolution=search_resolution,
                    rt_subframes=rt_subframes,
                    analysis_up_axis=analysis_up_axis,
                    analysis_front_axis=analysis_front_axis,
                    render_runner=render_runner,
                )
                winner, candidates = _score_candidates(
                    reference_id=reference_id,
                    reference_mask=references[reference_id][0],
                    reference_image=reference_images[reference_id],
                    registry_path=rendered,
                )
                _write_object(
                    scores_path,
                    {
                        "schema_version": SCHEMA_VERSION,
                        "reference_view_id": reference_id,
                        "phase": phase,
                        "view_specs_sha256": _sha256_file(specs_path),
                        "winner": winner,
                        "candidates": candidates,
                    },
                )
                new_render_batches += 1
            phase_records.append(
                {
                    "phase": phase,
                    "winner": winner,
                    "candidate_count": len(candidates),
                }
            )
            # Candidate lists are sorted by the global objective, while the
            # sequential phase winner is additionally constrained by the
            # incumbent trust region.  Always retain that physically trusted
            # winner in the rerank pool; otherwise a high structure score from
            # a silhouette-regressive candidate can crowd it out entirely.
            phase_candidate_pool.append(winner)
            phase_candidate_pool.extend(candidates[:FINALIST_COUNT])
            seed = {
                "analysis_direction": winner["analysis_direction"],
                "analysis_up_axis": winner["analysis_up_axis"],
                "focal_length_mm": winner["focal_length_mm"],
                "distance_multiplier": winner["distance_multiplier"],
                "target_offset_u": winner.get("target_offset_u", 0.0),
                "target_offset_v": winner.get("target_offset_v", 0.0),
                "roll_degrees": winner.get("roll_degrees", 0.0),
                "principal_point_u": winner.get("principal_point_u", 0.0),
                "principal_point_v": winner.get("principal_point_v", 0.0),
                "radial_distortion_k1": winner.get("radial_distortion_k1", 0.0),
                "radial_distortion_k2": winner.get("radial_distortion_k2", 0.0),
                "projection_mode": winner.get("projection_mode", "perspective"),
                "orthographic_span_multiplier": winner.get(
                    "orthographic_span_multiplier", 2.0
                ),
                "frame_anchor_affine": winner["calibration"][
                    "frame_anchor_affine"
                ],
            }
            print(
                f"[CAMERA] {reference_id}/{phase} "
                f"IoU={winner['projection_iou']:.4f} "
                f"boundary_p95={winner['boundary_p95_px']:.2f}px",
                flush=True,
            )
            if (
                max_new_render_batches is not None
                and new_render_batches >= max_new_render_batches
            ):
                # The score document above is the atomic resume checkpoint.
                # Never recycle between rendering and scoring a phase.
                raise _RenderBatchBudgetReached(
                    "Camera render-batch budget reached after sealed phase "
                    f"{reference_id}/{phase}",
                    checkpoint=scores_path,
                )
        trusted_finalist = min(
            (record["winner"] for record in phase_records),
            key=lambda item: (
                -float(item["projection_iou"]),
                float(item["boundary_p95_px"]),
                -float(item["score"]),
                str(item["view_id"]),
            ),
        )
        finalists[reference_id] = _global_finalists(
            phase_candidate_pool,
            count=FINALIST_COUNT,
            required=(trusted_finalist,),
        )
        winners[reference_id] = trusted_finalist
        phases[reference_id] = phase_records

    finalist_specs = {
        "schema_version": VIEW_SPEC_SCHEMA_VERSION,
        "views": [
            _spec_from_score(
                score,
                view_id=f"rerank_{reference_id}_{rank:02d}",
                phase="full_resolution_rerank",
                mark_phase_incumbent=rank == 1,
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
        render_runner=render_runner,
    )
    for reference_id in requested:
        winner, candidates = _score_candidates(
            reference_id=reference_id,
            reference_mask=references[reference_id][0],
            reference_image=reference_images[reference_id],
            registry_path=finalist_rendered,
        )
        winners[reference_id] = winner
        phases[reference_id].append(
            {
                "phase": "full_resolution_rerank",
                "winner": winner,
                "candidate_count": len(candidates),
            }
        )
        _write_object(
            destination / reference_id / "full_resolution_scores.json",
            {
                "schema_version": SCHEMA_VERSION,
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

    final_specs = {
        "schema_version": VIEW_SPEC_SCHEMA_VERSION,
        "views": [
            _spec_from_score(
                winners[reference_id],
                view_id=reference_id,
                phase="final",
                bind_frame_anchor=True,
            )
            for reference_id in requested
        ],
    }
    final_specs_path = _write_object(destination / "final_view_specs.json", final_specs)
    final_rendered = _seal_full_resolution_winners(
        rendered_path=finalist_rendered,
        winners=winners,
        output_path=destination / "sealed_finalists" / "part_registry.rendered.json",
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
        "source_initial_view_specs": (
            str(initial_view_specs) if initial_view_specs is not None else None
        ),
        "search_phases": list(active_phases),
        "seed_search": seed_audit,
        "whole_asset_only": True,
        "per_part_geometric_warp_applied": False,
        "robust_rigid_part_consensus": {
            "enabled": True,
            "asset_specific_part_rules": False,
            "camera_consensus_is_plurality_objective": True,
            "outlier_part_transforms_applied": False,
            "assembly_clustering": "prim_path_grandparent_subtree",
            "assembly_cluster_requires_coherent_2d_residual_direction": True,
            "unobservable_internal_parts_are_indeterminate": True,
            "minimum_part_pixels": RIGID_CONSENSUS_MINIMUM_PART_PIXELS,
            "minimum_inlier_parts": RIGID_CONSENSUS_MINIMUM_INLIER_PARTS,
            "minimum_inlier_pixel_coverage": (
                RIGID_CONSENSUS_MINIMUM_INLIER_COVERAGE
            ),
        },
        "camera_objective_version": CAMERA_OBJECTIVE_VERSION,
        "camera_intrinsics_optimized": [
            "projection_mode",
            "focal_length_mm",
            "principal_point_u",
            "principal_point_v",
            "radial_distortion_k1",
            "radial_distortion_k2",
            "orthographic_aperture",
        ],
        "camera_extrinsics_optimized": [
            "orbit_azimuth",
            "orbit_elevation",
            "camera_roll",
            "camera_distance",
            "optical_axis_target_u",
            "optical_axis_target_v",
        ],
        "image_frame_residual_optimized": [
            "bounded_uniform_scale_0.985_to_1.015",
            "bounded_in_plane_rotation_plus_or_minus_1_degree",
            "bounded_crop_translation_plus_or_minus_0.02_frame",
        ],
        "image_frame_residual_shared_anchor_per_batch": True,
        "image_frame_residual_shared_anchor_per_reference": True,
        "part_balanced_size_strata": ["small", "medium", "large"],
        "multiview_residual_diagnosis": _classify_multiview_residuals(
            final_scores
        ),
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
    parser.add_argument(
        "--render-backend",
        choices=("subprocess", "inprocess", "supervisor"),
        default="subprocess",
        help=(
            "subprocess preserves the legacy one-Isaac-process-per-batch "
            "behavior; inprocess reuses this Isaac process; supervisor "
            "rotates bounded inprocess children"
        ),
    )
    parser.add_argument(
        "--max-new-render-batches",
        type=int,
        help=(
            "internal inprocess child budget; supervisor defaults to "
            f"{SUPERVISOR_RENDER_BATCH_LIMIT}"
        ),
    )
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
    return parser.parse_args()


def _calibrate_from_args(
    args: argparse.Namespace,
    *,
    render_runner: RenderRunner | None = None,
) -> dict[str, Any]:
    return calibrate(
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
        render_runner=render_runner,
        max_new_render_batches=args.max_new_render_batches,
    )


def _make_inprocess_render_runner(
    *,
    render_part_views: Callable[..., Mapping[str, Any]],
    axis_vectors: Mapping[str, tuple[float, float, float]],
) -> RenderRunner:
    """Adapt the reusable USD renderer to the camera batch contract."""

    def run(
        *,
        registry: Path,
        output_dir: Path,
        view_specs: Path,
        resolution: int,
        rt_subframes: int,
        analysis_up_axis: str,
        analysis_front_axis: str,
    ) -> Path:
        report = render_part_views(
            registry_path=registry,
            output_dir=output_dir,
            resolution=resolution,
            view_names=None,
            rt_subframes=rt_subframes,
            analysis_up_axis=axis_vectors[analysis_up_axis],
            analysis_front_axis=axis_vectors[analysis_front_axis],
            lighting_profile="geometry",
            showcase=False,
            generate_part_evidence=False,
            custom_view_specs_path=view_specs,
        )
        output_registry = report.get("output_registry")
        if not isinstance(output_registry, str) or not output_registry:
            raise RuntimeError("In-process renderer did not report output_registry")
        return Path(output_registry)

    return run


def _run_inprocess_backend(
    args: argparse.Namespace,
    *,
    simulation_app_factory: Callable[[Mapping[str, bool]], Any] | None = None,
    render_module_loader: Callable[[], Any] | None = None,
) -> tuple[int, dict[str, Any] | None]:
    """Run all allowed batches under exactly one SimulationApp instance."""

    if simulation_app_factory is None:
        from isaacsim import SimulationApp

        simulation_app_factory = SimulationApp
    app = simulation_app_factory({"headless": True, "create_new_stage": False})
    try:
        # Importing this module initializes its Isaac-facing dependencies, so
        # it must happen only after SimulationApp has finished starting.
        render_module = (
            render_module_loader()
            if render_module_loader is not None
            else importlib.import_module("qwen_material_pipeline.usd.render")
        )
        runner = _make_inprocess_render_runner(
            render_part_views=render_module.render_part_views,
            axis_vectors=render_module.AXIS_VECTORS,
        )
        try:
            return 0, _calibrate_from_args(args, render_runner=runner)
        except _RenderBatchBudgetReached as exc:
            checkpoint = exc.checkpoint.expanduser().resolve(strict=True)
            output_dir = args.output_dir.expanduser().resolve()
            if not checkpoint.is_relative_to(output_dir):
                raise RuntimeError(
                    "Camera session rotation checkpoint escaped its output root: "
                    f"{checkpoint}"
                ) from exc
            _write_object(
                output_dir / SUPERVISOR_ROTATION_MARKER,
                {
                    "schema_version": SUPERVISOR_ROTATION_SCHEMA_VERSION,
                    "output_dir": str(output_dir),
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": _sha256_file(checkpoint),
                    "max_new_render_batches": args.max_new_render_batches,
                },
            )
            print(f"[CAMERA] {exc}; requesting a fresh Isaac session", flush=True)
            # Isaac's python.sh normalizes every non-zero Python exit to 1.
            # A zero exit plus the hash-bound marker is therefore the only
            # portable way to distinguish an intentional rotation from a
            # failed Kit process.  The supervisor verifies and consumes it.
            return 0, None
    finally:
        app.close()


def _forwarded_calibration_arguments(
    args: argparse.Namespace,
    *,
    render_backend: str,
    max_new_render_batches: int | None,
) -> list[str]:
    output = [
        "--registry",
        str(args.registry),
        "--reference-manifest",
        str(args.reference_manifest),
    ]
    if args.spatial_mapping is not None:
        output.extend(["--spatial-mapping", str(args.spatial_mapping)])
    if args.initial_view_specs is not None:
        output.extend(["--initial-view-specs", str(args.initial_view_specs)])
    if args.search_phases:
        output.extend(["--search-phases", args.search_phases])
    output.extend(
        [
            "--isaac-python",
            str(args.isaac_python),
            "--render-backend",
            render_backend,
            "--output-dir",
            str(args.output_dir),
        ]
    )
    if args.reference_ids:
        output.extend(["--reference-ids", args.reference_ids])
    output.extend(
        [
            "--search-resolution",
            str(args.search_resolution),
            "--final-resolution",
            str(args.final_resolution),
            "--rt-subframes",
            str(args.rt_subframes),
            "--analysis-up-axis",
            args.analysis_up_axis,
            f"--analysis-front-axis={args.analysis_front_axis}",
        ]
    )
    if max_new_render_batches is not None:
        output.extend(
            ["--max-new-render-batches", str(max_new_render_batches)]
        )
    return output


def _run_supervisor_backend(args: argparse.Namespace) -> int:
    configured_batch_limit = (
        args.max_new_render_batches
        if args.max_new_render_batches is not None
        else SUPERVISOR_RENDER_BATCH_LIMIT
    )
    if (
        isinstance(configured_batch_limit, bool)
        or not isinstance(configured_batch_limit, int)
        or configured_batch_limit <= 0
    ):
        raise ValueError("max_new_render_batches must be a positive integer")
    environment = os.environ.copy()
    tools_root = Path(__file__).resolve().parents[2]
    inherited_pythonpath = environment.get("PYTHONPATH")
    pythonpath_entries = (
        []
        if not inherited_pythonpath
        else [
            entry
            for entry in inherited_pythonpath.split(os.pathsep)
            if entry and Path(entry).expanduser().resolve() != tools_root
        ]
    )
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(tools_root), *pythonpath_entries)
    )
    output_dir = args.output_dir.expanduser().resolve()
    rotation_marker = output_dir / SUPERVISOR_ROTATION_MARKER
    final_report = output_dir / "camera_calibration_report.json"
    last_rotation: tuple[str, str] | None = None
    failures = 0
    current_batch_limit = configured_batch_limit
    degradation_logged = False
    while True:
        if rotation_marker.is_symlink():
            raise RuntimeError(
                "Camera session rotation marker must not be a symlink: "
                f"{rotation_marker}"
            )
        rotation_marker.unlink(missing_ok=True)
        session_batch_limit = current_batch_limit
        command = [
            str(args.isaac_python.expanduser().resolve(strict=True)),
            "-m",
            "qwen_material_pipeline",
            "calibrate-cameras",
            *_forwarded_calibration_arguments(
                args,
                render_backend="inprocess",
                max_new_render_batches=session_batch_limit,
            ),
        ]
        print("[CAMERA] supervisor starting bounded Isaac session", flush=True)
        try:
            completed = subprocess.run(command, check=False, env=environment)
            return_code = int(completed.returncode)
        except FileNotFoundError:
            return_code = 127
        if return_code == 0 and final_report.is_file():
            if rotation_marker.exists() or rotation_marker.is_symlink():
                raise RuntimeError(
                    "Camera child published both a final report and a session "
                    "rotation marker"
                )
            return 0
        if return_code == 0 and rotation_marker.is_file():
            marker = _read_object(rotation_marker)
            expected_keys = {
                "schema_version",
                "output_dir",
                "checkpoint",
                "checkpoint_sha256",
                "max_new_render_batches",
            }
            if set(marker) != expected_keys:
                raise RuntimeError("Camera session rotation marker has invalid fields")
            raw_checkpoint = marker.get("checkpoint")
            marker_batch_limit = marker.get("max_new_render_batches")
            if (
                marker.get("schema_version")
                != SUPERVISOR_ROTATION_SCHEMA_VERSION
                or marker.get("output_dir") != str(output_dir)
                or isinstance(marker_batch_limit, bool)
                or not isinstance(marker_batch_limit, int)
                or marker_batch_limit != session_batch_limit
                or not isinstance(raw_checkpoint, str)
            ):
                raise RuntimeError("Camera session rotation marker is invalid")
            checkpoint = Path(raw_checkpoint).expanduser().resolve(strict=True)
            if (
                not checkpoint.is_relative_to(output_dir)
                or not checkpoint.name.endswith("_scores.json")
            ):
                raise RuntimeError(
                    "Camera session rotation checkpoint is outside its output root"
                )
            checkpoint_sha256 = _sha256_file(checkpoint)
            if marker.get("checkpoint_sha256") != checkpoint_sha256:
                raise RuntimeError(
                    "Camera session rotation checkpoint hash does not match"
                )
            rotation = (str(checkpoint), checkpoint_sha256)
            if rotation == last_rotation:
                raise RuntimeError(
                    "Camera supervisor made no checkpoint progress between sessions"
                )
            last_rotation = rotation
            rotation_marker.unlink()
            failures = 0
            continue
        if return_code == 0:
            raise RuntimeError(
                "Camera child exited successfully without a final report or a "
                "validated session rotation marker"
            )
        if current_batch_limit != 1:
            current_batch_limit = 1
            if not degradation_logged:
                print(
                    "[CAMERA] child session failed; reducing all remaining "
                    "Isaac sessions to one new render batch "
                    f"(exit code {return_code})",
                    flush=True,
                )
                degradation_logged = True
        failures += 1
        if failures >= RENDER_RETRY_ATTEMPTS:
            raise RuntimeError(
                "Camera calibration failed in "
                f"{RENDER_RETRY_ATTEMPTS} fresh Isaac sessions "
                f"(last exit code {return_code})"
            )
        print(
            "[CAMERA] bounded Isaac session failed; retrying from verified "
            f"phase checkpoints in {RENDER_RETRY_DELAY_SECONDS:.0f}s "
            f"({failures}/{RENDER_RETRY_ATTEMPTS})",
            flush=True,
        )
        time.sleep(RENDER_RETRY_DELAY_SECONDS)


def _print_report(args: argparse.Namespace, report: Mapping[str, Any]) -> None:
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


def main() -> int:
    args = parse_args()
    if args.render_backend == "supervisor":
        return _run_supervisor_backend(args)
    if args.render_backend == "inprocess":
        exit_code, report = _run_inprocess_backend(args)
        if exit_code != 0:
            return exit_code
        if report is None:
            return 0
    else:
        if args.max_new_render_batches is not None:
            raise ValueError(
                "--max-new-render-batches is valid only for inprocess or "
                "supervisor backends"
            )
        report = _calibrate_from_args(args)
    _print_report(args, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
