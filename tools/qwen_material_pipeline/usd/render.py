#!/usr/bin/env python3
"""Render canonical RGB, part-ID and per-part crop views with Isaac Replicator.

The renderer writes semantic labels only to the in-memory stage.  It never saves
or changes the source USD.
"""

from __future__ import annotations

import argparse
import cv2
import hashlib
import json
import math
import re
import traceback
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any, TypeVar

import numpy as np

from qwen_material_pipeline.core.progress import (
    ProgressCallback,
    emit_progress_event,
    report_progress,
)


VIEW_DIRECTIONS = {
    "front": (0.0, -1.0, 0.15),
    "rear": (0.0, 1.0, 0.15),
    "left": (-1.0, 0.0, 0.15),
    "right": (1.0, 0.0, 0.15),
    "top": (0.0, 0.0, 1.0),
    "iso": (0.8, -0.8, 0.55),
}


def _simulation_app_launch_config() -> dict[str, bool]:
    """Start Kit without creating an unused asynchronous USD stage.

    The renderer opens the registry's asset explicitly after Kit is ready.
    Avoiding the default throwaway stage removes an unnecessary loader-thread
    race from long unattended candidate tournaments.
    """

    return {"headless": True, "create_new_stage": False}


def _pose_direction(
    azimuth_degrees: int, elevation_degrees: int
) -> tuple[float, float, float]:
    """Return one deterministic analysis-space spherical camera direction."""

    azimuth = math.radians(float(azimuth_degrees))
    elevation = math.radians(float(elevation_degrees))
    planar = math.cos(elevation)
    return (
        planar * math.sin(azimuth),
        -planar * math.cos(azimuth),
        math.sin(elevation),
    )


# Six established canonical views plus upper-hemisphere samples.  The 75-degree
# ring is intentionally near, but not equal to, a mathematical top view: real
# inspection photographs often retain enough side elevation to expose controls
# mounted on a vertical face.
POSE_BANK_DIRECTIONS = {
    **VIEW_DIRECTIONS,
    **{
        f"pose_a{azimuth:03d}_e015": _pose_direction(azimuth, 15)
        for azimuth in (45, 135, 225, 315)
    },
    **{
        f"pose_a{azimuth:03d}_e035": _pose_direction(azimuth, 35)
        for azimuth in range(0, 360, 45)
    },
    **{
        f"pose_a{azimuth:03d}_e060": _pose_direction(azimuth, 60)
        for azimuth in range(0, 360, 45)
    },
    **{
        f"pose_a{azimuth:03d}_e075": _pose_direction(azimuth, 75)
        for azimuth in range(0, 360, 45)
    },
    **{
        f"pose_a{azimuth:03d}_e075_r180": _pose_direction(azimuth, 75)
        for azimuth in range(0, 360, 45)
    },
    **{
        f"pose_a{azimuth:03d}_e082": _pose_direction(azimuth, 82)
        for azimuth in range(0, 360, 45)
    },
    **{
        f"pose_a{azimuth:03d}_e082_r180": _pose_direction(azimuth, 82)
        for azimuth in range(0, 360, 45)
    },
    **{
        f"pose_a{azimuth:03d}_e082_toproll": _pose_direction(azimuth, 82)
        for azimuth in range(0, 360, 45)
    },
    **{
        f"pose_a{azimuth:03d}_e082_toproll_r180": _pose_direction(azimuth, 82)
        for azimuth in range(0, 360, 45)
    },
}
VIEW_PRESETS = {
    "pose-bank-26": tuple(
        name
        for name in POSE_BANK_DIRECTIONS
        if "_e075" not in name and "_e082" not in name
    ),
    "pose-bank-34": tuple(
        name
        for name in POSE_BANK_DIRECTIONS
        if "_e082" not in name and not name.endswith("_r180")
    ),
    "pose-bank-42": tuple(
        name for name in POSE_BANK_DIRECTIONS if "_e082" not in name
    ),
    "pose-bank-58": tuple(
        name for name in POSE_BANK_DIRECTIONS if "_toproll" not in name
    ),
    "pose-bank-74": tuple(POSE_BANK_DIRECTIONS),
}

AXIS_VECTORS = {
    "x": (1.0, 0.0, 0.0),
    "-x": (-1.0, 0.0, 0.0),
    "y": (0.0, 1.0, 0.0),
    "-y": (0.0, -1.0, 0.0),
    "z": (0.0, 0.0, 1.0),
    "-z": (0.0, 0.0, -1.0),
}

ISOLATED_EVIDENCE_SCHEMA_VERSION = "qwen-isolated-part-evidence/v1"
ISOLATED_EVIDENCE_SOURCE_PIXEL_FLOOR = 12
ISOLATED_EVIDENCE_TARGET_LONG_EDGE = 224
ISOLATED_EVIDENCE_CANVAS_SIZE = 320
ISOLATED_EVIDENCE_MAX_VIEWS = 3
EVIDENCE_LIGHTING_PROFILES = ("geometry", "material-neutral")
PROGRESS_SCOPE = "qwen_material_pipeline"
ISAAC_STARTUP_PROGRESS_STAGE = "isaac_startup"
RENDER_BUSINESS_PROGRESS_STAGE = "render_business"
RENDER_CAPTURE_PROGRESS_STAGE = "render_capture"
RENDER_VIEWS_PROGRESS_STAGE = "render_views"
POST_CLOSE_MIN_KIT_UPDATES = 2
POST_CLOSE_MAX_KIT_UPDATES = 120

_ProgressItem = TypeVar("_ProgressItem")


class _RenderCleanupError(RuntimeError):
    """Report one or more failures while releasing a render invocation."""

    def __init__(self, failures: Sequence[tuple[str, BaseException]]) -> None:
        self.failures = tuple(failures)
        details = "; ".join(
            f"{operation}: {type(error).__name__}: {error}"
            for operation, error in self.failures
        )
        super().__init__(f"Render resource cleanup failed ({details})")


def _get_kit_app() -> Any:
    """Return Kit's application interface after SimulationApp has started."""

    import omni.kit.app

    return omni.kit.app.get_app()


def _wait_for_stage_to_close(context: Any, app: Any) -> None:
    """Pump Kit until stage-close hooks retire renderer resources.

    Replicator owns RenderProduct and HydraTexture teardown through its USD
    stage-closing/stage-closed hooks.  Those hooks and the renderer retire work
    over Kit updates, so opening another stage immediately after close_stage()
    can otherwise race resources from the previous render batch.
    """

    updates = 0
    while updates < POST_CLOSE_MIN_KIT_UPDATES or context.get_stage() is not None:
        if updates >= POST_CLOSE_MAX_KIT_UPDATES:
            raise RuntimeError(
                "USD stage remained open after "
                f"{POST_CLOSE_MAX_KIT_UPDATES} post-close Kit updates"
            )
        app.update()
        updates += 1


class _RenderLifecycle:
    """Own every Kit/Replicator resource created by one render invocation.

    Camera calibration can call :func:`render_part_views` repeatedly inside a
    single SimulationApp.  Replicator shutdown is asynchronous, so merely
    calling ``stop`` leaves the next invocation racing an orchestrator in the
    STOPPING state.  Annotators are detached explicitly, while Replicator's
    stage-close hooks remain the sole owner of RenderProduct and HydraTexture
    destruction.  Cleanup attempts every independent operation even when an
    earlier one raises.
    """

    def __init__(self) -> None:
        self._context: Any | None = None
        self._stage_opened = False
        self._annotators: list[tuple[Any, Any]] = []
        self._orchestrator: Any | None = None
        self._orchestrator_step_attempted = False
        self._cleaned = False

    def register_open_stage(self, context: Any) -> None:
        """Record a successfully opened stage for unconditional closing."""

        if self._stage_opened:
            raise RuntimeError("Render lifecycle already owns an open USD stage")
        self._context = context
        self._stage_opened = True

    def attach_annotator(self, annotator: Any, render_product: Any) -> None:
        """Record an annotator before attach so partial attaches are releasable."""

        self._annotators.append((annotator, render_product))
        annotator.attach(render_product)

    def step(self, orchestrator: Any, **kwargs: Any) -> Any:
        """Mark a Replicator step as attempted before entering Replicator."""

        if self._orchestrator is not None and self._orchestrator is not orchestrator:
            raise RuntimeError("Render lifecycle cannot own multiple orchestrators")
        self._orchestrator = orchestrator
        self._orchestrator_step_attempted = True
        return orchestrator.step(**kwargs)

    def cleanup(self) -> None:
        """Synchronously release all owned resources, collecting every failure."""

        if self._cleaned:
            return
        self._cleaned = True
        failures: list[tuple[str, BaseException]] = []

        def attempt(operation: str, callback: Callable[[], Any]) -> None:
            try:
                callback()
            except BaseException as error:
                failures.append((operation, error))

        if self._orchestrator_step_attempted:
            assert self._orchestrator is not None
            attempt("orchestrator.stop", self._orchestrator.stop)
            # stop() is non-blocking in supported Replicator releases.  Waiting
            # is mandatory before a later render_part_views call may step.
            attempt(
                "orchestrator.wait_until_complete",
                self._orchestrator.wait_until_complete,
            )

        for index, (annotator, render_product) in enumerate(
            reversed(self._annotators), start=1
        ):
            attempt(
                f"annotator.detach[{index}]",
                lambda annotator=annotator, render_product=render_product: (
                    annotator.detach(render_product)
                ),
            )

        if self._stage_opened:
            assert self._context is not None
            stage_close_requested = False

            def close_stage() -> None:
                nonlocal stage_close_requested
                result = self._context.close_stage()
                if result is not True:
                    raise RuntimeError(
                        "omni.usd context.close_stage() did not return True"
                    )
                stage_close_requested = True

            attempt("usd_context.close_stage", close_stage)
            if stage_close_requested:
                attempt(
                    "usd_context.await_stage_closed",
                    lambda: _wait_for_stage_to_close(
                        self._context,
                        _get_kit_app(),
                    ),
                )

        if failures:
            error = _RenderCleanupError(failures)
            raise error from failures[0][1]


def _counted_progress_items(
    items: Sequence[_ProgressItem],
    *,
    progress_callback: ProgressCallback | None,
    stage: str,
    unit: str,
    detail: Callable[[_ProgressItem], str] | None = None,
) -> Iterator[_ProgressItem]:
    """Yield items while advancing progress only after each item succeeds.

    The update after ``yield`` is intentional: if the caller raises while
    processing an item, that item is never counted and the generator cannot
    emit a misleading ``complete`` event.
    """

    total = len(items)
    if total <= 0:
        raise ValueError(f"{stage} requires at least one {unit}")
    report_progress(
        progress_callback,
        scope=PROGRESS_SCOPE,
        stage=stage,
        state="start",
        current=0,
        total=total,
        unit=unit,
        detail=f"Starting {stage}",
    )
    for current, item in enumerate(items, start=1):
        yield item
        report_progress(
            progress_callback,
            scope=PROGRESS_SCOPE,
            stage=stage,
            state="update",
            current=current,
            total=total,
            unit=unit,
            detail=detail(item) if detail is not None else f"Completed {current}/{total}",
        )
    report_progress(
        progress_callback,
        scope=PROGRESS_SCOPE,
        stage=stage,
        state="complete",
        current=total,
        total=total,
        unit=unit,
        detail=f"Completed {stage}",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_registry(path: str | Path) -> tuple[Path, dict[str, Any]]:
    registry_path = Path(path).expanduser().resolve(strict=True)
    document = json.loads(registry_path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or not isinstance(document.get("parts"), list):
        raise ValueError("Registry must be a JSON object containing parts")
    return registry_path, document


def _normalize(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    length = math.sqrt(sum(value * value for value in vector))
    if length <= 1e-12:
        raise ValueError("Camera direction cannot be zero")
    return tuple(value / length for value in vector)


def _resolve_view_directions(
    requested: list[str] | None,
) -> tuple[dict[str, tuple[float, float, float]], list[str]]:
    """Expand canonical names and pose-bank presets without duplicates."""

    tokens = requested or list(VIEW_DIRECTIONS)
    expanded: list[str] = []
    presets: list[str] = []
    for token in tokens:
        if token in VIEW_PRESETS:
            presets.append(token)
            expanded.extend(VIEW_PRESETS[token])
        else:
            expanded.append(token)
    names = list(dict.fromkeys(expanded))
    unknown = sorted(set(names) - set(POSE_BANK_DIRECTIONS))
    if unknown:
        supported = sorted({*VIEW_DIRECTIONS, *VIEW_PRESETS})
        raise ValueError(
            "Unknown canonical views or pose presets: "
            f"{', '.join(unknown)}; expected values from {supported}"
        )
    return {name: POSE_BANK_DIRECTIONS[name] for name in names}, presets


def _load_custom_view_specs(
    path: str | Path,
) -> dict[str, dict[str, Any]]:
    """Load validated arbitrary camera poses for continuous calibration.

    The specification is deliberately independent from USD/Isaac imports so
    it can be generated and unit-tested in the ordinary pipeline runtime.
    Directions and up axes use the same analysis-space basis as canonical
    views.  A view may change camera distance/focal length, select perspective
    or true orthographic projection, and move the optical-axis target within
    the asset frame, but never authors a transform on the source asset.
    """

    source = Path(path).expanduser().resolve(strict=True)
    document = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("Custom camera view specification must be an object")
    if document.get("schema_version") != "qwen-camera-view-specs/v1":
        raise ValueError(
            "Custom camera view specification has an unsupported schema_version"
        )
    raw_views = document.get("views")
    if not isinstance(raw_views, list) or not raw_views:
        raise ValueError("Custom camera view specification requires non-empty views")
    output: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(raw_views):
        if not isinstance(raw, dict):
            raise ValueError(f"Custom camera view {index} must be an object")
        view_id = raw.get("view_id")
        if (
            not isinstance(view_id, str)
            or not view_id
            or re.fullmatch(r"[A-Za-z0-9_.-]+", view_id) is None
        ):
            raise ValueError(
                f"Custom camera view {index}.view_id must be a safe non-empty ID"
            )
        if view_id in output:
            raise ValueError(f"Duplicate custom camera view ID: {view_id}")

        def vector(field: str) -> tuple[float, float, float]:
            value = raw.get(field)
            if (
                not isinstance(value, list)
                or len(value) != 3
                or any(
                    isinstance(component, bool)
                    or not isinstance(component, (int, float))
                    or not math.isfinite(float(component))
                    for component in value
                )
            ):
                raise ValueError(
                    f"Custom camera view {view_id}.{field} must contain "
                    "three finite numbers"
                )
            return _normalize(tuple(float(component) for component in value))

        direction = vector("analysis_direction")
        up_axis = vector("analysis_up_axis")
        if abs(_dot(direction, up_axis)) >= 0.999:
            raise ValueError(
                f"Custom camera view {view_id} direction and up axis are parallel"
            )
        focal_length = raw.get("focal_length_mm", 45.0)
        distance_multiplier = raw.get("distance_multiplier", 2.15)
        target_offset_u = raw.get("target_offset_u", 0.0)
        target_offset_v = raw.get("target_offset_v", 0.0)
        roll_degrees = raw.get("roll_degrees", 0.0)
        principal_point_u = raw.get("principal_point_u", 0.0)
        principal_point_v = raw.get("principal_point_v", 0.0)
        radial_distortion_k1 = raw.get("radial_distortion_k1", 0.0)
        radial_distortion_k2 = raw.get("radial_distortion_k2", 0.0)
        projection_mode = raw.get("projection_mode", "perspective")
        orthographic_span_multiplier = raw.get(
            "orthographic_span_multiplier", 2.0
        )
        if projection_mode not in {"perspective", "orthographic"}:
            raise ValueError(
                f"Custom camera view {view_id}.projection_mode must be "
                "'perspective' or 'orthographic'"
            )
        for field, value, minimum, maximum in (
            ("focal_length_mm", focal_length, 12.0, 2000.0),
            ("distance_multiplier", distance_multiplier, 1.05, 100.0),
            ("target_offset_u", target_offset_u, -1.0, 1.0),
            ("target_offset_v", target_offset_v, -1.0, 1.0),
            ("roll_degrees", roll_degrees, -15.0, 15.0),
            ("principal_point_u", principal_point_u, -0.20, 0.20),
            ("principal_point_v", principal_point_v, -0.20, 0.20),
            ("radial_distortion_k1", radial_distortion_k1, -0.35, 0.35),
            ("radial_distortion_k2", radial_distortion_k2, -0.20, 0.20),
            (
                "orthographic_span_multiplier",
                orthographic_span_multiplier,
                0.1,
                20.0,
            ),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not minimum <= float(value) <= maximum
            ):
                raise ValueError(
                    f"Custom camera view {view_id}.{field} must be within "
                    f"[{minimum}, {maximum}]"
                )
        output[view_id] = {
            "analysis_direction": direction,
            "analysis_up_axis": up_axis,
            "focal_length_mm": float(focal_length),
            "distance_multiplier": float(distance_multiplier),
            "target_offset_u": float(target_offset_u),
            "target_offset_v": float(target_offset_v),
            "roll_degrees": float(roll_degrees),
            "principal_point_u": float(principal_point_u),
            "principal_point_v": float(principal_point_v),
            "radial_distortion_k1": float(radial_distortion_k1),
            "radial_distortion_k2": float(radial_distortion_k2),
            "projection_mode": projection_mode,
            "orthographic_span_multiplier": float(
                orthographic_span_multiplier
            ),
            "calibration": raw.get("calibration"),
        }
    return output


def _load_assembly_pose_overrides(path: str | Path) -> list[dict[str, Any]]:
    """Load bounded world-space rigid transforms for assembly subtrees.

    The contract deliberately targets Xform subtrees rather than Mesh Parts.
    One transform therefore preserves the internal CAD assembly exactly and
    cannot become a per-Part image warp.
    """

    source = Path(path).expanduser().resolve(strict=True)
    document = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("schema_version") != (
        "qwen-assembly-pose-overrides/v1"
    ):
        raise ValueError("Assembly pose overrides have an unsupported schema")
    raw_overrides = document.get("overrides")
    if not isinstance(raw_overrides, list) or not raw_overrides:
        raise ValueError("Assembly pose overrides require non-empty overrides")
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_overrides):
        if not isinstance(raw, dict):
            raise ValueError(f"Assembly pose override {index} must be an object")
        prim_path = raw.get("prim_path")
        translation = raw.get("world_translation")
        if (
            not isinstance(prim_path, str)
            or not prim_path.startswith("/")
            or prim_path in seen
        ):
            raise ValueError(f"Assembly pose override {index} has invalid prim_path")
        if (
            not isinstance(translation, list)
            or len(translation) != 3
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in translation
            )
        ):
            raise ValueError(
                f"Assembly pose override {prim_path} requires finite translation"
            )
        seen.add(prim_path)
        output.append(
            {
                "prim_path": prim_path,
                "world_translation": [float(value) for value in translation],
            }
        )
    return output


def _apply_assembly_pose_overrides(
    *, stage: Any, overrides: Sequence[dict[str, Any]], maximum_translation: float
) -> None:
    """Apply unsaved assembly translations while preserving authored ops."""

    from pxr import Gf, Usd, UsdGeom

    for override in overrides:
        prim_path = str(override["prim_path"])
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or prim.GetTypeName() != "Xform" or prim.IsInstanceProxy():
            raise ValueError(
                f"Assembly pose override must target a writable Xform: {prim_path}"
            )
        translation = tuple(float(value) for value in override["world_translation"])
        length = math.sqrt(sum(value * value for value in translation))
        if length > maximum_translation:
            raise ValueError(
                f"Assembly pose override exceeds bounded asset-relative motion: "
                f"{prim_path} ({length:.6f} > {maximum_translation:.6f})"
            )
        xform = UsdGeom.Xformable(prim)
        op = xform.AddTranslateOp(
            UsdGeom.XformOp.PrecisionDouble,
            "qwenAssemblyPose",
        )
        # An appended xformOp can be composed before or after authored rotate/
        # scale ops.  Solving its actual three world-space basis responses is
        # therefore safer than assuming a particular authored op order.  This
        # keeps the public contract a true world translation for arbitrary
        # assembly hierarchies, not only translation-only CAD nodes.
        op.Set(Gf.Vec3d(0.0))
        base = xform.ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        ).ExtractTranslation()
        columns: list[list[float]] = []
        for axis in range(3):
            unit = [0.0, 0.0, 0.0]
            unit[axis] = 1.0
            op.Set(Gf.Vec3d(*unit))
            moved = xform.ComputeLocalToWorldTransform(
                Usd.TimeCode.Default()
            ).ExtractTranslation()
            columns.append([float(moved[i] - base[i]) for i in range(3)])
        response = np.asarray(columns, dtype=np.float64).T
        if (
            not np.isfinite(response).all()
            or abs(float(np.linalg.det(response))) < 1e-9
        ):
            raise ValueError(
                f"Assembly pose override has a singular transform basis: {prim_path}"
            )
        local_delta = np.linalg.solve(
            response,
            np.asarray(translation, dtype=np.float64),
        )
        op.Set(Gf.Vec3d(*(float(value) for value in local_delta)))
        achieved = (
            xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            .ExtractTranslation()
            - base
        )
        error = math.sqrt(
            sum((float(achieved[i]) - translation[i]) ** 2 for i in range(3))
        )
        if error > 1e-6:
            raise ValueError(
                f"Assembly pose override could not realize world translation: "
                f"{prim_path} (error={error:.9f})"
            )


def _dot(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
    return sum(left[index] * right[index] for index in range(3))


def _cross(
    left: tuple[float, float, float], right: tuple[float, float, float]
) -> tuple[float, float, float]:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _analysis_basis(
    up_axis: tuple[float, float, float],
    front_axis: tuple[float, float, float],
) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]:
    """Build a right-handed analysis-to-world camera basis.

    ``front_axis`` is the world-space direction from the asset toward a
    canonical front-view camera.  The returned columns are the world-space
    directions of analysis +X (right), +Y (rear), and +Z (up).  Transforming
    cameras through this basis is equivalent to globally orienting the asset,
    but does not author any transform on the USD stage.
    """

    up = _normalize(up_axis)
    requested_front = _normalize(front_axis)
    alignment = _dot(requested_front, up)
    planar_front = tuple(
        requested_front[index] - alignment * up[index] for index in range(3)
    )
    try:
        front = _normalize(planar_front)
    except ValueError as exc:
        raise ValueError("Analysis front axis cannot be parallel to up axis") from exc

    rear = tuple(-value for value in front)
    right = _normalize(_cross(rear, up))
    # Recompute rear from the other two unit axes to remove floating-point
    # projection drift and keep the basis strictly right-handed.
    rear = _normalize(_cross(up, right))
    return right, rear, up


def _analysis_to_world(
    vector: tuple[float, float, float],
    basis: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ],
) -> tuple[float, float, float]:
    """Map an analysis-space vector into the unchanged USD world space."""

    right, rear, up = basis
    return tuple(
        vector[0] * right[index] + vector[1] * rear[index] + vector[2] * up[index]
        for index in range(3)
    )


def _showcase_scene_spec(
    *,
    minimum: tuple[float, float, float],
    maximum: tuple[float, float, float],
    center: tuple[float, float, float],
    diagonal: float,
    basis: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ],
) -> dict[str, Any]:
    """Return world-space geometry and lighting for the opt-in showcase mode.

    Keeping this calculation independent of USD/Kit makes the unusual X-up
    assets just as predictable as conventional Z-up assets and lets the scene
    placement be covered by normal unit tests.
    """

    right, rear, up = basis
    bbox_corners = [
        (x, y, z)
        for x in (minimum[0], maximum[0])
        for y in (minimum[1], maximum[1])
        for z in (minimum[2], maximum[2])
    ]
    lowest_offset = min(
        _dot(
            tuple(corner[index] - center[index] for index in range(3)),
            up,
        )
        for corner in bbox_corners
    )
    ground_offset = lowest_offset - max(diagonal * 0.012, 1e-4)
    ground_center = tuple(
        center[index] + up[index] * ground_offset for index in range(3)
    )
    half_size = diagonal * 1.45

    def offset_position(
        origin: tuple[float, float, float],
        analysis_offset: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        world_offset = _analysis_to_world(analysis_offset, basis)
        return tuple(
            origin[index] + world_offset[index] * diagonal for index in range(3)
        )

    ground_corners = []
    for right_sign, rear_sign in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        ground_corners.append(
            tuple(
                ground_center[index]
                + right[index] * half_size * right_sign
                + rear[index] * half_size * rear_sign
                for index in range(3)
            )
        )

    return {
        "ground_center": ground_center,
        "ground_corners": ground_corners,
        "ground_normal": up,
        "ground_half_size": half_size,
        "lights": [
            {
                "name": "Key",
                "position": offset_position(center, (1.15, -1.35, 1.65)),
                "color": (1.0, 0.89, 0.78),
                "intensity": 65000.0,
                "radius": diagonal * 0.18,
            },
            {
                "name": "Fill",
                "position": offset_position(center, (-1.25, -0.55, 0.85)),
                "color": (0.72, 0.83, 1.0),
                "intensity": 35000.0,
                "radius": diagonal * 0.24,
            },
            {
                "name": "Rim",
                "position": offset_position(center, (0.30, 1.35, 1.25)),
                "color": (0.84, 0.91, 1.0),
                "intensity": 45000.0,
                "radius": diagonal * 0.16,
            },
        ],
        "dome": {"intensity": 340.0, "color": (0.78, 0.83, 0.93)},
    }


def _add_showcase_scene(stage, spec: dict[str, Any]) -> dict[str, Any]:
    """Author an unsaved ground and three-point rig on the open Kit stage."""

    from pxr import Gf, UsdGeom, UsdLux

    root_path = "/__QwenShowcaseRuntime"
    suffix = 1
    while stage.GetPrimAtPath(root_path).IsValid():
        root_path = f"/__QwenShowcaseRuntime_{suffix}"
        suffix += 1
    UsdGeom.Xform.Define(stage, root_path)

    ground_path = f"{root_path}/Ground"
    ground = UsdGeom.Mesh.Define(stage, ground_path)
    ground.CreatePointsAttr([Gf.Vec3f(*point) for point in spec["ground_corners"]])
    ground.CreateFaceVertexCountsAttr([4])
    ground.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    ground.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    ground.CreateDoubleSidedAttr(True)
    ground.CreateNormalsAttr([Gf.Vec3f(*spec["ground_normal"])])
    ground.SetNormalsInterpolation(UsdGeom.Tokens.constant)
    ground.CreateDisplayColorAttr([Gf.Vec3f(0.18, 0.20, 0.23)])

    dome_path = f"{root_path}/Dome"
    dome = UsdLux.DomeLight.Define(stage, dome_path)
    dome.CreateIntensityAttr(float(spec["dome"]["intensity"]))
    dome.CreateColorAttr(Gf.Vec3f(*spec["dome"]["color"]))

    light_paths = []
    for light_spec in spec["lights"]:
        light_path = f"{root_path}/{light_spec['name']}"
        light = UsdLux.SphereLight.Define(stage, light_path)
        light.CreateIntensityAttr(float(light_spec["intensity"]))
        light.CreateColorAttr(Gf.Vec3f(*light_spec["color"]))
        light.CreateRadiusAttr(float(light_spec["radius"]))
        light.CreateNormalizeAttr(True)
        UsdGeom.Xformable(light).AddTranslateOp().Set(Gf.Vec3d(*light_spec["position"]))
        light_paths.append(light_path)

    return {
        "runtime_root": root_path,
        "ground": ground_path,
        "dome": dome_path,
        "lights": light_paths,
    }


def _clear_render_evidence(registry: dict[str, Any]) -> None:
    """Remove stale render paths before producing a replacement evidence set."""

    for part in registry["parts"]:
        if isinstance(part, dict):
            part["renders"] = []
            part.pop("isolated_evidence", None)
    registry.pop("render_set", None)


def _camera_up_axis(
    direction: tuple[float, float, float],
    requested_up: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Return a stable camera up vector without changing source geometry."""

    normalized_direction = _normalize(direction)
    normalized_up = _normalize(requested_up)
    alignment = abs(
        sum(normalized_direction[index] * normalized_up[index] for index in range(3))
    )
    if alignment < 0.98:
        return normalized_up

    # A look-at camera cannot use an up vector parallel to its viewing axis.
    # Pick the world basis least aligned with this particular view.  These are
    # effectively the physical top/bottom views, so preserving a stable roll
    # is more useful than forcing an invalid requested up axis.
    candidates = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    return min(
        candidates,
        key=lambda axis: abs(
            sum(normalized_direction[index] * axis[index] for index in range(3))
        ),
    )


def _apply_radial_distortion(
    pixels: "Any",
    *,
    k1: float,
    k2: float,
    interpolation: int,
) -> "Any":
    """Apply the camera contract's Brown radial warp to one rendered plane."""

    if math.isclose(k1, 0.0, abs_tol=1e-12) and math.isclose(k2, 0.0, abs_tol=1e-12):
        return pixels
    height, width = pixels.shape[:2]
    map_x, map_y = _radial_distortion_maps(
        height,
        width,
        k1=k1,
        k2=k2,
    )
    source = pixels
    restore_dtype = None
    # Replicator semantic IDs are commonly uint32, a dtype OpenCV remap does
    # not accept.  IDs are small deterministic annotator labels, so bridge via
    # int32 and restore the exact contract dtype after nearest-neighbor remap.
    if pixels.dtype == np.uint32:
        if pixels.size and int(np.max(pixels)) > np.iinfo(np.int32).max:
            raise ValueError("Semantic ID exceeds the radial remap int32 range")
        source = pixels.astype(np.int32)
        restore_dtype = pixels.dtype
    distorted = cv2.remap(
        source,
        map_x,
        map_y,
        interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return (
        distorted.astype(restore_dtype, copy=False)
        if restore_dtype is not None
        else distorted
    )


def _radial_distortion_maps(
    height: int,
    width: int,
    *,
    k1: float,
    k2: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return inverse maps for the Brown radial model used by OpenCV remap."""

    x, y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    scale = 0.5 * float(max(width, height))
    distorted_x = (x - 0.5 * (width - 1)) / scale
    distorted_y = (y - 0.5 * (height - 1)) / scale
    source_x = distorted_x.copy()
    source_y = distorted_y.copy()
    # Fixed-point inversion is stable over the deliberately bounded contract.
    for _ in range(8):
        radius2 = source_x * source_x + source_y * source_y
        factor = 1.0 + float(k1) * radius2 + float(k2) * radius2 * radius2
        valid = np.abs(factor) > 1e-6
        source_x = np.where(valid, distorted_x / factor, distorted_x)
        source_y = np.where(valid, distorted_y / factor, distorted_y)
    return (
        (0.5 * (width - 1) + source_x * scale).astype(np.float32),
        (0.5 * (height - 1) + source_y * scale).astype(np.float32),
    )


def _part_from_label(value: Any) -> str | None:
    if isinstance(value, dict):
        for key in ("part", "class"):
            label = value.get(key)
            if isinstance(label, str):
                normalized = label.split(",")[-1].strip()
                if re.fullmatch(r"p\d+", normalized, re.IGNORECASE):
                    return normalized.upper()
                return normalized
    return None


def _part_color(part_id: str) -> tuple[int, int, int]:
    number = int(part_id[1:]) if part_id[1:].isdigit() else sum(map(ord, part_id))
    # Golden-angle hue stepping keeps neighboring stable IDs visually distinct.
    hue = (number * 0.618033988749895) % 1.0
    import colorsys

    red, green, blue = colorsys.hsv_to_rgb(hue, 0.72, 0.95)
    return int(red * 255), int(green * 255), int(blue * 255)


def _font(size: int):
    from PIL import ImageFont

    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _raw_mask_image(mask_by_part, width: int, height: int):
    """Return a lossless stable-color part-ID image with no annotations."""

    import numpy as np
    from PIL import Image

    pixels = np.full((height, width, 3), 28, dtype=np.uint8)
    for part_id, mask in mask_by_part.items():
        pixels[mask] = _part_color(part_id)
    return Image.fromarray(pixels, mode="RGB")


def _label_mask_image(mask_by_part, width: int, height: int):
    """Return the human-readable part-ID overview used in Qwen prompts."""

    import numpy as np
    from PIL import ImageDraw

    image = _raw_mask_image(mask_by_part, width, height)
    draw = ImageDraw.Draw(image)
    font = _font(max(12, width // 48))
    for part_id, mask in mask_by_part.items():
        ys, xs = np.nonzero(mask)
        if len(xs) < 12:
            continue
        x = int(np.median(xs))
        y = int(np.median(ys))
        box = draw.textbbox((x, y), part_id, font=font, anchor="mm")
        draw.rectangle(box, fill=(0, 0, 0))
        draw.text((x, y), part_id, fill=(255, 255, 255), font=font, anchor="mm")
    return image


def _crop_with_margin(image, mask, margin_ratio: float = 0.12):
    import numpy as np

    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    width = int(xs.max() - xs.min() + 1)
    height = int(ys.max() - ys.min() + 1)
    margin = max(8, int(max(width, height) * margin_ratio))
    left = max(0, int(xs.min()) - margin)
    top = max(0, int(ys.min()) - margin)
    right = min(image.width, int(xs.max()) + margin + 1)
    bottom = min(image.height, int(ys.max()) + margin + 1)
    return image.crop((left, top, right, bottom))


def _highlighted_context_crop(
    image,
    mask,
    part_id: str,
    margin_ratio: float = 0.38,
):
    """Show one neutral target part in context with a thin red outline.

    The image is an identity aid for the vision model, not material evidence:
    non-target geometry is darkened/desaturated, while the target keeps the
    neutral CAD render.  A larger margin than the ordinary crop preserves the
    surrounding assembly relationships needed to match a single photograph.
    """

    import numpy as np
    from PIL import Image, ImageDraw, ImageFilter

    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    width = int(xs.max() - xs.min() + 1)
    height = int(ys.max() - ys.min() + 1)
    margin = max(18, int(max(width, height) * margin_ratio))
    left = max(0, int(xs.min()) - margin)
    top = max(0, int(ys.min()) - margin)
    right = min(image.width, int(xs.max()) + margin + 1)
    bottom = min(image.height, int(ys.max()) + margin + 1)

    source = np.asarray(image.convert("RGB"))[top:bottom, left:right].copy()
    target = np.asarray(mask[top:bottom, left:right], dtype=bool)
    gray = np.dot(source[..., :3], (0.299, 0.587, 0.114))
    context = np.clip(gray[..., None] * 0.48 + 22.0, 0, 255).astype(np.uint8)
    result = np.repeat(context, 3, axis=2)
    result[target] = source[target]

    mask_image = Image.fromarray(target.astype(np.uint8) * 255, mode="L")
    outline_size = max(3, (max(source.shape[:2]) // 80) * 2 + 1)
    dilated = np.asarray(mask_image.filter(ImageFilter.MaxFilter(outline_size))) > 0
    outline = dilated & ~target
    result[outline] = (235, 55, 45)

    label_height = max(28, result.shape[0] // 10)
    output = Image.new(
        "RGB", (result.shape[1], result.shape[0] + label_height), (24, 24, 24)
    )
    output.paste(Image.fromarray(result, mode="RGB"), (0, label_height))
    draw = ImageDraw.Draw(output)
    draw.text(
        (8, label_height // 2),
        f"TARGET {part_id} - GEOMETRY ONLY",
        fill=(255, 255, 255),
        font=_font(max(12, label_height // 2)),
        anchor="lm",
    )
    return output


def _isolated_target_crop(
    image,
    mask,
    part_id: str,
    view_id: str,
    *,
    canvas_size: int = ISOLATED_EVIDENCE_CANVAS_SIZE,
    target_long_edge: int = ISOLATED_EVIDENCE_TARGET_LONG_EDGE,
):
    """Return a material-neutral, mask-isolated and enlarged target view.

    The original semantic mask remains the only source of geometry.  Target
    RGB is converted to grayscale so an imported CAD display colour cannot
    leak into Qwen's material decision.  Upscaling improves shape readability
    but is recorded separately from the original visible-pixel count; callers
    must never treat normalized pixels as new photo observations.
    """

    import numpy as np
    from PIL import Image, ImageDraw, ImageFilter

    if canvas_size < 128:
        raise ValueError("isolated evidence canvas_size must be at least 128")
    if not 32 <= target_long_edge <= canvas_size - 16:
        raise ValueError(
            "isolated evidence target_long_edge must fit inside the canvas"
        )
    ys, xs = np.nonzero(mask)
    source_pixels = int(len(xs))
    if source_pixels == 0:
        return None

    left = int(xs.min())
    top = int(ys.min())
    right = int(xs.max()) + 1
    bottom = int(ys.max()) + 1
    source = np.asarray(image.convert("RGB"))[top:bottom, left:right].copy()
    target = np.asarray(mask[top:bottom, left:right], dtype=bool)
    source_height, source_width = target.shape
    scale = target_long_edge / max(source_width, source_height)
    normalized_size = (
        max(1, int(round(source_width * scale))),
        max(1, int(round(source_height * scale))),
    )

    luminance = np.dot(source[..., :3], (0.299, 0.587, 0.114))
    neutral = np.repeat(luminance[..., None], 3, axis=2)
    neutral = np.clip(neutral * 0.82 + 32.0, 0, 255).astype(np.uint8)
    neutral_image = Image.fromarray(neutral, mode="RGB").resize(
        normalized_size,
        Image.Resampling.LANCZOS,
    )
    normalized_mask_image = Image.fromarray(
        target.astype(np.uint8) * 255,
        mode="L",
    ).resize(normalized_size, Image.Resampling.NEAREST)
    normalized_mask = np.asarray(normalized_mask_image) > 0
    normalized_pixels = int(np.count_nonzero(normalized_mask))

    header_height = 38
    canvas = Image.new(
        "RGB",
        (canvas_size, canvas_size + header_height),
        (116, 116, 116),
    )
    target_layer = np.full(
        (normalized_size[1], normalized_size[0], 3),
        116,
        dtype=np.uint8,
    )
    neutral_pixels = np.asarray(neutral_image)
    target_layer[normalized_mask] = neutral_pixels[normalized_mask]
    target_image = Image.fromarray(target_layer, mode="RGB")
    outline_size = max(3, (target_long_edge // 70) * 2 + 1)
    dilated = (
        np.asarray(normalized_mask_image.filter(ImageFilter.MaxFilter(outline_size)))
        > 0
    )
    outline = dilated & ~normalized_mask
    outlined = np.asarray(target_image).copy()
    outlined[outline] = (235, 55, 45)
    target_image = Image.fromarray(outlined, mode="RGB")
    paste_x = (canvas_size - normalized_size[0]) // 2
    paste_y = header_height + (canvas_size - normalized_size[1]) // 2
    canvas.paste(target_image, (paste_x, paste_y))

    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, canvas_size, header_height), fill=(24, 24, 24))
    draw.text(
        (8, header_height // 2),
        f"TARGET {part_id} | {view_id} | SOURCE {source_pixels}px",
        fill=(255, 255, 255),
        font=_font(max(12, header_height // 2 - 1)),
        anchor="lm",
    )
    return (
        canvas,
        {
            "view_id": view_id,
            "source_visible_pixels": source_pixels,
            "normalized_visible_pixels": normalized_pixels,
            "normalization_scale": round(float(scale), 8),
            "source_bbox_xyxy": [left, top, right, bottom],
            "target_long_edge": target_long_edge,
            "canvas_size": [canvas_size, canvas_size + header_height],
            "material_neutralized": True,
            "background_removed": True,
        },
    )


def _make_multiview_part_evidence(
    *,
    part_id: str,
    records: list[dict[str, Any]],
    context_path: str | None,
    output_path: Path,
) -> dict[str, Any]:
    """Pack one assembly context plus up to three isolated target views."""

    from PIL import Image, ImageDraw

    if not records:
        raise ValueError("multiview part evidence requires at least one view")
    selected = sorted(
        records,
        key=lambda item: (
            -int(item["source_visible_pixels"]),
            str(item["view_id"]),
        ),
    )[:ISOLATED_EVIDENCE_MAX_VIEWS]
    cell_size = 320
    canvas = Image.new("RGB", (cell_size * 2, cell_size * 2), (232, 232, 232))
    draw = ImageDraw.Draw(canvas)
    panels: list[tuple[str, Path]] = []
    if context_path is not None:
        panels.append(("ASSEMBLY CONTEXT", Path(context_path)))
    panels.extend(
        (
            (
                f"ISOLATED {record['view_id']} "
                f"({record['source_visible_pixels']}px source)",
                Path(str(record["path"])),
            )
        )
        for record in selected
    )
    for index, (label, source_path) in enumerate(panels[:4]):
        with Image.open(source_path.expanduser().resolve(strict=True)) as opened:
            image = opened.convert("RGB")
        header_height = 30
        available = cell_size - 14
        available_height = cell_size - header_height - 10
        scale = min(available / image.width, available_height / image.height)
        resized = image.resize(
            (
                max(1, int(round(image.width * scale))),
                max(1, int(round(image.height * scale))),
            ),
            Image.Resampling.LANCZOS,
        )
        column = index % 2
        row = index // 2
        left = column * cell_size
        top = row * cell_size
        x = left + (cell_size - resized.width) // 2
        y = top + header_height + (available_height - resized.height) // 2
        canvas.paste(resized, (x, y))
        draw.rectangle(
            (left, top, left + cell_size - 1, top + header_height),
            fill=(28, 28, 28),
        )
        draw.text(
            (left + 8, top + header_height // 2),
            label,
            fill=(255, 255, 255),
            font=_font(14),
            anchor="lm",
        )
        draw.rectangle(
            (left, top, left + cell_size - 1, top + cell_size - 1),
            outline=(120, 120, 120),
            width=2,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)

    source_pixels_by_view = {
        str(record["view_id"]): int(record["source_visible_pixels"])
        for record in selected
    }
    normalized_pixels_by_view = {
        str(record["view_id"]): int(record["normalized_visible_pixels"])
        for record in selected
    }
    eligible_views = sorted(
        view_id
        for view_id, pixels in source_pixels_by_view.items()
        if pixels >= ISOLATED_EVIDENCE_SOURCE_PIXEL_FLOOR
    )
    return {
        "schema_version": ISOLATED_EVIDENCE_SCHEMA_VERSION,
        "path": str(output_path),
        "sha256": _sha256(output_path),
        "context_path": context_path,
        "selected_view_ids": [str(record["view_id"]) for record in selected],
        "source_visible_pixels_by_view": source_pixels_by_view,
        "normalized_visible_pixels_by_view": normalized_pixels_by_view,
        "source_max_visible_pixels": max(source_pixels_by_view.values()),
        "normalized_max_visible_pixels": max(normalized_pixels_by_view.values()),
        "source_evidence_view_count": len(eligible_views),
        "source_evidence_view_ids": eligible_views,
        "source_pixel_floor": ISOLATED_EVIDENCE_SOURCE_PIXEL_FLOOR,
        "target_long_edge": ISOLATED_EVIDENCE_TARGET_LONG_EDGE,
        "material_neutralized": True,
        "background_removed": True,
    }


def _make_contact_sheets(crops_by_part, output_dir: Path, cell_size: int = 220):
    from PIL import Image, ImageDraw

    sheet_paths: list[str] = []
    items = sorted(crops_by_part.items())
    columns = 4
    rows_per_sheet = 4
    page_size = columns * rows_per_sheet
    font = _font(18)
    for page_index in range(0, len(items), page_size):
        page = items[page_index : page_index + page_size]
        sheet = Image.new(
            "RGB", (columns * cell_size, rows_per_sheet * cell_size), (245, 245, 245)
        )
        draw = ImageDraw.Draw(sheet)
        for slot, (part_id, crop_path) in enumerate(page):
            crop = Image.open(crop_path).convert("RGB")
            target_width = cell_size - 16
            target_height = cell_size - 42
            scale = min(target_width / crop.width, target_height / crop.height)
            resized = (
                max(1, int(round(crop.width * scale))),
                max(1, int(round(crop.height * scale))),
            )
            crop = crop.resize(resized, Image.Resampling.LANCZOS)
            col = slot % columns
            row = slot // columns
            x = col * cell_size + (cell_size - crop.width) // 2
            y = row * cell_size + 30 + (cell_size - 38 - crop.height) // 2
            sheet.paste(crop, (x, y))
            draw.text(
                (col * cell_size + cell_size // 2, row * cell_size + 16),
                part_id,
                fill=(20, 20, 20),
                font=font,
                anchor="mm",
            )
        output = output_dir / f"contact_sheet_{page_index // page_size + 1:02d}.png"
        sheet.save(output)
        sheet_paths.append(str(output))
    return sheet_paths


def _render_part_views_once(
    *,
    registry_path: str | Path,
    output_dir: str | Path,
    resolution: int = 768,
    view_names: list[str] | None = None,
    rt_subframes: int = 8,
    analysis_up_axis: tuple[float, float, float] = (0.0, 0.0, 1.0),
    analysis_front_axis: tuple[float, float, float] = (0.0, -1.0, 0.0),
    lighting_profile: str = "geometry",
    showcase: bool = False,
    generate_part_evidence: bool = True,
    custom_view_specs_path: str | Path | None = None,
    assembly_pose_overrides_path: str | Path | None = None,
    progress_callback: ProgressCallback | None = None,
    _lifecycle: _RenderLifecycle,
) -> dict[str, Any]:
    if (
        isinstance(rt_subframes, bool)
        or not isinstance(rt_subframes, int)
        or rt_subframes <= 0
    ):
        raise ValueError("RT subframes must be a positive integer")

    import numpy as np
    from PIL import Image
    from pxr import Usd, UsdGeom
    import omni.replicator.core as rep
    import omni.kit.app
    import omni.usd
    from isaacsim.core.utils.semantics import add_labels, add_update_semantics

    registry_file, registry = _load_registry(registry_path)
    asset_path = Path(registry.get("asset_usd", "")).expanduser().resolve(strict=True)
    source_sha256_before = _sha256(asset_path) if showcase else None
    destination = Path(output_dir).expanduser().resolve()
    rgb_dir = destination / "rgb"
    id_dir = destination / "part_ids"
    annotated_id_dir = destination / "part_ids_annotated"
    crop_dir = destination / "parts"
    highlight_dir = destination / "part_highlights"
    isolated_dir = destination / "part_isolated"
    evidence_dir = destination / "part_evidence"
    required_directories = [rgb_dir, id_dir, annotated_id_dir]
    if generate_part_evidence:
        required_directories.extend(
            [crop_dir, highlight_dir, isolated_dir, evidence_dir]
        )
    for directory in required_directories:
        directory.mkdir(parents=True, exist_ok=True)

    custom_view_specs = (
        _load_custom_view_specs(custom_view_specs_path)
        if custom_view_specs_path is not None
        else None
    )
    assembly_pose_overrides = (
        _load_assembly_pose_overrides(assembly_pose_overrides_path)
        if assembly_pose_overrides_path is not None
        else []
    )
    if custom_view_specs is None:
        view_directions, view_presets = _resolve_view_directions(view_names)
    else:
        view_directions = {
            name: spec["analysis_direction"]
            for name, spec in custom_view_specs.items()
        }
        view_presets = []
    names = list(view_directions)
    part_evidence_view_ids = (
        (set(VIEW_DIRECTIONS) if view_presets else set(names))
        if generate_part_evidence
        else set()
    )
    if resolution < 128:
        raise ValueError("Resolution must be at least 128")
    if lighting_profile not in EVIDENCE_LIGHTING_PROFILES:
        raise ValueError(
            "Unknown evidence lighting profile: "
            f"{lighting_profile!r}; expected one of "
            f"{', '.join(EVIDENCE_LIGHTING_PROFILES)}"
        )
    if showcase and lighting_profile != "geometry":
        raise ValueError(
            "Showcase rendering cannot be combined with material-neutral "
            "evidence lighting"
        )
    normalized_analysis_up = _normalize(analysis_up_axis)
    normalized_analysis_front = _normalize(analysis_front_axis)
    analysis_basis = _analysis_basis(normalized_analysis_up, normalized_analysis_front)

    # A rendered registry may be used as input for a re-render.  Never carry
    # its old horizontal crops into a newly oriented evidence set.
    _clear_render_evidence(registry)

    context = omni.usd.get_context()
    if not context.open_stage(str(asset_path)):
        raise RuntimeError(f"Unable to open USD stage in Kit: {asset_path}")
    _lifecycle.register_open_stage(context)
    stage = context.get_stage()
    if stage is None:
        raise RuntimeError("Kit did not return an open stage")

    part_by_path = {
        item["prim_path"]: item
        for item in registry["parts"]
        if isinstance(item, dict)
        and isinstance(item.get("prim_path"), str)
        and isinstance(item.get("part_id"), str)
    }
    for prim_path, part in part_by_path.items():
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsA(UsdGeom.Mesh):
            raise ValueError(
                f"Registry prim is not a Mesh in the opened stage: {prim_path}"
            )
        try:
            add_labels(prim, labels=[part["part_id"]], instance_name="part")
        except Exception:
            # Some referenced look-wrapper stages cannot resolve the newer
            # LabelsAPI type in Isaac Sim 5.0.  The legacy API remains readable
            # by Replicator and is authored only in this unsaved in-memory stage.
            add_update_semantics(prim, part["part_id"], type_label="part")
    for _ in range(3):
        omni.kit.app.get_app().update()

    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=True,
    )
    mesh_prims = [stage.GetPrimAtPath(path) for path in part_by_path]
    union = None
    for prim in mesh_prims:
        bounds = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
        if union is None:
            union = bounds
        else:
            union.UnionWith(bounds)
    if union is None or union.IsEmpty():
        raise ValueError("Unable to calculate asset bounds")
    minimum = union.GetMin()
    maximum = union.GetMax()
    center = tuple(float((minimum[i] + maximum[i]) * 0.5) for i in range(3))
    extents = tuple(float(maximum[i] - minimum[i]) for i in range(3))
    diagonal = max(math.sqrt(sum(value * value for value in extents)), 1e-3)
    camera_distance = diagonal * 2.15
    if assembly_pose_overrides:
        _apply_assembly_pose_overrides(
            stage=stage,
            overrides=assembly_pose_overrides,
            maximum_translation=0.20 * diagonal,
        )
        for _ in range(3):
            omni.kit.app.get_app().update()

    showcase_runtime = None
    if showcase:
        showcase_spec = _showcase_scene_spec(
            minimum=tuple(float(minimum[index]) for index in range(3)),
            maximum=tuple(float(maximum[index]) for index in range(3)),
            center=center,
            diagonal=diagonal,
            basis=analysis_basis,
        )
        showcase_runtime = _add_showcase_scene(stage, showcase_spec)
        for _ in range(3):
            omni.kit.app.get_app().update()
    else:
        if lighting_profile == "material-neutral":
            # Material QA needs an orientation-stable response. A white dome
            # avoids the fixed distant light making identical paint appear
            # bright in one canonical view and dark in another.
            rep.create.light(
                light_type="dome",
                intensity=400.0,
                color=(1.0, 1.0, 1.0),
            )
        else:
            # Directional contrast remains useful for Qwen geometry reading
            # and spatial registration. Material QA opts into the profile
            # above instead of changing this established evidence rig.
            rep.create.light(
                light_type="dome",
                intensity=260.0,
                color=(0.86, 0.90, 1.0),
            )
            rep.create.light(
                light_type="distant",
                rotation=(45, -30, 25),
                intensity=1600.0,
            )

    captures = []
    for name in names:
        analysis_direction = _normalize(view_directions[name])
        custom_spec = (
            custom_view_specs.get(name)
            if custom_view_specs is not None
            else None
        )
        analysis_camera_up = (
            custom_spec["analysis_up_axis"]
            if custom_spec is not None
            else (
                (1.0, 0.0, 0.0)
                if "_toproll" in name
                else _camera_up_axis(analysis_direction, (0.0, 0.0, 1.0))
            )
        )
        world_direction = _normalize(
            _analysis_to_world(analysis_direction, analysis_basis)
        )
        camera_up = _normalize(_analysis_to_world(analysis_camera_up, analysis_basis))
        if name.endswith("_r180"):
            camera_up = tuple(-value for value in camera_up)
        roll_degrees = (
            float(custom_spec["roll_degrees"])
            if custom_spec is not None
            else 0.0
        )
        effective_distance = (
            diagonal * float(custom_spec["distance_multiplier"])
            if custom_spec is not None
            else camera_distance
        )
        focal_length = (
            float(custom_spec["focal_length_mm"])
            if custom_spec is not None
            else 45.0
        )
        projection_mode = (
            str(custom_spec["projection_mode"])
            if custom_spec is not None
            else "perspective"
        )
        orthographic_span_multiplier = (
            float(custom_spec["orthographic_span_multiplier"])
            if custom_spec is not None
            else 2.0
        )
        position = tuple(
            center[i] + world_direction[i] * effective_distance for i in range(3)
        )
        # A general perspective camera is not required to point exactly at the
        # asset bounding-box center.  The two bounded target offsets complete
        # the rigid camera extrinsics without modifying the source USD.  They
        # are measured in asset-diagonal units in the camera image plane.
        view_forward = tuple(-value for value in world_direction)
        camera_right = _normalize(_cross(view_forward, camera_up))
        target_offset_u = (
            float(custom_spec["target_offset_u"])
            if custom_spec is not None
            else 0.0
        )
        target_offset_v = (
            float(custom_spec["target_offset_v"])
            if custom_spec is not None
            else 0.0
        )
        principal_point_u = (
            float(custom_spec["principal_point_u"])
            if custom_spec is not None
            else 0.0
        )
        principal_point_v = (
            float(custom_spec["principal_point_v"])
            if custom_spec is not None
            else 0.0
        )
        radial_distortion_k1 = (
            float(custom_spec["radial_distortion_k1"])
            if custom_spec is not None
            else 0.0
        )
        radial_distortion_k2 = (
            float(custom_spec["radial_distortion_k2"])
            if custom_spec is not None
            else 0.0
        )
        look_at_target = tuple(
            center[i]
            + diagonal * target_offset_u * camera_right[i]
            + diagonal * target_offset_v * camera_up[i]
            for i in range(3)
        )
        camera = rep.create.camera(
            name=f"Qwen_{name}",
            position=position,
            look_at=look_at_target,
            look_at_up_axis=camera_up,
            focal_length=focal_length,
            clipping_range=(max(diagonal * 0.001, 1e-5), diagonal * 200.0),
        )
        if projection_mode == "orthographic":
            # USD orthographic aperture is expressed in tenths of a stage
            # unit.  The downstream whole-image similarity resolves crop
            # scale, so a two-diagonal field safely contains every pose while
            # projection type alone determines whether depth causes parallax.
            camera_root = camera.get_output_prims()["prims"][0]
            camera_prim = (
                camera_root
                if camera_root.IsA(UsdGeom.Camera)
                else next(
                    (
                        child
                        for child in camera_root.GetChildren()
                        if child.IsA(UsdGeom.Camera)
                    ),
                    None,
                )
            )
            if camera_prim is None:
                raise RuntimeError(
                    f"Replicator camera {name!r} has no UsdGeom.Camera prim"
                )
            usd_camera = UsdGeom.Camera(camera_prim)
            aperture = diagonal * orthographic_span_multiplier * 10.0
            usd_camera.GetProjectionAttr().Set(UsdGeom.Tokens.orthographic)
            usd_camera.GetHorizontalApertureAttr().Set(aperture)
            usd_camera.GetVerticalApertureAttr().Set(aperture)
        else:
            camera_root = camera.get_output_prims()["prims"][0]
            camera_prim = (
                camera_root
                if camera_root.IsA(UsdGeom.Camera)
                else next(
                    (
                        child
                        for child in camera_root.GetChildren()
                        if child.IsA(UsdGeom.Camera)
                    ),
                    None,
                )
            )
            if camera_prim is None:
                raise RuntimeError(
                    f"Replicator camera {name!r} has no UsdGeom.Camera prim"
                )
            usd_camera = UsdGeom.Camera(camera_prim)
        # USD aperture offsets are the physical principal point for both
        # perspective and orthographic cameras.  Values in the camera
        # contract are fractions of half the active sensor gate.
        horizontal_aperture = float(usd_camera.GetHorizontalApertureAttr().Get())
        vertical_aperture = float(usd_camera.GetVerticalApertureAttr().Get())
        usd_camera.GetHorizontalApertureOffsetAttr().Set(
            principal_point_u * 0.5 * horizontal_aperture
        )
        usd_camera.GetVerticalApertureOffsetAttr().Set(
            principal_point_v * 0.5 * vertical_aperture
        )
        render_product = rep.create.render_product(
            camera, (resolution, resolution), name=f"Qwen_{name}"
        )
        rgb = rep.AnnotatorRegistry.get_annotator("rgb")
        _lifecycle.attach_annotator(rgb, render_product)
        segmentation = rep.AnnotatorRegistry.get_annotator(
            "semantic_segmentation",
            init_params={"semanticTypes": ["part"], "colorize": False},
        )
        _lifecycle.attach_annotator(segmentation, render_product)
        captures.append(
            (
                name,
                position,
                analysis_direction,
                analysis_camera_up,
                world_direction,
                camera_up,
                focal_length,
                effective_distance / diagonal,
                target_offset_u,
                target_offset_v,
                roll_degrees,
                principal_point_u,
                principal_point_v,
                radial_distortion_k1,
                radial_distortion_k2,
                look_at_target,
                projection_mode,
                orthographic_span_multiplier,
                rgb,
                segmentation,
                render_product,
            )
        )

    # The first step initializes render vars and semantic buffers; the second
    # produces stable data on both Isaac Sim 5.x and 6.x.
    capture_steps = (1, 2)
    for _ in _counted_progress_items(
        capture_steps,
        progress_callback=progress_callback,
        stage=RENDER_CAPTURE_PROGRESS_STAGE,
        unit="steps",
        detail=lambda step: f"Replicator capture step {step}/2",
    ):
        _lifecycle.step(
            rep.orchestrator,
            rt_subframes=rt_subframes,
            delta_time=0.0,
        )

    crop_candidates: dict[str, tuple[int, str]] = {}
    highlight_candidates: dict[str, tuple[int, str]] = {}
    isolated_records_by_part: dict[str, list[dict[str, Any]]] = {}
    view_records = []
    allowed_part_ids = {part["part_id"] for part in part_by_path.values()}
    for (
        name,
        camera_position,
        analysis_direction,
        analysis_camera_up,
        world_direction,
        camera_up,
        focal_length,
        distance_multiplier,
        target_offset_u,
        target_offset_v,
        roll_degrees,
        principal_point_u,
        principal_point_v,
        radial_distortion_k1,
        radial_distortion_k2,
        look_at_target,
        projection_mode,
        orthographic_span_multiplier,
        rgb_annotator,
        segmentation_annotator,
        _,
    ) in _counted_progress_items(
        captures,
        progress_callback=progress_callback,
        stage=RENDER_VIEWS_PROGRESS_STAGE,
        unit="views",
        detail=lambda capture: f"Rendered view {capture[0]}",
    ):
        rgb_array = np.asarray(rgb_annotator.get_data())
        if rgb_array.ndim != 3 or rgb_array.shape[2] < 3:
            raise RuntimeError(f"Invalid RGB output for view {name}: {rgb_array.shape}")

        segmentation = segmentation_annotator.get_data()
        raw_ids = np.asarray(segmentation.get("data"))
        if raw_ids.ndim == 3:
            raw_ids = raw_ids[:, :, 0]
        id_to_labels = segmentation.get("info", {}).get("idToLabels", {})
        rgb_pixels = rgb_array[:, :, :3].astype(np.uint8)
        rgb_pixels = _apply_radial_distortion(
            rgb_pixels,
            k1=radial_distortion_k1,
            k2=radial_distortion_k2,
            interpolation=cv2.INTER_LINEAR,
        )
        raw_ids = _apply_radial_distortion(
            raw_ids,
            k1=radial_distortion_k1,
            k2=radial_distortion_k2,
            interpolation=cv2.INTER_NEAREST,
        )
        # Rebuild masks from the same distorted Part-ID plane that is written
        # below; RGB and semantic evidence remain pixel-aligned.
        mask_by_part = {}
        for raw_id, labels in id_to_labels.items():
            part_id = _part_from_label(labels)
            if part_id not in allowed_part_ids:
                continue
            try:
                numeric_id = int(raw_id)
            except (TypeError, ValueError):
                continue
            mask = raw_ids == numeric_id
            if int(mask.sum()) > 0:
                mask_by_part[part_id] = mask
        if lighting_profile == "material-neutral":
            # Physical reference captures use a black viewport background.
            # Keep only labelled asset pixels so background/tone differences
            # cannot leak into material QA or its human-facing comparisons.
            foreground = np.zeros(raw_ids.shape[:2], dtype=bool)
            for mask in mask_by_part.values():
                foreground |= mask
            rgb_pixels = rgb_pixels.copy()
            rgb_pixels[~foreground] = 0
        rgb_image = Image.fromarray(rgb_pixels, mode="RGB")
        rgb_path = rgb_dir / f"{name}.png"
        rgb_image.save(rgb_path)

        actual_height, actual_width = raw_ids.shape[:2]
        raw_id_image = _raw_mask_image(mask_by_part, actual_width, actual_height)
        id_path = id_dir / f"{name}.png"
        raw_id_image.save(id_path)
        annotated_id_image = _label_mask_image(
            mask_by_part, actual_width, actual_height
        )
        annotated_id_path = annotated_id_dir / f"{name}.png"
        annotated_id_image.save(annotated_id_path)
        visible_parts = []
        for part_id, mask in mask_by_part.items():
            pixel_count = int(mask.sum())
            visible_parts.append({"part_id": part_id, "pixels": pixel_count})
            # Dense pose-bank samples are needed as whole-assembly RGB/Part-ID
            # evidence only.  Emitting per-part crops for every sample would
            # multiply a large CAD assembly into tens of thousands of files
            # without adding independent source-photo material evidence.
            if not generate_part_evidence or name not in part_evidence_view_ids:
                continue
            crop = _crop_with_margin(rgb_image, mask)
            if crop is None:
                continue
            crop_path = crop_dir / f"{part_id}_{name}.png"
            crop.save(crop_path)
            highlighted = _highlighted_context_crop(rgb_image, mask, part_id)
            highlight_path = highlight_dir / f"{part_id}_{name}.png"
            if highlighted is not None:
                highlighted.save(highlight_path)
            isolated = _isolated_target_crop(rgb_image, mask, part_id, name)
            isolated_path = isolated_dir / f"{part_id}_{name}.png"
            isolated_metadata: dict[str, Any] | None = None
            if isolated is not None:
                isolated_image, isolated_metadata = isolated
                isolated_image.save(isolated_path)
                isolated_records_by_part.setdefault(part_id, []).append(
                    {
                        **isolated_metadata,
                        "path": str(isolated_path),
                        "sha256": _sha256(isolated_path),
                    }
                )
            previous = crop_candidates.get(part_id)
            if previous is None or pixel_count > previous[0]:
                crop_candidates[part_id] = (pixel_count, str(crop_path))
                if highlighted is not None:
                    highlight_candidates[part_id] = (
                        pixel_count,
                        str(highlight_path),
                    )
            part = next(
                item for item in registry["parts"] if item["part_id"] == part_id
            )
            render_record = {
                "view_id": name,
                "image": str(crop_path),
                "visible_pixels": pixel_count,
            }
            if highlighted is not None:
                render_record["highlight_path"] = str(highlight_path)
            if isolated_metadata is not None:
                render_record["isolated_path"] = str(isolated_path)
                render_record["isolated_sha256"] = _sha256(isolated_path)
                render_record["isolated_normalized_visible_pixels"] = isolated_metadata[
                    "normalized_visible_pixels"
                ]
                render_record["isolated_normalization_scale"] = isolated_metadata[
                    "normalization_scale"
                ]
                render_record["isolated_material_neutralized"] = True
                render_record["isolated_background_removed"] = True
            part.setdefault("renders", []).append(render_record)
        view_records.append(
            {
                "view_id": name,
                "rgb": str(rgb_path),
                "part_ids": str(id_path),
                "part_ids_raw": str(id_path),
                "part_ids_annotated": str(annotated_id_path),
                "camera_position": list(camera_position),
                "analysis_direction": list(analysis_direction),
                "analysis_camera_up_axis": list(analysis_camera_up),
                "world_direction": list(world_direction),
                "camera_up_axis": list(camera_up),
                "focal_length_mm": float(focal_length),
                "camera_distance_multiplier": float(distance_multiplier),
                "camera_target_offset_u": float(target_offset_u),
                "camera_target_offset_v": float(target_offset_v),
                "camera_roll_degrees": float(roll_degrees),
                "camera_principal_point_u": float(principal_point_u),
                "camera_principal_point_v": float(principal_point_v),
                "camera_radial_distortion_k1": float(radial_distortion_k1),
                "camera_radial_distortion_k2": float(radial_distortion_k2),
                "camera_look_at_target": list(look_at_target),
                "camera_projection_mode": projection_mode,
                "camera_orthographic_span_multiplier": float(
                    orthographic_span_multiplier
                ),
                **(
                    {"camera_calibration": custom_view_specs[name].get("calibration")}
                    if custom_view_specs is not None
                    else {}
                ),
                "visible_parts": sorted(
                    visible_parts, key=lambda item: item["part_id"]
                ),
                "segmentation_ids": sorted(int(value) for value in np.unique(raw_ids)),
                "segmentation_labels": {
                    str(key): value for key, value in id_to_labels.items()
                },
            }
        )

    best_crops = {part_id: value[1] for part_id, value in crop_candidates.items()}
    best_highlights = {
        part_id: value[1] for part_id, value in highlight_candidates.items()
    }
    best_evidence: dict[str, str] = {}
    isolated_evidence_by_part: dict[str, dict[str, Any]] = {}
    part_records = {
        str(part["part_id"]): part
        for part in registry["parts"]
        if isinstance(part, dict) and isinstance(part.get("part_id"), str)
    }
    for part_id, records in sorted(isolated_records_by_part.items()):
        evidence_path = evidence_dir / f"{part_id}.png"
        evidence = _make_multiview_part_evidence(
            part_id=part_id,
            records=records,
            context_path=best_highlights.get(part_id),
            output_path=evidence_path,
        )
        isolated_evidence_by_part[part_id] = evidence
        best_evidence[part_id] = str(evidence_path)
        part_records[part_id]["isolated_evidence"] = evidence
    contact_sheets = _make_contact_sheets(best_crops, destination)
    source_sha256_after = _sha256(asset_path) if showcase else None
    if showcase and source_sha256_after != source_sha256_before:
        raise RuntimeError("Source USD changed during showcase rendering")
    registry["render_set"] = {
        "asset_usd": str(asset_path),
        "resolution": [resolution, resolution],
        "analysis_up_axis": list(normalized_analysis_up),
        "analysis_front_axis": list(normalized_analysis_front),
        "lighting_profile": lighting_profile,
        "requested_view_tokens": (
            list(names)
            if custom_view_specs is not None
            else list(view_names or VIEW_DIRECTIONS)
        ),
        "custom_view_specs": (
            str(Path(custom_view_specs_path).expanduser().resolve(strict=True))
            if custom_view_specs_path is not None
            else None
        ),
        "assembly_pose_overrides": (
            str(Path(assembly_pose_overrides_path).expanduser().resolve(strict=True))
            if assembly_pose_overrides_path is not None
            else None
        ),
        "assembly_pose_override_count": len(assembly_pose_overrides),
        "rt_subframes": rt_subframes,
        "expanded_view_count": len(names),
        "view_presets": view_presets,
        "part_evidence_view_ids": sorted(part_evidence_view_ids),
        "part_evidence_generated": generate_part_evidence,
        "analysis_basis_world": {
            "right": list(analysis_basis[0]),
            "rear": list(analysis_basis[1]),
            "up": list(analysis_basis[2]),
        },
        "views": view_records,
        "contact_sheets": contact_sheets,
        "best_crops": best_crops,
        "best_highlights": best_highlights,
        "best_evidence": best_evidence,
        "isolated_evidence_policy": {
            "schema_version": ISOLATED_EVIDENCE_SCHEMA_VERSION,
            "source_pixel_floor": ISOLATED_EVIDENCE_SOURCE_PIXEL_FLOOR,
            "target_long_edge": ISOLATED_EVIDENCE_TARGET_LONG_EDGE,
            "canvas_size": ISOLATED_EVIDENCE_CANVAS_SIZE,
            "maximum_views": ISOLATED_EVIDENCE_MAX_VIEWS,
            "material_neutralized": True,
            "background_removed": True,
        },
    }
    if showcase:
        registry["render_set"]["mode"] = "showcase"
        registry["render_set"]["runtime_scene"] = showcase_runtime
        registry["render_set"]["source_usd_sha256_before"] = source_sha256_before
        registry["render_set"]["source_usd_sha256_after"] = source_sha256_after
        registry["render_set"]["source_usd_unchanged"] = True
    rendered_registry = destination / "part_registry.rendered.json"
    rendered_registry.write_text(
        json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report = {
        "source_registry": str(registry_file),
        "output_registry": str(rendered_registry),
        "view_count": len(view_records),
        "part_count": len(part_by_path),
        "parts_with_crops": len(best_crops),
        "parts_with_highlights": len(best_highlights),
        "parts_with_isolated_evidence": len(best_evidence),
        "part_evidence_generated": generate_part_evidence,
        "contact_sheets": contact_sheets,
    }
    if showcase:
        report["mode"] = "showcase"
        report["source_usd_sha256_before"] = source_sha256_before
        report["source_usd_sha256_after"] = source_sha256_after
        report["source_usd_unchanged"] = True
        report["runtime_scene"] = showcase_runtime
    return report


def render_part_views(
    *,
    registry_path: str | Path,
    output_dir: str | Path,
    resolution: int = 768,
    view_names: list[str] | None = None,
    rt_subframes: int = 8,
    analysis_up_axis: tuple[float, float, float] = (0.0, 0.0, 1.0),
    analysis_front_axis: tuple[float, float, float] = (0.0, -1.0, 0.0),
    lighting_profile: str = "geometry",
    showcase: bool = False,
    generate_part_evidence: bool = True,
    custom_view_specs_path: str | Path | None = None,
    assembly_pose_overrides_path: str | Path | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Render one evidence set and synchronously release all Kit resources."""

    lifecycle = _RenderLifecycle()
    try:
        return _render_part_views_once(
            registry_path=registry_path,
            output_dir=output_dir,
            resolution=resolution,
            view_names=view_names,
            rt_subframes=rt_subframes,
            analysis_up_axis=analysis_up_axis,
            analysis_front_axis=analysis_front_axis,
            lighting_profile=lighting_profile,
            showcase=showcase,
            generate_part_evidence=generate_part_evidence,
            custom_view_specs_path=custom_view_specs_path,
            assembly_pose_overrides_path=assembly_pose_overrides_path,
            progress_callback=progress_callback,
            _lifecycle=lifecycle,
        )
    finally:
        lifecycle.cleanup()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render RGB and labelled part views")
    parser.add_argument("--registry", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resolution", type=int, default=768)
    parser.add_argument(
        "--views",
        default=",".join(VIEW_DIRECTIONS),
        help=(
            "Comma-separated canonical views or presets. Supported preset: "
            "pose-bank-26"
        ),
    )
    parser.add_argument(
        "--view-specs",
        help=(
            "JSON qwen-camera-view-specs/v1 containing arbitrary analysis-space "
            "camera directions, up axes, focal lengths and distance multipliers; "
            "when supplied it replaces --views"
        ),
    )
    parser.add_argument(
        "--assembly-pose-overrides",
        help=(
            "optional qwen-assembly-pose-overrides/v1 document; applies "
            "bounded rigid transforms to whole Xform subtrees in memory only"
        ),
    )
    parser.add_argument("--rt-subframes", type=int, default=8)
    parser.add_argument(
        "--lighting-profile",
        choices=EVIDENCE_LIGHTING_PROFILES,
        default="geometry",
        help=(
            "geometry keeps directional shape contrast; material-neutral uses "
            "orientation-stable white dome lighting and a black background"
        ),
    )
    parser.add_argument(
        "--showcase",
        action="store_true",
        help=(
            "opt in to an unsaved ground plane and soft three-point lighting; "
            "the default evidence lighting remains unchanged"
        ),
    )
    parser.add_argument(
        "--rgb-only",
        action="store_true",
        help=(
            "render assembly RGB/Part-ID views without per-part crops, "
            "highlights, isolated evidence, or contact sheets"
        ),
    )
    parser.add_argument(
        "--analysis-up-axis",
        choices=tuple(AXIS_VECTORS),
        default="z",
        help=(
            "physical up direction used to orient analysis cameras; source USD "
            "transforms and physics are never modified"
        ),
    )
    parser.add_argument(
        "--analysis-front-axis",
        choices=tuple(AXIS_VECTORS),
        default="-y",
        help=(
            "world direction toward the canonical front camera; must not be "
            "parallel to --analysis-up-axis"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report_progress(
        emit_progress_event,
        scope=PROGRESS_SCOPE,
        stage=ISAAC_STARTUP_PROGRESS_STAGE,
        state="start",
        current=0,
        total=1,
        unit="steps",
        detail="Starting Isaac Sim",
    )

    app = None
    exit_code = 0
    business_started = False
    report: dict[str, Any] | None = None
    try:
        from isaacsim import SimulationApp

        app = SimulationApp(_simulation_app_launch_config())
        report_progress(
            emit_progress_event,
            scope=PROGRESS_SCOPE,
            stage=ISAAC_STARTUP_PROGRESS_STAGE,
            state="complete",
            current=1,
            total=1,
            unit="steps",
            detail="Isaac Sim is ready",
        )
        report_progress(
            emit_progress_event,
            scope=PROGRESS_SCOPE,
            stage=RENDER_BUSINESS_PROGRESS_STAGE,
            state="start",
            current=0,
            total=1,
            unit="jobs",
            detail="Entered render workload",
        )
        business_started = True
        report = render_part_views(
            registry_path=args.registry,
            output_dir=args.output_dir,
            resolution=args.resolution,
            view_names=[
                value.strip() for value in args.views.split(",") if value.strip()
            ],
            rt_subframes=args.rt_subframes,
            analysis_up_axis=AXIS_VECTORS[args.analysis_up_axis],
            analysis_front_axis=AXIS_VECTORS[args.analysis_front_axis],
            lighting_profile=args.lighting_profile,
            showcase=args.showcase,
            generate_part_evidence=not args.rgb_only,
            custom_view_specs_path=args.view_specs,
            assembly_pose_overrides_path=args.assembly_pose_overrides,
            progress_callback=emit_progress_event,
        )
    except Exception:
        traceback.print_exc()
        exit_code = 1
    finally:
        if app is not None:
            try:
                app.close()
            except Exception:
                traceback.print_exc()
                exit_code = 1
    if exit_code == 0 and report is not None:
        if business_started:
            report_progress(
                emit_progress_event,
                scope=PROGRESS_SCOPE,
                stage=RENDER_BUSINESS_PROGRESS_STAGE,
                state="complete",
                current=1,
                total=1,
                unit="jobs",
                detail="Render workload and Isaac shutdown completed",
            )
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
