#!/usr/bin/env python3
"""Fast geometry-only Part-ID rasterization for camera candidate search.

This module deliberately does not replace the authoritative Isaac render.  It
loads the immutable USD assembly once, rasterizes camera hypotheses with
Kaolin's CUDA rasterizer, and returns stable Part-ID images suitable for the
existing camera objective.  The selected Top-K hypotheses are still rendered
and scored by Isaac at full resolution before they can be published.

Only the camera moves.  Mesh points and per-part transforms are read from the
composed USD stage and baked into one immutable world-space triangle buffer.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from qwen_material_pipeline.evidence.spatial import _part_color


USD_DEFAULT_HORIZONTAL_APERTURE_MM = 20.955
FAST_RASTER_BACKEND = "kaolin_cuda_part_id/v1"

_AXIS_VECTORS = {
    "x": (1.0, 0.0, 0.0),
    "-x": (-1.0, 0.0, 0.0),
    "y": (0.0, 1.0, 0.0),
    "-y": (0.0, -1.0, 0.0),
    "z": (0.0, 0.0, 1.0),
    "-z": (0.0, 0.0, -1.0),
}


class FastCameraRasterUnavailable(RuntimeError):
    """Raised when the optional fast-search runtime cannot be used safely."""


class FastCameraRasterError(RuntimeError):
    """Raised when a configured fast raster violates its sealed contract."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise FastCameraRasterError(f"JSON root must be an object: {resolved}")
    return value


def _unit_vector(value: Sequence[float], *, label: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if vector.shape != (3,) or not np.isfinite(vector).all() or norm <= 1e-12:
        raise FastCameraRasterError(f"{label} must be a finite non-zero 3-D vector")
    return vector / norm


def _axis_vector(value: str, *, label: str) -> np.ndarray:
    try:
        return np.asarray(_AXIS_VECTORS[value], dtype=np.float64)
    except KeyError as exc:
        raise FastCameraRasterError(
            f"{label} must be one of {', '.join(sorted(_AXIS_VECTORS))}"
        ) from exc


def _analysis_basis(up_axis: str, front_axis: str) -> np.ndarray:
    """Return the analysis-to-world basis used by the Isaac renderer."""

    up = _axis_vector(up_axis, label="analysis up axis")
    requested_front = _axis_vector(front_axis, label="analysis front axis")
    planar_front = requested_front - float(np.dot(requested_front, up)) * up
    front = _unit_vector(planar_front, label="analysis front axis")
    rear = -front
    right = _unit_vector(np.cross(rear, up), label="analysis right axis")
    rear = _unit_vector(np.cross(up, right), label="analysis rear axis")
    return np.column_stack((right, rear, up))


def _triangulate_faces(
    counts: Sequence[int],
    indices: Sequence[int],
    *,
    point_count: int,
    vertex_offset: int,
) -> np.ndarray:
    """Triangulate USD n-gons with a deterministic fan."""

    triangles: list[tuple[int, int, int]] = []
    offset = 0
    for face_index, raw_count in enumerate(counts):
        count = int(raw_count)
        face = [int(value) for value in indices[offset : offset + count]]
        offset += count
        if count < 3:
            continue
        if any(value < 0 or value >= point_count for value in face):
            raise FastCameraRasterError(
                f"USD face {face_index} references an invalid mesh point"
            )
        triangles.extend(
            (
                face[0] + vertex_offset,
                face[index] + vertex_offset,
                face[index + 1] + vertex_offset,
            )
            for index in range(1, count - 1)
        )
    if offset != len(indices):
        raise FastCameraRasterError("USD face counts do not consume all indices")
    return np.asarray(triangles, dtype=np.int64).reshape(-1, 3)


def _registry_bounds(parts: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, float]:
    bounds: list[np.ndarray] = []
    for row in parts:
        raw = row.get("world_bbox")
        candidate = np.asarray(raw, dtype=np.float64)
        if candidate.shape != (2, 3) or not np.isfinite(candidate).all():
            raise FastCameraRasterError(
                f"Part {row.get('part_id')!r} has an invalid world_bbox"
            )
        bounds.append(candidate)
    if not bounds:
        raise FastCameraRasterError("Part registry contains no world bounds")
    minimum = np.min(np.asarray([item[0] for item in bounds]), axis=0)
    maximum = np.max(np.asarray([item[1] for item in bounds]), axis=0)
    diagonal = float(np.linalg.norm(maximum - minimum))
    if not math.isfinite(diagonal) or diagonal <= 0.0:
        raise FastCameraRasterError("Part registry has degenerate world bounds")
    return (minimum + maximum) * 0.5, diagonal


class FastPartIdRasterizer:
    """One-load CUDA rasterizer for immutable whole-assembly camera search."""

    def __init__(
        self,
        *,
        registry_path: Path,
        resolution: int,
        analysis_up_axis: str,
        analysis_front_axis: str,
        device: str = "cuda",
    ) -> None:
        if isinstance(resolution, bool) or not isinstance(resolution, int):
            raise FastCameraRasterError("Fast raster resolution must be an integer")
        if resolution < 64:
            raise FastCameraRasterError("Fast raster resolution must be at least 64")
        started = time.monotonic()
        try:
            import torch
            from kaolin.render import mesh as kaolin_mesh
            from pxr import Usd, UsdGeom
        except ImportError as exc:
            raise FastCameraRasterUnavailable(
                "fast camera search requires torch, Kaolin, and USD Python bindings"
            ) from exc
        if device != "cuda" or not torch.cuda.is_available():
            raise FastCameraRasterUnavailable(
                "fast camera search requires a CUDA device"
            )

        self._torch = torch
        self._kaolin_mesh = kaolin_mesh
        self._device = device
        self.resolution = resolution
        self.registry_path = registry_path.expanduser().resolve(strict=True)
        self.registry = _read_object(self.registry_path)
        parts = [
            row
            for row in self.registry.get("parts", [])
            if isinstance(row, Mapping) and isinstance(row.get("part_id"), str)
        ]
        if len(parts) != len(self.registry.get("parts", [])) or not parts:
            raise FastCameraRasterError(
                "Part registry must contain only valid, identified mesh records"
            )
        self.parts = parts
        self.part_colors_bgr = [
            np.asarray(
                (
                    _part_color(str(row["part_id"]))[2],
                    _part_color(str(row["part_id"]))[1],
                    _part_color(str(row["part_id"]))[0],
                ),
                dtype=np.uint8,
            )
            for row in parts
        ]
        self._center, self._diagonal = _registry_bounds(parts)
        self._basis = _analysis_basis(analysis_up_axis, analysis_front_axis)

        asset_value = self.registry.get("asset_usd")
        expected_asset_hash = self.registry.get("asset_sha256")
        if not isinstance(asset_value, str) or not asset_value:
            raise FastCameraRasterError("Part registry has no asset_usd")
        asset_path = Path(asset_value).expanduser().resolve(strict=True)
        if (
            not isinstance(expected_asset_hash, str)
            or len(expected_asset_hash) != 64
            or _sha256_file(asset_path) != expected_asset_hash
        ):
            raise FastCameraRasterError("Part registry asset hash does not match the USD")
        stage = Usd.Stage.Open(str(asset_path))
        if stage is None:
            raise FastCameraRasterUnavailable(f"Unable to open USD stage: {asset_path}")
        xforms = UsdGeom.XformCache(Usd.TimeCode.Default())
        vertex_chunks: list[np.ndarray] = []
        triangle_chunks: list[np.ndarray] = []
        triangle_parts: list[np.ndarray] = []
        vertex_offset = 0
        for part_index, row in enumerate(parts):
            prim_path = str(row["prim_path"])
            prim = stage.GetPrimAtPath(prim_path)
            mesh = UsdGeom.Mesh(prim)
            if not prim.IsValid() or not mesh:
                raise FastCameraRasterError(
                    f"Registry Part-ID is not a composed USD mesh: {prim_path}"
                )
            points_value = mesh.GetPointsAttr().Get(Usd.TimeCode.Default())
            counts_value = mesh.GetFaceVertexCountsAttr().Get(Usd.TimeCode.Default())
            indices_value = mesh.GetFaceVertexIndicesAttr().Get(Usd.TimeCode.Default())
            points = np.asarray(points_value, dtype=np.float64)
            counts = [int(value) for value in counts_value]
            indices = [int(value) for value in indices_value]
            if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
                raise FastCameraRasterError(f"USD mesh has invalid points: {prim_path}")
            transform = np.asarray(
                xforms.GetLocalToWorldTransform(prim), dtype=np.float64
            )
            homogeneous = np.column_stack(
                (points, np.ones(len(points), dtype=np.float64))
            )
            transformed = homogeneous @ transform
            world = transformed[:, :3] / transformed[:, 3:4]
            triangles = _triangulate_faces(
                counts,
                indices,
                point_count=len(points),
                vertex_offset=vertex_offset,
            )
            vertex_chunks.append(world.astype(np.float32))
            triangle_chunks.append(triangles)
            triangle_parts.append(
                np.full(len(triangles), part_index, dtype=np.int64)
            )
            vertex_offset += len(points)
        vertices = np.concatenate(vertex_chunks, axis=0)
        triangles = np.concatenate(triangle_chunks, axis=0)
        face_to_part = np.concatenate(triangle_parts, axis=0)
        if not len(triangles):
            raise FastCameraRasterError("USD assembly contains no rasterizable triangles")

        self._vertices = torch.from_numpy(vertices).to(device=device)
        self._triangles = torch.from_numpy(triangles).to(device=device)
        self._face_to_part = torch.from_numpy(face_to_part).to(device=device)
        self._part_colors = torch.from_numpy(np.asarray(self.part_colors_bgr)).to(
            device=device
        )
        self._face_features = torch.zeros(
            (1, len(triangles), 3, 1), dtype=torch.float32, device=device
        )
        self.audit = {
            "backend": FAST_RASTER_BACKEND,
            "asset_usd": str(asset_path),
            "asset_sha256": expected_asset_hash,
            "part_count": len(parts),
            "vertex_count": int(len(vertices)),
            "triangle_count": int(len(triangles)),
            "resolution": resolution,
            "device": device,
            "initialization_seconds": round(time.monotonic() - started, 6),
            "candidate_count": 0,
            "raster_seconds": 0.0,
            "released_before_isaac_verification": False,
        }

    def release(self) -> None:
        """Release search buffers before the authoritative Isaac subprocess."""

        if self.audit["released_before_isaac_verification"]:
            return
        self._vertices = None
        self._triangles = None
        self._face_to_part = None
        self._part_colors = None
        self._face_features = None
        self._torch.cuda.empty_cache()
        self.audit["released_before_isaac_verification"] = True

    def _camera(self, spec: Mapping[str, Any]) -> tuple[np.ndarray, ...]:
        direction = _unit_vector(
            spec.get("analysis_direction", []), label="analysis camera direction"
        )
        declared_up = _unit_vector(
            spec.get("analysis_up_axis", []), label="analysis camera up axis"
        )
        world_direction = _unit_vector(
            self._basis @ direction, label="world camera direction"
        )
        world_up = _unit_vector(self._basis @ declared_up, label="world camera up")
        distance = float(spec.get("distance_multiplier", 2.15))
        target_u = float(spec.get("target_offset_u", 0.0))
        target_v = float(spec.get("target_offset_v", 0.0))
        if not all(math.isfinite(value) for value in (distance, target_u, target_v)):
            raise FastCameraRasterError("Camera distance and target offsets must be finite")
        if distance <= 1.0:
            raise FastCameraRasterError("Camera distance multiplier must exceed one")
        position = self._center + world_direction * self._diagonal * distance
        nominal_forward = -world_direction
        right = _unit_vector(
            np.cross(nominal_forward, world_up), label="camera right axis"
        )
        target = (
            self._center
            + self._diagonal * target_u * right
            + self._diagonal * target_v * world_up
        )
        forward = _unit_vector(target - position, label="camera forward axis")
        right = _unit_vector(np.cross(forward, world_up), label="camera right axis")
        up = _unit_vector(np.cross(right, forward), label="camera up axis")
        return position, right, up, forward

    def render_part_ids(self, spec: Mapping[str, Any]) -> np.ndarray:
        """Rasterize one camera hypothesis into the stable BGR Part-ID contract."""

        if self.audit["released_before_isaac_verification"]:
            raise FastCameraRasterError("Fast camera rasterizer has been released")
        started = time.monotonic()
        torch = self._torch
        position, right, up, forward = self._camera(spec)
        with torch.no_grad():
            position_tensor = torch.as_tensor(
                position, dtype=torch.float32, device=self._device
            )
            right_tensor = torch.as_tensor(
                right, dtype=torch.float32, device=self._device
            )
            up_tensor = torch.as_tensor(up, dtype=torch.float32, device=self._device)
            forward_tensor = torch.as_tensor(
                forward, dtype=torch.float32, device=self._device
            )
            relative = self._vertices - position_tensor
            camera_x = relative @ right_tensor
            camera_y = relative @ up_tensor
            depth = relative @ forward_tensor
            projection_mode = str(spec.get("projection_mode", "perspective"))
            if projection_mode == "perspective":
                focal = float(spec.get("focal_length_mm", 45.0))
                if not math.isfinite(focal) or focal <= 0.0:
                    raise FastCameraRasterError("Camera focal length must be positive")
                scale = 2.0 * focal / USD_DEFAULT_HORIZONTAL_APERTURE_MM
                safe_depth = torch.where(depth > 1e-8, depth, torch.ones_like(depth))
                image = torch.stack(
                    (scale * camera_x / safe_depth, scale * camera_y / safe_depth),
                    dim=1,
                )
            elif projection_mode == "orthographic":
                span_multiplier = float(
                    spec.get("orthographic_span_multiplier", 2.0)
                )
                if not math.isfinite(span_multiplier) or span_multiplier <= 0.0:
                    raise FastCameraRasterError(
                        "Orthographic span multiplier must be positive"
                    )
                scale = 2.0 / (self._diagonal * span_multiplier)
                image = torch.stack((scale * camera_x, scale * camera_y), dim=1)
            else:
                raise FastCameraRasterError(
                    f"Unsupported camera projection mode: {projection_mode!r}"
                )
            face_depth = (-depth)[self._triangles].unsqueeze(0)
            face_image = image[self._triangles].unsqueeze(0)
            near = max(self._diagonal * 0.001, 1e-5)
            valid_faces = (depth[self._triangles] > near).all(dim=1).unsqueeze(0)
            _features, face_indices = self._kaolin_mesh.rasterize(
                self.resolution,
                self.resolution,
                face_depth,
                face_image,
                self._face_features,
                valid_faces=valid_faces,
                backend="cuda",
            )
            selected = face_indices[0]
            foreground = selected >= 0
            part_indices = self._face_to_part[selected.clamp(min=0)]
            output = self._part_colors[part_indices]
            output = torch.where(
                foreground.unsqueeze(-1), output, torch.zeros_like(output)
            )
            result = output.cpu().numpy().astype(np.uint8, copy=False)
        self.audit["candidate_count"] += 1
        self.audit["raster_seconds"] = round(
            float(self.audit["raster_seconds"]) + time.monotonic() - started,
            6,
        )
        return result


__all__ = [
    "FAST_RASTER_BACKEND",
    "FastCameraRasterError",
    "FastCameraRasterUnavailable",
    "FastPartIdRasterizer",
    "_analysis_basis",
    "_registry_bounds",
    "_triangulate_faces",
]
