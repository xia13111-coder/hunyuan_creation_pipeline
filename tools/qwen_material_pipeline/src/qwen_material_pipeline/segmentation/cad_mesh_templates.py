#!/usr/bin/env python3
"""Project isolated CAD meshes through the sealed whole-asset cameras.

The ordinary renderer-authored Part-ID image is *modal*: pixels hidden by
another CAD part are absent.  That mask is the correct visibility/location
prior, but it is not the complete shape of the target mesh.  This module
projects every face of one target mesh without changing its transform or the
camera, producing an amodal silhouette that can be used as a separate shape
prior by photo segmentation.

The projection is deliberately CPU-only.  It consumes the camera parameters
already sealed in a rendered Part-ID registry and therefore does not start an
Isaac session for every mesh.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

if TYPE_CHECKING:
    import cv2
    import numpy as np
else:
    cv2 = None
    np = None


SCHEMA_VERSION = "qwen-cad-amodal-part-templates/v1"
USD_DEFAULT_HORIZONTAL_APERTURE_MM = 20.955
MINIMUM_TEMPLATE_PIXELS = 4


class CadMeshTemplateError(ValueError):
    """Raised when an isolated mesh template cannot be reproduced safely."""


def _load_numeric_dependencies() -> None:
    """Load NumPy/OpenCV only after Kit has initialized in Isaac processes.

    Isaac Sim 5 extensions initialize against their bundled numerical runtime.
    Importing the pip-installed NumPy through OpenCV first can make optional Kit
    extensions fail during startup even though this CPU projection tool itself
    later succeeds.  Ordinary library callers still load the modules lazily on
    their first projection operation.
    """

    global cv2, np
    if cv2 is None:
        import cv2 as loaded_cv2

        cv2 = loaded_cv2
    if np is None:
        import numpy as loaded_np

        np = loaded_np


def _start_isaac_if_needed():
    """Load USD bindings without requiring an already-running Kit process."""

    try:
        from pxr import Usd  # noqa: F401

        return None
    except ImportError:
        try:
            from isaacsim import SimulationApp
        except ImportError as exc:
            raise CadMeshTemplateError(
                "USD Python bindings are unavailable; run with Isaac Sim python.sh"
            ) from exc
        return SimulationApp({"headless": True})


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _read_object(path: Path, label: str) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CadMeshTemplateError(f"unable to read {label}: {resolved}") from exc
    if not isinstance(value, dict):
        raise CadMeshTemplateError(f"{label} must be a JSON object")
    return value


def _finite_vector(value: Any, *, size: int, label: str) -> np.ndarray:
    _load_numeric_dependencies()
    if (
        not isinstance(value, list)
        or len(value) != size
        or any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            for item in value
        )
    ):
        raise CadMeshTemplateError(f"{label} must contain {size} finite numbers")
    return np.asarray(value, dtype=np.float64)


def _camera_axes(view: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    position = _finite_vector(view.get("camera_position"), size=3, label="camera position")
    target = _finite_vector(
        view.get("camera_look_at_target"), size=3, label="camera look-at target"
    )
    declared_up = _finite_vector(
        view.get("camera_up_axis"), size=3, label="camera up axis"
    )
    forward = target - position
    forward_norm = float(np.linalg.norm(forward))
    up_norm = float(np.linalg.norm(declared_up))
    if forward_norm <= 1e-12 or up_norm <= 1e-12:
        raise CadMeshTemplateError("camera axes are degenerate")
    forward /= forward_norm
    declared_up /= up_norm
    right = np.cross(forward, declared_up)
    right_norm = float(np.linalg.norm(right))
    if right_norm <= 1e-12:
        raise CadMeshTemplateError("camera up axis is parallel to its view direction")
    right /= right_norm
    up = np.cross(right, forward)
    up /= float(np.linalg.norm(up))
    return right, up, forward


def _project_points(
    points_world: np.ndarray,
    *,
    view: Mapping[str, Any],
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Project world points using the same square-pixel USD camera contract."""

    if points_world.ndim != 2 or points_world.shape[1] != 3:
        raise CadMeshTemplateError("mesh world points must have shape (N, 3)")
    position = _finite_vector(view.get("camera_position"), size=3, label="camera position")
    right, up, forward = _camera_axes(view)
    relative = points_world - position
    camera_x = relative @ right
    camera_y = relative @ up
    camera_z = relative @ forward
    projection = str(view.get("camera_projection_mode", "perspective"))
    if projection == "perspective":
        focal_length = view.get("focal_length_mm")
        if (
            isinstance(focal_length, bool)
            or not isinstance(focal_length, (int, float))
            or not math.isfinite(float(focal_length))
            or float(focal_length) <= 0.0
        ):
            raise CadMeshTemplateError("camera focal length must be positive")
        pixels_per_camera_unit = (
            float(focal_length) / USD_DEFAULT_HORIZONTAL_APERTURE_MM * width
        )
        valid = camera_z > 1e-8
        safe_z = np.where(valid, camera_z, 1.0)
        x = width * 0.5 + pixels_per_camera_unit * camera_x / safe_z
        y = height * 0.5 - pixels_per_camera_unit * camera_y / safe_z
    elif projection == "orthographic":
        multiplier = view.get("camera_orthographic_span_multiplier")
        diagonal = view.get("camera_asset_diagonal")
        if (
            isinstance(multiplier, bool)
            or not isinstance(multiplier, (int, float))
            or float(multiplier) <= 0.0
            or isinstance(diagonal, bool)
            or not isinstance(diagonal, (int, float))
            or float(diagonal) <= 0.0
        ):
            raise CadMeshTemplateError(
                "orthographic templates require camera_asset_diagonal and a positive span"
            )
        pixels_per_camera_unit = width / (float(diagonal) * float(multiplier))
        valid = np.ones(len(points_world), dtype=bool)
        x = width * 0.5 + pixels_per_camera_unit * camera_x
        y = height * 0.5 - pixels_per_camera_unit * camera_y
    else:
        raise CadMeshTemplateError(f"unsupported camera projection: {projection!r}")
    return np.column_stack((x, y)), valid


def _rasterize_faces(
    projected_points: np.ndarray,
    valid_points: np.ndarray,
    face_vertex_counts: Sequence[int],
    face_vertex_indices: Sequence[int],
    *,
    width: int,
    height: int,
) -> np.ndarray:
    _load_numeric_dependencies()
    mask = np.zeros((height, width), dtype=np.uint8)
    offset = 0
    point_count = len(projected_points)
    for face_index, raw_count in enumerate(face_vertex_counts):
        count = int(raw_count)
        indices = np.asarray(face_vertex_indices[offset : offset + count], dtype=np.int64)
        offset += count
        if count < 3:
            continue
        if np.any(indices < 0) or np.any(indices >= point_count):
            raise CadMeshTemplateError(f"face {face_index} references an invalid point")
        if not bool(np.all(valid_points[indices])):
            continue
        polygon = np.rint(projected_points[indices]).astype(np.int32)
        # fillPoly handles both convex and concave CAD n-gons.  Unioning every
        # face deliberately ignores inter-mesh occlusion while retaining the
        # target mesh's exact transform and camera projection.
        cv2.fillPoly(mask, [polygon.reshape(-1, 1, 2)], 255)
    if offset != len(face_vertex_indices):
        raise CadMeshTemplateError("face counts do not consume all face indices")
    return mask


def _mask_audit(mask: np.ndarray) -> dict[str, Any]:
    _load_numeric_dependencies()
    binary = np.asarray(mask, dtype=np.uint8) > 0
    ys, xs = np.where(binary)
    if len(xs) < MINIMUM_TEMPLATE_PIXELS:
        raise CadMeshTemplateError(
            f"isolated mesh template has fewer than {MINIMUM_TEMPLATE_PIXELS} pixels"
        )
    component_count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        binary.astype(np.uint8), connectivity=8
    )
    components = sorted(
        (int(stats[index, cv2.CC_STAT_AREA]) for index in range(1, component_count)),
        reverse=True,
    )
    return {
        "mask_size": [int(binary.shape[1]), int(binary.shape[0])],
        "mask_pixels": int(np.count_nonzero(binary)),
        "bbox_pixels": [
            int(xs.min()),
            int(ys.min()),
            int(xs.max()) + 1,
            int(ys.max()) + 1,
        ],
        "connected_component_count": len(components),
        "connected_component_pixels": components,
    }


def _write_mask(path: Path, mask: np.ndarray) -> dict[str, Any]:
    _load_numeric_dependencies()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), np.asarray(mask, dtype=np.uint8)):
        raise CadMeshTemplateError(f"unable to write template mask: {path}")
    return {"path": str(path), "sha256": _sha256_file(path), **_mask_audit(mask)}


def build_templates(
    *,
    registry_path: Path,
    spatial_report_path: Path,
    evidence_path: Path,
    output_dir: Path,
    part_ids: set[str] | None = None,
) -> dict[str, Any]:
    _load_numeric_dependencies()
    """Build aligned amodal masks for every observed Part-ID/view pair."""

    try:
        from pxr import Usd, UsdGeom
    except ImportError as exc:  # pragma: no cover - production environment owns USD
        raise CadMeshTemplateError("USD Python bindings are required") from exc

    registry_path = registry_path.expanduser().resolve(strict=True)
    spatial_report_path = spatial_report_path.expanduser().resolve(strict=True)
    evidence_path = evidence_path.expanduser().resolve(strict=True)
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise CadMeshTemplateError(f"template output already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    raw_dir = output_dir / "render_masks"
    aligned_dir = output_dir / "masks"
    registry = _read_object(registry_path, "calibrated Part-ID registry")
    spatial = _read_object(spatial_report_path, "spatial mapping report")
    evidence = _read_object(evidence_path, "Part-ID evidence")
    asset_path = Path(str(registry.get("asset_usd", ""))).expanduser().resolve(strict=True)
    expected_asset_hash = registry.get("asset_sha256")
    if not isinstance(expected_asset_hash, str) or _sha256_file(asset_path) != expected_asset_hash:
        raise CadMeshTemplateError("registry asset hash does not match the USD")

    parts = {
        str(row.get("part_id")): row
        for row in registry.get("parts", [])
        if isinstance(row, Mapping) and isinstance(row.get("part_id"), str)
    }
    render_set = registry.get("render_set")
    if not isinstance(render_set, Mapping):
        raise CadMeshTemplateError("registry has no render_set")
    views = {
        str(row.get("view_id")): row
        for row in render_set.get("views", [])
        if isinstance(row, Mapping) and isinstance(row.get("view_id"), str)
    }
    world_bounds = [
        row.get("world_bbox")
        for row in parts.values()
        if isinstance(row, Mapping) and isinstance(row.get("world_bbox"), list)
    ]
    asset_diagonal: float | None = None
    if world_bounds:
        try:
            minimum = np.min(
                np.asarray([bounds[0] for bounds in world_bounds], dtype=np.float64),
                axis=0,
            )
            maximum = np.max(
                np.asarray([bounds[1] for bounds in world_bounds], dtype=np.float64),
                axis=0,
            )
            candidate_diagonal = float(np.linalg.norm(maximum - minimum))
            if math.isfinite(candidate_diagonal) and candidate_diagonal > 0.0:
                asset_diagonal = candidate_diagonal
        except (IndexError, TypeError, ValueError):
            asset_diagonal = None
    alignments = {
        str(row.get("reference_view_id")): row
        for row in spatial.get("view_alignments", [])
        if isinstance(row, Mapping)
        and isinstance(row.get("reference_view_id"), str)
    }
    stage = Usd.Stage.Open(str(asset_path))
    if stage is None:
        raise CadMeshTemplateError(f"unable to open USD stage: {asset_path}")
    xforms = UsdGeom.XformCache()
    raw_cache: dict[tuple[str, str], tuple[np.ndarray, dict[str, Any]]] = {}
    records: list[dict[str, Any]] = []

    for part in evidence.get("parts", []):
        if not isinstance(part, Mapping) or part.get("status") != "observed":
            continue
        part_id = part.get("part_id")
        if (
            not isinstance(part_id, str)
            or (part_ids is not None and part_id not in part_ids)
        ):
            continue
        registry_part = parts.get(part_id)
        if not isinstance(registry_part, Mapping):
            raise CadMeshTemplateError(f"evidence references unknown Part-ID: {part_id}")
        prim_path = registry_part.get("prim_path")
        if not isinstance(prim_path, str):
            raise CadMeshTemplateError(f"registry Part-ID {part_id} has no prim_path")
        prim = stage.GetPrimAtPath(prim_path)
        mesh = UsdGeom.Mesh(prim)
        if not prim.IsValid() or not mesh:
            raise CadMeshTemplateError(f"Part-ID {part_id} is not a USD mesh: {prim_path}")
        points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
        counts = [int(value) for value in mesh.GetFaceVertexCountsAttr().Get()]
        indices = [int(value) for value in mesh.GetFaceVertexIndicesAttr().Get()]
        transform = np.asarray(xforms.GetLocalToWorldTransform(prim), dtype=np.float64)
        homogeneous = np.column_stack((points, np.ones(len(points), dtype=np.float64)))
        transformed = homogeneous @ transform
        world_points = transformed[:, :3] / transformed[:, 3:4]

        for observation in part.get("observations", []):
            if not isinstance(observation, Mapping):
                continue
            reference_view_id = str(observation.get("view_id"))
            alignment = alignments.get(reference_view_id)
            if not isinstance(alignment, Mapping):
                raise CadMeshTemplateError(
                    f"missing spatial alignment for reference view {reference_view_id}"
                )
            render_view_id = alignment.get("selected_render_view_id")
            if not isinstance(render_view_id, str) or render_view_id not in views:
                raise CadMeshTemplateError(
                    f"alignment for {reference_view_id} has no calibrated render view"
                )
            render_view = views[render_view_id]
            part_ids_path = Path(str(render_view.get("part_ids", ""))).expanduser().resolve(
                strict=True
            )
            rendered_ids = cv2.imread(str(part_ids_path), cv2.IMREAD_COLOR)
            if rendered_ids is None:
                raise CadMeshTemplateError(f"unable to decode Part-ID image: {part_ids_path}")
            render_height, render_width = rendered_ids.shape[:2]
            cache_key = (render_view_id, part_id)
            if cache_key not in raw_cache:
                projection_view = dict(render_view)
                if asset_diagonal is not None:
                    projection_view["camera_asset_diagonal"] = asset_diagonal
                projected, valid = _project_points(
                    world_points,
                    view=projection_view,
                    width=render_width,
                    height=render_height,
                )
                raw_mask = _rasterize_faces(
                    projected,
                    valid,
                    counts,
                    indices,
                    width=render_width,
                    height=render_height,
                )
                raw_doc = _write_mask(
                    raw_dir / f"{render_view_id}__{part_id}.png", raw_mask
                )
                raw_cache[cache_key] = (raw_mask, raw_doc)
            raw_mask, raw_doc = raw_cache[cache_key]

            source_image_path = Path(str(observation.get("image", ""))).expanduser().resolve(
                strict=True
            )
            source = cv2.imread(str(source_image_path), cv2.IMREAD_COLOR)
            if source is None:
                raise CadMeshTemplateError(f"unable to decode source image: {source_image_path}")
            quarter_turns = alignment.get("quarter_turns_ccw")
            if isinstance(quarter_turns, bool) or not isinstance(quarter_turns, int):
                raise CadMeshTemplateError("quarter_turns_ccw must be an integer")
            rotated = np.rot90(raw_mask, int(quarter_turns) % 4).copy()
            bbox_affine = np.asarray(alignment.get("bbox_affine"), dtype=np.float32)
            ecc_warp = np.asarray(alignment.get("ecc_warp"), dtype=np.float32)
            if bbox_affine.shape != (2, 3) or ecc_warp.shape != (2, 3):
                raise CadMeshTemplateError("spatial affine matrices must have shape (2, 3)")
            normalized = cv2.warpAffine(
                rotated,
                bbox_affine,
                (source.shape[1], source.shape[0]),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            aligned = cv2.warpAffine(
                normalized,
                ecc_warp,
                (source.shape[1], source.shape[0]),
                flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            aligned_doc = _write_mask(
                aligned_dir / f"{reference_view_id}__{part_id}.png", aligned
            )
            modal_path = Path(str(observation.get("mask", ""))).expanduser().resolve(
                strict=True
            )
            modal = cv2.imread(str(modal_path), cv2.IMREAD_GRAYSCALE)
            if modal is None or modal.shape != aligned.shape:
                raise CadMeshTemplateError(
                    f"modal/amodal dimensions differ for {reference_view_id}/{part_id}"
                )
            modal_binary = modal >= 128
            amodal_binary = aligned >= 128
            intersection = int(np.count_nonzero(modal_binary & amodal_binary))
            records.append(
                {
                    "view_id": reference_view_id,
                    "part_id": part_id,
                    "mesh_prim_path": prim_path,
                    "render_view_id": render_view_id,
                    "quarter_turns_ccw": int(quarter_turns) % 4,
                    "raw_amodal_mask": raw_doc,
                    "aligned_amodal_mask": aligned_doc,
                    "modal_visibility_mask": {
                        "path": str(modal_path),
                        "sha256": _sha256_file(modal_path),
                        "mask_pixels": int(np.count_nonzero(modal_binary)),
                    },
                    "modal_amodal_intersection_pixels": intersection,
                    "modal_precision_against_amodal": intersection
                    / max(1, int(np.count_nonzero(modal_binary))),
                    "visible_fraction_of_amodal": intersection
                    / max(1, int(np.count_nonzero(amodal_binary))),
                    "projection_contract": {
                        "whole_asset_camera_unchanged": True,
                        "whole_asset_transform_unchanged": True,
                        "per_mesh_pose_change_allowed": False,
                        "other_mesh_occlusion_disabled_for_shape_only": True,
                        "horizontal_aperture_mm": USD_DEFAULT_HORIZONTAL_APERTURE_MM,
                    },
                }
            )
    if not records:
        raise CadMeshTemplateError("no observed Part-ID templates were generated")
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "inputs": {
            "registry": {"path": str(registry_path), "sha256": _sha256_file(registry_path)},
            "spatial_report": {
                "path": str(spatial_report_path),
                "sha256": _sha256_file(spatial_report_path),
            },
            "part_id_evidence": {
                "path": str(evidence_path),
                "sha256": _sha256_file(evidence_path),
            },
            "asset_usd": {"path": str(asset_path), "sha256": expected_asset_hash},
        },
        "policy": {
            "shape_authority": "isolated_target_mesh_amodal_projection",
            "visibility_authority": "whole_assembly_renderer_part_id_projection",
            "camera_authority": "sealed_whole_asset_camera",
            "per_mesh_pose_change_allowed": False,
        },
        "records": sorted(records, key=lambda row: (row["view_id"], row["part_id"])),
        "summary": {
            "template_count": len(records),
            "part_count": len({row["part_id"] for row in records}),
            "view_count": len({row["view_id"] for row in records}),
        },
    }
    result["integrity"] = {"result_sha256": _canonical_sha256(result)}
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--spatial-report", required=True, type=Path)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--part-id", action="append", default=[])
    args = parser.parse_args(argv)
    app = _start_isaac_if_needed()
    _load_numeric_dependencies()
    exit_code = 0
    try:
        result = build_templates(
            registry_path=args.registry,
            spatial_report_path=args.spatial_report,
            evidence_path=args.evidence,
            output_dir=args.output_dir,
            part_ids=set(args.part_id) if args.part_id else None,
        )
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2), flush=True)
    except Exception:
        traceback.print_exc()
        exit_code = 1
    finally:
        if app is not None:
            app.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
