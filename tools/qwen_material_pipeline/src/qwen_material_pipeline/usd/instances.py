#!/usr/bin/env python3
"""Create a non-destructive editable USD layer for an instanceable assembly.

The output references the source asset and authors only ``instanceable = false``
opinions on composed instance roots.  This exposes each occurrence Mesh at its
original composed path so downstream material tools can bind looks without
copying or rewriting geometry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import traceback
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "qwen-editable-instance-layer/v1"
GEOMETRY_ATTRIBUTES = (
    "points",
    "faceVertexCounts",
    "faceVertexIndices",
    "holeIndices",
    "orientation",
    "subdivisionScheme",
)


def _start_isaac_if_needed(headless: bool = True):
    try:
        from pxr import Usd  # noqa: F401

        return None
    except ImportError:
        try:
            from isaacsim import SimulationApp
        except ImportError as exc:
            raise RuntimeError(
                "pxr is unavailable. Run this script with Isaac Sim python.sh."
            ) from exc
        return SimulationApp({"headless": headless})


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mesh_paths(stage, *, instance_proxies: bool) -> list[str]:
    from pxr import Usd, UsdGeom

    traversal = (
        Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies())
        if instance_proxies
        else stage.Traverse()
    )
    return sorted(
        prim.GetPath().pathString for prim in traversal if prim.IsA(UsdGeom.Mesh)
    )


def _geometry_state(stage, mesh_paths: list[str]) -> dict[str, tuple[str, ...]]:
    return {
        path: tuple(
            repr(stage.GetPrimAtPath(path).GetAttribute(name).Get())
            for name in GEOMETRY_ATTRIBUTES
        )
        for path in mesh_paths
    }


def _world_matrix_state(stage, mesh_paths: list[str]) -> dict[str, tuple[float, ...]]:
    from pxr import Usd, UsdGeom

    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    result: dict[str, tuple[float, ...]] = {}
    for path in mesh_paths:
        matrix = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(path))
        result[path] = tuple(
            float(matrix[row][column]) for row in range(4) for column in range(4)
        )
    return result


def _matrix_states_close(
    left: dict[str, tuple[float, ...]], right: dict[str, tuple[float, ...]]
) -> bool:
    if set(left) != set(right):
        return False
    return all(
        all(
            math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-12)
            for a, b in zip(left[path], right[path], strict=True)
        )
        for path in left
    )


def create_editable_instance_layer(
    *, source_usd: str | Path, output_usd: str | Path
) -> dict[str, Any]:
    """Reference ``source_usd`` and expose all instance occurrence Mesh prims."""

    from pxr import Usd, UsdGeom

    source_path = Path(source_usd).expanduser().resolve(strict=True)
    output_path = Path(output_usd).expanduser().resolve()
    if output_path == source_path:
        raise ValueError("Output layer must not overwrite the source asset")
    if output_path.exists():
        raise ValueError(f"Refusing to overwrite existing output: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    source_sha256_before = _sha256_file(source_path)
    source_stage = Usd.Stage.Open(str(source_path), load=Usd.Stage.LoadAll)
    if source_stage is None:
        raise RuntimeError(f"Unable to open source stage: {source_path}")
    source_default = source_stage.GetDefaultPrim()
    if not source_default or not source_default.IsValid():
        raise ValueError("Source stage has no valid default prim")
    default_path = source_default.GetPath()

    instance_paths = sorted(
        prim.GetPath().pathString
        for prim in source_stage.Traverse()
        if prim.IsInstance()
    )
    if not instance_paths:
        raise ValueError("Source stage has no composed instance roots to expand")

    source_mesh_paths = _mesh_paths(source_stage, instance_proxies=True)
    if not source_mesh_paths:
        raise ValueError("Source stage has no Mesh occurrences behind its instances")
    source_meshes = [
        UsdGeom.Mesh(source_stage.GetPrimAtPath(path)) for path in source_mesh_paths
    ]
    point_occurrence_count = sum(
        len(mesh.GetPointsAttr().Get() or []) for mesh in source_meshes
    )
    face_occurrence_count = sum(
        len(mesh.GetFaceVertexCountsAttr().Get() or []) for mesh in source_meshes
    )
    source_geometry = _geometry_state(source_stage, source_mesh_paths)
    source_world_matrices = _world_matrix_state(source_stage, source_mesh_paths)

    temporary = output_path.with_name(
        f".{output_path.stem}.tmp-{os.getpid()}{output_path.suffix}"
    )
    temporary.unlink(missing_ok=True)
    stage = Usd.Stage.CreateNew(str(temporary))
    if stage is None:
        raise RuntimeError(f"Unable to create output stage: {temporary}")
    UsdGeom.SetStageUpAxis(stage, UsdGeom.GetStageUpAxis(source_stage))
    UsdGeom.SetStageMetersPerUnit(stage, UsdGeom.GetStageMetersPerUnit(source_stage))
    stage.SetTimeCodesPerSecond(source_stage.GetTimeCodesPerSecond())
    stage.SetFramesPerSecond(source_stage.GetFramesPerSecond())
    stage.SetStartTimeCode(source_stage.GetStartTimeCode())
    stage.SetEndTimeCode(source_stage.GetEndTimeCode())

    root = stage.DefinePrim(default_path, source_default.GetTypeName() or "Xform")
    reference_path = os.path.relpath(source_path, start=temporary.parent)
    root.GetReferences().AddReference(reference_path, default_path)
    stage.SetDefaultPrim(root)
    for path in instance_paths:
        stage.OverridePrim(path).SetInstanceable(False)
    stage.GetRootLayer().Save()
    stage = None

    composed = Usd.Stage.Open(str(temporary), load=Usd.Stage.LoadAll)
    if composed is None:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Unable to reopen editable layer: {temporary}")
    output_mesh_paths = _mesh_paths(composed, instance_proxies=False)
    if output_mesh_paths != source_mesh_paths:
        temporary.unlink(missing_ok=True)
        raise RuntimeError("Mesh occurrence paths changed while expanding instances")
    if _geometry_state(composed, output_mesh_paths) != source_geometry:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            "Mesh geometry or topology changed while expanding instances"
        )
    if not _matrix_states_close(
        _world_matrix_state(composed, output_mesh_paths), source_world_matrices
    ):
        temporary.unlink(missing_ok=True)
        raise RuntimeError("Mesh world transforms changed while expanding instances")
    remaining_instances = sum(prim.IsInstance() for prim in composed.Traverse())
    if remaining_instances:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"Editable layer still contains {remaining_instances} instance roots"
        )

    source_sha256_after = _sha256_file(source_path)
    if source_sha256_after != source_sha256_before:
        temporary.unlink(missing_ok=True)
        raise RuntimeError("Source USD changed while creating editable layer")

    composed = None
    temporary.replace(output_path)
    return {
        "schema_version": SCHEMA_VERSION,
        "source_usd": str(source_path),
        "source_sha256": source_sha256_before,
        "output_usd": str(output_path),
        "output_sha256": _sha256_file(output_path),
        "default_prim": default_path.pathString,
        "deinstanced_prim_count": len(instance_paths),
        "mesh_occurrence_count": len(source_mesh_paths),
        "point_occurrence_count": point_occurrence_count,
        "face_occurrence_count": face_occurrence_count,
        "geometry_and_topology_values_unchanged": True,
        "world_transforms_unchanged": True,
        "source_unchanged": True,
        "deinstanced_prim_paths": instance_paths,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-usd", required=True)
    parser.add_argument("--output-usd", required=True)
    parser.add_argument("--report")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    app = _start_isaac_if_needed(headless=True)
    exit_code = 0
    try:
        report = create_editable_instance_layer(
            source_usd=args.source_usd, output_usd=args.output_usd
        )
        if args.report:
            report_path = Path(args.report).expanduser().resolve()
            report_path.parent.mkdir(parents=True, exist_ok=True)
            if report_path.exists():
                raise ValueError(
                    f"Refusing to overwrite existing report: {report_path}"
                )
            report_path.write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    except Exception:
        traceback.print_exc()
        exit_code = 1
    finally:
        if app is not None:
            app.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
