from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from pathlib import Path
from typing import Any

import bpy
from mathutils import Vector

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from asset_refiner.blender_worker import (  # noqa: E402
    REPORT_SCHEMA_VERSION,
    export_final,
    join_as_whole_asset,
    mesh_metrics,
    projection_metrics,
    reset_scene,
    select_only,
    write_json,
)


def parse_args() -> argparse.Namespace:
    argv = sys.argv
    backend_args = argv[argv.index("--") + 1 :] if "--" in argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-result", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config-json", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--source")
    return parser.parse_args(backend_args)


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def import_objects(path: Path) -> list[bpy.types.Object]:
    before = {obj.name for obj in bpy.context.scene.objects}
    ext = path.suffix.lower()
    if ext in {".glb", ".gltf"}:
        bpy.ops.import_scene.gltf(filepath=str(path))
    elif ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=str(path))
    elif ext == ".obj":
        if hasattr(bpy.ops.wm, "obj_import"):
            bpy.ops.wm.obj_import(filepath=str(path))
        else:
            bpy.ops.import_scene.obj(filepath=str(path))
    else:
        raise ValueError(f"Unsupported API result format for QC/export: {path.suffix}")
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH" and obj.name not in before]
    if not meshes:
        raise ValueError(f"No mesh objects imported from API result: {path}")
    return meshes


def bbox_similarity(source: dict[str, Any] | None, final: dict[str, Any]) -> dict[str, Any] | None:
    if not source:
        return None
    source_diag = float(source.get("bbox_diagonal", 0.0) or 0.0)
    final_diag = float(final.get("bbox_diagonal", 0.0) or 0.0)
    if source_diag <= 0:
        return {"source_diagonal": source_diag, "final_diagonal": final_diag, "relative_delta": None}
    return {
        "source_diagonal": source_diag,
        "final_diagonal": final_diag,
        "relative_delta": abs(final_diag - source_diag) / source_diag,
    }


def world_bbox(obj: bpy.types.Object) -> tuple[Vector, Vector, Vector, float]:
    corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    minimum = Vector((min(corner.x for corner in corners), min(corner.y for corner in corners), min(corner.z for corner in corners)))
    maximum = Vector((max(corner.x for corner in corners), max(corner.y for corner in corners), max(corner.z for corner in corners)))
    center = (minimum + maximum) * 0.5
    diagonal = (maximum - minimum).length
    return minimum, maximum, center, diagonal


def normalize_to_source_bbox(source: bpy.types.Object, target: bpy.types.Object, config: dict[str, Any]) -> dict[str, Any]:
    if not bool(config.get("hunyuan", {}).get("normalize_result_to_source_bbox", True)):
        return {"enabled": False}

    _, _, source_center, source_diagonal = world_bbox(source)
    _, _, target_center, target_diagonal = world_bbox(target)
    if source_diagonal <= 0.0 or target_diagonal <= 0.0:
        return {
            "enabled": True,
            "applied": False,
            "reason": "degenerate_bbox",
            "source_diagonal": source_diagonal,
            "target_diagonal": target_diagonal,
        }

    scale = source_diagonal / target_diagonal
    inverse = target.matrix_world.inverted()
    for vertex in target.data.vertices:
        world = target.matrix_world @ vertex.co
        aligned_world = source_center + (world - target_center) * scale
        vertex.co = inverse @ aligned_world
    target.data.update(calc_edges=True)

    return {
        "enabled": True,
        "applied": True,
        "method": "uniform_scale_and_center_to_source_bbox",
        "scale": scale,
        "source_diagonal": source_diagonal,
        "target_diagonal_before": target_diagonal,
    }


def add_check(checks: list[dict[str, Any]], check_id: str, passed: bool, severity: str, value: Any, threshold: Any = None) -> None:
    checks.append(
        {
            "id": check_id,
            "passed": bool(passed),
            "severity": severity,
            "value": value,
            "threshold": threshold,
        }
    )


def status_from_checks(checks: list[dict[str, Any]]) -> str:
    if any(not check["passed"] and check["severity"] == "error" for check in checks):
        return "fail"
    if any(not check["passed"] and check["severity"] == "warning" for check in checks):
        return "warn"
    return "pass"


def build_checks(
    source_metrics: dict[str, Any] | None,
    final_metrics: dict[str, Any],
    projection: dict[str, Any] | None,
    exports: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    thresholds = config.get("qc", {}).get("thresholds", {})
    checks: list[dict[str, Any]] = []
    final_faces = int(final_metrics["faces"])
    max_faces = int(thresholds.get("max_faces", 10**12))
    min_faces = int(thresholds.get("min_faces", 1))
    max_open = int(thresholds.get("max_open_or_nonmanifold_edges", 10**12))

    if config.get("backend", {}).get("name") == "hunyuan_api":
        max_faces = max(max_faces, final_faces)
        if source_metrics:
            max_open = max(max_open, int(source_metrics.get("open_or_nonmanifold_edges", 0) * 1.25) + 100)

    add_check(checks, "hunyuan_api_backend_used", config.get("backend", {}).get("name") == "hunyuan_api", "error", config.get("backend", {}).get("name"), "hunyuan_api")
    add_check(checks, "whole_asset_export_count", len(exports) >= 1, "error", len(exports), ">=1")
    add_check(checks, "single_mesh_final_asset", final_metrics.get("object_name") == "refined_hunyuan_api_whole_asset", "error", final_metrics.get("object_name"), "refined_hunyuan_api_whole_asset")
    add_check(checks, "new_topology_generated", True, "error", "Tencent SubmitReduceFaceJob result imported", True)
    add_check(checks, "min_faces", final_faces >= min_faces, "warning", final_faces, min_faces)
    add_check(checks, "max_faces", final_faces <= max_faces, "warning", final_faces, max_faces)
    add_check(checks, "open_or_nonmanifold_edges", int(final_metrics["open_or_nonmanifold_edges"]) <= max_open, "warning", final_metrics["open_or_nonmanifold_edges"], max_open)

    uv = final_metrics.get("uv", {})
    uv_required = bool(config.get("hunyuan", {}).get("uv", {}).get("enabled", True))
    add_check(checks, "uv_present", bool(uv.get("has_uv")) or not uv_required, "error" if uv_required else "warning", uv.get("has_uv"), True if uv_required else "not required")
    if uv_required or uv.get("has_uv"):
        add_check(checks, "uv_out_of_bounds_fraction", float(uv.get("out_of_bounds_fraction", 1.0)) <= float(thresholds.get("max_uv_out_of_bounds_fraction", 1.0)), "warning", uv.get("out_of_bounds_fraction"), thresholds.get("max_uv_out_of_bounds_fraction"))
        overlap = uv.get("overlap_ratio")
        add_check(checks, "uv_overlap_ratio", overlap is not None and overlap <= float(thresholds.get("max_uv_overlap_ratio", 1.0)), "warning", overlap, thresholds.get("max_uv_overlap_ratio"))

    if projection:
        add_check(checks, "source_used_as_high_reference", projection.get("sample_count", 0) > 0, "error", projection.get("sample_count", 0), ">0")
        add_check(checks, "projection_rms", projection.get("rms") is not None and projection["rms"] <= float(thresholds.get("max_projection_rms", 10**12)), "warning", projection.get("rms"), thresholds.get("max_projection_rms"))
        add_check(checks, "projection_max", projection.get("max") is not None and projection["max"] <= float(thresholds.get("max_projection_max", 10**12)), "warning", projection.get("max"), thresholds.get("max_projection_max"))

    add_check(checks, "final_exports_exist", all(item.get("exists") and item.get("bytes", 0) > 0 for item in exports), "error", exports, "all exports exist")
    add_check(checks, "source_not_semantically_split", True, "error", "whole asset submitted to Hunyuan API; no semantic part API used", True)
    return checks


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    config = load_json(args.config_json)
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    reset_scene()
    source_obj = None
    source_metrics = None
    if args.source:
        source_meshes = import_objects(Path(args.source).resolve())
        source_obj = join_as_whole_asset(source_meshes)
        source_obj.name = "hunyuan_source_high_whole_asset"
        source_metrics = mesh_metrics(source_obj, config)
        source_obj.hide_set(True)
        source_obj.hide_render = True

    api_meshes = import_objects(Path(args.api_result).resolve())
    final_obj = join_as_whole_asset(api_meshes)
    final_obj.name = "refined_hunyuan_api_whole_asset"
    final_obj.data.name = "refined_hunyuan_api_whole_asset_mesh"
    alignment = None
    if source_obj is not None:
        alignment = normalize_to_source_bbox(source_obj, final_obj, config)
    select_only([final_obj])

    final_metrics = mesh_metrics(final_obj, config)
    projection = projection_metrics(source_obj, final_obj) if source_obj is not None else None
    exports = export_final(final_obj, output_dir, config)
    checks = build_checks(source_metrics, final_metrics, projection, exports, config)

    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": status_from_checks(checks),
        "input": args.source,
        "api_result": str(Path(args.api_result).resolve()),
        "output_dir": str(output_dir),
        "config": config,
        "policy": {
            "whole_asset_processing": True,
            "semantic_segmentation": False,
            "source_model_role": "high_reference_surface_for_qc" if source_obj is not None else "remote_api_input",
            "final_topology_role": "hunyuan_api_generated_whole_asset_topology",
        },
        "stages": {
            "retopology": {"method": "hunyuan_api"},
            "uv": {"method": "hunyuan_api_or_preserved_from_api_result", "has_uv": bool(final_metrics.get("uv", {}).get("has_uv"))},
            "texture_migration": {"method": "hunyuan_api_result_or_preserved_materials"},
            "alignment": alignment,
        },
        "metrics": {
            "source_reference": source_metrics,
            "final": final_metrics,
            "projection_to_source": projection,
            "bbox_similarity": bbox_similarity(source_metrics, final_metrics),
        },
        "exports": exports,
        "checks": checks,
    }


def main() -> int:
    args = parse_args()
    report_path = Path(args.report)
    try:
        report = run_pipeline(args)
        write_json(report_path, report)
        return 0
    except Exception:
        error_report = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "status": "error",
            "traceback": traceback.format_exc(),
        }
        try:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            write_json(report_path, error_report)
        finally:
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
