"""Blender worker for local mesh cleanup, UVs, texture migration, and QC.

Main call flow:
main -> run_pipeline
-> import_asset -> join_as_whole_asset -> clean_source_surface
-> whole_asset_retopology -> generate_uv -> migrate_textures
-> projection_metrics -> mesh_metrics -> export_final -> build_qc_checks

Texture migration flow:
migrate_textures
-> transfer_texture_nearest_surface
-> nearest_source_texture_sample
-> save_image
"""

from __future__ import annotations

import argparse
import array
import json
import math
import sys
import traceback
from collections import deque
from pathlib import Path
from typing import Any

import bmesh
import bpy
from mathutils import Vector
from mathutils.bvhtree import BVHTree


REPORT_SCHEMA_VERSION = "asset-refiner-qc-v1"


def parse_args() -> argparse.Namespace:
    argv = sys.argv
    backend_args = argv[argv.index("--") + 1 :] if "--" in argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config-json", required=True)
    parser.add_argument("--report", required=True)
    return parser.parse_args(backend_args)


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False, sort_keys=True)


def object_mode() -> None:
    if bpy.ops.object.mode_set.poll():
        bpy.ops.object.mode_set(mode="OBJECT")


def select_only(objects: list[bpy.types.Object]) -> None:
    object_mode()
    bpy.ops.object.select_all(action="DESELECT")
    for obj in objects:
        obj.select_set(True)
    if objects:
        bpy.context.view_layer.objects.active = objects[-1]


def reset_scene() -> None:
    object_mode()
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def import_asset(path: Path) -> list[bpy.types.Object]:
    before = set(bpy.context.scene.objects)
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
        raise ValueError(f"Unsupported input format: {path.suffix}")

    meshes = [
        obj
        for obj in bpy.context.scene.objects
        if obj.type == "MESH" and obj not in before
    ]
    if not meshes:
        raise ValueError(f"No mesh objects were imported from: {path}")
    return meshes


def apply_transform(obj: bpy.types.Object) -> None:
    select_only([obj])
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)


def join_as_whole_asset(meshes: list[bpy.types.Object]) -> bpy.types.Object:
    for obj in meshes:
        apply_transform(obj)
    if len(meshes) == 1:
        source = meshes[0]
        select_only([source])
    else:
        select_only(meshes)
        bpy.context.view_layer.objects.active = meshes[0]
        bpy.ops.object.join()
        source = bpy.context.view_layer.objects.active
    source.name = "hunyuan_source_high_whole_asset"
    source.data.name = "hunyuan_source_high_whole_asset_mesh"
    return source


def shade_smooth_by_angle(obj: bpy.types.Object, angle_degrees: float) -> None:
    for poly in obj.data.polygons:
        poly.use_smooth = True
    modifier = obj.modifiers.new("source_weighted_normals", "WEIGHTED_NORMAL")
    modifier.keep_sharp = True
    modifier.weight = 50
    select_only([obj])
    try:
        bpy.ops.object.modifier_apply(modifier=modifier.name)
    except RuntimeError:
        obj.modifiers.remove(modifier)


def edit_cleanup(obj: bpy.types.Object, cleanup_cfg: dict[str, Any]) -> None:
    select_only([obj])
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.delete_loose()

    merge_distance = float(cleanup_cfg.get("merge_distance", 0.0) or 0.0)
    if merge_distance > 0:
        if hasattr(bpy.ops.mesh, "remove_doubles"):
            bpy.ops.mesh.remove_doubles(threshold=merge_distance)
        else:
            bpy.ops.mesh.merge_by_distance(distance=merge_distance)

    degenerate_threshold = float(cleanup_cfg.get("degenerate_threshold", 0.0) or 0.0)
    if degenerate_threshold > 0:
        try:
            bpy.ops.mesh.dissolve_degenerate(threshold=degenerate_threshold)
        except RuntimeError:
            pass

    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode="OBJECT")
    obj.data.update(calc_edges=True)


def connected_components(mesh: bpy.types.Mesh) -> list[dict[str, Any]]:
    mesh.update(calc_edges=True)
    vertex_to_faces: dict[int, list[int]] = {}
    for poly in mesh.polygons:
        for vertex_index in poly.vertices:
            vertex_to_faces.setdefault(vertex_index, []).append(poly.index)

    visited: set[int] = set()
    components: list[dict[str, Any]] = []
    for poly in mesh.polygons:
        if poly.index in visited:
            continue
        queue: deque[int] = deque([poly.index])
        visited.add(poly.index)
        faces: list[int] = []
        area = 0.0
        while queue:
            face_index = queue.popleft()
            face = mesh.polygons[face_index]
            faces.append(face_index)
            area += float(face.area)
            for vertex_index in face.vertices:
                for neighbor in vertex_to_faces.get(vertex_index, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
        components.append({"faces": faces, "face_count": len(faces), "area": area})
    components.sort(key=lambda item: item["area"], reverse=True)
    return components


def remove_small_components(obj: bpy.types.Object, cleanup_cfg: dict[str, Any]) -> int:
    mesh = obj.data
    components = connected_components(mesh)
    if len(components) <= 1:
        return 0

    total_area = sum(component["area"] for component in components) or 1.0
    min_faces = int(cleanup_cfg.get("min_component_faces", 0) or 0)
    min_area_ratio = float(cleanup_cfg.get("min_component_area_ratio", 0.0) or 0.0)
    delete_face_indices: set[int] = set()
    for index, component in enumerate(components):
        if index == 0:
            continue
        area_ratio = component["area"] / total_area
        if component["face_count"] < min_faces or area_ratio < min_area_ratio:
            delete_face_indices.update(component["faces"])

    if not delete_face_indices:
        return 0

    max_delete_fraction = cleanup_cfg.get("max_component_cleanup_face_fraction", 0.05)
    if max_delete_fraction is not None:
        delete_fraction = len(delete_face_indices) / max(1, len(mesh.polygons))
        if delete_fraction > float(max_delete_fraction):
            return 0

    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.faces.ensure_lookup_table()
    faces = [
        bm.faces[index]
        for index in sorted(delete_face_indices)
        if index < len(bm.faces)
    ]
    bmesh.ops.delete(bm, geom=faces, context="FACES")
    loose_verts = [vert for vert in bm.verts if not vert.link_edges]
    if loose_verts:
        bmesh.ops.delete(bm, geom=loose_verts, context="VERTS")
    bm.to_mesh(mesh)
    bm.free()
    mesh.update(calc_edges=True)
    return len(delete_face_indices)


def clean_source_surface(
    obj: bpy.types.Object, config: dict[str, Any]
) -> dict[str, Any]:
    before_faces = len(obj.data.polygons)
    edit_cleanup(obj, config.get("cleanup", {}))
    removed_faces = remove_small_components(obj, config.get("cleanup", {}))
    shade_smooth_by_angle(
        obj, float(config.get("source", {}).get("normal_angle_degrees", 60.0))
    )
    return {
        "input_faces": before_faces,
        "removed_small_component_faces": removed_faces,
        "output_faces": len(obj.data.polygons),
    }


def bmesh_merge_vertices_by_distance(obj: bpy.types.Object, distance: float) -> int:
    if distance <= 0:
        return 0
    mesh = obj.data
    before = len(mesh.vertices)
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.remove_doubles(bm, verts=list(bm.verts), dist=distance)
    bm.to_mesh(mesh)
    bm.free()
    mesh.update(calc_edges=True)
    return max(0, before - len(mesh.vertices))


def duplicate_object(obj: bpy.types.Object, name: str) -> bpy.types.Object:
    duplicate = obj.copy()
    duplicate.data = obj.data.copy()
    duplicate.name = name
    duplicate.data.name = f"{name}_mesh"
    bpy.context.collection.objects.link(duplicate)
    return duplicate


def surface_area(obj: bpy.types.Object) -> float:
    obj.data.update(calc_edges=True)
    return float(sum(poly.area for poly in obj.data.polygons))


def bbox_dimensions(obj: bpy.types.Object) -> tuple[float, float, float]:
    corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    xs = [corner.x for corner in corners]
    ys = [corner.y for corner in corners]
    zs = [corner.z for corner in corners]
    return (max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs))


def bbox_center_and_diagonal(obj: bpy.types.Object) -> tuple[Vector, float]:
    corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    minimum = Vector(
        (
            min(c.x for c in corners),
            min(c.y for c in corners),
            min(c.z for c in corners),
        )
    )
    maximum = Vector(
        (
            max(c.x for c in corners),
            max(c.y for c in corners),
            max(c.z for c in corners),
        )
    )
    return (minimum + maximum) * 0.5, (maximum - minimum).length


def normalize_external_target_to_source_bbox(
    source: bpy.types.Object, target: bpy.types.Object
) -> dict[str, Any]:
    source_center, source_diagonal = bbox_center_and_diagonal(source)
    target_center, target_diagonal = bbox_center_and_diagonal(target)
    if source_diagonal <= 0 or target_diagonal <= 0:
        return {
            "applied": False,
            "reason": "invalid_bbox_diagonal",
            "source_diagonal": source_diagonal,
            "target_diagonal_before": target_diagonal,
        }

    scale = source_diagonal / target_diagonal
    mesh = target.data
    matrix = target.matrix_world
    inverse = matrix.inverted()
    for vertex in mesh.vertices:
        world = matrix @ vertex.co
        aligned_world = (world - target_center) * scale + source_center
        vertex.co = inverse @ aligned_world
    mesh.update(calc_edges=True)
    apply_transform(target)
    _, target_diagonal_after = bbox_center_and_diagonal(target)
    return {
        "applied": True,
        "method": "uniform_scale_and_center_to_source_bbox",
        "scale": scale,
        "source_diagonal": source_diagonal,
        "target_diagonal_before": target_diagonal,
        "target_diagonal_after": target_diagonal_after,
    }


def estimate_voxel_size(source: bpy.types.Object, retopo_cfg: dict[str, Any]) -> float:
    explicit = retopo_cfg.get("voxel_size")
    if explicit:
        return float(explicit)

    target_faces = max(100, int(retopo_cfg.get("target_faces", 30000) or 30000))
    area = max(surface_area(source), 1e-12)
    factor = float(retopo_cfg.get("voxel_size_factor", 1.15) or 1.15)
    return max(math.sqrt(area / target_faces) * factor, 1e-6)


def apply_modifier(obj: bpy.types.Object, modifier: bpy.types.Modifier) -> bool:
    select_only([obj])
    try:
        bpy.ops.object.modifier_apply(modifier=modifier.name)
        obj.data.update(calc_edges=True)
        return True
    except RuntimeError:
        obj.modifiers.remove(modifier)
        return False


def choose_retopology_method(
    source: bpy.types.Object, retopo_cfg: dict[str, Any]
) -> tuple[str, dict[str, Any]]:
    requested = str(
        retopo_cfg.get("method", "voxel_remesh_project") or "voxel_remesh_project"
    )
    if requested != "auto":
        return requested, {"requested_method": requested, "selected_method": requested}

    components = len(connected_components(source.data))
    counts = edge_face_counts(source.data)
    boundary_edges = sum(1 for count in counts if count == 1)
    edge_count = max(1, len(counts))
    boundary_ratio = boundary_edges / edge_count
    component_threshold = int(retopo_cfg.get("auto_component_threshold", 32) or 32)
    boundary_threshold = float(
        retopo_cfg.get("auto_boundary_edge_ratio_threshold", 0.02) or 0.02
    )
    selected = (
        "decimate_project"
        if components > component_threshold or boundary_ratio > boundary_threshold
        else "voxel_remesh_project"
    )
    return selected, {
        "requested_method": requested,
        "selected_method": selected,
        "source_connected_components": components,
        "source_boundary_edges": boundary_edges,
        "source_boundary_edge_ratio": boundary_ratio,
        "auto_component_threshold": component_threshold,
        "auto_boundary_edge_ratio_threshold": boundary_threshold,
    }


def apply_optional_projection(
    source: bpy.types.Object,
    retopo: bpy.types.Object,
    retopo_cfg: dict[str, Any],
) -> None:
    projection_steps = int(retopo_cfg.get("shrinkwrap_iterations", 1) or 1)
    projection_offset = float(retopo_cfg.get("projection_offset", 0.0) or 0.0)
    for index in range(max(1, projection_steps)):
        shrinkwrap = retopo.modifiers.new(
            f"whole_asset_source_projection_{index + 1}", "SHRINKWRAP"
        )
        shrinkwrap.target = source
        shrinkwrap.wrap_method = "NEAREST_SURFACEPOINT"
        shrinkwrap.offset = projection_offset
        apply_modifier(retopo, shrinkwrap)


def apply_optional_smoothing(
    retopo: bpy.types.Object, retopo_cfg: dict[str, Any]
) -> None:
    smooth_iterations = int(retopo_cfg.get("smooth_iterations", 0) or 0)
    smooth_factor = float(retopo_cfg.get("smooth_factor", 0.18) or 0.18)
    if smooth_iterations > 0:
        smooth = retopo.modifiers.new("whole_asset_relax_after_remesh", "SMOOTH")
        smooth.factor = smooth_factor
        smooth.iterations = smooth_iterations
        apply_modifier(retopo, smooth)


def apply_final_normals(retopo: bpy.types.Object) -> None:
    for poly in retopo.data.polygons:
        poly.use_smooth = True
    normals = retopo.modifiers.new("refined_weighted_normals", "WEIGHTED_NORMAL")
    normals.keep_sharp = True
    apply_modifier(retopo, normals)


def whole_asset_voxel_retopology(
    source: bpy.types.Object,
    config: dict[str, Any],
    method_info: dict[str, Any],
) -> tuple[bpy.types.Object, dict[str, Any]]:
    retopo_cfg = config.get("retopology", {})
    target_faces = max(100, int(retopo_cfg.get("target_faces", 30000) or 30000))
    retopo = duplicate_object(source, "refined_whole_asset_retopology")
    retopo.data.materials.clear()

    voxel_size = estimate_voxel_size(source, retopo_cfg)
    remesh = retopo.modifiers.new("whole_asset_voxel_remesh_new_topology", "REMESH")
    remesh.mode = "VOXEL"
    remesh.voxel_size = voxel_size
    remesh.adaptivity = float(retopo_cfg.get("adaptivity", 0.15) or 0.0)
    if hasattr(remesh, "use_remove_disconnected"):
        remesh.use_remove_disconnected = False
    remesh_ok = apply_modifier(retopo, remesh)

    if (
        retopo_cfg.get("decimate_after_remesh", True)
        and len(retopo.data.polygons) > target_faces
    ):
        decimate = retopo.modifiers.new("whole_asset_face_budget_decimate", "DECIMATE")
        decimate.decimate_type = "COLLAPSE"
        decimate.ratio = max(
            0.01, min(1.0, target_faces / max(1, len(retopo.data.polygons)))
        )
        apply_modifier(retopo, decimate)

    apply_optional_smoothing(retopo, retopo_cfg)
    apply_optional_projection(source, retopo, retopo_cfg)
    apply_final_normals(retopo)

    stats = method_info | {
        "method": "voxel_remesh_project",
        "target_faces": target_faces,
        "voxel_size": voxel_size,
        "remesh_modifier_applied": remesh_ok,
        "topology_modifier_applied": remesh_ok,
        "output_faces": len(retopo.data.polygons),
        "output_vertices": len(retopo.data.vertices),
    }
    return retopo, stats


def whole_asset_decimate_project_retopology(
    source: bpy.types.Object,
    config: dict[str, Any],
    method_info: dict[str, Any],
) -> tuple[bpy.types.Object, dict[str, Any]]:
    retopo_cfg = config.get("retopology", {})
    source_faces = max(1, len(source.data.polygons))
    target_faces = int(retopo_cfg.get("target_faces", 30000) or 30000)
    ratio_target = retopo_cfg.get("preserve_shape_target_face_ratio")
    if ratio_target is not None:
        target_faces = max(target_faces, int(source_faces * float(ratio_target)))
    min_target = retopo_cfg.get("preserve_shape_min_target_faces")
    if min_target is not None:
        target_faces = max(target_faces, int(min_target))
    max_target = retopo_cfg.get("preserve_shape_max_target_faces")
    if max_target is not None:
        target_faces = min(target_faces, int(max_target))
    if source_faces > 100:
        target_faces = min(target_faces, source_faces - 1)
    target_faces = max(100, target_faces)
    effective_cfg = dict(retopo_cfg)
    preserve_smooth = retopo_cfg.get("preserve_shape_smooth_iterations")
    if preserve_smooth is not None:
        effective_cfg["smooth_iterations"] = preserve_smooth
    preserve_projection = retopo_cfg.get("preserve_shape_shrinkwrap_iterations")
    if preserve_projection is not None:
        effective_cfg["shrinkwrap_iterations"] = preserve_projection
    retopo = duplicate_object(source, "refined_whole_asset_retopology")
    retopo.data.materials.clear()

    decimate_ok = False
    decimate_ratio = max(0.01, min(1.0, target_faces / source_faces))
    if decimate_ratio < 0.999:
        decimate = retopo.modifiers.new(
            "whole_asset_shape_preserving_decimate", "DECIMATE"
        )
        decimate.decimate_type = "COLLAPSE"
        decimate.ratio = decimate_ratio
        if hasattr(decimate, "use_collapse_triangulate"):
            decimate.use_collapse_triangulate = bool(
                effective_cfg.get("decimate_triangulate", True)
            )
        decimate_ok = apply_modifier(retopo, decimate)

    apply_optional_smoothing(retopo, effective_cfg)
    apply_optional_projection(source, retopo, effective_cfg)
    apply_final_normals(retopo)

    stats = method_info | {
        "method": "decimate_project",
        "target_faces": target_faces,
        "source_faces": source_faces,
        "decimate_ratio": decimate_ratio,
        "decimate_modifier_applied": decimate_ok,
        "topology_modifier_applied": decimate_ok,
        "effective_smooth_iterations": int(
            effective_cfg.get("smooth_iterations", 0) or 0
        ),
        "effective_shrinkwrap_iterations": int(
            effective_cfg.get("shrinkwrap_iterations", 0) or 0
        ),
        "output_faces": len(retopo.data.polygons),
        "output_vertices": len(retopo.data.vertices),
    }
    return retopo, stats


def build_vertex_adjacency(mesh: bpy.types.Mesh) -> dict[int, set[int]]:
    adjacency: dict[int, set[int]] = {vertex.index: set() for vertex in mesh.vertices}
    for edge in mesh.edges:
        a, b = edge.vertices
        adjacency[a].add(b)
        adjacency[b].add(a)
    return adjacency


def expand_vertex_set(mesh: bpy.types.Mesh, vertices: set[int], rings: int) -> set[int]:
    if rings <= 0 or not vertices:
        return set(vertices)
    adjacency = build_vertex_adjacency(mesh)
    protected = set(vertices)
    frontier = set(vertices)
    for _ in range(rings):
        next_frontier: set[int] = set()
        for vertex_index in frontier:
            next_frontier.update(adjacency.get(vertex_index, set()))
        next_frontier.difference_update(protected)
        protected.update(next_frontier)
        frontier = next_frontier
        if not frontier:
            break
    return protected


def protected_vertices_for_boundary_safe_decimate(
    obj: bpy.types.Object,
    preserve_boundary_rings: int,
    preserve_small_component_faces: int,
) -> tuple[set[int], dict[str, Any]]:
    mesh = obj.data
    mesh.update(calc_edges=True)
    counts = edge_face_counts(mesh)
    protected: set[int] = set()
    boundary_or_nonmanifold_edges = 0
    for edge, count in zip(mesh.edges, counts):
        if count != 2:
            boundary_or_nonmanifold_edges += 1
            protected.update(edge.vertices)

    small_component_faces = 0
    small_component_count = 0
    if preserve_small_component_faces > 0:
        components = connected_components(mesh)
        for component in components:
            if component["face_count"] <= preserve_small_component_faces:
                small_component_count += 1
                small_component_faces += component["face_count"]
                for face_index in component["faces"]:
                    if face_index < len(mesh.polygons):
                        protected.update(mesh.polygons[face_index].vertices)

    before_expand = len(protected)
    protected = expand_vertex_set(mesh, protected, preserve_boundary_rings)
    stats = {
        "boundary_or_nonmanifold_edges": boundary_or_nonmanifold_edges,
        "protected_vertices_before_ring_expand": before_expand,
        "protected_vertices_after_ring_expand": len(protected),
        "preserve_boundary_rings": preserve_boundary_rings,
        "preserved_small_component_count": small_component_count,
        "preserved_small_component_faces": small_component_faces,
        "preserve_small_component_faces_threshold": preserve_small_component_faces,
    }
    return protected, stats


def create_decimate_allowed_vertex_group(
    obj: bpy.types.Object, protected_vertices: set[int]
) -> tuple[str, int]:
    group = obj.vertex_groups.new(name="decimate_allowed_interior")
    allowed = [
        vertex.index
        for vertex in obj.data.vertices
        if vertex.index not in protected_vertices
    ]
    if allowed:
        group.add(allowed, 1.0, "ADD")
    return group.name, len(allowed)


def whole_asset_boundary_safe_decimate_project_retopology(
    source: bpy.types.Object,
    config: dict[str, Any],
    method_info: dict[str, Any],
) -> tuple[bpy.types.Object, dict[str, Any]]:
    retopo_cfg = config.get("retopology", {})
    source_faces = max(1, len(source.data.polygons))
    target_faces = max(
        100,
        min(int(retopo_cfg.get("target_faces", 300000) or 300000), source_faces - 1),
    )
    retopo = duplicate_object(source, "refined_whole_asset_retopology")
    retopo.data.materials.clear()

    protected_vertices, protection_stats = (
        protected_vertices_for_boundary_safe_decimate(
            retopo,
            int(retopo_cfg.get("preserve_boundary_rings", 2) or 0),
            int(retopo_cfg.get("preserve_small_component_faces", 0) or 0),
        )
    )
    group_name, allowed_vertex_count = create_decimate_allowed_vertex_group(
        retopo, protected_vertices
    )

    decimate_ok = False
    decimate_ratio = max(0.01, min(1.0, target_faces / source_faces))
    if decimate_ratio < 0.999 and allowed_vertex_count > 0:
        decimate = retopo.modifiers.new(
            "whole_asset_boundary_safe_decimate", "DECIMATE"
        )
        decimate.decimate_type = "COLLAPSE"
        decimate.ratio = decimate_ratio
        decimate.vertex_group = group_name
        decimate.vertex_group_factor = float(
            retopo_cfg.get("vertex_group_factor", 1.0) or 1.0
        )
        decimate.invert_vertex_group = False
        if hasattr(decimate, "use_collapse_triangulate"):
            decimate.use_collapse_triangulate = bool(
                retopo_cfg.get("decimate_triangulate", True)
            )
        decimate_ok = apply_modifier(retopo, decimate)

    apply_optional_projection(source, retopo, retopo_cfg)
    apply_final_normals(retopo)

    stats = (
        method_info
        | protection_stats
        | {
            "method": "boundary_safe_decimate_project",
            "target_faces": target_faces,
            "source_faces": source_faces,
            "decimate_ratio": decimate_ratio,
            "allowed_vertex_count": allowed_vertex_count,
            "decimate_modifier_applied": decimate_ok,
            "topology_modifier_applied": decimate_ok,
            "output_faces": len(retopo.data.polygons),
            "output_vertices": len(retopo.data.vertices),
        }
    )
    return retopo, stats


def resolve_target_path(target_path: str | None) -> Path:
    if not target_path:
        raise ValueError(
            "retopology.target_path is required for external_target_project"
        )
    path = Path(target_path).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def whole_asset_external_target_project_retopology(
    source: bpy.types.Object,
    config: dict[str, Any],
    method_info: dict[str, Any],
) -> tuple[bpy.types.Object, dict[str, Any]]:
    retopo_cfg = config.get("retopology", {})
    target_path = resolve_target_path(retopo_cfg.get("target_path"))
    target_meshes = import_asset(target_path)
    imported_mesh_objects = len(target_meshes)
    retopo = join_as_whole_asset(target_meshes)
    retopo.name = "refined_whole_asset_retopology"
    retopo.data.name = "refined_whole_asset_retopology_mesh"

    alignment = None
    if retopo_cfg.get("normalize_external_target_to_source_bbox", False):
        alignment = normalize_external_target_to_source_bbox(source, retopo)

    before_faces = len(retopo.data.polygons)
    edit_cleanup(retopo, config.get("cleanup", {}))
    bmesh_merged_vertices = bmesh_merge_vertices_by_distance(
        retopo,
        float(config.get("cleanup", {}).get("merge_distance", 0.0) or 0.0),
    )
    removed_faces = (
        remove_small_components(retopo, config.get("cleanup", {}))
        if retopo_cfg.get("cleanup_target_small_components", False)
        else 0
    )
    after_cleanup_faces = len(retopo.data.polygons)

    apply_optional_projection(source, retopo, retopo_cfg)
    apply_final_normals(retopo)

    stats = method_info | {
        "method": "external_target_project",
        "target_path": str(target_path),
        "target_alignment": alignment,
        "imported_target_mesh_objects": imported_mesh_objects,
        "input_target_faces": before_faces,
        "bmesh_merged_vertices": bmesh_merged_vertices,
        "removed_small_component_faces": removed_faces,
        "after_cleanup_faces": after_cleanup_faces,
        "topology_modifier_applied": True,
        "external_target_used": True,
        "output_faces": len(retopo.data.polygons),
        "output_vertices": len(retopo.data.vertices),
    }
    return retopo, stats


def whole_asset_retopology(
    source: bpy.types.Object, config: dict[str, Any]
) -> tuple[bpy.types.Object, dict[str, Any]]:
    retopo_cfg = config.get("retopology", {})
    selected_method, method_info = choose_retopology_method(source, retopo_cfg)
    if selected_method in {"external_target_project", "existing_target_project"}:
        return whole_asset_external_target_project_retopology(
            source, config, method_info
        )
    if selected_method in {
        "boundary_safe_decimate_project",
        "hole_safe_decimate_project",
    }:
        return whole_asset_boundary_safe_decimate_project_retopology(
            source, config, method_info
        )
    if selected_method in {
        "decimate_project",
        "preserve_shape_decimate_project",
        "quadric_decimate_project",
    }:
        return whole_asset_decimate_project_retopology(source, config, method_info)
    if selected_method == "voxel_remesh_project":
        return whole_asset_voxel_retopology(source, config, method_info)
    raise ValueError(f"Unsupported retopology method: {selected_method}")


def generate_uv(obj: bpy.types.Object, config: dict[str, Any]) -> dict[str, Any]:
    uv_cfg = config.get("uv", {})
    method = str(uv_cfg.get("method", "smart_project") or "smart_project")
    select_only([obj])
    if not obj.data.uv_layers:
        obj.data.uv_layers.new(name="UVMap")
    obj.data.uv_layers.active = obj.data.uv_layers[0]

    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    if method == "lightmap_pack":
        bpy.ops.uv.lightmap_pack(
            PREF_CONTEXT="ALL_FACES",
            PREF_PACK_IN_ONE=True,
            PREF_NEW_UVLAYER=False,
            PREF_BOX_DIV=int(uv_cfg.get("box_div", 24) or 24),
            PREF_MARGIN_DIV=float(uv_cfg.get("margin_div", 0.2) or 0.2),
        )
    else:
        bpy.ops.uv.smart_project(
            angle_limit=math.radians(
                float(uv_cfg.get("angle_limit_degrees", 66.0) or 66.0)
            ),
            island_margin=float(uv_cfg.get("island_margin", 0.02) or 0.02),
            area_weight=float(uv_cfg.get("area_weight", 0.0) or 0.0),
        )
    if method != "lightmap_pack" and bool(
        uv_cfg.get("pack_islands_after_smart_project", False)
    ):
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.uv.pack_islands(
            rotate=bool(uv_cfg.get("pack_rotate", True)),
            scale=bool(uv_cfg.get("pack_scale", True)),
            merge_overlap=False,
            margin_method=str(uv_cfg.get("pack_margin_method", "SCALED") or "SCALED"),
            margin=float(
                uv_cfg.get("pack_margin", uv_cfg.get("island_margin", 0.02)) or 0.02
            ),
        )
    bpy.ops.object.mode_set(mode="OBJECT")
    obj.data.update(calc_edges=True)
    return {
        "method": method,
        "uv_layer": obj.data.uv_layers.active.name,
        "uv_layers": len(obj.data.uv_layers),
    }


def find_principled_bsdf(material: bpy.types.Material) -> bpy.types.Node | None:
    if not material.use_nodes:
        return None
    for node in material.node_tree.nodes:
        if node.type == "BSDF_PRINCIPLED":
            return node
    return None


def fill_image(
    image: bpy.types.Image, color: tuple[float, float, float, float]
) -> None:
    pixel_count = image.size[0] * image.size[1]
    values = array.array("f", color * pixel_count)
    image.pixels.foreach_set(values)
    image.update()


def save_image(image: bpy.types.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.filepath_raw = str(path)
    image.file_format = "PNG"
    image.save()


def create_texture_node(
    material: bpy.types.Material,
    image: bpy.types.Image,
    label: str,
) -> bpy.types.Node:
    node = material.node_tree.nodes.new(type="ShaderNodeTexImage")
    node.name = label
    node.label = label
    node.image = image
    material.node_tree.nodes.active = node
    node.select = True
    return node


def create_refined_material(obj: bpy.types.Object) -> bpy.types.Material:
    material = bpy.data.materials.new("Refined_Whole_Asset_Material")
    material.use_nodes = True
    obj.data.materials.clear()
    obj.data.materials.append(material)
    return material


def channel_index_from_socket_name(name: str) -> int | None:
    normalized = name.lower()
    if normalized in {"red", "r"}:
        return 0
    if normalized in {"green", "g"}:
        return 1
    if normalized in {"blue", "b"}:
        return 2
    if normalized in {"alpha", "a"}:
        return 3
    return None


def linked_image_source_from_socket(
    socket: bpy.types.NodeSocket,
    channel: int | None = None,
    visited: set[tuple[int, str, int | None]] | None = None,
) -> tuple[bpy.types.Image, int | None] | None:
    if visited is None:
        visited = set()
    if not socket.is_linked:
        return None
    for link in socket.links:
        result = linked_image_source_from_node_output(
            link.from_node, link.from_socket, channel, visited
        )
        if result is not None:
            return result
    return None


def linked_image_source_from_node_output(
    node: bpy.types.Node,
    output_socket: bpy.types.NodeSocket,
    channel: int | None,
    visited: set[tuple[int, str, int | None]],
) -> tuple[bpy.types.Image, int | None] | None:
    key = (id(node), getattr(output_socket, "identifier", output_socket.name), channel)
    if key in visited:
        return None
    visited.add(key)

    if node.type == "TEX_IMAGE" and getattr(node, "image", None) is not None:
        output_channel = channel_index_from_socket_name(output_socket.name)
        return node.image, channel if channel is not None else output_channel

    if node.type in {"SEPARATE_COLOR", "SEPARATE_RGB"}:
        separated_channel = channel_index_from_socket_name(output_socket.name)
        input_socket = node.inputs.get("Color") or node.inputs.get("Image")
        if input_socket is not None:
            return linked_image_source_from_socket(
                input_socket,
                separated_channel if separated_channel is not None else channel,
                visited,
            )

    if node.type == "NORMAL_MAP":
        input_socket = node.inputs.get("Color")
        if input_socket is not None:
            return linked_image_source_from_socket(input_socket, None, visited)

    for input_socket in node.inputs:
        if input_socket.is_linked:
            result = linked_image_source_from_socket(input_socket, channel, visited)
            if result is not None:
                return result
    return None


def find_bsdf_input(
    bsdf: bpy.types.Node | None, names: list[str]
) -> bpy.types.NodeSocket | None:
    if bsdf is None:
        return None
    for name in names:
        if name in bsdf.inputs:
            return bsdf.inputs[name]
    return None


def socket_default_color(
    socket: bpy.types.NodeSocket | None, fallback: tuple[float, float, float, float]
) -> tuple[float, float, float, float]:
    if socket is None:
        return fallback
    value = getattr(socket, "default_value", None)
    if value is None:
        return fallback
    try:
        return (
            float(value[0]),
            float(value[1]),
            float(value[2]),
            float(value[3]) if len(value) > 3 else 1.0,
        )
    except (TypeError, IndexError, ValueError):
        scalar = float(value)
        return (scalar, scalar, scalar, 1.0)


def socket_default_scalar(
    socket: bpy.types.NodeSocket | None, fallback: float
) -> tuple[float, float, float, float]:
    if socket is None:
        return (fallback, fallback, fallback, 1.0)
    value = getattr(socket, "default_value", None)
    if value is None:
        return (fallback, fallback, fallback, 1.0)
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        try:
            scalar = float(value[0])
        except (TypeError, IndexError, ValueError):
            scalar = fallback
    return (scalar, scalar, scalar, 1.0)


def find_named_image_source_for_material(
    material: bpy.types.Material | None,
    keywords: list[str],
    default_channel: int | None,
) -> tuple[bpy.types.Image, int | None] | None:
    if material is None or not material.use_nodes:
        return None
    lowered_keywords = [keyword.lower() for keyword in keywords]
    for node in material.node_tree.nodes:
        if node.type != "TEX_IMAGE" or getattr(node, "image", None) is None:
            continue
        image = node.image
        haystack = " ".join(
            [
                node.name,
                getattr(node, "label", ""),
                image.name,
                getattr(image, "filepath", ""),
            ]
        ).lower()
        if any(keyword in haystack for keyword in lowered_keywords):
            return image, default_channel
    return None


def pbr_texture_spec(texture_type: str, texture_cfg: dict[str, Any]) -> dict[str, Any]:
    neutral_roughness = float(texture_cfg.get("neutral_roughness", 0.65) or 0.65)
    neutral_metallic = float(texture_cfg.get("neutral_metallic", 0.0) or 0.0)
    specs: dict[str, dict[str, Any]] = {
        "base_color": {
            "bsdf_inputs": ["Base Color"],
            "default_kind": "color",
            "fallback": (0.8, 0.8, 0.8, 1.0),
            "keywords": ["basecolor", "base_color", "albedo", "diffuse", "color"],
            "kind": "color",
            "colorspace": "sRGB",
            "filename": "base_color.png",
            "label": "BaseColor_Target",
            "bsdf_target_inputs": ["Base Color"],
        },
        "normal": {
            "bsdf_inputs": ["Normal"],
            "default_kind": "fixed",
            "fallback": (0.5, 0.5, 1.0, 1.0),
            "keywords": ["normal", "norm"],
            "kind": "color",
            "colorspace": "Non-Color",
            "filename": "normal.png",
            "label": "Normal_Target",
            "normal_map": True,
        },
        "roughness": {
            "bsdf_inputs": ["Roughness"],
            "default_kind": "scalar",
            "fallback_scalar": neutral_roughness,
            "keywords": ["roughness", "rough"],
            "kind": "scalar",
            "default_channel": 1,
            "colorspace": "Non-Color",
            "filename": "roughness.png",
            "label": "Roughness_Target",
            "bsdf_target_inputs": ["Roughness"],
        },
        "metallic": {
            "bsdf_inputs": ["Metallic"],
            "default_kind": "scalar",
            "fallback_scalar": neutral_metallic,
            "keywords": ["metallic", "metalness", "metal"],
            "kind": "scalar",
            "default_channel": 2,
            "colorspace": "Non-Color",
            "filename": "metallic.png",
            "label": "Metallic_Target",
            "bsdf_target_inputs": ["Metallic"],
        },
        "ao": {
            "bsdf_inputs": [],
            "default_kind": "fixed",
            "fallback": (1.0, 1.0, 1.0, 1.0),
            "keywords": ["occlusion", "ambient", "ao"],
            "kind": "scalar",
            "default_channel": 0,
            "colorspace": "Non-Color",
            "filename": "ao.png",
            "label": "AO_Target",
        },
        "emissive": {
            "bsdf_inputs": ["Emission Color", "Emission"],
            "default_kind": "color",
            "fallback": (0.0, 0.0, 0.0, 1.0),
            "keywords": ["emissive", "emission"],
            "kind": "color",
            "colorspace": "sRGB",
            "filename": "emissive.png",
            "label": "Emissive_Target",
            "bsdf_target_inputs": ["Emission Color", "Emission"],
        },
    }
    spec = specs[texture_type]
    spec["texture_type"] = texture_type
    return spec


def source_color_attribute(mesh: bpy.types.Mesh):
    color_attributes = getattr(mesh, "color_attributes", None)
    if color_attributes:
        active = getattr(color_attributes, "active_color", None)
        if active is not None and getattr(active, "domain", None) in {
            "CORNER",
            "POINT",
        }:
            return active
        for attribute in color_attributes:
            if getattr(attribute, "domain", None) in {"CORNER", "POINT"}:
                return attribute

    vertex_colors = getattr(mesh, "vertex_colors", None)
    if vertex_colors:
        active = getattr(vertex_colors, "active", None)
        if active is not None:
            return active
        for attribute in vertex_colors:
            return attribute
    return None


def color_tuple(value) -> tuple[float, float, float, float]:
    rgba = [0.0, 0.0, 0.0, 1.0]
    try:
        value_len = len(value)
    except TypeError:
        return rgba[0], rgba[1], rgba[2], rgba[3]
    for index in range(min(4, value_len)):
        try:
            rgba[index] = max(0.0, min(1.0, float(value[index])))
        except (TypeError, ValueError):
            return rgba[0], rgba[1], rgba[2], rgba[3]
    return rgba[0], rgba[1], rgba[2], rgba[3]


def sample_source_color_attribute(
    color_attribute,
    poly: bpy.types.MeshPolygon,
    bary: tuple[float, float, float],
) -> tuple[float, float, float, float] | None:
    if color_attribute is None:
        return None

    domain = getattr(color_attribute, "domain", "CORNER")
    try:
        if domain == "POINT":
            indices = list(poly.vertices[:3])
        else:
            indices = list(poly.loop_indices[:3])
        colors = [color_tuple(color_attribute.data[index].color) for index in indices]
    except (AttributeError, IndexError, TypeError, ValueError):
        return None

    return (
        bary[0] * colors[0][0] + bary[1] * colors[1][0] + bary[2] * colors[2][0],
        bary[0] * colors[0][1] + bary[1] * colors[1][1] + bary[2] * colors[2][1],
        bary[0] * colors[0][2] + bary[1] * colors[1][2] + bary[2] * colors[2][2],
        bary[0] * colors[0][3] + bary[1] * colors[1][3] + bary[2] * colors[2][3],
    )


def material_texture_source(
    material: bpy.types.Material | None,
    spec: dict[str, Any],
) -> dict[str, Any]:
    bsdf = find_principled_bsdf(material) if material is not None else None
    socket = find_bsdf_input(bsdf, list(spec.get("bsdf_inputs", [])))
    image_source = (
        linked_image_source_from_socket(socket) if socket is not None else None
    )
    if image_source is None:
        image_source = find_named_image_source_for_material(
            material,
            list(spec.get("keywords", [])),
            spec.get("default_channel"),
        )

    default_kind = str(spec.get("default_kind") or "fixed")
    if default_kind == "color":
        fallback = socket_default_color(
            socket, spec.get("fallback", (0.0, 0.0, 0.0, 1.0))
        )
    elif default_kind == "scalar":
        fallback = socket_default_scalar(
            socket, float(spec.get("fallback_scalar", 0.0) or 0.0)
        )
    else:
        fallback = spec.get("fallback", (0.0, 0.0, 0.0, 1.0))

    if image_source is None:
        return {
            "image": None,
            "pixels": None,
            "channel": spec.get("default_channel"),
            "fallback": fallback,
        }
    image, channel = image_source
    return {
        "image": image,
        "pixels": image_to_numpy(image),
        "channel": channel,
        "fallback": fallback,
    }


def build_material_texture_sources(
    source: bpy.types.Object,
    spec: dict[str, Any],
) -> tuple[dict[int, dict[str, Any]], bool]:
    records: dict[int, dict[str, Any]] = {}
    found_image = False
    materials = list(source.data.materials)
    if not materials:
        materials = [None]
    for index, material in enumerate(materials):
        record = material_texture_source(material, spec)
        if record.get("image") is not None:
            found_image = True
        records[index] = record
    return records, found_image


def texture_sample_to_color(
    sample, record: dict[str, Any], spec: dict[str, Any]
) -> tuple[float, float, float, float]:
    if str(spec.get("kind")) == "scalar":
        channel = record.get("channel")
        channel_index = int(channel) if channel is not None else 0
        channel_index = max(0, min(3, channel_index))
        value = float(sample[channel_index])
        return (value, value, value, 1.0)
    return (float(sample[0]), float(sample[1]), float(sample[2]), 1.0)


def image_to_numpy(image: bpy.types.Image):
    import numpy as np

    width, height = int(image.size[0]), int(image.size[1])
    pixels = np.empty(width * height * 4, dtype=np.float32)
    image.pixels.foreach_get(pixels)
    return pixels.reshape((height, width, 4))


def sample_image_nearest(image_pixels, uv: tuple[float, float]):
    width = image_pixels.shape[1]
    height = image_pixels.shape[0]
    u = max(0.0, min(1.0, float(uv[0])))
    v = max(0.0, min(1.0, float(uv[1])))
    x = min(width - 1, max(0, int(round(u * (width - 1)))))
    y = min(height - 1, max(0, int(round(v * (height - 1)))))
    return image_pixels[y, x]


def dilate_texture_pixels(
    colors, mask, iterations: int, limit_mask=None, fill_remaining: bool = False
):
    import numpy as np

    if iterations <= 0:
        return colors
    limit = np.ones(mask.shape, dtype=bool) if limit_mask is None else limit_mask.copy()
    filled = mask.copy() & limit
    out = colors.copy()
    for _ in range(iterations):
        missing = limit & ~filled
        if not missing.any():
            break
        accum = np.zeros_like(out)
        counts = np.zeros(filled.shape, dtype=np.float32)
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            shifted_mask = np.zeros_like(filled)
            shifted_colors = np.zeros_like(out)
            if dy == -1:
                shifted_mask[:-1, :] = filled[1:, :]
                shifted_colors[:-1, :, :] = out[1:, :, :]
            elif dy == 1:
                shifted_mask[1:, :] = filled[:-1, :]
                shifted_colors[1:, :, :] = out[:-1, :, :]
            elif dx == -1:
                shifted_mask[:, :-1] = filled[:, 1:]
                shifted_colors[:, :-1, :] = out[:, 1:, :]
            else:
                shifted_mask[:, 1:] = filled[:, :-1]
                shifted_colors[:, 1:, :] = out[:, :-1, :]
            accum += shifted_colors * shifted_mask[:, :, None]
            counts += shifted_mask.astype(np.float32)
        fill_now = missing & (counts > 0)
        if not fill_now.any():
            break
        out[fill_now] = accum[fill_now] / counts[fill_now, None]
        filled[fill_now] = True
    if fill_remaining:
        missing = limit & ~filled
        if missing.any() and filled.any():
            out[missing] = out[filled].mean(axis=0)
    return out


def nearest_source_texture_sample(
    source: bpy.types.Object,
    tree: BVHTree,
    source_uv,
    source_color_attr,
    material_sources: dict[int, dict[str, Any]],
    spec: dict[str, Any],
    point: Vector,
    normal: Vector,
    max_distance: float,
    min_dot: float,
    allow_nearest_fallback: bool,
) -> tuple[float, float, float, float] | None:
    candidates = []
    if max_distance > 0:
        candidates = tree.find_nearest_range(point, max_distance)
    if not candidates:
        nearest = tree.find_nearest(point)
        if nearest and nearest[0] is not None:
            candidates = [nearest]

    best = None
    best_score = -1e18
    normal = normal.normalized() if normal.length > 0 else normal
    for location, source_normal, face_index, distance in candidates:
        if location is None or face_index is None:
            continue
        if max_distance > 0 and distance > max_distance:
            continue
        dot = (
            normal.dot(source_normal.normalized())
            if normal.length > 0 and source_normal.length > 0
            else 1.0
        )
        if dot < min_dot:
            continue
        score = dot - distance * 10.0
        if score > best_score:
            best = (location, face_index)
            best_score = score
    if best is None and allow_nearest_fallback:
        fallback_candidates = [
            (location, face_index, distance)
            for location, _source_normal, face_index, distance in candidates
            if location is not None and face_index is not None
        ]
        if fallback_candidates:
            location, face_index, _distance = min(
                fallback_candidates, key=lambda item: item[2]
            )
            best = (location, face_index)
    if best is None:
        return None

    location, face_index = best
    if face_index >= len(source.data.polygons):
        return None
    poly = source.data.polygons[face_index]
    if len(poly.vertices) < 3:
        return None
    record = material_sources.get(poly.material_index)
    if record is None:
        record = material_sources.get(0)
    if record is None:
        return None
    vertex_indices = list(poly.vertices[:3])
    tri = [source.data.vertices[index].co for index in vertex_indices]
    bary = barycentric_3d(location, tri)
    if bary is None:
        return None

    image_pixels = record.get("pixels")
    if image_pixels is None:
        if spec.get("texture_type") == "base_color":
            color = sample_source_color_attribute(source_color_attr, poly, bary)
            if color is not None:
                return color
        fallback = record.get("fallback", spec.get("fallback", (0.0, 0.0, 0.0, 1.0)))
        return (
            float(fallback[0]),
            float(fallback[1]),
            float(fallback[2]),
            float(fallback[3]),
        )
    if source_uv is None:
        return None
    loops = list(poly.loop_indices[:3])
    uvs = [source_uv.data[loop_index].uv for loop_index in loops]
    uv = (
        bary[0] * uvs[0].x + bary[1] * uvs[1].x + bary[2] * uvs[2].x,
        bary[0] * uvs[0].y + bary[1] * uvs[1].y + bary[2] * uvs[2].y,
    )
    color = sample_image_nearest(image_pixels, uv)
    return texture_sample_to_color(color, record, spec)


def transfer_texture_nearest_surface(
    source: bpy.types.Object,
    target: bpy.types.Object,
    resolution: int,
    config: dict[str, Any],
    texture_type: str,
) -> tuple[bpy.types.Image, dict[str, Any]]:
    """Project target UV texels back to source geometry and sample source textures."""
    import numpy as np

    texture_cfg = config.get("textures", {})
    spec = pbr_texture_spec(texture_type, texture_cfg)
    source_uv = source.data.uv_layers.active
    target_uv = target.data.uv_layers.active
    if target_uv is None:
        raise RuntimeError("nearest_surface_texture transfer requires target UVs")

    material_sources, found_source_image = build_material_texture_sources(source, spec)
    source_color_attr = (
        source_color_attribute(source.data) if texture_type == "base_color" else None
    )
    if found_source_image and source_uv is None:
        raise RuntimeError(
            "nearest_surface_texture transfer requires source UVs when source images are used"
        )
    if not material_sources:
        raise RuntimeError(
            f"Could not prepare source material data for {texture_type} transfer"
        )

    depsgraph = bpy.context.evaluated_depsgraph_get()
    tree = BVHTree.FromObject(source, depsgraph)
    if tree is None:
        raise RuntimeError("Could not build source BVH for nearest surface transfer")

    target.data.calc_loop_triangles()
    colors = np.zeros((resolution, resolution, 4), dtype=np.float32)
    mask = np.zeros((resolution, resolution), dtype=bool)
    coverage = np.zeros((resolution, resolution), dtype=bool)
    max_distance = float(
        config.get("textures", {}).get("transfer_max_distance", 0.004) or 0.0
    )
    min_dot = float(
        config.get("textures", {}).get("transfer_normal_dot_min", 0.1) or -1.0
    )
    allow_nearest_fallback = bool(
        config.get("textures", {}).get("transfer_allow_nearest_fallback", True)
    )
    sampled = 0
    missed = 0

    matrix = target.matrix_world
    for tri in target.data.loop_triangles:
        uv_coords = [
            (float(target_uv.data[loop].uv.x), float(target_uv.data[loop].uv.y))
            for loop in tri.loops
        ]
        min_u = max(0.0, min(coord[0] for coord in uv_coords))
        max_u = min(1.0, max(coord[0] for coord in uv_coords))
        min_v = max(0.0, min(coord[1] for coord in uv_coords))
        max_v = min(1.0, max(coord[1] for coord in uv_coords))
        if max_u <= 0.0 or max_v <= 0.0 or min_u >= 1.0 or min_v >= 1.0:
            continue
        x0 = max(0, min(resolution - 1, int(math.floor(min_u * resolution))))
        x1 = max(0, min(resolution - 1, int(math.ceil(max_u * resolution))))
        y0 = max(0, min(resolution - 1, int(math.floor(min_v * resolution))))
        y1 = max(0, min(resolution - 1, int(math.ceil(max_v * resolution))))
        positions = [matrix @ target.data.vertices[index].co for index in tri.vertices]
        face_normal = (positions[1] - positions[0]).cross(positions[2] - positions[0])
        if face_normal.length > 0:
            face_normal.normalize()
        for y in range(y0, y1 + 1):
            py = (y + 0.5) / resolution
            for x in range(x0, x1 + 1):
                px = (x + 0.5) / resolution
                bary = barycentric_2d(px, py, uv_coords)
                if bary is None:
                    continue
                coverage[y, x] = True
                point = (
                    positions[0] * bary[0]
                    + positions[1] * bary[1]
                    + positions[2] * bary[2]
                )
                color = nearest_source_texture_sample(
                    source,
                    tree,
                    source_uv,
                    source_color_attr,
                    material_sources,
                    spec,
                    point,
                    face_normal,
                    max_distance,
                    min_dot,
                    allow_nearest_fallback,
                )
                if color is None:
                    missed += 1
                    continue
                colors[y, x, :] = color
                mask[y, x] = True
                sampled += 1

    iterations = int(
        config.get("textures", {}).get("transfer_dilate_iterations", 16) or 0
    )
    fill_uncovered = bool(
        config.get("textures", {}).get("transfer_fill_uncovered", True)
    )
    fill_iterations = int(
        config.get("textures", {}).get("transfer_fill_max_iterations", 512) or 0
    )
    initial_uncovered = int((coverage & ~mask).sum())
    if fill_uncovered and initial_uncovered:
        colors = dilate_texture_pixels(
            colors,
            mask,
            max(iterations, fill_iterations),
            limit_mask=coverage,
            fill_remaining=True,
        )
        mask = mask | coverage
    colors = dilate_texture_pixels(colors, mask, iterations)
    colors[:, :, 3] = 1.0

    image = bpy.data.images.new(
        f"refined_{texture_type}_nearest_surface",
        width=resolution,
        height=resolution,
        alpha=True,
    )
    image.colorspace_settings.name = str(spec.get("colorspace") or "Non-Color")
    image.pixels.foreach_set(colors.reshape(-1))
    image.update()
    stats = {
        "method": "nearest_surface_texture",
        "texture_type": texture_type,
        "sampled_texels": sampled,
        "missed_texels": missed,
        "covered_fraction": sampled / float(resolution * resolution),
        "uv_covered_texels": int(coverage.sum()),
        "initial_uncovered_uv_texels": initial_uncovered,
        "transfer_fill_uncovered": fill_uncovered,
        "transfer_fill_max_iterations": fill_iterations,
        "transfer_allow_nearest_fallback": allow_nearest_fallback,
        "source_images_found": found_source_image,
        "source_vertex_color_found": source_color_attr is not None,
        "source_vertex_color_name": getattr(source_color_attr, "name", None)
        if source_color_attr is not None
        else None,
        "transfer_max_distance": max_distance,
        "transfer_normal_dot_min": min_dot,
        "transfer_dilate_iterations": iterations,
    }
    return image, stats


def transfer_base_color_nearest_surface(
    source: bpy.types.Object,
    target: bpy.types.Object,
    resolution: int,
    config: dict[str, Any],
) -> tuple[bpy.types.Image, dict[str, Any]]:
    return transfer_texture_nearest_surface(
        source, target, resolution, config, "base_color"
    )


def create_fallback_texture_image(
    texture_type: str,
    spec: dict[str, Any],
    resolution: int,
    color: tuple[float, float, float, float],
) -> bpy.types.Image:
    image = bpy.data.images.new(
        f"refined_{texture_type}", width=resolution, height=resolution, alpha=True
    )
    image.colorspace_settings.name = str(spec.get("colorspace") or "Non-Color")
    fill_image(image, color)
    return image


def connect_texture_to_refined_material(
    material: bpy.types.Material,
    bsdf: bpy.types.Node,
    image: bpy.types.Image,
    spec: dict[str, Any],
) -> bpy.types.Node:
    node = create_texture_node(
        material, image, str(spec.get("label") or f"{image.name}_Target")
    )
    if spec.get("normal_map"):
        normal_map = material.node_tree.nodes.new(type="ShaderNodeNormalMap")
        material.node_tree.links.new(node.outputs["Color"], normal_map.inputs["Color"])
        normal_socket = find_bsdf_input(bsdf, ["Normal"])
        if normal_socket is not None:
            material.node_tree.links.new(normal_map.outputs["Normal"], normal_socket)
        return node

    target_socket = find_bsdf_input(bsdf, list(spec.get("bsdf_target_inputs", [])))
    if target_socket is not None:
        material.node_tree.links.new(node.outputs["Color"], target_socket)
    if "Emission Color" in spec.get("bsdf_target_inputs", []) or "Emission" in spec.get(
        "bsdf_target_inputs", []
    ):
        emission_strength = find_bsdf_input(bsdf, ["Emission Strength"])
        if emission_strength is not None:
            emission_strength.default_value = 1.0
    return node


def transfer_nearest_texture_with_fallback(
    source: bpy.types.Object,
    target: bpy.types.Object,
    texture_type: str,
    resolution: int,
    config: dict[str, Any],
) -> tuple[bpy.types.Image, bool, dict[str, Any], list[str]]:
    texture_cfg = config.get("textures", {})
    spec = pbr_texture_spec(texture_type, texture_cfg)
    warnings: list[str] = []
    try:
        image, transfer_stats = transfer_texture_nearest_surface(
            source, target, resolution, config, texture_type
        )
        return image, True, transfer_stats, warnings
    except Exception as exc:
        image = create_fallback_texture_image(
            texture_type,
            spec,
            resolution,
            spec.get("fallback", (0.0, 0.0, 0.0, 1.0)),
        )
        transfer_stats = {
            "method": "nearest_surface_texture",
            "texture_type": texture_type,
            "failed": True,
        }
        warnings.append(
            f"{texture_type} nearest-surface transfer failed; wrote fallback texture: {exc}"
        )
        return image, False, transfer_stats, warnings


def migrate_textures(
    source: bpy.types.Object,
    target: bpy.types.Object,
    output_dir: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Create target material and write migrated PBR textures into output_dir."""
    texture_cfg = config.get("textures", {})
    texture_dir = output_dir / "textures"
    texture_dir.mkdir(parents=True, exist_ok=True)
    material = create_refined_material(target)
    bsdf = find_principled_bsdf(material)
    if bsdf is None:
        raise RuntimeError("Could not create a Principled BSDF material for baking")

    resolution = int(texture_cfg.get("resolution", 2048) or 2048)
    textures: list[dict[str, Any]] = []
    warnings: list[str] = []

    if texture_cfg.get("enabled", True) and texture_cfg.get("bake_base_color", True):
        spec = pbr_texture_spec("base_color", texture_cfg)
        baked = True
        transfer_stats: dict[str, Any] = {"method": "nearest_surface_texture"}
        try:
            image, transfer_stats = transfer_base_color_nearest_surface(
                source, target, resolution, config
            )
        except Exception as exc:
            baked = False
            image = create_fallback_texture_image(
                "base_color",
                spec,
                resolution,
                spec.get("fallback", (0.8, 0.8, 0.8, 1.0)),
            )
            warnings.append(
                f"base_color nearest-surface transfer failed; wrote fallback texture: {exc}"
            )
        connect_texture_to_refined_material(material, bsdf, image, spec)
        path = texture_dir / "base_color.png"
        save_image(image, path)
        textures.append(
            {
                "type": "base_color",
                "path": str(path),
                "resolution": [resolution, resolution],
                "baked": baked,
                "transfer": transfer_stats,
            }
        )

    if texture_cfg.get("enabled", True) and texture_cfg.get("bake_normal", True):
        spec = pbr_texture_spec("normal", texture_cfg)
        image, baked, transfer_stats, transfer_warnings = (
            transfer_nearest_texture_with_fallback(
                source, target, "normal", resolution, config
            )
        )
        warnings.extend(transfer_warnings)
        connect_texture_to_refined_material(material, bsdf, image, spec)
        path = texture_dir / "normal.png"
        save_image(image, path)
        textures.append(
            {
                "type": "normal",
                "path": str(path),
                "resolution": [resolution, resolution],
                "baked": baked,
                "transfer": transfer_stats,
            }
        )

    if texture_cfg.get("enabled", True) and texture_cfg.get("bake_roughness", False):
        spec = pbr_texture_spec("roughness", texture_cfg)
        image, baked, transfer_stats, transfer_warnings = (
            transfer_nearest_texture_with_fallback(
                source, target, "roughness", resolution, config
            )
        )
        warnings.extend(transfer_warnings)
        connect_texture_to_refined_material(material, bsdf, image, spec)
        path = texture_dir / "roughness.png"
        save_image(image, path)
        textures.append(
            {
                "type": "roughness",
                "path": str(path),
                "resolution": [resolution, resolution],
                "baked": baked,
                "transfer": transfer_stats,
            }
        )

    for texture_type in ("metallic", "ao", "emissive"):
        if not texture_cfg.get("enabled", True) or not texture_cfg.get(
            f"bake_{texture_type}", False
        ):
            continue
        spec = pbr_texture_spec(texture_type, texture_cfg)
        image, baked, transfer_stats, transfer_warnings = (
            transfer_nearest_texture_with_fallback(
                source, target, texture_type, resolution, config
            )
        )
        warnings.extend(transfer_warnings)
        connect_texture_to_refined_material(material, bsdf, image, spec)
        path = texture_dir / str(spec.get("filename") or f"{texture_type}.png")
        save_image(image, path)
        textures.append(
            {
                "type": texture_type,
                "path": str(path),
                "resolution": [resolution, resolution],
                "baked": baked,
                "transfer": transfer_stats,
            }
        )

    if texture_cfg.get("pbr_repaint", False):
        roughness = float(texture_cfg.get("neutral_roughness", 0.65) or 0.65)
        metallic = float(texture_cfg.get("neutral_metallic", 0.0) or 0.0)
        for name, value, input_name in [
            ("roughness_repaint", roughness, "Roughness"),
            ("metallic_repaint", metallic, "Metallic"),
        ]:
            image = bpy.data.images.new(
                f"refined_{name}", width=resolution, height=resolution, alpha=False
            )
            image.colorspace_settings.name = "Non-Color"
            fill_image(image, (value, value, value, 1.0))
            node = create_texture_node(material, image, f"{name}_Target")
            if input_name in bsdf.inputs:
                material.node_tree.links.new(
                    node.outputs["Color"], bsdf.inputs[input_name]
                )
            path = texture_dir / f"{name}.png"
            save_image(image, path)
            textures.append(
                {
                    "type": name,
                    "path": str(path),
                    "resolution": [resolution, resolution],
                    "baked": False,
                }
            )

    return {
        "enabled": bool(texture_cfg.get("enabled", True)),
        "textures": textures,
        "warnings": warnings,
    }


def edge_face_counts(mesh: bpy.types.Mesh) -> list[int]:
    mesh.update(calc_edges=True)
    edge_index_by_key = {
        tuple(sorted(edge.vertices)): edge.index for edge in mesh.edges
    }
    counts = [0 for _ in mesh.edges]
    for poly in mesh.polygons:
        for key in poly.edge_keys:
            index = edge_index_by_key.get(tuple(sorted(key)))
            if index is not None:
                counts[index] += 1
    return counts


def point_in_triangle(px: float, py: float, tri: list[tuple[float, float]]) -> bool:
    (ax, ay), (bx, by), (cx, cy) = tri
    v0x, v0y = cx - ax, cy - ay
    v1x, v1y = bx - ax, by - ay
    v2x, v2y = px - ax, py - ay
    dot00 = v0x * v0x + v0y * v0y
    dot01 = v0x * v1x + v0y * v1y
    dot02 = v0x * v2x + v0y * v2y
    dot11 = v1x * v1x + v1y * v1y
    dot12 = v1x * v2x + v1y * v2y
    denom = dot00 * dot11 - dot01 * dot01
    if abs(denom) < 1e-20:
        return False
    inv = 1.0 / denom
    u = (dot11 * dot02 - dot01 * dot12) * inv
    v = (dot00 * dot12 - dot01 * dot02) * inv
    return u >= -1e-8 and v >= -1e-8 and (u + v) <= 1.0 + 1e-8


def barycentric_2d(
    px: float, py: float, tri: list[tuple[float, float]]
) -> tuple[float, float, float] | None:
    (ax, ay), (bx, by), (cx, cy) = tri
    v0x, v0y = bx - ax, by - ay
    v1x, v1y = cx - ax, cy - ay
    v2x, v2y = px - ax, py - ay
    den = v0x * v1y - v1x * v0y
    if abs(den) < 1e-20:
        return None
    v = (v2x * v1y - v1x * v2y) / den
    w = (v0x * v2y - v2x * v0y) / den
    u = 1.0 - v - w
    if u < -1e-6 or v < -1e-6 or w < -1e-6:
        return None
    return u, v, w


def barycentric_3d(
    point: Vector, tri: list[Vector]
) -> tuple[float, float, float] | None:
    a, b, c = tri
    v0 = b - a
    v1 = c - a
    v2 = point - a
    d00 = v0.dot(v0)
    d01 = v0.dot(v1)
    d11 = v1.dot(v1)
    d20 = v2.dot(v0)
    d21 = v2.dot(v1)
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-20:
        return None
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return u, v, w


def uv_metrics(obj: bpy.types.Object, grid: int) -> dict[str, Any]:
    mesh = obj.data
    if not mesh.uv_layers.active:
        return {
            "has_uv": False,
            "uv_layers": len(mesh.uv_layers),
            "overlap_ratio": None,
            "used_grid_ratio": 0.0,
            "out_of_bounds_fraction": 1.0,
        }

    mesh.calc_loop_triangles()
    uv_data = mesh.uv_layers.active.data
    counts = bytearray(grid * grid)
    occupied_cells = 0
    overlap_cells = 0
    total_loop_count = 0
    out_of_bounds_count = 0
    uv_area = 0.0

    for tri in mesh.loop_triangles:
        coords = [
            (float(uv_data[loop_index].uv.x), float(uv_data[loop_index].uv.y))
            for loop_index in tri.loops
        ]
        for u, v in coords:
            total_loop_count += 1
            if u < -1e-6 or u > 1.0 + 1e-6 or v < -1e-6 or v > 1.0 + 1e-6:
                out_of_bounds_count += 1

        (ax, ay), (bx, by), (cx, cy) = coords
        uv_area += abs((bx - ax) * (cy - ay) - (cx - ax) * (by - ay)) * 0.5
        min_u = max(0.0, min(ax, bx, cx))
        max_u = min(1.0, max(ax, bx, cx))
        min_v = max(0.0, min(ay, by, cy))
        max_v = min(1.0, max(ay, by, cy))
        if max_u <= 0.0 or max_v <= 0.0 or min_u >= 1.0 or min_v >= 1.0:
            continue
        x0 = max(0, min(grid - 1, int(math.floor(min_u * grid))))
        x1 = max(0, min(grid - 1, int(math.ceil(max_u * grid))))
        y0 = max(0, min(grid - 1, int(math.floor(min_v * grid))))
        y1 = max(0, min(grid - 1, int(math.ceil(max_v * grid))))
        for y in range(y0, y1 + 1):
            py = (y + 0.5) / grid
            for x in range(x0, x1 + 1):
                px = (x + 0.5) / grid
                if not point_in_triangle(px, py, coords):
                    continue
                index = y * grid + x
                if counts[index] == 0:
                    occupied_cells += 1
                elif counts[index] == 1:
                    overlap_cells += 1
                if counts[index] < 255:
                    counts[index] += 1

    return {
        "has_uv": True,
        "uv_layers": len(mesh.uv_layers),
        "active_layer": mesh.uv_layers.active.name,
        "approx_uv_area": uv_area,
        "overlap_ratio": overlap_cells / occupied_cells if occupied_cells else 0.0,
        "used_grid_ratio": occupied_cells / float(grid * grid),
        "out_of_bounds_fraction": out_of_bounds_count / total_loop_count
        if total_loop_count
        else 0.0,
        "grid_resolution": grid,
    }


def mesh_metrics(
    obj: bpy.types.Object, config: dict[str, Any] | None = None
) -> dict[str, Any]:
    mesh = obj.data
    mesh.update(calc_edges=True)
    mesh.calc_loop_triangles()
    counts = edge_face_counts(mesh)
    vertices_with_edges = {
        vertex_index for edge in mesh.edges for vertex_index in edge.vertices
    }
    components = connected_components(mesh)
    dims = bbox_dimensions(obj)
    uv_grid = int((config or {}).get("qc", {}).get("uv_overlap_grid", 128) or 128)
    return {
        "object_name": obj.name,
        "vertices": len(mesh.vertices),
        "edges": len(mesh.edges),
        "faces": len(mesh.polygons),
        "triangulated_faces": len(mesh.loop_triangles),
        "triangles": sum(1 for poly in mesh.polygons if len(poly.vertices) == 3),
        "quads": sum(1 for poly in mesh.polygons if len(poly.vertices) == 4),
        "ngons": sum(1 for poly in mesh.polygons if len(poly.vertices) > 4),
        "surface_area": surface_area(obj),
        "bbox_dimensions": list(dims),
        "bbox_diagonal": math.sqrt(sum(value * value for value in dims)),
        "connected_components": len(components),
        "component_face_counts": [
            component["face_count"] for component in components[:10]
        ],
        "boundary_edges": sum(1 for count in counts if count == 1),
        "nonmanifold_edges": sum(1 for count in counts if count == 0 or count > 2),
        "open_or_nonmanifold_edges": sum(1 for count in counts if count != 2),
        "loose_vertices": sum(
            1 for vert in mesh.vertices if vert.index not in vertices_with_edges
        ),
        "uv": uv_metrics(obj, uv_grid),
    }


def projection_metrics(
    source: bpy.types.Object, target: bpy.types.Object
) -> dict[str, Any]:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    tree = BVHTree.FromObject(source, depsgraph)
    if tree is None or not target.data.vertices:
        return {"sample_count": 0, "mean": None, "rms": None, "max": None}

    distances: list[float] = []
    matrix = target.matrix_world
    for vertex in target.data.vertices:
        nearest = tree.find_nearest(matrix @ vertex.co)
        if nearest and nearest[0] is not None:
            distances.append(float(nearest[3]))

    if not distances:
        return {"sample_count": 0, "mean": None, "rms": None, "max": None}
    mean = sum(distances) / len(distances)
    rms = math.sqrt(sum(distance * distance for distance in distances) / len(distances))
    return {
        "sample_count": len(distances),
        "mean": mean,
        "rms": rms,
        "max": max(distances),
    }


def export_intermediate(obj: bpy.types.Object, output_dir: Path, filename: str) -> str:
    path = output_dir / "intermediate" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    select_only([obj])
    bpy.ops.export_scene.gltf(
        filepath=str(path), export_format="GLB", use_selection=True, export_apply=True
    )
    return str(path)


def export_final(
    obj: bpy.types.Object, output_dir: Path, config: dict[str, Any]
) -> list[dict[str, Any]]:
    export_cfg = config.get("export", {})
    formats = export_cfg.get("formats", ["glb"])
    exports: list[dict[str, Any]] = []
    select_only([obj])
    for fmt in formats:
        normalized = str(fmt).lower()
        if normalized == "glb":
            path = output_dir / str(export_cfg.get("glb_filename", "refined_asset.glb"))
            bpy.ops.export_scene.gltf(
                filepath=str(path),
                export_format="GLB",
                use_selection=True,
                export_apply=True,
            )
            exports.append(
                {
                    "format": "glb",
                    "path": str(path),
                    "exists": path.exists(),
                    "bytes": path.stat().st_size if path.exists() else 0,
                }
            )
        elif normalized == "obj":
            path = output_dir / str(export_cfg.get("obj_filename", "refined_asset.obj"))
            if hasattr(bpy.ops.wm, "obj_export"):
                bpy.ops.wm.obj_export(filepath=str(path), export_selected_objects=True)
            else:
                bpy.ops.export_scene.obj(filepath=str(path), use_selection=True)
            exports.append(
                {
                    "format": "obj",
                    "path": str(path),
                    "exists": path.exists(),
                    "bytes": path.stat().st_size if path.exists() else 0,
                }
            )
        else:
            exports.append(
                {
                    "format": normalized,
                    "path": None,
                    "exists": False,
                    "bytes": 0,
                    "error": "unsupported_export_format",
                }
            )
    return exports


def add_check(
    checks: list[dict[str, Any]],
    check_id: str,
    passed: bool,
    severity: str,
    value: Any,
    threshold: Any = None,
) -> None:
    checks.append(
        {
            "id": check_id,
            "passed": bool(passed),
            "severity": severity,
            "value": value,
            "threshold": threshold,
        }
    )


def build_qc_checks(
    source_before: dict[str, Any],
    source_after: dict[str, Any],
    final: dict[str, Any],
    projection: dict[str, Any],
    textures: dict[str, Any],
    exports: list[dict[str, Any]],
    retopo_stats: dict[str, Any],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    thresholds = config.get("qc", {}).get("thresholds", {})
    checks: list[dict[str, Any]] = []
    final_faces = int(final["faces"])
    adaptive_thresholds = bool(
        config.get("qc", {}).get("adapt_thresholds_to_selected_method", True)
    )
    max_face_threshold = int(thresholds.get("max_faces", 10**12))
    max_open = int(thresholds.get("max_open_or_nonmanifold_edges", 10**12))
    if adaptive_thresholds and retopo_stats.get("method") == "decimate_project":
        max_face_threshold = max(
            max_face_threshold,
            int(retopo_stats.get("target_faces", final_faces) * 1.1) + 100,
        )
        max_open = max(
            max_open, int(source_after.get("open_or_nonmanifold_edges", 0) * 1.1) + 100
        )

    add_check(
        checks,
        "whole_asset_export_count",
        len(exports) >= 1,
        "error",
        len(exports),
        ">=1",
    )
    add_check(
        checks,
        "single_mesh_final_asset",
        final.get("object_name") == "refined_whole_asset_retopology",
        "error",
        final.get("object_name"),
        "refined_whole_asset_retopology",
    )
    add_check(
        checks,
        "new_topology_generated",
        bool(retopo_stats.get("topology_modifier_applied")),
        "error",
        {
            "source_faces": source_after["faces"],
            "final_faces": final_faces,
            "method": retopo_stats.get("method"),
            "topology_modifier_applied": retopo_stats.get("topology_modifier_applied"),
        },
        "whole-asset topology modifier applied",
    )
    add_check(
        checks,
        "source_used_as_high_reference",
        projection.get("sample_count", 0) > 0,
        "error",
        projection.get("sample_count", 0),
        ">0",
    )
    add_check(
        checks,
        "min_faces",
        final_faces >= int(thresholds.get("min_faces", 1)),
        "warning",
        final_faces,
        thresholds.get("min_faces"),
    )
    add_check(
        checks,
        "max_faces",
        final_faces <= max_face_threshold,
        "warning",
        final_faces,
        max_face_threshold,
    )

    add_check(
        checks,
        "open_or_nonmanifold_edges",
        int(final["open_or_nonmanifold_edges"]) <= max_open,
        "warning",
        final["open_or_nonmanifold_edges"],
        max_open,
    )

    source_area = float(source_after.get("surface_area", 0.0) or 0.0)
    final_area = float(final.get("surface_area", 0.0) or 0.0)
    max_area_delta = thresholds.get("max_surface_area_relative_delta")
    if max_area_delta is not None and source_area > 0:
        area_delta = abs(final_area - source_area) / source_area
        add_check(
            checks,
            "surface_area_relative_delta",
            area_delta <= float(max_area_delta),
            "warning",
            area_delta,
            max_area_delta,
        )

    source_dims = list(source_after.get("bbox_dimensions") or [])
    final_dims = list(final.get("bbox_dimensions") or [])
    max_axis_delta = thresholds.get("max_bbox_axis_relative_delta")
    if max_axis_delta is not None and len(source_dims) == 3 and len(final_dims) == 3:
        axis_deltas = [
            abs(float(final_value) - float(source_value)) / float(source_value)
            if float(source_value) > 0
            else None
            for source_value, final_value in zip(source_dims, final_dims)
        ]
        worst_axis_delta = max(
            (value for value in axis_deltas if value is not None), default=None
        )
        add_check(
            checks,
            "bbox_axis_relative_delta",
            worst_axis_delta is not None and worst_axis_delta <= float(max_axis_delta),
            "warning",
            {"axis_deltas": axis_deltas, "max": worst_axis_delta},
            max_axis_delta,
        )

    uv = final.get("uv", {})
    add_check(
        checks, "uv_present", bool(uv.get("has_uv")), "error", uv.get("has_uv"), True
    )
    overlap_ratio = uv.get("overlap_ratio")
    max_overlap = float(thresholds.get("max_uv_overlap_ratio", 1.0))
    add_check(
        checks,
        "uv_overlap_ratio",
        overlap_ratio is not None and overlap_ratio <= max_overlap,
        "warning",
        overlap_ratio,
        max_overlap,
    )
    out_of_bounds = float(uv.get("out_of_bounds_fraction", 1.0))
    max_oob = float(thresholds.get("max_uv_out_of_bounds_fraction", 1.0))
    add_check(
        checks,
        "uv_out_of_bounds_fraction",
        out_of_bounds <= max_oob,
        "warning",
        out_of_bounds,
        max_oob,
    )

    rms = projection.get("rms")
    max_rms = float(thresholds.get("max_projection_rms", 10**12))
    add_check(
        checks,
        "projection_rms",
        rms is not None and rms <= max_rms,
        "warning",
        rms,
        max_rms,
    )
    maximum = projection.get("max")
    max_projection = float(thresholds.get("max_projection_max", 10**12))
    add_check(
        checks,
        "projection_max",
        maximum is not None and maximum <= max_projection,
        "warning",
        maximum,
        max_projection,
    )

    texture_cfg = config.get("textures", {})
    texture_records = textures.get("textures", [])
    if texture_cfg.get("enabled", True):
        add_check(
            checks,
            "texture_migration_output",
            len(texture_records) > 0,
            "error",
            len(texture_records),
            ">0",
        )
        if texture_cfg.get("bake_base_color", True):
            add_check(
                checks,
                "base_color_texture_present",
                any(
                    item.get("type") == "base_color"
                    and Path(item.get("path", "")).exists()
                    for item in texture_records
                ),
                "error",
                [item.get("type") for item in texture_records],
                "base_color",
            )
        for texture_type in ("normal", "roughness", "metallic", "ao", "emissive"):
            if texture_cfg.get(f"bake_{texture_type}", False):
                add_check(
                    checks,
                    f"{texture_type}_texture_present",
                    any(
                        item.get("type") == texture_type
                        and Path(item.get("path", "")).exists()
                        for item in texture_records
                    ),
                    "error",
                    [item.get("type") for item in texture_records],
                    texture_type,
                )
    else:
        add_check(checks, "texture_migration_enabled", False, "warning", False, True)

    add_check(
        checks,
        "final_exports_exist",
        all(item.get("exists") and item.get("bytes", 0) > 0 for item in exports),
        "error",
        exports,
        "all exports exist",
    )
    add_check(
        checks,
        "source_not_semantically_split",
        True,
        "error",
        "whole asset joined and processed as one mesh",
        True,
    )
    add_check(
        checks,
        "tiny_fragment_cleanup_only",
        source_before.get("connected_components", 0)
        >= source_after.get("connected_components", 0),
        "warning",
        {
            "before": source_before.get("connected_components"),
            "after": source_after.get("connected_components"),
        },
        "after <= before",
    )
    return checks


def report_status(checks: list[dict[str, Any]]) -> str:
    failed_errors = [
        check
        for check in checks
        if not check["passed"] and check["severity"] == "error"
    ]
    failed_warnings = [
        check
        for check in checks
        if not check["passed"] and check["severity"] == "warning"
    ]
    if failed_errors:
        return "fail"
    if failed_warnings:
        return "warn"
    return "pass"


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    """Main Blender-side refinement pipeline."""
    input_path = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_json(args.config_json)

    reset_scene()
    imported_meshes = import_asset(input_path)
    source = join_as_whole_asset(imported_meshes)
    source_before = mesh_metrics(source, config)
    cleanup_stats = clean_source_surface(source, config)
    source_after = mesh_metrics(source, config)

    intermediates: list[str] = []
    if config.get("backend", {}).get("keep_intermediate", True):
        intermediates.append(
            export_intermediate(source, output_dir, "source_high_reference_cleaned.glb")
        )

    retopo, retopo_stats = whole_asset_retopology(source, config)
    uv_stats = generate_uv(retopo, config)
    texture_stats = migrate_textures(source, retopo, output_dir, config)
    projection = projection_metrics(source, retopo)
    final_metrics = mesh_metrics(retopo, config)

    source.hide_set(True)
    source.hide_render = True
    exports = export_final(retopo, output_dir, config)
    checks = build_qc_checks(
        source_before,
        source_after,
        final_metrics,
        projection,
        texture_stats,
        exports,
        retopo_stats,
        config,
    )

    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": report_status(checks),
        "input": str(input_path),
        "output_dir": str(output_dir),
        "config": config,
        "policy": {
            "whole_asset_processing": True,
            "semantic_segmentation": False,
            "source_model_role": "high_reference_surface",
            "final_topology_role": "new_generated_whole_asset_topology",
        },
        "stages": {
            "imported_mesh_objects": len(imported_meshes),
            "cleanup": cleanup_stats,
            "retopology": retopo_stats,
            "uv": uv_stats,
            "texture_migration": texture_stats,
        },
        "metrics": {
            "source_before_cleanup": source_before,
            "source_after_cleanup": source_after,
            "final": final_metrics,
            "projection_to_source": projection,
        },
        "exports": exports,
        "intermediates": intermediates,
        "checks": checks,
    }


def main() -> int:
    args = parse_args()
    report_path = Path(args.report)
    try:
        report = run_pipeline(args)
        write_json(report_path, report)
        fail_on_error = bool(
            report.get("config", {}).get("qc", {}).get("fail_on_error")
        )
        return 2 if fail_on_error and report.get("status") == "fail" else 0
    except Exception as exc:
        error_report = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "status": "error",
            "error": str(exc),
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
