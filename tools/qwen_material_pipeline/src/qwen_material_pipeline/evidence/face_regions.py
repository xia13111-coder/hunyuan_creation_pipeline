#!/usr/bin/env python3
"""Build deterministic, read-only face-region evidence for registered USD meshes.

This is deliberately a standalone opt-in tool.  It opens the registry asset for
reading, never authors or saves a USD layer, and writes its output atomically
only after every registered mesh and requested projection has validated.
"""

from __future__ import annotations

import argparse
import colorsys
import hashlib
import json
import math
import shutil
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

if __package__ in {None, ""}:
    package_root = next(
        parent
        for parent in Path(__file__).resolve().parents
        if parent.name == "qwen_material_pipeline"
    )
    sys.path.insert(0, str(package_root.parent))

from qwen_material_pipeline.core.progress import (  # noqa: E402
    ProgressCallback,
    emit_progress_event,
    report_progress,
)


Vector3 = tuple[float, float, float]
PROGRESS_SCOPE = "qwen_material_pipeline"
PROJECTION_PROGRESS_STAGE = "face_regions/views"


@dataclass(frozen=True)
class FaceGeometry:
    vertices: tuple[int, ...]
    area: float
    centroid: Vector3
    normal: Vector3 | None


@dataclass
class ProjectionMesh:
    part_id: str
    points_world: list[Vector3]
    face_vertices: list[tuple[int, ...]]
    face_labels: list[int]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dot(left: Vector3, right: Vector3) -> float:
    return sum(left[index] * right[index] for index in range(3))


def _sub(left: Vector3, right: Vector3) -> Vector3:
    return tuple(left[index] - right[index] for index in range(3))


def _cross(left: Vector3, right: Vector3) -> Vector3:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _length(vector: Vector3) -> float:
    return math.sqrt(_dot(vector, vector))


def _normalize(vector: Vector3) -> Vector3:
    length = _length(vector)
    if length <= 1e-15:
        raise ValueError("Cannot normalize a zero vector")
    return tuple(value / length for value in vector)


def _finite_vector(value: Sequence[Any], *, label: str) -> Vector3:
    if len(value) != 3:
        raise ValueError(f"{label} must contain exactly three numbers")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{label} contains a non-finite value")
    return result


def _inclusive_ranges(indices: Iterable[int]) -> list[list[int]]:
    ordered = sorted(indices)
    if len(ordered) != len(set(ordered)):
        raise ValueError("Face indices must be unique before range compaction")
    if not ordered:
        return []
    ranges: list[list[int]] = []
    start = previous = ordered[0]
    for index in ordered[1:]:
        if index == previous + 1:
            previous = index
            continue
        ranges.append([start, previous])
        start = previous = index
    ranges.append([start, previous])
    return ranges


def _decode_faces(
    point_count: int,
    face_vertex_counts: Sequence[int],
    face_vertex_indices: Sequence[int],
) -> list[tuple[int, ...]]:
    if point_count <= 0:
        raise ValueError("Mesh must contain points")
    counts = [int(value) for value in face_vertex_counts]
    indices = [int(value) for value in face_vertex_indices]
    if any(count < 3 for count in counts):
        raise ValueError("Every face must contain at least three vertices")
    if sum(counts) != len(indices):
        raise ValueError("Face counts do not exactly cover face vertex indices")
    if any(index < 0 or index >= point_count for index in indices):
        raise ValueError("Face vertex index is outside the point array")

    faces = []
    offset = 0
    for count in counts:
        vertices = tuple(indices[offset : offset + count])
        offset += count
        if len(set(vertices)) != len(vertices):
            raise ValueError("A face contains a repeated point index")
        faces.append(vertices)
    return faces


def _polygon_geometry(
    points: Sequence[Vector3], vertices: tuple[int, ...]
) -> FaceGeometry:
    anchor = points[vertices[0]]
    area = 0.0
    normal_sum = (0.0, 0.0, 0.0)
    centroid_sum = (0.0, 0.0, 0.0)
    for offset in range(1, len(vertices) - 1):
        left = points[vertices[offset]]
        right = points[vertices[offset + 1]]
        twice_area_normal = _cross(_sub(left, anchor), _sub(right, anchor))
        triangle_area = _length(twice_area_normal) * 0.5
        if triangle_area <= 1e-18:
            continue
        triangle_centroid = tuple(
            (anchor[index] + left[index] + right[index]) / 3.0 for index in range(3)
        )
        area += triangle_area
        normal_sum = tuple(
            normal_sum[index] + twice_area_normal[index] for index in range(3)
        )
        centroid_sum = tuple(
            centroid_sum[index] + triangle_centroid[index] * triangle_area
            for index in range(3)
        )

    if area <= 1e-18:
        centroid = tuple(
            sum(points[vertex][index] for vertex in vertices) / len(vertices)
            for index in range(3)
        )
        normal = None
    else:
        centroid = tuple(value / area for value in centroid_sum)
        normal = _normalize(normal_sum) if _length(normal_sum) > 1e-15 else None
    return FaceGeometry(vertices=vertices, area=area, centroid=centroid, normal=normal)


def _edge_incidence(
    faces: Sequence[tuple[int, ...]], *, skip_collapsed: bool = False
) -> dict[tuple[int, int], list[int]]:
    incidence: dict[tuple[int, int], list[int]] = {}
    for face_index, vertices in enumerate(faces):
        if len(set(vertices)) < 3:
            if skip_collapsed:
                continue
            raise ValueError("A topology face has fewer than three unique vertices")
        for offset, first in enumerate(vertices):
            second = vertices[(offset + 1) % len(vertices)]
            if first == second:
                if skip_collapsed:
                    continue
                raise ValueError("A face contains a zero-length topology edge")
            edge = (first, second) if first < second else (second, first)
            incidence.setdefault(edge, []).append(face_index)
    return incidence


def _weld_points(
    points: Sequence[Vector3], tolerance: float
) -> tuple[list[int], dict[str, Any]]:
    """Map nearly identical CAD seam vertices to deterministic representatives.

    Every point is compared only with already accepted representatives, so a
    chain of individually close points cannot bridge a real gap larger than the
    tolerance.  Faces that would collapse after welding are rejected by the
    caller; this makes an over-large tolerance fail closed.
    """

    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("Weld tolerance must be finite and positive")
    inverse = 1.0 / tolerance
    bins: dict[tuple[int, int, int], list[int]] = {}
    representatives: list[int] = []
    point_to_welded: list[int] = []
    maximum_distance = 0.0
    for point_index, point in enumerate(points):
        cell = tuple(math.floor(value * inverse) for value in point)
        matches = []
        for x_offset in (-1, 0, 1):
            for y_offset in (-1, 0, 1):
                for z_offset in (-1, 0, 1):
                    candidate_cell = (
                        cell[0] + x_offset,
                        cell[1] + y_offset,
                        cell[2] + z_offset,
                    )
                    for welded_index in bins.get(candidate_cell, []):
                        representative = representatives[welded_index]
                        distance = _length(_sub(point, points[representative]))
                        if distance <= tolerance:
                            matches.append((representative, welded_index, distance))
        if matches:
            _, welded_index, distance = min(matches)
            point_to_welded.append(welded_index)
            maximum_distance = max(maximum_distance, distance)
        else:
            welded_index = len(representatives)
            representatives.append(point_index)
            point_to_welded.append(welded_index)
            bins.setdefault(cell, []).append(welded_index)
    return point_to_welded, {
        "raw_point_count": len(points),
        "welded_point_count": len(representatives),
        "merged_point_count": len(points) - len(representatives),
        "maximum_representative_distance": maximum_distance,
    }


def _components(neighbors: Sequence[set[int]]) -> list[list[int]]:
    unseen = set(range(len(neighbors)))
    components = []
    while unseen:
        seed = min(unseen)
        stack = [seed]
        unseen.remove(seed)
        component = []
        while stack:
            current = stack.pop()
            component.append(current)
            additions = sorted(neighbors[current] & unseen, reverse=True)
            for neighbor in additions:
                unseen.remove(neighbor)
                stack.append(neighbor)
        components.append(sorted(component))
    return sorted(components, key=lambda item: item[0])


def _normal_coherent_components(
    neighbors: Sequence[set[int]],
    normals: Sequence[Vector3 | None],
    cosine_threshold: float,
) -> list[list[int]]:
    """Split smooth adjacency without allowing transitive normal drift.

    A plain connected-component pass can join a chain of small bevels whose
    first and last faces differ by 90 degrees even though every adjacent pair
    is below the crease threshold.  Such a region is too broad for automatic
    face-material recovery.  Each deterministic region therefore keeps the
    seed face as an additional normal anchor.
    """

    if len(neighbors) != len(normals):
        raise ValueError("neighbors and normals must have the same length")
    unseen = set(range(len(neighbors)))
    components: list[list[int]] = []
    while unseen:
        seed = min(unseen)
        seed_normal = normals[seed]
        stack = [seed]
        unseen.remove(seed)
        component: list[int] = []
        while stack:
            current = stack.pop()
            component.append(current)
            additions = []
            for neighbor in sorted(neighbors[current] & unseen, reverse=True):
                neighbor_normal = normals[neighbor]
                if (
                    seed_normal is None
                    or neighbor_normal is None
                    or _dot(seed_normal, neighbor_normal) < cosine_threshold - 1e-12
                ):
                    continue
                additions.append(neighbor)
            for neighbor in additions:
                unseen.remove(neighbor)
                stack.append(neighbor)
        components.append(sorted(component))
    return sorted(components, key=lambda item: item[0])


def _bbox(points: Iterable[Vector3]) -> list[list[float]]:
    values = list(points)
    if not values:
        raise ValueError("Cannot calculate an empty bounding box")
    return [
        [min(point[index] for point in values) for index in range(3)],
        [max(point[index] for point in values) for index in range(3)],
    ]


def _face_reference(indices: list[int], explicit_face_limit: int) -> dict[str, Any]:
    ranges = _inclusive_ranges(indices)
    reference: dict[str, Any] = {
        "face_ranges": ranges,
        "face_range_count": len(ranges),
    }
    if len(indices) <= explicit_face_limit:
        reference["face_indices"] = indices
    return reference


def _region_metrics(
    face_indices: list[int],
    geometries: Sequence[FaceGeometry],
    points: Sequence[Vector3],
    edge_map: dict[tuple[int, int], list[int]],
    edge_faces: Sequence[tuple[int, ...]],
    explicit_face_limit: int,
) -> dict[str, Any]:
    area = sum(geometries[index].area for index in face_indices)
    if area > 1e-18:
        centroid = tuple(
            sum(
                geometries[index].centroid[axis] * geometries[index].area
                for index in face_indices
            )
            / area
            for axis in range(3)
        )
    else:
        centroid = tuple(
            sum(geometries[index].centroid[axis] for index in face_indices)
            / len(face_indices)
            for axis in range(3)
        )

    weighted_normal = tuple(
        sum(
            (geometries[index].normal or (0.0, 0.0, 0.0))[axis] * geometries[index].area
            for index in face_indices
        )
        for axis in range(3)
    )
    mean_normal = (
        _normalize(weighted_normal) if _length(weighted_normal) > 1e-15 else None
    )
    deviations = []
    if mean_normal is not None:
        for index in face_indices:
            normal = geometries[index].normal
            if normal is not None:
                deviations.append(
                    math.degrees(
                        math.acos(max(-1.0, min(1.0, _dot(mean_normal, normal))))
                    )
                )

    face_set = set(face_indices)
    used_vertices = sorted(
        {vertex for index in face_indices for vertex in geometries[index].vertices}
    )
    region_edges = {
        tuple(sorted((first, vertices[(offset + 1) % len(vertices)])))
        for index in face_indices
        for vertices in (edge_faces[index],)
        for offset, first in enumerate(vertices)
        if first != vertices[(offset + 1) % len(vertices)]
        and tuple(sorted((first, vertices[(offset + 1) % len(vertices)]))) in edge_map
    }
    boundary_edges = sum(
        1
        for edge in region_edges
        if set(edge_map[edge]) - face_set or len(edge_map[edge]) != 2
    )
    result = {
        "face_count": len(face_indices),
        **_face_reference(face_indices, explicit_face_limit),
        "point_count": len(used_vertices),
        "area_world": area,
        "centroid_world": list(centroid),
        "bbox_world": _bbox(points[index] for index in used_vertices),
        "mean_normal_world": list(mean_normal) if mean_normal is not None else None,
        "max_normal_deviation_degrees": max(deviations) if deviations else None,
        "degenerate_face_count": sum(
            geometries[index].normal is None for index in face_indices
        ),
        "boundary_edge_count": boundary_edges,
    }
    return result


def analyze_mesh_topology(
    *,
    points_world: Sequence[Sequence[float]],
    face_vertex_counts: Sequence[int],
    face_vertex_indices: Sequence[int],
    crease_angle_degrees: float = 35.0,
    explicit_face_limit: int = 128,
    weld_tolerance_ratio: float = 1e-9,
) -> tuple[dict[str, Any], list[tuple[int, ...]], list[int]]:
    """Analyze one mesh and return evidence plus per-face patch ordinals."""

    if not 0.0 <= crease_angle_degrees < 180.0:
        raise ValueError("crease_angle_degrees must be in [0, 180)")
    if explicit_face_limit < 0:
        raise ValueError("explicit_face_limit cannot be negative")
    if not math.isfinite(weld_tolerance_ratio) or weld_tolerance_ratio <= 0.0:
        raise ValueError("weld_tolerance_ratio must be finite and positive")
    points = [
        _finite_vector(point, label=f"points_world[{index}]")
        for index, point in enumerate(points_world)
    ]
    faces = _decode_faces(len(points), face_vertex_counts, face_vertex_indices)
    if not faces:
        raise ValueError("Mesh must contain faces")
    geometries = [_polygon_geometry(points, vertices) for vertices in faces]
    raw_edges = _edge_incidence(faces)
    bounds = _bbox(points)
    diagonal = _length(tuple(bounds[1][index] - bounds[0][index] for index in range(3)))
    weld_tolerance = max(diagonal * weld_tolerance_ratio, 1e-12)
    point_to_welded, weld_audit = _weld_points(points, weld_tolerance)
    welded_faces = [
        tuple(point_to_welded[vertex] for vertex in vertices) for vertices in faces
    ]
    collapsed_faces = [
        index for index, vertices in enumerate(welded_faces) if len(set(vertices)) < 3
    ]
    if any(geometries[index].normal is not None for index in collapsed_faces):
        raise ValueError(
            "Weld tolerance collapses a nondegenerate face; reduce weld_tolerance_ratio"
        )
    welded_edges = _edge_incidence(welded_faces, skip_collapsed=True)

    raw_topology_neighbors = [set() for _ in faces]
    for incident_faces in raw_edges.values():
        for left_offset, left in enumerate(incident_faces):
            for right in incident_faces[left_offset + 1 :]:
                raw_topology_neighbors[left].add(right)
                raw_topology_neighbors[right].add(left)
    raw_topology_components = _components(raw_topology_neighbors)

    welded_topology_neighbors = [set() for _ in faces]
    for incident_faces in welded_edges.values():
        for left_offset, left in enumerate(incident_faces):
            for right in incident_faces[left_offset + 1 :]:
                welded_topology_neighbors[left].add(right)
                welded_topology_neighbors[right].add(left)
    welded_topology_components = _components(welded_topology_neighbors)

    cosine_threshold = math.cos(math.radians(crease_angle_degrees))
    patch_neighbors = [set() for _ in faces]
    for incident_faces in welded_edges.values():
        if len(incident_faces) != 2:
            continue
        left, right = incident_faces
        left_normal = geometries[left].normal
        right_normal = geometries[right].normal
        if (
            left_normal is not None
            and right_normal is not None
            and _dot(left_normal, right_normal) >= cosine_threshold - 1e-12
        ):
            patch_neighbors[left].add(right)
            patch_neighbors[right].add(left)
    patches = _normal_coherent_components(
        patch_neighbors,
        [geometry.normal for geometry in geometries],
        cosine_threshold,
    )

    component_by_face = {}
    component_records = []
    raw_component_records = []
    for ordinal, component in enumerate(raw_topology_components, start=1):
        raw_component_records.append(
            {
                "component_id": f"RC{ordinal:04d}",
                **_region_metrics(
                    component,
                    geometries,
                    points,
                    raw_edges,
                    faces,
                    explicit_face_limit,
                ),
            }
        )

    for ordinal, component in enumerate(welded_topology_components, start=1):
        component_id = f"C{ordinal:04d}"
        for face_index in component:
            component_by_face[face_index] = component_id
        component_records.append(
            {
                "component_id": component_id,
                **_region_metrics(
                    component,
                    geometries,
                    points,
                    welded_edges,
                    welded_faces,
                    explicit_face_limit,
                ),
            }
        )

    patch_records = []
    patch_by_face = [-1] * len(faces)
    total_area = sum(geometry.area for geometry in geometries)
    for ordinal, patch in enumerate(patches, start=1):
        patch_id = f"R{ordinal:04d}"
        for face_index in patch:
            patch_by_face[face_index] = ordinal - 1
        metrics = _region_metrics(
            patch,
            geometries,
            points,
            welded_edges,
            welded_faces,
            explicit_face_limit,
        )
        metrics["area_fraction_of_part"] = (
            metrics["area_world"] / total_area if total_area > 1e-18 else 0.0
        )
        patch_records.append(
            {
                "region_id": patch_id,
                "candidate_kind": "unclassified_geometric_surface_patch",
                "welded_topology_component_id": component_by_face[patch[0]],
                **metrics,
            }
        )
    if any(value < 0 for value in patch_by_face):
        raise RuntimeError("Internal error: patches did not exactly cover every face")

    welded_points: list[Vector3 | None] = [None] * weld_audit["welded_point_count"]
    for raw_point_index, welded_point_index in enumerate(point_to_welded):
        if welded_points[welded_point_index] is None:
            welded_points[welded_point_index] = points[raw_point_index]
    patch_adjacency: dict[tuple[int, int], dict[str, Any]] = {}
    for edge, incident_faces in welded_edges.items():
        if len(incident_faces) != 2:
            continue
        left_face, right_face = incident_faces
        left_patch = patch_by_face[left_face]
        right_patch = patch_by_face[right_face]
        if left_patch == right_patch:
            continue
        pair = tuple(sorted((left_patch, right_patch)))
        record = patch_adjacency.setdefault(
            pair,
            {
                "shared_edge_count": 0,
                "shared_boundary_length_world": 0.0,
                "dihedral_angles_degrees": [],
            },
        )
        first = welded_points[edge[0]]
        second = welded_points[edge[1]]
        if first is None or second is None:
            raise RuntimeError(
                "Internal error: welded edge has no representative point"
            )
        record["shared_edge_count"] += 1
        record["shared_boundary_length_world"] += _length(_sub(first, second))
        left_normal = geometries[left_face].normal
        right_normal = geometries[right_face].normal
        if left_normal is not None and right_normal is not None:
            record["dihedral_angles_degrees"].append(
                math.degrees(
                    math.acos(max(-1.0, min(1.0, _dot(left_normal, right_normal))))
                )
            )

    for patch_index, record in enumerate(patch_records):
        neighbors = []
        for pair, adjacency in sorted(patch_adjacency.items()):
            if patch_index not in pair:
                continue
            neighbor_index = pair[1] if pair[0] == patch_index else pair[0]
            angles = adjacency["dihedral_angles_degrees"]
            neighbors.append(
                {
                    "region_id": patch_records[neighbor_index]["region_id"],
                    "shared_edge_count": adjacency["shared_edge_count"],
                    "shared_boundary_length_world": adjacency[
                        "shared_boundary_length_world"
                    ],
                    "minimum_dihedral_degrees": min(angles) if angles else None,
                    "maximum_dihedral_degrees": max(angles) if angles else None,
                    "mean_dihedral_degrees": (
                        sum(angles) / len(angles) if angles else None
                    ),
                }
            )
        record["adjacent_patch_count"] = len(neighbors)
        record["adjacent_patches"] = neighbors

    evidence = {
        "point_count": len(points),
        "face_count": len(faces),
        "total_area_world": total_area,
        "bbox_world": bounds,
        "raw_edge_count": len(raw_edges),
        "raw_boundary_edge_count": sum(len(value) == 1 for value in raw_edges.values()),
        "raw_nonmanifold_edge_count": sum(
            len(value) > 2 for value in raw_edges.values()
        ),
        "weld_tolerance_world": weld_tolerance,
        "weld_tolerance_ratio": weld_tolerance_ratio,
        "weld_audit": weld_audit,
        "weld_collapsed_degenerate_face_count": len(collapsed_faces),
        "weld_collapsed_degenerate_face_ranges": _inclusive_ranges(collapsed_faces),
        "welded_edge_count": len(welded_edges),
        "welded_boundary_edge_count": sum(
            len(value) == 1 for value in welded_edges.values()
        ),
        "welded_nonmanifold_edge_count": sum(
            len(value) > 2 for value in welded_edges.values()
        ),
        "degenerate_face_count": sum(
            geometry.normal is None for geometry in geometries
        ),
        "raw_topology_component_count": len(raw_component_records),
        "raw_topology_components": raw_component_records,
        "welded_topology_component_count": len(component_records),
        "welded_topology_components": component_records,
        "crease_angle_degrees": crease_angle_degrees,
        "surface_patch_count": len(patch_records),
        "surface_patch_adjacency_count": len(patch_adjacency),
        "surface_patch_method": "smooth_edge_plus_seed_normal_coherence/v2",
        "surface_patch_semantics": (
            "unclassified geometry candidates; not inferred material regions"
        ),
        "surface_patches": patch_records,
    }
    return evidence, faces, patch_by_face


def _region_colors(region_keys: Sequence[str]) -> dict[int, tuple[int, int, int]]:
    colors: dict[int, tuple[int, int, int]] = {}
    used = set()
    for label, key in enumerate(region_keys, start=1):
        salt = 0
        while True:
            digest = hashlib.sha256(f"{key}:{salt}".encode("utf-8")).digest()
            hue = int.from_bytes(digest[:2], "big") / 65535.0
            saturation = 0.58 + digest[2] / 255.0 * 0.32
            value = 0.72 + digest[3] / 255.0 * 0.25
            rgb_float = colorsys.hsv_to_rgb(hue, saturation, value)
            color = tuple(int(round(channel * 255.0)) for channel in rgb_float)
            if color not in used:
                used.add(color)
                colors[label] = color
                break
            salt += 1
    return colors


def _camera_frame(
    camera_position: Sequence[float],
    world_direction: Sequence[float],
    camera_up_axis: Sequence[float],
    camera_look_at_target: Sequence[float] | None = None,
) -> tuple[Vector3, Vector3, Vector3, Vector3]:
    position = _finite_vector(camera_position, label="camera_position")
    requested_up = _normalize(_finite_vector(camera_up_axis, label="camera_up_axis"))
    if camera_look_at_target is None:
        outward = _normalize(
            _finite_vector(world_direction, label="world_direction")
        )
        forward = tuple(-value for value in outward)
    else:
        target = _finite_vector(
            camera_look_at_target, label="camera_look_at_target"
        )
        forward = _normalize(
            tuple(target[index] - position[index] for index in range(3))
        )
    right = _normalize(_cross(forward, requested_up))
    up = _normalize(_cross(right, forward))
    return position, right, up, forward


def _rasterize_region_labels(
    meshes: Sequence[ProjectionMesh],
    *,
    camera_position: Sequence[float],
    world_direction: Sequence[float],
    camera_up_axis: Sequence[float],
    width: int,
    height: int,
    focal_length_mm: float,
    horizontal_aperture_mm: float,
    projection_mode: str = "perspective",
    orthographic_span_multiplier: float = 2.0,
    asset_diagonal: float | None = None,
    camera_look_at_target: Sequence[float] | None = None,
) -> Any:
    import numpy as np

    if width < 16 or height < 16:
        raise ValueError("Projection dimensions must be at least 16 pixels")
    if projection_mode not in {"perspective", "orthographic"}:
        raise ValueError(f"Unsupported camera projection mode: {projection_mode}")
    if focal_length_mm <= 0.0 or horizontal_aperture_mm <= 0.0:
        raise ValueError("Camera focal length and aperture must be positive")
    if projection_mode == "orthographic" and (
        isinstance(asset_diagonal, bool)
        or not isinstance(asset_diagonal, (int, float))
        or not math.isfinite(float(asset_diagonal))
        or float(asset_diagonal) <= 0.0
        or isinstance(orthographic_span_multiplier, bool)
        or not isinstance(orthographic_span_multiplier, (int, float))
        or not math.isfinite(float(orthographic_span_multiplier))
        or float(orthographic_span_multiplier) <= 0.0
    ):
        raise ValueError(
            "Orthographic projection requires a positive asset diagonal and span"
        )
    position, right, up, forward = _camera_frame(
        camera_position,
        world_direction,
        camera_up_axis,
        camera_look_at_target,
    )
    pixels_per_camera_unit = (
        focal_length_mm / horizontal_aperture_mm * width
        if projection_mode == "perspective"
        else width
        / (float(asset_diagonal) * float(orthographic_span_multiplier))
    )
    depth_buffer = np.full((height, width), np.inf, dtype=np.float64)
    labels = np.zeros((height, width), dtype=np.int32)

    for mesh in meshes:
        if len(mesh.face_vertices) != len(mesh.face_labels):
            raise ValueError(
                f"Projection labels do not cover all faces of {mesh.part_id}"
            )
        points = np.asarray(mesh.points_world, dtype=np.float64)
        delta = points - np.asarray(position, dtype=np.float64)
        camera_x = delta @ np.asarray(right)
        camera_y = delta @ np.asarray(up)
        camera_z = delta @ np.asarray(forward)
        if projection_mode == "perspective":
            screen_x = width * 0.5 + camera_x / camera_z * pixels_per_camera_unit
            screen_y = height * 0.5 - camera_y / camera_z * pixels_per_camera_unit
        else:
            screen_x = width * 0.5 + camera_x * pixels_per_camera_unit
            screen_y = height * 0.5 - camera_y * pixels_per_camera_unit

        for face_index, vertices in enumerate(mesh.face_vertices):
            label = int(mesh.face_labels[face_index])
            if label <= 0:
                raise ValueError("Projection region labels must be positive")
            anchor = vertices[0]
            for offset in range(1, len(vertices) - 1):
                triangle = (anchor, vertices[offset], vertices[offset + 1])
                z_values = camera_z[list(triangle)]
                if np.any(z_values <= 1e-8):
                    continue
                x_values = screen_x[list(triangle)]
                y_values = screen_y[list(triangle)]
                min_x = max(0, int(math.floor(float(np.min(x_values)))))
                max_x = min(width - 1, int(math.ceil(float(np.max(x_values)))))
                min_y = max(0, int(math.floor(float(np.min(y_values)))))
                max_y = min(height - 1, int(math.ceil(float(np.max(y_values)))))
                if min_x > max_x or min_y > max_y:
                    continue

                x0, x1, x2 = (float(value) for value in x_values)
                y0, y1, y2 = (float(value) for value in y_values)
                denominator = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
                if abs(denominator) <= 1e-12:
                    continue
                pixel_y, pixel_x = np.mgrid[min_y : max_y + 1, min_x : max_x + 1]
                sample_x = pixel_x + 0.5
                sample_y = pixel_y + 0.5
                weight0 = (
                    (y1 - y2) * (sample_x - x2) + (x2 - x1) * (sample_y - y2)
                ) / denominator
                weight1 = (
                    (y2 - y0) * (sample_x - x2) + (x0 - x2) * (sample_y - y2)
                ) / denominator
                weight2 = 1.0 - weight0 - weight1
                inside = (weight0 >= -1e-9) & (weight1 >= -1e-9) & (weight2 >= -1e-9)
                if projection_mode == "perspective":
                    inverse_depth = (
                        weight0 / z_values[0]
                        + weight1 / z_values[1]
                        + weight2 / z_values[2]
                    )
                    valid = inside & (inverse_depth > 0.0)
                    candidate_depth = np.where(valid, 1.0 / inverse_depth, np.inf)
                else:
                    linear_depth = (
                        weight0 * z_values[0]
                        + weight1 * z_values[1]
                        + weight2 * z_values[2]
                    )
                    valid = inside & (linear_depth > 0.0)
                    candidate_depth = np.where(valid, linear_depth, np.inf)
                depth_slice = depth_buffer[min_y : max_y + 1, min_x : max_x + 1]
                update = candidate_depth < depth_slice
                if not np.any(update):
                    continue
                depth_slice[update] = candidate_depth[update]
                label_slice = labels[min_y : max_y + 1, min_x : max_x + 1]
                label_slice[update] = label
    return labels


def _write_projection_images(
    *,
    labels: Any,
    rgb_source: Path,
    colors: dict[int, tuple[int, int, int]],
    label_metadata: dict[int, dict[str, str]],
    destination: Path,
    view_id: str,
    semantic_visible_parts: Sequence[dict[str, Any]],
    semantic_pixel_scale: float,
) -> dict[str, Any]:
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    height, width = labels.shape
    rgb = (
        Image.open(rgb_source)
        .convert("RGB")
        .resize((width, height), Image.Resampling.LANCZOS)
    )
    rgb_pixels = np.asarray(rgb, dtype=np.uint8)
    color_pixels = np.zeros((height, width, 3), dtype=np.uint8)
    color_pixels[:] = (22, 24, 28)
    visible_records = []
    observed_pixels_by_part: dict[str, int] = {}
    for label in sorted(int(value) for value in np.unique(labels) if int(value) > 0):
        mask = labels == label
        color_pixels[mask] = colors[label]
        ys, xs = np.nonzero(mask)
        metadata = label_metadata[label]
        observed_pixels_by_part[metadata["part_id"]] = observed_pixels_by_part.get(
            metadata["part_id"], 0
        ) + int(mask.sum())
        visible_records.append(
            {
                "numeric_region_id": label,
                **metadata,
                "color_rgb": list(colors[label]),
                "visible_pixels": int(mask.sum()),
                "image_bbox_xyxy": [
                    int(xs.min()),
                    int(ys.min()),
                    int(xs.max()),
                    int(ys.max()),
                ],
                "image_centroid_xy": [float(xs.mean()), float(ys.mean())],
            }
        )

    mask = labels > 0
    expected_pixels_by_part = {}
    for record in semantic_visible_parts:
        if (
            not isinstance(record, dict)
            or not isinstance(record.get("part_id"), str)
            or not isinstance(record.get("pixels"), int)
            or record["pixels"] < 0
        ):
            raise ValueError(f"Invalid semantic visible_parts record for {view_id}")
        expected_pixels_by_part[record["part_id"]] = (
            record["pixels"] * semantic_pixel_scale
        )
    part_comparisons = []
    for part_id in sorted(set(expected_pixels_by_part) | set(observed_pixels_by_part)):
        expected = expected_pixels_by_part.get(part_id, 0.0)
        observed = observed_pixels_by_part.get(part_id, 0)
        part_comparisons.append(
            {
                "part_id": part_id,
                "semantic_pixels_scaled": expected,
                "cpu_projection_pixels": observed,
                "absolute_error_pixels": abs(observed - expected),
                "relative_error": abs(observed - expected) / max(expected, 1.0),
            }
        )
    expected_total = sum(expected_pixels_by_part.values())
    observed_total = int(mask.sum())
    total_relative_error = abs(observed_total - expected_total) / max(
        expected_total, 1.0
    )
    alignment = {
        "status": (
            "approximate_coverage_match"
            if total_relative_error <= 0.15
            else "diagnostic_only_coverage_mismatch"
        ),
        "semantic_pixels_scaled_total": expected_total,
        "cpu_projection_pixels_total": observed_total,
        "total_coverage_relative_error": total_relative_error,
        "part_comparisons": part_comparisons,
    }
    overlay_pixels = np.clip(rgb_pixels.astype(float) * 0.38, 0, 255).astype(np.uint8)
    overlay_pixels[mask] = np.clip(
        rgb_pixels[mask].astype(float) * 0.38 + color_pixels[mask].astype(float) * 0.62,
        0,
        255,
    ).astype(np.uint8)
    overlay = Image.fromarray(overlay_pixels, mode="RGB")
    draw = ImageDraw.Draw(overlay)
    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    font = (
        ImageFont.truetype(str(font_path), max(11, width // 64))
        if font_path.exists()
        else ImageFont.load_default()
    )
    placed_label_boxes: list[tuple[int, int, int, int]] = []
    labels_per_part: dict[str, int] = {}
    for record in sorted(
        visible_records, key=lambda item: item["visible_pixels"], reverse=True
    ):
        if len(placed_label_boxes) >= 18:
            break
        if record["visible_pixels"] < max(80, width * height // 5000):
            continue
        if labels_per_part.get(record["part_id"], 0) >= 2:
            continue
        x, y = record["image_centroid_xy"]
        text = f"{record['part_id']}:{record['region_id']}"
        box = draw.textbbox((x, y), text, font=font, anchor="mm")
        padded_box = (box[0] - 3, box[1] - 2, box[2] + 3, box[3] + 2)
        if any(
            padded_box[0] <= existing[2]
            and padded_box[2] >= existing[0]
            and padded_box[1] <= existing[3]
            and padded_box[3] >= existing[1]
            for existing in placed_label_boxes
        ):
            continue
        draw.rectangle(box, fill=(0, 0, 0))
        draw.text((x, y), text, font=font, fill=(255, 255, 255), anchor="mm")
        placed_label_boxes.append(padded_box)
        labels_per_part[record["part_id"]] = (
            labels_per_part.get(record["part_id"], 0) + 1
        )

    ids_dir = destination / "region_ids"
    overlays_dir = destination / "region_overlays"
    labels_dir = destination / "region_labels"
    projections_dir = destination / "projections"
    for directory in (ids_dir, overlays_dir, labels_dir, projections_dir):
        directory.mkdir(parents=True, exist_ok=True)
    Image.fromarray(color_pixels, mode="RGB").save(ids_dir / f"{view_id}.png")
    overlay.save(overlays_dir / f"{view_id}.png")
    np.save(labels_dir / f"{view_id}.npy", labels, allow_pickle=False)

    projection = {
        "view_id": view_id,
        "resolution": [width, height],
        "rgb_source": str(rgb_source),
        "visible_region_count": len(visible_records),
        "covered_pixel_count": int(mask.sum()),
        "semantic_alignment": alignment,
        "regions": visible_records,
    }
    (projections_dir / f"{view_id}.json").write_text(
        json.dumps(projection, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "view_id": view_id,
        "region_ids": f"region_ids/{view_id}.png",
        "region_overlay": f"region_overlays/{view_id}.png",
        "numeric_labels": f"region_labels/{view_id}.npy",
        "projection": f"projections/{view_id}.json",
        "visible_region_count": len(visible_records),
        "covered_pixel_count": int(mask.sum()),
        "semantic_alignment_status": alignment["status"],
        "total_coverage_relative_error": alignment["total_coverage_relative_error"],
    }


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return document


def _load_registered_meshes(
    *,
    registry_path: Path,
    crease_angle_degrees: float,
    explicit_face_limit: int,
    weld_tolerance_ratio: float,
    progress_callback: ProgressCallback | None = None,
) -> tuple[dict[str, Any], list[ProjectionMesh], Path, str, Any]:
    from pxr import Gf, Usd, UsdGeom

    registry = _load_json(registry_path, label="Registry")
    parts = registry.get("parts")
    if not isinstance(parts, list) or not parts:
        raise ValueError("Registry must contain a non-empty parts array")
    if registry.get("part_count") != len(parts):
        raise ValueError("Registry part_count does not match parts")
    asset_path = (
        Path(str(registry.get("asset_usd", ""))).expanduser().resolve(strict=True)
    )
    asset_sha256 = _sha256(asset_path)
    if registry.get("asset_sha256") != asset_sha256:
        raise ValueError("Registry asset_sha256 does not match the USD file")
    stage = Usd.Stage.Open(str(asset_path))
    if stage is None:
        raise ValueError(f"Unable to open USD stage: {asset_path}")
    registry_ids = [item.get("part_id") for item in parts if isinstance(item, dict)]
    registry_paths = [item.get("prim_path") for item in parts if isinstance(item, dict)]
    if len(registry_ids) != len(parts) or len(set(registry_ids)) != len(parts):
        raise ValueError("Registry part IDs must be unique strings")
    if len(registry_paths) != len(parts) or len(set(registry_paths)) != len(parts):
        raise ValueError("Registry prim paths must be unique strings")
    stage_mesh_paths = {
        str(prim.GetPath()) for prim in stage.Traverse() if prim.IsA(UsdGeom.Mesh)
    }
    if set(registry_paths) != stage_mesh_paths:
        missing = sorted(stage_mesh_paths - set(registry_paths))
        extra = sorted(set(registry_paths) - stage_mesh_paths)
        raise ValueError(
            f"Registry does not exactly cover stage Mesh prims; missing={missing}, extra={extra}"
        )

    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    part_records = []
    projection_meshes = []
    global_label = 1
    label_metadata: dict[int, dict[str, str]] = {}
    ordered_parts = sorted(parts, key=lambda item: item["part_id"])
    part_total = len(ordered_parts)
    report_progress(
        progress_callback,
        scope=PROGRESS_SCOPE,
        stage="face_topology",
        state="start",
        current=0,
        total=part_total,
        unit="parts",
        detail="face topology analysis started",
    )
    for part_index, part in enumerate(ordered_parts, start=1):
        part_id = part["part_id"]
        prim_path = part["prim_path"]
        mesh = UsdGeom.Mesh(stage.GetPrimAtPath(prim_path))
        points_raw = mesh.GetPointsAttr().Get(Usd.TimeCode.Default())
        counts = mesh.GetFaceVertexCountsAttr().Get(Usd.TimeCode.Default())
        indices = mesh.GetFaceVertexIndicesAttr().Get(Usd.TimeCode.Default())
        holes = mesh.GetHoleIndicesAttr().Get(Usd.TimeCode.Default()) or []
        if points_raw is None or counts is None or indices is None:
            raise ValueError(f"{part_id} is missing required Mesh topology attributes")
        if holes:
            raise ValueError(
                f"{part_id} contains hole faces, which are not silently triangulated"
            )
        if len(points_raw) != part.get("point_count") or len(counts) != part.get(
            "face_count"
        ):
            raise ValueError(f"{part_id} registry geometry counts do not match USD")
        transform = xform_cache.GetLocalToWorldTransform(mesh.GetPrim())
        points_world = []
        for point in points_raw:
            transformed = transform.Transform(
                Gf.Vec3d(float(point[0]), float(point[1]), float(point[2]))
            )
            points_world.append(
                (float(transformed[0]), float(transformed[1]), float(transformed[2]))
            )

        try:
            evidence, faces, patch_by_face = analyze_mesh_topology(
                points_world=points_world,
                face_vertex_counts=counts,
                face_vertex_indices=indices,
                crease_angle_degrees=crease_angle_degrees,
                explicit_face_limit=explicit_face_limit,
                weld_tolerance_ratio=weld_tolerance_ratio,
            )
        except (ValueError, RuntimeError) as exc:
            raise type(exc)(f"{part_id}: {exc}") from exc
        evidence["part_id"] = part_id
        evidence["prim_path"] = prim_path
        face_labels = []
        for patch_index, patch in enumerate(evidence["surface_patches"]):
            label = global_label
            global_label += 1
            patch["numeric_region_id"] = label
            label_metadata[label] = {
                "part_id": part_id,
                "region_id": patch["region_id"],
            }
        for patch_index in patch_by_face:
            face_labels.append(
                evidence["surface_patches"][patch_index]["numeric_region_id"]
            )
        part_records.append(evidence)
        projection_meshes.append(
            ProjectionMesh(
                part_id=part_id,
                points_world=points_world,
                face_vertices=faces,
                face_labels=face_labels,
            )
        )
        report_progress(
            progress_callback,
            scope=PROGRESS_SCOPE,
            stage="face_topology",
            state="update",
            current=part_index,
            total=part_total,
            unit="parts",
            detail=f"face topology part {part_id} completed",
        )
    if stage.GetRootLayer().dirty:
        raise RuntimeError("USD root layer became dirty during read-only analysis")
    report_progress(
        progress_callback,
        scope=PROGRESS_SCOPE,
        stage="face_topology",
        state="complete",
        current=part_total,
        total=part_total,
        unit="parts",
        detail="face topology analysis completed",
    )
    metadata = {
        "label_metadata": label_metadata,
        "meters_per_unit": float(UsdGeom.GetStageMetersPerUnit(stage)),
    }
    manifest = {
        "schema_version": "qwen-face-region-evidence/v1",
        "asset_usd": str(asset_path),
        "asset_sha256": asset_sha256,
        "registry": str(registry_path),
        "registry_sha256": _sha256(registry_path),
        "part_count": len(part_records),
        "crease_angle_degrees": crease_angle_degrees,
        "explicit_face_limit": explicit_face_limit,
        "weld_tolerance_ratio": weld_tolerance_ratio,
        "meters_per_unit": metadata["meters_per_unit"],
        "welded_topology_component_count": sum(
            part["welded_topology_component_count"] for part in part_records
        ),
        "surface_patch_count": sum(
            part["surface_patch_count"] for part in part_records
        ),
        "surface_patch_method": "smooth_edge_plus_seed_normal_coherence/v2",
        "face_count": sum(part["face_count"] for part in part_records),
        "parts": part_records,
    }
    return manifest, projection_meshes, asset_path, asset_sha256, metadata


def _load_projection_views(
    *,
    rendered_registry_path: Path,
    asset_path: Path,
    requested_views: list[str] | None,
    projection_max_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from PIL import Image

    document = _load_json(rendered_registry_path, label="Rendered registry")
    render_set = document.get("render_set")
    if not isinstance(render_set, dict) or not isinstance(
        render_set.get("views"), list
    ):
        raise ValueError("Rendered registry has no render_set.views")
    rendered_asset = (
        Path(str(render_set.get("asset_usd", ""))).expanduser().resolve(strict=True)
    )
    if rendered_asset != asset_path:
        raise ValueError(
            "Rendered registry asset does not match the topology registry asset"
        )
    resolution = render_set.get("resolution")
    if (
        not isinstance(resolution, list)
        or len(resolution) != 2
        or not all(isinstance(value, int) and value > 0 for value in resolution)
    ):
        raise ValueError("Rendered registry resolution is invalid")
    source_width, source_height = resolution
    scale = min(1.0, projection_max_size / max(source_width, source_height))
    width = max(16, int(round(source_width * scale)))
    height = max(16, int(round(source_height * scale)))

    records_by_id = {}
    for record in render_set["views"]:
        if not isinstance(record, dict) or not isinstance(record.get("view_id"), str):
            raise ValueError("Rendered registry contains an invalid view record")
        view_id = record["view_id"]
        if view_id in records_by_id:
            raise ValueError(f"Rendered registry repeats view {view_id}")
        rgb_path = Path(str(record.get("rgb", ""))).expanduser().resolve(strict=True)
        with Image.open(rgb_path) as image:
            if image.size != (source_width, source_height):
                raise ValueError(
                    f"RGB dimensions do not match render_set for {view_id}"
                )
        _camera_frame(
            record.get("camera_position", []),
            record.get("world_direction", []),
            record.get("camera_up_axis", []),
        )
        records_by_id[view_id] = {
            **record,
            "rgb_path": rgb_path,
            "projection_resolution": (width, height),
        }
    selected = requested_views or sorted(records_by_id)
    if not selected:
        raise ValueError("At least one projection view is required")
    if len(selected) != len(set(selected)):
        raise ValueError("Projection view IDs must be unique")
    unknown = sorted(set(selected) - set(records_by_id))
    if unknown:
        raise ValueError(
            f"Requested views are absent from rendered registry: {unknown}"
        )
    return [records_by_id[view_id] for view_id in selected], {
        "rendered_registry": str(rendered_registry_path),
        "rendered_registry_sha256": _sha256(rendered_registry_path),
        "source_resolution": [source_width, source_height],
        "projection_resolution": [width, height],
    }


def _projection_asset_diagonal(meshes: Sequence[ProjectionMesh]) -> float:
    points = [point for mesh in meshes for point in mesh.points_world]
    if not points:
        raise ValueError("Projection meshes contain no world-space points")
    minimum = [min(point[index] for point in points) for index in range(3)]
    maximum = [max(point[index] for point in points) for index in range(3)]
    diagonal = math.sqrt(
        sum((maximum[index] - minimum[index]) ** 2 for index in range(3))
    )
    if not math.isfinite(diagonal) or diagonal <= 0.0:
        raise ValueError("Projection asset bounds have a non-positive diagonal")
    return diagonal


def build_face_region_evidence(
    *,
    registry_path: str | Path,
    output_dir: str | Path,
    rendered_registry_path: str | Path | None = None,
    view_names: list[str] | None = None,
    crease_angle_degrees: float = 35.0,
    explicit_face_limit: int = 128,
    weld_tolerance_ratio: float = 1e-9,
    projection_max_size: int = 512,
    focal_length_mm: float = 45.0,
    horizontal_aperture_mm: float = 20.955,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    registry_file = Path(registry_path).expanduser().resolve(strict=True)
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"Output directory already exists: {destination}")
    if projection_max_size < 16:
        raise ValueError("projection_max_size must be at least 16")
    temporary = destination.parent / f".{destination.name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir(parents=True, exist_ok=False)
    try:
        manifest, meshes, asset_path, asset_sha256_before, metadata = (
            _load_registered_meshes(
                registry_path=registry_file,
                crease_angle_degrees=crease_angle_degrees,
                explicit_face_limit=explicit_face_limit,
                weld_tolerance_ratio=weld_tolerance_ratio,
                progress_callback=progress_callback,
            )
        )
        parts_dir = temporary / "parts"
        parts_dir.mkdir()
        part_summaries = []
        for part in manifest.pop("parts"):
            part_path = parts_dir / f"{part['part_id']}.json"
            part_path.write_text(
                json.dumps(part, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            part_summaries.append(
                {
                    "part_id": part["part_id"],
                    "prim_path": part["prim_path"],
                    "face_count": part["face_count"],
                    "raw_topology_component_count": part[
                        "raw_topology_component_count"
                    ],
                    "welded_topology_component_count": part[
                        "welded_topology_component_count"
                    ],
                    "surface_patch_count": part["surface_patch_count"],
                    "evidence": f"parts/{part['part_id']}.json",
                }
            )
        manifest["parts"] = part_summaries

        projection_records = []
        if rendered_registry_path is not None:
            rendered_file = (
                Path(rendered_registry_path).expanduser().resolve(strict=True)
            )
            views, projection_source = _load_projection_views(
                rendered_registry_path=rendered_file,
                asset_path=asset_path,
                requested_views=view_names,
                projection_max_size=projection_max_size,
            )
            asset_diagonal = _projection_asset_diagonal(meshes)
            region_keys = [
                f"{metadata['label_metadata'][label]['part_id']}:{metadata['label_metadata'][label]['region_id']}"
                for label in sorted(metadata["label_metadata"])
            ]
            colors = _region_colors(region_keys)
            projection_total = len(views)
            report_progress(
                progress_callback,
                scope=PROGRESS_SCOPE,
                stage=PROJECTION_PROGRESS_STAGE,
                state="start",
                current=0,
                total=projection_total,
                unit="views",
                detail="face-region projection started",
            )
            for projection_index, view in enumerate(views, start=1):
                width, height = view["projection_resolution"]
                raw_view_focal_length = view.get(
                    "focal_length_mm", focal_length_mm
                )
                if (
                    isinstance(raw_view_focal_length, bool)
                    or not isinstance(raw_view_focal_length, (int, float))
                    or not math.isfinite(float(raw_view_focal_length))
                    or float(raw_view_focal_length) <= 0.0
                ):
                    raise ValueError(
                        f"Projection view {view['view_id']!r} has an invalid "
                        "focal_length_mm"
                    )
                view_focal_length = float(raw_view_focal_length)
                projection_mode = str(
                    view.get("camera_projection_mode", "perspective")
                )
                raw_span = view.get("camera_orthographic_span_multiplier", 2.0)
                if (
                    isinstance(raw_span, bool)
                    or not isinstance(raw_span, (int, float))
                    or not math.isfinite(float(raw_span))
                    or float(raw_span) <= 0.0
                ):
                    raise ValueError(
                        f"Projection view {view['view_id']!r} has an invalid "
                        "camera_orthographic_span_multiplier"
                    )
                orthographic_span_multiplier = float(raw_span)
                labels = _rasterize_region_labels(
                    meshes,
                    camera_position=view["camera_position"],
                    world_direction=view["world_direction"],
                    camera_up_axis=view["camera_up_axis"],
                    width=width,
                    height=height,
                    focal_length_mm=view_focal_length,
                    horizontal_aperture_mm=horizontal_aperture_mm,
                    projection_mode=projection_mode,
                    orthographic_span_multiplier=orthographic_span_multiplier,
                    asset_diagonal=asset_diagonal,
                    camera_look_at_target=view.get("camera_look_at_target"),
                )
                projection_record = _write_projection_images(
                    labels=labels,
                    rgb_source=view["rgb_path"],
                    colors=colors,
                    label_metadata=metadata["label_metadata"],
                    destination=temporary,
                    view_id=view["view_id"],
                    semantic_visible_parts=view.get("visible_parts", []),
                    semantic_pixel_scale=(
                        width
                        * height
                        / (
                            projection_source["source_resolution"][0]
                            * projection_source["source_resolution"][1]
                        )
                    ),
                )
                projection_records.append(
                    {
                        **projection_record,
                        "focal_length_mm": view_focal_length,
                        "projection_mode": projection_mode,
                        "orthographic_span_multiplier": (
                            orthographic_span_multiplier
                        ),
                    }
                )
                report_progress(
                    progress_callback,
                    scope=PROGRESS_SCOPE,
                    stage=PROJECTION_PROGRESS_STAGE,
                    state="update",
                    current=projection_index,
                    total=projection_total,
                    unit="views",
                    detail=f"face-region projection {view['view_id']} completed",
                )
            report_progress(
                progress_callback,
                scope=PROGRESS_SCOPE,
                stage=PROJECTION_PROGRESS_STAGE,
                state="complete",
                current=projection_total,
                total=projection_total,
                unit="views",
                detail="face-region projection completed",
            )
            manifest["projection_contract"] = {
                **projection_source,
                "method": (
                    "CPU USD perspective-or-orthographic triangulation "
                    "with global z-buffer"
                ),
                "focal_length_source": (
                    "render_set.views[*].focal_length_mm_with_cli_fallback"
                ),
                "focal_length_mm": focal_length_mm,
                "horizontal_aperture_mm": horizontal_aperture_mm,
                "asset_diagonal": asset_diagonal,
                "views": projection_records,
                "limitations": [
                    "Uses the recorded camera frame and USD projection mode; it is not an RTX G-buffer.",
                    "Polygons are deterministically fan-triangulated and lens distortion is not modeled.",
                    "Region IDs locate candidate surfaces; they are not material classifications.",
                ],
            }
        elif view_names:
            raise ValueError("view_names require rendered_registry_path")

        asset_sha256_after = _sha256(asset_path)
        if asset_sha256_after != asset_sha256_before:
            raise RuntimeError(
                "Source USD changed during face-region evidence generation"
            )
        manifest["source_usd_sha256_before"] = asset_sha256_before
        manifest["source_usd_sha256_after"] = asset_sha256_after
        manifest["source_usd_unchanged"] = True
        manifest["projection_view_count"] = len(projection_records)
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.rename(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {
        "output_dir": str(destination),
        "manifest": str(destination / "manifest.json"),
        "part_count": manifest["part_count"],
        "face_count": manifest["face_count"],
        "welded_topology_component_count": manifest["welded_topology_component_count"],
        "surface_patch_count": manifest["surface_patch_count"],
        "projection_view_count": manifest["projection_view_count"],
        "source_usd_unchanged": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate deterministic read-only USD face-region evidence"
    )
    parser.add_argument("--registry", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--rendered-registry",
        help="Optional rendered registry whose recorded cameras drive CPU region projections",
    )
    parser.add_argument(
        "--views",
        help="Optional comma-separated subset of rendered-registry view IDs",
    )
    parser.add_argument("--crease-angle-degrees", type=float, default=35.0)
    parser.add_argument("--explicit-face-limit", type=int, default=128)
    parser.add_argument("--weld-tolerance-ratio", type=float, default=1e-9)
    parser.add_argument("--projection-max-size", type=int, default=512)
    parser.add_argument("--focal-length-mm", type=float, default=45.0)
    parser.add_argument("--horizontal-aperture-mm", type=float, default=20.955)
    return parser.parse_args()


def _start_isaac_if_needed() -> Any:
    """Initialize Isaac Sim 5 before importing USD Python bindings."""

    try:
        from pxr import Usd  # noqa: F401

        return None
    except ImportError:
        try:
            from isaacsim import SimulationApp
        except ImportError as exc:
            raise RuntimeError(
                "pxr is unavailable. Run this tool with Isaac Sim python.sh."
            ) from exc
        app = SimulationApp({"headless": True})
        try:
            from pxr import Usd  # noqa: F401
        except ImportError:
            app.close()
            raise
        return app


def main() -> int:
    args = parse_args()
    app = _start_isaac_if_needed()
    try:
        report = build_face_region_evidence(
            registry_path=args.registry,
            output_dir=args.output_dir,
            rendered_registry_path=args.rendered_registry,
            view_names=(
                [value.strip() for value in args.views.split(",") if value.strip()]
                if args.views
                else None
            ),
            crease_angle_degrees=args.crease_angle_degrees,
            explicit_face_limit=args.explicit_face_limit,
            weld_tolerance_ratio=args.weld_tolerance_ratio,
            projection_max_size=args.projection_max_size,
            focal_length_mm=args.focal_length_mm,
            horizontal_aperture_mm=args.horizontal_aperture_mm,
            progress_callback=emit_progress_event,
        )
    finally:
        if app is not None:
            app.close()
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
