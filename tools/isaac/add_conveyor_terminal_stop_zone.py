#!/usr/bin/env python3
"""Split the right conveyor end into graded slowdown zones and a final stop."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade


DEFAULT_CONVEYOR = "/tn__8U_Z2odv1qbE/Conveyor_brep_36"
DEFAULT_MESH = f"{DEFAULT_CONVEYOR}/CollisionMesh"
DEFAULT_CURVE = f"{DEFAULT_CONVEYOR}/MotionPath"
DEFAULT_RIGHT = "/tn__8U_Z2odv1qbE/ConveyorRightStraight_brep_36"
DEFAULT_DECEL_PREFIX = "/tn__8U_Z2odv1qbE/ConveyorDecelZone_brep_36"
DEFAULT_STOP = "/tn__8U_Z2odv1qbE/ConveyorTerminalZeroSpeed_brep_36"
DEFAULT_MATERIAL = "/tn__8U_Z2odv1qbE/PhysicsMaterials/ConveyorHighFriction"
MACHINE_BODY = "/tn__8U_Z2odv1qbE/tn__8U_Z2odv1qbE"
DECEL_SPEEDS = tuple(round(speed / 100.0, 2) for speed in range(19, -1, -1))


def _face_records(counts, indices):
    cursor = 0
    for count in counts:
        count = int(count)
        yield count, indices[cursor : cursor + count]
        cursor += count
    if cursor != len(indices):
        raise ValueError("faceVertexCounts and faceVertexIndices do not agree")


def _centroid(points, face):
    count = float(len(face))
    return Gf.Vec3d(
        sum(float(points[i][0]) for i in face) / count,
        sum(float(points[i][1]) for i in face) / count,
        sum(float(points[i][2]) for i in face) / count,
    )


def _clip_at_y(polygon, boundary, keep_below):
    """Clip a polygon against one side of a Y-aligned plane."""
    if not polygon:
        return []

    def inside(point):
        value = float(point[1]) - boundary
        return value <= 1.0e-8 if keep_below else value >= -1.0e-8

    clipped = []
    previous = polygon[-1]
    previous_inside = inside(previous)
    for current in polygon:
        current_inside = inside(current)
        if current_inside != previous_inside:
            denominator = float(current[1] - previous[1])
            if abs(denominator) > 1.0e-12:
                t = (boundary - float(previous[1])) / denominator
                clipped.append(previous + (current - previous) * t)
        if current_inside:
            clipped.append(current)
        previous = current
        previous_inside = current_inside
    return clipped


def _append_polygon(points_out, counts_out, indices_out, polygon, transform):
    """Triangulate a polygon fan and append transformed, unwelded vertices."""
    if len(polygon) < 3:
        return 0
    triangles = 0
    for index in range(1, len(polygon) - 1):
        for point in (polygon[0], polygon[index], polygon[index + 1]):
            indices_out.append(len(points_out))
            points_out.append(Gf.Vec3f(transform.Transform(Gf.Vec3d(point))))
        counts_out.append(3)
        triangles += 1
    return triangles


def _extent(points):
    mins = [min(float(point[axis]) for point in points) for axis in range(3)]
    maxs = [max(float(point[axis]) for point in points) for axis in range(3)]
    return [Gf.Vec3f(*mins), Gf.Vec3f(*maxs)]


def _apply_codeless_api(prim, schema_name):
    """Author a codeless API token without requiring its plugin in usd-core."""
    current = prim.GetMetadata("apiSchemas")
    items = list(current.explicitItems) if current else []
    if schema_name not in items:
        items.append(schema_name)
    prim.SetMetadata("apiSchemas", Sdf.TokenListOp.CreateExplicit(items))


def _define_spline_zone(
    stage, path, points, counts, indices, speed, path_start, path_end, material, zone_name
):
    body = UsdGeom.Xform.Define(stage, path)
    prim = body.GetPrim()
    rigid_body = UsdPhysics.RigidBodyAPI.Apply(prim)
    rigid_body.CreateRigidBodyEnabledAttr(True)
    rigid_body.CreateKinematicEnabledAttr(True)
    rigid_body.CreateVelocityAttr(Gf.Vec3f(0.0))
    rigid_body.CreateAngularVelocityAttr(Gf.Vec3f(0.0))
    UsdPhysics.FilteredPairsAPI.Apply(prim).CreateFilteredPairsRel().AddTarget(
        MACHINE_BODY
    )

    mesh = UsdGeom.Mesh.Define(stage, f"{path}/CollisionMesh")
    mesh.GetPointsAttr().Set(points)
    mesh.GetFaceVertexCountsAttr().Set(counts)
    mesh.GetFaceVertexIndicesAttr().Set(indices)
    mesh.GetExtentAttr().Set(_extent(points))
    mesh.GetSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    mesh.GetDoubleSidedAttr().Set(True)
    mesh.GetPurposeAttr().Set(UsdGeom.Tokens.default_)
    mesh.GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)
    mesh_prim = mesh.GetPrim()
    UsdPhysics.CollisionAPI.Apply(mesh_prim).CreateCollisionEnabledAttr(True)
    UsdPhysics.MeshCollisionAPI.Apply(mesh_prim).CreateApproximationAttr("none")
    UsdShade.MaterialBindingAPI.Apply(mesh_prim).Bind(
        material, materialPurpose="physics"
    )

    _apply_codeless_api(prim, "PhysxSplinesSurfaceVelocityAPI")
    prim.CreateAttribute(
        "physxSplinesSurfaceVelocity:surfaceVelocityEnabled", Sdf.ValueTypeNames.Bool
    ).Set(True)
    prim.CreateAttribute(
        "physxSplinesSurfaceVelocity:surfaceVelocityMagnitude", Sdf.ValueTypeNames.Float
    ).Set(float(speed))
    prim.CreateAttribute("conveyor:surfaceSpeed", Sdf.ValueTypeNames.Float).Set(
        float(speed)
    )
    prim.CreateAttribute("conveyor:zone", Sdf.ValueTypeNames.Token).Set(zone_name)

    curve = UsdGeom.BasisCurves.Define(stage, f"{path}/MotionPath")
    curve.GetPointsAttr().Set([Gf.Vec3f(path_start), Gf.Vec3f(path_end)])
    curve.GetCurveVertexCountsAttr().Set([2])
    curve.GetTypeAttr().Set(UsdGeom.Tokens.linear)
    curve.GetWrapAttr().Set(UsdGeom.Tokens.nonperiodic)
    curve.GetWidthsAttr().Set([0.01])
    curve.GetPurposeAttr().Set(UsdGeom.Tokens.guide)
    curve.GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)
    prim.CreateRelationship(
        "physxSplinesSurfaceVelocity:surfaceVelocityCurve"
    ).SetTargets([curve.GetPath()])
    return prim


def split_terminal_zone(stage, stop_length, stop_advance):
    conveyor = stage.GetPrimAtPath(DEFAULT_CONVEYOR)
    source_prim = stage.GetPrimAtPath(DEFAULT_MESH)
    curve_prim = stage.GetPrimAtPath(DEFAULT_CURVE)
    if not conveyor or not source_prim or not curve_prim:
        raise RuntimeError("The expected brep_36 conveyor prims were not found")
    decel_paths = [f"{DEFAULT_DECEL_PREFIX}_{index + 1:02d}" for index in range(len(DECEL_SPEEDS) - 1)]
    for path in (DEFAULT_RIGHT, DEFAULT_STOP, *decel_paths):
        if stage.GetPrimAtPath(path):
            raise RuntimeError(f"Conveyor region already exists at {path}")

    source_mesh = UsdGeom.Mesh(source_prim)
    points = source_mesh.GetPointsAttr().Get()
    counts = source_mesh.GetFaceVertexCountsAttr().Get()
    indices = source_mesh.GetFaceVertexIndicesAttr().Get()
    curve = UsdGeom.BasisCurves(curve_prim)
    curve_points = curve.GetPointsAttr().Get()
    if len(curve_points) < 3:
        raise RuntimeError("MotionPath needs at least three points")

    xforms = UsdGeom.XformCache(Usd.TimeCode.Default())
    curve_to_world = xforms.GetLocalToWorldTransform(curve_prim)
    mesh_to_world = xforms.GetLocalToWorldTransform(source_prim)
    world_to_mesh = mesh_to_world.GetInverse()
    root_prim = conveyor.GetParent()
    world_to_root = xforms.GetLocalToWorldTransform(root_prim).GetInverse()
    mesh_to_root = mesh_to_world * world_to_root

    left = world_to_mesh.Transform(curve_to_world.Transform(Gf.Vec3d(curve_points[0])))
    right = world_to_mesh.Transform(curve_to_world.Transform(Gf.Vec3d(curve_points[-1])))
    center = (left + right) * 0.5
    radius = 0.5 * (right - left).GetLength()
    if radius <= 0.0:
        raise RuntimeError("MotionPath has an invalid radius")
    min_y = min(float(point[1]) for point in points)
    stop_boundary_y = min_y + float(stop_length) + float(stop_advance)
    zone_length = float(stop_length) / len(DECEL_SPEEDS)
    zone_edges = [
        stop_boundary_y - zone_length * index for index in range(len(DECEL_SPEEDS))
    ]

    right_leg_points = [
        point
        for point in points
        if float(point[0]) > float(center[0])
        and float(point[1]) <= float(right[1]) + 1.0e-4
    ]
    right_min_x = min(float(point[0]) for point in right_leg_points)
    right_max_x = max(float(point[0]) for point in right_leg_points)
    belt_top_z = max(float(point[2]) for point in right_leg_points)

    def plane_region(upper_y, lower_y):
        local_points = [
            Gf.Vec3d(right_min_x, upper_y, belt_top_z),
            Gf.Vec3d(right_min_x, lower_y, belt_top_z),
            Gf.Vec3d(right_max_x, lower_y, belt_top_z),
            Gf.Vec3d(right_max_x, upper_y, belt_top_z),
        ]
        return {
            "points": [Gf.Vec3f(mesh_to_root.Transform(point)) for point in local_points],
            "counts": [3, 3],
            "indices": [0, 1, 2, 0, 2, 3],
            "faces": 2,
        }

    main_points = []
    main_counts = []
    main_indices = []
    right_region = plane_region(float(right[1]), stop_boundary_y)
    right_points = right_region["points"]
    right_counts = right_region["counts"]
    right_indices = right_region["indices"]
    zone_meshes = [
        plane_region(upper_y, lower_y)
        for upper_y, lower_y in zip(zone_edges[:-1], zone_edges[1:])
    ]
    zone_meshes.append(plane_region(zone_edges[-1], min_y))
    identity = Gf.Matrix4d(1.0)
    main_faces = 0
    right_faces = right_region["faces"]
    for record in _face_records(counts, indices):
        polygon = [Gf.Vec3d(points[int(index)]) for index in record[1]]
        face_center = _centroid(points, record[1])
        # The right leg lies below the curve's right endpoint.  Splitting by
        # its terminal Y range keeps the U bend moving and makes only the last
        # short part of the right straight leg static.
        is_right_leg = (
            float(face_center[0]) > float(center[0])
            and min(float(point[1]) for point in polygon) < float(right[1])
        )
        if is_right_leg:
            main_polygon = _clip_at_y(polygon, float(right[1]), keep_below=False)
        else:
            main_polygon = polygon
        main_faces += _append_polygon(
            main_points, main_counts, main_indices, main_polygon, identity
        )

    if not right_counts or not main_counts or any(not zone["counts"] for zone in zone_meshes):
        raise RuntimeError("The requested split did not produce every conveyor region")

    source_mesh.GetPointsAttr().Set(main_points)
    source_mesh.GetFaceVertexCountsAttr().Set(main_counts)
    source_mesh.GetFaceVertexIndicesAttr().Set(main_indices)
    source_mesh.GetExtentAttr().Set(_extent(main_points))
    source_prim.RemoveProperty("normals")
    for prop in list(source_prim.GetProperties()):
        if prop.GetName().startswith("primvars:"):
            source_prim.RemoveProperty(prop.GetName())

    material_prim = stage.GetPrimAtPath(DEFAULT_MATERIAL)
    if not material_prim:
        raise RuntimeError(f"Physics material not found at {DEFAULT_MATERIAL}")
    material = UsdShade.Material(material_prim)

    full_speed = float(
        conveyor.GetAttribute(
            "physxSplinesSurfaceVelocity:surfaceVelocityMagnitude"
        ).Get()
    )
    right_join_root = world_to_root.Transform(curve_to_world.Transform(Gf.Vec3d(curve_points[-1])))
    right_end_root = world_to_root.Transform(
        mesh_to_world.Transform(Gf.Vec3d(right[0], stop_boundary_y, right[2]))
    )
    _define_spline_zone(
        stage,
        DEFAULT_RIGHT,
        right_points,
        right_counts,
        right_indices,
        full_speed,
        right_join_root,
        right_end_root,
        material,
        "right_straight_moving",
    )

    for index, (path, speed) in enumerate(zip(decel_paths, DECEL_SPEEDS[:-1])):
        zone = zone_meshes[index]
        upper_root = world_to_root.Transform(
            mesh_to_world.Transform(Gf.Vec3d(right[0], zone_edges[index], right[2]))
        )
        lower_root = world_to_root.Transform(
            mesh_to_world.Transform(Gf.Vec3d(right[0], zone_edges[index + 1], right[2]))
        )
        _define_spline_zone(
            stage,
            path,
            zone["points"],
            zone["counts"],
            zone["indices"],
            speed,
            upper_root,
            lower_root,
            material,
            f"deceleration_{index + 1:02d}",
        )

    conveyor.CreateAttribute(
        "conveyor:decelerationLength", Sdf.ValueTypeNames.Float
    ).Set(float(stop_length) + float(stop_advance))
    conveyor.CreateAttribute(
        "conveyor:stopAdvance", Sdf.ValueTypeNames.Float
    ).Set(float(stop_advance))
    conveyor.CreateAttribute(
        "conveyor:decelerationSpeeds", Sdf.ValueTypeNames.FloatArray
    ).Set(list(DECEL_SPEEDS))

    final_zone = zone_meshes[-1]
    stop_mesh = UsdGeom.Mesh.Define(stage, DEFAULT_STOP)
    stop_mesh.GetPointsAttr().Set(final_zone["points"])
    stop_mesh.GetFaceVertexCountsAttr().Set(final_zone["counts"])
    stop_mesh.GetFaceVertexIndicesAttr().Set(final_zone["indices"])
    stop_mesh.GetExtentAttr().Set(_extent(final_zone["points"]))
    stop_mesh.GetSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    stop_mesh.GetDoubleSidedAttr().Set(True)
    stop_mesh.GetPurposeAttr().Set(UsdGeom.Tokens.default_)
    stop_mesh.GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)

    stop_prim = stop_mesh.GetPrim()
    UsdPhysics.CollisionAPI.Apply(stop_prim).CreateCollisionEnabledAttr(True)
    UsdPhysics.MeshCollisionAPI.Apply(stop_prim).CreateApproximationAttr("none")
    stop_prim.CreateAttribute("conveyor:zone", Sdf.ValueTypeNames.Token).Set(
        "terminal_zero_speed"
    )
    stop_prim.CreateAttribute("conveyor:surfaceSpeed", Sdf.ValueTypeNames.Float).Set(0.0)
    stop_prim.CreateAttribute("conveyor:zoneLength", Sdf.ValueTypeNames.Float).Set(
        zone_edges[-1] - min_y
    )

    binding = UsdShade.MaterialBindingAPI.Apply(stop_prim)
    binding.Bind(material, materialPurpose="physics")

    return {
        "radius": radius,
        "right_join_y": float(right[1]),
        "terminal_min_y": min_y,
        "stop_boundary_y": stop_boundary_y,
        "stop_advance": float(stop_advance),
        "zero_speed_start_y": zone_edges[-1],
        "main_faces": main_faces,
        "right_moving_faces": right_faces,
        "deceleration_speeds": list(DECEL_SPEEDS),
        "deceleration_zone_length": zone_length,
        "deceleration_zone_faces": [zone["faces"] for zone in zone_meshes],
        "stop_faces": final_zone["faces"],
        "stop_points": len(final_zone["points"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("usd", type=Path)
    parser.add_argument("--stop-length", type=float, default=0.18)
    parser.add_argument("--stop-advance", type=float, default=0.10)
    parser.add_argument("--backup", type=Path)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    usd_path = args.usd.resolve()
    if args.stop_length <= 0.0:
        raise ValueError("--stop-length must be positive")
    if args.stop_advance < 0.0:
        raise ValueError("--stop-advance cannot be negative")
    stage = Usd.Stage.Open(str(usd_path))
    if not stage:
        raise RuntimeError(f"Unable to open {usd_path}")

    if args.apply:
        backup = (
            args.backup.resolve()
            if args.backup
            else usd_path.with_name(f"{usd_path.stem}.before_terminal_zero_zone.bak.usd")
        )
        if backup.exists():
            raise FileExistsError(f"Refusing to overwrite existing backup: {backup}")
        shutil.copy2(usd_path, backup)

    report = split_terminal_zone(stage, args.stop_length, args.stop_advance)
    if args.apply:
        stage.GetRootLayer().Save()
        print(f"saved={usd_path}")
        print(f"backup={backup}")
    else:
        print("dry_run=true")
    for key, value in report.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
