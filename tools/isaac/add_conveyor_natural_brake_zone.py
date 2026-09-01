#!/usr/bin/env python3
"""Create a continuous low-friction braking region on the U conveyor."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

from add_conveyor_terminal_stop_zone import (
    DEFAULT_CONVEYOR,
    DEFAULT_CURVE,
    DEFAULT_MATERIAL,
    DEFAULT_MESH,
    DEFAULT_RIGHT,
    _append_polygon,
    _apply_codeless_api,
    _centroid,
    _clip_at_y,
    _define_spline_zone,
    _extent,
    _face_records,
)


BRAKE_PRIM = "/tn__8U_Z2odv1qbE/ConveyorNaturalBrake_brep_36"
BRAKE_MATERIAL = "/tn__8U_Z2odv1qbE/PhysicsMaterials/ConveyorNaturalBrake"


def build_natural_brake(stage, brake_start_world_y, dynamic_friction):
    conveyor = stage.GetPrimAtPath(DEFAULT_CONVEYOR)
    source_prim = stage.GetPrimAtPath(DEFAULT_MESH)
    curve_prim = stage.GetPrimAtPath(DEFAULT_CURVE)
    if not conveyor or not source_prim or not curve_prim:
        raise RuntimeError("The expected brep_36 conveyor prims were not found")
    for path in (DEFAULT_RIGHT, BRAKE_PRIM):
        if stage.GetPrimAtPath(path):
            raise RuntimeError(f"Conveyor region already exists at {path}")

    source_mesh = UsdGeom.Mesh(source_prim)
    points = source_mesh.GetPointsAttr().Get()
    counts = source_mesh.GetFaceVertexCountsAttr().Get()
    indices = source_mesh.GetFaceVertexIndicesAttr().Get()
    curve_points = UsdGeom.BasisCurves(curve_prim).GetPointsAttr().Get()

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
    right_world = curve_to_world.Transform(Gf.Vec3d(curve_points[-1]))
    brake_boundary_mesh_y = float(
        world_to_mesh.Transform(
            Gf.Vec3d(right_world[0], brake_start_world_y, right_world[2])
        )[1]
    )
    min_y = min(float(point[1]) for point in points)
    if not min_y < brake_boundary_mesh_y < float(right[1]):
        raise ValueError("Brake start must lie inside the right straight conveyor leg")

    regions = {
        "main": {"points": [], "counts": [], "indices": [], "faces": 0},
        "right": {"points": [], "counts": [], "indices": [], "faces": 0},
        "brake": {"points": [], "counts": [], "indices": [], "faces": 0},
    }
    identity = Gf.Matrix4d(1.0)
    for record in _face_records(counts, indices):
        polygon = [Gf.Vec3d(points[int(index)]) for index in record[1]]
        face_center = _centroid(points, record[1])
        is_right_leg = (
            float(face_center[0]) > float(center[0])
            and min(float(point[1]) for point in polygon) < float(right[1])
        )
        if is_right_leg:
            above_brake = _clip_at_y(
                polygon, brake_boundary_mesh_y, keep_below=False
            )
            right_polygon = _clip_at_y(
                above_brake, float(right[1]), keep_below=True
            )
            main_polygon = _clip_at_y(
                polygon, float(right[1]), keep_below=False
            )
            brake_polygon = _clip_at_y(
                polygon, brake_boundary_mesh_y, keep_below=True
            )
        else:
            main_polygon = polygon
            right_polygon = []
            brake_polygon = []

        for name, clipped, transform in (
            ("main", main_polygon, identity),
            ("right", right_polygon, mesh_to_root),
            ("brake", brake_polygon, mesh_to_root),
        ):
            region = regions[name]
            region["faces"] += _append_polygon(
                region["points"],
                region["counts"],
                region["indices"],
                clipped,
                transform,
            )

    if any(not region["counts"] for region in regions.values()):
        raise RuntimeError("The split did not produce every conveyor region")

    main = regions["main"]
    source_mesh.GetPointsAttr().Set(main["points"])
    source_mesh.GetFaceVertexCountsAttr().Set(main["counts"])
    source_mesh.GetFaceVertexIndicesAttr().Set(main["indices"])
    source_mesh.GetExtentAttr().Set(_extent(main["points"]))
    source_prim.RemoveProperty("normals")
    for prop in list(source_prim.GetProperties()):
        if prop.GetName().startswith("primvars:"):
            source_prim.RemoveProperty(prop.GetName())

    high_friction = UsdShade.Material(stage.GetPrimAtPath(DEFAULT_MATERIAL))
    full_speed = float(
        conveyor.GetAttribute(
            "physxSplinesSurfaceVelocity:surfaceVelocityMagnitude"
        ).Get()
    )
    right_join_root = world_to_root.Transform(right_world)
    right_end_root = world_to_root.Transform(
        mesh_to_world.Transform(
            Gf.Vec3d(right[0], brake_boundary_mesh_y, right[2])
        )
    )
    right_region = regions["right"]
    _define_spline_zone(
        stage,
        DEFAULT_RIGHT,
        right_region["points"],
        right_region["counts"],
        right_region["indices"],
        full_speed,
        right_join_root,
        right_end_root,
        high_friction,
        "right_straight_moving",
    )

    brake_material = UsdShade.Material.Define(stage, BRAKE_MATERIAL)
    brake_material_prim = brake_material.GetPrim()
    physics_material = UsdPhysics.MaterialAPI.Apply(brake_material_prim)
    physics_material.CreateDynamicFrictionAttr(float(dynamic_friction))
    physics_material.CreateStaticFrictionAttr(0.9)
    physics_material.CreateRestitutionAttr(0.0)
    _apply_codeless_api(brake_material_prim, "PhysxMaterialAPI")
    brake_material_prim.CreateAttribute(
        "physxMaterial:frictionCombineMode", Sdf.ValueTypeNames.Token
    ).Set("min")

    brake = regions["brake"]
    brake_mesh = UsdGeom.Mesh.Define(stage, BRAKE_PRIM)
    brake_mesh.GetPointsAttr().Set(brake["points"])
    brake_mesh.GetFaceVertexCountsAttr().Set(brake["counts"])
    brake_mesh.GetFaceVertexIndicesAttr().Set(brake["indices"])
    brake_mesh.GetExtentAttr().Set(_extent(brake["points"]))
    brake_mesh.GetSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    brake_mesh.GetDoubleSidedAttr().Set(True)
    brake_mesh.GetPurposeAttr().Set(UsdGeom.Tokens.default_)
    brake_mesh.GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)
    brake_prim = brake_mesh.GetPrim()
    UsdPhysics.CollisionAPI.Apply(brake_prim).CreateCollisionEnabledAttr(True)
    UsdPhysics.MeshCollisionAPI.Apply(brake_prim).CreateApproximationAttr("none")
    UsdShade.MaterialBindingAPI.Apply(brake_prim).Bind(
        brake_material, materialPurpose="physics"
    )
    brake_prim.CreateAttribute("conveyor:zone", Sdf.ValueTypeNames.Token).Set(
        "continuous_natural_brake"
    )
    brake_prim.CreateAttribute(
        "conveyor:brakeStartWorldY", Sdf.ValueTypeNames.Float
    ).Set(float(brake_start_world_y))
    brake_prim.CreateAttribute(
        "conveyor:dynamicFriction", Sdf.ValueTypeNames.Float
    ).Set(float(dynamic_friction))

    estimated_distance = full_speed * full_speed / (2.0 * dynamic_friction * 9.81)
    estimated_stop_y = float(brake_start_world_y) - estimated_distance
    conveyor.CreateAttribute(
        "conveyor:brakeStartWorldY", Sdf.ValueTypeNames.Float
    ).Set(float(brake_start_world_y))
    conveyor.CreateAttribute(
        "conveyor:estimatedStopWorldY", Sdf.ValueTypeNames.Float
    ).Set(estimated_stop_y)
    conveyor.CreateAttribute(
        "conveyor:brakeDynamicFriction", Sdf.ValueTypeNames.Float
    ).Set(float(dynamic_friction))

    return {
        "brake_start_world_y": float(brake_start_world_y),
        "brake_dynamic_friction": float(dynamic_friction),
        "estimated_stop_world_y": estimated_stop_y,
        "main_faces": main["faces"],
        "right_faces": right_region["faces"],
        "brake_faces": brake["faces"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("usd", type=Path)
    parser.add_argument("--brake-start-world-y", type=float, default=-0.05)
    parser.add_argument("--dynamic-friction", type=float, default=0.015)
    parser.add_argument("--backup", type=Path)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if not 0.0 < args.dynamic_friction < 1.0:
        raise ValueError("--dynamic-friction must be between zero and one")

    usd_path = args.usd.resolve()
    stage = Usd.Stage.Open(str(usd_path))
    if not stage:
        raise RuntimeError(f"Unable to open {usd_path}")
    if args.apply:
        backup = (
            args.backup.resolve()
            if args.backup
            else usd_path.with_name(f"{usd_path.stem}.before_natural_brake.bak.usd")
        )
        if backup.exists():
            raise FileExistsError(f"Refusing to overwrite existing backup: {backup}")
        shutil.copy2(usd_path, backup)

    report = build_natural_brake(
        stage, args.brake_start_world_y, args.dynamic_friction
    )
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
