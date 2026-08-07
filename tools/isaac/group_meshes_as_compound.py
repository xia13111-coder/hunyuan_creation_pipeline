#!/usr/bin/env python3
"""Move selected USD meshes into one independent compound rigid body.

The input layer is never saved.  A new USD is exported only after namespace,
transform, collider ownership, relationship, and mass-conservation checks pass.

Run this with a Python environment that provides Pixar USD, for example:

    "$ISAACSIM_ROOT/python.sh" \
      tools/isaac/group_meshes_as_compound.py \
      --input asset.usd --output asset_grouped.usd \
      --mesh /World/PartA/Mesh \
      --mesh /World/PartB/Mesh \
      --mesh /World/PivotPart/Mesh \
      --part-name PartA --part-name PartB --part-name PivotPart \
      --pivot-mesh /World/PivotPart/Mesh
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Group selected collider meshes under one compound rigid body."
    )
    parser.add_argument("--input", required=True, help="Source USD; never modified")
    parser.add_argument("--output", required=True, help="New grouped USD")
    parser.add_argument(
        "--mesh",
        action="append",
        required=True,
        help="Absolute mesh prim path; repeat for every selected mesh",
    )
    parser.add_argument(
        "--part-name",
        action="append",
        help="Unique wrapper name matching each --mesh; defaults to Part_01, ...",
    )
    parser.add_argument(
        "--pivot-mesh",
        help="Mesh whose current world frame becomes the compound-body frame",
    )
    parser.add_argument("--group-name", default="MovingAssembly")
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Make the compound body dynamic instead of temporarily kinematic",
    )
    parser.add_argument(
        "--revolute",
        action="store_true",
        help=(
            "Create a revolute joint from the original rigid body to the "
            "compound at the pivot-mesh frame; implies --dynamic"
        ),
    )
    parser.add_argument(
        "--revolute-axis",
        choices=("X", "Y", "Z"),
        default="X",
        help="Allowed rotation axis in the pivot-mesh frame",
    )
    parser.add_argument(
        "--joint-name",
        default="MovingAssemblyRevoluteJoint",
        help="Sibling prim name for the optional revolute joint",
    )
    parser.add_argument(
        "--report",
        help="JSON report path; defaults beside output with .group_report.json suffix",
    )
    return parser.parse_args()


def fail(message: str) -> None:
    raise RuntimeError(message)


def matrix_rows(matrix: Gf.Matrix4d) -> list[list[float]]:
    return [[float(matrix[row][col]) for col in range(4)] for row in range(4)]


def max_matrix_error(left: Gf.Matrix4d, right: Gf.Matrix4d) -> float:
    return max(
        abs(float(left[row][col]) - float(right[row][col]))
        for row in range(4)
        for col in range(4)
    )


def matrix_det3(matrix: Gf.Matrix4d) -> float:
    a00, a01, a02 = matrix[0][0], matrix[0][1], matrix[0][2]
    a10, a11, a12 = matrix[1][0], matrix[1][1], matrix[1][2]
    a20, a21, a22 = matrix[2][0], matrix[2][1], matrix[2][2]
    return float(
        a00 * (a11 * a22 - a12 * a21)
        - a01 * (a10 * a22 - a12 * a20)
        + a02 * (a10 * a21 - a11 * a20)
    )


def pose_from_matrix(matrix: Gf.Matrix4d) -> tuple[Gf.Vec3f, Gf.Quatf]:
    translation = matrix.ExtractTranslation()
    rotation = matrix.ExtractRotationQuat().GetNormalized()
    imaginary = rotation.GetImaginary()
    return (
        Gf.Vec3f(
            float(translation[0]),
            float(translation[1]),
            float(translation[2]),
        ),
        Gf.Quatf(
            float(rotation.GetReal()),
            Gf.Vec3f(
                float(imaginary[0]),
                float(imaginary[1]),
                float(imaginary[2]),
            ),
        ),
    )


def matrix_from_pose(position: Gf.Vec3f, rotation: Gf.Quatf) -> Gf.Matrix4d:
    transform = Gf.Transform()
    transform.SetRotation(Gf.Rotation(Gf.Quatd(rotation)))
    transform.SetTranslation(Gf.Vec3d(position))
    return transform.GetMatrix()


def triangulated_indices(counts: Any, indices: Any):
    cursor = 0
    for count in counts or []:
        face = indices[cursor : cursor + count]
        cursor += count
        for index in range(1, count - 1):
            yield face[0], face[index], face[index + 1]


def mesh_world_volume(mesh_prim: Usd.Prim, world: Gf.Matrix4d) -> float:
    """Match add_physics.py's absolute signed-volume weighting."""
    mesh = UsdGeom.Mesh(mesh_prim)
    points = mesh.GetPointsAttr().Get() or []
    counts = mesh.GetFaceVertexCountsAttr().Get() or []
    indices = mesh.GetFaceVertexIndicesAttr().Get() or []
    if not points or not counts or not indices:
        return 0.0

    transformed = [
        world.Transform(Gf.Vec3d(float(point[0]), float(point[1]), float(point[2])))
        for point in points
    ]
    volume = 0.0
    for index0, index1, index2 in triangulated_indices(counts, indices):
        point0 = transformed[index0]
        point1 = transformed[index1]
        point2 = transformed[index2]
        volume += Gf.Dot(point0, Gf.Cross(point1, point2)) / 6.0
    return abs(float(volume))


def value_digest(value: Any) -> str:
    return hashlib.sha256(repr(value).encode("utf-8")).hexdigest()


def relationship_targets(prim: Usd.Prim) -> dict[str, list[str]]:
    return {
        relationship.GetName(): [str(path) for path in relationship.GetTargets()]
        for relationship in prim.GetRelationships()
        if relationship.GetTargets()
    }


def mesh_signature(prim: Usd.Prim) -> dict[str, Any]:
    mesh = UsdGeom.Mesh(prim)
    xform_ops = [
        (
            operation.GetOpName(),
            str(operation.GetOpType()),
            repr(operation.Get()),
        )
        for operation in UsdGeom.Xformable(prim).GetOrderedXformOps()
    ]
    return {
        "type": prim.GetTypeName(),
        "applied_schemas": sorted(prim.GetAppliedSchemas()),
        "point_count": len(mesh.GetPointsAttr().Get() or []),
        "face_count": len(mesh.GetFaceVertexCountsAttr().Get() or []),
        "points_sha256": value_digest(mesh.GetPointsAttr().Get()),
        "face_counts_sha256": value_digest(mesh.GetFaceVertexCountsAttr().Get()),
        "face_indices_sha256": value_digest(mesh.GetFaceVertexIndicesAttr().Get()),
        "xform_ops": xform_ops,
        "child_types": sorted(
            (child.GetName(), child.GetTypeName()) for child in prim.GetChildren()
        ),
        "relationships": relationship_targets(prim),
    }


def nearest_rigid_body(prim: Usd.Prim) -> Usd.Prim:
    current = prim
    while current and current.IsValid() and not current.IsPseudoRoot():
        if current.HasAPI(UsdPhysics.RigidBodyAPI):
            return current
        current = current.GetParent()
    return Usd.Prim()


def enabled_colliders(stage: Usd.Stage) -> list[Usd.Prim]:
    result = []
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        enabled = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
        if enabled is not False:
            result.append(prim)
    return result


def all_rigid_bodies(stage: Usd.Stage) -> list[Usd.Prim]:
    return [
        prim
        for prim in stage.Traverse()
        if prim.HasAPI(UsdPhysics.RigidBodyAPI)
        and UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Get() is not False
    ]


def path_is_under(path: Sdf.Path, parent: Sdf.Path) -> bool:
    return path == parent or path.HasPrefix(parent)


def validate_names(names: list[str], count: int) -> list[str]:
    if not names:
        names = [f"Part_{index + 1:02d}" for index in range(count)]
    if len(names) != count:
        fail("--part-name count must match --mesh count")
    if len(set(names)) != len(names):
        fail("--part-name values must be unique")
    for name in names:
        if not Sdf.Path.IsValidIdentifier(name):
            fail(f"Invalid USD identifier for --part-name: {name!r}")
    return names


def capture_joint_state(stage: Usd.Stage) -> list[dict[str, Any]]:
    records = []
    for prim in stage.TraverseAll():
        if not prim.IsA(UsdPhysics.Joint):
            continue
        joint = UsdPhysics.Joint(prim)
        revolute = (
            UsdPhysics.RevoluteJoint(prim)
            if prim.IsA(UsdPhysics.RevoluteJoint)
            else None
        )
        records.append(
            {
                "path": str(prim.GetPath()),
                "type": prim.GetTypeName(),
                "body0": [
                    str(path)
                    for path in joint.GetBody0Rel().GetTargets()
                ],
                "body1": [
                    str(path)
                    for path in joint.GetBody1Rel().GetTargets()
                ],
                "enabled": joint.GetJointEnabledAttr().Get(),
                "collision_enabled": joint.GetCollisionEnabledAttr().Get(),
                "local_pos0": repr(joint.GetLocalPos0Attr().Get()),
                "local_rot0": repr(joint.GetLocalRot0Attr().Get()),
                "local_pos1": repr(joint.GetLocalPos1Attr().Get()),
                "local_rot1": repr(joint.GetLocalRot1Attr().Get()),
                "axis": (
                    str(revolute.GetAxisAttr().Get()) if revolute else None
                ),
                "lower_limit": (
                    repr(revolute.GetLowerLimitAttr().Get())
                    if revolute
                    else None
                ),
                "upper_limit": (
                    repr(revolute.GetUpperLimitAttr().Get())
                    if revolute
                    else None
                ),
                "applied_schemas": sorted(prim.GetAppliedSchemas()),
            }
        )
    return records


def joint_targets_direct_rigid_bodies(
    stage: Usd.Stage, prim: Usd.Prim
) -> bool:
    """Return whether a joint has usable direct rigid-body/world targets."""
    joint = UsdPhysics.Joint(prim)
    body0 = joint.GetBody0Rel().GetTargets()
    body1 = joint.GetBody1Rel().GetTargets()
    if len(body0) > 1 or len(body1) > 1:
        return False
    if not body0 and not body1:
        return False

    all_targets = [*body0, *body1]
    for target in all_targets:
        target_prim = stage.GetPrimAtPath(target)
        if (
            not target_prim.IsValid()
            or not target_prim.HasAPI(UsdPhysics.RigidBodyAPI)
        ):
            return False
    if body0 and body1 and body0[0] == body1[0]:
        return False
    return True


def unresolved_relationships(stage: Usd.Stage) -> list[dict[str, str]]:
    unresolved = []
    for prim in stage.TraverseAll():
        for relationship in prim.GetRelationships():
            for target in relationship.GetTargets():
                if target.IsPrimPath() and not stage.GetPrimAtPath(target).IsValid():
                    unresolved.append(
                        {
                            "prim": str(prim.GetPath()),
                            "relationship": relationship.GetName(),
                            "target": str(target),
                        }
                    )
    return unresolved


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    report_path = (
        Path(args.report).expanduser().resolve()
        if args.report
        else output_path.with_suffix(".group_report.json")
    )

    if input_path == output_path:
        fail("Input and output must differ; the source USD is never overwritten")
    if not input_path.is_file():
        fail(f"Input USD does not exist: {input_path}")
    if output_path.exists():
        fail(f"Output already exists: {output_path}")
    if report_path.exists():
        fail(f"Report already exists: {report_path}")
    if not Sdf.Path.IsValidIdentifier(args.group_name):
        fail(f"Invalid USD identifier for --group-name: {args.group_name!r}")
    if args.revolute and not Sdf.Path.IsValidIdentifier(args.joint_name):
        fail(f"Invalid USD identifier for --joint-name: {args.joint_name!r}")

    source_stat = input_path.stat()
    stage = Usd.Stage.Open(str(input_path), Usd.Stage.LoadAll)
    if not stage:
        fail(f"Could not open USD: {input_path}")
    default_prim = stage.GetDefaultPrim()
    if not default_prim.IsValid():
        fail("Input stage has no valid default prim")

    mesh_paths = [Sdf.Path(path) for path in args.mesh]
    if len(mesh_paths) < 2:
        fail("At least two --mesh paths are required")
    if len(set(mesh_paths)) != len(mesh_paths):
        fail("--mesh paths must be unique")
    for index, path in enumerate(mesh_paths):
        for other_path in mesh_paths[index + 1 :]:
            if path_is_under(path, other_path) or path_is_under(other_path, path):
                fail("Selected mesh paths cannot be ancestors of one another")

    part_names = validate_names(args.part_name or [], len(mesh_paths))
    pivot_path = Sdf.Path(args.pivot_mesh) if args.pivot_mesh else mesh_paths[-1]
    if pivot_path not in mesh_paths:
        fail("--pivot-mesh must also be listed as --mesh")

    meshes = []
    for path in mesh_paths:
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid() or not prim.IsA(UsdGeom.Mesh):
            fail(f"Selected path is not a valid Mesh: {path}")
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            fail(f"Selected Mesh has no PhysicsCollisionAPI: {path}")
        meshes.append(prim)

    old_owners = [nearest_rigid_body(prim) for prim in meshes]
    if not all(owner.IsValid() for owner in old_owners):
        fail("Every selected Mesh must currently resolve to a rigid body")
    old_owner_paths = {owner.GetPath() for owner in old_owners}
    if len(old_owner_paths) != 1:
        fail(
            "Selected Meshes do not share one original rigid body: "
            + ", ".join(sorted(str(path) for path in old_owner_paths))
        )
    old_body = old_owners[0]
    old_body_path = old_body.GetPath()
    if old_body == default_prim:
        fail(
            "The default prim itself is the original rigid body; a non-rigid outer "
            "container is required to create a sibling compound body"
        )
    if old_body.GetParent() != default_prim:
        fail(
            f"Expected original rigid body {old_body_path} to be a direct child of "
            f"default prim {default_prim.GetPath()}"
        )

    group_path = default_prim.GetPath().AppendChild(args.group_name)
    if stage.GetPrimAtPath(group_path).IsValid():
        fail(f"Group path already exists: {group_path}")
    joint_path = (
        default_prim.GetPath().AppendChild(args.joint_name)
        if args.revolute
        else Sdf.Path.emptyPath
    )
    if args.revolute and joint_path == group_path:
        fail("--joint-name and --group-name must create different prim paths")
    if args.revolute and stage.GetPrimAtPath(joint_path).IsValid():
        fail(f"Joint path already exists: {joint_path}")

    old_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    old_world = {
        path: old_cache.GetLocalToWorldTransform(stage.GetPrimAtPath(path))
        for path in mesh_paths
    }
    old_parent_world = {
        path: old_cache.GetLocalToWorldTransform(stage.GetPrimAtPath(path).GetParent())
        for path in mesh_paths
    }
    old_joints = capture_joint_state(stage)
    nested_joints = [
        prim
        for prim in stage.TraverseAll()
        if prim.IsA(UsdPhysics.Joint)
        and any(path_is_under(prim.GetPath(), path) for path in mesh_paths)
    ]
    valid_nested_joints = [
        prim.GetPath()
        for prim in nested_joints
        if joint_targets_direct_rigid_bodies(stage, prim)
    ]
    if valid_nested_joints:
        fail(
            "Selected Mesh subtree contains valid joints that require explicit "
            "handling: "
            + ", ".join(str(path) for path in valid_nested_joints)
        )
    removed_invalid_joints = [prim.GetPath() for prim in nested_joints]
    removed_invalid_joint_strings = {
        str(path) for path in removed_invalid_joints
    }
    preserved_old_joints = [
        record
        for record in old_joints
        if record["path"] not in removed_invalid_joint_strings
    ]
    for invalid_joint_path in removed_invalid_joints:
        if not stage.RemovePrim(invalid_joint_path):
            fail(f"Could not remove invalid joint: {invalid_joint_path}")

    old_signatures = {
        path: mesh_signature(stage.GetPrimAtPath(path)) for path in mesh_paths
    }
    old_collider_count = len(enabled_colliders(stage))
    old_rigid_bodies = [str(prim.GetPath()) for prim in all_rigid_bodies(stage)]
    old_rigid_body_paths = set(old_rigid_bodies)
    old_unresolved = unresolved_relationships(stage)

    old_mass_attr = UsdPhysics.MassAPI(old_body).GetMassAttr()
    old_mass = old_mass_attr.Get() if old_mass_attr else None
    if old_mass is None or not math.isfinite(float(old_mass)) or float(old_mass) <= 0:
        fail(f"Original rigid body has no usable authored mass: {old_body_path}")
    old_mass = float(old_mass)

    all_body_meshes = [
        prim
        for prim in Usd.PrimRange(old_body)
        if prim.IsA(UsdGeom.Mesh)
        and nearest_rigid_body(prim).GetPath() == old_body_path
    ]
    volume_by_path = {
        prim.GetPath(): mesh_world_volume(
            prim, old_cache.GetLocalToWorldTransform(prim)
        )
        for prim in all_body_meshes
    }
    total_volume = sum(volume_by_path.values())
    selected_volume = sum(volume_by_path.get(path, 0.0) for path in mesh_paths)
    if total_volume <= 0.0 or selected_volume <= 0.0:
        fail("Could not compute positive volume weights for mass splitting")
    moving_mass = old_mass * selected_volume / total_volume
    base_mass = old_mass - moving_mass
    if moving_mass <= 0.0 or base_mass <= 0.0:
        fail("Computed invalid mass split")

    pivot_world = old_world[pivot_path]
    default_world = old_cache.GetLocalToWorldTransform(default_prim)
    if abs(matrix_det3(default_world)) < 1e-12:
        fail("Default prim world transform is singular")
    group_local = pivot_world * default_world.GetInverse()
    if abs(matrix_det3(group_local)) < 1e-12:
        fail("Pivot Mesh world transform is singular")

    group = UsdGeom.Xform.Define(stage, group_path)
    group_transform = UsdGeom.Xformable(group).MakeMatrixXform()
    group_transform.Set(group_local)
    group.GetPrim().SetDisplayName(
        f"{args.group_name} — {len(mesh_paths)} Mesh Compound"
    )

    wrapper_paths = []
    destination_paths = []
    for source_path, part_name in zip(mesh_paths, part_names):
        wrapper_path = group_path.AppendChild(part_name)
        wrapper = UsdGeom.Xform.Define(stage, wrapper_path)
        # Absorb the old ancestor transform into the wrapper.  The Mesh keeps
        # its exact authored translate/orient/scale ops and therefore remains
        # semantically unchanged.
        wrapper_local = old_parent_world[source_path] * pivot_world.GetInverse()
        UsdGeom.Xformable(wrapper).MakeMatrixXform().Set(wrapper_local)
        wrapper_paths.append(wrapper_path)
        destination_paths.append(wrapper_path.AppendChild("Mesh"))

    for source_path, destination_path in zip(mesh_paths, destination_paths):
        # NamespaceEditor represents one pending edit at a time.  Apply each
        # move atomically so a later MovePrimAtPath call cannot replace an
        # earlier pending edit.
        editor = Usd.NamespaceEditor(stage)
        if not editor.MovePrimAtPath(source_path, destination_path):
            fail(f"Namespace editor rejected move: {source_path} -> {destination_path}")
        can_apply = editor.CanApplyEdits()
        if not can_apply:
            fail(
                f"Namespace edit cannot be applied for {source_path}: {can_apply}"
            )
        if not editor.ApplyEdits():
            fail(f"Namespace editor failed to move: {source_path}")

    for destination_path in destination_paths:
        moved_prim = stage.GetPrimAtPath(destination_path)
        if not moved_prim.IsValid():
            fail(f"Moved Mesh is missing: {destination_path}")

    rigid_body = UsdPhysics.RigidBodyAPI.Apply(group.GetPrim())
    rigid_body.CreateRigidBodyEnabledAttr(True)
    compound_kinematic = not (args.dynamic or args.revolute)
    rigid_body.CreateKinematicEnabledAttr(compound_kinematic)
    rigid_body.CreateVelocityAttr(Gf.Vec3f(0.0))
    rigid_body.CreateAngularVelocityAttr(Gf.Vec3f(0.0))
    UsdPhysics.MassAPI.Apply(group.GetPrim()).CreateMassAttr(moving_mass)
    UsdPhysics.MassAPI(old_body).CreateMassAttr(base_mass)

    joint_report = None
    if args.revolute:
        # The group frame is the selected pivot Mesh's old world frame.  Use
        # that same world frame for both joint anchors, expressed locally in
        # each rigid body.  With body1 equal to the group this makes its local
        # joint pose identity, while body0 receives the exact offset.
        current_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        body0_world = current_cache.GetLocalToWorldTransform(old_body)
        body1_world = current_cache.GetLocalToWorldTransform(group.GetPrim())
        if abs(matrix_det3(body0_world)) < 1e-12:
            fail(f"Original rigid-body transform is singular: {old_body_path}")
        if abs(matrix_det3(body1_world)) < 1e-12:
            fail(f"Compound rigid-body transform is singular: {group_path}")

        local0_matrix = pivot_world * body0_world.GetInverse()
        local1_matrix = pivot_world * body1_world.GetInverse()
        local_pos0, local_rot0 = pose_from_matrix(local0_matrix)
        local_pos1, local_rot1 = pose_from_matrix(local1_matrix)

        revolute = UsdPhysics.RevoluteJoint.Define(stage, joint_path)
        revolute.CreateJointEnabledAttr(True)
        revolute.CreateCollisionEnabledAttr(False)
        revolute.CreateBody0Rel().SetTargets([old_body_path])
        revolute.CreateBody1Rel().SetTargets([group_path])
        revolute.CreateLocalPos0Attr(local_pos0)
        revolute.CreateLocalRot0Attr(local_rot0)
        revolute.CreateLocalPos1Attr(local_pos1)
        revolute.CreateLocalRot1Attr(local_rot1)
        revolute.CreateAxisAttr(args.revolute_axis)

        joint_world0 = matrix_from_pose(local_pos0, local_rot0) * body0_world
        joint_world1 = matrix_from_pose(local_pos1, local_rot1) * body1_world
        anchor_error0 = max_matrix_error(joint_world0, pivot_world)
        anchor_error1 = max_matrix_error(joint_world1, pivot_world)
        if max(anchor_error0, anchor_error1) > 1e-6:
            fail(
                "Revolute joint anchor frames do not coincide: "
                f"body0_error={anchor_error0}, body1_error={anchor_error1}"
            )
        joint_report = {
            "path": str(joint_path),
            "axis": args.revolute_axis,
            "body0": str(old_body_path),
            "body1": str(group_path),
            "local_pos0": [float(value) for value in local_pos0],
            "local_rot0_real_imaginary": [
                float(local_rot0.GetReal()),
                *[float(value) for value in local_rot0.GetImaginary()],
            ],
            "local_pos1": [float(value) for value in local_pos1],
            "local_rot1_real_imaginary": [
                float(local_rot1.GetReal()),
                *[float(value) for value in local_rot1.GetImaginary()],
            ],
            "anchor_world_max_abs_error": {
                "body0": anchor_error0,
                "body1": anchor_error1,
            },
            "collision_enabled": False,
        }

    # Validate the composed in-memory stage before writing anything.
    final_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    transform_errors = {}
    signature_matches = {}
    owners = {}
    for source_path, destination_path in zip(mesh_paths, destination_paths):
        moved_prim = stage.GetPrimAtPath(destination_path)
        transform_errors[str(destination_path)] = max_matrix_error(
            old_world[source_path],
            final_cache.GetLocalToWorldTransform(moved_prim),
        )
        signature_matches[str(destination_path)] = (
            old_signatures[source_path] == mesh_signature(moved_prim)
        )
        owner = nearest_rigid_body(moved_prim)
        owners[str(destination_path)] = str(owner.GetPath()) if owner.IsValid() else ""

    if any(error > 1e-10 for error in transform_errors.values()):
        fail(f"World-transform preservation failed: {transform_errors}")
    if not all(signature_matches.values()):
        fail(f"Mesh geometry/schema signature changed: {signature_matches}")
    if set(owners.values()) != {str(group_path)}:
        fail(f"Moved collider ownership is incorrect: {owners}")

    new_colliders = enabled_colliders(stage)
    if len(new_colliders) != old_collider_count:
        fail(
            f"Enabled collider count changed: {old_collider_count} -> "
            f"{len(new_colliders)}"
        )
    group_colliders = [
        prim
        for prim in new_colliders
        if nearest_rigid_body(prim).GetPath() == group_path
    ]
    if len(group_colliders) != len(mesh_paths):
        fail(
            f"Compound body owns {len(group_colliders)} colliders, expected "
            f"{len(mesh_paths)}"
        )

    rigid_bodies = all_rigid_bodies(stage)
    rigid_body_paths = {str(body.GetPath()) for body in rigid_bodies}
    expected_rigid_body_paths = old_rigid_body_paths | {str(group_path)}
    if rigid_body_paths != expected_rigid_body_paths:
        fail(
            "Unexpected enabled rigid-body paths: "
            f"expected={sorted(expected_rigid_body_paths)}, "
            f"actual={sorted(rigid_body_paths)}"
        )
    for body in rigid_bodies:
        ancestor = nearest_rigid_body(body.GetParent())
        if ancestor.IsValid():
            fail(
                f"Nested rigid bodies are not allowed: {ancestor.GetPath()} -> "
                f"{body.GetPath()}"
            )

    new_unresolved = unresolved_relationships(stage)
    if new_unresolved != old_unresolved:
        fail(
            "Namespace move introduced or changed unresolved relationship targets: "
            f"before={old_unresolved}, after={new_unresolved}"
        )
    for old_path in mesh_paths:
        remaining_prim = stage.GetPrimAtPath(old_path)
        if remaining_prim.IsValid():
            fail(
                f"Original Mesh path still exists after move: {old_path} "
                f"(type={remaining_prim.GetTypeName()!r}, "
                f"active={remaining_prim.IsActive()}, "
                f"properties={[prop.GetName() for prop in remaining_prim.GetProperties()]}, "
                f"children={[child.GetName() for child in remaining_prim.GetAllChildren()]}, "
                f"prim_stack={[(str(spec.layer.identifier), str(spec.path), str(spec.specifier)) for spec in remaining_prim.GetPrimStack()]})"
            )

    for prim in stage.TraverseAll():
        for relationship in prim.GetRelationships():
            for target in relationship.GetTargets():
                if any(path_is_under(target, old_path) for old_path in mesh_paths):
                    fail(
                        f"Stale relationship target after namespace move: "
                        f"{prim.GetPath()}.{relationship.GetName()} -> {target}"
                    )

    resulting_joints = capture_joint_state(stage)
    resulting_joints_by_path = {
        record["path"]: record for record in resulting_joints
    }
    for preserved_joint in preserved_old_joints:
        resulting_joint = resulting_joints_by_path.get(preserved_joint["path"])
        if resulting_joint != preserved_joint:
            fail(
                "Existing joint changed while grouping: "
                f"before={preserved_joint}, after={resulting_joint}"
            )
    expected_joint_count = len(preserved_old_joints) + int(args.revolute)
    if len(resulting_joints) != expected_joint_count:
        fail(
            f"Unexpected resulting joint count: expected={expected_joint_count}, "
            f"actual={len(resulting_joints)}, joints={resulting_joints}"
        )
    if args.revolute:
        resulting_joint = resulting_joints_by_path.get(str(joint_path))
        if (
            resulting_joint is None
            or resulting_joint["type"] != "PhysicsRevoluteJoint"
            or resulting_joint["body0"] != [str(old_body_path)]
            or resulting_joint["body1"] != [str(group_path)]
        ):
            fail(f"Resulting revolute joint is incorrect: {resulting_joint}")
    mass_sum_error = abs((base_mass + moving_mass) - old_mass)
    if mass_sum_error > 1e-10:
        fail(f"Mass conservation failed: error={mass_sum_error}")

    current_stat = input_path.stat()
    if (
        current_stat.st_mtime_ns != source_stat.st_mtime_ns
        or current_stat.st_size != source_stat.st_size
    ):
        fail("Input USD changed while grouping; refusing to export a stale result")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output_path.with_name(f".{output_path.stem}.tmp.usd")
    if temporary_output.exists():
        temporary_output.unlink()
    if not stage.GetRootLayer().Export(str(temporary_output)):
        fail(f"Failed to export temporary USD: {temporary_output}")

    reopened = Usd.Stage.Open(str(temporary_output), Usd.Stage.LoadAll)
    if not reopened:
        temporary_output.unlink(missing_ok=True)
        fail("Exported USD could not be reopened")
    reopened_group = reopened.GetPrimAtPath(group_path)
    if (
        not reopened_group.IsValid()
        or not reopened_group.HasAPI(UsdPhysics.RigidBodyAPI)
    ):
        temporary_output.unlink(missing_ok=True)
        fail("Exported USD lost the compound rigid body")
    reopened_colliders = enabled_colliders(reopened)
    if len(reopened_colliders) != old_collider_count:
        temporary_output.unlink(missing_ok=True)
        fail("Exported USD changed enabled collider count")
    reopened_rigid_body_paths = {
        str(prim.GetPath()) for prim in all_rigid_bodies(reopened)
    }
    if reopened_rigid_body_paths != expected_rigid_body_paths:
        temporary_output.unlink(missing_ok=True)
        fail(
            "Exported USD changed rigid-body paths: "
            f"expected={sorted(expected_rigid_body_paths)}, "
            f"actual={sorted(reopened_rigid_body_paths)}"
        )
    reopened_joints_by_path = {
        record["path"]: record for record in capture_joint_state(reopened)
    }
    if reopened_joints_by_path != resulting_joints_by_path:
        temporary_output.unlink(missing_ok=True)
        fail(
            "Exported USD changed joint definitions: "
            f"expected={resulting_joints_by_path}, "
            f"actual={reopened_joints_by_path}"
        )

    os.replace(temporary_output, output_path)

    report = {
        "input": str(input_path),
        "output": str(output_path),
        "default_prim": str(default_prim.GetPath()),
        "original_rigid_body": str(old_body_path),
        "compound_rigid_body": str(group_path),
        "pivot_source_mesh": str(pivot_path),
        "pivot_world_matrix": matrix_rows(pivot_world),
        "path_mapping": {
            str(source): str(destination)
            for source, destination in zip(mesh_paths, destination_paths)
        },
        "world_transform_max_abs_error": transform_errors,
        "mesh_signature_matches": signature_matches,
        "colliders": {
            "total_before": old_collider_count,
            "total_after": len(reopened_colliders),
            "compound_body": len(group_colliders),
        },
        "rigid_bodies_before": old_rigid_bodies,
        "rigid_bodies_after": sorted(rigid_body_paths),
        "mass_kg": {
            "total_before": old_mass,
            "base_after": base_mass,
            "compound_after": moving_mass,
            "conservation_error": mass_sum_error,
            "weighting": "absolute world-space signed mesh volume",
            "selected_volume_m3": selected_volume,
            "total_volume_m3": total_volume,
        },
        "joints_before": old_joints,
        "removed_invalid_joint_paths": [
            str(path) for path in removed_invalid_joints
        ],
        "joints_after": resulting_joints,
        "revolute_joint": joint_report,
        "compound_kinematic_enabled": compound_kinematic,
        "unresolved_relationships": new_unresolved,
        "validation": "PASS",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"PASS: {output_path}")
    print(f"REPORT: {report_path}")
    print(f"GROUP: {group_path}")
    print(f"MASS: base={base_mass:.9f} kg moving={moving_mass:.9f} kg")
    print(f"MAX_WORLD_ERROR: {max(transform_errors.values()):.3e}")


if __name__ == "__main__":
    main()
