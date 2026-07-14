"""Add Isaac Sim physics authoring data to USD assets.

The script also prepares imported CAD USD geometry for PhysX by normalizing
units, optionally centering visible geometry at the world origin, centering
mesh-local origins, fixing world-space mesh winding, and diagnosing topology
that cannot define a reliable closed volume.

Example:
    /home/user/isaacsim500/python.sh ./tools/isaac/add_physics.py \
      --folder ./downloads_refined_mesh/postprocess_glbs \
      --material-file ./materials.json \
      --out-dir ./output_intermediate \
      --headless \
      --material plastic \
      --approx sdf
"""

import argparse
import fnmatch
import json
import math
import os
import re

from isaac_sim_compat import get_simulation_app_class


SimulationApp = get_simulation_app_class()

def log(*a):
    print(*a, flush=True)

def build_sim(headless):
    return SimulationApp({"headless": headless})

def parse_args():
    ap = argparse.ArgumentParser(description="批量为 USD 添加物理（动态刚体 + 支持 SDF/凸分解等碰撞体）")
    ap.add_argument("--folder", required=True, help="输入 USD 根目录（递归 .usd/.usda/.usdc）")
    ap.add_argument("--config", help="规则 JSON（可选，支持 defaults / rules）")
    ap.add_argument("--material-file", required=True, help="材质参数 JSON（必须，定义 materials）")
    ap.add_argument("--out-dir", help="输出根目录（镜像结构）；不填则覆盖到原目录旁（加后缀）")
    ap.add_argument("--suffix", default="_phys", help="输出文件名后缀（默认 _phys）")
    ap.add_argument("--headless", action="store_true", help="无界面运行")

    ap.add_argument("--set-mass", type=float, help="设置整个资产总质量(kg)，多刚体时按体积权重分配")
    ap.add_argument("--material", help="统一材质标签（如 steel/rubber/wood/...），覆盖密度/摩擦/回弹/组合模式")
    ap.add_argument("--center-origin", action="store_true", help="把资产包围盒中心移动到世界原点")

    ap.add_argument("--approx", help="碰撞近似（sdf/convexHull/convexDecomposition/triangleMesh/meshSimplification/box/sphere）")
    ap.add_argument("--force-sdf", action="store_true",
                    help="无论规则/命令行指定为何，动态刚体一律强制使用 SDF")

    ap.add_argument("--sdf-res", type=int, default=256, help="SDF 分辨率（sdfResolution）>0 才会启用 SDF（默认 256）")
    ap.add_argument("--sdf-subgrid", type=int, default=6, help="SDF subgrid（默认 6）")
    ap.add_argument("--sdf-band", type=float, default=0.01, help="SDF 窄带厚度（默认 0.01）")
    ap.add_argument("--sdf-margin", type=float, default=0.01, help="SDF margin（默认 0.01）")
    ap.add_argument("--sdf-remesh", action="store_true", help="SDF 启用重网格（如可用）")
    ap.add_argument("--sdf-tri-reduce", type=float, default=1.0, help="SDF 三角面数缩减系数（如可用）")

    ap.add_argument("--vhacd-max-hulls", type=int, default=64)
    ap.add_argument("--vhacd-max-verts-per-hull", type=int, default=64)
    ap.add_argument("--vhacd-resolution", type=int, default=100_000)

    ap.add_argument("--contact-offset", type=float, default=0.01, help="接触偏移（默认 0.01）")
    ap.add_argument("--rest-offset", type=float, default=0.0, help="静止偏移（默认 0.0）")
    ap.add_argument("--pos-iters", type=int, default=8, help="位置迭代次数（默认 8）")
    ap.add_argument("--vel-iters", type=int, default=1, help="速度迭代次数（默认 1）")

    return ap.parse_args()

args = parse_args()
sim = build_sim(headless=args.headless)

import omni.usd
import omni.kit.commands
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade


def load_rules(path):
    if not path:
        return {"defaults": {}, "rules": []}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data.setdefault("defaults", {})
    data.setdefault("rules", [])
    return data

def load_materials(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "materials" not in data or not isinstance(data["materials"], dict):
        raise ValueError("materials.json 缺少根键 'materials' 或其类型不是对象")
    return data

RULES = load_rules(args.config)
MATS = load_materials(args.material_file)
MATERIAL_PRESETS = {k.lower(): v for k, v in MATS["materials"].items()}



def ensure_parent_xform_for_mesh(stage, mesh_prim):
    parent = mesh_prim.GetParent()
    if parent and parent.IsA(UsdGeom.Xform):
        return parent, mesh_prim

    parent_path = mesh_prim.GetPath().GetParentPath()
    xf_path = parent_path.AppendChild(mesh_prim.GetName() + "_Xform")
    UsdGeom.Xform.Define(stage, xf_path)

    new_mesh_path = xf_path.AppendChild(mesh_prim.GetName())
    omni.kit.commands.execute(
        "MovePrim", path_from=str(mesh_prim.GetPath()), path_to=str(new_mesh_path)
    )
    return stage.GetPrimAtPath(xf_path), stage.GetPrimAtPath(new_mesh_path)

def ensure_velocity_attrs_on_xform(xf_prim, use_physx_names=False):
    for n in ("physics:velocity","physics:angularVelocity","physx:linearVelocity","physx:angularVelocity"):
        if xf_prim.HasProperty(n):
            xf_prim.RemoveProperty(n)
    if use_physx_names:
        v  = xf_prim.CreateAttribute("physx:linearVelocity",  Sdf.ValueTypeNames.Float3)
        av = xf_prim.CreateAttribute("physx:angularVelocity", Sdf.ValueTypeNames.Float3)
    else:
        v  = xf_prim.CreateAttribute("physics:velocity",        Sdf.ValueTypeNames.Float3)
        av = xf_prim.CreateAttribute("physics:angularVelocity", Sdf.ValueTypeNames.Float3)
    v.Set(Gf.Vec3f(0,0,0)); av.Set(Gf.Vec3f(0,0,0))

def find_usd_files(root):
    if os.path.isfile(root):
        return [os.path.abspath(root)] if root.lower().endswith((".usd", ".usda", ".usdc")) else []
    out = []
    for r, _, fs in os.walk(root):
        for n in fs:
            if n.lower().endswith((".usd", ".usda", ".usdc")):
                out.append(os.path.join(r, n))
    return sorted(out)

def xform_has_geom(prim):
    for p in Usd.PrimRange(prim):
        if p.IsA(UsdGeom.Mesh) or p.IsA(UsdGeom.Cube) or p.IsA(UsdGeom.Sphere) or p.IsA(UsdGeom.Capsule):
            return True
    return False

def deinstance_visible_subtree(root_prim):
    """Make instanceable CAD assemblies editable before authoring physics."""
    changed = 0
    while True:
        pass_changed = 0
        for prim in list(Usd.PrimRange(root_prim)):
            if prim.IsInstanceable():
                prim.SetInstanceable(False)
                pass_changed += 1
        changed += pass_changed
        if pass_changed == 0:
            return changed

def _path_is_under(path, parent_path):
    path_str = path.pathString
    parent_str = parent_path.pathString
    return path_str == parent_str or path_str.startswith(parent_str + "/")

def cleanup_exported_stage(out_path, prune_prototypes=False):
    stage = Usd.Stage.Open(out_path, Usd.Stage.LoadAll)
    if not stage:
        return

    changed = False
    for path in (
        Sdf.Path("/Render"),
        Sdf.Path("/OmniverseKit_Persp"),
        Sdf.Path("/OmniverseKit_Front"),
        Sdf.Path("/OmniverseKit_Top"),
        Sdf.Path("/OmniverseKit_Right"),
    ):
        if stage.GetPrimAtPath(path).IsValid():
            stage.RemovePrim(path)
            changed = True

    if prune_prototypes:
        proto_paths = [
            prim.GetPath()
            for prim in stage.TraverseAll()
            if prim.GetName() == "Prototypes"
        ]
        if proto_paths:
            kept_meshes = sum(
                1
                for prim in stage.TraverseAll()
                if prim.IsA(UsdGeom.Mesh)
                and not any(_path_is_under(prim.GetPath(), proto) for proto in proto_paths)
            )
            if kept_meshes > 0:
                for proto_path in sorted(proto_paths, key=lambda p: len(p.pathString), reverse=True):
                    stage.RemovePrim(proto_path)
                remaining_meshes = sum(1 for prim in stage.TraverseAll() if prim.IsA(UsdGeom.Mesh))
                if remaining_meshes == kept_meshes:
                    log(f"  cleaned CAD Prototypes scope(s); kept {remaining_meshes} visible mesh(es)")
                    changed = True
                else:
                    log("  [WARN] skipped Prototypes cleanup because visible mesh count changed unexpectedly")

    if changed:
        stage.Export(out_path)
        log(f"  cleaned exported stage: {out_path}")

def _scale_vec3_array(values, factor):
    if not values:
        return values
    vec_type = type(values[0])
    return [vec_type(float(v[0]) * factor, float(v[1]) * factor, float(v[2]) * factor) for v in values]

def _scale_vec3_value(value, factor):
    vec_type = type(value)
    return vec_type(float(value[0]) * factor, float(value[1]) * factor, float(value[2]) * factor)

def normalize_stage_units_to_meters(stage):
    try:
        current_mpu = float(UsdGeom.GetStageMetersPerUnit(stage))
    except Exception:
        current_mpu = 1.0

    target_mpu = 1.0
    if abs(current_mpu - target_mpu) < 1e-12:
        return False

    factor = current_mpu / target_mpu
    scaled_points = 0
    scaled_xforms = 0
    scaled_prims = 0

    for prim in stage.TraverseAll():
        if prim.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(prim)
            points_attr = mesh.GetPointsAttr()
            points = points_attr.Get()
            if points:
                points_attr.Set(_scale_vec3_array(points, factor))
                scaled_points += 1

        if prim.IsA(UsdGeom.Boundable):
            extent_attr = UsdGeom.Boundable(prim).GetExtentAttr()
            extent = extent_attr.Get()
            if extent:
                extent_attr.Set(_scale_vec3_array(extent, factor))

        if prim.IsA(UsdGeom.Cube):
            cube = UsdGeom.Cube(prim)
            size = cube.GetSizeAttr().Get()
            if size is not None:
                cube.GetSizeAttr().Set(float(size) * factor)
                scaled_prims += 1
        elif prim.IsA(UsdGeom.Sphere):
            sphere = UsdGeom.Sphere(prim)
            radius = sphere.GetRadiusAttr().Get()
            if radius is not None:
                sphere.GetRadiusAttr().Set(float(radius) * factor)
                scaled_prims += 1
        elif prim.IsA(UsdGeom.Capsule):
            capsule = UsdGeom.Capsule(prim)
            radius = capsule.GetRadiusAttr().Get()
            height = capsule.GetHeightAttr().Get()
            if radius is not None:
                capsule.GetRadiusAttr().Set(float(radius) * factor)
            if height is not None:
                capsule.GetHeightAttr().Set(float(height) * factor)
            scaled_prims += 1
        elif prim.IsA(UsdGeom.Cylinder):
            cylinder = UsdGeom.Cylinder(prim)
            radius = cylinder.GetRadiusAttr().Get()
            height = cylinder.GetHeightAttr().Get()
            if radius is not None:
                cylinder.GetRadiusAttr().Set(float(radius) * factor)
            if height is not None:
                cylinder.GetHeightAttr().Set(float(height) * factor)
            scaled_prims += 1
        elif prim.IsA(UsdGeom.Cone):
            cone = UsdGeom.Cone(prim)
            radius = cone.GetRadiusAttr().Get()
            height = cone.GetHeightAttr().Get()
            if radius is not None:
                cone.GetRadiusAttr().Set(float(radius) * factor)
            if height is not None:
                cone.GetHeightAttr().Set(float(height) * factor)
            scaled_prims += 1

        if prim.IsA(UsdGeom.Xformable):
            xformable = UsdGeom.Xformable(prim)
            for op in xformable.GetOrderedXformOps():
                try:
                    op_type = op.GetOpType()
                    value = op.Get()
                    if value is None:
                        continue
                    if op_type == UsdGeom.XformOp.TypeTranslate:
                        op.Set(_scale_vec3_value(value, factor))
                        scaled_xforms += 1
                    elif op_type == UsdGeom.XformOp.TypeTransform:
                        translation = value.ExtractTranslation()
                        value.SetTranslateOnly(_scale_vec3_value(translation, factor))
                        op.Set(value)
                        scaled_xforms += 1
                except Exception as exc:
                    log(f"  [WARN] failed to scale xform op on {prim.GetPath().pathString}: {exc}")

    UsdGeom.SetStageMetersPerUnit(stage, target_mpu)
    log(
        f"  normalized stage units: metersPerUnit {current_mpu:g} -> {target_mpu:g}, "
        f"scale={factor:g}, mesh_points={scaled_points}, xform_ops={scaled_xforms}, primitives={scaled_prims}"
    )
    return True

def center_stage_geometry_at_origin(stage, anchor_prim):
    sank = _sink_root_center_origin_to_geometry(anchor_prim)
    if sank:
        log("  center-origin: moved existing root offset down to geometry child")

    bbox = compute_world_aabb(stage, anchor_prim)
    if not bbox:
        log("  [WARN] skipped center-origin: failed to compute world bounds")
        return False

    center = bbox.GetMin() + (bbox.GetSize() * 0.5)
    offset = Gf.Vec3d(-float(center[0]), -float(center[1]), -float(center[2]))
    if max(abs(offset[0]), abs(offset[1]), abs(offset[2])) < 1e-9:
        log("  center-origin: asset already centered")
        return False

    _translate_anchor_geometry(anchor_prim, offset)

    new_bbox = compute_world_aabb(stage, anchor_prim)
    if new_bbox:
        new_center = new_bbox.GetMin() + (new_bbox.GetSize() * 0.5)
        log(
            "  center-origin: "
            f"offset=({offset[0]:.6g}, {offset[1]:.6g}, {offset[2]:.6g}), "
            f"new_center=({new_center[0]:.6g}, {new_center[1]:.6g}, {new_center[2]:.6g})"
        )
    else:
        log(
            "  center-origin: "
            f"offset=({offset[0]:.6g}, {offset[1]:.6g}, {offset[2]:.6g})"
        )
    return True

def _sink_root_center_origin_to_geometry(anchor_prim):
    """Keep the top asset Xform at zero by moving old root offsets below it."""
    if anchor_prim.IsA(UsdGeom.Mesh) or not anchor_prim.IsA(UsdGeom.Xformable):
        return False

    xformable = UsdGeom.Xformable(anchor_prim)
    for op in xformable.GetOrderedXformOps():
        if op.GetOpName() != "xformOp:translate:centerOrigin":
            continue
        current = op.Get() or Gf.Vec3d(0.0, 0.0, 0.0)
        offset = Gf.Vec3d(float(current[0]), float(current[1]), float(current[2]))
        if max(abs(offset[0]), abs(offset[1]), abs(offset[2])) > 1e-12:
            _translate_anchor_geometry(anchor_prim, offset)
            op.Set(Gf.Vec3d(0.0, 0.0, 0.0))
            return True
        op.Set(Gf.Vec3d(0.0, 0.0, 0.0))
        return False
    return False

def _translate_anchor_geometry(anchor_prim, offset):
    """Move geometry under the asset root while keeping the root transform zero."""
    if anchor_prim.IsA(UsdGeom.Mesh):
        _translate_mesh_points(UsdGeom.Mesh(anchor_prim), offset)
        return

    moved = 0
    for child in anchor_prim.GetChildren():
        if child.IsA(UsdGeom.Mesh) or xform_has_geom(child):
            if _translate_prim_as_unit(child, offset):
                moved += 1

    if moved == 0:
        log("  [WARN] center-origin: no geometry child found; asset root was not translated")

def _translate_prim_as_unit(prim, offset):
    if prim.IsA(UsdGeom.Mesh):
        _translate_mesh_points(UsdGeom.Mesh(prim), offset)
        return True

    if not prim.IsA(UsdGeom.Xformable):
        return False

    xformable = UsdGeom.Xformable(prim)
    op = None
    for candidate in xformable.GetOrderedXformOps():
        if candidate.GetOpName() == "xformOp:translate:centerOrigin":
            op = candidate
            break
    if op is None:
        op = xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble, "centerOrigin")
    _put_xform_op_first(xformable, op)

    current = op.Get()
    if current is None:
        current = Gf.Vec3d(0.0, 0.0, 0.0)
    op.Set(Gf.Vec3d(
        float(current[0]) + float(offset[0]),
        float(current[1]) + float(offset[1]),
        float(current[2]) + float(offset[2]),
    ))
    return True

def _put_xform_op_first(xformable, op):
    ordered = list(xformable.GetOrderedXformOps())
    reordered = [op] + [candidate for candidate in ordered if candidate.GetOpName() != op.GetOpName()]
    try:
        reset_stack = bool(xformable.GetResetXformStack())
    except Exception:
        reset_stack = False
    try:
        xformable.SetXformOpOrder(reordered, reset_stack)
    except TypeError:
        xformable.SetXformOpOrder(reordered)

def _translate_mesh_points(mesh, offset):
    points_attr = mesh.GetPointsAttr()
    points = points_attr.Get()
    if not points:
        return
    points_attr.Set([
        type(point)(
            float(point[0]) + float(offset[0]),
            float(point[1]) + float(offset[1]),
            float(point[2]) + float(offset[2]),
        )
        for point in points
    ])

def _offset_vec3_array(values, offset):
    if not values:
        return values
    return [
        type(value)(
            float(value[0]) + float(offset[0]),
            float(value[1]) + float(offset[1]),
            float(value[2]) + float(offset[2]),
        )
        for value in values
    ]

def _points_bbox_center(points):
    min_x = min(float(point[0]) for point in points)
    min_y = min(float(point[1]) for point in points)
    min_z = min(float(point[2]) for point in points)
    max_x = max(float(point[0]) for point in points)
    max_y = max(float(point[1]) for point in points)
    max_z = max(float(point[2]) for point in points)
    return Gf.Vec3d(
        (min_x + max_x) * 0.5,
        (min_y + max_y) * 0.5,
        (min_z + max_z) * 0.5,
    )

def center_mesh_local_origins(root_prim):
    """Move mesh points near their own local origins and compensate with xform ops."""
    centered = 0
    skipped = 0
    for prim in Usd.PrimRange(root_prim):
        if not prim.IsA(UsdGeom.Mesh):
            continue

        xformable = UsdGeom.Xformable(prim)
        if xformable.GetOrderedXformOps():
            skipped += 1
            continue

        mesh = UsdGeom.Mesh(prim)
        points_attr = mesh.GetPointsAttr()
        points = points_attr.Get()
        if not points:
            continue

        center = _points_bbox_center(points)
        if max(abs(center[0]), abs(center[1]), abs(center[2])) < 1e-12:
            continue

        negative_center = Gf.Vec3d(-center[0], -center[1], -center[2])
        points_attr.Set(_offset_vec3_array(points, negative_center))

        extent_attr = UsdGeom.Boundable(prim).GetExtentAttr()
        extent = extent_attr.Get()
        if extent:
            extent_attr.Set(_offset_vec3_array(extent, negative_center))

        xformable.AddTranslateOp(
            UsdGeom.XformOp.PrecisionDouble,
            "meshLocalOrigin",
        ).Set(center)
        centered += 1

    if centered:
        log(f"  centered mesh local origins: {centered} mesh(es)")
    if skipped:
        log(f"  mesh local origin centering skipped meshes with existing xform ops: {skipped}")
    return centered

def compute_world_aabb(stage, prim):
    try:
        purposes = [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy, UsdGeom.Tokens.guide]
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), purposes, useExtentsHint=False)
        bbox = cache.ComputeWorldBound(prim)
        return bbox.ComputeAlignedBox()
    except Exception:
        return None

def get_anchor_root(stage):
    dp = stage.GetDefaultPrim()
    if dp and dp.IsValid():
        return dp
    w = stage.GetPrimAtPath("/World")
    if w and w.IsValid():
        return w
    for prim in stage.GetPseudoRoot().GetChildren():
        if prim.IsA(UsdGeom.Xform) or prim.IsA(UsdGeom.Scope):
            return prim
    return stage.GetPseudoRoot()

def ensure_material_under_anchor(
    stage, anchor_prim, base_name, static_friction, restitution, combine=None, dyn_ratio=0.9
):
    mats_root = anchor_prim.GetPath().AppendChild("PhysicsMaterials")
    if not stage.GetPrimAtPath(mats_root):
        UsdGeom.Xform.Define(stage, mats_root)
    mat_path = mats_root.AppendChild(f"{base_name}_PhysMat")
    material = UsdShade.Material.Define(stage, mat_path)

    pmat = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    pmat.CreateStaticFrictionAttr(float(static_friction))
    pmat.CreateDynamicFrictionAttr(float(static_friction) * float(dyn_ratio))
    pmat.CreateRestitutionAttr(float(restitution))

    try:
        xmat = PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim())
        xmat.CreateStaticFrictionAttr(float(static_friction))
        xmat.CreateDynamicFrictionAttr(float(static_friction) * float(dyn_ratio))
        xmat.CreateRestitutionAttr(float(restitution))
        if combine:
            try:
                xmat.CreateFrictionCombineModeAttr(str(combine))
                xmat.CreateRestitutionCombineModeAttr(str(combine))
            except Exception:
                pass
    except Exception:
        pass
    return material

def _triangulate_counts_indices(counts, indices):
    tris = []
    i = 0
    for c in counts or []:
        if c >= 3:
            face = indices[i: i + c]
            for k in range(1, c - 1):
                tris.append((face[0], face[k], face[k + 1]))
        i += c
    return tris

def _mesh_signed_volume_units3(mesh, xform=None):
    pts = mesh.GetPointsAttr().Get()
    if not pts:
        return 0.0
    counts = mesh.GetFaceVertexCountsAttr().Get() or []
    idx = mesh.GetFaceVertexIndicesAttr().Get() or []
    tris = _triangulate_counts_indices(counts, idx)
    if not tris:
        return 0.0

    if xform is not None:
        points = [xform.Transform(Gf.Vec3d(float(p[0]), float(p[1]), float(p[2]))) for p in pts]
    else:
        points = [Gf.Vec3d(float(p[0]), float(p[1]), float(p[2])) for p in pts]

    vol = 0.0
    for i0, i1, i2 in tris:
        p0 = points[i0]
        p1 = points[i1]
        p2 = points[i2]
        vol += Gf.Dot(p0, Gf.Cross(p1, p2))
    return float(vol) / 6.0

def _mesh_volume_epsilon(mesh, xform=None):
    pts = mesh.GetPointsAttr().Get()
    if not pts:
        return 1e-15
    if xform is not None:
        points = [xform.Transform(Gf.Vec3d(float(p[0]), float(p[1]), float(p[2]))) for p in pts]
    else:
        points = [Gf.Vec3d(float(p[0]), float(p[1]), float(p[2])) for p in pts]
    min_x = min(float(p[0]) for p in points)
    min_y = min(float(p[1]) for p in points)
    min_z = min(float(p[2]) for p in points)
    max_x = max(float(p[0]) for p in points)
    max_y = max(float(p[1]) for p in points)
    max_z = max(float(p[2]) for p in points)
    diag = math.sqrt((max_x - min_x) ** 2 + (max_y - min_y) ** 2 + (max_z - min_z) ** 2)
    return max((diag ** 3) * 1e-12, 1e-15)

def _reverse_mesh_face_winding(mesh):
    counts = list(mesh.GetFaceVertexCountsAttr().Get() or [])
    indices = list(mesh.GetFaceVertexIndicesAttr().Get() or [])
    if not counts or not indices:
        return False

    reversed_indices = []
    cursor = 0
    for count in counts:
        face = indices[cursor:cursor + count]
        reversed_indices.extend(reversed(face))
        cursor += count
    if len(reversed_indices) != len(indices):
        return False

    mesh.GetFaceVertexIndicesAttr().Set(reversed_indices)
    mesh.CreateOrientationAttr(UsdGeom.Tokens.rightHanded)
    try:
        mesh.GetNormalsAttr().Clear()
    except Exception:
        pass
    try:
        mesh.CreateDoubleSidedAttr(True)
    except Exception:
        pass
    return True

def _mesh_topology_status(mesh):
    """Return boundary, non-manifold, and inconsistent shared-edge counts."""
    counts = list(mesh.GetFaceVertexCountsAttr().Get() or [])
    indices = list(mesh.GetFaceVertexIndicesAttr().Get() or [])
    edge_uses = {}
    cursor = 0
    invalid_faces = 0

    for count in counts:
        face = indices[cursor:cursor + count]
        cursor += count
        if count < 3 or len(face) != count:
            invalid_faces += 1
            continue
        for index, start in enumerate(face):
            end = face[(index + 1) % count]
            if start == end:
                invalid_faces += 1
                continue
            key = (min(start, end), max(start, end))
            direction = 1 if start < end else -1
            edge_uses.setdefault(key, []).append(direction)

    boundary_edges = sum(len(uses) == 1 for uses in edge_uses.values())
    nonmanifold_edges = sum(len(uses) > 2 for uses in edge_uses.values())
    inconsistent_edges = sum(
        len(uses) == 2 and uses[0] == uses[1]
        for uses in edge_uses.values()
    )
    return {
        "boundary_edges": boundary_edges,
        "nonmanifold_edges": nonmanifold_edges,
        "inconsistent_edges": inconsistent_edges,
        "invalid_faces": invalid_faces,
        "watertight": bool(edge_uses)
        and boundary_edges == 0
        and nonmanifold_edges == 0
        and inconsistent_edges == 0
        and invalid_faces == 0,
    }

def _mesh_topology_problem(mesh):
    status = _mesh_topology_status(mesh)
    if status["watertight"]:
        return None
    return (
        f"boundary={status['boundary_edges']}, "
        f"nonmanifold={status['nonmanifold_edges']}, "
        f"inconsistent={status['inconsistent_edges']}, "
        f"invalid_faces={status['invalid_faces']}"
    )

def _mesh_sdf_problem(mesh):
    topology_problem = _mesh_topology_problem(mesh)
    if topology_problem:
        return topology_problem
    signed_volume = _mesh_signed_volume_units3(mesh)
    if abs(signed_volume) <= _mesh_volume_epsilon(mesh):
        return "closed volume is too small"
    return None

def fix_inverted_mesh_winding(stage, root_prim):
    """Make closed CAD meshes outward-facing after hierarchy transforms."""
    fixed = 0
    skipped = 0
    xcache = UsdGeom.XformCache(Usd.TimeCode.Default())
    for prim in Usd.PrimRange(root_prim):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)

        # USD's leftHanded token changes the interpreted front face without
        # changing indices. Normalize it before evaluating the actual winding.
        if mesh.GetOrientationAttr().Get() == UsdGeom.Tokens.leftHanded:
            mesh.CreateOrientationAttr(UsdGeom.Tokens.rightHanded)
            try:
                mesh.GetNormalsAttr().Clear()
            except Exception:
                pass

        topology_problem = _mesh_topology_problem(mesh)
        if topology_problem:
            skipped += 1
            log(
                f"  [WARN] winding volume check skipped for {prim.GetPath().pathString}: "
                f"{topology_problem}"
            )
            continue

        world_xform = xcache.GetLocalToWorldTransform(prim)
        signed_volume = _mesh_signed_volume_units3(mesh, world_xform)
        if abs(signed_volume) <= _mesh_volume_epsilon(mesh, world_xform):
            skipped += 1
            log(
                f"  [WARN] winding volume check skipped for {prim.GetPath().pathString}: "
                "closed volume is too small"
            )
            continue
        if signed_volume < 0.0 and _reverse_mesh_face_winding(mesh):
            fixed += 1
    if fixed:
        log(f"  fixed inverted world-space mesh winding: {fixed} mesh(es)")
    if skipped:
        log(f"  mesh winding volume check skipped unsafe meshes: {skipped}")
    return fixed

def prepare_geometry_for_physics(stage, anchor_prim, center_origin=False):
    """Normalize and clean imported geometry before authoring PhysX schemas."""
    normalize_stage_units_to_meters(stage)

    deinstanced = deinstance_visible_subtree(anchor_prim)
    if deinstanced:
        log(f"  de-instanced {deinstanced} instanceable prim(s) for editable CAD hierarchy")

    if center_origin:
        center_stage_geometry_at_origin(stage, anchor_prim)

    center_mesh_local_origins(anchor_prim)
    fix_inverted_mesh_winding(stage, anchor_prim)
    return {"deinstanced": deinstanced}

def _mat3_det_from_mat4(m4):
    a00, a01, a02 = m4[0][0], m4[0][1], m4[0][2]
    a10, a11, a12 = m4[1][0], m4[1][1], m4[1][2]
    a20, a21, a22 = m4[2][0], m4[2][1], m4[2][2]
    return (a00 * (a11 * a22 - a12 * a21)
            - a01 * (a10 * a22 - a12 * a20)
            + a02 * (a10 * a21 - a11 * a20))

def _mesh_local_volume_units3(mesh):
    return abs(_mesh_signed_volume_units3(mesh))

def _meters_per_unit(stage):
    try:
        return float(UsdGeom.GetStageMetersPerUnit(stage))
    except Exception:
        return 1.0

def compute_precise_volume_m3(stage, prim):
    mpu = _meters_per_unit(stage)
    xcache = UsdGeom.XformCache(Usd.TimeCode.Default())
    vol_units3 = 0.0
    for p in Usd.PrimRange(prim):
        if not p.IsValid():
            continue
        det = abs(_mat3_det_from_mat4(xcache.GetLocalToWorldTransform(p)))
        if det == 0.0:
            continue
        if p.IsA(UsdGeom.Mesh):
            v_local = _mesh_local_volume_units3(UsdGeom.Mesh(p))
            vol_units3 += v_local * det
        elif p.IsA(UsdGeom.Cube):
            size = float(UsdGeom.Cube(p).GetSizeAttr().Get() or 0.0)
            vol_units3 += (size ** 3) * det
        elif p.IsA(UsdGeom.Sphere):
            r = float(UsdGeom.Sphere(p).GetRadiusAttr().Get() or 0.0)
            vol_units3 += ((4.0 / 3.0) * math.pi * (r ** 3)) * det
        elif p.IsA(UsdGeom.Capsule):
            cap = UsdGeom.Capsule(p)
            r = float(cap.GetRadiusAttr().Get() or 0.0)
            h = float(cap.GetHeightAttr().Get() or 0.0)
            vol_units3 += ((math.pi * r * r * h) + (4.0 / 3.0) * math.pi * (r ** 3)) * det
    vol_m3 = vol_units3 * (mpu ** 3)
    log(f"Total Volume: {vol_m3:.6e} m^3")
    return float(vol_m3)

def estimate_mass_by_density_precise(stage, prim, density, fallback_mass=1.0):
    vol_m3 = compute_precise_volume_m3(stage, prim)
    if vol_m3 and vol_m3 > 0.0:
        m = density * vol_m3
    else:
        rng = compute_world_aabb(stage, prim)
        if not rng:
            return float(fallback_mass)
        size = rng.GetSize()
        mpu = _meters_per_unit(stage)
        vol_m3_fb = max(size[0], 1e-9) * max(size[1], 1e-9) * max(size[2], 1e-9) * (mpu ** 3)
        m = density * vol_m3_fb
    return float(min(max(m, 0.01), 10000.0))

def compute_mass_distribution_weight(stage, prim, density):
    vol_m3 = compute_precise_volume_m3(stage, prim)
    if not vol_m3 or vol_m3 <= 0.0:
        rng = compute_world_aabb(stage, prim)
        if rng:
            size = rng.GetSize()
            mpu = _meters_per_unit(stage)
            vol_m3 = (
                max(size[0], 1e-9)
                * max(size[1], 1e-9)
                * max(size[2], 1e-9)
                * (mpu ** 3)
            )
    if not vol_m3 or vol_m3 <= 0.0:
        return 1.0
    return max(float(vol_m3) * max(float(density), 1e-9), 1e-9)

def distribute_total_mass(stage, records, total_mass):
    rigid_records = [
        record
        for record in records
        if record["params"].get("body_kind", "rigid") == "rigid"
    ]
    if not rigid_records:
        return {}

    total_mass = float(total_mass)
    if total_mass <= 0.0:
        raise ValueError("--set-mass must be greater than 0 when provided")

    weights = [
        compute_mass_distribution_weight(
            stage,
            record["rb_xform"],
            float(record["params"].get("density", 500.0)),
        )
        for record in rigid_records
    ]
    total_weight = sum(weights)
    if total_weight <= 0.0:
        weights = [1.0 for _ in rigid_records]
        total_weight = float(len(rigid_records))

    masses = {}
    assigned = 0.0
    for index, (record, weight) in enumerate(zip(rigid_records, weights)):
        if index == len(rigid_records) - 1:
            mass = max(total_mass - assigned, 0.0)
        else:
            mass = total_mass * (weight / total_weight)
            assigned += mass
        masses[record["rb_xform"].GetPath().pathString] = float(mass)
    return masses

def bind_mat_to_body_and_meshes(stage, body_prim, material):
    UsdShade.MaterialBindingAPI.Apply(body_prim).Bind(
        material, UsdShade.Tokens.weakerThanDescendants, "physics"
    )
    for p in Usd.PrimRange(body_prim):
        if p.IsA(UsdGeom.Mesh):
            UsdShade.MaterialBindingAPI.Apply(p).Bind(
                material, UsdShade.Tokens.weakerThanDescendants, "physics"
            )

def normalize_mesh_approx(name):
    if not name:
        return "convexHull"
    n = str(name).strip().lower()
    alias = {
        "convex": "convexHull",
        "convexdecomp": "convexDecomposition",
        "convexdecomposition": "convexDecomposition",
        "vhacd": "convexDecomposition",
        "box": "boundingCube", "cube": "boundingCube",
        "sphere": "boundingSphere",
        "trianglemesh": "triangleMesh", "mesh": "triangleMesh",
        "meshsimplification": "meshSimplification",
        "sdf": "sdf",
        "sphereapproximation": "sphereApproximation",
    }
    return alias.get(n, name)

def approx_suggests_static(approx):
    return approx in ("triangleMesh", "meshSimplification")

def _set_approx_token(mesh_prim, token_str):
    attr = mesh_prim.CreateAttribute("physics:approximation", Sdf.ValueTypeNames.Token)
    attr.Set(token_str)

def _author_sdf_on_mesh(mesh_prim, params):
    _set_approx_token(mesh_prim, "sdf")
    UsdPhysics.CollisionAPI.Apply(mesh_prim)
    UsdPhysics.MeshCollisionAPI.Apply(mesh_prim)

    sdf_api = PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(mesh_prim)
    if hasattr(sdf_api, "CreateSdfResolutionAttr"):
        sdf_api.CreateSdfResolutionAttr(int(max(1, params.get("sdf_res", args.sdf_res))))
    if hasattr(sdf_api, "CreateSdfSubgridResolutionAttr"):
        sdf_api.CreateSdfSubgridResolutionAttr(int(params.get("sdf_subgrid", args.sdf_subgrid)))
    if hasattr(sdf_api, "CreateSdfNarrowBandThicknessAttr"):
        sdf_api.CreateSdfNarrowBandThicknessAttr(float(params.get("sdf_band", args.sdf_band)))
    if hasattr(sdf_api, "CreateSdfMarginAttr"):
        sdf_api.CreateSdfMarginAttr(float(params.get("sdf_margin", args.sdf_margin)))
    sdf_remesh = bool(params.get("sdf_remesh", args.sdf_remesh))
    if hasattr(sdf_api, "CreateSdfEnableRemeshingAttr"):
        sdf_api.CreateSdfEnableRemeshingAttr(sdf_remesh)
        if sdf_remesh:
            log(f"  SDF remeshing enabled on {mesh_prim.GetPath().pathString}")
    elif sdf_remesh:
        log(
            f"  [WARN] SDF remeshing requested but unsupported on "
            f"{mesh_prim.GetPath().pathString}"
        )
    if "sdf_tri_reduce" in params and hasattr(sdf_api, "CreateSdfTriangleCountReductionFactorAttr"):
        sdf_api.CreateSdfTriangleCountReductionFactorAttr(float(params["sdf_tri_reduce"]))

    col = PhysxSchema.PhysxCollisionAPI.Apply(mesh_prim)
    if hasattr(col, "CreateContactOffsetAttr"):
        col.CreateContactOffsetAttr(float(params.get("contact_offset", args.contact_offset)))
    if hasattr(col, "CreateRestOffsetAttr"):
        col.CreateRestOffsetAttr(float(params.get("rest_offset", args.rest_offset)))

def _author_convex_decomposition(mesh_prim):
    _set_approx_token(mesh_prim, "convexDecomposition")
    UsdPhysics.CollisionAPI.Apply(mesh_prim)
    UsdPhysics.MeshCollisionAPI.Apply(mesh_prim)
    try:
        decomp = PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(mesh_prim)
        if hasattr(decomp, "CreateMaxConvexHullsAttr"):
            decomp.CreateMaxConvexHullsAttr(int(args.vhacd_max_hulls))
        if hasattr(decomp, "CreateMaxHullVerticesAttr"):
            decomp.CreateMaxHullVerticesAttr(int(args.vhacd_max_verts_per_hull))
        if hasattr(decomp, "CreateResolutionAttr"):
            decomp.CreateResolutionAttr(int(args.vhacd_resolution))
    except Exception:
        pass
    col = PhysxSchema.PhysxCollisionAPI.Apply(mesh_prim)
    if hasattr(col, "CreateContactOffsetAttr"): col.CreateContactOffsetAttr(float(args.contact_offset))
    if hasattr(col, "CreateRestOffsetAttr"):    col.CreateRestOffsetAttr(float(args.rest_offset))

def add_rigid_and_collider(stage, body_prim, params):
    req_approx = normalize_mesh_approx(params.get("approx", args.approx or "convexHull"))
    body_kind = params.get("body_kind", "rigid")
    force_sdf = bool(params.get("force_sdf", args.force_sdf))

    if body_kind == "rigid" and (approx_suggests_static(req_approx) or force_sdf or req_approx=="sdf"):
        req_approx = "sdf"

    def _author_mesh(mesh_prim):
        mesh_approx = req_approx
        if mesh_approx == "sdf":
            sdf_problem = _mesh_sdf_problem(UsdGeom.Mesh(mesh_prim))
            if sdf_problem:
                log(
                    f"  [WARN] SDF topology warning on {mesh_prim.GetPath().pathString}: "
                    f"{sdf_problem}; keeping requested sdf"
                )

        if mesh_approx == "sdf":
            _author_sdf_on_mesh(mesh_prim, params)
            used = "sdf"
        elif mesh_approx == "convexDecomposition":
            _author_convex_decomposition(mesh_prim)
            used = "convexDecomposition"
        elif mesh_approx in ("convexHull", "boundingCube", "boundingSphere", "sphereApproximation"):
            _set_approx_token(mesh_prim, mesh_approx)
            UsdPhysics.CollisionAPI.Apply(mesh_prim)
            UsdPhysics.MeshCollisionAPI.Apply(mesh_prim)
            col = PhysxSchema.PhysxCollisionAPI.Apply(mesh_prim)
            if hasattr(col, "CreateContactOffsetAttr"): col.CreateContactOffsetAttr(float(params.get("contact_offset", args.contact_offset)))
            if hasattr(col, "CreateRestOffsetAttr"):    col.CreateRestOffsetAttr(float(params.get("rest_offset", args.rest_offset)))
            used = mesh_approx
        else:
            _author_sdf_on_mesh(mesh_prim, params)
            used = "sdf"
        log(f"  collider on {mesh_prim.GetPath().pathString}: approx={used}")
        return used

    used_rb_xform = None

    if body_prim.IsA(UsdGeom.Mesh):
        rb_xform, mesh_fixed = ensure_parent_xform_for_mesh(stage, body_prim)
        used_rb_xform = rb_xform
        if body_kind == "rigid":
            UsdPhysics.RigidBodyAPI.Apply(rb_xform)
            try: PhysxSchema.PhysxRigidBodyAPI.Apply(rb_xform)
            except Exception: pass
            ensure_velocity_attrs_on_xform(rb_xform, use_physx_names=False)
        _author_mesh(mesh_fixed)

    elif body_prim.IsA(UsdGeom.Xform) and xform_has_geom(body_prim):
        used_rb_xform = body_prim
        if body_kind == "rigid":
            UsdPhysics.RigidBodyAPI.Apply(body_prim)
            try: PhysxSchema.PhysxRigidBodyAPI.Apply(body_prim)
            except Exception: pass
            ensure_velocity_attrs_on_xform(body_prim, use_physx_names=False)
        for p in Usd.PrimRange(body_prim):
            if p.IsA(UsdGeom.Mesh):
                _author_mesh(p)
    else:
        log(f"  [Skip] 非 Mesh/Xform 或无几何：{body_prim.GetPath().pathString}")
        return None

    return used_rb_xform

def normalize_tag(tag):
    return (str(tag).strip().lower() if tag else None)

def get_preset(tag):
    t = normalize_tag(tag)
    return MATERIAL_PRESETS.get(t) if t else None

def match_params(prim):
    params = dict(RULES.get("defaults", {}))
    ppath = prim.GetPath().pathString
    name = prim.GetName()

    for r in RULES.get("rules", []):
        hit = False
        if r.get("path_glob") and fnmatch.fnmatch(ppath, r["path_glob"]):
            hit = True
        if (not hit) and r.get("name_regex") and re.search(r["name_regex"], name, flags=re.IGNORECASE):
            hit = True
        if hit:
            for k, v in r.items():
                if k in ("path_glob", "name_regex"):
                    continue
                params[k] = v
            break

    if args.material:
        chosen = normalize_tag(args.material)
        preset = get_preset(chosen)
        if not preset:
            log(f"[WARN] --material '{args.material}' 不在 materials.json 中；可用标签：{list(MATERIAL_PRESETS.keys())}")
        else:
            for k in ("density", "friction", "restitution", "combine", "dyn_ratio"):
                if k in preset:
                    params[k] = preset[k]
            params["material_tag"] = chosen
    else:
        rule_tag = normalize_tag(params.get("material_tag"))
        if rule_tag:
            preset = get_preset(rule_tag)
            if preset:
                for k in ("density", "friction", "restitution", "combine", "dyn_ratio"):
                    if k in preset and k not in params:
                        params[k] = preset[k]
                params["material_tag"] = rule_tag

    if args.approx:
        params["approx"] = args.approx
    if args.force_sdf:
        params["force_sdf"] = True

    params.setdefault("density", 500.0)
    params.setdefault("friction", 0.6)
    params.setdefault("restitution", 0.1)
    params.setdefault("approx", "convexHull")
    params.setdefault("pos_iters", args.pos_iters)
    params.setdefault("vel_iters", args.vel_iters)
    params.setdefault("contact_offset", args.contact_offset)
    params.setdefault("rest_offset", args.rest_offset)
    params.setdefault("mass", None)
    params.setdefault("body_kind", "rigid")
    params.setdefault("dyn_ratio", 0.9)

    params.setdefault("sdf_res", args.sdf_res)
    params.setdefault("sdf_subgrid", args.sdf_subgrid)
    params.setdefault("sdf_band", args.sdf_band)
    params.setdefault("sdf_margin", args.sdf_margin)
    params.setdefault("sdf_remesh", args.sdf_remesh)
    params.setdefault("sdf_tri_reduce", args.sdf_tri_reduce)

    return params

def build_out_path(in_path):
    base_dir, fname = os.path.split(in_path)
    stem, ext = os.path.splitext(fname)
    out_name = f"{stem}{args.suffix}{ext}"
    if args.out_dir:
        rel_dir = "." if os.path.isfile(args.folder) else os.path.relpath(base_dir, args.folder)
        out_dir = os.path.normpath(os.path.join(args.out_dir, rel_dir))
    else:
        out_dir = base_dir
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, out_name)

def _set_solver_iters(rb_xform, pos_iters, vel_iters):
    try:
        prb = PhysxSchema.PhysxRigidBodyAPI.Apply(rb_xform)
        if hasattr(prb, "CreateSolverPositionIterationCountAttr"):
            prb.CreateSolverPositionIterationCountAttr(int(pos_iters))
        if hasattr(prb, "CreateSolverVelocityIterationCountAttr"):
            prb.CreateSolverVelocityIterationCountAttr(int(vel_iters))
    except Exception:
        pass

def process_usd(in_path):
    log(f"\n>>> Processing: {in_path}")
    ctx = omni.usd.get_context()
    ctx.open_stage(in_path)
    stage = ctx.get_stage()
    if not stage:
        log("  !! failed to open stage")
        return

    try:
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        log("  stage upAxis = Z")
    except Exception as e:
        log(f"  [WARN] failed to set stage upAxis=Z: {e}")

    anchor = get_anchor_root(stage)
    log("  anchor root =", anchor.GetPath().pathString)
    geometry_prep = prepare_geometry_for_physics(
        stage,
        anchor,
        center_origin=args.center_origin,
    )

    records = []
    for prim in list(anchor.GetChildren()):
        if prim.IsA(UsdGeom.Mesh) or prim.IsA(UsdGeom.Cube) or prim.IsA(UsdGeom.Sphere) or prim.IsA(UsdGeom.Capsule):
            body = prim
        elif prim.IsA(UsdGeom.Xform) and xform_has_geom(prim):
            body = prim
        else:
            continue

        params = match_params(body)
        rb_xform = add_rigid_and_collider(stage, body, params)
        if rb_xform is None:
            continue
        records.append({"body": body, "rb_xform": rb_xform, "params": params})

    fixed_masses = {}
    if args.set_mass is not None:
        fixed_masses = distribute_total_mass(stage, records, args.set_mass)
        log(
            f"  total mass={float(args.set_mass):.3f} kg distributed over "
            f"{len(fixed_masses)} rigid body/bodies"
        )

    for record in records:
        rb_xform = record["rb_xform"]
        params = record["params"]
        if params.get("body_kind", "rigid") == "rigid":
            if args.set_mass is not None:
                m = fixed_masses.get(rb_xform.GetPath().pathString, float(args.set_mass))
            elif params.get("mass") is not None:
                m = float(params["mass"])
            else:
                m = estimate_mass_by_density_precise(stage, rb_xform, float(params["density"]), fallback_mass=1.0)
            UsdPhysics.MassAPI.Apply(rb_xform).CreateMassAttr(float(m))
            _set_solver_iters(rb_xform, int(params["pos_iters"]), int(params["vel_iters"]))
            log(f"  mass={m:.3f} kg @ {rb_xform.GetPath().pathString}")
        else:
            log("  body_kind=static")

        material = ensure_material_under_anchor(
            stage,
            anchor,
            rb_xform.GetName(),
            float(params["friction"]),
            float(params["restitution"]),
            params.get("combine"),
            float(params.get("dyn_ratio", 0.9)),
        )
        bind_mat_to_body_and_meshes(stage, rb_xform, material)
        mu_s = float(params["friction"]); mu_d = mu_s * float(params.get("dyn_ratio", 0.9))
        tag = params.get("material_tag", "n/a")
        log(f"  material[{tag}] : {material.GetPath().pathString}  μs={mu_s:.3f}  μd≈{mu_d:.3f}  e={float(params['restitution']):.3f}")

    out_path = build_out_path(in_path)
    try:
        stage.Export(out_path)
        log(f"  ✓ exported: {out_path}")
        cleanup_exported_stage(out_path, prune_prototypes=bool(geometry_prep["deinstanced"]))
    except Exception as e:
        log(f"  !! export failed: {e}")

def main():
    usd_list = find_usd_files(args.folder)
    if not usd_list:
        log("未找到 USD 文件")
        return
    for p in usd_list:
        process_usd(p)

if __name__ == "__main__":
    try:
        main()
    finally:
        sim.close()
