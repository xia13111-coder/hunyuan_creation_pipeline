
"""

./python.sh /home/user/ubtech/physic_setting/add_physics.py \
  --folder /home/user/create_data/downloads \
  --material-file /home/user/ubtech/physic_setting/materials.json \
  --out-dir /home/user/ubtech/physic_setting/table_7 \
  --suffix _phys --headless \
  --material plastic --set-mass 30.0 \
  --approx sdf --force-sdf \
  --sdf-res 256 --sdf-subgrid 6 --sdf-band 0.01 --sdf-margin 0.01
"""

import argparse
import fnmatch
import json
import math
import os
import re
import sys


from omni.isaac.kit import SimulationApp

def log(*a):
    print(*a, flush=True)

def build_sim(headless):
    return SimulationApp({"headless": headless})

def parse_args():
    ap = argparse.ArgumentParser(description="批量为 USD 添加物理（动态刚体 + 支持 SDF/凸分解等碰撞体）")
    ap.add_argument("--folder", required=True, help="输入 USD 根目录（递归 .usd/.usda/.usdc）")
    ap.add_argument("--config", help="规则 JSON（可选，支持 defaults / rules / material_name_map）")
    ap.add_argument("--material-file", required=True, help="材质参数 JSON（必须，定义 materials 和可选 name_map）")
    ap.add_argument("--out-dir", help="输出根目录（镜像结构）；不填则覆盖到原目录旁（加后缀）")
    ap.add_argument("--suffix", default="_phys", help="输出文件名后缀（默认 _phys）")
    ap.add_argument("--headless", action="store_true", help="无界面运行")


    ap.add_argument("--set-mass", type=float, help="给所有物体直接设置固定质量(kg)，仅覆盖质量")
    ap.add_argument("--material", help="统一材质标签（如 steel/rubber/wood/...），覆盖密度/摩擦/回弹/组合模式")


    ap.add_argument("--approx", help="碰撞近似（sdf/convexHull/convexDecomposition/triangleMesh/meshSimplification/box/sphere）")
    ap.add_argument("--force-sdf", action="store_true",
                    help="无论规则/命令行指定为何，动态刚体一律强制用SDF（避免回退与报错）")


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
        return {"defaults": {}, "rules": [], "material_name_map": []}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data.setdefault("defaults", {})
    data.setdefault("rules", [])
    data.setdefault("material_name_map", [])
    return data

def load_materials(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "materials" not in data or not isinstance(data["materials"], dict):
        raise ValueError("materials.json 缺少根键 'materials' 或其类型不是对象")
    data.setdefault("name_map", [])
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

def _mat3_det_from_mat4(m4):
    a00, a01, a02 = m4[0][0], m4[0][1], m4[0][2]
    a10, a11, a12 = m4[1][0], m4[1][1], m4[1][2]
    a20, a21, a22 = m4[2][0], m4[2][1], m4[2][2]
    return (a00 * (a11 * a22 - a12 * a21)
            - a01 * (a10 * a22 - a12 * a20)
            + a02 * (a10 * a21 - a11 * a20))

def _mesh_local_volume_units3(mesh):
    pts = mesh.GetPointsAttr().Get()
    if not pts:
        return 0.0
    counts = mesh.GetFaceVertexCountsAttr().Get() or []
    idx = mesh.GetFaceVertexIndicesAttr().Get() or []
    tris = _triangulate_counts_indices(counts, idx)
    if not tris:
        return 0.0
    vol = 0.0
    for i0, i1, i2 in tris:
        p0 = Gf.Vec3d(*pts[i0]); p1 = Gf.Vec3d(*pts[i1]); p2 = Gf.Vec3d(*pts[i2])
        vol += Gf.Dot(p0, Gf.Cross(p1, p2))
    return abs(vol) / 6.0

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

def bind_mat_to_body_and_meshes(stage, body_prim, material):
    UsdShade.MaterialBindingAPI.Apply(body_prim).Bind(
        material, UsdShade.Tokens.weakerThanDescendants, "physics"
    )
    for p in Usd.PrimRange(body_prim):
        if p.IsA(UsdGeom.Mesh):
            UsdShade.MaterialBindingAPI.Apply(p).Bind(
                material, UsdShade.Tokens.weakerThanDescendants, "physics"
            )

def _get_bound_material_name(mesh_prim):
    try:
        mb = UsdShade.MaterialBindingAPI(mesh_prim)
        res = mb.ComputeBoundMaterial()
        if isinstance(res, tuple) and res and res[0]:
            return res[0].GetPrim().GetName()
        if hasattr(mb, "GetDirectBinding"):
            db = mb.GetDirectBinding()
            if db and db.GetMaterial():
                return db.GetMaterial().GetPrim().GetName()
    except Exception:
        pass
    return None

def detect_material_tag(body_prim):
    names = []
    for p in Usd.PrimRange(body_prim):
        if p.IsA(UsdGeom.Mesh):
            n = _get_bound_material_name(p)
            if n:
                names.append(n)
    if not names:
        return None

    def _scan_name_map(name_map, names_list):
        for n in names_list:
            for rule in (name_map or []):
                if re.search(rule.get("regex", ""), n, flags=re.IGNORECASE):
                    return str(rule.get("tag", "")).strip().lower() or None
        return None

    tag = _scan_name_map(MATS.get("name_map", []), names)
    if tag:
        return tag
    return _scan_name_map(RULES.get("material_name_map", []), names)

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
    if "sdf_remesh" in params and hasattr(sdf_api, "CreateSdfEnableRemeshingAttr"):
        sdf_api.CreateSdfEnableRemeshingAttr(bool(params["sdf_remesh"]))
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
        if req_approx == "sdf":
            _author_sdf_on_mesh(mesh_prim, params)
            used = "sdf"
        elif req_approx == "convexDecomposition":
            _author_convex_decomposition(mesh_prim)
            used = "convexDecomposition"
        elif req_approx in ("convexHull", "boundingCube", "boundingSphere", "sphereApproximation"):
            _set_approx_token(mesh_prim, req_approx)
            UsdPhysics.CollisionAPI.Apply(mesh_prim)
            UsdPhysics.MeshCollisionAPI.Apply(mesh_prim)
            col = PhysxSchema.PhysxCollisionAPI.Apply(mesh_prim)
            if hasattr(col, "CreateContactOffsetAttr"): col.CreateContactOffsetAttr(float(params.get("contact_offset", args.contact_offset)))
            if hasattr(col, "CreateRestOffsetAttr"):    col.CreateRestOffsetAttr(float(params.get("rest_offset", args.rest_offset)))
            used = req_approx
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
        if not rule_tag:
            rule_tag = normalize_tag(detect_material_tag(prim))
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
        rel_dir = os.path.relpath(base_dir, args.folder)
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

    anchor = get_anchor_root(stage)
    log("  anchor root =", anchor.GetPath().pathString)

    for prim in anchor.GetChildren():
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

        if params.get("body_kind", "rigid") == "rigid":
            if args.set_mass is not None:
                m = float(args.set_mass)
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
