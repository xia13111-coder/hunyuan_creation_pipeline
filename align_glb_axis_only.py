import os
import sys
import traceback
import numpy as np
import bpy
from mathutils import Matrix, Vector

# =========================
# 参数
# =========================
def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    import argparse
    p = argparse.ArgumentParser("GLB 仅绕 Z 轴旋转以实现 L/M/S → X/Y/Z 映射（不移位/不缩放）")
    p.add_argument("--input", required=True, help="输入文件或目录（递归处理 .glb）")
    p.add_argument("--ext", default=".glb", help="扩展名（默认 .glb ）")
    p.add_argument("--axis-map", default="X=L,Y=M,Z=S",
                   help="X/Y/Z 对应 L/M/S（最长/中/最短）。示例：'X=L,Z=M,Y=S'")
    p.add_argument("--overwrite", action="store_true", help="覆盖原 .glb；否则在文件名后加后缀")
    p.add_argument("--suffix", default="_axis", help="未覆盖时输出文件名后缀（默认 _axis）")
    p.add_argument("--prefer-deg", type=int, default=0,
                   help="破平优先角度，仅 0/90/180/270 四选一，默认 0")
    return p.parse_args(argv)

def log(*a): print(*a, flush=True)

# =========================
# Blender 基本操作
# =========================
def enable_addons():
    try: bpy.ops.preferences.addon_enable(module="io_scene_gltf2")
    except Exception: pass

def clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)

def import_glb(path):
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=path)
    after = set(bpy.data.objects)
    return [ob for ob in (after - before)]

def export_glb(out_path):
    op = bpy.ops.export_scene.gltf
    kw = dict(
        filepath=out_path,
        export_format="GLB",
        use_selection=False,
        export_apply=False,   # 我们已将旋转应用到几何
        export_yup=True       # 保持 glTF 约定
    )
    op(**kw)

# =========================
# 几何辅助
# =========================
def world_bbox_minmax(objs):
    inf, ninf = float("inf"), float("-inf")
    mn = Vector((inf, inf, inf)); mx = Vector((ninf, ninf, ninf)); has=False
    for ob in objs:
        if ob.type != "MESH": continue
        has=True
        for c in ob.bound_box:
            w = ob.matrix_world @ Vector(c)
            mn.x=min(mn.x,w.x); mn.y=min(mn.y,w.y); mn.z=min(mn.z,w.z)
            mx.x=max(mx.x,w.x); mx.y=max(mx.y,w.y); mx.z=max(mx.z,w.z)
    return (mn,mx) if has else (Vector((0,0,0)), Vector((0,0,0)))

def get_vertices_world(objs):
    vs=[]
    for ob in objs:
        if ob.type!="MESH": continue
        for v in ob.data.vertices:
            w = ob.matrix_world @ v.co
            vs.append((w.x,w.y,w.z))
    return np.asarray(vs, dtype=np.float64) if vs else np.empty((0,3), dtype=np.float64)

# =========================
# 只绕 Z 轴的映射选择
# =========================
def parse_axis_map(s):
    alias = {
        "L":0,"LONG":0,"LONGEST":0,
        "M":1,"MID":1,"MIDDLE":1,"MEDIUM":1,
        "S":2,"SHORT":2,"SHORTEST":2
    }
    out={}
    for t in [t for t in s.replace(";",",").split(",") if t.strip()]:
        if "=" not in t: continue
        ax,val = t.split("=",1)
        ax=ax.strip().upper(); val=val.strip().upper()
        if ax not in ("X","Y","Z"): raise ValueError(f"非法轴: {ax}")
        if val not in alias: raise ValueError(f"非法排名: {val}")
        out[ax]=alias[val]
    miss=[a for a in "XYZ" if a not in out]
    ranks=[r for r in (0,1,2) if r not in out.values()]
    for ax,rk in zip(miss,ranks): out[ax]=rk
    return out

def choose_rotation_by_axis_map_zonly(P_world, center_np, rank_map, prefer_deg=0):
    """
    仅在 Z 轴的 {0,90,180,270} 四种旋转里选择，使得 AABB 的 L/M/S 尽量匹配 X/Y/Z。
    说明：只能改变 X/Y 的对应关系，Z 方向长度保持不变；若映射要求 Z 的排名改变且原始 Z
    并非该排名，则会取惩罚最小的可行解。
    """
    prefer_deg = (prefer_deg % 360)
    if prefer_deg not in (0,90,180,270): prefer_deg = 0

    def Rz(deg):
        t = np.deg2rad(deg); c, s = np.cos(t), np.sin(t)
        return np.array([[c,-s,0],
                         [s, c,0],
                         [0, 0,1]], dtype=float)

    best_key, best_R = None, None
    for deg in (0, 90, 180, 270):
        R = Rz(deg)
        Q = (R @ (P_world - center_np).T).T + center_np
        mn, mx = Q.min(0), Q.max(0)
        ext = np.abs(mx - mn)                # [ex, ey, ez]

        # AABB 边长从大到小排序，得到每个轴的“排名”（0=L,1=M,2=S）
        idx = np.argsort(-ext)
        rank = np.empty(3, dtype=int)
        rank[idx[0]] = 0; rank[idx[1]] = 1; rank[idx[2]] = 2

        # 与期望排名的差距（惩罚越小越好）
        pen = (abs(int(rank[0]) - rank_map["X"]) +
               abs(int(rank[1]) - rank_map["Y"]) +
               abs(int(rank[2]) - rank_map["Z"]))

        # 破平：越接近 prefer_deg 越好；再最大化 AABB（稳定些）
        yaw_cost = 0 if deg == prefer_deg else 1 if (deg - prefer_deg) % 360 in (90,270) else 2
        key = (pen, yaw_cost, -float(ext[0]), -float(ext[1]), -float(ext[2]))

        if best_key is None or key < best_key:
            best_key, best_R = key, R
    return best_R

def npR_to_M4(R):
    return Matrix(((R[0,0],R[0,1],R[0,2],0.0),
                   (R[1,0],R[1,1],R[1,2],0.0),
                   (R[2,0],R[2,1],R[2,2],0.0),
                   (0.0,0.0,0.0,1.0)))

def apply_rotation_around(objs, R4, pivot):
    root = bpy.data.objects.new("AxisMapAlignRoot", None)
    bpy.context.collection.objects.link(root)
    for ob in objs:
        ob.parent = root
        ob.matrix_parent_inverse = root.matrix_world.inverted()

    Tpos = Matrix.Translation(pivot); Tneg = Matrix.Translation(-pivot)
    root.matrix_world = (Tpos @ R4 @ Tneg) @ root.matrix_world

    # 解父级，保留变换；并“应用旋转”到几何
    bpy.ops.object.select_all(action="DESELECT")
    for ob in objs: ob.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    bpy.ops.object.parent_clear(type="CLEAR_KEEP_TRANSFORM")
    bpy.data.objects.remove(root, do_unlink=True)

    bpy.ops.object.select_all(action="DESELECT")
    for ob in objs:
        if ob.type=="MESH": ob.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)

# =========================
# 流程
# =========================
def find_files(root, ext=".glb"):
    ext = ext.lower()
    if os.path.isfile(root) and root.lower().endswith(ext):
        return [os.path.abspath(root)]
    out=[]
    for r,_,names in os.walk(root):
        for n in names:
            if n.lower().endswith(ext):
                out.append(os.path.join(r,n))
    return out

def main():
    a = parse_args()
    enable_addons()
    files = find_files(a.input, a.ext)
    if not files:
        log("未找到任何", a.ext, "文件"); sys.exit(3)

    rank_map = parse_axis_map(a.axis_map)
    total, ok, fail = len(files), 0, 0
    log(f"发现 {total} 个 {a.ext}，仅做“绕 Z 轴”的轴映射对齐……")

    for i, glb in enumerate(files,1):
        try:
            log(f"[{i}/{total}] 导入：{glb}")
            clear_scene()
            objs = import_glb(glb)
            if not objs:
                log("  ! 未发现对象，跳过"); continue

            # 围绕包围盒中心旋转（不平移）
            mn, mx = world_bbox_minmax(objs)
            pivot = (mn + mx) * 0.5
            P = get_vertices_world(objs)
            if P.size == 0:
                log("  ! 无顶点，跳过"); continue

            R = choose_rotation_by_axis_map_zonly(
                P, np.array([pivot.x, pivot.y, pivot.z]),
                rank_map, prefer_deg=a.prefer_deg
            )
            apply_rotation_around(objs, npR_to_M4(R), pivot)

            # 导出 GLB
            if a.overwrite:
                out_path = glb
            else:
                base,ext = os.path.splitext(glb)
                out_path = base + a.suffix + ".glb"

            log("  导出：", out_path)
            export_glb(out_path)
            ok += 1
        except Exception:
            log("  ✗ 异常：", glb); traceback.print_exc(); fail += 1

    log(f"\n完成：成功 {ok}，失败 {fail}")
    if ok == 0 and total > 0: sys.exit(4)

if __name__ == "__main__":
    main()
