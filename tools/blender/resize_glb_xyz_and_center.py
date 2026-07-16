import os
import sys
import traceback
import numpy as np
import bpy
from mathutils import Matrix, Vector


# =============== 参数 ===============
def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    import argparse

    p = argparse.ArgumentParser(
        "将 GLB 缩放到目标 X/Y/Z 长度，并把包围盒中心移到原点后导出"
    )
    p.add_argument("--input", required=True, help="输入文件或目录（递归处理 .glb）")
    p.add_argument("--ext", default=".glb", help="扩展名（默认 .glb ）")
    p.add_argument("--len-x", type=float, required=True, help="目标 X 长度")
    p.add_argument("--len-y", type=float, required=True, help="目标 Y 长度")
    p.add_argument("--len-z", type=float, required=True, help="目标 Z 长度")
    p.add_argument(
        "--unit",
        choices=["m", "cm", "mm"],
        default="m",
        help="以上长度的单位（默认 m）",
    )
    p.add_argument(
        "--overwrite", action="store_true", help="覆盖原 .glb；否则加后缀另存"
    )
    p.add_argument(
        "--suffix", default="_scaled", help="未覆盖时输出后缀（默认 _scaled）"
    )
    return p.parse_args(argv)


def log(*a):
    print(*a, flush=True)


# =============== 基本 ===============
def enable_addons():
    try:
        bpy.ops.preferences.addon_enable(module="io_scene_gltf2")
    except Exception:
        pass


def clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def import_glb(path):
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=path)
    after = set(bpy.data.objects)
    return [ob for ob in (after - before)]


def export_glb(out_path):
    kw = dict(
        filepath=out_path,
        export_format="GLB",
        use_selection=False,
        export_apply=False,  # 我们已将变换应用到几何
        export_yup=True,
    )
    bpy.ops.export_scene.gltf(**kw)


# =============== 几何辅助 ===============
def world_bbox_minmax(objs):
    inf, ninf = float("inf"), float("-inf")
    mn = Vector((inf, inf, inf))
    mx = Vector((ninf, ninf, ninf))
    has = False
    for ob in objs:
        if ob.type != "MESH":
            continue
        has = True
        for c in ob.bound_box:
            w = ob.matrix_world @ Vector(c)
            mn.x = min(mn.x, w.x)
            mn.y = min(mn.y, w.y)
            mn.z = min(mn.z, w.z)
            mx.x = max(mx.x, w.x)
            mx.y = max(mx.y, w.y)
            mx.z = max(mx.z, w.z)
    return (mn, mx) if has else (Vector((0, 0, 0)), Vector((0, 0, 0)))


def apply_matrix_to_group(objs, M):
    # 用临时根节点施加矩阵，然后清父级、应用变换到网格
    root = bpy.data.objects.new("ResizeRoot", None)
    bpy.context.collection.objects.link(root)
    for ob in objs:
        ob.parent = root
        ob.matrix_parent_inverse = root.matrix_world.inverted()

    root.matrix_world = M @ root.matrix_world

    bpy.ops.object.select_all(action="DESELECT")
    for ob in objs:
        ob.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    bpy.ops.object.parent_clear(type="CLEAR_KEEP_TRANSFORM")
    bpy.data.objects.remove(root, do_unlink=True)

    # 应用到网格数据
    bpy.ops.object.select_all(action="DESELECT")
    for ob in objs:
        if ob.type == "MESH":
            ob.select_set(True)
    if any(ob.select_get() for ob in objs if ob.type == "MESH"):
        bpy.context.view_layer.objects.active = [
            ob for ob in objs if ob.type == "MESH"
        ][0]
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)


# =============== 查找文件 ===============
def find_files(root, ext=".glb"):
    ext = ext.lower()
    if os.path.isfile(root) and root.lower().endswith(ext):
        return [os.path.abspath(root)]
    out = []
    for r, _, names in os.walk(root):
        for n in names:
            if n.lower().endswith(ext):
                out.append(os.path.join(r, n))
    return out


# =============== 主流程 ===============
def main():
    a = parse_args()
    unit_scale = {"m": 1.0, "cm": 0.01, "mm": 0.001}[a.unit]
    target = np.array([a.len_x, a.len_y, a.len_z], dtype=float) * unit_scale

    if np.any(target <= 0):
        raise SystemExit("目标长度必须为正数")

    enable_addons()
    files = find_files(a.input, a.ext)
    if not files:
        log("未找到任何", a.ext, "文件")
        sys.exit(3)

    total, ok, fail = len(files), 0, 0
    log(f"发现 {total} 个 {a.ext}，开始按目标尺寸缩放并居中到原点……")

    for i, glb in enumerate(files, 1):
        try:
            log(f"[{i}/{total}] 导入：{glb}")
            clear_scene()
            objs = import_glb(glb)
            if not objs:
                log("  ! 未发现对象，跳过")
                continue

            mn, mx = world_bbox_minmax(objs)
            ext = np.array([mx.x - mn.x, mx.y - mn.y, mx.z - mn.z], dtype=float)
            if np.any(ext <= 1e-9):
                log(f"  ! 包围盒某些轴长度≈0，无法缩放到目标值：{ext}")
                continue

            scale = target / ext
            pivot = (mn + mx) * 0.5  # 包围盒中心（世界坐标）

            # 组合矩阵：先绕包围盒中心缩放，再把中心移到世界原点
            S4 = Matrix(
                (
                    (scale[0], 0, 0, 0),
                    (0, scale[1], 0, 0),
                    (0, 0, scale[2], 0),
                    (0, 0, 0, 1),
                )
            )
            Tpos = Matrix.Translation(pivot)
            Tneg = Matrix.Translation(-pivot)
            # M_total = T(-pivot) @ (T(pivot) @ S @ T(-pivot))
            M_total = Tneg @ (Tpos @ S4 @ Tneg)

            apply_matrix_to_group(objs, M_total)

            # 导出
            if a.overwrite:
                out_path = glb
            else:
                base, ext_ = os.path.splitext(glb)
                out_path = base + a.suffix + ".glb"
            log("  导出：", out_path)
            export_glb(out_path)
            ok += 1
        except Exception:
            log("  ✗ 异常：", glb)
            traceback.print_exc()
            fail += 1

    log(f"\n完成：成功 {ok}，失败 {fail}")
    if ok == 0 and total > 0:
        sys.exit(4)


if __name__ == "__main__":
    main()
