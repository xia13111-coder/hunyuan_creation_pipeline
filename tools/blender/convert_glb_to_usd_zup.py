import os, sys, traceback
import bpy


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    import argparse
    p = argparse.ArgumentParser("GLB → USD（Z-up），每个模型输出到独立文件夹")
    p.add_argument("--input", required=True, help="输入文件或目录（递归处理 .glb）")
    p.add_argument("--ext", default=".glb", help="待转换扩展名（默认 .glb ）")
    p.add_argument("--usd-format", choices=["usdc","usda","usd"], default="usdc",
                   help="USD 格式：usdc（二进制）、usda（ASCII）或 usd（自动）")
    p.add_argument("--overwrite", action="store_true",
                   help="覆盖同名文件夹（若存在则复用并覆盖其中同名 USD）")
    p.add_argument("--suffix", default="_zup",
                   help="未覆盖时输出文件夹后缀（默认 _zup）")
    p.add_argument("--visible-only", action="store_true", help="只导出可见对象")
    return p.parse_args(argv)

def log(*a): print(*a, flush=True)

def enable_addons():
    for mod in ("io_scene_gltf2", "io_scene_usd"):
        try: bpy.ops.preferences.addon_enable(module=mod)
        except Exception: pass

def clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)

def import_glb(path):
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=path)
    after = set(bpy.data.objects)
    return [ob for ob in (after - before)]

def export_usd(out_path, out_dir, visible_only=False):


    tex_dir = os.path.join(out_dir, "textures")
    os.makedirs(tex_dir, exist_ok=True)


    tried = False
    for kw_variant in [
        dict(filepath=out_path, check_existing=False,
             export_animation=False,
             selected_objects_only=False,
             visible_objects_only=bool(visible_only),
             export_textures=True,  # 拷贝贴图
             export_materials=True,
             export_uvmaps=True,
             export_normals=True,
             export_color_management=True,
             use_instancing=True,
             texture_dir=tex_dir,   # 常见参数名1
             ),
        dict(filepath=out_path, check_existing=False,
             export_animation=False,
             selected_objects_only=False,
             visible_objects_only=bool(visible_only),
             export_textures=True,
             export_materials=True,
             export_uvmaps=True,
             export_normals=True,
             use_instancing=True,
             texture_directory=tex_dir,  # 常见参数名2
             ),
        dict(filepath=out_path, check_existing=False,
             export_animation=False,
             selected_objects_only=False,
             visible_objects_only=bool(visible_only),
             export_textures=True,
             export_materials=True,
             export_uvmaps=True,
             export_normals=True,
             use_instancing=True,
             ),  # 不显式指定目录，由导出器决定
    ]:
        try:
            bpy.ops.wm.usd_export(**kw_variant)
            tried = True
            break
        except Exception as e:
            # 尝试下一个变体
            last_err = e
            continue

    if not tried:
        log("  ! 高级导出参数不兼容，回退到最小参数：", last_err)
        bpy.ops.wm.usd_export(filepath=out_path, check_existing=False)

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

def set_stage_up_axis_z(out_path):

    try:
        from pxr import Usd, UsdGeom
        stage = Usd.Stage.Open(out_path)
        if stage:
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
            stage.GetRootLayer().Save()
            return True
    except Exception as e:
        log("  ! 无法写 upAxis（pxr 不可用或打开失败）：", e)
    return False


def main():
    a = parse_args()
    enable_addons()

    files = find_files(a.input, a.ext)
    if not files:
        log("未找到任何", a.ext, "文件"); sys.exit(3)

    # 输出后缀名
    if a.usd_format in ("usdc", "usda"):
        usd_ext = "." + a.usd_format
    else:
        usd_ext = ".usd"

    total, ok, fail = len(files), 0, 0
    log(f"发现 {total} 个 {a.ext}，转为 USD（Z-up），每个模型输出到独立文件夹……")

    for i, glb in enumerate(files, 1):
        try:
            log(f"[{i}/{total}] 导入：{glb}")
            clear_scene()
            objs = import_glb(glb)
            if not objs:
                log("  ! 未发现对象，跳过"); continue

            base_noext = os.path.splitext(glb)[0]
            parent_dir  = os.path.dirname(base_noext)
            stem        = os.path.basename(base_noext)

            # 目标文件夹
            out_dir = os.path.join(parent_dir, stem if a.overwrite else stem + a.suffix)
            os.makedirs(out_dir, exist_ok=True)

            # 文件名使用文件夹名
            out_path = os.path.join(out_dir, os.path.basename(out_dir) + usd_ext)

            log("  导出到：", out_path)
            export_usd(out_path, out_dir, visible_only=a.visible_only)

            # 写 upAxis = Z（双保险）
            if set_stage_up_axis_z(out_path):
                log("  已写入 Stage upAxis = Z")

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
