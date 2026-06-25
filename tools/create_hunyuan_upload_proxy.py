from __future__ import annotations

import argparse
import sys
from pathlib import Path

import bpy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a lightweight GLB for Hunyuan ReduceFace upload. "
            "The proxy keeps geometry but strips heavy textures/materials; "
            "use the original GLB as --input for local texture migration."
        )
    )
    parser.add_argument("--input", required=True, help="Original GLB/GLTF file")
    parser.add_argument("--output", required=True, help="Geometry-only proxy GLB output")
    parser.add_argument("--keep-uvs", action="store_true", help="Keep UV coordinates in the proxy")
    parser.add_argument("--keep-normals", action="store_true", help="Keep normals in the proxy")
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    return parser.parse_args(argv)


def import_model(path: Path) -> None:
    suffix = path.suffix.lower()
    if suffix in {".glb", ".gltf"}:
        bpy.ops.import_scene.gltf(filepath=str(path))
    elif suffix == ".fbx":
        bpy.ops.import_scene.fbx(filepath=str(path))
    elif suffix == ".obj":
        if hasattr(bpy.ops.wm, "obj_import"):
            bpy.ops.wm.obj_import(filepath=str(path))
        else:
            bpy.ops.import_scene.obj(filepath=str(path))
    else:
        raise ValueError(f"Unsupported proxy input format: {path.suffix}")


def strip_heavy_materials() -> None:
    for obj in bpy.context.scene.objects:
        if obj.type == "MESH":
            obj.data.materials.clear()
    for image in list(bpy.data.images):
        if image.name not in {"Render Result", "Viewer Node"}:
            bpy.data.images.remove(image)
    for material in list(bpy.data.materials):
        bpy.data.materials.remove(material)


def select_meshes() -> None:
    bpy.ops.object.select_all(action="DESELECT")
    for obj in bpy.context.scene.objects:
        obj.select_set(obj.type == "MESH")
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not meshes:
        raise ValueError("No mesh objects found in input")
    bpy.context.view_layer.objects.active = meshes[0]


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    import_model(input_path)
    strip_heavy_materials()
    select_meshes()

    bpy.ops.export_scene.gltf(
        filepath=str(output_path),
        export_format="GLB",
        use_selection=True,
        export_apply=True,
        export_materials="NONE",
        export_texcoords=bool(args.keep_uvs),
        export_normals=bool(args.keep_normals),
    )

    print(f"Wrote Hunyuan upload proxy: {output_path}")


if __name__ == "__main__":
    main()
