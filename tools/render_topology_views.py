from __future__ import annotations

import argparse
from pathlib import Path

import bpy
from mathutils import Vector


def parse_args() -> argparse.Namespace:
    import sys

    argv = sys.argv
    args = argv[argv.index("--") + 1 :] if "--" in argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(args)


def reset_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def import_asset(path: Path) -> list[bpy.types.Object]:
    ext = path.suffix.lower()
    if ext in {".glb", ".gltf"}:
        bpy.ops.import_scene.gltf(filepath=str(path))
    elif ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=str(path))
    elif ext == ".obj":
        if hasattr(bpy.ops.wm, "obj_import"):
            bpy.ops.wm.obj_import(filepath=str(path))
        else:
            bpy.ops.import_scene.obj(filepath=str(path))
    else:
        raise ValueError(path)
    return [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]


def join_meshes(meshes: list[bpy.types.Object]) -> bpy.types.Object:
    bpy.ops.object.select_all(action="DESELECT")
    for obj in meshes:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    if len(meshes) > 1:
        bpy.ops.object.join()
    obj = bpy.context.view_layer.objects.active
    obj.name = "topology_subject"
    return obj


def bbox(obj: bpy.types.Object) -> tuple[Vector, Vector, Vector]:
    corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    low = Vector((min(c.x for c in corners), min(c.y for c in corners), min(c.z for c in corners)))
    high = Vector((max(c.x for c in corners), max(c.y for c in corners), max(c.z for c in corners)))
    return low, high, (low + high) * 0.5


def setup_materials(obj: bpy.types.Object) -> None:
    obj.data.materials.clear()
    body = bpy.data.materials.new("mat_soft_clay")
    body.diffuse_color = (0.72, 0.74, 0.72, 1.0)
    obj.data.materials.append(body)

    wire = obj.modifiers.new("wire_overlay", "WIREFRAME")
    wire.thickness = 0.0012
    wire.use_even_offset = True
    wire.use_replace = False
    wire.material_offset = 1
    wire_mat = bpy.data.materials.new("mat_wire_black")
    wire_mat.diffuse_color = (0.0, 0.0, 0.0, 1.0)
    obj.data.materials.append(wire_mat)


def normalize_for_render(obj: bpy.types.Object) -> None:
    low, high, center = bbox(obj)
    dims = high - low
    scale = 4.0 / max(dims.x, dims.y, dims.z, 1e-6)
    obj.location = obj.location - center
    obj.scale = obj.scale * scale
    bpy.context.view_layer.update()


def create_camera(center: Vector, distance: float, view: str) -> bpy.types.Object:
    if view == "front":
        location = center + Vector((0, -distance, distance * 0.12))
        rotation = (1.450, 0, 0)
    elif view == "top":
        location = center + Vector((0, 0, distance))
        rotation = (0, 0, 0)
    elif view == "side":
        location = center + Vector((distance, 0, distance * 0.12))
        rotation = (1.450, 0, 1.5708)
    else:
        location = center + Vector((distance * 0.75, -distance * 0.95, distance * 0.55))
        direction = center - location
        rotation = direction.to_track_quat("-Z", "Y").to_euler()
    bpy.ops.object.camera_add(location=location, rotation=rotation)
    camera = bpy.context.object
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = distance * 1.12
    return camera


def render_view(obj: bpy.types.Object, output_dir: Path, view: str) -> None:
    low, high, center = bbox(obj)
    dims = high - low
    distance = max(dims.x, dims.y, dims.z) * 1.8
    camera = create_camera(center, distance, view)
    bpy.context.scene.camera = camera
    path = output_dir / f"{view}.png"
    bpy.context.scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)
    bpy.data.objects.remove(camera, do_unlink=True)


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reset_scene()
    obj = join_meshes(import_asset(Path(args.input)))
    normalize_for_render(obj)
    setup_materials(obj)
    bpy.context.scene.render.engine = "BLENDER_EEVEE"
    if hasattr(bpy.context.scene, "eevee"):
        bpy.context.scene.eevee.taa_render_samples = 64
    bpy.context.scene.world.color = (1, 1, 1)
    bpy.context.scene.render.resolution_x = 1600
    bpy.context.scene.render.resolution_y = 1600
    bpy.ops.object.light_add(type="AREA", location=(0, -3, 3))
    light = bpy.context.object
    light.data.energy = 450
    light.data.size = 4
    for view in ["front", "side", "top", "iso"]:
        render_view(obj, output_dir, view)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
