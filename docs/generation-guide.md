# Choosing Between Hunyuan and SAM3D

[English](./generation-guide.md) | [中文](./generation-guide.zh.md) | [Documentation index](./README.md)

Both paths create a raw GLB, then normally continue through Hunyuan refine,
Blender postprocess, and Isaac Sim physics. The difference is how the initial
geometry is created:

- **Hunyuan** generates a complete asset from one image and infers
  unseen regions.
- **SAM3D** segments a selected object and reconstructs it from one or more
  photographs.

## Quick choice

| Input or requirement | Use | Why |
| --- | --- | --- |
| One clean product image | Start with Hunyuan image-to-3D | Usually the simplest way to obtain a visually complete asset. |
| Several views of the same object | SAM3D `multi` | All views contribute to one reconstruction. |
| One target in a cluttered scene | SAM3D | Its segmentation prompt selects the object. |
| Many unrelated images | Hunyuan `--input-dir` | Each image becomes an independent job. |
| Engineering dimensions and assembly transforms must be preserved | STEP/STP CAD path | Neither image path is an engineering measurement tool. |

## Important differences

| Topic | Hunyuan | SAM3D |
| --- | --- | --- |
| Input | Image URL or image directory | One image or multiple views of one object |
| Multiple images | Independent jobs | Views combined into one reconstruction in `multi` mode |
| Prompt | Not used; Hunyuan accepts images only | `--sam3d-prompt` only selects the 2-D object |
| Hidden regions | Generated from model priors | Still inferred; extra views reduce uncertainty |
| Runtime | Tencent Cloud API | Local GPU for reconstruction |
| Main controls | Source and `--face-count` | Mode, segmentation prompt, seed, and steps |
| Common failures | Plausible but incorrect hidden geometry | Wrong masks, inconsistent views, or occlusion |

Use Hunyuan for quick generation from a single product image. This project does
not support text-only Hunyuan input. Remember that `--input-dir` is a batch input: four images
of one object produce four separate jobs.

Use SAM3D when you have complementary views of the same physical object or need
to select one object from a scene. Multi-view images should show the same
instance at similar scale with limited occlusion.

## Decision flow

```text
Need engineering dimensions or assembly accuracy?
├─ Yes -> use STEP/STP
└─ No
   ├─ No image -> unsupported by the current generation workflows
   ├─ Multiple views of one object -> SAM3D multi
   ├─ Need to select an object from a scene -> SAM3D single
   └─ One clear subject image -> Hunyuan image-to-3D
```

## Quality limits

Neither path recovers exact internal structures, hole locations, wall
thickness, or dimensions that are not visible. `--len-x/y/z` changes only the
final bounding box; it does not recover part-level measurements.

Check the raw GLB silhouette and component completeness before refine. Refine
can simplify geometry and transfer appearance, but it cannot correct a wrong
reconstruction automatically. For simulation assets, also verify collision,
mass, origin, and physical behavior in the final USD.

Commands and parameters: [Hunyuan module](./modules/hunyuan.md) and
[SAM3D module](./modules/sam3d.md).
