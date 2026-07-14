# Choosing Between Hunyuan And SAM3D

[English](./generation-guide.md) | [中文](./generation-guide.zh.md) | [Documentation index](./README.md)

The project has two image-to-3D paths, but they solve different problems:

- **Hunyuan** is generative image-to-3D and text-to-3D. It creates a complete model from a single image or text semantics and infers unseen regions.
- **SAM3D** reconstructs a selected object from one or more image observations. It first segments the requested target, then runs single-view or multi-view reconstruction.

Both paths produce a raw GLB. The production workflow then runs Hunyuan refine, Blender postprocess, and Isaac Sim physics by default, so this choice concerns how the initial geometry is obtained.

## Main Differences

| Dimension | Hunyuan generation | SAM3D reconstruction |
| --- | --- | --- |
| Primary goal | Generate a complete asset from semantics and appearance | Reconstruct a selected object visible in one or more images |
| Input | Text, one image URL, or an image directory | One image or multiple views of the same object |
| Meaning of multiple files | Each image in `--input-dir` becomes an independent generation job | `multi` treats all directory images as views of one object |
| Prompt behavior | `--prompt` describes a text-to-3D asset | `--sam3d-prompt` only selects the 2D segmentation target; it does not describe 3D generation |
| Unseen regions | Filled using a generative prior and may differ from the real object | Single view still requires inference; complementary views reduce, but do not eliminate, unknown regions |
| Image requirements | A clear, lightly occluded subject is generally more stable | The prompt must segment the same target reliably; multi-view images should keep subject identity, scale, and framing consistent |
| Runtime | Tencent Cloud API with credentials, network access, service quota, and job polling | Local GPU execution in the `hunyuan_sam3d` environment |
| Main controls | Source image/text and `--face-count` | Mode, segmentation prompt, seed, and inference steps |
| Typical risks | Plausible but inaccurate hidden geometry; variation in generated details | Wrong masks, inconsistent subjects, insufficient view coverage, or occlusion-induced geometry errors |
| Downstream path | Hunyuan refine by default | Also Hunyuan refine by default, then the two paths converge |

## Prefer Hunyuan When

- Only a text description is available.
- Only one clear image is available and a visually complete asset is needed quickly.
- The target is a concept asset or general prop rather than a strict reproduction of hidden real-world structure.
- A requested raw face count is useful.
- Tencent Cloud is available but the local machine is not suitable for SAM3D.

Examples include generating a conceptual rack from `industrial storage rack`, creating a display asset from one product image, or processing a directory of unrelated product images as separate jobs.

Remember that `--input-dir` is a batch directory, not a multi-view input. Four views of one object in that directory produce four independent Hunyuan jobs rather than one fused reconstruction.

## Prefer SAM3D When

- Complementary front, side, and rear views of the same physical object are available.
- A scene contains several objects and `--sam3d-prompt` is needed to select one target.
- The first reconstruction stage should run locally and fixed seeds are useful for comparisons.
- A single image has a clear target boundary and segmentation-first reconstruction is desired.

For multi-view input, every image should show the same object instance, use a similar subject scale, provide complementary angles, minimize occlusion, and allow the same prompt to identify the target consistently.

## Quick Selection

| Available information or requirement | Recommended path | Reason |
| --- | --- | --- |
| Text only | Hunyuan text-to-3D | SAM3D requires images. |
| One clean product image | Start with Hunyuan image-to-3D | It is the simpler path to a complete visual asset; SAM3D single is useful as a comparison. |
| Multiple complementary views of one object | SAM3D multi | The images jointly contribute to one reconstruction. |
| A cluttered scene with one target object | SAM3D | The segmentation prompt explicitly selects the target. |
| Many unrelated single images | Hunyuan `--input-dir` | Each image becomes an independent generation job. |
| Hunyuan generation service is temporarily unavailable | SAM3D for the first stage | Raw reconstruction is local, although the normal downstream refine still requires Hunyuan. |
| Engineering dimensions, hole locations, and assembly transforms must be preserved | Neither; use the STEP/STP CAD path | Generative reconstruction is not engineering measurement or CAD conversion. |

## Decision Flow

```text
Are images available?
├─ No -> Hunyuan text-to-3D
└─ Yes
   ├─ Are complementary views of the same object available?
   │  ├─ Yes -> SAM3D multi
   │  └─ No
   │     ├─ Must one target be selected from a cluttered scene?
   │     │  ├─ Yes -> SAM3D single
   │     │  └─ No -> Start with Hunyuan image-to-3D; compare SAM3D single if useful
   └─ Is engineering-level dimensional and assembly accuracy required?
      └─ Yes -> Use STEP/STP instead of either generation path
```

## Quality Expectations

Neither Hunyuan nor SAM3D is a CAD reverse-engineering system. Internal structures, exact holes, wall thickness, and real dimensions that are not directly observed cannot be guaranteed. `--len-x/y/z` sets only the final bounding-box dimensions; it does not recover part-level engineering dimensions.

Inspect the raw GLB silhouette and component completeness before refine. Refine prepares reduced geometry and local Blender output but cannot turn an incorrect structure into the real structure automatically. Critical simulation assets still require final USD checks for collision, mass, origin, and physical behavior.

See the [Hunyuan module](./modules/hunyuan.md) and [SAM3D module](./modules/sam3d.md) for commands and parameters.
