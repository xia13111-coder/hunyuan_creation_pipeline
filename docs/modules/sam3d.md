# SAM 3D Objects Pipeline

[English](./sam3d.md) | [中文](./sam3d.zh.md) | [Documentation index](../README.md)

SAM3D reconstructs a selected object from one or more images, then sends the
raw GLB through Hunyuan refine, Blender postprocess, and Isaac Sim physics. See
[Choosing between Hunyuan and SAM3D](../guides/generation-guide.md) before selecting a
generation path.

## Flow

```text
input image(s)
-> prepare_sam3d_input: filter and convert to RGB PNG
-> run_reconstruct.py: SAM3 mask + single/multi-view reconstruction
-> raw GLB
-> Hunyuan ReduceFace and local Blender refine
-> Blender alignment, sizing, centering, and USD export
-> Isaac Sim physics and final dependency collection
```

Do not use `--skip-refine` in the standard workflow.

## Local weights and environment

```bash
conda activate hunyuan_sam3d
```

The pipeline launches SAM3D with the current Python executable. It discovers
the project defaults, or you can set every local path explicitly in the root
`.env` file:

```dotenv
SAM3D_SINGLE_VIEW_ROOT=/absolute/path/to/sam-3d-objects
SAM3D_MULTI_VIEW_ROOT=/absolute/path/to/sam-3d-objects-multiview
SAM3D_PIPELINE_CONFIG=/absolute/path/to/checkpoints/pipeline.yaml
SAM3_REPOSITORY=/absolute/path/to/sam3
SAM3_CHECKPOINT=/absolute/path/to/sam3.pt
SAM3D_MOGE_CHECKPOINT=/absolute/path/to/moge-vitl/model.pt
SAM3D_DINOV2_REPOSITORY=/absolute/path/to/facebookresearch_dinov2_main
SAM3D_DINOV2_CHECKPOINT=/absolute/path/to/dinov2_vitl14_reg4_pretrain.pth
```

Normal SAM3/SAM3D inference forces Hugging Face and Transformers offline. A
missing, empty, or incomplete source directory, configuration, or weight fails
immediately instead of falling back to a download.

At runtime, the wrapper creates a temporary local overlay from the upstream
`pipeline.yaml` and inserts absolute paths for MoGe, DINOv2, and SAM3D
checkpoints. It never modifies the upstream YAML.

Hunyuan generation and ReduceFace are Tencent Cloud APIs, not local model
weights. SAM3D reconstruction is therefore fully offline, while downstream
Hunyuan refine still requires network access and cloud credentials. USD and
physics stages also require Isaac Sim Python.

## Input rules

`--sam3d-input` accepts one image or a directory containing `.png`, `.jpg`,
`.jpeg`, `.webp`, or `.bmp` files.

- If the directory has an `images/` child, only that child is read.
- Discovery is not recursive below the selected directory.
- Files are sorted by name; mask-like names (`mask_*` or `*_mask`) are ignored.
- Images are copied as RGB PNG. Existing alpha is not used as a mask.
- One prompt is applied to every view; precomputed masks are not required.

Multi-view images must show the same object. Prefer complementary angles with
similar subject scale, stable framing, and little occlusion.

## Command

```bash
hunyuan-asset-pipeline \
  --sam3d-input ./data/sam3d_images \
  --sam3d-mode auto \
  --sam3d-prompt "metal shelves" \
  --sam3d-seed 42 \
  --sam3d-steps 50 \
  --output-dir ./outputs/sam3d_example/generation \
  --intermediate-output-dir ./outputs/sam3d_example/intermediate \
  --final-output-dir ./outputs/sam3d_example/final \
  --result-json ./outputs/sam3d_example/pipeline_result.json \
  --refine-config-path ./configs/refinement/hunyuan_reduce_local_postprocess.yaml \
  --refine-temp-upload uguu \
  --len-x 0.4 --len-y 0.3 --len-z 0.8 \
  --orientation "X=L,Y=M,Z=S" \
  --approx sdf
```

## SAM3D options

| Option | Default | Meaning |
| --- | --- | --- |
| `--sam3d-input` | — | Image or image directory. Mutually exclusive with `--sam3d-glb`, `--existing-glb`, and `--manual-stp`. |
| `--sam3d-mode` | `auto` | `single`, `multi`, or automatic selection from image count. |
| `--sam3d-prompt` | — | Object name used for 2-D segmentation; required with `--sam3d-input`. |
| `--sam3d-confidence-threshold` | `0.5` | Minimum SAM3 grounding score. Lowering it increases false positives. |
| `--sam3d-seed` | `42` | Reconstruction random seed. |
| `--sam3d-steps` | `50` | Stage-1 sampling steps; higher values cost more time and do not set face count. |
| `--output-dir` | `./downloads` | Workspace root. Existing run names receive `_2`, `_3`, and so on. |
| `--sam3d-glb` | — | Resume downstream processing from an existing SAM3D GLB. |

Mode behavior:

| Mode | Behavior |
| --- | --- |
| `auto` | One valid image uses `single`; more than one uses `multi`. |
| `single` | Uses the first image after sorting. |
| `multi` | Sends all valid images to the multi-view backend; use at least two views. |

Use a concise English object name for `--sam3d-prompt`, such as
`industrial control cabinet` or `red plastic crate`. It selects the object; it
does not describe 3-D geometry, dimensions, or physical properties. A view with
no valid mask is skipped, and the run fails if all views are skipped.

Keep seed `42` while comparing prompts or image sets. Change it only when
testing reconstruction variation. More sampling steps do not guarantee better
geometry and do not control ReduceFace or collision quality.

## Downstream options

| Option | Meaning |
| --- | --- |
| `--refine-config-path` | Hunyuan ReduceFace and local Blender refine configuration. |
| `--refine-temp-upload` | Temporary host used to make a local GLB reachable by Hunyuan. |
| `--skip-refine` | Diagnostic bypass; not recommended for normal runs. |
| `--len-x/y/z` | Final bounding-box dimensions in meters. |
| `--orientation` | Long/middle/short bounding-box axis mapping. |
| `--approx` | Isaac Sim collision approximation. |
| `--set-mass` | Total final mass in kilograms. |

## Outputs and resume

Single-view workspaces contain `image.png` and usually `result_obj0.glb`;
multi-view workspaces contain `images/`, `masks/`, and usually `result.glb`.
The selected GLB is recorded as `generation.selected_glb` in the file passed to
`--result-json`.

If reconstruction succeeded but a later stage failed, resume from that GLB:

```bash
hunyuan-asset-pipeline \
  --sam3d-glb ./outputs/sam3d_example/generation/sam3d/sam3d_images/result.glb \
  --intermediate-output-dir ./outputs/sam3d_resume/intermediate \
  --final-output-dir ./outputs/sam3d_resume/final \
  --refine-config-path ./configs/refinement/hunyuan_reduce_local_postprocess.yaml \
  --refine-temp-upload uguu \
  --len-x 0.4 --len-y 0.3 --len-z 0.8 \
  --orientation "X=L,Y=M,Z=S" \
  --approx sdf
```

## Troubleshooting

| Message | Action |
| --- | --- |
| `No module named 'sam3.model'` | Activate `hunyuan_sam3d` and check the SAM3 submodule under the single-view checkout. |
| `Local model path ... is missing` | Set the reported local path in `.env`; the pipeline will not download it. |
| `SAM3 segmentation produced no valid masks` | Use a direct object name and inspect visibility. Lower the threshold only after verifying the low-score mask. |
| `Multi-view SAM3D script not found` | Check the default checkout or set `SAM3D_MULTI_VIEW_ROOT`. |
| `FailedOperation.RequestTimeout` | Let ReduceFace retry, or resume the existing GLB with `--sam3d-glb`. |
| Pillow, spconv, or AMP warnings | Usually compatibility warnings; they are not fatal if reconstruction finishes. |
