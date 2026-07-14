# SAM 3D Objects Pipeline Guide

[English](./sam3d.md) | [中文](./sam3d.zh.md) | [Documentation index](../README.md)

For input and use-case differences from Hunyuan generation, see [Choosing Between Hunyuan And SAM3D](../generation-guide.md).

This guide explains how to reconstruct a GLB from one or more images with SAM 3D Objects and continue through Hunyuan refine, Blender postprocess, and Isaac Sim physics authoring.

## End-to-End Flow

```text
input image(s)
-> asset_pipeline.jobs.sam3d.prepare_sam3d_input
   -> filter images and convert them to RGB PNG
   -> single: image.png
   -> multi: images/00000.png, images/00001.png, ...
-> tools/sam3d/run_reconstruct.py
   -> SAM3 generates a mask for each image from --sam3d-prompt
   -> single: SAM 3D Objects single-view reconstruction
   -> multi: SAM 3D Objects multi-view reconstruction
   -> export a raw GLB
-> asset_refiner
   -> Tencent Hunyuan ReduceFace
   -> Blender low-mesh projection and UV unwrap
-> Blender axis mapping, resize, centering, and GLB-to-USD
-> Isaac Sim collision, rigid body, and mass
-> collect final USD
```

Do not use `--skip-refine` for the normal production path. Hunyuan refine and the local Blender worker prepare the reduced mesh and downstream export.

## Environment

Hunyuan, refine, the API, the CLI, and SAM3D use the same environment:

```bash
conda activate hunyuan_sam3d
```

The pipeline launches SAM3D with the current process's `sys.executable` and
validates the environment name at startup. It no longer searches for or maintains
a second Python path.

Default checkout and checkpoint locations:

```text
./tools/sam3d/third_party/sam-3d-objects/
./tools/sam3d/third_party/sam-3d-objects-multiview/
./tools/sam3d/third_party/sam-3d-objects/checkpoints/sam3.pt
```

Override them when needed:

```bash
export SAM3D_SINGLE_VIEW_ROOT="/path/to/sam-3d-objects"
export SAM3D_MULTI_VIEW_ROOT="/path/to/sam-3d-objects-multiview"
export SAM3_CHECKPOINT="/path/to/sam3.pt"
```

Hunyuan refine also requires `TENCENTCLOUD_SECRET_ID`, `TENCENTCLOUD_SECRET_KEY`, and Blender. USD conversion and physics authoring require Isaac Sim Python.

## Image Input

`--sam3d-input` accepts one image file or a directory of images. Supported extensions are:

```text
.png  .jpg  .jpeg  .webp  .bmp
```

A single image can be passed directly:

```text
data/shelf.jpg
```

Multiple images can live directly in one directory:

```text
data/sam3d_images/
├── front.jpg
├── left.jpg
├── right.jpg
└── back.jpg
```

An `images/` child directory is also supported:

```text
data/sam3d_images/
└── images/
    ├── 00000.jpg
    ├── 00001.jpg
    └── 00002.jpg
```

Input rules:

- If the input directory contains `images/`, only that child directory is read.
- Image discovery is not recursive below the selected directory.
- Files are sorted by name. Multi-view inputs are copied as `00000.png`, `00001.png`, and so on.
- Files whose names start with `mask_` or end with `_mask` are ignored.
- Every source image is copied as RGB PNG; an existing alpha channel is not used directly as a mask.
- Precomputed masks are not required. The wrapper applies the same `--sam3d-prompt` to every view and generates masks automatically.

Multi-view inputs should show the same object. Prefer complete, lightly occluded views with similar subject scale and complementary front, side, and rear angles. Camera calibration is not required by this wrapper, but clear backgrounds and stable framing usually produce more consistent masks.

## Complete Command

```bash
python ./run_asset_pipeline.py \
  --sam3d-input ./data/sam3d_images \
  --sam3d-mode auto \
  --sam3d-prompt "metal shelves" \
  --sam3d-seed 42 \
  --sam3d-steps 50 \
  --output-dir ./sam3d_downloads \
  --intermediate-output-dir ./sam3d_output_intermediate \
  --final-output-dir ./sam3d_output_final \
  --result-json ./sam3d_pipeline_result.json \
  --refine-config-path ./configs/hunyuan_reduce_local_postprocess.yaml \
  --refine-temp-upload uguu \
  --len-x 0.4 \
  --len-y 0.3 \
  --len-z 0.8 \
  --orientation "X=L,Y=M,Z=S" \
  --approx sdf
```

## SAM3D Options

| Option | Default | Purpose |
| --- | --- | --- |
| `--sam3d-input` | none | Source image file or directory. Required for the image path and mutually exclusive with `--sam3d-glb`, `--existing-glb`, and `--manual-stp`. |
| `--sam3d-mode` | `auto` | Input mode: `auto`, `single`, or `multi`. It controls both the prepared directory layout and the reconstruction backend. |
| `--sam3d-prompt` | none | SAM3 text segmentation prompt. Required with `--sam3d-input`. It identifies the object in each image; it is not a text-to-3D description. |
| `--sam3d-seed` | `42` | Reconstruction random seed. It helps reproduce geometry under the same software, model, input, and parameters; it does not choose the segmentation target. |
| `--sam3d-steps` | `50` | Stage-1 geometry sampling steps, passed as `stage1_inference_steps`/`stage1_steps`. Higher values take longer and do not represent face count. Stage 2 is currently fixed at 25 steps. |
| `--output-dir` | `./downloads` | SAM3D workspace root. Runs use `<output-dir>/sam3d/<source-name>` and append `_2`, `_3`, and so on instead of overwriting an existing run. |
| `--sam3d-glb` | none | An existing SAM3D GLB. Skips image segmentation and SAM3D reconstruction, while Hunyuan refine and Blender/Isaac processing still run by default. |

### `--sam3d-mode`

| Mode | Actual behavior | Recommended use |
| --- | --- | --- |
| `auto` | Resolves to `single` for one valid image and `multi` for more than one. | Recommended default. |
| `single` | Uses only the first image after sorting and ignores the rest. | One product image, or a deliberate single-view run. |
| `multi` | Sends every valid image to the multi-view project; prepare at least two complementary views. | Multiple angles of the same object. |

The resolved mode and valid image count are printed in the log:

```text
SAM3D source images: ... | image_count=4 | mode=multi
```

### `--sam3d-prompt`

The prompt should name the object to segment, for example:

```text
metal shelves
industrial control cabinet
red plastic crate
```

Prefer a concise English object name with only useful visual qualifiers. Avoid actions, physical properties, target dimensions, or long generation prose because this prompt is used only for 2D segmentation. The same prompt is applied to every view. A view with no valid mask is excluded; the run fails if no view produces a valid mask.

### `--sam3d-seed` and `--sam3d-steps`

- Keep `--sam3d-seed 42` fixed while comparing image sets or prompts so the source of a change is easier to identify.
- Try another seed when geometry is unstable; different seeds can produce different details.
- `--sam3d-steps 50` is the current baseline. More steps increase stage-1 inference time and do not guarantee better geometry.
- This option does not control Hunyuan ReduceFace density or Isaac Sim collision quality.

## Downstream Options

| Option | Meaning |
| --- | --- |
| `--refine-config-path` | Hunyuan ReduceFace and local Blender postprocess config. Production uses `configs/hunyuan_reduce_local_postprocess.yaml`. |
| `--refine-temp-upload` | Temporary upload provider for local GLBs. `uguu` makes a local file downloadable by the Hunyuan API. |
| `--skip-refine` | Skips Hunyuan refine. Use only for diagnostics; normal SAM3D assets should retain refine. |
| `--len-x`, `--len-y`, `--len-z` | Refined asset target size along X/Y/Z in meters. |
| `--orientation` | Bounding-box longest/middle/shortest axis mapping, such as `X=L,Y=M,Z=S`. |
| `--approx` | Isaac Sim collision approximation, such as `sdf`. |
| `--set-mass` | Total final asset mass in kilograms. |

## Outputs

A single-view workspace usually contains:

```text
sam3d_downloads/sam3d/<source-name>/
├── image.png
├── 0.png
├── result_obj0.glb
├── result_obj0.obj
├── result_obj0.ply
└── result_obj0_refined_mesh/ or result_refined_mesh/
```

A multi-view workspace usually contains:

```text
sam3d_downloads/sam3d/<source-name>/
├── images/
│   ├── 00000.png
│   └── 00001.png
├── masks/
│   ├── 00000.png
│   └── 00001.png
├── result.glb
├── result.obj
├── result.ply
└── result_refined_mesh/
```

The GLB selection order is `scene_combined.glb`, `result.glb`, `result_obj0.glb`, followed by any other GLB in the workspace. The selected path is stored as `generation.selected_glb` in `sam3d_pipeline_result.json`.

## Resume From An Existing SAM3D GLB

When SAM3D has generated a GLB but a later Hunyuan refine request fails, reuse that GLB instead of reconstructing the images again:

```bash
python ./run_asset_pipeline.py \
  --sam3d-glb ./sam3d_downloads/sam3d/sam3d_images/result.glb \
  --intermediate-output-dir ./sam3d_output_intermediate \
  --final-output-dir ./sam3d_output_final \
  --result-json ./sam3d_pipeline_result.json \
  --refine-config-path ./configs/hunyuan_reduce_local_postprocess.yaml \
  --refine-temp-upload uguu \
  --len-x 0.4 \
  --len-y 0.3 \
  --len-z 0.8 \
  --orientation "X=L,Y=M,Z=S" \
  --approx sdf
```

## Troubleshooting

| Log or symptom | Cause and action |
| --- | --- |
| `No module named 'sam3.model'` | The environment is incomplete or the SAM3 submodule is missing. Activate `hunyuan_sam3d` and inspect `tools/sam3d/third_party/sam-3d-objects/submodules/sam3`; check `SAM3D_SINGLE_VIEW_ROOT` only for a custom checkout. |
| `SAM3 segmentation produced no valid masks` | The prompt did not identify the target reliably. Check spelling, use a more direct object name, and verify that the target is clearly visible. |
| `Multi-view SAM3D script not found` | The multi-view checkout is not at the default location. Set `SAM3D_MULTI_VIEW_ROOT`. |
| `FailedOperation.RequestTimeout` | The Hunyuan ReduceFace backend timed out temporarily. The current config retries the submission automatically; the existing `result.glb` can also be resumed with `--sam3d-glb`. |
| Pillow, spconv, or AMP `Warning` messages | These are normally compatibility/deprecation warnings. They are not the failure cause when the run ends with `SAM3D reconstruction finished`. |
