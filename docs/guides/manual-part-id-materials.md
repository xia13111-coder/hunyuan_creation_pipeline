# STEP/STP automatic materials

[English](./manual-part-id-materials.md) | [中文](./manual-part-id-materials.zh.md)

The workflow uses 2–4 photographs of the same workpiece to select an NVIDIA
Base MDL for each CAD Part-ID. Photographs provide appearance only; CAD
geometry, dimensions, and assembly relationships are unchanged.

## Run

First point-select and confirm the whole-workpiece foreground in each image:

```bash
qwen-material sam3-foreground-ui \
  --reference front=./references/front.jpg \
  --reference side=./references/side.jpg \
  --output ./annotations/sam3_foreground_annotations.json
```

Then run:

```bash
manual-material-pipeline \
  --stp ./input/asset.stp \
  --sam3-annotations ./annotations/sam3_foreground_annotations.json \
  --output ./outputs/manual/asset_run
```

The program handles camera alignment, part segmentation, material selection,
eligible colour adjustment, USD binding, and final validation. The only manual
step is whole-workpiece foreground confirmation; per-part annotation is not
required.

## Results

```text
RUN_ROOT/pipeline_result.json
RUN_ROOT/visual_material/
RUN_ROOT/final/
```

Add `--resume` to continue the same output directory. On failure, inspect the
first `FAILED` message; free GPU memory or disk space before resuming when
needed. MVInverse is non-commercial only. See
[third-party notices](../../legal/THIRD_PARTY_NOTICES.md) for other licenses.
