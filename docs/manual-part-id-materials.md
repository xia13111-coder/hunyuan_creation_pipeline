# Automatic Part-ID materials for CAD

[English](./manual-part-id-materials.md) | [中文](./manual-part-id-materials.zh.md)

Use 2–4 photographs to assign NVIDIA Base MDLs to one STEP/STP assembly. CAD
continues to define geometry and dimensions.

## Run

Install the commands once:

```bash
conda activate hunyuan_sam3d
python -m pip install -e . -e ./tools/qwen_material_pipeline
```

The commands automatically load `.env` from the project root. Non-empty shell
variables take precedence.

1. Confirm the whole-workpiece foreground in each view:

```bash
qwen-material sam3-foreground-ui \
  --reference front=./references/front.jpg \
  --reference side=./references/side.jpg \
  --reference top=./references/top.jpg \
  --reference iso=./references/iso.jpg \
  --output ./annotations/sam3_foreground_annotations.json
```

2. Start a new unattended run:

```bash
manual-material-pipeline \
  --stp ./input/asset.stp \
  --sam3-annotations ./annotations/sam3_foreground_annotations.json \
  --output ./outputs/manual/asset_run
```

The annotation stores image hashes and view IDs. Recreate it after changing an
image. Use a new output directory to start from zero; add `--resume` only to
continue the same verified run.

Optional physics arguments are `--material`, `--approx`, `--sdf-resolution`,
`--set-mass`, and `--cad-option KEY=VALUE`. STEP/STP does not use target-size
arguments.

## What the command does

```text
STEP/STP -> USD and physics preparation -> camera registration
         -> evidence for each visible Part-ID
         -> NVIDIA Base retrieval and MDL render comparison
         -> apply materials to every Mesh -> final validation
```

SAM3 foreground confirmation is the only required human step. Local image
fitting does not modify CAD. Hidden or uncertain parts receive the configured
default material, and selected MDL parameters are not changed. The run stops
if alignment, assignment coverage, visual comparison, or final delivery checks
fail.

## Outputs

```text
RUN_ROOT/{cad_usd,intermediate,visual_material,final}/
RUN_ROOT/pipeline_result.json
```

See the [detailed workflow](./modules/visual-materials.md),
[environment template](../.env.example), and
[third-party notices](../THIRD_PARTY_NOTICES.md). MVInverse is non-commercial.
