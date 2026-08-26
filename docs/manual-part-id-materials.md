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
continue the same request. Visual-material checkpoints are reused only after
their input and output hashes pass validation.

Optional physics arguments are `--material`, `--approx`, `--sdf-resolution`,
`--set-mass`, and `--cad-option KEY=VALUE`. STEP/STP does not use target-size
arguments.

## What the command does

```text
STEP/STP -> USD and physics preparation -> camera registration
         -> isolated model-image template for each visible Part-ID
         -> SAM3 + EntitySeg first pass
         -> neighbour-guided second pass and iterative mask fusion
         -> colour-blind material identity selection
         -> reviewed colour calibration where eligible
         -> apply materials to every Mesh -> final validation
```

SAM3 foreground confirmation is the only required human step. The current
mainline is generic: it has no asset name, Part-ID list, view-specific prompt,
or hand-authored material mapping. All registered views must contribute real
evidence. Local image fitting changes only 2-D proposals and never modifies
CAD, a Mesh transform, or the delivered camera.

Hidden or uncertain parts receive the configured default material. An exact
library preset remains unchanged; a corresponding-material assignment may
change only a reviewed colour input after material identity is fixed. A local
colour-quality rejection keeps the best measured result and records `REVIEW`
instead of aborting the complete run. Broken hashes, incomplete Part-ID
coverage, identity changes, or invalid delivery data still fail closed.

## Outputs

```text
RUN_ROOT/{cad_usd,intermediate,visual_material,final}/
RUN_ROOT/pipeline_result.json
```

The evidence and selection audits are under `RUN_ROOT/visual_material/analysis/`:

```text
part_id_cad_amodal_templates/manifest.json
part_id_relation_guidance/request.json
part_id_hybrid_masks/manifest.json
part_id_reference_evidence.json
part_id_material_plan.json
```

See the [detailed workflow](./modules/visual-materials.md),
[environment template](../.env.example), and
[third-party notices](../THIRD_PARTY_NOTICES.md). MVInverse is non-commercial.
