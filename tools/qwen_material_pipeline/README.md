# Automatic material toolkit

[English](./README.md) | [中文](./README.zh.md) | [Project README](../../README.md)

Reference-driven NVIDIA MDL retrieval and USD material tools for hand-authored
STEP/STP assets. Most users should run the root `manual-material-pipeline`
command; this package supplies its material stages.

## Scope

The default configuration is `configs/pipeline/manual_part_id_materials.json`.
It predicts material identity before colour, selects an NVIDIA
`Materials/Base` entry per CAD Part-ID, preserves exact presets, and runs
identity-preserving actual-CAD colour calibration only for corresponding
materials.

This package handles reference-image evidence, camera registration, Part-ID
mapping, material retrieval, USD binding, and validation. CAD conversion and the
top-level workflow remain in `asset_pipeline`.

## Setup

```bash
conda activate hunyuan_sam3d
python -m pip install -e . -e ./tools/qwen_material_pipeline
qwen-material --help
```

Set local model and application paths in `.env`; copy `.env.example` as a
starting point. Do not commit the populated file.

Optional verified Qwen3.5/SigLIP2 setup:

```bash
bash tools/qwen_material_pipeline/scripts/setup_qwen35_runtime.sh
```

SAM3, MVInverse, DINOv2, NVIDIA materials, and the Base observation bank remain
separate local dependencies and must pass preflight.

## Pipeline

```text
normalized CAD + confirmed photos
  -> registered RGB/Part-ID evidence
  -> SAM3/MVInverse/SigLIP2/DINOv2/Qwen3.5
  -> Base MDL candidate renders
  -> one assignment for every Part-ID
  -> record the selected MDL, bind it in USD, and validate the result
```

Each visible Part-ID is evaluated independently. Invisible parts receive a
configured default material. Models rank candidates; validated code performs
the USD binding. Exact matches keep library defaults. Corresponding materials
keep the selected MDL identity while the main pipeline renders and selects only
reviewed colour parameters per part/component scope.

## Commands

| Command | Purpose |
| --- | --- |
| `sam3-foreground-ui` | confirm whole-workpiece foreground |
| `staged` | run the material inference stages |
| `catalog` | build/inspect the NVIDIA MDL catalog |
| `base-bank` | build/verify Base observations |
| `part-id-qwen` | rank candidates per Part-ID |
| `exact-mdl-tournament` | render-compare MDL candidates |
| `compare` | compare references and renders |
| `final-visual-gate` | validate collected USD |
| `usd` | USD part indexing, rendering, binding, validation |

`qwen-material --help` is the command reference. Most users only need
`manual-material-pipeline` and `sam3-foreground-ui`.

## Resume and outputs

`--resume` reuses an artifact only when its inputs, configuration, schema, model,
and hashes still match.

Important run artifacts:

```text
visual_material/renders/
visual_material/analysis/{reference_manifest,qwen_inference_ledger,
  part_id_reference_evidence,part_id_qwen_choices,material_selection_lock}.json
visual_material/analysis/mvinverse/
visual_material/visual_quality/
visual_material/final_visual_acceptance/
```

Do not include local results, caches, models, or workspaces in a source release.

## Documentation and tests

- [User command](../../docs/manual-part-id-materials.md)
- [Behavior and troubleshooting](../../docs/modules/visual-materials.md)
- [Architecture (Chinese)](./docs/architecture.zh.md)
- [MVInverse (Chinese)](./docs/mvinverse.zh.md)

```bash
PYTHONPATH=./tools python -m pytest -q -p no:cacheprovider \
  tools/qwen_material_pipeline/tests
```

## License

First-party code uses [Apache License 2.0](./LICENSE). MVInverse remains
non-commercial; models and NVIDIA assets retain separate terms. See
[third-party notices](../../THIRD_PARTY_NOTICES.md).
