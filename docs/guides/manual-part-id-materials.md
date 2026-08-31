# STEP/STP automatic materials

[English](./manual-part-id-materials.md) | [中文](./manual-part-id-materials.zh.md)

This workflow uses 2–4 photographs of the same workpiece to select and bind an
NVIDIA Base MDL for every CAD Part-ID in a STEP/STP assembly. Photographs
provide appearance only; CAD remains authoritative for geometry, dimensions,
and assembly relationships.

## Before you start

- one `.stp` or `.step` file;
- 2–4 views of the same workpiece;
- Isaac Sim, NVIDIA Base Materials, and about 24 GB of VRAM;
- local models prepared by `qwen-material setup-models`.

See the project [README](../../README.md) for installation, automatic model
downloads, and `.env` setup. Verify these entry points after installation:

```bash
hunyuan-asset-pipeline --help
manual-material-pipeline --help
qwen-material --help
```

## Run

### 1. Confirm the whole-workpiece foreground

Point-select and confirm the whole workpiece in every image. View names may be
customized but must be unique.

```bash
qwen-material sam3-foreground-ui \
  --reference front=./references/front.jpg \
  --reference side=./references/side.jpg \
  --reference top=./references/top.jpg \
  --reference iso=./references/iso.jpg \
  --output ./annotations/sam3_foreground_annotations.json
```

Only the whole workpiece is annotated; per-part annotation is not required.
The file records image paths, view names, and content hashes. Recreate it after
changing any image.

### 2. Run automatic material assignment

```bash
manual-material-pipeline \
  --stp ./input/asset.stp \
  --sam3-annotations ./annotations/sam3_foreground_annotations.json \
  --output ./outputs/manual/asset_run
```

Common optional arguments:

| Argument | Meaning |
| --- | --- |
| `--resume` | Continue the same inputs and output, reusing only hash-verified stages. |
| `--material NAME` | Physics-material preset; default is `plastic`. |
| `--approx NAME` | Collision approximation; default is `sdf`. |
| `--sdf-resolution N` | SDF resolution; default is `32`. |
| `--set-mass KG` | Set the total workpiece mass. |
| `--cad-option KEY=VALUE` | Pass a repeatable CAD Converter option. |
| `--config FILE` | Use another automatic-material configuration. |

Use a new output directory for a new job. Add `--resume` to an existing output
only when the input, configuration, and models are unchanged. STEP/STP retains
engineering dimensions, so target-size and orientation arguments are not used.

## Core workflow

```text
STEP/STP
  -> CAD to USD, then unit, origin, and physics preparation
  -> render the CAD assembly and isolated model-image Part-ID templates
  -> align the CAD assembly camera with the photograph cameras
  -> locate photo parts from model-image shape and assembly position
  -> two-pass SAM3 + EntitySeg segmentation and iterative fusion
  -> MVInverse, SigLIP2, DINOv2, and Qwen candidate generation/ranking
  -> lock material identity, then calibrate eligible corresponding colours
  -> bind every Mesh and inspect actual-CAD renders
  -> collect dependencies and validate the final USD
```

Important rules:

- Part-ID shape comes from CAD model images. The pipeline does not simply
  translate a CAD mask over a photograph; it re-estimates photo location from
  camera alignment, neighboring-part relationships, and segmentation.
- Local fitting changes only 2-D photo candidates. It never moves an individual
  Mesh. Pose changes apply to the whole rigid assembly and camera parameters.
- SAM3 proposes regions, EntitySeg contributes object boundaries, and CAD model
  images constrain shape. The result combines all three rather than choosing
  one segmentation output unchanged.
- Material identity is selected before colour. A trustworthy exact library
  preset is retained; only corresponding materials with reviewed colour inputs
  are eligible for colour adjustment.
- Colour candidates are compared through actual-CAD renders. If local colour
  quality remains low, the best measured candidate is retained and marked
  `REVIEW`; invalid delivery data or a changed material identity still fails.
- Every Mesh receives a result. Parts that are hidden or cannot be inferred
  reliably receive a safe default instead of fabricated photo evidence.
- The mainline contains no asset names, fixed Part-ID lists, view-specific
  handwritten prompts, or manual material mappings.

## Outputs

| Path | Contents |
| --- | --- |
| `RUN_ROOT/cad_usd/` | Raw CAD Converter USD. |
| `RUN_ROOT/intermediate/` | USD after unit, geometry, and physics preparation. |
| `RUN_ROOT/visual_material/renders/` | Assembly and Part-ID CAD renders. |
| `RUN_ROOT/visual_material/camera_calibration/` | Camera search, alignment, and checks. |
| `RUN_ROOT/visual_material/analysis/` | Segmentation, retrieval, selection, and colour evidence. |
| `RUN_ROOT/visual_material/visual_quality/` | Reference-versus-render comparisons. |
| `RUN_ROOT/final/` | Collected USD that passed final checks. |
| `RUN_ROOT/pipeline_result.json` | Inputs, status, and primary result paths. |

For a single-part investigation, start with:

```text
visual_material/analysis/part_id_cad_amodal_templates/manifest.json
visual_material/analysis/part_id_relation_guidance/request.json
visual_material/analysis/part_id_hybrid_masks/manifest.json
visual_material/analysis/part_id_reference_evidence.json
visual_material/analysis/part_id_material_plan.json
visual_material/analysis/material_selection_lock.json
```

## Resume and failures

Inspect the first `FAILED` line. Later exceptions are often consequences of the
first failed stage.

| Situation | Action |
| --- | --- |
| `CUDA out of memory` | Stop other GPU processes, then use `--resume` on the same output. |
| `No space left on device` | Free disk space or move the run to a larger disk, then continue. |
| SAM3 annotation/policy mismatch | Recreate annotations after image changes; do not mix a new annotation with an old run. |
| Incomplete camera or Part-ID evidence | Inspect `camera_calibration/` and Part-ID comparisons; do not skip failed views. |
| EntitySeg deterministic warning | This is a reproducibility warning; use the later traceback and `FAILED` status to determine failure. |

See [Architecture](../development/architecture.md) for the code call graph.
MVInverse is non-commercial only; see
[third-party notices](../../legal/THIRD_PARTY_NOTICES.md) for other licenses.
