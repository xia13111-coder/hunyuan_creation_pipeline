# Hunyuan Asset Creation Pipeline

[English](./README.md) | [中文](./README.zh.md) | [Documentation](./docs/README.md)

Convert images, GLBs, or hand-authored STEP/STP assemblies into
USD assets with optional mesh refinement, Isaac Sim physics, and
reference-driven NVIDIA MDL materials.

## Workflows

| Input | Output |
| --- | --- |
| Hunyuan images | generated/refined GLB and USD |
| SAM3D images | reconstructed GLB and USD |
| Existing GLB | refined, oriented USD |
| STEP/STP | dimension-preserving USD with optional physics/materials |

The STEP/STP material workflow aligns CAD renders with 2–4 photographs, evaluates
NVIDIA `Materials/Base` candidates per CAD Part-ID, and validates the final USD.
If alignment or visual evidence is insufficient, it stops and reports the
failed check.

## Requirements

- Linux x86-64 and an NVIDIA CUDA GPU;
- Conda with Python 3.10;
- separately installed Blender and Isaac Sim;
- locally installed Qwen3.5, SAM3, MVInverse, SigLIP2, and DINOv2 weights, plus
  a prepared NVIDIA Base observation bank;
- a mounted NVIDIA MDL material library;
- Tencent Cloud credentials only for Hunyuan generation or refinement stages.

The complete local material workflow currently targets a GPU with about 24 GB
of VRAM.
Models, NVIDIA assets, vendor runtimes, and credentials are not included in the
source release. Dependencies are loaded per workflow: STEP/STP without visual
materials does not require Qwen, SAM3, MVInverse, SigLIP2, DINOv2, or the NVIDIA
MDL library.

## Install

```bash
git clone https://github.com/xia13111-coder/hunyuan_creation_pipeline.git
cd hunyuan_creation_pipeline

conda env create -f environment.yml
conda activate hunyuan_sam3d
python -m pip install -e . -e ./tools/qwen_material_pipeline
cp .env.example .env
```

Fill local paths or credentials in `.env`. Pipeline commands load this file
automatically; variables already set in the environment take precedence. Keep
all local model paths there:

| Module | `.env` variables |
| --- | --- |
| Qwen / Qwen3.5 | `QWEN_MODEL_PATH`, `QWEN35_MODEL_PATH` |
| MVInverse | `MVINVERSE_REPOSITORY`, `MVINVERSE_CHECKPOINT` |
| SAM3 | `SAM3_REPOSITORY`, `SAM3_CHECKPOINT` |
| EntitySeg / CropFormer | `ENTITYSEG_PYTHON`, `ENTITYSEG_CROPFORMER_ROOT`, `ENTITYSEG_CONFIG`, `ENTITYSEG_CHECKPOINT` |
| SAM3D | `SAM3D_SINGLE_VIEW_ROOT`, `SAM3D_MULTI_VIEW_ROOT`, `SAM3D_PIPELINE_CONFIG`, `SAM3D_MOGE_CHECKPOINT`, `SAM3D_DINOV2_REPOSITORY`, `SAM3D_DINOV2_CHECKPOINT` |
| Material retrieval | `SIGLIP2_MODEL_PATH`, `DINOV2_MODEL_PATH` |

EntitySeg runs in its own CropFormer/Detectron2 environment. Install the
pipeline-owned compatibility layer into that environment (not into
`~/.local`) so the isolated child process also works with
`PYTHONNOUSERSITE=1`:

```bash
"$ENTITYSEG_PYTHON" -m pip install \
  -r tools/qwen_material_pipeline/requirements-entityseg.txt
PYTHONNOUSERSITE=1 "$ENTITYSEG_PYTHON" -c \
  'import black, cloudpickle, mmcv, yapf'
```

The runtime fixes `PIPELINE_LOCAL_MODELS_ONLY=1`. Normal local inference does
not download weights and fails clearly when a path is missing or
incomplete. Hunyuan generation and ReduceFace are Tencent Cloud APIs, so they
still require network access and credentials.

Verify the entry points:

```bash
hunyuan-asset-pipeline --help
manual-material-pipeline --help
qwen-material --help
```

Container deployment is documented in [docker/README.md](./docker/README.md).

## Quick start

Run these commands from the repository root and use a separate output directory
for each job.

### Hunyuan image generation

```bash
RUN=./outputs/hunyuan/basket
hunyuan-asset-pipeline \
  --input-dir ./input/basket_images \
  --output-dir "$RUN/generation" \
  --intermediate-output-dir "$RUN/intermediate" \
  --final-output-dir "$RUN/final" \
  --result-json "$RUN/pipeline_result.json" \
  --refine-config-path ./configs/refinement/hunyuan_reduce_local_postprocess.yaml \
  --refine-temp-upload uguu \
  --len-x 0.4 --len-y 0.3 --len-z 0.3 \
  --material plastic --approx sdf
```

Hunyuan accepts only a local image directory or one public image URL; text-only
generation is not supported. Replace `--input-dir` with `--image-url URL` for a
public image.

### SAM3D image reconstruction

Images in the input directory must show the same object:

```bash
RUN=./outputs/sam3d/cabinet
hunyuan-asset-pipeline \
  --sam3d-input ./input/cabinet_views \
  --sam3d-mode auto \
  --sam3d-prompt "industrial cabinet" \
  --output-dir "$RUN/generation" \
  --intermediate-output-dir "$RUN/intermediate" \
  --final-output-dir "$RUN/final" \
  --result-json "$RUN/pipeline_result.json" \
  --refine-config-path ./configs/refinement/hunyuan_reduce_local_postprocess.yaml \
  --refine-temp-upload uguu \
  --len-x 0.4 --len-y 0.3 --len-z 0.8 \
  --material steel --approx sdf
```

### Existing GLB

```bash
RUN=./outputs/glb/model
hunyuan-asset-pipeline \
  --existing-glb ./input/model.glb \
  --intermediate-output-dir "$RUN/intermediate" \
  --final-output-dir "$RUN/final" \
  --result-json "$RUN/pipeline_result.json" \
  --refine-config-path ./configs/refinement/hunyuan_reduce_local_postprocess.yaml \
  --refine-temp-upload uguu \
  --len-x 0.4 --len-y 0.3 --len-z 0.8 \
  --material plastic --approx sdf
```

Add `--skip-refine` when the GLB is already refined.

### STEP/STP without visual materials

This path converts CAD to USD, adds physics, and collects dependencies. It does
not invoke Qwen, MVInverse, or the reference-image workflow:

```bash
RUN=./outputs/manual/asset_physics
hunyuan-asset-pipeline \
  --manual-stp ./input/asset.stp \
  --cad-usd-output-dir "$RUN/cad_usd" \
  --intermediate-output-dir "$RUN/intermediate" \
  --final-output-dir "$RUN/final" \
  --result-json "$RUN/pipeline_result.json" \
  --material steel \
  --approx sdf \
  --manual-sdf-resolution 32
```

### STEP/STP with reference-image materials

First confirm the whole-workpiece foreground in each reference image:

```bash
qwen-material sam3-foreground-ui \
  --reference front=./references/front.jpg \
  --reference side=./references/side.jpg \
  --reference top=./references/top.jpg \
  --reference iso=./references/iso.jpg \
  --output ./annotations/sam3_foreground_annotations.json
```

Then run the automatic pipeline:

```bash
manual-material-pipeline \
  --stp ./input/asset.stp \
  --sam3-annotations ./annotations/sam3_foreground_annotations.json \
  --output ./outputs/manual/asset_run
```

STEP/STP dimensions are preserved. Use a new output directory for a from-zero
run; `--resume` is only for the same `live` request, and reusable visual stages
must pass their hash checks.

See [Hunyuan](./docs/modules/hunyuan.md), [SAM3D](./docs/modules/sam3d.md),
[CAD](./docs/modules/cad.md), the
[generation guide](./docs/guides/generation-guide.md), and
[CAD visual materials](./docs/guides/manual-part-id-materials.md).

## Outputs

```text
RUN_ROOT/
├── generation/       # Hunyuan/SAM3D only
├── cad_usd/           # STEP/STP only
├── intermediate/
├── visual_material/   # automatic visual materials only
├── final/
└── pipeline_result.json
```

Each workflow creates only the directories it uses.

Outputs may contain local paths, photographs, or private input data. Git ignores
them by default; review them before sharing.

## Repository

```text
asset_pipeline/                 orchestration and workflows
asset_refiner/                  mesh refinement
tools/{blender,isaac,sam3d}/    vendor-runtime workers
tools/qwen_material_pipeline/   material inference and USD tools
configs/                        versioned configuration
requirements/                   purpose-specific dependency overlays
docs/{guides,development,release,modules}/
                                detailed documentation
legal/                          third-party license inventory
.github/                        contribution and security policies
tests/                          tests
outputs/                        generated runs
```

Key references:

- [Documentation index](./docs/README.md)
- [Architecture](./docs/development/architecture.md)
- [Visual materials](./docs/modules/visual-materials.md)
- [Public release checklist](./docs/release/public-release-checklist.md)
- [Changelog](./CHANGELOG.md)

## Test

```bash
python -m pytest -q -p no:cacheprovider tests
PYTHONPATH=./tools python -m pytest -q -p no:cacheprovider \
  tools/qwen_material_pipeline/tests
python ./tools/release/check_public_tree.py
```

## License

First-party code and documentation use
[Apache License 2.0](./LICENSE). MVInverse is non-commercial, and models,
NVIDIA software, MDL materials, and generated assets retain separate terms.
See the [legal-file index](./legal/README.md),
[third-party notices](./legal/THIRD_PARTY_NOTICES.md),
[contribution guide](./.github/CONTRIBUTING.md), and
[security policy](./.github/SECURITY.md).
