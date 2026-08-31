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
- separately installed Isaac Sim;
- locally installed Qwen3.5, SAM3, MVInverse, SigLIP2, and DINOv2 weights;
- a mounted NVIDIA MDL material library;
- Tencent Cloud credentials only for Hunyuan generation or refinement stages.

The complete local material workflow currently targets a GPU with about 24 GB
of VRAM.
Models, NVIDIA MDL assets, vendor runtimes, and credentials are not included;
the Base observation bank is bundled. Dependencies are loaded per workflow:
STEP/STP without visual
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

## `.env` setup

The program loads `.env` from the repository root automatically; do not run
`source .env`. Configure it in three steps.

1. Create the file if it does not exist:

```bash
cp .env.example .env
```

2. Accept access for [SAM3](https://huggingface.co/facebook/sam3) and
[EntitySeg](https://huggingface.co/datasets/qqlu1992/Adobe_EntitySeg), then download
the dependencies:

```bash
hf auth login
qwen-material setup-models --model-root /data/hunyuan-models
```

`--model-root` is the model/tool storage directory. It may be changed to any
directory with at least 30 GB free. The command installs Blender, prepares
Qwen3.5, MVInverse, SAM3, EntitySeg, SigLIP2, and DINOv2, and writes their paths
to `.env`. Do not fill those paths manually. Re-run the same command to resume
an interrupted download.

3. Open `.env` and fill only these two values:

```dotenv
ISAAC_PYTHON=/opt/isaacsim/python.sh
VISUAL_MATERIAL_ROOT=/data/NVIDIA/Materials
```

- `ISAAC_PYTHON`: install [Isaac Sim](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_workstation.html),
  then use the absolute path to `python.sh` in its installation directory.
- `VISUAL_MATERIAL_ROOT`: download and extract NVIDIA's **Base Materials Pack**
  from [Downloadable Asset Packs](https://docs.omniverse.nvidia.com/usd/latest/usd_content_samples/downloadable_packs.html),
  then use the `Materials` directory that directly contains `Base/`.

Tencent Cloud credentials are needed only for Hunyuan generation/refinement.
Set `SAM3D_*` only when using SAM3D reconstruction and automatic discovery
fails. Leave the other optional values blank. See [.env.example](./.env.example)
for the template and [docker/README.md](./docker/README.md) for container paths.

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

Use 2–4 photographs of the same workpiece to automate part matching, material
selection, colour adjustment, and USD binding. The only manual step is
confirming the whole-workpiece foreground; see
[automatic materials](./docs/guides/manual-part-id-materials.md).

See [Hunyuan](./docs/modules/hunyuan.md), [SAM3D](./docs/modules/sam3d.md),
[CAD](./docs/modules/cad.md), the
[generation guide](./docs/guides/generation-guide.md).

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
docs/{guides,development,modules}/
                                detailed documentation
legal/                          third-party license inventory
tests/                          tests
outputs/                        generated runs
```

Key references:

- [Documentation index](./docs/README.md)
- [Architecture](./docs/development/architecture.md)
- [Visual materials](./docs/modules/visual-materials.md)

## Test

```bash
python -m pytest -q -p no:cacheprovider tests
python -m pytest -q -p no:cacheprovider tools/qwen_material_pipeline/tests
```

## License

First-party code and documentation use
[Apache License 2.0](./LICENSE). MVInverse is non-commercial, and models,
NVIDIA software, MDL materials, and generated assets retain separate terms.
See the [legal-file index](./legal/README.md),
[third-party notices](./legal/THIRD_PARTY_NOTICES.md).
