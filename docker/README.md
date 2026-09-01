# Complete Docker Setup

This deployment runs the current mainline source with the API, Blender, Isaac
Sim, Qwen3.5, MVInverse, SAM3, EntitySeg, DINOv2, SigLIP2, and NVIDIA Base
material workflow. Large models and material libraries stay on the host and are
mounted read-only. Inputs, outputs, and caches belong on a writable data disk.

## Prerequisites

Use Linux with an NVIDIA GPU, Docker Engine, Docker Compose v2, and NVIDIA
Container Toolkit. Verify container GPU access first:

```bash
docker run --rm --gpus all --entrypoint nvidia-smi \
  hunyuan-allinone:isaac-6.0.1-materials -L
```

If the image is absent, download every part and the SHA256 manifest from the
[Docker offline-image release](https://github.com/xia13111-coder/hunyuan_creation_pipeline/releases/tag/docker-isaac-6.0.1),
then import it:

```bash
cd docker/offline-images
sha256sum -c hunyuan-pipeline-isaac-6.0.1-offline.parts.sha256
cat hunyuan-pipeline-isaac-6.0.1-offline.tar.part-* | docker load
cd ../..
```

## First start

```bash
docker/run-full.sh init
${EDITOR:-nano} docker/.env.runtime
docker/run-full.sh preflight
docker/run-full.sh up
docker/run-full.sh smoke
```

`docker/.env.runtime` is read by Compose and is separate from the root `.env`.
Every path below is an absolute host path. Compose mounts external runtimes and
models read-only at the same absolute path, so do not use `$HOME`, relative
paths, or container-only paths.

### Images and containers

| Variable | Default/value | Purpose and source |
| --- | --- | --- |
| `PIPELINE_IMAGE` | `hunyuan-allinone:isaac-6.0.1-materials` | Final runtime image. Import the offline release above or build it with `docker/run-full.sh build` after both base images exist. |
| `PIPELINE_ENV_IMAGE` | `hunyuan-allinone:isaac-6.0.1` | Build-only Conda/dependency image, produced from `docker/Dockerfile`. It is not needed when running an already imported final image. |
| `ISAACSIM_BASE_IMAGE` | Local Isaac Sim 6.0.1 image tag | Build-only base. Pull the official image from [NVIDIA NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/-/containers/isaac-sim/-); `nvcr.io/nvidia/isaac-sim:6.0.1` is accepted. |
| `PIPELINE_CONTAINER_NAME` | `hunyuan-pipeline-601` | Pipeline/API container name; use the changed name in later `docker exec` commands. |
| `HUB_CONTAINER_NAME` | `isaac-hub-cache` | Isaac Hub cache container name. |
| `HUB_IMAGE` | `nvcr.io/nvidia/omniverse/hub_workstation_cache:2.0.0` | Started automatically. See the official [Hub Workstation Cache documentation](https://docs.omniverse.nvidia.com/utilities/latest/cache/hub-workstation.html); import it separately for offline use. |
| `PIPELINE_UID` / `PIPELINE_GID` | Host user/group IDs | Set to `id -u` and `id -g` so generated files belong to the host user. |
| `PIPELINE_MIN_FREE_GB` | `50` | Minimum free GiB required on `ASSET_ROOT` by host preflight; it is not a quota. |
| `PIPELINE_DOCKER_MIN_BUILD_FREE_GB` | `15` | Minimum free GiB required under Docker's data root before `build`; the candidate and current images coexist briefly. |
| `PIPELINE_START_TIMEOUT_SECONDS` | `300` | Maximum time in seconds for Compose to wait for a healthy service. |
| `PIPELINE_MIN_NOFILE` | `65536` | Minimum container soft open-file limit; full reference rendering opens many GPU/render resources. Compose sets the same `nofile` limit. |

To build instead of importing the final image, first prepare the base images:

```bash
docker build -f docker/Dockerfile \
  -t hunyuan-allinone:isaac-6.0.1 .
docker pull nvcr.io/nvidia/isaac-sim:6.0.1
```

Set `ISAACSIM_BASE_IMAGE=nvcr.io/nvidia/isaac-sim:6.0.1` in `.env.runtime`,
then run `docker/run-full.sh build`.

### Host directories

| Variable | Value | Mount and purpose |
| --- | --- | --- |
| `PROJECT_ROOT` | This repository root | Mounted read-only at `/workspace/hunyuan3.0_assets_creation` and at the identical absolute host path. The latter preserves the hash-bound reference-image paths stored in reviewed SAM3 annotation manifests. It must resolve to the current checkout. |
| `ASSET_ROOT` | Inputs, outputs, and run results | Writable at `/workspace/assets`; place it on a large data disk. |
| `MODEL_CACHE_ROOT` | Container home, model, Torch, and Hub cache root | Writable; the launcher creates `home/` and `ov-hub/`. This is not one model path. |
| `ISAAC_CACHE_ROOT` | Isaac Sim cache/log root | Writable; the launcher creates `cache/`, `logs/`, `config/`, `data/`, and `pkg/`. Nothing is downloaded manually into it. |

### External Python runtimes and models

| Variable | Must point to | Source |
| --- | --- | --- |
| `QWEN35_RUNTIME_DIR` | Qwen3.5 isolated runtime root | Created by `setup_qwen35_runtime.sh` documented in the root README; it is not the checkpoint directory. |
| `QWEN35_PYTHON` | Executable Python in that runtime | Normally `QWEN35_RUNTIME_DIR/env/bin/python`. It is mounted at the same host path inside the container. |
| `QWEN35_MODEL_PATH` | Qwen3.5-4B directory with `config.json` and safetensors | [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B); use the repository setup script for the pinned revision. |
| `MVINVERSE_REPOSITORY` | MVInverse source root | Compatible source is bundled; upstream: [Maddog241/mvinverse](https://github.com/Maddog241/mvinverse). |
| `MVINVERSE_CHECKPOINT` | Directory with `config.json` and `model.safetensors` | [maddog241/mvinverse](https://huggingface.co/maddog241/mvinverse). |
| `SAM3_REPOSITORY` | SAM3 source compatible with the checkpoint | Prefer SAM 3D Objects' `submodules/sam3`; upstream: [facebookresearch/sam3](https://github.com/facebookresearch/sam3). |
| `SAM3_CHECKPOINT` | `sam3.pt` | Gated [facebook/sam3](https://huggingface.co/facebook/sam3). |
| `SAM3D_SINGLE_VIEW_ROOT` | SAM 3D Objects source root | Clone [facebookresearch/sam-3d-objects](https://github.com/facebookresearch/sam-3d-objects) with submodules. |
| `SAM3D_MULTI_VIEW_ROOT` | Multi-view extension root | Clone [devinli123/MV-SAM3D](https://github.com/devinli123/MV-SAM3D). Complete Docker preflight requires this directory. |
| `SAM3D_PIPELINE_CONFIG` | Checkpoint package's `checkpoints/pipeline.yaml` | Gated [facebook/sam-3d-objects](https://huggingface.co/facebook/sam-3d-objects). |
| `SAM3D_MOGE_CHECKPOINT` | MoGe `model.pt` | [Ruicheng/moge-vitl](https://huggingface.co/Ruicheng/moge-vitl). |
| `SAM3D_DINOV2_REPOSITORY` | DINOv2 source containing `hubconf.py` | [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2). |
| `SAM3D_DINOV2_CHECKPOINT` | `dinov2_vitl14_reg4_pretrain.pth` | Meta's [official checkpoint](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth). |
| `SIGLIP2_MODEL_PATH` | SigLIP2 directory | [google/siglip2-base-patch16-224](https://huggingface.co/google/siglip2-base-patch16-224); downloaded by the Qwen3.5 setup script. |
| `DINOV2_MODEL_PATH` | Transformers DINOv2 directory with `config.json` | [facebook/dinov2-with-registers-large](https://huggingface.co/facebook/dinov2-with-registers-large). Do not use the SAM3D `.pth` path here. |

Exact download commands and directory examples are in the root
[`.env` section](../README.md#env-variables-and-downloads).

### EntitySeg environment

| Variable | Must point to | Source and reason |
| --- | --- | --- |
| `ENTITYSEG_RUNTIME_DIR` | EntitySeg virtual-environment root | Created locally; it should contain `bin/python` and dependencies. |
| `ENTITYSEG_BASE_RUNTIME_DIR` | Base Python prefix containing the resolved `ENTITYSEG_PYTHON` target | It must also be mounted when the virtual environment's Python is a symlink. |
| `ENTITYSEG_DETECTRON2_ROOT` | Root containing `detectron2/__init__.py` | Prepare the compatible revision using the [CropFormer instructions](https://github.com/qqlu/Entity/blob/main/Entityv2/CODE.md). |
| `ENTITYSEG_PYTHON` | Executable EntitySeg Python | It must import Detectron2, CropFormer, and the packages in `requirements/entityseg.txt`. |
| `ENTITYSEG_CROPFORMER_ROOT` | `Entity/Entityv2/CropFormer` | Clone [qqlu/Entity](https://github.com/qqlu/Entity). |
| `ENTITYSEG_CONFIG` | `cropformer_swin_tiny_3x.yaml` | Under CropFormer's `configs/entityv2/entity_segmentation/`. |
| `ENTITYSEG_CHECKPOINT` | `CropFormer_swin_tiny_3x_5cea5e.pth` | Authors' gated [CropFormer model directory](https://huggingface.co/datasets/qqlu1992/Adobe_EntitySeg/tree/main/CropFormer_model/Entity_Segmentation/CropFormer_swin_tiny_3x). |

### Materials, generated data, and controls

| Variable | Value | Source/default |
| --- | --- | --- |
| `VISUAL_MATERIAL_ROOT` | NVIDIA `Materials` directory containing `Base/` | Download and extract the **Base Materials Pack** from NVIDIA's [Downloadable Asset Packs](https://docs.omniverse.nvidia.com/usd/latest/usd_content_samples/downloadable_packs.html). |
| `NVIDIA_BASE_OBSERVATION_BANK` | Observation bank containing `index_manifest.json` | Generated locally from Isaac Sim, Base materials, SigLIP2, and DINOv2 using the root README command; mounted read-only. |
| `VISUAL_RETRIEVAL_CACHE` | Writable retrieval cache | Created locally. Change it after changing retrieval models or materials. |
| `PIPELINE_LOCAL_MODELS_ONLY` | `1` | Enforced; jobs never download missing models implicitly. |
| `ACCEPT_EULA` | `Y` | Accepts the Isaac Sim container license, as required by NVIDIA's container instructions. |
| `PRIVACY_CONSENT` | `Y` | NVIDIA container privacy/telemetry consent. Review NVIDIA's terms before setting it. |
| `PIPELINE_MAX_WORKERS` | `1` | Concurrent API jobs; keep `1` when heavy models share one GPU. |
| `PIPELINE_MAX_LOG_LINES` | `2000` | In-memory log lines retained per API job. |
| `REFINE_MESH_TEMP_UPLOAD` | `uguu` | Third-party temporary upload used for a local GLB before cloud refinement; avoid it for sensitive assets. |
| `TENCENTCLOUD_SECRET_ID` / `TENCENTCLOUD_SECRET_KEY` | Tencent Cloud API credentials | Create them in the [Tencent Cloud API key console](https://console.cloud.tencent.com/cam/capi). Leave blank for STEP/STP automatic materials. |
| `TENCENTCLOUD_REGION` | Region such as `ap-guangzhou` | Blank defaults to `ap-guangzhou`. |

Keep at least 50 GiB free on the output filesystem. Do not place `ASSET_ROOT`
or caches on a nearly full system disk.

The launcher creates writable directories, verifies all models and mounts,
checks GPU access, and starts the Isaac Hub cache. Missing required components
fail before a pipeline job is accepted.

## Usage

- Health: <http://127.0.0.1:8000/health>
- API documentation: <http://127.0.0.1:8000/docs>

For STEP/STP material assignment, put inputs under the host `ASSET_ROOT`; inside
the container the same data is always under `/workspace/assets`:

```bash
docker exec -it hunyuan-pipeline-601 manual-material-pipeline \
  --stp /workspace/assets/input/model.stp \
  --sam3-annotations /workspace/assets/input/sam3_foreground_annotations.json \
  --output /workspace/assets/runs/model \
  --visual-inference-mode live
```

Tencent credentials are optional for STEP/STP material assignment. They are
required only by cloud generation/refinement.

## Operations

```bash
docker/run-full.sh status       # container status
docker/run-full.sh logs         # follow logs
docker/run-full.sh shell        # container shell
docker/run-full.sh smoke        # complete runtime check
docker/run-full.sh down         # stop the API container
docker/run-full.sh up           # start and wait; do not recreate a healthy container
docker/run-full.sh build        # rebuild after dependency/Dockerfile changes
docker/run-full.sh replace      # remove an old same-name container and recreate
```

Source is mounted read-only. Run `replace` after Python changes to restart the
API process; use `build` only
after dependency or Dockerfile changes.

## Troubleshooting

| Symptom | Action |
| --- | --- |
| Restart loop | Run `preflight`, then inspect `logs` |
| `No space left on device` | Move outputs/caches to a data disk; remove old images only after review |
| CUDA OOM | Run one heavy model stage per GPU and stop unrelated GPU processes |
| EntitySeg cannot import Detectron2 | Check `ENTITYSEG_BASE_RUNTIME_DIR` and `ENTITYSEG_DETECTRON2_ROOT` |
| API is healthy but cloud generation fails | Configure Tencent secret ID, key, and region |

Never commit `docker/.env.runtime` or bake credentials into an image.
