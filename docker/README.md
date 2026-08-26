# Docker Operations

[English](./README.md) | [中文](./README.zh.md) | [Project README](../README.md)

The offline image contains the pipeline runtime, shared visual-model
dependencies, Blender 4.5.0, Isaac Sim 6.0.1, the CLI, and the HTTP API. The
separate Qwen3.5 runtime, model weights, NVIDIA Base observation bank, and
NVIDIA Materials remain host mounts. Paths under `/workspace`, `/isaac-sim`, `/opt/blender`, and
`/home/pipeline` are container paths and do not change between hosts.

## 1. Load the offline bundle

Download all 17 parts and the checksum manifest from
[Docker Offline Bundle - Isaac Sim 6.0.1](https://github.com/xia13111-coder/hunyuan_creation_pipeline/releases/tag/docker-isaac-6.0.1):

```text
hunyuan-pipeline-isaac-6.0.1-offline.tar.part-000 ... part-016
hunyuan-pipeline-isaac-6.0.1-offline.parts.sha256
```

The repository may require an authorized GitHub account. To download with the
[GitHub CLI](https://cli.github.com/):

```bash
mkdir -p hunyuan-docker-bundle
cd hunyuan-docker-bundle
gh auth login
gh release download docker-isaac-6.0.1 \
  --repo xia13111-coder/hunyuan_creation_pipeline \
  --pattern 'hunyuan-pipeline-isaac-6.0.1-offline.*'
```

Bundle details:

| Item | Value |
| --- | --- |
| Total size | `32,108,196,352` bytes (about 29.9 GiB) |
| Parts | 17; 1900 MiB each except the final part |
| Full image | `hunyuan-allinone:isaac-6.0.1-materials` |
| Full image ID | `sha256:913fe7c41298e99ec12701afdecc4a784f2e3f07b534c1b0bc9db77099f86855` |
| Hub image | `nvcr.io/nvidia/omniverse/hub_workstation_cache:2.0.0` |
| Tencent SDK | AI3D/common `3.0.1462` |

The host needs Linux, an RTX-capable NVIDIA GPU, Docker Engine, NVIDIA Container
Toolkit, `acl`, and enough disk space. Keep at least 80 GB free when the bundle
and Docker data root share a filesystem.

```bash
nvidia-smi
docker --version
docker info >/dev/null
sha256sum -c hunyuan-pipeline-isaac-6.0.1-offline.parts.sha256
cat hunyuan-pipeline-isaac-6.0.1-offline.tar.part-* | docker load
```

Do not load the bundle if any checksum fails. Verify the images and GPU access:

```bash
docker image inspect hunyuan-allinone:isaac-6.0.1-materials \
  --format '{{index .RepoTags 0}} {{.Id}}'
docker image inspect nvcr.io/nvidia/omniverse/hub_workstation_cache:2.0.0 \
  --format '{{index .RepoTags 0}} {{.Id}}'
docker run --rm --gpus all --entrypoint nvidia-smi \
  hunyuan-allinone:isaac-6.0.1-materials
```

Keep the checksum manifest with any archived copy of the bundle.

## 2. Prepare host directories

Run from the project root:

```bash
export PROJECT_ROOT="$(pwd -P)"
export RUNTIME_ROOT="${RUNTIME_ROOT:-$PROJECT_ROOT/docker/runtime}"
export ASSET_ROOT="${ASSET_ROOT:-$RUNTIME_ROOT/assets}"
export ISAAC_CACHE_ROOT="${ISAAC_CACHE_ROOT:-$RUNTIME_ROOT/isaac-sim-6.0.1}"
export MODEL_CACHE_ROOT="${MODEL_CACHE_ROOT:-$RUNTIME_ROOT/model-cache}"

mkdir -p "$ASSET_ROOT"/{input,sam3d-visualization}
mkdir -p "$ISAAC_CACHE_ROOT"/{cache/main,cache/computecache,config,data,logs,pkg}
mkdir -p "$MODEL_CACHE_ROOT"/{ov-hub,huggingface,torch-hub,models,mvinverse,nvidia-materials,retrieval,nvidia_base_observation_bank_v1}
```

| Variable | Default | Purpose |
| --- | --- | --- |
| `PROJECT_ROOT` | Current directory | Read-only source and configuration |
| `ASSET_ROOT` | `docker/runtime/assets` | Inputs and outputs |
| `ISAAC_CACHE_ROOT` | `docker/runtime/isaac-sim-6.0.1` | Writable Isaac Sim caches |
| `MODEL_CACHE_ROOT` | `docker/runtime/model-cache` | Model and Hub caches |

SAM3D additionally needs `tools/sam3d/third_party/` and populated Hugging Face
and Torch Hub caches. The image runs as UID/GID `1234:1234`; grant it access to
writable mounts:

```bash
command -v setfacl >/dev/null || sudo apt-get install -y acl
sudo chown -R "$(id -u):$(id -g)" "$ASSET_ROOT"
sudo chown -R 1234:1234 \
  "$ISAAC_CACHE_ROOT" "$MODEL_CACHE_ROOT/ov-hub" "$MODEL_CACHE_ROOT/retrieval"
find "$ASSET_ROOT" -type d -exec setfacl \
  -m "u:$(id -u):rwx,u:1234:rwx,d:u:$(id -u):rwx,d:u:1234:rwx" {} +
find "$ASSET_ROOT" -type f -exec setfacl \
  -m "u:$(id -u):rw-,u:1234:rw-" {} +
```

Put inputs in `$ASSET_ROOT/input`. Source, model weights, the observation bank,
and NVIDIA Materials are mounted read-only. Outputs and the Isaac, Hub, and
visual-retrieval caches are writable.

## 3. Configure and start

Create the local credential file:

```bash
cp docker/env.runtime.example docker/.env.runtime
chmod 600 docker/.env.runtime
${EDITOR:-nano} docker/.env.runtime
```

Do not commit `.env.runtime`. Tencent credentials are required for Hunyuan and
refine jobs, but not for manual STEP/STP or existing GLB jobs with
`--skip-refine`.

Start the Isaac Hub cache. Create the container only if it does not exist:

```bash
docker start isaac-hub-cache 2>/dev/null || docker run --name isaac-hub-cache \
  --restart unless-stopped --network=host -u 1234:1234 \
  -v "$MODEL_CACHE_ROOT/ov-hub:/var/cache/hub:rw" \
  -d nvcr.io/nvidia/omniverse/hub_workstation_cache:2.0.0
```

Isaac Sim can exit or segfault when Hub is unavailable.

Set the model, observation-bank, cache, and material paths. When
`ASSET_MODEL_VOLUME="$MODEL_CACHE_ROOT"` is set, the Qwen3.5 setup script creates
the `env/` and `model/` directories under this runtime root. The runtime is
mounted at the same absolute path so its Python environment keeps its original
prefix.

```bash
export QWEN35_RUNTIME_DIR="${QWEN35_RUNTIME_DIR:-$MODEL_CACHE_ROOT/qwen35_4b_runtime}"
export QWEN35_PYTHON="${QWEN35_PYTHON:-$QWEN35_RUNTIME_DIR/env/bin/python}"
export QWEN35_MODEL_DIR="${QWEN35_MODEL_DIR:-$QWEN35_RUNTIME_DIR/model}"
export SIGLIP2_MODEL_DIR="${SIGLIP2_MODEL_DIR:-$MODEL_CACHE_ROOT/models/siglip2-base-patch16-224}"
export DINOV2_MODEL_DIR="${DINOV2_MODEL_DIR:-$MODEL_CACHE_ROOT/models/dinov2-with-registers-large}"
export MVINVERSE_MODEL_DIR="${MVINVERSE_MODEL_DIR:-$MODEL_CACHE_ROOT/mvinverse/model}"
export NVIDIA_MATERIAL_DIR="${NVIDIA_MATERIAL_DIR:-$MODEL_CACHE_ROOT/nvidia-materials}"
export NVIDIA_BASE_BANK_DIR="${NVIDIA_BASE_BANK_DIR:-$MODEL_CACHE_ROOT/nvidia_base_observation_bank_v1}"
export VISUAL_RETRIEVAL_CACHE_DIR="${VISUAL_RETRIEVAL_CACHE_DIR:-$MODEL_CACHE_ROOT/retrieval}"

test -x "$QWEN35_PYTHON"
test -f "$QWEN35_MODEL_DIR/config.json"
test -f "$SIGLIP2_MODEL_DIR/config.json"
test -f "$DINOV2_MODEL_DIR/config.json"
test -f "$MVINVERSE_MODEL_DIR/config.json"
test -d "$NVIDIA_MATERIAL_DIR/Base"
test -f "$NVIDIA_BASE_BANK_DIR/index_manifest.json"
test -d "$VISUAL_RETRIEVAL_CACHE_DIR"
```

The observation bank records the material-root and retrieval-model identities
used when it was built. Build it against the same container destinations shown
below; a bank created against different host paths is rejected rather than
silently reused.

Start the pipeline container:

```bash
docker run --name hunyuan-pipeline-601 \
  --gpus all \
  --network=host \
  --env-file docker/.env.runtime \
  -e ROOT_DIR=/workspace/hunyuan3.0_assets_creation \
  -v "$PROJECT_ROOT:/workspace/hunyuan3.0_assets_creation:ro" \
  -v "$ASSET_ROOT:/workspace/assets:rw" \
  -v "$ASSET_ROOT/sam3d-visualization:/workspace/hunyuan3.0_assets_creation/tools/sam3d/third_party/sam-3d-objects-multiview/visualization:rw" \
  -v "$ISAAC_CACHE_ROOT/cache/main:/isaac-sim/.cache:rw" \
  -v "$ISAAC_CACHE_ROOT/cache/computecache:/isaac-sim/.nv/ComputeCache:rw" \
  -v "$ISAAC_CACHE_ROOT/logs:/isaac-sim/.nvidia-omniverse/logs:rw" \
  -v "$ISAAC_CACHE_ROOT/config:/isaac-sim/.nvidia-omniverse/config:rw" \
  -v "$ISAAC_CACHE_ROOT/data:/isaac-sim/.local/share/ov/data:rw" \
  -v "$ISAAC_CACHE_ROOT/pkg:/isaac-sim/.local/share/ov/pkg:rw" \
  -v "$MODEL_CACHE_ROOT/ov-hub:/var/cache/hub:rw" \
  -v "$MODEL_CACHE_ROOT/huggingface:/home/pipeline/.cache/huggingface:ro" \
  -v "$MODEL_CACHE_ROOT/torch-hub:/home/pipeline/.cache/torch/hub:ro" \
  -v "$QWEN35_RUNTIME_DIR:$QWEN35_RUNTIME_DIR:ro" \
  -e QWEN35_PYTHON="$QWEN35_PYTHON" \
  -e QWEN35_MODEL_PATH="$QWEN35_MODEL_DIR" \
  -v "$SIGLIP2_MODEL_DIR:/workspace/models/siglip2:ro" \
  -e SIGLIP2_MODEL_PATH=/workspace/models/siglip2 \
  -v "$DINOV2_MODEL_DIR:/workspace/models/dinov2:ro" \
  -e DINOV2_MODEL_PATH=/workspace/models/dinov2 \
  -v "$MVINVERSE_MODEL_DIR:/workspace/hunyuan3.0_assets_creation/tools/qwen_material_pipeline/models/mvinverse/model:ro" \
  -v "$NVIDIA_MATERIAL_DIR:/workspace/materials/nvidia:ro" \
  -e VISUAL_MATERIAL_ROOT=/workspace/materials/nvidia \
  -v "$NVIDIA_BASE_BANK_DIR:/workspace/models/nvidia_base_observation_bank_v1:ro" \
  -e NVIDIA_BASE_OBSERVATION_BANK=/workspace/models/nvidia_base_observation_bank_v1 \
  -v "$VISUAL_RETRIEVAL_CACHE_DIR:/workspace/cache/visual-retrieval:rw" \
  -e VISUAL_RETRIEVAL_CACHE=/workspace/cache/visual-retrieval \
  -d hunyuan-allinone:isaac-6.0.1-materials
```

In every `-v` argument, the path before `:` is on the host and the path after it
is inside the container. The API uses host networking and listens at
`http://127.0.0.1:8000`.

```bash
docker logs --tail 100 hunyuan-pipeline-601
curl --noproxy '*' --fail http://127.0.0.1:8000/health
docker exec hunyuan-pipeline-601 qwen-material base-bank verify \
  --material-root /workspace/materials/nvidia/Base \
  --output-dir /workspace/models/nvidia_base_observation_bank_v1
```

The first Isaac job can take several minutes while caches are initialized.

## 4. Run CLI jobs

Use container paths under `/workspace/assets`. Do not run API and CLI jobs on the
same GPU at the same time.

Existing GLB:

```bash
docker exec hunyuan-pipeline-601 python -m asset_pipeline.cli \
  --existing-glb /workspace/assets/input/model.glb \
  --refine-config-path configs/refinement/hunyuan_reduce_local_postprocess.yaml \
  --intermediate-output-dir /workspace/assets/glb/intermediate \
  --final-output-dir /workspace/assets/glb/final \
  --result-json /workspace/assets/glb/result.json \
  --len-x 0.4 --len-y 0.3 --len-z 0.8 \
  --orientation "X=L,Y=M,Z=S" --material plastic --approx sdf
```

Add `--skip-refine` when refinement is not required.

Manual STEP/STP without automatic visual materials:

```bash
docker exec hunyuan-pipeline-601 python -m asset_pipeline.cli \
  --manual-stp /workspace/assets/input/manual_asset.stp \
  --cad-usd-output-dir /workspace/assets/manual/cad_usd \
  --intermediate-output-dir /workspace/assets/manual/intermediate \
  --final-output-dir /workspace/assets/manual/final \
  --result-json /workspace/assets/manual/result.json \
  --material steel --approx sdf --manual-sdf-resolution 32
```

SAM3D images:

```bash
docker exec hunyuan-pipeline-601 python -m asset_pipeline.cli \
  --sam3d-input /workspace/assets/input/sam3d_images \
  --sam3d-mode auto --sam3d-prompt "storage shelves" \
  --output-dir /workspace/assets/sam3d/work \
  --refine-config-path configs/refinement/hunyuan_reduce_local_postprocess.yaml \
  --intermediate-output-dir /workspace/assets/sam3d/intermediate \
  --final-output-dir /workspace/assets/sam3d/final \
  --result-json /workspace/assets/sam3d/result.json \
  --len-x 0.4 --len-y 0.3 --len-z 0.8 \
  --orientation "X=L,Y=M,Z=S" --material plastic --approx sdf
```

See the [project README](../README.md) and [module index](../docs/README.md) for
other inputs and parameters.

## 5. HTTP API and maintenance

The API supports Hunyuan, existing GLB, and manual STEP/STP jobs. Use the CLI for
SAM3D images. Automatic visual materials require Qwen3.5, SAM3, MVInverse,
SigLIP2, DINOv2, the Base observation bank, and NVIDIA Materials. Project-local
SAM3 and MVInverse code is supplied by the read-only source mount; the remaining
runtime paths are configured above.

```bash
curl --noproxy '*' http://127.0.0.1:8000/health
curl --noproxy '*' http://127.0.0.1:8000/credentials/tencent-cloud
curl --noproxy '*' http://127.0.0.1:8000/jobs
curl --noproxy '*' http://127.0.0.1:8000/jobs/<job_id>
curl --noproxy '*' 'http://127.0.0.1:8000/jobs/<job_id>/logs?tail=200'
```

Jobs are kept in process memory and disappear after restart. See the
[HTTP API guide](../docs/modules/api.md) for request bodies.

```bash
docker logs -f --tail 200 hunyuan-pipeline-601
docker exec -it hunyuan-pipeline-601 bash
docker stop hunyuan-pipeline-601
docker start hunyuan-pipeline-601
docker restart hunyuan-pipeline-601
docker inspect hunyuan-pipeline-601 --format '{{json .State.Health}}'
```

New CLI processes see changes in the source mount immediately; restart the
container for the API to reload them. Rebuild only after dependency, Dockerfile,
Blender, or Isaac Sim changes.

## 6. Build or publish images

The current bundle needs no target-side build. If only the older
`hunyuan-allinone:isaac-6.0.1` base is available, build the visual-material layer:

```bash
DOCKER_BUILDKIT=1 docker build \
  -f docker/Dockerfile.visual-materials \
  -t hunyuan-allinone:isaac-6.0.1-materials .
```

To publish a new complete bundle, build from the current clean source tree and
split the two runtime images:

```bash
export PROJECT_ROOT="$(pwd -P)"
export IMAGE_BUNDLE_DIR="${IMAGE_BUNDLE_DIR:-$PROJECT_ROOT/docker/offline-images}"
mkdir -p "$IMAGE_BUNDLE_DIR"
set -o pipefail

DOCKER_BUILDKIT=1 docker build \
  -f docker/Dockerfile.full \
  -t hunyuan-allinone:isaac-6.0.1-materials .

docker save \
  hunyuan-allinone:isaac-6.0.1-materials \
  nvcr.io/nvidia/omniverse/hub_workstation_cache:2.0.0 \
  | split --bytes=1900M --numeric-suffixes=0 --suffix-length=3 - \
      "$IMAGE_BUNDLE_DIR/hunyuan-pipeline-isaac-6.0.1-offline.tar.part-"

cd "$IMAGE_BUNDLE_DIR"
sha256sum hunyuan-pipeline-isaac-6.0.1-offline.tar.part-* \
  > hunyuan-pipeline-isaac-6.0.1-offline.parts.sha256
```

Do not add build-stage images to the bundle. Do not run `apt upgrade` in the
final `Dockerfile.full` stage; replacing libc from the Isaac image can break
Vulkan initialization.

## 7. Acceptance and troubleshooting

```bash
docker exec hunyuan-pipeline-601 python -m unittest discover -s tests -v
docker exec hunyuan-pipeline-601 /opt/blender/blender --version
docker exec hunyuan-pipeline-601 /isaac-sim/python.sh -c \
  'from isaacsim import SimulationApp; app = SimulationApp({"headless": True}); print("Isaac Sim OK"); app.close()'
curl --noproxy '*' --fail http://127.0.0.1:8000/health
```

The current release contains Tencent AI3D/common `3.0.1462`. For an older image
that reports `models has no attribute SubmitHunyuanTo3DProJobRequest`, upgrade
both packages together, then restart the container:

```bash
docker exec -u 0 hunyuan-pipeline-601 \
  /opt/conda/envs/hunyuan_sam3d/bin/python -m pip install \
  --no-cache-dir --no-deps --upgrade \
  tencentcloud-sdk-python-ai3d==3.0.1462 \
  tencentcloud-sdk-python-common==3.0.1462
docker restart hunyuan-pipeline-601
```

This repair is lost when an old container is recreated. Rebuild from the current
source for a permanent fix; do not `docker commit` a container carrying secrets.

| Problem | Action |
| --- | --- |
| Host cannot write `$ASSET_ROOT` | Repeat the ACL commands in section 2. |
| Cache reports `Permission denied` | Give UID 1234 write access to Isaac and Hub caches. |
| `SimulationApp` exits or segfaults | Confirm `isaac-hub-cache` is running with host networking. |
| SAM3D module or weights are missing | Check `tools/sam3d/third_party/` and model caches. |
| SAM3D native extension fails | Match its CUDA, PyTorch, Python, and system ABI to the image. |
| Pro/Rapid request class is missing | Upgrade both Tencent SDK packages as shown above. |
| Hunyuan/refine credentials are missing | Update `.env.runtime` and restart. |
| API is unreachable | Check container logs, health, and host port 8000. |
| Disk is full | Inspect `docker system df`; remove only reviewed caches or old images. |
