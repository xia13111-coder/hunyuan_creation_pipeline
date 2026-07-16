# Docker Operations

[English](./README.md) | [中文](./README.zh.md) | [Project README](../README.md)

The full image contains `hunyuan_sam3d`, Blender 4.5.0, Isaac Sim 6.0.1, the
pipeline CLI, and the HTTP API. This guide starts after the offline tar has
arrived on the target and covers verification, loading, configuration, startup,
and pipeline operation. The target does not rebuild or log in to NGC.

No host username, disk mount point, or checkout location is hard-coded below.
Host directories are derived from the current project directory. Paths beginning
with `/workspace`, `/isaac-sim`, `/opt/blender`, or `/home/pipeline` are fixed
paths inside the image and do not change when the bundle moves to another
computer.

## 1. Prepare the Tar and Target

Download all 17 `tar.part-*` files and the checksum manifest from
[Docker Offline Bundle - Isaac Sim 6.0.1](https://github.com/xia13111-coder/hunyuan_creation_pipeline/releases/tag/docker-isaac-6.0.1),
then place them together in any directory on the target:

- `hunyuan-pipeline-isaac-6.0.1-offline.tar.part-001` through `part-017`
- `hunyuan-pipeline-isaac-6.0.1-offline.parts.sha256`

GitHub Release assets must each remain under 2 GiB, so the offline tar is
published in 1900 MiB parts. Concatenating them in filename order recreates one
standard `docker save` tar; they are not 17 independent images.

Current bundle information:

| Item | Value |
| --- | --- |
| Total size | `32,746,894,336` bytes, approximately 30.5 GiB |
| Parts | 17; the first 16 are 1900 MiB and the last is approximately 830 MiB |
| Full image | `hunyuan-allinone:isaac-6.0.1` |
| Hub image | `nvcr.io/nvidia/omniverse/hub_workstation_cache:2.0.0` |
| Tencent SDK | AI3D/common `3.0.1462` with Pro/Rapid request support |

The target requires:

- Linux and an RTX-capable NVIDIA GPU
- Docker Engine
- NVIDIA Container Toolkit
- `acl` installed (`setfacl` is used for shared asset-directory permissions)
- At least 40 GB for the tar, plus Docker image and cache space; keep at least
  80 GB free while loading when the tar and Docker data root share a filesystem
- A project checkout, mounted read-only for current source and configuration

```bash
nvidia-smi
docker --version
docker info >/dev/null
```

## 2. Verify and Load the Tar

Run this from the bundle directory:

```bash
export BUNDLE_DIR="$(pwd -P)"
cd "$BUNDLE_DIR"
ls -lh hunyuan-pipeline-isaac-6.0.1-offline.tar.part-*
```

Verify every part first; all 17 results must be `OK`:

```bash
sha256sum -c hunyuan-pipeline-isaac-6.0.1-offline.parts.sha256
```

If any part reports `FAILED`, download that part again and do not run
`docker load`. After a successful checksum, concatenate directly into
`docker load` without allocating another full tar on disk:

```bash
cat hunyuan-pipeline-isaac-6.0.1-offline.tar.part-* | docker load
```

To create one tar for external-drive distribution, merge it first:

```bash
cat hunyuan-pipeline-isaac-6.0.1-offline.tar.part-* \
  > hunyuan-pipeline-isaac-6.0.1-offline.tar
docker load -i hunyuan-pipeline-isaac-6.0.1-offline.tar
```

Expected output includes:

```text
Loaded image: nvcr.io/nvidia/omniverse/hub_workstation_cache:2.0.0
Loaded image: hunyuan-allinone:isaac-6.0.1
```

Check both tags and container GPU access:

```bash
docker image inspect hunyuan-allinone:isaac-6.0.1 \
  --format '{{index .RepoTags 0}} {{.Id}}'
docker image inspect nvcr.io/nvidia/omniverse/hub_workstation_cache:2.0.0 \
  --format '{{index .RepoTags 0}} {{.Id}}'

docker run --rm --gpus all \
  --entrypoint nvidia-smi \
  hunyuan-allinone:isaac-6.0.1
```

Keep the offline parts and checksum manifest as the reusable installer for other
computers. Delete this copy only when another verified backup exists.

## 3. Prepare Target Directories

The tar contains Docker images only. Also prepare:

- Current project source and configuration; the run command mounts it read-only.
- Complete `tools/sam3d/third_party/` when SAM3D is used.
- Hugging Face and Torch Hub model caches for offline SAM3D.
- Inputs, runtime credentials, and output directories.

Targets that do not run SAM3D do not need its `third_party` or model caches.
Enter the project root first, then run:

```bash
export PROJECT_ROOT="$(pwd -P)"
export RUNTIME_ROOT="${RUNTIME_ROOT:-$PROJECT_ROOT/docker/runtime}"
export ASSET_ROOT="${ASSET_ROOT:-$RUNTIME_ROOT/assets}"
export ISAAC_CACHE_ROOT="${ISAAC_CACHE_ROOT:-$RUNTIME_ROOT/isaac-sim-6.0.1}"
export MODEL_CACHE_ROOT="${MODEL_CACHE_ROOT:-$RUNTIME_ROOT/model-cache}"

mkdir -p "$ASSET_ROOT"/{input,sam3d-visualization}
mkdir -p "$ISAAC_CACHE_ROOT"/{cache/main,cache/computecache,config,data,logs,pkg}
mkdir -p "$MODEL_CACHE_ROOT"/{ov-hub,huggingface,torch-hub}
```

`PROJECT_ROOT` must be the current project root. Other directories default to
`./docker/runtime/`; set the corresponding environment variable before running
the block to place one on another disk. `pwd -P` supplies the explicit host path
required by Docker bind mounts without depending on a particular username or
directory layout.

| Host variable | Default location | Purpose |
| --- | --- | --- |
| `PROJECT_ROOT` | Current project directory | Read-only source and configuration |
| `ASSET_ROOT` | `./docker/runtime/assets` | Inputs, outputs, and SAM3D visualization |
| `ISAAC_CACHE_ROOT` | `./docker/runtime/isaac-sim-6.0.1` | Writable Isaac Sim caches |
| `MODEL_CACHE_ROOT` | `./docker/runtime/model-cache` | Hub, Hugging Face, and Torch Hub caches |

For offline SAM3D, place the prepared Hugging Face and Torch Hub cache contents
under `$MODEL_CACHE_ROOT/huggingface` and `$MODEL_CACHE_ROOT/torch-hub`.

The full image runs as `1234:1234`. Keep the asset directory owned by the host
user and grant container UID 1234 access through ACLs, so the host can add inputs
while the container creates outputs. Isaac and Hub caches belong to the container.

```bash
command -v setfacl >/dev/null || sudo apt-get install -y acl

sudo chown -R "$(id -u):$(id -g)" "$ASSET_ROOT"
sudo chown -R 1234:1234 "$ISAAC_CACHE_ROOT" "$MODEL_CACHE_ROOT/ov-hub"

find "$ASSET_ROOT" -type d -exec setfacl \
  -m "u:$(id -u):rwx,u:1234:rwx,d:u:$(id -u):rwx,d:u:1234:rwx" {} +
find "$ASSET_ROOT" -type f -exec setfacl \
  -m "u:$(id -u):rw-,u:1234:rw-" {} +
```

The source checkout is mounted read-only. Put inputs in `$ASSET_ROOT/input`; all
outputs remain under `$ASSET_ROOT`. Hugging Face and Torch Hub caches are reused
read-only.

## 4. Runtime Environment

Create the runtime file, restrict its permissions, and open it in an editor:

```bash
touch docker/.env.runtime
chmod 600 docker/.env.runtime
${EDITOR:-nano} docker/.env.runtime
```

Enter the following content in the editor. Do not paste this `dotenv` block into
the shell one line at a time:

```dotenv
ACCEPT_EULA=Y
PRIVACY_CONSENT=Y
REFINE_MESH_TEMP_UPLOAD=uguu
PIPELINE_MAX_WORKERS=1
PIPELINE_MAX_LOG_LINES=2000

# Add these for Hunyuan generation or refine:
# TENCENTCLOUD_SECRET_ID=your-secret-id
# TENCENTCLOUD_SECRET_KEY=your-secret-key
```

Manual STEP/STP and existing GLB jobs with `--skip-refine` do not need Tencent
credentials. Hunyuan generation, SAM3D's following refine stage, and default GLB
refine do require them.

## 5. Start Hub

If the target already has an `isaac-hub-cache` container, start it:

```bash
docker start isaac-hub-cache
docker ps --filter name=isaac-hub-cache
```

If the container does not exist but the Hub image is local, create it once:

```bash
docker run --name isaac-hub-cache \
  --restart unless-stopped \
  --network=host \
  -u 1234:1234 \
  -v "$MODEL_CACHE_ROOT/ov-hub:/var/cache/hub:rw" \
  -d nvcr.io/nvidia/omniverse/hub_workstation_cache:2.0.0
```

Keep Hub running before Isaac Sim 6.0.1 starts. Without it, `SimulationApp` may
exit or segfault.

## 6. Start the Full Container

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
  -d hunyuan-allinone:isaac-6.0.1
```

On each `-v` option, variables such as `$PROJECT_ROOT` and `$ASSET_ROOT` are
dynamic target-host paths. Paths after the colon, including `/workspace` and
`/isaac-sim`, are internal container paths. On another computer, rerun the
variable block in section 3 and leave the container paths unchanged.

The container starts the HTTP API. Host networking is required for Hub, so no
port mapping is needed; the API is at `http://127.0.0.1:8000`.

```bash
docker ps --filter name=hunyuan-pipeline-601
docker logs --tail 100 hunyuan-pipeline-601
curl --noproxy '*' http://127.0.0.1:8000/health
```

The first Isaac job may take several minutes to initialize plugins and GPU
caches.

## 7. Run Pipeline CLI Jobs

Use internal container paths under `/workspace/assets` in CLI arguments, not
paths from the target host. Do not run an API job and a CLI job concurrently on
the same GPU.

### Existing GLB

```bash
docker exec hunyuan-pipeline-601 python run_asset_pipeline.py \
  --existing-glb /workspace/assets/input/model.glb \
  --refine-config-path configs/hunyuan_reduce_local_postprocess.yaml \
  --intermediate-output-dir /workspace/assets/glb/intermediate \
  --final-output-dir /workspace/assets/glb/final \
  --result-json /workspace/assets/glb/result.json \
  --len-x 0.4 --len-y 0.3 --len-z 0.8 \
  --orientation "X=L,Y=M,Z=S" \
  --material plastic --approx sdf
```

Add `--skip-refine` when refine is not needed.

### Manual STEP/STP

```bash
docker exec hunyuan-pipeline-601 python run_asset_pipeline.py \
  --manual-stp /workspace/assets/input/manual_asset.stp \
  --cad-usd-output-dir /workspace/assets/manual/cad_usd \
  --intermediate-output-dir /workspace/assets/manual/intermediate \
  --final-output-dir /workspace/assets/manual/final \
  --result-json /workspace/assets/manual/result.json \
  --material steel \
  --approx sdf \
  --manual-sdf-resolution 32
```

### SAM3D Images

```bash
docker exec hunyuan-pipeline-601 python run_asset_pipeline.py \
  --sam3d-input /workspace/assets/input/sam3d_images \
  --sam3d-mode auto \
  --sam3d-prompt "goods shelves" \
  --output-dir /workspace/assets/sam3d/work \
  --refine-config-path configs/hunyuan_reduce_local_postprocess.yaml \
  --intermediate-output-dir /workspace/assets/sam3d/intermediate \
  --final-output-dir /workspace/assets/sam3d/final \
  --result-json /workspace/assets/sam3d/result.json \
  --len-x 0.4 --len-y 0.3 --len-z 0.8 \
  --orientation "X=L,Y=M,Z=S" \
  --material plastic --approx sdf
```

See the [project README](../README.md) for Hunyuan commands and the
[module documentation](../docs/README.md) for SAM3D, refine, CAD, and physics
parameters.

## 8. HTTP API

The API currently supports Hunyuan generation and existing-GLB processing. Use
the CLI above for SAM3D images and STEP/STP.

```bash
# Health and Tencent credential status
curl --noproxy '*' http://127.0.0.1:8000/health
curl --noproxy '*' http://127.0.0.1:8000/credentials/tencent-cloud

# Jobs and logs
curl --noproxy '*' http://127.0.0.1:8000/jobs
curl --noproxy '*' http://127.0.0.1:8000/jobs/<job_id>
curl --noproxy '*' 'http://127.0.0.1:8000/jobs/<job_id>/logs?tail=200'
```

See the [HTTP API guide](../docs/modules/api.md) for request examples and fields.
Jobs live in process memory and disappear after a container restart.

## 9. Container Maintenance

```bash
# Logs and shell
docker logs -f --tail 200 hunyuan-pipeline-601
docker exec -it hunyuan-pipeline-601 bash

# Stop, start, restart, and inspect health
docker stop hunyuan-pipeline-601
docker start hunyuan-pipeline-601
docker restart hunyuan-pipeline-601
docker inspect hunyuan-pipeline-601 --format '{{json .State.Health}}'
```

The source is bind-mounted read-only. New CLI processes see host source changes
immediately; restart the container for the API to reload them. Rebuild only when
`environment.yml`, a Dockerfile, Blender, or the Isaac Sim version changes.

Recreating the container does not delete host assets or caches:

```bash
docker stop hunyuan-pipeline-601
docker rm hunyuan-pipeline-601
# Rerun the docker run command in section 6.
```

## 10. Appendix: Create the Tar on the Source

The target does not run this section. Run it only on the source machine when
publishing a new image:

```bash
export PROJECT_ROOT="$(pwd -P)"
export IMAGE_BUNDLE_DIR="${IMAGE_BUNDLE_DIR:-$PROJECT_ROOT/docker/offline-images}"
mkdir -p "$IMAGE_BUNDLE_DIR"
set -o pipefail

docker save \
  hunyuan-allinone:isaac-6.0.1 \
  nvcr.io/nvidia/omniverse/hub_workstation_cache:2.0.0 \
  | split --bytes=1900M --numeric-suffixes=1 --suffix-length=3 - \
      "$IMAGE_BUNDLE_DIR/hunyuan-pipeline-isaac-6.0.1-offline.tar.part-"

cd "$IMAGE_BUNDLE_DIR"
sha256sum hunyuan-pipeline-isaac-6.0.1-offline.tar.part-* \
  > hunyuan-pipeline-isaac-6.0.1-offline.parts.sha256
```

Do not add `nvcr.io/nvidia/isaac-sim:6.0.1` or
`hunyuan-sam3d-env:latest`; the full image already contains their runtime layers.

## 11. Acceptance and Troubleshooting

Run basic acceptance checks after loading and starting the container:

```bash
docker exec hunyuan-pipeline-601 python -m unittest discover -s tests -v
docker exec hunyuan-pipeline-601 /opt/blender/blender --version
docker exec hunyuan-pipeline-601 /isaac-sim/python.sh -c \
  'from isaacsim import SimulationApp; app = SimulationApp({"headless": True}); print("Isaac Sim OK"); app.close()'
curl --noproxy '*' --fail http://127.0.0.1:8000/health
```

### Hunyuan Pro/Rapid SDK Compatibility

GitHub Release `docker-isaac-6.0.1` already contains AI3D/common `3.0.1462`, so
this section is unnecessary for that bundle. The earlier offline tar contains
`tencentcloud-sdk-python-ai3d==3.0.1424` and
`tencentcloud-sdk-python-common==3.0.1443`, while the current code uses the
Pro/Rapid APIs. If the log reports
`models has no attribute SubmitHunyuanTo3DProJobRequest`, run once in the current
container:

```bash
docker exec -u 0 hunyuan-pipeline-601 \
  /opt/conda/envs/hunyuan_sam3d/bin/python -m pip install \
  --no-cache-dir --no-deps --upgrade \
  tencentcloud-sdk-python-ai3d==3.0.1462 \
  tencentcloud-sdk-python-common==3.0.1462
```

Verify both versions and all four request classes:

```bash
docker exec hunyuan-pipeline-601 python -c '
import importlib.metadata as metadata
from tencentcloud.ai3d.v20250513 import models
print("ai3d", metadata.version("tencentcloud-sdk-python-ai3d"))
print("common", metadata.version("tencentcloud-sdk-python-common"))
for name in (
    "SubmitHunyuanTo3DProJobRequest", "QueryHunyuanTo3DProJobRequest",
    "SubmitHunyuanTo3DRapidJobRequest", "QueryHunyuanTo3DRapidJobRequest",
):
    print(name, hasattr(models, name))
'
```

Both versions must be `3.0.1462`, and all four checks must be `True`. Restart the
container and wait for it to become healthy:

```bash
docker restart hunyuan-pipeline-601
until [ "$(docker inspect hunyuan-pipeline-601 \
  --format '{{.State.Health.Status}}')" = healthy ]; do sleep 2; done
curl --noproxy '*' --fail http://127.0.0.1:8000/health
```

Optionally remove empty output from the failed attempt before rerunning the same
pipeline command:

```bash
docker exec hunyuan-pipeline-601 rm -rf \
  /workspace/assets/hunyuan/downloads \
  /workspace/assets/hunyuan/intermediate \
  /workspace/assets/hunyuan/final \
  /workspace/assets/hunyuan/result.json
```

This repair lives only in the current container's writable layer and must be
repeated after recreating a container from the old tar. The project
`environment.yml` now pins both AI3D and common to `3.0.1462`; rebuild the image
and create a new tar for a permanent fix. Do not use `docker commit` on a running
container that carries Tencent credentials, because they can enter image
configuration metadata.

| Problem | Action |
| --- | --- |
| The host cannot write `$ASSET_ROOT` | Repeat the owner and ACL commands in section 3 so both the host UID and container UID 1234 have access. |
| Container cache `Permission denied` | Ensure UID 1234 can write `$ISAAC_CACHE_ROOT` and `$MODEL_CACHE_ROOT/ov-hub`. |
| `SimulationApp` exits or segfaults | Confirm `isaac-hub-cache` runs with host networking. |
| SAM3D module or weights are missing | Check the host `tools/sam3d/third_party/` directory. |
| SAM3D native extension fails | Its CUDA, PyTorch, Python, and system ABI must match the image. |
| `SubmitHunyuanTo3DProJobRequest` is missing | Apply the SDK compatibility procedure above to upgrade AI3D and common together. |
| Hunyuan/refine credentials are missing | Update `docker/.env.runtime` and restart the container. |
| API is unreachable | Check logs, health, and whether host port 8000 is already used. |
| Disk is full | Run `docker system df`, then remove only reviewed build cache or old images. |

Do not run `apt upgrade` in the final `Dockerfile.full` stage. Replacing libc
packages pinned by the official Isaac image can cause Vulkan initialization
crashes.
