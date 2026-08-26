# Docker 操作指南

[English](./README.md) | [中文](./README.zh.md) | [项目 README](../README.zh.md)

离线镜像包含流水线运行环境、共享视觉模型依赖、Blender 4.5.0、Isaac Sim 6.0.1、CLI
和 HTTP API。Qwen3.5 独立环境、模型权重和材质观察库从宿主机挂载。`/workspace`、
`/isaac-sim`、`/opt/blender` 和 `/home/pipeline` 开头的是容器路径，换机器时保持不变。

## 1. 导入离线包

从 [Docker Offline Bundle - Isaac Sim 6.0.1](https://github.com/xia13111-coder/hunyuan_creation_pipeline/releases/tag/docker-isaac-6.0.1)
下载 17 个分卷和校验文件：

```text
hunyuan-pipeline-isaac-6.0.1-offline.tar.part-000 ... part-016
hunyuan-pipeline-isaac-6.0.1-offline.parts.sha256
```

仓库可能要求有权限的 GitHub 账号。也可以使用
[GitHub CLI](https://cli.github.com/) 下载：

```bash
mkdir -p hunyuan-docker-bundle
cd hunyuan-docker-bundle
gh auth login
gh release download docker-isaac-6.0.1 \
  --repo xia13111-coder/hunyuan_creation_pipeline \
  --pattern 'hunyuan-pipeline-isaac-6.0.1-offline.*'
```

离线包信息：

| 项目 | 值 |
| --- | --- |
| 总大小 | `32,108,196,352` 字节（约 29.9 GiB） |
| 分卷 | 17 个；除最后一个外均为 1900 MiB |
| 完整镜像 | `hunyuan-allinone:isaac-6.0.1-materials` |
| 完整镜像 ID | `sha256:913fe7c41298e99ec12701afdecc4a784f2e3f07b534c1b0bc9db77099f86855` |
| Hub 镜像 | `nvcr.io/nvidia/omniverse/hub_workstation_cache:2.0.0` |
| 腾讯云 SDK | AI3D/common `3.0.1462` |

宿主机需要 Linux、支持光追的 NVIDIA GPU、Docker Engine、NVIDIA Container Toolkit、
`acl` 和足够的磁盘空间。离线包与 Docker data root 位于同一分区时，建议至少预留
80 GB。

```bash
nvidia-smi
docker --version
docker info >/dev/null
sha256sum -c hunyuan-pipeline-isaac-6.0.1-offline.parts.sha256
cat hunyuan-pipeline-isaac-6.0.1-offline.tar.part-* | docker load
```

任一校验失败都不要导入。随后检查镜像和 GPU 容器能力：

```bash
docker image inspect hunyuan-allinone:isaac-6.0.1-materials \
  --format '{{index .RepoTags 0}} {{.Id}}'
docker image inspect nvcr.io/nvidia/omniverse/hub_workstation_cache:2.0.0 \
  --format '{{index .RepoTags 0}} {{.Id}}'
docker run --rm --gpus all --entrypoint nvidia-smi \
  hunyuan-allinone:isaac-6.0.1-materials
```

归档分卷时应同时保留校验文件。

## 2. 准备宿主机目录

从项目根目录执行：

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

| 变量 | 默认位置 | 用途 |
| --- | --- | --- |
| `PROJECT_ROOT` | 当前目录 | 只读源码和配置 |
| `ASSET_ROOT` | `docker/runtime/assets` | 输入和输出 |
| `ISAAC_CACHE_ROOT` | `docker/runtime/isaac-sim-6.0.1` | Isaac Sim 可写缓存 |
| `MODEL_CACHE_ROOT` | `docker/runtime/model-cache` | 模型与 Hub 缓存 |

SAM3D 还需要 `tools/sam3d/third_party/` 以及已准备好的 Hugging Face 和 Torch Hub 缓存。
镜像以 UID/GID `1234:1234` 运行，需要授权其写入运行目录：

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

输入放在 `$ASSET_ROOT/input`。源码、模型、观察库和 NVIDIA Materials 只读挂载；输出、
Isaac/Hub 缓存和视觉检索缓存可写。

## 3. 配置并启动

创建本机密钥文件：

```bash
cp docker/env.runtime.example docker/.env.runtime
chmod 600 docker/.env.runtime
${EDITOR:-nano} docker/.env.runtime
```

不要提交 `.env.runtime`。Hunyuan 生成和精修任务需要腾讯云凭据；手工建模 STEP/STP 以及带
`--skip-refine` 的已有 GLB 任务不需要。

启动 Isaac Hub 缓存。容器不存在时才会创建：

```bash
docker start isaac-hub-cache 2>/dev/null || docker run --name isaac-hub-cache \
  --restart unless-stopped --network=host -u 1234:1234 \
  -v "$MODEL_CACHE_ROOT/ov-hub:/var/cache/hub:rw" \
  -d nvcr.io/nvidia/omniverse/hub_workstation_cache:2.0.0
```

Hub 不可用时，Isaac Sim 可能退出或发生段错误。

设置模型、观察库、缓存和材质目录。以
`ASSET_MODEL_VOLUME="$MODEL_CACHE_ROOT"` 运行 Qwen3.5 安装脚本时，脚本会在同一目录中
创建 `env/` 和 `model/`。容器按相同绝对路径挂载该运行时，避免 Python 环境前缀变化。

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

观察库记录构建时使用的材质根目录和检索模型。应按下方相同的容器目标路径构建；身份不匹配
时流程会拒绝复用，而不会静默使用旧结果。

启动流水线容器：

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

每个 `-v` 参数中，冒号左侧是宿主机路径，右侧是容器路径。API 使用 host 网络，地址为
`http://127.0.0.1:8000`。

```bash
docker logs --tail 100 hunyuan-pipeline-601
curl --noproxy '*' --fail http://127.0.0.1:8000/health
docker exec hunyuan-pipeline-601 qwen-material base-bank verify \
  --material-root /workspace/materials/nvidia/Base \
  --output-dir /workspace/models/nvidia_base_observation_bank_v1
```

首次 Isaac 任务可能需要几分钟初始化缓存。

## 4. 运行 CLI 任务

参数统一使用 `/workspace/assets` 下的容器路径。同一块 GPU 上不要同时运行 API 和 CLI
任务。

已有 GLB：

```bash
docker exec hunyuan-pipeline-601 python run_asset_pipeline.py \
  --existing-glb /workspace/assets/input/model.glb \
  --refine-config-path configs/refinement/hunyuan_reduce_local_postprocess.yaml \
  --intermediate-output-dir /workspace/assets/glb/intermediate \
  --final-output-dir /workspace/assets/glb/final \
  --result-json /workspace/assets/glb/result.json \
  --len-x 0.4 --len-y 0.3 --len-z 0.8 \
  --orientation "X=L,Y=M,Z=S" --material plastic --approx sdf
```

无需精修时添加 `--skip-refine`。

不含自动视觉材质的手工建模 STEP/STP：

```bash
docker exec hunyuan-pipeline-601 python run_asset_pipeline.py \
  --manual-stp /workspace/assets/input/manual_asset.stp \
  --cad-usd-output-dir /workspace/assets/manual/cad_usd \
  --intermediate-output-dir /workspace/assets/manual/intermediate \
  --final-output-dir /workspace/assets/manual/final \
  --result-json /workspace/assets/manual/result.json \
  --material steel --approx sdf --manual-sdf-resolution 32
```

SAM3D 图片：

```bash
docker exec hunyuan-pipeline-601 python run_asset_pipeline.py \
  --sam3d-input /workspace/assets/input/sam3d_images \
  --sam3d-mode auto --sam3d-prompt "goods shelves" \
  --output-dir /workspace/assets/sam3d/work \
  --refine-config-path configs/refinement/hunyuan_reduce_local_postprocess.yaml \
  --intermediate-output-dir /workspace/assets/sam3d/intermediate \
  --final-output-dir /workspace/assets/sam3d/final \
  --result-json /workspace/assets/sam3d/result.json \
  --len-x 0.4 --len-y 0.3 --len-z 0.8 \
  --orientation "X=L,Y=M,Z=S" --material plastic --approx sdf
```

其他输入和参数见[项目 README](../README.zh.md)及[模块索引](../docs/README.zh.md)。

## 5. HTTP API 与维护

API 支持 Hunyuan、已有 GLB 和手工建模 STEP/STP 任务。SAM3D 图片使用 CLI。自动视觉材质
还需要 Qwen3.5、SAM3、MVInverse、SigLIP2、DINOv2、Base 材质观察库和 NVIDIA Materials。

```bash
curl --noproxy '*' http://127.0.0.1:8000/health
curl --noproxy '*' http://127.0.0.1:8000/credentials/tencent-cloud
curl --noproxy '*' http://127.0.0.1:8000/jobs
curl --noproxy '*' http://127.0.0.1:8000/jobs/<job_id>
curl --noproxy '*' 'http://127.0.0.1:8000/jobs/<job_id>/logs?tail=200'
```

任务记录保存在进程内存中，重启后会消失。请求格式见
[HTTP API 文档](../docs/modules/api.zh.md)。

```bash
docker logs -f --tail 200 hunyuan-pipeline-601
docker exec -it hunyuan-pipeline-601 bash
docker stop hunyuan-pipeline-601
docker start hunyuan-pipeline-601
docker restart hunyuan-pipeline-601
docker inspect hunyuan-pipeline-601 --format '{{json .State.Health}}'
```

CLI 新进程会立即读取源码挂载中的改动；API 需要重启容器。只有依赖、Dockerfile、
Blender 或 Isaac Sim 变化后才需要重建镜像。

## 6. 构建或发布镜像

当前离线包无需在目标机器构建。如果只有旧的 `hunyuan-allinone:isaac-6.0.1` 基础镜像，
可构建视觉材质层：

```bash
DOCKER_BUILDKIT=1 docker build \
  -f docker/Dockerfile.visual-materials \
  -t hunyuan-allinone:isaac-6.0.1-materials .
```

发布新的完整离线包时，从当前干净源码构建并分割两个运行镜像：

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

离线包不应包含构建阶段镜像。不要在 `Dockerfile.full` 最终阶段执行 `apt upgrade`；替换
Isaac 镜像中的 libc 可能导致 Vulkan 初始化失败。

## 7. 验收与排错

```bash
docker exec hunyuan-pipeline-601 python -m unittest discover -s tests -v
docker exec hunyuan-pipeline-601 /opt/blender/blender --version
docker exec hunyuan-pipeline-601 /isaac-sim/python.sh -c \
  'from isaacsim import SimulationApp; app = SimulationApp({"headless": True}); print("Isaac Sim OK"); app.close()'
curl --noproxy '*' --fail http://127.0.0.1:8000/health
```

当前版本已包含腾讯云 AI3D/common `3.0.1462`。旧镜像若出现
`models has no attribute SubmitHunyuanTo3DProJobRequest`，需同时升级两个包并重启：

```bash
docker exec -u 0 hunyuan-pipeline-601 \
  /opt/conda/envs/hunyuan_sam3d/bin/python -m pip install \
  --no-cache-dir --no-deps --upgrade \
  tencentcloud-sdk-python-ai3d==3.0.1462 \
  tencentcloud-sdk-python-common==3.0.1462
docker restart hunyuan-pipeline-601
```

旧容器重建后会丢失这个临时修复。永久修复应基于当前源码重建；不要对含密钥的运行容器
执行 `docker commit`。

| 问题 | 处理方法 |
| --- | --- |
| 宿主机无法写 `$ASSET_ROOT` | 重新执行第 2 节 ACL 命令。 |
| 缓存提示 `Permission denied` | 确保 UID 1234 可写 Isaac 和 Hub 缓存。 |
| `SimulationApp` 退出或段错误 | 确认 `isaac-hub-cache` 使用 host 网络运行。 |
| 找不到 SAM3D 模块或权重 | 检查 `tools/sam3d/third_party/` 和模型缓存。 |
| SAM3D 扩展加载失败 | CUDA、PyTorch、Python 和系统 ABI 必须与镜像一致。 |
| 缺少 Pro/Rapid 请求类 | 按上面的方法同时升级两个腾讯云 SDK 包。 |
| Hunyuan 生成或精修缺少凭据 | 更新 `.env.runtime` 并重启。 |
| API 无法访问 | 检查日志、健康状态和宿主机 8000 端口。 |
| 磁盘不足 | 用 `docker system df` 检查，只删除确认无用的缓存或旧镜像。 |
