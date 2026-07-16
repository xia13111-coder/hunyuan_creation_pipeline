# Docker 操作指南

[English](./README.md) | [中文](./README.zh.md) | [项目 README](../README.zh.md)

完整镜像包含 `hunyuan_sam3d`、Blender 4.5.0、Isaac Sim 6.0.1、pipeline CLI 和
HTTP API。本指南从目标机器已经拿到离线 tar 开始，依次完成校验、导入、目录配置、
容器启动和 pipeline 运行。目标机器不需要重新构建镜像，也不需要登录 NGC。

本文不写死用户名、磁盘挂载点或项目安装位置。宿主机目录都从当前项目目录动态生成；
`/workspace/...`、`/isaac-sim/...`、`/opt/blender/...` 和 `/home/pipeline/...` 是
镜像内部固定路径，换电脑时不需要修改。

## 1. 准备 tar 和目标机器

从 [Docker Offline Bundle - Isaac Sim 6.0.1](https://github.com/xia13111-coder/hunyuan_creation_pipeline/releases/tag/docker-isaac-6.0.1)
下载全部 17 个 `tar.part-*` 分卷和校验文件，并放到目标机器的同一目录：

- `hunyuan-pipeline-isaac-6.0.1-offline.tar.part-001` 到 `part-017`
- `hunyuan-pipeline-isaac-6.0.1-offline.parts.sha256`

GitHub Release 的单附件不能超过 2 GiB，因此离线 tar 以 1900 MiB 分卷发布。这些
分卷按文件名顺序拼接后就是标准 `docker save` tar，不是 17 个独立镜像。

当前离线包信息：

| 项目 | 值 |
| --- | --- |
| 总大小 | `32,746,894,336` 字节，约 30.5 GiB |
| 分卷 | 17 个；前 16 个为 1900 MiB，最后一个约 830 MiB |
| 完整镜像 | `hunyuan-allinone:isaac-6.0.1` |
| Hub 镜像 | `nvcr.io/nvidia/omniverse/hub_workstation_cache:2.0.0` |
| 腾讯云 SDK | AI3D/common `3.0.1462`，支持 Pro/Rapid 请求 |

目标机器需要：

- Linux 和支持光追的 NVIDIA GPU
- 已安装 Docker Engine
- 已安装并配置 NVIDIA Container Toolkit
- 已安装 `acl`（提供 `setfacl`，用于共享资产目录权限）
- 至少 40 GB 用于保存 tar，并另外预留 Docker 镜像和运行缓存空间；如果 tar 和
  Docker data-root 在同一分区，导入时建议至少有 80 GB 可用空间
- 当前项目目录；运行时会只读挂载源码和配置

```bash
nvidia-smi
docker --version
docker info >/dev/null
```

## 2. 校验并导入 tar

在 tar 所在目录执行：

```bash
export BUNDLE_DIR="$(pwd -P)"
cd "$BUNDLE_DIR"
ls -lh hunyuan-pipeline-isaac-6.0.1-offline.tar.part-*
```

先校验全部分卷，17 项都必须显示 `OK`：

```bash
sha256sum -c hunyuan-pipeline-isaac-6.0.1-offline.parts.sha256
```

如果任何一项提示 `FAILED`，不要执行 `docker load`，应重新下载对应分卷。校验通过后，
可以直接拼接到 `docker load`，不额外占用一个完整 tar 的磁盘空间：

```bash
cat hunyuan-pipeline-isaac-6.0.1-offline.tar.part-* | docker load
```

如需得到单个 tar 文件用于移动硬盘分发，可以先合并再导入：

```bash
cat hunyuan-pipeline-isaac-6.0.1-offline.tar.part-* \
  > hunyuan-pipeline-isaac-6.0.1-offline.tar
docker load -i hunyuan-pipeline-isaac-6.0.1-offline.tar
```

正常输出应包含：

```text
Loaded image: nvcr.io/nvidia/omniverse/hub_workstation_cache:2.0.0
Loaded image: hunyuan-allinone:isaac-6.0.1
```

检查镜像标签和 GPU 容器能力：

```bash
docker image inspect hunyuan-allinone:isaac-6.0.1 \
  --format '{{index .RepoTags 0}} {{.Id}}'
docker image inspect nvcr.io/nvidia/omniverse/hub_workstation_cache:2.0.0 \
  --format '{{index .RepoTags 0}} {{.Id}}'

docker run --rm --gpus all \
  --entrypoint nvidia-smi \
  hunyuan-allinone:isaac-6.0.1
```

离线分卷是重新部署到其他电脑时使用的安装包，建议和校验文件一起长期保留。
只有确认存在另一份已校验备份时，才考虑删除当前副本。

## 3. 在目标机器准备目录

tar 只包含 Docker 镜像。目标机器还要准备：

- 当前项目源码和配置；以下运行命令会只读挂载项目目录。
- 使用 SAM3D 时需要完整的 `tools/sam3d/third_party/`。
- 离线运行 SAM3D 时需要 Hugging Face 和 Torch Hub 模型缓存。
- 输入文件、运行密钥和用于保存结果的目录。

如果不运行 SAM3D，可以不复制 `third_party` 和 SAM3D 模型缓存。先进入项目根目录，
再执行：

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

`PROJECT_ROOT` 必须是当前项目根目录。其余目录默认位于 `./docker/runtime/`，也可以在
运行上述命令前通过同名环境变量改到其他磁盘。这里使用 `pwd -P` 是因为 Docker bind
mount 要求宿主机路径明确；路径由当前目录生成，不依赖某台电脑的用户名或目录结构。

| 宿主机变量 | 默认位置 | 用途 |
| --- | --- | --- |
| `PROJECT_ROOT` | 当前项目目录 | 只读挂载源码和配置 |
| `ASSET_ROOT` | `./docker/runtime/assets` | 输入、输出和 SAM3D visualization |
| `ISAAC_CACHE_ROOT` | `./docker/runtime/isaac-sim-6.0.1` | Isaac Sim 可写缓存 |
| `MODEL_CACHE_ROOT` | `./docker/runtime/model-cache` | Hub、Hugging Face 和 Torch Hub 缓存 |

离线使用 SAM3D 时，把已准备好的 Hugging Face 和 Torch Hub 缓存内容分别放入
`$MODEL_CACHE_ROOT/huggingface` 和 `$MODEL_CACHE_ROOT/torch-hub`。

完整镜像以 `1234:1234` 运行。资产目录由宿主机用户拥有，并通过 ACL 同时授权容器
UID 1234；这样宿主机可以复制输入，容器也可以创建输出。Isaac 和 Hub 缓存只交给容器。

```bash
command -v setfacl >/dev/null || sudo apt-get install -y acl

sudo chown -R "$(id -u):$(id -g)" "$ASSET_ROOT"
sudo chown -R 1234:1234 "$ISAAC_CACHE_ROOT" "$MODEL_CACHE_ROOT/ov-hub"

find "$ASSET_ROOT" -type d -exec setfacl \
  -m "u:$(id -u):rwx,u:1234:rwx,d:u:$(id -u):rwx,d:u:1234:rwx" {} +
find "$ASSET_ROOT" -type f -exec setfacl \
  -m "u:$(id -u):rw-,u:1234:rw-" {} +
```

项目源码只读挂载，输入文件放在 `$ASSET_ROOT/input`，所有输出保存在
`$ASSET_ROOT`。Hugging Face 和 Torch Hub 缓存只读挂载，不会被容器修改。

## 4. 配置环境变量

先创建运行配置文件并设置权限：

```bash
touch docker/.env.runtime
chmod 600 docker/.env.runtime
${EDITOR:-nano} docker/.env.runtime
```

在编辑器中写入以下内容。不要把这个 `dotenv` 代码块逐行粘贴到 shell：

```dotenv
ACCEPT_EULA=Y
PRIVACY_CONSENT=Y
REFINE_MESH_TEMP_UPLOAD=uguu
PIPELINE_MAX_WORKERS=1
PIPELINE_MAX_LOG_LINES=2000

# 使用 Hunyuan 生成或 refine 时填写：
# TENCENTCLOUD_SECRET_ID=your-secret-id
# TENCENTCLOUD_SECRET_KEY=your-secret-key
```

该文件已被 Git 忽略。手工 STEP/STP 和带 `--skip-refine` 的已有 GLB 不需要腾讯云
凭据；Hunyuan 生成、SAM3D 后续 refine 和默认 GLB refine 需要凭据。

## 5. 启动 Hub

如果目标机器已有 `isaac-hub-cache` 容器，直接启动：

```bash
docker start isaac-hub-cache
docker ps --filter name=isaac-hub-cache
```

如果容器不存在，但本地 Hub 镜像存在，则创建一次：

```bash
docker run --name isaac-hub-cache \
  --restart unless-stopped \
  --network=host \
  -u 1234:1234 \
  -v "$MODEL_CACHE_ROOT/ov-hub:/var/cache/hub:rw" \
  -d nvcr.io/nvidia/omniverse/hub_workstation_cache:2.0.0
```

Isaac Sim 6.0.1 启动前应保证 Hub 正常运行，否则 `SimulationApp` 可能退出或段错误。

## 6. 启动完整容器

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

命令左侧的 `$PROJECT_ROOT`、`$ASSET_ROOT` 等是目标电脑上的动态路径；冒号右侧的
`/workspace/...`、`/isaac-sim/...` 等全部是容器内部路径。迁移到其他电脑时只需重新
执行第 3 节的变量命令，不要修改容器内部路径。

容器默认启动 HTTP API。使用 host 网络时无需映射端口，访问地址为
`http://127.0.0.1:8000`。

检查启动结果：

```bash
docker ps --filter name=hunyuan-pipeline-601
docker logs --tail 100 hunyuan-pipeline-601
curl --noproxy '*' http://127.0.0.1:8000/health
```

冷启动 Isaac Sim 时，首次任务可能需要几分钟初始化插件和 GPU 缓存。

## 7. 在容器中运行 Pipeline

CLI 中统一使用容器内部路径 `/workspace/assets`，不要填目标电脑上的宿主机路径。
同一张 GPU 上不要同时运行 API 任务和 CLI 任务。

### 已有 GLB

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

不需要 refine 时加 `--skip-refine`。

### 手工 STEP/STP

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

### SAM3D 图片

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

Hunyuan 图片生成参数见[项目 README](../README.zh.md)，SAM3D、refine、CAD 和物理参数
见[模块文档](../docs/README.zh.md)。

## 8. HTTP API

当前 API 支持 Hunyuan 生成和已有 GLB 后处理。SAM3D 图片和 STEP/STP 使用上一节 CLI。

```bash
# 健康状态和腾讯云凭据状态
curl --noproxy '*' http://127.0.0.1:8000/health
curl --noproxy '*' http://127.0.0.1:8000/credentials/tencent-cloud

# 查看任务
curl --noproxy '*' http://127.0.0.1:8000/jobs
curl --noproxy '*' http://127.0.0.1:8000/jobs/<job_id>
curl --noproxy '*' 'http://127.0.0.1:8000/jobs/<job_id>/logs?tail=200'
```

提交任务的 JSON 示例和字段说明见 [HTTP API 文档](../docs/modules/api.zh.md)。API 任务
保存在进程内存中，容器重启后任务记录会丢失。

## 9. 容器维护

```bash
# 日志和终端
docker logs -f --tail 200 hunyuan-pipeline-601
docker exec -it hunyuan-pipeline-601 bash

# 停止、启动和重启
docker stop hunyuan-pipeline-601
docker start hunyuan-pipeline-601
docker restart hunyuan-pipeline-601

# 查看健康状态
docker inspect hunyuan-pipeline-601 --format '{{json .State.Health}}'
```

项目源码是宿主机只读挂载：CLI 新进程会立即读取新代码，API 需要重启容器。修改
`environment.yml`、Dockerfile、Blender 或 Isaac Sim 版本后才需要重建镜像。

重新创建容器不会删除宿主机资产和缓存：

```bash
docker stop hunyuan-pipeline-601
docker rm hunyuan-pipeline-601
# 然后重新执行第 6 节 docker run 命令
```

## 10. 附录：在源机器生成 tar

目标机器不需要执行本节。只有发布新镜像时，才在已经构建好镜像的源机器执行：

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

离线包不需要额外包含 `nvcr.io/nvidia/isaac-sim:6.0.1` 或
`hunyuan-sam3d-env:latest`，因为它们需要的运行层已经合并在完整镜像中。

## 11. 验收和排错

导入并启动容器后执行基本验收：

```bash
docker exec hunyuan-pipeline-601 python -m unittest discover -s tests -v
docker exec hunyuan-pipeline-601 /opt/blender/blender --version
docker exec hunyuan-pipeline-601 /isaac-sim/python.sh -c \
  'from isaacsim import SimulationApp; app = SimulationApp({"headless": True}); print("Isaac Sim OK"); app.close()'
curl --noproxy '*' --fail http://127.0.0.1:8000/health
```

### Hunyuan Pro/Rapid SDK 兼容修复

GitHub Release `docker-isaac-6.0.1` 已包含 AI3D/common `3.0.1462`，不需要执行本节。
早期离线 tar 内含 `tencentcloud-sdk-python-ai3d==3.0.1424` 和
`tencentcloud-sdk-python-common==3.0.1443`，但当前代码使用 Pro/Rapid 接口。如果日志出现
`models has no attribute SubmitHunyuanTo3DProJobRequest`，在现有容器中执行一次：

```bash
docker exec -u 0 hunyuan-pipeline-601 \
  /opt/conda/envs/hunyuan_sam3d/bin/python -m pip install \
  --no-cache-dir --no-deps --upgrade \
  tencentcloud-sdk-python-ai3d==3.0.1462 \
  tencentcloud-sdk-python-common==3.0.1462
```

验证版本和四个请求类：

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

两个版本都应为 `3.0.1462`，四项检查都应为 `True`。然后重启并等待健康状态：

```bash
docker restart hunyuan-pipeline-601
until [ "$(docker inspect hunyuan-pipeline-601 \
  --format '{{.State.Health.Status}}')" = healthy ]; do sleep 2; done
curl --noproxy '*' --fail http://127.0.0.1:8000/health
```

可删除本次失败产生的空输出后重新运行原命令：

```bash
docker exec hunyuan-pipeline-601 rm -rf \
  /workspace/assets/hunyuan/downloads \
  /workspace/assets/hunyuan/intermediate \
  /workspace/assets/hunyuan/final \
  /workspace/assets/hunyuan/result.json
```

该修复只保存在当前容器的可写层，使用旧 tar 重新创建容器后需要再次执行。项目
`environment.yml` 已将 AI3D 和 common 固定为 `3.0.1462`，以后应基于它重新构建镜像并
生成新 tar。不要用 `docker commit` 保存带腾讯云凭据的运行容器，以免密钥进入镜像配置。

| 问题 | 处理方法 |
| --- | --- |
| 宿主机无法写 `$ASSET_ROOT` | 重新执行第 3 节的 owner 和 ACL 命令，确保宿主机 UID 与容器 UID 1234 都有权限。 |
| 容器缓存 `Permission denied` | 检查 `$ISAAC_CACHE_ROOT` 和 `$MODEL_CACHE_ROOT/ov-hub` 是否允许 UID 1234 写入。 |
| `SimulationApp` 退出或段错误 | 检查 `isaac-hub-cache` 是否运行并使用 host 网络。 |
| 找不到 SAM3D 模块或权重 | 检查宿主机 `tools/sam3d/third_party/` 是否完整。 |
| SAM3D 扩展加载失败 | 扩展需要与镜像内 CUDA、PyTorch 和 Python ABI 一致。 |
| 缺少 `SubmitHunyuanTo3DProJobRequest` | 按上面的 SDK 兼容修复同时升级 AI3D 和 common。 |
| Hunyuan/refine 缺少凭据 | 填写 `docker/.env.runtime` 并重启容器。 |
| API 无法访问 | 检查容器日志、健康状态和宿主机 8000 端口。 |
| 磁盘不足 | 使用 `docker system df` 检查，再清理确认无用的 build cache 或旧镜像。 |

不要在 `Dockerfile.full` 最终阶段执行 `apt upgrade`。替换 Isaac 官方镜像固定的 libc
可能导致 Vulkan 初始化崩溃。
