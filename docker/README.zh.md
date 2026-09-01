# 完整 Docker 使用说明

这套配置运行当前主线代码，并包含 API、Blender、Isaac Sim、Qwen3.5、MVInverse、
SAM3、EntitySeg、DINOv2、SigLIP2 和 NVIDIA Base 材质流程。模型和材质库体积很大，
保存在宿主机，通过只读目录挂入容器；输入、输出和缓存放在可写数据盘。

## 准备

宿主机需要 Linux、NVIDIA GPU、Docker Engine、Docker Compose v2 和 NVIDIA
Container Toolkit。先确认 GPU 可被容器使用：

```bash
docker run --rm --gpus all --entrypoint nvidia-smi \
  hunyuan-allinone:isaac-6.0.1-materials -L
```

没有镜像时，从项目的
[Docker 离线镜像发布页](https://github.com/xia13111-coder/hunyuan_creation_pipeline/releases/tag/docker-isaac-6.0.1)
下载全部分卷和 SHA256 文件，再导入：

```bash
cd docker/offline-images
sha256sum -c hunyuan-pipeline-isaac-6.0.1-offline.parts.sha256
cat hunyuan-pipeline-isaac-6.0.1-offline.tar.part-* | docker load
cd ../..
```

## 第一次启动

```bash
docker/run-full.sh init
${EDITOR:-nano} docker/.env.runtime
docker/run-full.sh preflight
docker/run-full.sh up
docker/run-full.sh smoke
```

`docker/.env.runtime` 由 Compose 读取，和根目录 `.env` 是两份不同的配置。这里所有路径
都必须是宿主机绝对路径；Compose 会把外部模型按相同绝对路径只读挂载，不能写
`$HOME`、相对路径或容器内路径。

### 镜像和容器

| 变量 | 默认值/填写内容 | 作用和获取方式 |
| --- | --- | --- |
| `PIPELINE_IMAGE` | `hunyuan-allinone:isaac-6.0.1-materials` | 最终运行镜像。通过上面的离线发布包 `docker load` 获得，或在两个基础镜像就绪后执行 `docker/run-full.sh build` 构建。 |
| `PIPELINE_ENV_IMAGE` | `hunyuan-allinone:isaac-6.0.1` | 仅构建时使用的 Conda/项目依赖镜像，可用 `docker/Dockerfile` 构建；直接运行已导入的最终镜像时不会读取它。 |
| `ISAACSIM_BASE_IMAGE` | 本地 Isaac Sim 6.0.1 镜像标签 | 仅构建时使用。官方来源为 [NVIDIA NGC Isaac Sim](https://catalog.ngc.nvidia.com/orgs/nvidia/-/containers/isaac-sim/-)，可填写 `nvcr.io/nvidia/isaac-sim:6.0.1`。 |
| `PIPELINE_CONTAINER_NAME` | `hunyuan-pipeline-601` | API/流水线容器名称，可改，但后续 `docker exec` 命令也要使用新名称。 |
| `HUB_CONTAINER_NAME` | `isaac-hub-cache` | Isaac Hub 缓存容器名称。 |
| `HUB_IMAGE` | `nvcr.io/nvidia/omniverse/hub_workstation_cache:2.0.0` | 由启动脚本自动运行；来源和用途见 [NVIDIA Hub Workstation Cache](https://docs.omniverse.nvidia.com/utilities/latest/cache/hub-workstation.html)。离线使用时也必须先导入该镜像。 |
| `PIPELINE_UID` / `PIPELINE_GID` | 宿主机用户 ID/组 ID | 分别填写 `id -u` 和 `id -g` 的输出，使容器写出的文件归当前用户所有。 |
| `PIPELINE_MIN_FREE_GB` | `50` | `preflight` 要求 `ASSET_ROOT` 至少保留的空间，单位 GiB；不是磁盘配额。 |
| `PIPELINE_DOCKER_MIN_BUILD_FREE_GB` | `15` | 执行 `build` 前要求 Docker 数据根目录保留的空间，单位 GiB。构建期间候选镜像会与当前镜像短暂共存。 |
| `PIPELINE_START_TIMEOUT_SECONDS` | `300` | Compose 等待服务健康的最长时间，单位秒。 |
| `PIPELINE_MIN_NOFILE` | `65536` | 容器预检要求的文件描述符软上限；完整参考图渲染会同时打开大量 GPU/渲染资源。Compose 已设置同样的 `nofile` 上限。 |

如果不用离线最终镜像而要从源码构建，先准备基础镜像：

```bash
docker build -f docker/Dockerfile \
  -t hunyuan-allinone:isaac-6.0.1 .
docker pull nvcr.io/nvidia/isaac-sim:6.0.1
```

然后在 `.env.runtime` 中把 `ISAACSIM_BASE_IMAGE` 改为
`nvcr.io/nvidia/isaac-sim:6.0.1`，再执行 `docker/run-full.sh build`。

### 宿主机目录

| 变量 | 填写内容 | 读写方式和用途 |
| --- | --- | --- |
| `PROJECT_ROOT` | 当前代码仓库根目录 | 只读挂到容器 `/workspace/hunyuan3.0_assets_creation`，并保留同名宿主机绝对路径；后者用于原样回放 SAM3 标注清单中经过哈希校验的参考图路径。必须正好指向当前 checkout。 |
| `ASSET_ROOT` | 输入、输出和任务结果目录 | 可写挂到 `/workspace/assets`；应放在大容量数据盘。 |
| `MODEL_CACHE_ROOT` | 容器 Home、Hugging Face、Torch 和 Hub 缓存根目录 | 可写；启动脚本会创建 `home/` 和 `ov-hub/`。它不是某个模型的路径。 |
| `ISAAC_CACHE_ROOT` | Isaac Sim 缓存和日志根目录 | 可写；启动脚本会创建 `cache/`、`logs/`、`config/`、`data/`、`pkg/`。无需下载内容。 |

### 外部 Python、模型和权重

| 变量 | 必须指向 | 获取方式 |
| --- | --- | --- |
| `QWEN35_RUNTIME_DIR` | Qwen3.5 独立运行环境根目录 | 运行根 README 中的 `setup_qwen35_runtime.sh` 创建；它不是模型目录。 |
| `QWEN35_PYTHON` | 上述目录中的可执行 Python | 通常为 `QWEN35_RUNTIME_DIR/env/bin/python`。容器按同一路径挂载，所以不能使用容器专用路径替代。 |
| `QWEN35_MODEL_PATH` | 含 `config.json` 和 safetensors 的 Qwen3.5-4B 目录 | 来源为 [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)；推荐使用仓库脚本下载固定版本。 |
| `MVINVERSE_REPOSITORY` | MVInverse 源码目录 | 当前仓库已带兼容源码；上游为 [Maddog241/mvinverse](https://github.com/Maddog241/mvinverse)。 |
| `MVINVERSE_CHECKPOINT` | 含 `config.json` 和 `model.safetensors` 的目录 | 从 [maddog241/mvinverse](https://huggingface.co/maddog241/mvinverse) 下载。 |
| `SAM3_REPOSITORY` | 与权重匹配的 SAM3 源码目录 | 推荐填写 SAM 3D Objects 的 `submodules/sam3`；上游为 [facebookresearch/sam3](https://github.com/facebookresearch/sam3)。 |
| `SAM3_CHECKPOINT` | `sam3.pt` 文件 | 从受访问控制的 [facebook/sam3](https://huggingface.co/facebook/sam3) 下载。 |
| `SAM3D_SINGLE_VIEW_ROOT` | SAM 3D Objects 源码根目录 | 从 [facebookresearch/sam-3d-objects](https://github.com/facebookresearch/sam-3d-objects) 带子模块克隆。 |
| `SAM3D_MULTI_VIEW_ROOT` | 多视角扩展根目录 | 从 [devinli123/MV-SAM3D](https://github.com/devinli123/MV-SAM3D) 克隆。完整 Docker 检查要求该目录存在。 |
| `SAM3D_PIPELINE_CONFIG` | SAM 3D 权重包内的 `checkpoints/pipeline.yaml` | 从受访问控制的 [facebook/sam-3d-objects](https://huggingface.co/facebook/sam-3d-objects) 下载。 |
| `SAM3D_MOGE_CHECKPOINT` | MoGe 的 `model.pt` | 从 [Ruicheng/moge-vitl](https://huggingface.co/Ruicheng/moge-vitl) 下载。 |
| `SAM3D_DINOV2_REPOSITORY` | 含 `hubconf.py` 的 DINOv2 源码目录 | 从 [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2) 克隆。 |
| `SAM3D_DINOV2_CHECKPOINT` | `dinov2_vitl14_reg4_pretrain.pth` | 从 [Meta 官方地址](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth) 下载。 |
| `SIGLIP2_MODEL_PATH` | SigLIP2 模型目录 | [google/siglip2-base-patch16-224](https://huggingface.co/google/siglip2-base-patch16-224)；Qwen3.5 设置脚本会一并下载。 |
| `DINOV2_MODEL_PATH` | 含 `config.json` 的 Transformers DINOv2 目录 | [facebook/dinov2-with-registers-large](https://huggingface.co/facebook/dinov2-with-registers-large)；不要填成上面的 `.pth`。 |

具体下载命令和目录示例见根目录 [README 的 `.env` 章节](../README.zh.md#env-参数与资源下载)。

### EntitySeg 独立环境

| 变量 | 必须指向 | 获取方式和原因 |
| --- | --- | --- |
| `ENTITYSEG_RUNTIME_DIR` | EntitySeg 虚拟环境根目录 | 本机创建，不是下载模型；其中应包含 `bin/python` 和依赖。 |
| `ENTITYSEG_BASE_RUNTIME_DIR` | `ENTITYSEG_PYTHON` 解析符号链接后所在的基础 Python 前缀 | Docker 需要同时挂载它，否则虚拟环境的 Python 链接在容器内会失效。普通 venv 没有外部链接时可与运行目录按实际布局填写。 |
| `ENTITYSEG_DETECTRON2_ROOT` | 含 `detectron2/__init__.py` 的 Detectron2 源码/安装根目录 | 按 [CropFormer 安装说明](https://github.com/qqlu/Entity/blob/main/Entityv2/CODE.md) 准备；用于保留与旧版 CropFormer 匹配的 Detectron2。 |
| `ENTITYSEG_PYTHON` | EntitySeg 环境的可执行 Python | 必须可执行，并能导入 Detectron2、CropFormer 和 `requirements/entityseg.txt` 中的兼容包。 |
| `ENTITYSEG_CROPFORMER_ROOT` | `Entity/Entityv2/CropFormer` | 从 [qqlu/Entity](https://github.com/qqlu/Entity) 下载。 |
| `ENTITYSEG_CONFIG` | `cropformer_swin_tiny_3x.yaml` 文件 | 位于 CropFormer 的 `configs/entityv2/entity_segmentation/`。 |
| `ENTITYSEG_CHECKPOINT` | `CropFormer_swin_tiny_3x_5cea5e.pth` | 从作者发布的 [CropFormer 模型目录](https://huggingface.co/datasets/qqlu1992/Adobe_EntitySeg/tree/main/CropFormer_model/Entity_Segmentation/CropFormer_swin_tiny_3x) 下载并接受访问条件。 |

### 材质、观察库和运行选项

| 变量 | 填写内容 | 来源或默认行为 |
| --- | --- | --- |
| `VISUAL_MATERIAL_ROOT` | 包含 `Base/` 的 NVIDIA `Materials` 目录 | 从 NVIDIA [Downloadable Asset Packs](https://docs.omniverse.nvidia.com/usd/latest/usd_content_samples/downloadable_packs.html) 下载 **Base Materials Pack** 并解压。 |
| `NVIDIA_BASE_OBSERVATION_BANK` | 含 `index_manifest.json` 的观察库目录 | 不是下载项；按根 README 的命令用本机材质、Isaac Sim、SigLIP2 和 DINOv2 生成。以只读方式挂载。 |
| `VISUAL_RETRIEVAL_CACHE` | 可写检索缓存目录 | 本机创建；模型或材质库变化后应换新目录。 |
| `PIPELINE_LOCAL_MODELS_ONLY` | `1` | 固定为 `1`，禁止任务过程中隐式下载模型。 |
| `ACCEPT_EULA` | `Y` | 表示接受 Isaac Sim 容器许可；NVIDIA 的官方容器启动说明要求设置。 |
| `PRIVACY_CONSENT` | `Y` | NVIDIA 容器的隐私/遥测同意项；不接受时应先查看 NVIDIA 说明，不要随意伪造值。 |
| `PIPELINE_MAX_WORKERS` | `1` | API 并发任务数。一张 GPU 建议保持 `1`。 |
| `PIPELINE_MAX_LOG_LINES` | `2000` | 每个 API 任务保留在内存中的日志行数。 |
| `REFINE_MESH_TEMP_UPLOAD` | `uguu` | 云端精修本地 GLB 时使用的第三方临时上传服务；敏感资产不应使用。 |
| `TENCENTCLOUD_SECRET_ID` / `TENCENTCLOUD_SECRET_KEY` | 腾讯云 API 密钥 | 仅云端 Hunyuan/精修需要，从 [腾讯云 API 密钥管理](https://console.cloud.tencent.com/cam/capi) 创建；只跑 STEP/STP 自动赋材质时留空。 |
| `TENCENTCLOUD_REGION` | 例如 `ap-guangzhou` | 留空时默认 `ap-guangzhou`。 |

输出盘建议至少保留 50 GiB；不要把 `ASSET_ROOT` 和缓存目录放在空间不足的系统盘。

启动脚本会自动创建可写目录、验证全部模型和挂载、检查 GPU，并启动 Isaac Hub
缓存。任何必需组件缺失时都会在创建流水线任务前直接给出明确错误。

## 使用

API 地址：

- 健康检查：<http://127.0.0.1:8000/health>
- 接口页面：<http://127.0.0.1:8000/docs>

STEP/STP 自动赋材质示例。先把输入放到宿主机的 `ASSET_ROOT`，容器内统一使用
`/workspace/assets`：

```bash
docker exec -it hunyuan-pipeline-601 manual-material-pipeline \
  --stp /workspace/assets/input/model.stp \
  --sam3-annotations /workspace/assets/input/sam3_foreground_annotations.json \
  --output /workspace/assets/runs/model \
  --visual-inference-mode live
```

腾讯云密钥只用于可选的云端生成/细化；只运行 STEP/STP 自动赋材质时可以留空。

## 管理命令

```bash
docker/run-full.sh status       # 状态
docker/run-full.sh logs         # 日志
docker/run-full.sh shell        # 容器终端
docker/run-full.sh smoke        # 完整运行时检查
docker/run-full.sh down         # 停止 API 容器
docker/run-full.sh up           # 启动现有容器并等待健康；健康容器不会被重建
docker/run-full.sh build        # 依赖或 Dockerfile 变化后重建
docker/run-full.sh replace      # 删除同名旧容器并使用当前配置重建
```

源码以只读方式挂载，修改 Python 后执行 `docker/run-full.sh replace` 以重启 API 进程；
只有依赖或 Dockerfile 变化时才需要 `build`。

## 常见问题

| 现象 | 处理 |
| --- | --- |
| 容器反复重启 | 运行 `docker/run-full.sh preflight`，再看 `docker/run-full.sh logs` |
| `No space left on device` | 将输出和缓存改到大容量数据盘；确认后再清理旧镜像或缓存 |
| CUDA OOM | 同一 GPU 只运行一个重模型阶段，关闭其他占用 GPU 的进程 |
| EntitySeg 找不到 Detectron2 | 检查 `ENTITYSEG_BASE_RUNTIME_DIR` 和 `ENTITYSEG_DETECTRON2_ROOT` |
| API 健康但云端生成不可用 | 配置 `TENCENTCLOUD_SECRET_ID`、`TENCENTCLOUD_SECRET_KEY` 和区域 |

不要提交 `docker/.env.runtime`，也不要把密钥写入镜像。
