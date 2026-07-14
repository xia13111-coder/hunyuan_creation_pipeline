# Hunyuan Docker

这个目录集中放置项目的 Docker 相关文件：

- `Dockerfile`：构建可复用的 `hunyuan_sam3d` conda 环境镜像，适合运行 Python、Hunyuan refine 和 SAM3D 部分，也是完整镜像的环境层。
- `Dockerfile.full`：一体化镜像，基于环境层加入 Isaac Sim、Blender 4.5.0、Python 编排层和 HTTP API。
- `docker-entrypoint.full.sh`：完整镜像的入口脚本。
- `Dockerfile.dockerignore` / `Dockerfile.full.dockerignore`：对应 Dockerfile 的 build ignore 规则。

下面的构建命令都在项目根目录执行，build context 仍然是项目根目录，这样 Dockerfile 可以复制 `environment.yml` 和项目源码。

## 轻量版

```bash
docker build -f docker/Dockerfile -t hunyuan-sam3d-env:latest .
docker run --rm -it \
  -v "$(pwd)":/workspace/hunyuan3.0_assets_creation \
  -e TENCENTCLOUD_SECRET_ID=你的ID \
  -e TENCENTCLOUD_SECRET_KEY=你的KEY \
  hunyuan-sam3d-env:latest
```

这里必须挂载项目根目录。`tools/sam3d/third_party/` 包含约几十 GB 的上游源码、权重和
本机扩展，已排除在镜像构建上下文之外；挂载后容器内仍能按默认路径访问它们。
环境镜像使用 BuildKit 的 pip 缓存和断点续传配置；Torch/CUDA wheel 很大，网络中断后
重新执行同一条 `docker build` 命令即可继续复用已下载内容。完整镜像直接复用这个环境层，
不会重新安装全部 Python 依赖。

## 完整一体化版

### 1. 宿主机要求

- Linux
- Docker
- NVIDIA GPU 与驱动
- NVIDIA Container Toolkit

Isaac Sim 官方文档说明：

- 容器安装适合部署在远程无头 Linux 服务器
- 需要 `nvidia-container-toolkit`
- GPU 必须带 RT Cores，A100 / H100 这类无 RT Cores 的卡不支持

### 2. 构建镜像

如果你的环境还没登录过 NVIDIA NGC，先执行：

```bash
docker login nvcr.io
```

用户名一般填 `$oauthtoken`，密码填你的 NGC API Key。

完整镜像依赖轻量环境镜像。首次构建时，先执行上一节的环境镜像构建命令；之后默认用
`hunyuan-sam3d-env:latest` 作为环境层，再用 Isaac Sim 5.0.0 官方容器做基础镜像：

```bash
docker build -f docker/Dockerfile.full -t hunyuan-allinone \
  --build-arg PIPELINE_ENV_IMAGE=hunyuan-sam3d-env:latest \
  --build-arg ISAACSIM_BASE_IMAGE=nvcr.io/nvidia/isaac-sim:5.0.0 .
```

如需使用 Isaac Sim 6.0.1，Dockerfile 无需修改，只需替换基础镜像标签：

```bash
docker build -f docker/Dockerfile.full -t hunyuan-allinone:isaac-6.0.1 \
  --build-arg PIPELINE_ENV_IMAGE=hunyuan-sam3d-env:latest \
  --build-arg ISAACSIM_BASE_IMAGE=nvcr.io/nvidia/isaac-sim:6.0.1 .
```

也可以把 `ISAACSIM_BASE_IMAGE` 替换为本机已有的兼容 Isaac Sim 镜像。例如本机已有公司镜像源中的 Isaac Sim 5.0.0 时：

```bash
docker build -f docker/Dockerfile.full -t hunyuan-allinone \
  --build-arg PIPELINE_ENV_IMAGE=hunyuan-sam3d-env:latest \
  --build-arg ISAACSIM_BASE_IMAGE=glcr.rd.ubtrobot.com/pub/docker/isaac:isaac_sim_5.0.0 .
```

完整 Dockerfile 会兼容基础镜像中的 `/isaac-sim` 和 `/opt/isaac-sim` 两种安装位置，
并默认创建 UID/GID 为 `1234:1234`、HOME 为 `/home/pipeline` 的运行环境。需要匹配宿主机
权限时，可通过 `--build-arg APP_UID=... --build-arg APP_GID=...` 覆盖。环境镜像使用其他
标签时，同时传入 `--build-arg PIPELINE_ENV_IMAGE=<你的环境镜像标签>`。

### 3. 准备 Isaac Sim 缓存目录

```bash
mkdir -p ~/docker/isaac-sim/{cache/main,cache/computecache,config,data,logs,pkg}
mkdir -p ~/.cache/ov/hub
sudo chown -R 1234:1234 ~/docker/isaac-sim ~/.cache/ov/hub
```

### 4. 运行容器

```bash
docker run --rm -it --gpus all \
  -e ACCEPT_EULA=Y \
  -e PRIVACY_CONSENT=Y \
  -e TENCENTCLOUD_SECRET_ID=你的ID \
  -e TENCENTCLOUD_SECRET_KEY=你的KEY \
  -e ROOT_DIR=/workspace/hunyuan3.0_assets_creation \
  -e ROOT_DIR1=/workspace/assets \
  -v "$(pwd)":/workspace/hunyuan3.0_assets_creation \
  -v /你的资产目录:/workspace/assets \
  -v ~/docker/isaac-sim/cache/main:/isaac-sim/.cache:rw \
  -v ~/docker/isaac-sim/cache/computecache:/isaac-sim/.nv/ComputeCache:rw \
  -v ~/docker/isaac-sim/logs:/isaac-sim/.nvidia-omniverse/logs:rw \
  -v ~/docker/isaac-sim/config:/isaac-sim/.nvidia-omniverse/config:rw \
  -v ~/docker/isaac-sim/data:/isaac-sim/.local/share/ov/data:rw \
  -v ~/docker/isaac-sim/pkg:/isaac-sim/.local/share/ov/pkg:rw \
  -v ~/.cache/ov/hub:/var/cache/hub:rw \
  -p 8000:8000 \
  hunyuan-allinone
```

说明：

- 容器内默认路径：
  - `BLENDER_BIN=/opt/blender/blender`
  - `ISAACSIM_ROOT=/isaac-sim`
  - `ISAAC_PYTHON=/isaac-sim/python.sh`
- 容器默认启动 `uvicorn asset_pipeline.api:app`，对外提供 HTTP 接口。
- 容器里的主 Python 环境固定为 `hunyuan_sam3d`；无需设置 `SAM3D_PYTHON`。
- 运行 SAM3D 时，项目根目录挂载会同时提供 `tools/sam3d/third_party/`。其中的本机编译
  扩展必须与镜像内 CUDA、PyTorch 和系统 ABI 兼容；若不兼容，需要在容器内重新编译
  对应扩展。
- 如果你只做离线批处理，不做 Isaac Sim WebRTC 直播，通常不需要 `--network=host`。

### 5. 接口说明

健康检查：

```bash
curl http://127.0.0.1:8000/health
```

查看任务列表：

```bash
curl http://127.0.0.1:8000/jobs
```

现在只对外开放 3 个业务接口：

- `POST /jobs/generate-model`
- `POST /jobs/process-model`
- `POST /jobs/generate-and-process-model`

其中 `generate-and-process-model` 会按下面顺序直接调用 Python 文件，不依赖 `.sh`：

1. `python -m asset_pipeline.hunyuan_generation`
2. `python -m asset_refiner`
3. `tools/blender/align_glb_axis_only.py`
4. `tools/blender/resize_glb_xyz_and_center.py`
5. `tools/blender/convert_glb_to_usd_zup.py`
6. `tools/isaac/add_physics.py`
7. `tools/isaac/collect_usd_flat.py`

`process-model` 仍然只执行第 3-7 步，适合直接处理已有 GLB。

暴露给接口层的业务参数只有这些：

- `face_count`
- `refine_mesh`
- `refine_output_dir`
- `refine_config_path`
- `refine_temp_upload`
- `refine_fail_on_qc_error`
- `len_x`
- `len_y`
- `len_z`
- `orientation`
- `set_mass`
- `material`
- `approx`

说明：

- `face_count` 会映射到 `asset_pipeline.hunyuan_generation` 的 `--face-count`
- `refine_mesh` 默认为 `true`，表示混元生成 GLB 后先执行 refine mesh，再进入 Blender/Isaac 后处理
- `refine_config_path` 默认使用 `configs/hunyuan_reduce_local_postprocess.yaml`
- `refine_temp_upload` 默认取环境变量 `REFINE_MESH_TEMP_UPLOAD`，未设置时为 `uguu`；设为 `none`/`false`/`off`/`0` 可关闭临时上传
- refine 阶段会把最终 `refined_asset.glb` 汇总到 `postprocess_glbs` 目录，后续 Blender 步骤只处理这些最终 GLB，不处理 refine 中间文件
- `orientation` 会映射到 `tools/blender/align_glb_axis_only.py` 的 `--axis-map`
- `set_mass` 如果传了，就直接映射到 `tools/isaac/add_physics.py` 的 `--set-mass`
- `set_mass` 如果不传，就走 `tools/isaac/add_physics.py` 里的自动质量计算逻辑：按体积和材质密度估算
- `material` 取值来自 `materials.json`
- `approx` 直接映射到 `tools/isaac/add_physics.py` 的 `--approx`

只做后处理：

```bash
curl -X POST http://127.0.0.1:8000/jobs/process-model \
  -H 'Content-Type: application/json' \
  -d '{
    "input_path": "/workspace/assets/downloads_carton_0.4_0.3_0.3",
    "intermediate_output_dir": "/workspace/assets/carton_intermediate",
    "final_output_dir": "/workspace/assets/carton_final",
    "len_x": 0.4,
    "len_y": 0.3,
    "len_z": 0.3,
    "orientation": "X=L,Y=M,Z=S",
    "material": "plastic",
    "set_mass": 5.0,
    "approx": "convexDecomposition"
  }'
```

只做模型生成：

```bash
curl -X POST http://127.0.0.1:8000/jobs/generate-model \
  -H 'Content-Type: application/json' \
  -d '{
    "input_dir": "/workspace/hunyuan3.0_assets_creation/data",
    "output_dir": "/workspace/hunyuan3.0_assets_creation/downloads",
    "face_count": 150000
  }'
```

先生成再后处理：

```bash
curl -X POST http://127.0.0.1:8000/jobs/generate-and-process-model \
  -H 'Content-Type: application/json' \
  -d '{
    "input_dir": "/workspace/hunyuan3.0_assets_creation/data",
    "output_dir": "/workspace/hunyuan3.0_assets_creation/downloads",
    "intermediate_output_dir": "/workspace/assets/carton_intermediate",
    "final_output_dir": "/workspace/assets/carton_final",
    "face_count": 150000,
    "refine_mesh": true,
    "len_x": 0.4,
    "len_y": 0.3,
    "len_z": 0.3,
    "orientation": "X=L,Y=M,Z=S",
    "set_mass": null,
    "material": "plastic",
    "approx": "convexDecomposition"
  }'
```

查询单个任务状态：

```bash
curl http://127.0.0.1:8000/jobs/<job_id>
```

### 6. 可用接口

- `GET /health`
- `GET /jobs`
- `GET /jobs/{job_id}`
- `GET /jobs/{job_id}/logs`
- `POST /jobs/generate-model`
- `POST /jobs/process-model`
- `POST /jobs/generate-and-process-model`

### 7. 如果要进容器手动调试

覆盖默认命令即可：

```bash
docker run --rm -it --gpus all ... hunyuan-allinone bash
```
