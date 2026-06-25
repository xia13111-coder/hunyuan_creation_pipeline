# Hunyuan Docker

这个目录现在同时提供两种 Docker 方案：

- `Dockerfile`：仅复现 `hunyuan` conda 环境，适合只跑 Python 部分。
- `Dockerfile.full`：一体化镜像，包含 Isaac Sim 官方基础镜像、Blender 4.5.0、`hunyuan` conda 环境、Python 编排层和 HTTP API。

## 轻量版

```bash
docker build -t hunyuan-env .
docker run --rm -it \
  -v "$(pwd)":/workspace/hunyuan3.0_assets_creation \
  -e TENCENTCLOUD_SECRET_ID=你的ID \
  -e TENCENTCLOUD_SECRET_KEY=你的KEY \
  hunyuan-env
```

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

默认用 Isaac Sim 5.0.0 官方容器做基础镜像：

```bash
docker build -f Dockerfile.full -t hunyuan-allinone \
  --build-arg ISAACSIM_BASE_IMAGE=nvcr.io/nvidia/isaac-sim:5.0.0 .
```

如果你的环境只能拉取别的 Isaac Sim 5.0.x 标签，把上面的 `ISAACSIM_BASE_IMAGE` 改成对应值即可。

### 3. 准备 Isaac Sim 缓存目录

```bash
mkdir -p ~/docker/isaac-sim/{cache/main,cache/computecache,config,data,logs,pkg}
sudo chown -R 1234:1234 ~/docker/isaac-sim
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
  -p 8000:8000 \
  hunyuan-allinone
```

说明：

- 容器内默认路径：
  - `BLENDER_BIN=/opt/blender/blender`
  - `ISAACSIM_ROOT=/isaac-sim`
  - `ISAAC_PYTHON=/isaac-sim/python.sh`
- 容器默认启动 `uvicorn serve_api:app`，对外提供 HTTP 接口。
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

1. `hunyuan_to3d_batch.py`
2. `python -m asset_refiner`
3. `align_glb_axis_only.py`
4. `resize_glb_xyz_and_center.py`
5. `convert_glb_to_usd_zup.py`
6. `add_physics.py`
7. `collect_usd_flat.py`

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

- `face_count` 会映射到 `hunyuan_to3d_batch.py` 的 `--face-count`
- `refine_mesh` 默认为 `true`，表示混元生成 GLB 后先执行 refine mesh，再进入 Blender/Isaac 后处理
- `refine_config_path` 默认使用 `configs/hunyuan_reduce_local_postprocess.yaml`
- `refine_temp_upload` 默认取环境变量 `REFINE_MESH_TEMP_UPLOAD`，未设置时为 `uguu`；设为 `none`/`false`/`off`/`0` 可关闭临时上传
- refine 阶段会把最终 `refined_asset.glb` 汇总到 `postprocess_glbs` 目录，后续 Blender 步骤只处理这些最终 GLB，不处理 refine 中间文件
- `orientation` 会映射到 `align_glb_axis_only.py` 的 `--axis-map`
- `set_mass` 如果传了，就直接映射到 `add_physics.py` 的 `--set-mass`
- `set_mass` 如果不传，就走 `add_physics.py` 里的自动质量计算逻辑：按体积和材质密度估算
- `material` 取值来自 `materials.json`
- `approx` 直接映射到 `add_physics.py` 的 `--approx`

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
    "approx": "convexDecomposition",
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
