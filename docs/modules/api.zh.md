# HTTP API 模块

[English](./api.md) | [中文](./api.zh.md) | [文档索引](../README.zh.md)

`asset_pipeline/api.py` 通过 FastAPI 把 Hunyuan 生成和资产处理流程包装成后台任务。API
管理队列状态和日志，实际处理仍调用与 CLI 相同的函数。

## 启动

```bash
python -m uvicorn asset_pipeline.api:app --host 0.0.0.0 --port 8000
```

`asset_pipeline.api:app` 是唯一 ASGI 应用路径；仓库根目录不再保留重复的 API 包装文件。

## 接口

| 方法 | 路径 | 作用 |
| --- | --- | --- |
| `GET` | `/health` | 环境、可用材质、碰撞模式和腾讯云凭据状态。 |
| `GET` | `/credentials/tencent-cloud` | 最近一次凭据检查结果。 |
| `POST` | `/credentials/tencent-cloud/check` | 重新检查腾讯云凭据。 |
| `GET` | `/jobs` | 任务列表。 |
| `GET` | `/jobs/{job_id}` | 单个任务状态、结果和日志。 |
| `GET` | `/jobs/{job_id}/logs` | 任务日志。 |
| `POST` | `/jobs/generate-model` | 只运行 Hunyuan 原始生成。 |
| `POST` | `/jobs/process-model` | 对已有 GLB 运行 Blender/Isaac 后处理。 |
| `POST` | `/jobs/process-manual-cad` | 把 STEP/STP 转为 USD、添加物理、可选参考图赋材质、收集依赖并验证结果。 |
| `POST` | `/jobs/generate-and-process-model` | Hunyuan 生成、网格精修、Blender 和 Isaac 全流程。 |

两个 Hunyuan 生成接口只接受 `input_dir` 或 `image_url`，且必须二选一；`prompt` 字段会被拒绝。

## 任务模型

任务保存在当前 API 进程内存中：

```text
queued -> running -> succeeded / failed
```

`PIPELINE_MAX_WORKERS` 控制线程池并发数，默认 `1`。
`PIPELINE_MAX_LOG_LINES` 控制每个任务保留的日志行数。API 进程重启后，内存中的任务记录
不会保留。

## 示例

下面是 Docker API 示例，JSON 中的 `/workspace/assets/...` 是容器内部固定路径，
不是目标电脑上的绝对路径。宿主机目录映射方式见 Docker 操作手册。

```bash
curl --noproxy '*' -X POST http://127.0.0.1:8000/jobs/process-model \
  -H 'Content-Type: application/json' \
  -d '{
    "input_path": "/workspace/assets/model.glb",
    "intermediate_output_dir": "/workspace/assets/intermediate",
    "final_output_dir": "/workspace/assets/final",
    "len_x": 0.4,
    "len_y": 0.3,
    "len_z": 0.8,
    "orientation": "X=L,Y=M,Z=S",
    "material": "plastic",
    "set_mass": null,
    "approx": "sdf"
  }'
```

手工建模 STEP/STP（CAD）使用独立接口，不需要尺寸字段：

```bash
curl --noproxy '*' -X POST http://127.0.0.1:8000/jobs/process-manual-cad \
  -H 'Content-Type: application/json' \
  -d '{
    "input_path": "/workspace/assets/manual_asset.stp",
    "cad_usd_output_dir": "/workspace/assets/manual/cad_usd",
    "intermediate_output_dir": "/workspace/assets/manual/intermediate",
    "final_output_dir": "/workspace/assets/manual/final",
    "auto_visual_materials": false,
    "material": "steel",
    "approx": "sdf",
    "sdf_resolution": 32
  }'
```

接口调用 `manual_cad.run_manual_cad_workflow`。启用参考图赋材质时，先完成物理几何
准备，再选择 NVIDIA MDL；材质验证通过后才收集依赖。这个顺序可避免使用物体坐标的
程序化纹理在选材后发生变化。请求中没有 `len_x/len_y/len_z` 或 `orientation`，传入这些
字段会被拒绝。数据结构错误返回 HTTP 422，运行前检查错误返回 HTTP 400，模型或 Isaac
子进程失败时任务状态为 `failed`。

本地 SAM3 标注页面应在提交后台任务前单独完成。提交时把生成的 JSON 路径放入
`visual_foreground_annotations`；该字段只允许与 `auto_visual_materials: true`、
`visual_inference_mode: "live"` 和 STEP/STP 接口一起使用。后台任务不会停住等待
网页点击。

上例只运行 CAD 转换和物理处理。自动赋材质还需要 Qwen3.5、SAM3、MVInverse、SigLIP2、
DINOv2、Base 材质观察库、NVIDIA Materials 和参考图。依赖准备完成后，可把
`auto_visual_materials` 改为 `true`；完整字段见
[STEP/STP 工件参考图自动赋材质](./visual-materials.zh.md)。

## Docker

完整镜像默认启动 `asset_pipeline.api:app`。当前已验证的镜像基于 Isaac Sim 6.0.1。
API 提供 Hunyuan 生成、已有 GLB 后处理和手工建模 STEP/STP 处理；SAM3D 图片重建仍通过
同一容器内的命令行运行。

`hunyuan-allinone:isaac-6.0.1-materials` 已在统一 `hunyuan_sam3d` 中包含
Qwen/MVInverse Python 依赖。模型权重和 NVIDIA Materials 按设计不打入镜像，必须按
Docker 手册只读挂载。

离线镜像导出/导入、GPU、Hub/缓存挂载、CLI/API 命令、容器维护和验收步骤见
[Docker 操作手册](../../docker/README.zh.md)。
