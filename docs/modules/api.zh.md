# HTTP API 模块

[English](./api.md) | [中文](./api.zh.md) | [文档索引](../README.zh.md)

`asset_pipeline/api.py` 使用 FastAPI 把 Hunyuan 生成和 GLB 后处理 workflow 包装成后台任务。API 只负责任务队列、状态和日志，不重复实现 pipeline 逻辑。

## 启动

```bash
python -m uvicorn asset_pipeline.api:app --host 0.0.0.0 --port 8000
```

根目录 `serve_api.py` 只保留兼容 ASGI 导出，旧命令 `uvicorn serve_api:app` 仍可使用。

## 接口

| 方法 | 路径 | 作用 |
| --- | --- | --- |
| `GET` | `/health` | 环境、可用材质、collision 模式和腾讯云凭据状态。 |
| `GET` | `/credentials/tencent-cloud` | 最近一次凭据检查结果。 |
| `POST` | `/credentials/tencent-cloud/check` | 重新检查腾讯云凭据。 |
| `GET` | `/jobs` | 任务列表。 |
| `GET` | `/jobs/{job_id}` | 单个任务状态、结果和日志。 |
| `GET` | `/jobs/{job_id}/logs` | 任务日志。 |
| `POST` | `/jobs/generate-model` | 只运行 Hunyuan 原始生成。 |
| `POST` | `/jobs/process-model` | 对已有 GLB 运行 Blender/Isaac 后处理。 |
| `POST` | `/jobs/generate-and-process-model` | Hunyuan 生成、refine、Blender 和 Isaac 全流程。 |

## 任务模型

任务保存在当前 API 进程内存中：

```text
queued -> running -> succeeded / failed
```

`PIPELINE_MAX_WORKERS` 控制线程池并发数，默认 `1`。`PIPELINE_MAX_LOG_LINES` 控制每个任务保留的日志行数。API 进程重启后，内存任务记录不会持久化。

## 示例

```bash
curl -X POST http://127.0.0.1:8000/jobs/process-model \
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

## Docker

完整镜像默认启动 `asset_pipeline.api:app`。镜像构建、GPU、Isaac Sim 缓存挂载和更多 curl 示例见 [docker/README.md](../../docker/README.md)。
