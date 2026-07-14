# HTTP API Module

[English](./api.md) | [中文](./api.zh.md) | [Documentation index](../README.md)

`asset_pipeline/api.py` uses FastAPI to expose Hunyuan generation and GLB postprocess workflows as background jobs. The API owns queue state, status, and logs while reusing the same pipeline functions.

## Start

```bash
python -m uvicorn asset_pipeline.api:app --host 0.0.0.0 --port 8000
```

The root `serve_api.py` contains a compatibility ASGI export, so the old `uvicorn serve_api:app` command still works.

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Runtime, materials, collision modes, and Tencent credential status. |
| `GET` | `/credentials/tencent-cloud` | Last credential-check result. |
| `POST` | `/credentials/tencent-cloud/check` | Refresh Tencent Cloud credential status. |
| `GET` | `/jobs` | List jobs. |
| `GET` | `/jobs/{job_id}` | Return one job's status, result, and logs. |
| `GET` | `/jobs/{job_id}/logs` | Return job logs. |
| `POST` | `/jobs/generate-model` | Run raw Hunyuan generation only. |
| `POST` | `/jobs/process-model` | Run Blender/Isaac postprocess on an existing GLB. |
| `POST` | `/jobs/generate-and-process-model` | Run Hunyuan generation, refine, Blender, and Isaac. |

## Job Model

Jobs live in the current API process memory:

```text
queued -> running -> succeeded / failed
```

`PIPELINE_MAX_WORKERS` controls thread-pool concurrency and defaults to `1`. `PIPELINE_MAX_LOG_LINES` limits retained log lines per job. In-memory job history is lost when the API process restarts.

## Example

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

The full image starts `asset_pipeline.api:app` by default. See [docker/README.md](../../docker/README.md) for image builds, GPU setup, Isaac Sim cache mounts, and more curl examples.
