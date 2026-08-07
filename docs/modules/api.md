# HTTP API Module

[English](./api.md) | [中文](./api.zh.md) | [Documentation index](../README.md)

`asset_pipeline/api.py` exposes Hunyuan generation and asset-processing pipelines
as FastAPI background jobs. It manages queue state and logs while calling the
same functions as the CLI.

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
| `POST` | `/jobs/process-manual-cad` | Convert STEP/STP to USD, add physics, optionally assign reference-image materials, collect dependencies, and validate the result. |
| `POST` | `/jobs/generate-and-process-model` | Run Hunyuan generation, refine, Blender, and Isaac. |

The two Hunyuan generation endpoints accept exactly one of `input_dir` and
`image_url`. A `prompt` field is rejected.

## Job Model

Jobs live in the current API process memory:

```text
queued -> running -> succeeded / failed
```

`PIPELINE_MAX_WORKERS` controls thread-pool concurrency and defaults to `1`. `PIPELINE_MAX_LOG_LINES` limits retained log lines per job. In-memory job history is lost when the API process restarts.

## Example

This Docker API example uses fixed container paths under `/workspace/assets`;
they are not absolute paths from the target host. See the Docker operations
guide for the host-directory mapping.

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

Hand-authored CAD has a dedicated endpoint and requires no size fields:

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

The endpoint calls `manual_cad.run_manual_cad_workflow`. With reference-image
materials enabled, geometry preparation runs before NVIDIA MDL selection;
dependencies are collected only after the material result is validated. This
order keeps object-space procedural textures stable. The request has no
`len_x/len_y/len_z` or `orientation` fields and rejects them. Schema errors
return HTTP 422, preflight errors return HTTP 400, and subprocess failures mark
the job as `failed`.

Complete the local SAM3 annotation UI before submitting the background job.
Pass its JSON path as `visual_foreground_annotations`; this field is restricted
to the STEP/STP endpoint with `auto_visual_materials: true` and
`visual_inference_mode: "live"`. A queued job never waits for browser clicks.

The example above runs CAD + Physics only. To enable reference-image materials,
configure the dependencies listed in the visual-material guide, including
Qwen3.5, SAM3, MVInverse, SigLIP2, DINOv2, the Base observation bank, NVIDIA
Materials, and the reference images. Then set `auto_visual_materials` to
`true`. See
[Reference-image material assignment](./visual-materials.md).

## Docker

The full image starts `asset_pipeline.api:app` by default. The provided
container image uses Isaac Sim 6.0.1. The API exposes Hunyuan generation,
existing-GLB processing, and manual STEP/STP processing; run SAM3D image
reconstruction through the pipeline CLI inside the same container.

`hunyuan-allinone:isaac-6.0.1-materials` includes the shared visual-material
dependencies in `hunyuan_sam3d`. Qwen3.5 uses the separately mounted runtime
described in the Docker guide. Model weights, the Base observation bank, and
NVIDIA Materials remain read-only mounts and are not baked into the image.

See the [Docker operations guide](../../docker/README.md) for offline image
export/import, GPU setup, Hub/cache mounts, CLI/API commands, lifecycle
operations, and acceptance checks.
