# Raw Hunyuan Generation Module

[English](./hunyuan.md) | [中文](./hunyuan.zh.md) | [Documentation index](../README.md)

For input and use-case differences from SAM3D, see
[Choosing Between Hunyuan And SAM3D](../guides/generation-guide.md).

This module submits a local image or image URL to Tencent Hunyuan 3D, downloads a raw GLB, and hands it to the refine module. It does not implement ReduceFace, UVs, physics, or USD conversion.

## Code

```text
asset_pipeline/jobs/hunyuan.py
  run_generate_model_job
  -> run_hunyuan_job
     -> python -m asset_pipeline.hunyuan_generation

asset_pipeline/hunyuan_generation.py
  Tencent SDK requests, polling, and downloads
```

## Inputs

| CLI option | Purpose |
| --- | --- |
| `--input-dir` | Scan JPG, JPEG, PNG, and WEBP files for image-to-3D jobs. |
| `--image-url` | Use one publicly reachable image URL. |
| `--face-count` | Requested Hunyuan Pro face count, such as `150000`. |
| `--download-preview` | Download the preview image returned by the API. |
| `--output-dir` | Raw generation directory. Default: `./downloads`. |

`--input-dir` and `--image-url` are mutually exclusive Hunyuan inputs. This
project does not expose text-only Hunyuan generation. Validation fails before
submission when the image directory is missing or contains no supported image.

## Command

```bash
hunyuan-asset-pipeline \
  --input-dir ./data \
  --output-dir ./outputs/hunyuan_example/generation \
  --face-count 150000 \
  --refine-config-path ./configs/refinement/hunyuan_reduce_local_postprocess.yaml \
  --refine-temp-upload uguu \
  --intermediate-output-dir ./outputs/hunyuan_example/intermediate \
  --final-output-dir ./outputs/hunyuan_example/final \
  --len-x 0.4 --len-y 0.3 --len-z 0.3 \
  --material plastic \
  --approx sdf
```

To inspect the generation client directly:

```bash
python -m asset_pipeline.hunyuan_generation --help
```

## Credentials

```bash
export TENCENTCLOUD_SECRET_ID="your-secret-id"
export TENCENTCLOUD_SECRET_KEY="your-secret-key"
```

Do not store real credentials in YAML, documentation, or git.

## Handoff

`run_generate_model_job` returns raw model files. The complete workflow passes `output_dir` to `jobs.refine.run_refine_mesh_job` by default. Only an explicit `--skip-refine` sends raw GLBs directly to Blender postprocess.

```text
Hunyuan raw GLB
-> Refine Mesh
-> Blender postprocess
-> Isaac physics
```

## Troubleshooting

| Problem | Action |
| --- | --- |
| Missing SecretId/SecretKey | Export both Tencent Cloud variables and verify the current shell can read them. |
| API submission failure | Check region, endpoint, account permission, and service quota. |
| No downloaded result | Inspect the JobId, status, and ResultFile3Ds in the generation log. |
| Existing GLB should be processed | Use `--existing-glb` instead of invoking generation again. |
