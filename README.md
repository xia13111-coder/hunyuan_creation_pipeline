# Hunyuan Creation Pipeline

End-to-end asset pipeline for Tencent Hunyuan 3D generation, mesh refinement, Blender post-processing, USD conversion, and Isaac Sim physics/collection.

## Pipeline

```text
image/prompt
-> Hunyuan 3D GLB generation
-> refine mesh
-> axis align
-> resize and center
-> GLB to USD
-> add physics
-> collect final USD asset
```

## Requirements

- Python/conda environment from `environment.yml`
- Tencent Cloud credentials in environment variables
- Blender, usually `/opt/blender/blender`
- Isaac Sim, for example `/home/user/isaacsim500/python.sh`

The runner auto-detects Blender and Isaac Sim on the local machine when possible.

## Run Full Pipeline

Put input images in `data/`, then run:

```bash
cd /path/to/hunyuan_creation_pipeline

export TENCENTCLOUD_SECRET_ID="your-real-secret-id"
export TENCENTCLOUD_SECRET_KEY="your-real-secret-key"

/home/user/miniconda3/envs/hunyuan/bin/python ./run_asset_pipeline.py \
  --input-dir ./data \
  --output-dir ./downloads \
  --intermediate-output-dir ./output_intermediate \
  --final-output-dir ./output_final \
  --result-json ./pipeline_result.json \
  --face-count 150000 \
  --len-x 0.4 \
  --len-y 0.3 \
  --len-z 0.3 \
  --orientation "X=L,Y=M,Z=S" \
  --material plastic \
  --approx sdf
```

## Run From Existing GLB

```bash
/home/user/miniconda3/envs/hunyuan/bin/python ./run_asset_pipeline.py \
  --existing-glb ./downloads/example_asset \
  --intermediate-output-dir ./output_intermediate \
  --final-output-dir ./output_final \
  --len-x 0.4 \
  --len-y 0.3 \
  --len-z 0.3 \
  --material plastic \
  --approx sdf
```

## Run Hand-Made GLB

For manually modeled assets, use the generic back-half pipeline. It skips
Hunyuan generation and mesh refinement, preserves the authored GLB geometry by
default, then converts to USD, adds physics/mass/material, and collects the
final asset.

```bash
/home/user/miniconda3/envs/hunyuan/bin/python ./run_asset_pipeline.py \
  --manual-glb "/home/user/下载/焊机/未命名.glb" \
  --intermediate-output-dir ./manual_output_intermediate \
  --final-output-dir ./manual_output_final \
  --result-json ./manual_pipeline_result.json \
  --material steel \
  --approx sdf \
  --set-mass 30
```

If a manual model needs axis mapping or resizing before USD conversion, add:

```bash
  --manual-align \
  --manual-resize \
  --len-x 0.6 \
  --len-y 0.4 \
  --len-z 0.5
```

## Notes

- Do not commit real Tencent Cloud credentials.
- Generated model outputs are ignored by `.gitignore`.
- Refine mesh uses `configs/hunyuan_reduce_local_postprocess.yaml` by default.
- More Docker/API details are in `README.docker.md`.
