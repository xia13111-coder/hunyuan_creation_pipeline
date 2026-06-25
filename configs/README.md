# Config Guide

Use these configs by workflow:

| Config | Backend | Purpose |
| --- | --- | --- |
| `hunyuan_reduce_local_postprocess.yaml` | Hunyuan API + local Blender | Recommended current production path. Tencent ReduceFace only; local bbox alignment, UV, texture migration, QC, export. |
| `local_postprocess_from_target.yaml` | local Blender | Re-run local UV/texture/QC from an existing low-poly target with `--retopo-target`; useful for debugging without another Tencent API call. |
| `hunyuan_api.yaml` | Hunyuan API | Pure Tencent API path. Can run ReduceFace only with `--hunyuan-no-uv`, or ReduceFace plus Tencent UV without that flag. |
| `default.yaml` | local Blender | Pure local automatic retopology, UV, texture migration, QC, export. |
| `smoke.yaml` | local Blender | Fast low-resolution validation. |

## Recommended Commands

Hunyuan ReduceFace plus local UV and texture migration:

```bash
python -m asset_refiner \
  --input ./input/apple.glb \
  --output ./output/apple_hunyuan_reduce_local_postprocess \
  --config ./configs/hunyuan_reduce_local_postprocess.yaml \
  --hunyuan-temp-upload uguu
```

The recommended local postprocess config migrates PBR texture channels to the
new UV layout: `base_color.png`, `normal.png`, `roughness.png`, and
`metallic.png`.

For large GLBs, upload a geometry-only proxy to Hunyuan while keeping the
original file as the local texture source:

```bash
blender --background --factory-startup \
  --python tools/create_hunyuan_upload_proxy.py -- \
  --input ./input/1_phys.glb \
  --output ./output/1_phys_hunyuan_upload_proxy.glb

python -m asset_refiner \
  --input ./input/1_phys.glb \
  --output ./output/box \
  --config ./configs/hunyuan_reduce_local_postprocess.yaml \
  --hunyuan-temp-upload uguu \
  --hunyuan-upload-input ./output/1_phys_hunyuan_upload_proxy.glb
```

Reuse an already downloaded low-poly target:

```bash
python -m asset_refiner \
  --input ./input/apple.glb \
  --output ./output/apple_local_postprocess_from_target \
  --config ./configs/local_postprocess_from_target.yaml \
  --retopo-target ./output/apple_hunyuan_reduce_local_postprocess/intermediate/hunyuan_api/hunyuan_reduce_target.glb
```

Hunyuan ReduceFace only:

```bash
python -m asset_refiner \
  --input ./input/apple.glb \
  --output ./output/apple_hunyuan_reduce_only \
  --config ./configs/hunyuan_api.yaml \
  --hunyuan-temp-upload uguu \
  --hunyuan-no-uv
```
