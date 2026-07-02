# Config Guide

The pipeline now keeps one production refine path:

```text
Tencent Hunyuan ReduceFace
-> local Blender UV unwrap
-> nearest-surface texture migration
-> QC
-> refined GLB export
```

Use:

```bash
python -m asset_refiner \
  --input ./input/apple.glb \
  --output ./output/apple_refined \
  --config ./configs/hunyuan_reduce_local_postprocess.yaml \
  --hunyuan-temp-upload uguu \
  --hunyuan-local-postprocess
```

The local postprocess step migrates PBR texture channels to the new UV layout:
`base_color.png`, `normal.png`, `roughness.png`, and `metallic.png`.

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
  --hunyuan-upload-input ./output/1_phys_hunyuan_upload_proxy.glb \
  --hunyuan-local-postprocess
```
