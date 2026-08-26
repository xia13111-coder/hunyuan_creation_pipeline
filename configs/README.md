# Project configurations

Repository-owned runtime configuration is grouped by subsystem:

```text
configs/
├── physics/materials.json
└── refinement/hunyuan_reduce_local_postprocess.yaml
```

Package-internal settings remain with the package that owns them. In
particular, the Part-ID material pipeline reads
`tools/qwen_material_pipeline/configs/`; those files should not be duplicated
here.

## Mesh refinement

The default refine workflow is:

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
  --config ./configs/refinement/hunyuan_reduce_local_postprocess.yaml \
  --hunyuan-temp-upload uguu \
  --hunyuan-local-postprocess
```

The local postprocess step migrates PBR texture channels to the new UV layout:
`base_color.png`, `normal.png`, `roughness.png`, and `metallic.png`.

Some SAM3D GLBs have no image texture and store appearance in the mesh
`COLOR_0` vertex attribute. For the `base_color` channel, the Blender script
uses those vertex colors when needed and bakes them to the new UV layout. See
[`docs/modules/sam3d.md`](../docs/modules/sam3d.md) for all SAM3D options.

For large GLBs, upload a geometry-only proxy to Hunyuan while keeping the
original file as the local texture source:

```bash
blender --background --factory-startup \
  --python tools/blender/utilities/create_hunyuan_upload_proxy.py -- \
  --input ./input/1_phys.glb \
  --output ./output/1_phys_hunyuan_upload_proxy.glb

python -m asset_refiner \
  --input ./input/1_phys.glb \
  --output ./output/box \
  --config ./configs/refinement/hunyuan_reduce_local_postprocess.yaml \
  --hunyuan-temp-upload uguu \
  --hunyuan-upload-input ./output/1_phys_hunyuan_upload_proxy.glb \
  --hunyuan-local-postprocess
```

## Physics materials

`configs/physics/materials.json` is the single source of truth for named PhysX
friction, restitution, and density presets selected by `--material`. It does
not control NVIDIA MDL visual-material selection or colour calibration.

Runtime code resolves both configuration paths through
`asset_pipeline.project_layout.ProjectLayout`; avoid rebuilding these paths in
individual jobs.
