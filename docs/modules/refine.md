# Refine Mesh Module

[English](./refine.md) | [中文](./refine.zh.md) | [Documentation index](../README.md)

The refine module turns a raw GLB into a reduced GLB suitable for USD and physics processing. The production path is Hunyuan ReduceFace followed by local Blender projection, UV generation, and texture migration.

## Code

```text
asset_pipeline/jobs/refine.py
-> python -m asset_refiner
   -> asset_refiner/cli.py
   -> asset_refiner/runner.py
   -> asset_refiner/hunyuan_backend.py
      -> Tencent SubmitReduceFaceJob
      -> download reduced target
      -> Blender asset_refiner/blender_worker.py
```

## Flow

```text
source GLB
-> temporary upload so Hunyuan can reach the local file
-> Hunyuan ReduceFace target
-> import source and target
-> normalize target bounding box and project it to source
-> cleanup, normal repair, and UV unwrap
-> nearest-surface PBR image or COLOR_0 vertex-color migration
-> refined_asset.glb + qc_report.json
-> postprocess_glbs/<asset>_refined.glb
```

## Main Pipeline Options

| Option | Meaning |
| --- | --- |
| `--refine-config-path` | Configuration file. Production uses `configs/hunyuan_reduce_local_postprocess.yaml`. |
| `--refine-output-dir` | Refine workspace. When omitted, `*_refined_mesh` is created next to the input. |
| `--refine-temp-upload` | Temporary upload provider. Use `uguu` for local GLBs or `none` to disable. |
| `--refine-fail-on-qc-error` | Fail the complete pipeline when QC reports `fail`. |
| `--skip-refine` | Diagnostic bypass only. Normal Hunyuan/SAM3D assets should retain refine. |

## Important Configuration

| Setting | Current value | Purpose |
| --- | --- | --- |
| `cleanup.merge_distance` | `0.0001` | Merge nearby vertices. |
| `cleanup.degenerate_threshold` | `5.0e-07` | Remove very small or degenerate geometry. |
| `cleanup.min_component_faces` | `8` | Face-count threshold for small-component candidates. |
| `cleanup.min_component_area_ratio` | `1.0e-06` | Relative-area threshold for small-component candidates. |
| `cleanup.max_component_cleanup_face_fraction` | `0.05` | Limit the total face fraction removed as small components. |
| `hunyuan.retopology.polygon_type` | `quadrilateral` | Request quad-style ReduceFace output. |
| `hunyuan.retopology.face_level` | `high` | Request the higher detail level. |
| `retopology.method` | `external_target_project` | Use the reduced target as topology and project it back to the source surface. |
| `retopology.normalize_external_target_to_source_bbox` | `true` | Align source and target bounding boxes before projection. |
| `retopology.shrinkwrap_iterations` | `1` | Number of projection passes. |
| `uv.method` | `lightmap_pack` | Generate and pack UVs automatically. |
| `textures.resolution` | `2048` | Output texture resolution. |
| `textures.transfer_max_distance` | `0.004` | Maximum nearest-surface search distance. |
| `textures.transfer_normal_dot_min` | `0.05` | Source/target normal compatibility threshold. |
| `textures.transfer_dilate_iterations` | `24` | Expand valid pixels to reduce black UV seams. |
| `qc.thresholds.max_projection_rms` | `0.03` | Allowed RMS projection error. |
| `qc.thresholds.max_projection_max` | `0.12` | Allowed worst-point projection error. |
| `qc.thresholds.max_uv_overlap_ratio` | `0.03` | Allowed UV overlap ratio. |

See [configs/README.md](../../configs/README.md) for direct config usage.

## SAM3D Vertex Colors

A raw SAM3D GLB may contain no image, texture, or PBR material and store appearance in mesh `COLOR_0`. The Blender worker resolves `base_color` in this order:

1. Sample a source base-color image when available.
2. Otherwise interpolate source vertex colors.
3. Use the fallback color only when neither source exists.

`source_vertex_color_found: true` in `qc_report.json` confirms that the vertex-color fallback ran. `materials.json` contains PhysX properties and does not participate in visual texture migration.

## Outputs

```text
<refine-root>/
├── <asset>/
│   ├── refined_asset.glb
│   ├── qc_report.json
│   ├── resolved_config.json
│   ├── resolved_local_postprocess_config.json
│   └── intermediate/hunyuan_api/
└── postprocess_glbs/
    └── <asset>_refined.glb
```

The downstream Blender module reads only `postprocess_glbs`, so Hunyuan targets and intermediate GLBs are not processed again.

## Error Handling

- `FailedOperation.RequestTimeout` and other configured transient submission errors are retried automatically.
- A Hunyuan `DownloadError` can trigger a fresh temporary upload and URL submission.
- Refine fails and preserves logs when the API job returns `FAIL`, exceeds its total timeout, or local Blender exits nonzero.
