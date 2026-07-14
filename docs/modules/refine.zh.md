# Refine Mesh 模块

[English](./refine.md) | [中文](./refine.zh.md) | [文档索引](../README.zh.md)

Refine 模块把原始 GLB 变成适合后续 USD 和物理处理的低模 GLB。生产路径固定为 Hunyuan ReduceFace 加本地 Blender 投影、UV 和贴图迁移。

## 代码

```text
asset_pipeline/jobs/refine.py
-> python -m asset_refiner
   -> asset_refiner/cli.py
   -> asset_refiner/runner.py
   -> asset_refiner/hunyuan_backend.py
      -> Tencent SubmitReduceFaceJob
      -> 下载低模 target
      -> Blender asset_refiner/blender_worker.py
```

## 流程

```text
source GLB
-> 临时上传，让 Hunyuan API 能访问本地文件
-> Hunyuan ReduceFace target
-> 导入 source 和 target
-> target 包围盒归一化并投影到 source
-> 清理、normal 修复和 UV 展开
-> 最近表面迁移 PBR 图片或 COLOR_0 顶点色
-> refined_asset.glb + qc_report.json
-> postprocess_glbs/<asset>_refined.glb
```

## 主 Pipeline 参数

| 参数 | 说明 |
| --- | --- |
| `--refine-config-path` | 配置文件。生产配置为 `configs/hunyuan_reduce_local_postprocess.yaml`。 |
| `--refine-output-dir` | 指定 refine 工作目录；不传时在输入旁创建 `*_refined_mesh`。 |
| `--refine-temp-upload` | 临时上传 provider。`uguu` 适用于本地 GLB；`none` 关闭。 |
| `--refine-fail-on-qc-error` | QC 为 fail 时让整个 pipeline 失败。 |
| `--skip-refine` | 仅用于诊断。Hunyuan/SAM3D 正常资产不建议跳过。 |

## 重点配置

| 配置 | 当前值 | 作用 |
| --- | --- | --- |
| `cleanup.merge_distance` | `0.0001` | 合并距离过近的顶点。 |
| `cleanup.degenerate_threshold` | `5.0e-07` | 清理极小或退化几何。 |
| `cleanup.min_component_faces` | `8` | 小组件面数候选阈值。 |
| `cleanup.min_component_area_ratio` | `1.0e-06` | 小组件相对面积候选阈值。 |
| `cleanup.max_component_cleanup_face_fraction` | `0.05` | 限制一次可删除的小组件总面数。 |
| `hunyuan.retopology.polygon_type` | `quadrilateral` | 请求四边面风格的 ReduceFace 结果。 |
| `hunyuan.retopology.face_level` | `high` | 请求较高细节层级。 |
| `retopology.method` | `external_target_project` | 使用 Hunyuan 低模作为目标拓扑并投影回源表面。 |
| `retopology.normalize_external_target_to_source_bbox` | `true` | 投影前对齐 source 和 target 包围盒。 |
| `retopology.shrinkwrap_iterations` | `1` | 回投影次数。 |
| `uv.method` | `lightmap_pack` | 自动生成并打包 UV。 |
| `textures.resolution` | `2048` | 输出贴图分辨率。 |
| `textures.transfer_max_distance` | `0.004` | 最近表面采样的最大搜索距离。 |
| `textures.transfer_normal_dot_min` | `0.05` | 源/目标 normal 兼容阈值。 |
| `textures.transfer_dilate_iterations` | `24` | 扩张有效像素，减少 UV seam 黑边。 |
| `qc.thresholds.max_projection_rms` | `0.03` | 允许的 RMS 投影误差。 |
| `qc.thresholds.max_projection_max` | `0.12` | 允许的最大单点投影误差。 |
| `qc.thresholds.max_uv_overlap_ratio` | `0.03` | 允许的 UV 重叠比例。 |

配置文件的直接调用说明见 [configs/README.md](../../configs/README.md)。

## SAM3D 顶点色

SAM3D 原始 GLB 可能没有 image、texture 或 PBR material，而把颜色放在 mesh `COLOR_0` 中。Blender worker 对 `base_color` 使用以下顺序：

1. 优先采样源材质的 base-color 图片。
2. 没有图片时插值采样源顶点色。
3. 两者都没有时才写默认颜色。

`qc_report.json` 中的 `source_vertex_color_found: true` 表示顶点色 fallback 已启用。`materials.json` 是 PhysX 材质，不参与视觉贴图迁移。

## 输出

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

后续 Blender 模块只读取 `postprocess_glbs`，不会把 Hunyuan target 或其它中间 GLB 重复处理。

## 错误处理

- `FailedOperation.RequestTimeout` 和配置中的其它临时提交错误会自动重试。
- `DownloadError` 可能触发重新临时上传，再次提交 URL。
- API job 状态为 `FAIL`、超过总超时，或本地 Blender 退出非零时，refine 失败并保留日志。
