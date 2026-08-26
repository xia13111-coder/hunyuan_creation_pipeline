# 网格精修模块

[English](./refine.md) | [中文](./refine.zh.md) | [文档索引](../README.zh.md)

网格精修模块把原始 GLB 转为适合后续 USD 和物理处理的低模 GLB。默认流程使用 Hunyuan ReduceFace，再由本地 Blender 完成投影、UV 和贴图迁移。

## 代码

```text
asset_pipeline/jobs/refine.py
-> python -m asset_refiner
   -> asset_refiner/cli.py
   -> asset_refiner/runner.py
   -> asset_refiner/hunyuan_backend.py
      -> Tencent SubmitReduceFaceJob
      -> 下载低模结果
      -> Blender asset_refiner/blender_worker.py
```

## 流程

```text
源 GLB
-> 临时上传，让 Hunyuan API 能访问本地文件
-> Hunyuan ReduceFace 低模结果
-> 导入源模型和低模结果
-> 对齐包围盒并把低模投影回源表面
-> 清理、法线修复和 UV 展开
-> 最近表面迁移 PBR 图片或 COLOR_0 顶点色
-> refined_asset.glb + qc_report.json
-> postprocess_glbs/<asset>_refined.glb
```

## 主流程参数

| 参数 | 说明 |
| --- | --- |
| `--refine-config-path` | 配置文件。生产配置为 `configs/refinement/hunyuan_reduce_local_postprocess.yaml`。 |
| `--refine-output-dir` | 指定精修工作目录；不传时在输入旁创建 `*_refined_mesh`。 |
| `--refine-temp-upload` | 临时上传服务。`uguu` 适用于本地 GLB；`none` 关闭。 |
| `--refine-fail-on-qc-error` | QC 为 `fail` 时停止整个流程。 |
| `--skip-refine` | 仅用于诊断。Hunyuan/SAM3D 正常资产不建议跳过。 |

## 重点配置

下表只列出最常影响外观结果的设置。清理阈值和高级选项见完整配置说明。

| 配置 | 当前值 | 作用 |
| --- | --- | --- |
| `hunyuan.retopology.polygon_type` | `quadrilateral` | 请求四边面风格的 ReduceFace 结果。 |
| `hunyuan.retopology.face_level` | `high` | 请求较高细节层级。 |
| `retopology.method` | `external_target_project` | 使用 Hunyuan 低模作为目标拓扑并投影回源表面。 |
| `retopology.normalize_external_target_to_source_bbox` | `true` | 投影前对齐源模型和目标模型的包围盒。 |
| `uv.method` | `lightmap_pack` | 自动生成并打包 UV。 |
| `textures.resolution` | `2048` | 输出贴图分辨率。 |
| `textures.transfer_max_distance` | `0.004` | 最近表面采样的最大搜索距离。 |
| `textures.transfer_dilate_iterations` | `24` | 扩张有效像素，减少 UV 接缝黑边。 |
| `qc.thresholds.max_projection_rms` | `0.03` | 允许的 RMS 投影误差。 |
| `qc.thresholds.max_projection_max` | `0.12` | 允许的最大单点投影误差。 |
| `qc.thresholds.max_uv_overlap_ratio` | `0.03` | 允许的 UV 重叠比例。 |

全部设置和配置文件的直接调用方式见 [configs/README.md](../../configs/README.md)。

## SAM3D 顶点色

SAM3D 原始 GLB 可能没有图片贴图或 PBR 材质，而把颜色保存在网格的 `COLOR_0` 属性中。Blender 脚本按以下顺序确定 `base_color`：

1. 优先采样源材质的基础色贴图。
2. 没有图片时插值采样源顶点色。
3. 两者都没有时才写默认颜色。

`qc_report.json` 中的 `source_vertex_color_found: true` 表示流程使用了源顶点色。
`configs/physics/materials.json` 只定义 PhysX 材质，不参与视觉贴图迁移。

## 输出

```text
<精修目录>/
├── <asset>/
│   ├── refined_asset.glb
│   ├── qc_report.json
│   ├── resolved_config.json
│   ├── resolved_local_postprocess_config.json
│   └── intermediate/hunyuan_api/
└── postprocess_glbs/
    └── <asset>_refined.glb
```

后续 Blender 模块只读取 `postprocess_glbs`，不会重复处理 Hunyuan 低模或其他中间 GLB。

## 错误处理

- `FailedOperation.RequestTimeout` 和配置中的其他临时提交错误会自动重试。
- `DownloadError` 可能触发重新临时上传，再次提交 URL。
- API 任务状态为 `FAIL`、超过总超时，或本地 Blender 退出非零时，精修失败并保留日志。
