# 代码架构与调用关系

[English](./architecture.md) | [中文](./architecture.zh.md) | [文档索引](./README.zh.md)

## 分层原则

```text
入口层        run_asset_pipeline.py / serve_api.py
                 |
接口实现层    asset_pipeline/cli.py / api.py
                 |
工作流层      asset_pipeline/workflows.py
                 |
Job 层        asset_pipeline/jobs/*.py
                 |
执行层        asset_pipeline/command.py
                 |
外部进程      Hunyuan SDK / Blender / Isaac Sim / SAM3D
```

依赖只向下流动。`jobs` 不反向调用 CLI，独立工具也不知道完整 pipeline。这样每个函数的所属模块和上层调用者都比较明确。

## 包结构

| 文件 | 职责 |
| --- | --- |
| `asset_pipeline/runtime.py` | 查找 Blender、Isaac Python、SAM3D Python，提供项目路径和默认配置。 |
| `asset_pipeline/command.py` | 统一构造日志并运行外部子进程。 |
| `asset_pipeline/paths.py` | 文件筛选、运行目录命名和输出路径计算。 |
| `asset_pipeline/jobs/hunyuan.py` | 提交 Hunyuan 原始模型生成。 |
| `asset_pipeline/jobs/sam3d.py` | 准备单/多图目录并启动 SAM3D wrapper。 |
| `asset_pipeline/jobs/refine.py` | 对每个 GLB 启动 `asset_refiner` 并收集 refined GLB。 |
| `asset_pipeline/jobs/blender.py` | Blender 预检、轴对齐、尺寸和 GLB-to-USD。 |
| `asset_pipeline/jobs/isaac.py` | STEP/STP-to-USD、物理挂载和 USD 收集。 |
| `asset_pipeline/workflows.py` | 按输入类型组合 job，不实现底层算法。 |
| `asset_pipeline/cli.py` | 解析用户参数，选择一条 workflow。 |
| `asset_pipeline/api.py` | 把 workflow 包装成 FastAPI 后台任务。 |
| `asset_pipeline/hunyuan_generation.py` | Tencent SDK 客户端和可独立执行的生成 CLI。 |

`pipeline_runner.py` 只保留兼容导出。旧代码仍可使用 `import pipeline_runner`，新代码应直接 import 所属模块。

## 主 CLI

```text
run_asset_pipeline.py
-> asset_pipeline.cli.main
   -> runtime.configure_runtime
   -> ensure_generation_source
   -> run
```

`run` 的分支：

```text
--manual-stp
-> workflows.run_stp_physics_job

--sam3d-input
-> workflows.run_sam3d_image_and_process_model_job

--sam3d-glb / --existing-glb
-> jobs.refine.run_refine_mesh_job
-> workflows.run_process_model_job

图片 / 图片 URL / prompt
-> workflows.run_generate_and_process_model_job
```

## Hunyuan 生成

```text
workflows.run_generate_and_process_model_job
-> jobs.hunyuan.run_generate_model_job
   -> run_hunyuan_job
      -> python -m asset_pipeline.hunyuan_generation
-> jobs.refine.run_refine_mesh_job
-> workflows.run_process_model_job
```

## SAM3D

```text
workflows.run_sam3d_image_and_process_model_job
-> jobs.sam3d.run_sam3d_image_job
   -> prepare_sam3d_input
   -> command.run_command tools/sam3d/run_reconstruct.py
      -> SAM3 自动分割
      -> single-view 或 multi-view 重建
   -> select_sam3d_glb
-> jobs.refine.run_refine_mesh_job
-> workflows.run_process_model_job
```

## Refine

```text
jobs.refine.run_refine_mesh_job
-> python -m asset_refiner
   -> asset_refiner.cli.main
   -> asset_refiner.runner.run_refinement
   -> asset_refiner.hunyuan_backend.run_hunyuan_refinement
      -> 临时上传本地 GLB
      -> SubmitReduceFaceJob / DescribeReduceFaceJob
      -> 下载 Hunyuan target
      -> run_local_postprocess_worker
         -> Blender asset_refiner/blender_worker.py
```

Blender worker 的核心链路：

```text
run_pipeline
-> import_asset / join_as_whole_asset
-> clean_source_surface
-> whole_asset_retopology
-> generate_uv
-> migrate_textures
   -> 图片 PBR 最近表面采样
   -> 或 COLOR_0 顶点色到 base_color
-> export_final
-> build_qc_checks
```

## GLB 后处理

```text
workflows.run_process_model_job
-> run_postprocess_job
   -> jobs.blender.blender_preflight
   -> jobs.blender.run_align_job
      -> tools/blender/align_glb_axis_only.py
   -> jobs.blender.run_resize_job
      -> tools/blender/resize_glb_xyz_and_center.py
   -> jobs.blender.run_convert_job
      -> tools/blender/convert_glb_to_usd_zup.py
   -> jobs.isaac.run_add_physics_job
      -> tools/isaac/add_physics.py
   -> jobs.isaac.run_collect_job
      -> tools/isaac/collect_usd_flat.py
```

## 手工 CAD

```text
workflows.run_stp_physics_job
-> jobs.isaac.run_cad_to_usd_job
   -> tools/isaac/convert_cad_to_usd.py
-> 对每个 USD:
   -> jobs.isaac.run_add_physics_job(center_origin=True)
   -> jobs.isaac.run_collect_job
```

## 外部边界

CLI、API、Hunyuan、`asset_refiner` 编排代码和 SAM3D 统一运行在
`hunyuan_sam3d`。Blender 和 Isaac Sim 仍使用各自随程序提供的 Python，因此编排层通过
子进程调用，不在主 Python 中 import `bpy` 或 `pxr`；真正的 mesh/UV/texture 工作也仍由
Blender 子进程执行。
