# Code Architecture And Call Graph

[English](./architecture.md) | [中文](./architecture.zh.md) | [Documentation index](./README.md)

## Layers

```text
entry points       run_asset_pipeline.py / serve_api.py
                        |
interfaces         asset_pipeline/cli.py / api.py
                        |
workflows          asset_pipeline/workflows.py
                        |
jobs               asset_pipeline/jobs/*.py
                        |
execution          asset_pipeline/command.py
                        |
external processes Hunyuan SDK / Blender / Isaac Sim / SAM3D
```

Dependencies flow downward. Jobs do not call the CLI, and standalone tools do not know about complete workflows. This keeps function ownership and callers explicit.

## Package Ownership

| File | Responsibility |
| --- | --- |
| `asset_pipeline/runtime.py` | Discover Blender, Isaac Python, SAM3D Python, project paths, and defaults. |
| `asset_pipeline/command.py` | Log and execute external subprocesses consistently. |
| `asset_pipeline/paths.py` | Filter files, name run directories, and calculate output paths. |
| `asset_pipeline/jobs/hunyuan.py` | Submit raw Hunyuan model generation. |
| `asset_pipeline/jobs/sam3d.py` | Prepare single/multi-view inputs and start the SAM3D wrapper. |
| `asset_pipeline/jobs/refine.py` | Run `asset_refiner` per GLB and collect refined GLBs. |
| `asset_pipeline/jobs/blender.py` | Blender preflight, axis mapping, sizing, and GLB-to-USD. |
| `asset_pipeline/jobs/isaac.py` | STEP/STP-to-USD, physics authoring, and USD collection. |
| `asset_pipeline/workflows.py` | Compose jobs by source type without implementing backend algorithms. |
| `asset_pipeline/cli.py` | Parse user options and select one workflow. |
| `asset_pipeline/api.py` | Expose workflows as FastAPI background jobs. |
| `asset_pipeline/hunyuan_generation.py` | Tencent SDK client and standalone generation CLI. |

`pipeline_runner.py` now contains compatibility exports only. Existing callers may keep `import pipeline_runner`; new code should import the owning module directly.

## Main CLI

```text
run_asset_pipeline.py
-> asset_pipeline.cli.main
   -> runtime.configure_runtime
   -> ensure_generation_source
   -> run
```

Source dispatch:

```text
--manual-stp
-> workflows.run_stp_physics_job

--sam3d-input
-> workflows.run_sam3d_image_and_process_model_job

--sam3d-glb / --existing-glb
-> jobs.refine.run_refine_mesh_job
-> workflows.run_process_model_job

image / image URL / prompt
-> workflows.run_generate_and_process_model_job
```

## Hunyuan Generation

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
      -> SAM3 segmentation
      -> single-view or multi-view reconstruction
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
      -> temporary upload for local GLB
      -> SubmitReduceFaceJob / DescribeReduceFaceJob
      -> download Hunyuan target
      -> run_local_postprocess_worker
         -> Blender asset_refiner/blender_worker.py
```

Blender worker core path:

```text
run_pipeline
-> import_asset / join_as_whole_asset
-> clean_source_surface
-> whole_asset_retopology
-> generate_uv
-> migrate_textures
   -> nearest-surface PBR image sampling
   -> or COLOR_0 vertex color to base_color
-> export_final
-> build_qc_checks
```

## GLB Postprocess

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

## Manual CAD

```text
workflows.run_stp_physics_job
-> jobs.isaac.run_cad_to_usd_job
   -> tools/isaac/convert_cad_to_usd.py
-> for each USD:
   -> jobs.isaac.run_add_physics_job(center_origin=True)
   -> jobs.isaac.run_collect_job
```

## External Boundaries

The CLI, API, Hunyuan, `asset_refiner` orchestration code, and SAM3D all run in
`hunyuan_sam3d`. Blender and Isaac Sim still use the Python runtimes bundled with
those applications, so the orchestration layer invokes them as subprocesses instead
of importing `bpy` or `pxr`; mesh, UV, and texture work remains in Blender.
