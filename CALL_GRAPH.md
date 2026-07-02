# Function Call Graph

This file documents the main function relationships after the pipeline cleanup.
It focuses on production paths and omits small helpers such as path formatting,
JSON writes, and simple argument appenders.

## Main CLI Path

```text
run_asset_pipeline.main
-> configure_runtime
-> ensure_generation_source
-> run
```

`run_asset_pipeline.run` branches by source:

```text
--manual-glb
-> pipeline_runner.run_glb_physics_job

--existing-glb
-> pipeline_runner.run_refine_mesh_job
-> pipeline_runner.run_process_model_job

image / prompt / image-url
-> pipeline_runner.run_generate_and_process_model_job
```

## Pipeline Runner

Full generated-asset path:

```text
pipeline_runner.run_generate_and_process_model_job
-> run_generate_model_job
   -> run_hunyuan_job
      -> _run_command: python hunyuan_to3d_batch.py
-> run_refine_mesh_job
   -> _run_command: python -m asset_refiner
-> run_process_model_job
   -> run_postprocess_job
```

Hunyuan-style postprocess:

```text
pipeline_runner.run_process_model_job
-> run_postprocess_job
   -> _log_blender_preflight
   -> run_align_job
      -> _run_command: blender -P align_glb_axis_only.py
   -> run_resize_job
      -> _run_command: blender -P resize_glb_xyz_and_center.py
   -> run_convert_job
      -> _run_command: blender -P convert_glb_to_usd_zup.py
   -> run_add_physics_job
      -> _run_command: isaac python add_physics.py
   -> run_collect_job
      -> _run_command: isaac python collect_usd_flat.py
```

Manual/general GLB path:

```text
pipeline_runner.run_glb_physics_job
-> _log_blender_preflight
-> optional run_align_job
-> optional run_resize_job
-> run_convert_job
-> run_add_physics_job
-> run_collect_job
```

## Asset Refiner

`pipeline_runner.run_refine_mesh_job` starts:

```text
python -m asset_refiner
```

That expands to:

```text
asset_refiner.__main__
-> asset_refiner.cli.main
   -> _build_parser
   -> load_default_env_files
   -> _overrides_from_args
   -> runner.run_refinement
```

Backend dispatch:

```text
asset_refiner.runner.run_refinement
-> load_config + apply_overrides
-> save resolved_config.json
-> if backend.name == "hunyuan_api":
      hunyuan_backend.run_hunyuan_refinement
   else:
      build_backend_command
      -> subprocess: blender -P blender_worker.py
```

## Hunyuan ReduceFace Backend

Production refine path:

```text
hunyuan_backend.run_hunyuan_refinement
-> resolve_api_upload_input_ref
-> resolve_remote_input
   or planned_temp_upload_input
-> TencentCloudApiClient.from_config
-> if temporary upload is needed:
      upload_to_temporary_host
-> submit_reduce_face
   -> run_job
      -> submit_job_with_retry
      -> TencentCloudApiClient.call SubmitReduceFaceJob
      -> TencentCloudApiClient.call DescribeReduceFaceJob until DONE
-> choose_result_file
-> download_url
-> prepare_downloaded_model
-> run_local_postprocess_worker
   -> build_local_postprocess_config
   -> subprocess: blender -P asset_refiner/blender_worker.py
-> merge Hunyuan summary into qc_report.json
```

## Blender Worker

Local mesh and texture path:

```text
blender_worker.main
-> parse_args
-> run_pipeline
   -> reset_scene
   -> import_asset
   -> join_as_whole_asset
   -> mesh_metrics
   -> clean_source_surface
   -> whole_asset_retopology
   -> generate_uv
   -> migrate_textures
   -> projection_metrics
   -> mesh_metrics
   -> export_final
   -> build_qc_checks
```

Texture migration:

```text
blender_worker.migrate_textures
-> create_refined_material
-> pbr_texture_spec
-> transfer_texture_nearest_surface
   -> build_material_texture_sources
      -> material_texture_source
         -> linked_image_source_from_socket
         -> find_named_image_source_for_material
   -> BVHTree.FromObject(source)
   -> nearest_source_texture_sample
      -> source BVH nearest face
      -> source face UV interpolation
      -> sample_image_nearest
   -> dilate_texture_pixels
-> connect_texture_to_refined_material
-> save_image
```

## External Script Boundary

The orchestration layer calls these scripts as separate processes:

```text
hunyuan_to3d_batch.py          Tencent Hunyuan generation client
asset_refiner/blender_worker.py Blender-side refine and texture migration
align_glb_axis_only.py          Blender GLB axis alignment
resize_glb_xyz_and_center.py    Blender GLB resize and centering
convert_glb_to_usd_zup.py       Blender GLB to USD conversion
add_physics.py                  Isaac Sim physics authoring
collect_usd_flat.py             Isaac Sim USD collection
```
