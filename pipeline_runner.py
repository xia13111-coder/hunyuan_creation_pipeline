"""Backward-compatible imports for the modular :mod:`asset_pipeline` package.

New code should import the owning module directly, for example
``asset_pipeline.jobs.refine`` or ``asset_pipeline.workflows``.
"""

# Compatibility re-exports are intentionally imported for downstream callers.
# ruff: noqa: F401

from asset_pipeline import (
    UNIFIED_CONDA_ENV,
    active_environment_name,
    available_approx_types,
    available_materials,
    blender_bin,
    configure_runtime,
    default_cad_usd_output_dir,
    default_refine_config_path,
    default_refine_output_dir,
    default_refine_temp_upload,
    isaac_python,
    materials_file,
    prepare_sam3d_input,
    project_root,
    require_unified_environment,
    root_dir,
    run_add_physics_job,
    run_align_job,
    run_cad_to_usd_job,
    run_collect_job,
    run_convert_job,
    run_generate_and_process_model_job,
    run_generate_model_job,
    run_hunyuan_job,
    run_postprocess_job,
    run_process_model_job,
    run_refine_mesh_job,
    run_resize_job,
    run_sam3d_image_and_process_model_job,
    run_sam3d_image_job,
    run_stp_physics_job,
    runtime_summary,
    sam3d_python,
    select_sam3d_glb,
)


__all__ = [name for name in globals() if not name.startswith("_")]
