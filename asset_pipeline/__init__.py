"""Public and compatibility exports for the modular asset pipeline.

New hand-authored CAD integrations should call
``manual_cad.run_manual_cad_workflow``. Low-level job exports remain available
for diagnostics and older callers but do not replace the STEP/STP production
entry contract.
"""

from .jobs.blender import (
    blender_preflight,
    run_align_job,
    run_convert_job,
    run_resize_job,
)
from .jobs.cad import run_cad_to_usd_job
from .jobs.delivery import run_validate_visual_material_delivery_job
from .jobs.hunyuan import run_generate_model_job, run_hunyuan_job
from .jobs.isaac import run_add_physics_job, run_collect_job
from .visual_materials import (
    DEFAULT_CONFIG_PATH as DEFAULT_VISUAL_MATERIAL_CONFIG_PATH,
    load_visual_material_config,
    parse_visual_references,
    run_assign_visual_materials_job,
    run_final_visual_acceptance_job,
)
from .jobs.refine import run_refine_mesh_job
from .jobs.sam3d import prepare_sam3d_input, run_sam3d_image_job, select_sam3d_glb
from .manual_cad import (
    run_manual_cad_job,
    run_manual_cad_workflow,
    run_stp_physics_job,
)
from .runtime import (
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
    project_root,
    require_unified_environment,
    root_dir,
    runtime_summary,
    sam3d_python,
)
from .workflows import (
    run_generate_and_process_model_job,
    run_postprocess_job,
    run_process_model_job,
    run_sam3d_image_and_process_model_job,
)


__all__ = [
    "UNIFIED_CONDA_ENV",
    "active_environment_name",
    "available_approx_types",
    "available_materials",
    "blender_bin",
    "blender_preflight",
    "configure_runtime",
    "default_cad_usd_output_dir",
    "default_refine_config_path",
    "default_refine_output_dir",
    "default_refine_temp_upload",
    "DEFAULT_VISUAL_MATERIAL_CONFIG_PATH",
    "isaac_python",
    "materials_file",
    "prepare_sam3d_input",
    "load_visual_material_config",
    "parse_visual_references",
    "project_root",
    "require_unified_environment",
    "root_dir",
    "run_add_physics_job",
    "run_assign_visual_materials_job",
    "run_final_visual_acceptance_job",
    "run_align_job",
    "run_cad_to_usd_job",
    "run_collect_job",
    "run_convert_job",
    "run_generate_and_process_model_job",
    "run_generate_model_job",
    "run_hunyuan_job",
    "run_manual_cad_job",
    "run_manual_cad_workflow",
    "run_postprocess_job",
    "run_process_model_job",
    "run_refine_mesh_job",
    "run_resize_job",
    "run_validate_visual_material_delivery_job",
    "run_sam3d_image_and_process_model_job",
    "run_sam3d_image_job",
    "run_stp_physics_job",
    "runtime_summary",
    "sam3d_python",
    "select_sam3d_glb",
]
