"""Composable end-to-end asset workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from .command import LogCallback
from .jobs.blender import (
    blender_preflight,
    run_align_job,
    run_convert_job,
    run_resize_job,
)
from .jobs.hunyuan import run_generate_model_job
from .jobs.delivery import run_validate_visual_material_delivery_job
from .jobs.isaac import run_add_physics_job, run_collect_job
from .visual_materials import run_assign_visual_materials_job
from .jobs.refine import run_refine_mesh_job
from .jobs.sam3d import run_sam3d_image_job
from .manual_cad import (
    DEFAULT_MANUAL_SDF_RESOLUTION,
    run_manual_cad_job,
    run_manual_cad_workflow,
    run_stp_physics_job,
)
from .paths import collected_root_usd, list_files_by_suffix, suffixed_file_path


def run_postprocess_job(
    *,
    input_path: str,
    len_x: float,
    len_y: float,
    len_z: float,
    intermediate_output_dir: str,
    final_output_dir: str,
    axis_map: str = "X=L,Y=M,Z=S",
    unit: str = "m",
    usd_format: str = "usd",
    material_file: str | None = None,
    material: str = "plastic",
    set_mass: float | None = None,
    approx: str = "convexDecomposition",
    headless: bool = True,
    auto_visual_materials: bool = False,
    visual_material_references: Sequence[str] = (),
    visual_material_output_dir: str | None = None,
    visual_material_config: str | None = None,
    acknowledge_mvinverse_noncommercial: bool = False,
    log_cb: LogCallback = None,
) -> dict:
    """GLB path: align, resize, convert, optionally assign visuals, add physics."""
    blender_preflight(input_path, log_cb=log_cb)
    steps = [
        {
            "step": "align",
            "result": run_align_job(
                input_path=input_path, axis_map=axis_map, log_cb=log_cb
            ),
        },
        {
            "step": "resize",
            "result": run_resize_job(
                input_path=input_path,
                len_x=len_x,
                len_y=len_y,
                len_z=len_z,
                unit=unit,
                log_cb=log_cb,
            ),
        },
    ]
    convert_result = run_convert_job(
        input_path=input_path, usd_format=usd_format, log_cb=log_cb
    )
    steps.append({"step": "convert_usd", "result": convert_result})
    usd_input_root = convert_result["usd_root"]
    physics_input = convert_result.get("usd_input_path") or usd_input_root
    visual_material_result = None
    if auto_visual_materials:
        converted_usd_files = convert_result.get("usd_files")
        if not isinstance(converted_usd_files, list) or len(converted_usd_files) != 1:
            count = (
                len(converted_usd_files) if isinstance(converted_usd_files, list) else 0
            )
            raise ValueError(
                "Reference-image material assignment currently requires exactly "
                f"one converted USD per run; found {count}. Use one STEP/STP workpiece."
            )
        visual_material_result = run_assign_visual_materials_job(
            source_usd=converted_usd_files[0],
            references=visual_material_references,
            output_dir=visual_material_output_dir,
            config_path=visual_material_config,
            acknowledge_mvinverse_noncommercial=(acknowledge_mvinverse_noncommercial),
            log_cb=log_cb,
        )
        steps.append(
            {"step": "assign_visual_materials", "result": visual_material_result}
        )
        physics_input = visual_material_result["effective_usd"]

    physics_result = run_add_physics_job(
        folder=physics_input,
        out_dir=intermediate_output_dir,
        material_file=material_file,
        material=material,
        set_mass=set_mass,
        approx=approx,
        headless=headless,
        log_cb=log_cb,
    )
    physics_output_file = None
    collect_input = intermediate_output_dir
    if auto_visual_materials:
        physics_output_file = suffixed_file_path(
            physics_input, intermediate_output_dir, "_phys"
        )
        if not Path(physics_output_file).is_file():
            raise FileNotFoundError(
                "Physics job did not create expected materialized USD file: "
                f"{physics_output_file}"
            )
        physics_result["input_usd_file"] = physics_input
        physics_result["output_usd_file"] = physics_output_file
        collect_input = physics_output_file
    steps.append(
        {
            "step": "add_physics",
            "result": physics_result,
        }
    )
    collect_result = run_collect_job(
        folder=collect_input,
        out_dir=final_output_dir,
        headless=headless,
        log_cb=log_cb,
    )
    steps.append({"step": "collect_usd", "result": collect_result})
    delivery_validation_result = None
    if visual_material_result is not None and physics_output_file is not None:
        collected_root = collected_root_usd(physics_output_file, final_output_dir)
        collect_result["collected_root_usd"] = collected_root
        delivery_validation_result = run_validate_visual_material_delivery_job(
            look_usd=visual_material_result["effective_usd"],
            physics_usd=physics_output_file,
            collected_root_usd=collected_root,
            registry=visual_material_result["rendered_registry"],
            apply_report=visual_material_result["apply_report"],
            bundle_root=str(Path(collected_root).parent),
            output=str(
                Path(visual_material_result["output_dir"]) / "delivery_validation.json"
            ),
            log_cb=log_cb,
        )
        steps.append(
            {
                "step": "validate_visual_material_delivery",
                "result": delivery_validation_result,
            }
        )
    return {
        "input_path": input_path,
        "usd_input_root": usd_input_root,
        "physics_input": physics_input,
        "visual_material_output_dir": (
            visual_material_result["output_dir"]
            if visual_material_result is not None
            else None
        ),
        "physics_output_file": physics_output_file,
        "visual_material_delivery_validation": delivery_validation_result,
        "intermediate_output_dir": intermediate_output_dir,
        "final_output_dir": final_output_dir,
        "processed_glb_files": list_files_by_suffix(input_path, {".glb"}),
        "steps": steps,
    }


def run_process_model_job(
    *,
    input_path: str,
    len_x: float,
    len_y: float,
    len_z: float,
    orientation: str,
    intermediate_output_dir: str,
    final_output_dir: str,
    set_mass: float | None = None,
    material: str = "plastic",
    approx: str = "convexDecomposition",
    auto_visual_materials: bool = False,
    visual_material_references: Sequence[str] = (),
    visual_material_output_dir: str | None = None,
    visual_material_config: str | None = None,
    acknowledge_mvinverse_noncommercial: bool = False,
    log_cb: LogCallback = None,
) -> dict:
    return run_postprocess_job(
        input_path=input_path,
        len_x=len_x,
        len_y=len_y,
        len_z=len_z,
        intermediate_output_dir=intermediate_output_dir,
        final_output_dir=final_output_dir,
        axis_map=orientation,
        material=material,
        set_mass=set_mass,
        approx=approx,
        auto_visual_materials=auto_visual_materials,
        visual_material_references=visual_material_references,
        visual_material_output_dir=visual_material_output_dir,
        visual_material_config=visual_material_config,
        acknowledge_mvinverse_noncommercial=(acknowledge_mvinverse_noncommercial),
        usd_format="usd",
        headless=True,
        log_cb=log_cb,
    )


def run_sam3d_image_and_process_model_job(
    *,
    input_path: str,
    output_dir: str,
    intermediate_output_dir: str,
    final_output_dir: str,
    len_x: float,
    len_y: float,
    len_z: float,
    orientation: str,
    sam3d_mode: str = "auto",
    sam3d_prompt: str | None = None,
    sam3d_seed: int = 42,
    sam3d_steps: int = 50,
    sam3d_confidence_threshold: float = 0.5,
    set_mass: float | None = None,
    material: str = "plastic",
    approx: str = "convexDecomposition",
    refine_mesh: bool = True,
    refine_output_dir: str | None = None,
    refine_config_path: str | None = None,
    refine_temp_upload: str | None = None,
    refine_fail_on_qc_error: bool = False,
    auto_visual_materials: bool = False,
    visual_material_references: Sequence[str] = (),
    visual_material_output_dir: str | None = None,
    visual_material_config: str | None = None,
    acknowledge_mvinverse_noncommercial: bool = False,
    log_cb: LogCallback = None,
) -> dict:
    generation_result = run_sam3d_image_job(
        input_path=input_path,
        output_dir=output_dir,
        mode=sam3d_mode,
        prompt=sam3d_prompt,
        seed=sam3d_seed,
        steps=sam3d_steps,
        confidence_threshold=sam3d_confidence_threshold,
        log_cb=log_cb,
    )
    resolved_input = generation_result["postprocess_input_path"]
    refine_result = None
    if refine_mesh:
        refine_result = run_refine_mesh_job(
            input_path=resolved_input,
            output_dir=refine_output_dir,
            config_path=refine_config_path,
            temp_upload=refine_temp_upload,
            fail_on_qc_error=refine_fail_on_qc_error,
            log_cb=log_cb,
        )
        resolved_input = refine_result["postprocess_input_path"]

    process_result = run_process_model_job(
        input_path=resolved_input,
        len_x=len_x,
        len_y=len_y,
        len_z=len_z,
        orientation=orientation,
        intermediate_output_dir=intermediate_output_dir,
        final_output_dir=final_output_dir,
        set_mass=set_mass,
        material=material,
        approx=approx,
        auto_visual_materials=auto_visual_materials,
        visual_material_references=visual_material_references,
        visual_material_output_dir=visual_material_output_dir,
        visual_material_config=visual_material_config,
        acknowledge_mvinverse_noncommercial=(acknowledge_mvinverse_noncommercial),
        log_cb=log_cb,
    )
    return {
        "generation": generation_result,
        "refine_mesh": refine_result,
        "postprocess": process_result,
    }


def run_generate_and_process_model_job(
    *,
    output_dir: str,
    intermediate_output_dir: str,
    final_output_dir: str,
    len_x: float,
    len_y: float,
    len_z: float,
    orientation: str,
    set_mass: float | None = None,
    material: str = "plastic",
    approx: str = "convexDecomposition",
    input_dir: str | None = None,
    image_url: str | None = None,
    face_count: int | None = None,
    download_preview: bool = False,
    postprocess_input_path: str | None = None,
    refine_mesh: bool = True,
    refine_output_dir: str | None = None,
    refine_config_path: str | None = None,
    refine_temp_upload: str | None = None,
    refine_fail_on_qc_error: bool = False,
    auto_visual_materials: bool = False,
    visual_material_references: Sequence[str] = (),
    visual_material_output_dir: str | None = None,
    visual_material_config: str | None = None,
    acknowledge_mvinverse_noncommercial: bool = False,
    log_cb: LogCallback = None,
) -> dict:
    generation_result = run_generate_model_job(
        output_dir=output_dir,
        input_dir=input_dir,
        image_url=image_url,
        face_count=face_count,
        download_preview=download_preview,
        log_cb=log_cb,
    )
    resolved_input = postprocess_input_path or output_dir
    refine_result = None
    if refine_mesh:
        refine_result = run_refine_mesh_job(
            input_path=resolved_input,
            output_dir=refine_output_dir,
            config_path=refine_config_path,
            temp_upload=refine_temp_upload,
            fail_on_qc_error=refine_fail_on_qc_error,
            log_cb=log_cb,
        )
        resolved_input = refine_result["postprocess_input_path"]

    process_result = run_process_model_job(
        input_path=resolved_input,
        len_x=len_x,
        len_y=len_y,
        len_z=len_z,
        orientation=orientation,
        intermediate_output_dir=intermediate_output_dir,
        final_output_dir=final_output_dir,
        set_mass=set_mass,
        material=material,
        approx=approx,
        auto_visual_materials=auto_visual_materials,
        visual_material_references=visual_material_references,
        visual_material_output_dir=visual_material_output_dir,
        visual_material_config=visual_material_config,
        acknowledge_mvinverse_noncommercial=(acknowledge_mvinverse_noncommercial),
        log_cb=log_cb,
    )
    return {
        "generation": generation_result,
        "refine_mesh": refine_result,
        "postprocess": process_result,
    }


__all__ = [
    "DEFAULT_MANUAL_SDF_RESOLUTION",
    "run_generate_and_process_model_job",
    "run_manual_cad_job",
    "run_manual_cad_workflow",
    "run_postprocess_job",
    "run_process_model_job",
    "run_sam3d_image_and_process_model_job",
    "run_stp_physics_job",
]
