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
from .jobs.isaac import run_add_physics_job, run_cad_to_usd_job, run_collect_job
from .jobs.refine import run_refine_mesh_job
from .jobs.sam3d import run_sam3d_image_job
from .paths import list_files_by_suffix, mirrored_output_parent, suffixed_file_path


DEFAULT_MANUAL_SDF_RESOLUTION = 32


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
    log_cb: LogCallback = None,
) -> dict:
    """GLB path: align, resize, convert to USD, add physics, and collect."""
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
    steps.append(
        {
            "step": "add_physics",
            "result": run_add_physics_job(
                folder=physics_input,
                out_dir=intermediate_output_dir,
                material_file=material_file,
                material=material,
                set_mass=set_mass,
                approx=approx,
                headless=headless,
                log_cb=log_cb,
            ),
        }
    )
    steps.append(
        {
            "step": "collect_usd",
            "result": run_collect_job(
                folder=intermediate_output_dir,
                out_dir=final_output_dir,
                headless=headless,
                log_cb=log_cb,
            ),
        }
    )
    return {
        "input_path": input_path,
        "usd_input_root": usd_input_root,
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
        usd_format="usd",
        headless=True,
        log_cb=log_cb,
    )


def run_stp_physics_job(
    *,
    input_path: str,
    intermediate_output_dir: str,
    final_output_dir: str,
    cad_usd_output_dir: str | None = None,
    cad_converter_options: Sequence[str] = (),
    material_file: str | None = None,
    material: str = "plastic",
    set_mass: float | None = None,
    approx: str = "sdf",
    sdf_resolution: int = DEFAULT_MANUAL_SDF_RESOLUTION,
    headless: bool = True,
    log_cb: LogCallback = None,
) -> dict:
    """Manual CAD path that preserves the CAD USD hierarchy."""
    if sdf_resolution <= 0:
        raise ValueError("sdf_resolution must be greater than zero")

    cad_result = run_cad_to_usd_job(
        input_path=input_path,
        out_dir=cad_usd_output_dir,
        overwrite=True,
        headless=headless,
        converter_options=cad_converter_options,
        log_cb=log_cb,
    )
    steps = [{"step": "cad_to_usd", "result": cad_result}]

    physics_results = []
    collect_results = []
    for usd_file in cad_result["usd_files"]:
        physics_out_dir = mirrored_output_parent(
            usd_file,
            cad_result["out_dir"],
            intermediate_output_dir,
        )
        physics_result = run_add_physics_job(
            folder=usd_file,
            out_dir=physics_out_dir,
            material_file=material_file,
            material=material,
            set_mass=set_mass,
            approx=approx,
            sdf_resolution=sdf_resolution,
            sdf_remesh=str(approx).strip().lower() == "sdf",
            center_origin=True,
            headless=headless,
            log_cb=log_cb,
        )
        physics_usd_file = suffixed_file_path(usd_file, physics_out_dir, "_phys")
        if not Path(physics_usd_file).exists():
            raise FileNotFoundError(
                f"Physics job did not create expected USD file: {physics_usd_file}"
            )
        physics_result["input_usd_file"] = usd_file
        physics_result["output_usd_file"] = physics_usd_file
        physics_results.append(physics_result)

        collect_result = run_collect_job(
            folder=physics_usd_file,
            out_dir=final_output_dir,
            headless=headless,
            log_cb=log_cb,
        )
        collect_result["input_usd_file"] = physics_usd_file
        collect_results.append(collect_result)

    steps.append({"step": "add_physics", "result": {"jobs": physics_results}})
    steps.append({"step": "collect_usd", "result": {"jobs": collect_results}})
    return {
        "input_path": input_path,
        "cad_usd_output_dir": cad_result["out_dir"],
        "intermediate_output_dir": intermediate_output_dir,
        "final_output_dir": final_output_dir,
        "processed_cad_files": cad_result["cad_files"],
        "processed_usd_files": cad_result["usd_files"],
        "processed_glb_files": [],
        "sdf_resolution": sdf_resolution,
        "steps": steps,
    }


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
    set_mass: float | None = None,
    material: str = "plastic",
    approx: str = "convexDecomposition",
    refine_mesh: bool = True,
    refine_output_dir: str | None = None,
    refine_config_path: str | None = None,
    refine_temp_upload: str | None = None,
    refine_fail_on_qc_error: bool = False,
    log_cb: LogCallback = None,
) -> dict:
    generation_result = run_sam3d_image_job(
        input_path=input_path,
        output_dir=output_dir,
        mode=sam3d_mode,
        prompt=sam3d_prompt,
        seed=sam3d_seed,
        steps=sam3d_steps,
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
    prompt: str | None = None,
    image_url: str | None = None,
    face_count: int | None = None,
    download_preview: bool = False,
    postprocess_input_path: str | None = None,
    refine_mesh: bool = True,
    refine_output_dir: str | None = None,
    refine_config_path: str | None = None,
    refine_temp_upload: str | None = None,
    refine_fail_on_qc_error: bool = False,
    log_cb: LogCallback = None,
) -> dict:
    generation_result = run_generate_model_job(
        output_dir=output_dir,
        input_dir=input_dir,
        prompt=prompt,
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
        log_cb=log_cb,
    )
    return {
        "generation": generation_result,
        "refine_mesh": refine_result,
        "postprocess": process_result,
    }
