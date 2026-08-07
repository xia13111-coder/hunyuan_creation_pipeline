"""Isaac Sim Physics authoring and USD collection jobs."""

from __future__ import annotations

from ..command import (
    LogCallback,
    append_flag,
    append_option,
    isaac_tool_path,
    run_command,
)
from ..runtime import isaac_python, materials_file

# Compatibility re-exports. New code imports STEP/STP jobs from ``jobs.cad``.
from .cad import CAD_SUFFIXES, run_cad_to_usd_job, validate_cad_input_path


def run_add_physics_job(
    *,
    folder: str,
    out_dir: str,
    material_file: str | None = None,
    material: str = "plastic",
    set_mass: float | None = None,
    approx: str = "convexDecomposition",
    sdf_resolution: int | None = None,
    sdf_remesh: bool = False,
    center_origin: bool = False,
    headless: bool = True,
    log_cb: LogCallback = None,
) -> dict:
    resolved_material_file = material_file or str(materials_file())
    args = [
        str(isaac_python()),
        str(isaac_tool_path("add_physics.py")),
        "--folder",
        folder,
        "--material-file",
        resolved_material_file,
        "--out-dir",
        out_dir,
    ]
    append_flag(args, "--headless", headless)
    append_option(args, "--material", material)
    append_option(args, "--set-mass", set_mass)
    append_option(args, "--approx", approx)
    append_option(args, "--sdf-res", sdf_resolution)
    append_flag(args, "--sdf-remesh", sdf_remesh)
    append_flag(args, "--center-origin", center_origin)
    run_command(args, log_cb=log_cb)
    return {
        "folder": folder,
        "out_dir": out_dir,
        "material_file": resolved_material_file,
        "material": material,
        "set_mass": set_mass,
        "approx": approx,
        "sdf_resolution": sdf_resolution,
        "sdf_remesh": sdf_remesh,
        "center_origin": center_origin,
    }


def run_collect_job(
    *,
    folder: str,
    out_dir: str,
    headless: bool = True,
    log_cb: LogCallback = None,
) -> dict:
    args = [
        str(isaac_python()),
        str(isaac_tool_path("collect_usd_flat.py")),
        "--folder",
        folder,
        "--out-dir",
        out_dir,
    ]
    append_flag(args, "--headless", headless)
    run_command(args, log_cb=log_cb)
    return {"folder": folder, "out_dir": out_dir}


__all__ = [
    "CAD_SUFFIXES",
    "run_add_physics_job",
    "run_cad_to_usd_job",
    "run_collect_job",
    "validate_cad_input_path",
]
