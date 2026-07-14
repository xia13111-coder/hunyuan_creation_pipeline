"""Isaac Sim CAD conversion, physics authoring, and USD collection jobs."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from ..command import (
    LogCallback,
    append_flag,
    append_option,
    isaac_tool_path,
    run_command,
)
from ..paths import cad_usd_output_path, list_files_by_suffix
from ..runtime import default_cad_usd_output_dir, isaac_python, materials_file


def run_cad_to_usd_job(
    *,
    input_path: str,
    out_dir: str | None = None,
    overwrite: bool = True,
    headless: bool = True,
    converter_options: Sequence[str] = (),
    log_cb: LogCallback = None,
) -> dict:
    cad_files = list_files_by_suffix(input_path, {".stp", ".step"})
    if not cad_files:
        raise FileNotFoundError(
            f"No .stp/.step files found for CAD input: {input_path}"
        )

    resolved_out_dir = out_dir or default_cad_usd_output_dir(input_path)
    args = [
        str(isaac_python()),
        str(isaac_tool_path("convert_cad_to_usd.py")),
        "--input",
        input_path,
        "--out-dir",
        resolved_out_dir,
    ]
    append_flag(args, "--headless", headless)
    append_flag(args, "--overwrite", overwrite)
    for item in converter_options:
        args.extend(["--option", item])

    run_command(args, log_cb=log_cb)
    usd_files = sorted(
        cad_usd_output_path(path, input_path, resolved_out_dir) for path in cad_files
    )
    missing_usd_files = [path for path in usd_files if not Path(path).exists()]
    if missing_usd_files:
        raise FileNotFoundError(
            f"CAD conversion did not create expected USD file(s): {', '.join(missing_usd_files)}"
        )
    return {
        "input_path": input_path,
        "out_dir": resolved_out_dir,
        "overwrite": overwrite,
        "converter_options": list(converter_options),
        "cad_files": cad_files,
        "usd_files": usd_files,
    }


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
