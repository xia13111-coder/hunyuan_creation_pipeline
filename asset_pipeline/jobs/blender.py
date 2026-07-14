"""Blender axis, sizing, and USD conversion jobs."""

from __future__ import annotations

import subprocess

from ..command import (
    LogCallback,
    append_flag,
    append_option,
    blender_tool_path,
    log_message,
    run_command,
)
from ..paths import converted_usd_path, converted_usd_root, list_files_by_suffix
from ..runtime import blender_bin, root_dir


def blender_preflight(input_path: str, log_cb: LogCallback = None) -> list[str]:
    blender = blender_bin()
    glb_files = list_files_by_suffix(input_path, {".glb"})
    log_message(log_cb, f"Blender binary: {blender} | exists={blender.exists()}")
    log_message(log_cb, f"Postprocess input: {input_path} | glb_count={len(glb_files)}")
    if not blender.exists():
        raise FileNotFoundError(
            f"Blender binary not found: {blender}. Set BLENDER_BIN for this machine."
        )
    if blender.is_dir():
        raise RuntimeError(
            f"BLENDER_BIN points to a directory, not an executable: {blender}"
        )
    for glb in glb_files[:5]:
        log_message(log_cb, f"  GLB: {glb}")
    if len(glb_files) > 5:
        log_message(log_cb, f"  ... {len(glb_files) - 5} more GLB files")
    if not glb_files:
        raise FileNotFoundError(f"No .glb files found for Blender input: {input_path}")

    completed = subprocess.run(
        [str(blender), "--version"],
        cwd=str(root_dir()),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=30,
        check=False,
    )
    first_line = (completed.stdout or "").splitlines()[0] if completed.stdout else ""
    log_message(log_cb, f"Blender version check: {first_line or 'no output'}")
    if completed.returncode != 0:
        raise RuntimeError(
            f"Blender version check failed with exit code {completed.returncode}: {completed.stdout}"
        )
    return glb_files


def run_align_job(
    *,
    input_path: str,
    axis_map: str = "X=L,Y=M,Z=S",
    overwrite: bool = True,
    suffix: str | None = None,
    log_cb: LogCallback = None,
) -> dict:
    args = [
        str(blender_bin()),
        "-b",
        "-P",
        str(blender_tool_path("align_glb_axis_only.py")),
        "--",
        "--input",
        input_path,
        "--axis-map",
        axis_map,
    ]
    append_flag(args, "--overwrite", overwrite)
    append_option(args, "--suffix", suffix if not overwrite else None)
    run_command(args, log_cb=log_cb)
    return {"input_path": input_path, "axis_map": axis_map, "overwrite": overwrite}


def run_resize_job(
    *,
    input_path: str,
    len_x: float,
    len_y: float,
    len_z: float,
    unit: str = "m",
    overwrite: bool = True,
    suffix: str | None = None,
    log_cb: LogCallback = None,
) -> dict:
    args = [
        str(blender_bin()),
        "-b",
        "-P",
        str(blender_tool_path("resize_glb_xyz_and_center.py")),
        "--",
        "--input",
        input_path,
        "--len-x",
        str(len_x),
        "--len-y",
        str(len_y),
        "--len-z",
        str(len_z),
        "--unit",
        unit,
    ]
    append_flag(args, "--overwrite", overwrite)
    append_option(args, "--suffix", suffix if not overwrite else None)
    run_command(args, log_cb=log_cb)
    return {
        "input_path": input_path,
        "target_size": {"x": len_x, "y": len_y, "z": len_z, "unit": unit},
        "overwrite": overwrite,
    }


def run_convert_job(
    *,
    input_path: str,
    usd_format: str = "usd",
    overwrite: bool = True,
    suffix: str | None = None,
    visible_only: bool = False,
    log_cb: LogCallback = None,
) -> dict:
    args = [
        str(blender_bin()),
        "-b",
        "-P",
        str(blender_tool_path("convert_glb_to_usd_zup.py")),
        "--",
        "--input",
        input_path,
        "--usd-format",
        usd_format,
    ]
    append_flag(args, "--overwrite", overwrite)
    append_option(args, "--suffix", suffix if not overwrite else None)
    append_flag(args, "--visible-only", visible_only)
    run_command(args, log_cb=log_cb)
    usd_root = converted_usd_root(input_path, overwrite=overwrite, suffix=suffix)
    usd_input_path = converted_usd_path(
        input_path,
        usd_format=usd_format,
        overwrite=overwrite,
        suffix=suffix,
    )
    return {
        "input_path": input_path,
        "usd_format": usd_format,
        "overwrite": overwrite,
        "usd_root": usd_root,
        "usd_input_path": usd_input_path or usd_root,
        "usd_files": list_files_by_suffix(usd_root, {".usd", ".usda", ".usdc"}),
    }
