"""STEP/STP validation and CAD-to-USD execution boundary."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from ..command import LogCallback, append_flag, isaac_tool_path, run_command
from ..paths import cad_usd_output_path, list_files_by_suffix
from ..runtime import default_cad_usd_output_dir, isaac_python


CAD_SUFFIXES = {".stp", ".step"}


def validate_cad_input_path(
    input_path: str, *, require_single: bool = False
) -> list[str]:
    """Resolve a manual CAD input and reject non-STEP data at the boundary."""

    path = Path(input_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"STEP/STP CAD input does not exist: {path}")
    cad_files = list_files_by_suffix(str(path), CAD_SUFFIXES)
    if not cad_files:
        raise ValueError(
            "STEP/STP CAD input must be a .stp/.step file or a directory "
            f"containing one: {path}"
        )
    if require_single and len(cad_files) != 1:
        raise ValueError(
            "Reference-image material assignment requires exactly one STEP/STP "
            f"workpiece per run; found {len(cad_files)} in {path}"
        )
    return cad_files


def run_cad_to_usd_job(
    *,
    input_path: str,
    out_dir: str | None = None,
    overwrite: bool = True,
    headless: bool = True,
    converter_options: Sequence[str] = (),
    require_single: bool = False,
    log_cb: LogCallback = None,
) -> dict:
    """Convert validated STEP/STP inputs with the Omniverse CAD Converter."""

    cad_files = validate_cad_input_path(input_path, require_single=require_single)
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
            "CAD conversion did not create expected USD file(s): "
            + ", ".join(missing_usd_files)
        )
    return {
        "input_path": input_path,
        "out_dir": resolved_out_dir,
        "overwrite": overwrite,
        "converter_options": list(converter_options),
        "cad_files": cad_files,
        "usd_files": usd_files,
    }


__all__ = ["CAD_SUFFIXES", "run_cad_to_usd_job", "validate_cad_input_path"]
