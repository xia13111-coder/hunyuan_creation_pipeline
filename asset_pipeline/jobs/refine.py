"""Hunyuan ReduceFace plus local Blender refinement job."""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

from ..command import LogCallback, log_message, run_command
from ..paths import asset_run_name, list_files_by_suffix, unique_path
from ..runtime import (
    blender_bin,
    default_refine_config_path,
    default_refine_output_dir,
    default_refine_temp_upload,
    root_dir,
)


def _refined_glb_from_output(refine_dir: Path) -> Path:
    report_path = refine_dir / "qc_report.json"
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            for item in report.get("exports", []):
                export_path = item.get("path") if isinstance(item, dict) else None
                if not export_path:
                    continue
                candidate = Path(export_path)
                if not candidate.is_absolute():
                    candidate = refine_dir / candidate
                if candidate.suffix.lower() == ".glb" and candidate.exists():
                    return candidate
        except (OSError, ValueError, TypeError):
            pass
    candidate = refine_dir / "refined_asset.glb"
    if candidate.exists():
        return candidate
    for glb_file in list_files_by_suffix(str(refine_dir), {".glb"}):
        path = Path(glb_file)
        if "intermediate" not in path.parts:
            return path
    raise FileNotFoundError(
        f"Refine mesh finished but no final GLB was found in: {refine_dir}"
    )


def run_refine_mesh_job(
    *,
    input_path: str,
    output_dir: str | None = None,
    config_path: str | None = None,
    temp_upload: str | None = None,
    fail_on_qc_error: bool = False,
    log_cb: LogCallback = None,
) -> dict:
    glb_files = list_files_by_suffix(input_path, {".glb"})
    if not glb_files:
        raise FileNotFoundError(
            f"No .glb files found for refine mesh input: {input_path}"
        )

    config = Path(
        config_path
        or os.getenv("REFINE_MESH_CONFIG", str(default_refine_config_path()))
    ).expanduser()
    if not config.is_absolute():
        config = root_dir() / config
    if not config.exists():
        raise FileNotFoundError(f"Refine mesh config not found: {config}")

    refine_root = (
        Path(output_dir or default_refine_output_dir(input_path)).expanduser().resolve()
    )
    refine_root.mkdir(parents=True, exist_ok=True)
    final_glbs_dir = refine_root / "postprocess_glbs"
    if final_glbs_dir.exists():
        final_glbs_dir = refine_root / f"postprocess_glbs_{int(time.time())}"
    final_glbs_dir.mkdir(parents=True, exist_ok=False)

    provider = default_refine_temp_upload() if temp_upload is None else temp_upload
    provider = provider.strip() if isinstance(provider, str) else provider
    if provider and provider.lower() in {"none", "false", "off", "0"}:
        provider = None

    log_message(log_cb, f"Refine mesh input: {input_path} | glb_count={len(glb_files)}")
    log_message(log_cb, f"Refine mesh output: {refine_root}")
    log_message(log_cb, f"Refine mesh config: {config}")

    refined_items = []
    for glb_file in glb_files:
        asset_name = asset_run_name(glb_file, input_path)
        asset_output_dir = refine_root / asset_name
        args = [
            sys.executable,
            "-m",
            "asset_refiner",
            "--input",
            glb_file,
            "--output",
            str(asset_output_dir),
            "--config",
            str(config),
            "--blender",
            str(blender_bin()),
            "--hunyuan-local-postprocess",
        ]
        if provider:
            args.extend(["--hunyuan-temp-upload", provider])
        if fail_on_qc_error:
            args.append("--fail-on-qc-error")

        log_message(log_cb, f"Refining GLB: {glb_file}")
        run_command(args, log_cb=log_cb)
        final_glb = _refined_glb_from_output(asset_output_dir)
        copied_glb = unique_path(final_glbs_dir / f"{asset_name}_refined.glb")
        shutil.copy2(final_glb, copied_glb)
        refined_items.append(
            {
                "source_glb": glb_file,
                "refine_output_dir": str(asset_output_dir),
                "qc_report": str(asset_output_dir / "qc_report.json"),
                "refined_glb": str(final_glb),
                "postprocess_glb": str(copied_glb),
            }
        )
        log_message(log_cb, f"Refined GLB ready for postprocess: {copied_glb}")

    return {
        "input_path": input_path,
        "output_dir": str(refine_root),
        "config_path": str(config),
        "temp_upload": provider,
        "postprocess_input_path": str(final_glbs_dir),
        "refined_files": [item["postprocess_glb"] for item in refined_items],
        "items": refined_items,
    }
