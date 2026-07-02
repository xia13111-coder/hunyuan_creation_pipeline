"""Backend dispatcher for asset_refiner.

Call flow from CLI:
asset_refiner.cli.main -> run_refinement

run_refinement selects a backend from config["backend"]["name"]:
- "hunyuan_api" -> hunyuan_backend.run_hunyuan_refinement
- any other value -> Blender command -> blender_worker.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import apply_overrides, load_config, save_json
from .exceptions import BackendExecutionError
from .hunyuan_backend import is_http_url, run_hunyuan_refinement


@dataclass(frozen=True)
class RunResult:
    command: list[str]
    output_dir: Path
    report_path: Path
    log_path: Path
    config_path: Path
    report: dict[str, Any] | None


def _find_blender(config: dict[str, Any]) -> str:
    executable = config.get("backend", {}).get("blender_executable") or "blender"
    resolved = shutil.which(executable)
    if resolved:
        return resolved
    candidate = Path(executable)
    if candidate.exists():
        return str(candidate)
    raise FileNotFoundError(
        f"Blender executable not found: {executable}. "
        "Set backend.blender_executable in the config or pass --blender."
    )


def _worker_path() -> Path:
    return Path(__file__).with_name("blender_worker.py")


def build_backend_command(
    input_path: Path | str,
    output_dir: Path,
    resolved_config_path: Path,
    report_path: Path,
    config: dict[str, Any],
) -> list[str]:
    blender = _find_blender(config)
    return [
        blender,
        "--background",
        "--factory-startup",
        "--python",
        str(_worker_path()),
        "--",
        "--input",
        str(input_path),
        "--output",
        str(output_dir),
        "--config-json",
        str(resolved_config_path),
        "--report",
        str(report_path),
    ]


def run_refinement(
    input_path: str | Path,
    output_dir: str | Path,
    config_path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
    dry_run: bool = False,
) -> RunResult:
    """Resolve config and dispatch to Hunyuan API or local Blender backend."""
    config = apply_overrides(load_config(config_path), overrides)
    input_ref = str(input_path)
    if is_http_url(input_ref):
        source: Path | str = input_ref
    else:
        source_path = Path(input_path).expanduser().resolve()
        if not source_path.exists():
            raise FileNotFoundError(f"Input model does not exist: {source_path}")
        source = source_path

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)

    resolved_config_path = destination / "resolved_config.json"
    report_path = destination / "qc_report.json"
    backend_name = str(config.get("backend", {}).get("name") or "blender")
    log_path = destination / ("hunyuan_local_postprocess_blender.log" if backend_name == "hunyuan_api" else "blender.log")
    save_json(resolved_config_path, config)

    if backend_name == "hunyuan_api":
        command = [
            "hunyuan-api",
            "--input",
            str(source),
            "--output",
            str(destination),
            "--config-json",
            str(resolved_config_path),
        ]
        if dry_run:
            report = run_hunyuan_refinement(
                input_ref=str(source),
                output_dir=destination,
                report_path=report_path,
                config=config,
                dry_run=True,
            )
            return RunResult(command, destination, report_path, log_path, resolved_config_path, report)
        report = run_hunyuan_refinement(
            input_ref=str(source),
            output_dir=destination,
            report_path=report_path,
            config=config,
            dry_run=False,
        )
        return RunResult(command, destination, report_path, log_path, resolved_config_path, report)

    command = build_backend_command(source, destination, resolved_config_path, report_path, config)
    if dry_run:
        return RunResult(command, destination, report_path, log_path, resolved_config_path, None)

    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    log_path.write_text(
        (completed.stdout or "") + ("\n" if completed.stdout and completed.stderr else "") + (completed.stderr or ""),
        encoding="utf-8",
    )

    if completed.returncode != 0:
        excerpt = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
        raise BackendExecutionError(
            f"Blender backend failed with exit code {completed.returncode}. "
            f"Log: {log_path}\n{excerpt}"
        )

    if not report_path.exists():
        raise BackendExecutionError(f"Blender backend finished but did not write QC report: {report_path}")

    with report_path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    return RunResult(command, destination, report_path, log_path, resolved_config_path, report)
