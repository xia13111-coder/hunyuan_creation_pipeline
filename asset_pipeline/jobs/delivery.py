"""Final Look/Physics/collected visual-material delivery validation job."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from ..command import LogCallback, log_message, run_command
from ..project_layout import ProjectLayout
from ..runtime import isaac_python, root_dir


ISOLATED_ENV_REMOVE = (
    "CONDA_PREFIX",
    "CONDA_DEFAULT_ENV",
    "CONDA_SHLVL",
    "VIRTUAL_ENV",
    "PYTHONHOME",
    "PYTHONPATH",
)


def _read_report(path: Path) -> dict[str, Any]:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Unable to read visual-material delivery report {path}: {exc}"
        ) from exc
    if not isinstance(report, dict):
        raise RuntimeError(f"Visual-material delivery report must be an object: {path}")
    return report


def _input_file(value: str, label: str) -> Path:
    try:
        path = Path(value).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise FileNotFoundError(f"{label} does not exist: {value}") from exc
    if not path.is_file():
        raise ValueError(f"{label} must be a file: {path}")
    return path


def _input_directory(value: str, label: str) -> Path:
    try:
        path = Path(value).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise FileNotFoundError(f"{label} does not exist: {value}") from exc
    if not path.is_dir():
        raise ValueError(f"{label} must be a directory: {path}")
    return path


def run_validate_visual_material_delivery_job(
    *,
    look_usd: str,
    physics_usd: str,
    collected_root_usd: str,
    registry: str,
    apply_report: str,
    bundle_root: str,
    output: str,
    log_cb: LogCallback = None,
) -> dict[str, Any]:
    """Verify exact visual coverage and MDL dependencies in the final delivery."""

    isaac = isaac_python().expanduser().resolve()
    if not isaac.is_file() or not os.access(isaac, os.X_OK):
        raise FileNotFoundError(f"Isaac Sim Python is unavailable: {isaac}")

    resolved_output = Path(output).expanduser().resolve()
    command = [
        str(isaac),
        "-m",
        "qwen_material_pipeline",
        "usd",
        "validate-delivery",
        "--look-usd",
        str(_input_file(look_usd, "Look USD")),
        "--physics-usd",
        str(_input_file(physics_usd, "Physics USD")),
        "--collected-root-usd",
        str(_input_file(collected_root_usd, "collected root USD")),
        "--registry",
        str(_input_file(registry, "rendered registry")),
        "--apply-report",
        str(_input_file(apply_report, "material apply report")),
        "--bundle-root",
        str(_input_directory(bundle_root, "collected bundle root")),
        "--output",
        str(resolved_output),
    ]
    log_message(log_cb, "Visual material stage: validate_visual_material_delivery")
    try:
        run_command(
            command,
            log_cb=log_cb,
            env_remove=ISOLATED_ENV_REMOVE,
            env_overrides={
                "PYTHONPATH": str(
                    ProjectLayout.from_root(root_dir()).material_pythonpath
                )
            },
        )
    except Exception as exc:
        raise RuntimeError(
            f"Visual material stage failed (validate_visual_material_delivery): {exc}"
        ) from exc

    if not resolved_output.is_file():
        raise RuntimeError(
            "validate_visual_material_delivery did not create expected file: "
            f"{resolved_output}"
        )
    report_path = resolved_output.resolve(strict=True)
    report = _read_report(report_path)
    if report.get("overall_pass") is not True:
        raise RuntimeError(
            "Final Physics/collected USD failed visual-material delivery validation"
        )
    return {"report": str(report_path), **report}


__all__ = ["run_validate_visual_material_delivery_job"]
