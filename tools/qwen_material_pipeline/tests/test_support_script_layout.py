from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def _isolated_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["PYTHONNOUSERSITE"] = "1"
    return environment


def test_qwen35_smoke_script_resolves_package_from_any_working_directory(
    tmp_path: Path,
) -> None:
    script = PACKAGE_ROOT / "scripts" / "qwen35" / "smoke_qwen35_runtime.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        env=_isolated_environment(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "offline CUDA load/generate/unload/reload smoke" in completed.stdout


def test_result_viewer_server_resolves_sibling_assets_from_any_working_directory(
    tmp_path: Path,
) -> None:
    script = PACKAGE_ROOT / "web" / "result_viewer" / "serve.sh"
    completed = subprocess.run(
        ["bash", str(script), "--help"],
        cwd=tmp_path,
        env=_isolated_environment(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "serve.sh --port PORT" in completed.stdout
