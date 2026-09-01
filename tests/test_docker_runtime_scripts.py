from __future__ import annotations

import os
import resource
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ISAAC_SMOKE = ROOT / "docker" / "isaac-smoke.sh"


def _fake_isaac(tmp_path: Path, *, failures: int) -> tuple[Path, Path]:
    executable = tmp_path / "fake-isaac.sh"
    counter = tmp_path / "counter"
    executable.write_text(
        "#!/usr/bin/env bash\n"
        "set -eu\n"
        f"counter={counter!s}\n"
        "count=0\n"
        "if [ -f \"$counter\" ]; then count=$(cat \"$counter\"); fi\n"
        "count=$((count + 1))\n"
        "printf '%s\\n' \"$count\" > \"$counter\"\n"
        f"if [ \"$count\" -le {failures} ]; then exit 139; fi\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    return executable, counter


def test_isaac_smoke_retries_then_succeeds(tmp_path: Path) -> None:
    executable, counter = _fake_isaac(tmp_path, failures=2)
    environment = os.environ.copy()
    environment.update(
        ISAAC_PYTHON=str(executable),
        ISAAC_SMOKE_MAX_ATTEMPTS="3",
    )

    completed = subprocess.run(
        ["bash", str(ISAAC_SMOKE)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 0, completed.stderr
    assert counter.read_text(encoding="utf-8").strip() == "3"
    assert "retrying in a clean process" in completed.stderr


def test_isaac_smoke_returns_failure_after_limit(tmp_path: Path) -> None:
    executable, counter = _fake_isaac(tmp_path, failures=3)
    environment = os.environ.copy()
    environment.update(
        ISAAC_PYTHON=str(executable),
        ISAAC_SMOKE_MAX_ATTEMPTS="2",
    )

    completed = subprocess.run(
        ["bash", str(ISAAC_SMOKE)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 139
    assert counter.read_text(encoding="utf-8").strip() == "2"
    assert "failed after 2 attempts" in completed.stderr


def test_container_preflight_rejects_low_open_file_limit(monkeypatch) -> None:
    from asset_pipeline import docker_preflight

    monkeypatch.setenv("PIPELINE_MIN_NOFILE", "65536")
    monkeypatch.setattr(resource, "getrlimit", lambda _kind: (1024, 524288))

    try:
        docker_preflight._require_open_file_limit()
    except docker_preflight.DockerPreflightError as exc:
        assert "open-file soft limit is 1024" in str(exc)
    else:
        raise AssertionError("low open-file limit was accepted")


def test_container_preflight_accepts_sufficient_open_file_limit(monkeypatch) -> None:
    from asset_pipeline import docker_preflight

    monkeypatch.setenv("PIPELINE_MIN_NOFILE", "65536")
    monkeypatch.setattr(resource, "getrlimit", lambda _kind: (65536, 65536))

    docker_preflight._require_open_file_limit()
