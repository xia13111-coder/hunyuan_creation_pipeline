"""Shared subprocess helpers for external pipeline tools."""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path
from typing import Callable, Optional, Sequence

from .runtime import root_dir


LogCallback = Optional[Callable[[str], None]]


def log_message(log_cb: LogCallback, message: str) -> None:
    if log_cb:
        log_cb(message)


def script_path(script_name: str) -> Path:
    path = root_dir() / script_name
    if not path.exists():
        raise FileNotFoundError(f"Script not found: {path}")
    return path


def blender_tool_path(script_name: str) -> Path:
    return script_path(str(Path("tools/blender") / script_name))


def isaac_tool_path(script_name: str) -> Path:
    return script_path(str(Path("tools/isaac") / script_name))


def sam3d_tool_path(script_name: str) -> Path:
    return script_path(str(Path("tools/sam3d") / script_name))


def run_command(
    cmd: Sequence[str],
    *,
    log_cb: LogCallback = None,
    env_overrides: dict[str, str] | None = None,
) -> None:
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)
    pretty_cmd = " ".join(shlex.quote(part) for part in cmd)
    log_message(log_cb, f"$ {pretty_cmd}")

    executable = Path(cmd[0])
    if not executable.exists():
        raise FileNotFoundError(
            f"Executable not found: {executable}. "
            "Set BLENDER_BIN, ISAAC_PYTHON, or ISAACSIM_ROOT for this machine."
        )
    if executable.is_dir():
        raise RuntimeError(f"Executable path is a directory: {executable}")

    try:
        process = subprocess.Popen(
            list(cmd),
            cwd=str(root_dir()),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except OSError as exc:
        raise RuntimeError(f"Failed to start command: {pretty_cmd} | {exc}") from exc

    assert process.stdout is not None
    for line in process.stdout:
        log_message(log_cb, line.rstrip())
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"Command failed with exit code {return_code}: {pretty_cmd}")


def append_flag(args: list[str], flag: str, enabled: bool) -> None:
    if enabled:
        args.append(flag)


def append_option(args: list[str], name: str, value: object | None) -> None:
    if value is not None:
        args.extend([name, str(value)])
