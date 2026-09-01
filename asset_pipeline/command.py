"""Shared subprocess helpers for external pipeline tools."""

from __future__ import annotations

import os
import re
import shlex
import subprocess
from pathlib import Path
from typing import Callable, Optional, Sequence

from .progress import format_subprocess_output_line
from .runtime import root_dir


LogCallback = Optional[Callable[[str], None]]

_PYTHON_TRACEBACK_HEADER = "Traceback (most recent call last):"
_KIT_PYTHON_STDERR_MARKER = "[py stderr]: "
_KIT_EXTENSION_EXCEPTION_MARKER = "Exception: Extension python module:"
_PYTHON_EXCEPTION_TAIL_RE = re.compile(
    r"^(?:[A-Za-z_]\w*\.)*"
    r"(?:[A-Za-z_]\w*(?:Error|Exception)|"
    r"SystemExit|KeyboardInterrupt|GeneratorExit|"
    r"StopIteration|StopAsyncIteration|Warning)"
    r"(?::(?: .*)?)?$"
)


def _python_stderr_payload(output_line: str) -> str:
    """Return Python stderr content, removing Kit's structured log prefix."""

    if _KIT_PYTHON_STDERR_MARKER in output_line:
        return output_line.split(_KIT_PYTHON_STDERR_MARKER, 1)[1]
    return output_line


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
    env_remove: Sequence[str] = (),
) -> None:
    env = os.environ.copy()
    # A conda pipeline must not silently import packages from ~/.local.  Those
    # user-site packages are outside the declared runtime and can shadow core
    # dependencies (notably typing_extensions used while importing PyTorch).
    # Keep the child environment reproducible regardless of how the invoking
    # interactive shell was configured. Explicit stage overrides below still
    # take precedence for a deliberately exceptional tool.
    env["PYTHONNOUSERSITE"] = "1"
    for name in env_remove:
        env.pop(name, None)
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
    traceback_started = False
    traceback_is_kit_stderr = False
    pending_unwrapped_traceback = False
    pending_unwrapped_lines = 0
    uncaught_python_traceback = False
    for line in process.stdout:
        output_line = line.rstrip()
        python_payload = _python_stderr_payload(output_line)
        if pending_unwrapped_traceback:
            if (
                "[carb.scripting-python.plugin]" in output_line
                and _KIT_EXTENSION_EXCEPTION_MARKER in output_line
            ):
                # Kit logs a raw traceback while probing an optional extension,
                # then explicitly catches and reports that extension failure.
                # The application may remain usable and return success.  Do not
                # mistake this handled startup diagnostic for a child-script
                # exception, but keep rejecting every other zero-exit traceback.
                pending_unwrapped_traceback = False
                pending_unwrapped_lines = 0
            elif output_line.strip():
                pending_unwrapped_lines += 1
                if pending_unwrapped_lines > 4:
                    uncaught_python_traceback = True
                    pending_unwrapped_traceback = False
        if python_payload == _PYTHON_TRACEBACK_HEADER:
            if pending_unwrapped_traceback:
                uncaught_python_traceback = True
                pending_unwrapped_traceback = False
            traceback_started = True
            traceback_is_kit_stderr = _KIT_PYTHON_STDERR_MARKER in output_line
        elif (
            traceback_started
            and python_payload == python_payload.lstrip()
            and _PYTHON_EXCEPTION_TAIL_RE.fullmatch(python_payload)
        ):
            if traceback_is_kit_stderr or _KIT_PYTHON_STDERR_MARKER in output_line:
                uncaught_python_traceback = True
            else:
                pending_unwrapped_traceback = True
                pending_unwrapped_lines = 0
            traceback_started = False
        log_message(log_cb, format_subprocess_output_line(output_line))
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"Command failed with exit code {return_code}: {pretty_cmd}")
    if uncaught_python_traceback or pending_unwrapped_traceback:
        raise RuntimeError(
            "Command reported an uncaught Python traceback despite exit code 0: "
            f"{pretty_cmd}"
        )


def append_flag(args: list[str], flag: str, enabled: bool) -> None:
    if enabled:
        args.append(flag)


def append_option(args: list[str], name: str, value: object | None) -> None:
    if value is not None:
        args.extend([name, str(value)])
