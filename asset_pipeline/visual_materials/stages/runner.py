"""Subprocess execution, retry classification, affinity and stage progress."""

from __future__ import annotations

import functools
import os
import shutil
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from time import monotonic
from typing import Any

from ...command import LogCallback, log_message, run_command
from ...paths import unique_path
from ...progress import emit_progress
from ...project_layout import ProjectLayout
from ...runtime import root_dir
from ..contracts import ISOLATED_ENV_REMOVE


CommandRunner = Callable[..., None]


_VISUAL_CONTROL_CHILD_CPU_AFFINITY: tuple[int, ...] | None = None
_TASKSET_EXECUTABLE = shutil.which("taskset")


def _stable_native_child_cpu_affinity(
    original: Sequence[int],
) -> tuple[int, ...]:
    """Keep native children off hybrid-CPU efficiency cores when detectable.

    Linux exposes P-core hyperthread pairs through ``thread_siblings_list``
    while Intel E-cores have a singleton sibling list.  Mixing both classes
    has produced sporadic CPython/PyTorch import corruption on this host.  On
    non-hybrid or opaque systems the function returns the full original set.
    """

    allowed = tuple(sorted(set(int(cpu) for cpu in original)))
    if len(allowed) < 2:
        return allowed

    def parse_cpu_list(value: str) -> set[int]:
        output: set[int] = set()
        for token in value.strip().split(","):
            if not token:
                continue
            if "-" in token:
                start_text, end_text = token.split("-", 1)
                output.update(range(int(start_text), int(end_text) + 1))
            else:
                output.add(int(token))
        return output

    performant: list[int] = []
    for cpu in allowed:
        siblings_path = Path(
            f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list"
        )
        try:
            siblings = parse_cpu_list(siblings_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return allowed
        if len(siblings & set(allowed)) > 1:
            performant.append(cpu)
    stable = tuple(sorted(set(performant)))
    # Never turn an ordinary non-hybrid topology into an unexplained partial
    # affinity mask. Only use the subset when both core classes are present.
    return stable if stable and len(stable) < len(allowed) else allowed


def _is_isaac_native_command(command: Sequence[str]) -> bool:
    """Return whether a stage needs parallel Isaac/Kit CPU affinity.

    Local CPython model processes inherit the controller's fixed core.  This
    avoids a reproducible ``taskset``-mediated Python/PyTorch instability on
    the host, while Kit/Isaac rendering still receives the high-performance
    CPU subset it needs.
    """

    if not command:
        return False
    executable = str(command[0]).lower()
    return executable.endswith(".sh") and "isaac" in executable


def _isolated_pythonpath(command: Sequence[str]) -> str:
    """Return the source roots needed by one isolated model interpreter."""

    paths = [
        str(ProjectLayout.from_root(root_dir()).material_pythonpath),
    ]
    entityseg_python = os.getenv("ENTITYSEG_PYTHON", "").strip()
    detectron2_root = os.getenv("ENTITYSEG_DETECTRON2_ROOT", "").strip()
    if command and entityseg_python and detectron2_root:
        command_python = Path(command[0]).expanduser().resolve()
        configured_python = Path(entityseg_python).expanduser().resolve()
        if command_python == configured_python:
            paths.append(str(Path(detectron2_root).expanduser().absolute()))
    return os.pathsep.join(paths)


def _visual_control_cpu_stability_guard(
    function: Callable[..., Any],
) -> Callable[..., Any]:
    """Pin only the long-lived Python controller, not its compute children.

    The visual tournament performs many pure-Python plan transformations over
    a long period.  On the current heterogeneous host, unrestricted migration
    of that controller has produced non-deterministic interpreter corruption.
    Isaac commands receive the detected high-performance subset by
    :func:`_run_stage`; local Python model processes inherit the controller's
    fixed core to avoid taskset-mediated PyTorch instability.
    """

    @functools.wraps(function)
    def guarded(*args: Any, **kwargs: Any) -> Any:
        global _VISUAL_CONTROL_CHILD_CPU_AFFINITY
        if os.environ.get("ASSET_PIPELINE_DISABLE_CPU_STABILITY_GUARD") == "1":
            return function(*args, **kwargs)
        get_affinity = getattr(os, "sched_getaffinity", None)
        set_affinity = getattr(os, "sched_setaffinity", None)
        if not callable(get_affinity) or not callable(set_affinity):
            return function(*args, **kwargs)
        previous_child_affinity = _VISUAL_CONTROL_CHILD_CPU_AFFINITY
        try:
            original = tuple(sorted(get_affinity(0)))
            if not original:
                return function(*args, **kwargs)
            _VISUAL_CONTROL_CHILD_CPU_AFFINITY = _stable_native_child_cpu_affinity(
                original
            )
            set_affinity(0, {original[0]})
        except OSError:
            _VISUAL_CONTROL_CHILD_CPU_AFFINITY = previous_child_affinity
            return function(*args, **kwargs)
        try:
            return function(*args, **kwargs)
        finally:
            try:
                set_affinity(0, set(original))
            finally:
                _VISUAL_CONTROL_CHILD_CPU_AFFINITY = previous_child_affinity

    return guarded


NATIVE_CRASH_MAX_ATTEMPTS = 3
TRANSIENT_PYTHON_RUNTIME_MAX_ATTEMPTS = 2
ENTITYSEG_OOM_MAX_ATTEMPTS = 2
ENTITYSEG_OOM_RETRY_SHORT_EDGE = 640
_NATIVE_CRASH_OUTPUT_MARKERS = (
    "segmentation fault",
    "core dumped",
    "sigsegv",
    "sigabrt",
    "signal 11",
    "signal 6",
    "pure virtual method called",
    "terminate called without an active exception",
)
_TRANSIENT_PYTHON_RUNTIME_MARKERS = (
    # These impossible standard-library states have been observed only after
    # a long-lived controller has launched several native CUDA subprocesses.
    # A clean Python process immediately succeeds; retrying the identical
    # deterministic material evidence is therefore safe and avoids making an
    # unattended run depend on a manual restart.
    "TypeError: 'dict_itemiterator' object is not callable",
    "TypeError: 'list_iterator' object is not callable",
    "SystemError: unknown opcode",
)
_CUDA_OOM_OUTPUT_MARKERS = (
    "torch.OutOfMemoryError: CUDA out of memory",
    "CUDA out of memory. Tried to allocate",
)


def _is_entityseg_region_command(command: Sequence[str]) -> bool:
    return "qwen_material_pipeline.segmentation.entityseg_regions" in command


def _set_command_option(command: list[str], option: str, value: str) -> None:
    if option in command:
        value_index = command.index(option) + 1
        if value_index >= len(command):
            raise RuntimeError(f"missing value for command option {option}")
        command[value_index] = value
    else:
        command.extend((option, value))

def _isaac_crash_dump_snapshot(command: Sequence[str]) -> dict[str, tuple[int, int]]:
    """Return a cheap identity snapshot of dumps owned by this Isaac install."""

    if not command:
        return {}
    executable = Path(command[0]).expanduser().resolve()
    dump_root = executable.parent / "kit" / "data" / "Kit"
    if not dump_root.is_dir():
        return {}

    snapshot: dict[str, tuple[int, int]] = {}
    try:
        dump_paths = dump_root.rglob("*.dmp")
        for dump_path in dump_paths:
            try:
                stat = dump_path.stat()
            except OSError:
                continue
            snapshot[str(dump_path.resolve())] = (stat.st_mtime_ns, stat.st_size)
    except OSError:
        # Dump discovery is supporting evidence only. Explicit signal/fatal
        # output still detects a native crash if Kit's data directory is
        # temporarily unreadable.
        return {}
    return snapshot


def _native_crash_evidence(
    *,
    output: str,
    error: Exception,
    dumps_before: Mapping[str, tuple[int, int]],
    dumps_after: Mapping[str, tuple[int, int]],
) -> tuple[str, ...]:
    """Classify only strong native-process evidence, failing Python errors closed."""

    lowered_output = output.lower()
    lowered_error = str(error).lower()
    combined = f"{lowered_output}\n{lowered_error}"
    if "traceback (most recent call last):" in lowered_output:
        return ()

    evidence: list[str] = [
        f"output_marker={marker}"
        for marker in _NATIVE_CRASH_OUTPUT_MARKERS
        if marker in combined
    ]
    if (
        "[fatal]" in lowered_output
        and "crashreporter" in lowered_output
        and "[previous crash]" not in lowered_output.rsplit("[fatal]", 1)[-1]
    ):
        evidence.append("output_marker=kit_crashreporter_fatal")
    for exit_code in (-11, 139, -6, 134):
        if f"exit code {exit_code}" in lowered_error:
            evidence.append(f"process_exit_code={exit_code}")

    new_or_changed_dumps = sorted(
        path
        for path, identity in dumps_after.items()
        if dumps_before.get(path) != identity
    )
    evidence.extend(f"new_crash_dump={path}" for path in new_or_changed_dumps)
    return tuple(dict.fromkeys(evidence))


def _transient_python_runtime_evidence(output: str) -> tuple[str, ...]:
    """Recognize a narrow class of retryable fresh-process Python failures."""

    return tuple(
        f"output_marker={marker}"
        for marker in _TRANSIENT_PYTHON_RUNTIME_MARKERS
        if marker in output
    )


def _clean_native_retry_outputs(
    command: Sequence[str],
    *,
    attempt: int,
) -> tuple[str, ...]:
    """Remove partial files and archive a partial render directory before retry."""

    actions: list[str] = []
    for flag in ("--output", "--report"):
        if flag not in command:
            continue
        value_index = command.index(flag) + 1
        if value_index >= len(command):
            continue
        retry_output = Path(command[value_index])
        if retry_output.is_file() or retry_output.is_symlink():
            retry_output.unlink()
            actions.append(f"removed_partial={retry_output}")

    if "--output-dir" in command:
        value_index = command.index("--output-dir") + 1
        if value_index < len(command):
            output_dir = Path(command[value_index])
            if output_dir.exists() or output_dir.is_symlink():
                archive = unique_path(
                    output_dir.with_name(
                        f"{output_dir.name}.native_crash_attempt_{attempt:02d}"
                    )
                )
                output_dir.rename(archive)
                actions.append(f"archived_partial={archive}")
    return tuple(actions)


def _run_stage(
    name: str,
    command: list[str],
    log_cb: LogCallback,
    *,
    command_runner: CommandRunner,
    retry_native_crash: bool = False,
    required_files: Sequence[Path] = (),
) -> None:
    started_at = monotonic()
    emit_progress(
        log_cb,
        scope="visual_materials",
        stage=name,
        state="start",
        current=None,
        total=None,
        unit=None,
        detail="elapsed=0.000s",
    )
    log_message(log_cb, f"Visual material stage: {name}")
    output_tail: deque[str] = deque(maxlen=256)
    stage_command = list(command)

    def capture_log(message: str) -> None:
        output_tail.append(message)
        log_message(log_cb, message)

    def run_once() -> None:
        effective_command = stage_command
        if (
            command_runner is run_command
            and _VISUAL_CONTROL_CHILD_CPU_AFFINITY
            and len(_VISUAL_CONTROL_CHILD_CPU_AFFINITY) > 1
            and _TASKSET_EXECUTABLE is not None
            and _is_isaac_native_command(stage_command)
        ):
            # The controller is pinned for interpreter stability. Give only
            # native Isaac/Kit workloads the parallel stable subset; local
            # CPython workloads remain on the inherited fixed controller core.
            effective_command = [
                _TASKSET_EXECUTABLE,
                "-c",
                ",".join(
                    str(cpu) for cpu in _VISUAL_CONTROL_CHILD_CPU_AFFINITY
                ),
                *stage_command,
            ]
        command_runner(
            effective_command,
            log_cb=capture_log,
            env_remove=ISOLATED_ENV_REMOVE,
            env_overrides={
                "PYTHONPATH": _isolated_pythonpath(stage_command)
            },
        )
        missing_outputs = [str(path) for path in required_files if not path.is_file()]
        if missing_outputs:
            raise RuntimeError(
                f"{name} did not create expected file(s): "
                + ", ".join(missing_outputs)
            )

    attempt = 1
    while True:
        dumps_before = (
            _isaac_crash_dump_snapshot(command) if retry_native_crash else {}
        )
        try:
            run_once()
            break
        except Exception as exc:
            elapsed_seconds = monotonic() - started_at
            emit_progress(
                log_cb,
                scope="visual_materials",
                stage=name,
                state="failed",
                current=None,
                total=None,
                unit=None,
                detail=f"elapsed={elapsed_seconds:.3f}s attempt={attempt}",
            )
            dumps_after = (
                _isaac_crash_dump_snapshot(command) if retry_native_crash else {}
            )
            evidence = _native_crash_evidence(
                output="\n".join(output_tail),
                error=exc,
                dumps_before=dumps_before,
                dumps_after=dumps_after,
            )
            runtime_evidence = _transient_python_runtime_evidence(
                "\n".join(output_tail)
            )
            entityseg_oom_evidence = tuple(
                f"output_marker={marker}"
                for marker in _CUDA_OOM_OUTPUT_MARKERS
                if marker in "\n".join(output_tail)
            )
            native_retry = retry_native_crash and bool(evidence)
            runtime_retry = bool(runtime_evidence)
            entityseg_oom_retry = _is_entityseg_region_command(
                stage_command
            ) and bool(entityseg_oom_evidence)
            if not native_retry and not runtime_retry and not entityseg_oom_retry:
                raise RuntimeError(
                    f"Visual material stage failed ({name}): {exc}"
                ) from exc
            if native_retry:
                maximum_attempts = NATIVE_CRASH_MAX_ATTEMPTS
                retry_kind = "native process crash"
                retry_failure_label = "native-crash"
                retry_evidence = evidence
            elif entityseg_oom_retry:
                maximum_attempts = ENTITYSEG_OOM_MAX_ATTEMPTS
                retry_kind = "recoverable EntitySeg CUDA OOM"
                retry_failure_label = "EntitySeg CUDA OOM"
                retry_evidence = entityseg_oom_evidence
            else:
                maximum_attempts = TRANSIENT_PYTHON_RUNTIME_MAX_ATTEMPTS
                retry_kind = "transient Python runtime failure"
                retry_failure_label = "transient Python runtime"
                retry_evidence = runtime_evidence
            if attempt >= maximum_attempts:
                raise RuntimeError(
                    "Visual material stage failed after "
                    f"{attempt - 1} {retry_failure_label} retries ({name}): {exc}; "
                    f"evidence={'; '.join(retry_evidence)}"
                ) from exc

            cleanup_actions = _clean_native_retry_outputs(
                stage_command,
                attempt=attempt,
            )
            if entityseg_oom_retry:
                _set_command_option(
                    stage_command,
                    "--inference-short-edge",
                    str(ENTITYSEG_OOM_RETRY_SHORT_EDGE),
                )
                cleanup_actions = (
                    *cleanup_actions,
                    "entityseg_inference_short_edge="
                    f"{ENTITYSEG_OOM_RETRY_SHORT_EDGE}",
                )
            next_attempt = attempt + 1
            log_message(
                log_cb,
                f"Visual material stage {name} detected a {retry_kind}; "
                f"retrying in a clean process ({next_attempt}/"
                f"{maximum_attempts}). "
                f"evidence={'; '.join(retry_evidence)}"
                + (
                    f" cleanup={'; '.join(cleanup_actions)}"
                    if cleanup_actions
                    else ""
                ),
            )
            emit_progress(
                log_cb,
                scope="visual_materials",
                stage=name,
                state="retry",
                current=None,
                total=None,
                unit=None,
                detail=(
                    f"elapsed={elapsed_seconds:.3f}s "
                    f"next_attempt={next_attempt} "
                    f"max_attempts={maximum_attempts}"
                ),
            )
            output_tail.clear()
            attempt = next_attempt
    elapsed_seconds = monotonic() - started_at
    emit_progress(
        log_cb,
        scope="visual_materials",
        stage=name,
        state="complete",
        current=None,
        total=None,
        unit=None,
        detail=f"elapsed={elapsed_seconds:.3f}s",
    )
