from __future__ import annotations

from pathlib import Path

import pytest

from asset_pipeline.visual_materials import orchestrator
from asset_pipeline.visual_materials.stages import runner as stage_runner


def _progress_messages(messages: list[str]) -> list[str]:
    return [message for message in messages if message.startswith("[PROGRESS]")]


def test_run_stage_reports_start_and_complete_with_real_elapsed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    messages: list[str] = []
    times = iter((10.0, 12.375))
    monkeypatch.setattr(stage_runner, "monotonic", lambda: next(times))

    stage_runner._run_stage(
        "render_locked",
        ["/fake/isaac/python.sh"],
        messages.append,
        command_runner=lambda *_args, **_kwargs: None,
    )

    progress = _progress_messages(messages)
    assert len(progress) == 2
    assert progress[0] == (
        "[PROGRESS] visual_materials/render_locked START elapsed=0.000s"
    )
    assert progress[1] == (
        "[PROGRESS] visual_materials/render_locked COMPLETE elapsed=2.375s"
    )


def test_run_stage_restores_full_affinity_for_native_child(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[list[str]] = []

    def runner(command, **_kwargs) -> None:
        captured.append(command)

    monkeypatch.setattr(stage_runner, "run_command", runner)
    monkeypatch.setattr(stage_runner, "_TASKSET_EXECUTABLE", "/usr/bin/taskset")
    monkeypatch.setattr(
        stage_runner,
        "_VISUAL_CONTROL_CHILD_CPU_AFFINITY",
        (0, 2, 4),
    )

    stage_runner._run_stage(
        "native_child",
        ["/fake/isaac/python.sh", "--help"],
        None,
        command_runner=runner,
    )

    assert captured == [
        [
            "/usr/bin/taskset",
            "-c",
            "0,2,4",
            "/fake/isaac/python.sh",
            "--help",
        ]
    ]


def test_run_stage_does_not_taskset_local_python_model_child(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[list[str]] = []

    def runner(command, **_kwargs) -> None:
        captured.append(command)

    monkeypatch.setattr(stage_runner, "run_command", runner)
    monkeypatch.setattr(stage_runner, "_TASKSET_EXECUTABLE", "/usr/bin/taskset")
    monkeypatch.setattr(stage_runner, "_VISUAL_CONTROL_CHILD_CPU_AFFINITY", (0, 2))

    stage_runner._run_stage(
        "local_model",
        ["/env/bin/python", "-m", "qwen_material_pipeline"],
        None,
        command_runner=runner,
    )

    assert captured == [["/env/bin/python", "-m", "qwen_material_pipeline"]]


def test_stable_native_child_affinity_keeps_only_hyperthreaded_cores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakePath:
        def __init__(self, path: str) -> None:
            self.path = path

        def read_text(self, *, encoding: str) -> str:
            cpu = int(self.path.rsplit("cpu", 1)[1].split("/")[0])
            return {0: "0-1\n", 1: "0-1\n", 2: "2\n", 3: "3\n"}[cpu]

    monkeypatch.setattr(stage_runner, "Path", FakePath)
    assert stage_runner._stable_native_child_cpu_affinity((0, 1, 2, 3)) == (0, 1)


def test_run_stage_reports_failed_retry_and_cumulative_elapsed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    messages: list[str] = []
    calls = 0
    times = iter((20.0, 21.25, 23.5))
    monkeypatch.setattr(stage_runner, "monotonic", lambda: next(times))

    def runner(_command, *, log_cb, **_kwargs) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            log_cb("Segmentation fault (core dumped)")
            raise RuntimeError("native crash")

    stage_runner._run_stage(
        "apply_locked",
        ["/fake/isaac/python.sh"],
        messages.append,
        command_runner=runner,
        retry_native_crash=True,
    )

    progress = _progress_messages(messages)
    assert [
        next(
            state
            for state in ("START", "FAILED", "RETRY", "COMPLETE")
            if f" {state} " in message
        )
        for message in progress
    ] == ["START", "FAILED", "RETRY", "COMPLETE"]
    assert "elapsed=1.250s" in progress[1]
    assert "elapsed=1.250s" in progress[2]
    assert "elapsed=3.500s" in progress[3]


def test_run_stage_reports_terminal_failure_without_false_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    messages: list[str] = []
    times = iter((30.0, 30.625))
    monkeypatch.setattr(stage_runner, "monotonic", lambda: next(times))

    def runner(*_args, **_kwargs) -> None:
        raise RuntimeError("invalid material plan")

    with pytest.raises(RuntimeError, match="Visual material stage failed"):
        stage_runner._run_stage(
            "apply_invalid_plan",
            ["/fake/isaac/python.sh"],
            messages.append,
            command_runner=runner,
            retry_native_crash=True,
        )

    progress = _progress_messages(messages)
    assert progress == [
        "[PROGRESS] visual_materials/apply_invalid_plan START elapsed=0.000s",
        (
            "[PROGRESS] visual_materials/apply_invalid_plan FAILED "
            "elapsed=0.625s attempt=1"
        ),
    ]


def test_run_stage_missing_required_output_is_not_reported_complete(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    messages: list[str] = []
    times = iter((40.0, 40.25))
    monkeypatch.setattr(stage_runner, "monotonic", lambda: next(times))

    with pytest.raises(RuntimeError, match="did not create expected file"):
        stage_runner._run_stage(
            "apply_usd",
            ["/fake/isaac/python.sh"],
            messages.append,
            command_runner=lambda *_args, **_kwargs: None,
            required_files=(tmp_path / "look.usda", tmp_path / "apply.json"),
        )

    assert _progress_messages(messages) == [
        "[PROGRESS] visual_materials/apply_usd START elapsed=0.000s",
        "[PROGRESS] visual_materials/apply_usd FAILED elapsed=0.250s attempt=1",
    ]


def test_exact_mdl_candidate_progress_spans_the_global_tournament() -> None:
    messages: list[str] = []

    orchestrator._log_exact_mdl_candidate_progress(
        messages.append,
        state="start",
        group_index=1,
        group_total=4,
        candidate_index=1,
        candidate_total=32,
        candidate_id="candidate_green_01",
        global_current=0,
        global_total=128,
    )
    orchestrator._log_exact_mdl_candidate_progress(
        messages.append,
        state="complete",
        group_index=1,
        group_total=4,
        candidate_index=1,
        candidate_total=32,
        candidate_id="candidate_green_01",
        global_current=1,
        global_total=128,
        cache_status="CACHE_HIT",
    )

    assert " 0.0% " in messages[0]
    assert "group 1/4 candidate 1/32 id=candidate_green_01" in messages[0]
    assert messages[0].endswith("(candidate 0/128)")
    assert "visual_materials.exact_mdl_tournament/candidate COMPLETE" in messages[1]
    assert "cache=CACHE_HIT" in messages[1]
    assert messages[1].endswith("(candidate 1/128)")


def test_exact_mdl_group_progress_spans_all_groups() -> None:
    messages: list[str] = []

    orchestrator._log_exact_mdl_group_progress(
        messages.append,
        state="start",
        current=0,
        total=4,
        group_id="G06",
    )
    orchestrator._log_exact_mdl_group_progress(
        messages.append,
        state="complete",
        current=4,
        total=4,
        group_id="G05",
    )

    assert " 0.0% " in messages[0]
    assert messages[0].endswith("(group 0/4)")
    assert " 100.0% " in messages[1]
    assert messages[1].endswith("(group 4/4)")
