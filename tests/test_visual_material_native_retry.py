from pathlib import Path

import pytest

from asset_pipeline.visual_materials.stages.runner import _run_stage


def test_visual_stage_retries_one_native_crash_and_cleans_partial_outputs(
    tmp_path: Path,
) -> None:
    output = tmp_path / "look.usda"
    report = tmp_path / "report.json"
    render_dir = tmp_path / "renders"
    calls = 0
    messages: list[str] = []

    def runner(command, *, log_cb, env_remove, env_overrides):
        nonlocal calls
        calls += 1
        if calls == 1:
            output.write_text("partial", encoding="utf-8")
            report.write_text("partial", encoding="utf-8")
            render_dir.mkdir()
            (render_dir / "partial.png").write_bytes(b"partial")
            log_cb("[Fatal] [carb.crashreporter-breakpad.plugin] [crash]")
            log_cb("Segmentation fault (core dumped)")
            raise RuntimeError("Command failed with exit code 1")
        assert not output.exists()
        assert not report.exists()
        assert not render_dir.exists()
        output.write_text("complete", encoding="utf-8")
        report.write_text("{}", encoding="utf-8")
        render_dir.mkdir()

    _run_stage(
        "test_apply",
        [
            "/fake/isaac/python.sh",
            "--output",
            str(output),
            "--report",
            str(report),
            "--output-dir",
            str(render_dir),
        ],
        messages.append,
        command_runner=runner,
        retry_native_crash=True,
    )

    assert calls == 2
    assert output.read_text(encoding="utf-8") == "complete"
    archived = list(tmp_path.glob("renders.native_crash_attempt_01*"))
    assert len(archived) == 1
    assert (archived[0] / "partial.png").read_bytes() == b"partial"
    assert any("retrying in a clean process (2/3)" in message for message in messages)
    assert any("archived_partial=" in message for message in messages)


def test_visual_stage_does_not_retry_a_deterministic_failure(tmp_path: Path) -> None:
    calls = 0

    def runner(command, *, log_cb, env_remove, env_overrides):
        nonlocal calls
        calls += 1
        log_cb("invalid material plan")
        raise RuntimeError("Command failed with exit code 2")

    with pytest.raises(RuntimeError, match="Visual material stage failed"):
        _run_stage(
            "test_apply",
            ["/fake/isaac/python.sh", "--output", str(tmp_path / "look.usda")],
            None,
            command_runner=runner,
            retry_native_crash=True,
        )

    assert calls == 1


def test_visual_stage_retries_when_isaac_writes_a_new_crash_dump(
    tmp_path: Path,
) -> None:
    isaac_root = tmp_path / "isaac"
    executable = isaac_root / "python.sh"
    dump_dir = isaac_root / "kit" / "data" / "Kit" / "Isaac-Sim Python" / "5.0"
    dump_dir.mkdir(parents=True)
    executable.write_text("", encoding="utf-8")
    calls = 0

    def runner(command, *, log_cb, env_remove, env_overrides):
        nonlocal calls
        calls += 1
        if calls == 1:
            (dump_dir / "new-native-crash.dmp").write_bytes(b"dump")
            raise RuntimeError("Command failed with exit code 1")

    _run_stage(
        "test_render",
        [str(executable), "--output-dir", str(tmp_path / "renders")],
        None,
        command_runner=runner,
        retry_native_crash=True,
    )

    assert calls == 2


def test_visual_stage_stops_after_two_native_crash_retries() -> None:
    calls = 0
    messages: list[str] = []

    def runner(command, *, log_cb, env_remove, env_overrides):
        nonlocal calls
        calls += 1
        log_cb("Segmentation fault (core dumped)")
        raise RuntimeError("Command failed with exit code 139")

    with pytest.raises(RuntimeError, match="after 2 native-crash retries"):
        _run_stage(
            "test_render",
            ["/fake/isaac/python.sh"],
            messages.append,
            command_runner=runner,
            retry_native_crash=True,
        )

    assert calls == 3
    assert sum(" RETRY " in message for message in messages) == 2
    assert sum(" FAILED " in message for message in messages) == 3


def test_visual_stage_does_not_retry_python_traceback_with_fatal_log() -> None:
    calls = 0

    def runner(command, *, log_cb, env_remove, env_overrides):
        nonlocal calls
        calls += 1
        log_cb("[Fatal] [carb.crashreporter-breakpad.plugin] [crash]")
        log_cb("Traceback (most recent call last):")
        log_cb("ValueError: invalid deterministic MDL input")
        raise RuntimeError("Command failed with exit code 1")

    with pytest.raises(RuntimeError, match="Visual material stage failed"):
        _run_stage(
            "test_apply",
            ["/fake/isaac/python.sh"],
            None,
            command_runner=runner,
            retry_native_crash=True,
        )

    assert calls == 1


def test_visual_stage_retries_known_transient_python_runtime_state(
    tmp_path: Path,
) -> None:
    calls = 0
    output = tmp_path / "retrieval.json"
    messages: list[str] = []

    def runner(command, *, log_cb, env_remove, env_overrides):
        nonlocal calls
        calls += 1
        if calls == 1:
            output.write_text("partial", encoding="utf-8")
            log_cb("TypeError: 'dict_itemiterator' object is not callable")
            raise RuntimeError("Command failed with exit code 1")
        assert not output.exists()
        output.write_text("complete", encoding="utf-8")

    _run_stage(
        "test_retrieval",
        ["/fake/python", "--output", str(output)],
        messages.append,
        command_runner=runner,
    )

    assert calls == 2
    assert output.read_text(encoding="utf-8") == "complete"
    assert any("transient Python runtime failure" in message for message in messages)
    assert any("retrying in a clean process (2/2)" in message for message in messages)
