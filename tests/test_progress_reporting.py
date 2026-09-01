from __future__ import annotations

import sys
from pathlib import Path

import pytest

from asset_pipeline import manual_cad
from asset_pipeline.command import run_command
from asset_pipeline.progress import (
    PROGRESS_PREFIX,
    PROGRESS_SCHEMA_VERSION,
    ProgressProtocolError,
    encode_progress_event,
    format_progress_event,
    format_subprocess_output_line,
    make_progress_event,
    parse_progress_line,
    validate_progress_event,
)


def _event(**overrides: object) -> dict[str, object]:
    event: dict[str, object] = {
        "schema_version": PROGRESS_SCHEMA_VERSION,
        "scope": "pipeline",
        "stage": "mesh",
        "state": "update",
        "current": 2,
        "total": 4,
        "unit": "asset",
        "detail": "halfway",
    }
    event.update(overrides)
    return event


def test_progress_protocol_round_trip_and_ascii_bar() -> None:
    marked_line = encode_progress_event(_event())

    assert marked_line.startswith(PROGRESS_PREFIX)
    assert parse_progress_line(marked_line) == _event()
    assert format_progress_event(_event(), bar_width=8) == (
        "[PROGRESS] [####----] 50.0% "
        "pipeline/mesh UPDATE halfway (asset 2/4)"
    )


def test_indeterminate_progress_uses_no_fabricated_bar_or_count() -> None:
    event = make_progress_event(
        scope="visual",
        stage="model_load",
        state="start",
        current=None,
        total=None,
        unit=None,
        detail="Loading model",
    )

    assert format_progress_event(event) == (
        "[PROGRESS] visual/model_load START Loading model"
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"state": "completed"},
        {"current": True},
        {"current": 0.5},
        {"current": -1},
        {"current": 5},
        {"total": 0},
        {"unit": ""},
        {"current": None},
        {"detail": None},
        {"schema_version": "asset-pipeline-progress/v2"},
    ],
)
def test_progress_protocol_rejects_invalid_fields(
    overrides: dict[str, object],
) -> None:
    with pytest.raises(ProgressProtocolError):
        validate_progress_event(_event(**overrides))


def test_progress_protocol_rejects_missing_and_unknown_fields() -> None:
    missing = _event()
    missing.pop("detail")
    extra = {**_event(), "elapsed": 1.0}

    with pytest.raises(ProgressProtocolError, match="missing"):
        validate_progress_event(missing)
    with pytest.raises(ProgressProtocolError, match="unknown"):
        validate_progress_event(extra)


def test_subprocess_line_formatter_preserves_logs_and_warns_safely() -> None:
    assert format_subprocess_output_line("ordinary child log") == "ordinary child log"

    malformed = PROGRESS_PREFIX + "{not-json"
    rendered = format_subprocess_output_line(malformed)
    assert rendered.startswith("[WARNING] malformed asset progress line")
    assert malformed in rendered


def test_subprocess_line_formatter_translates_kit_wrapped_progress() -> None:
    progress_line = encode_progress_event(_event())
    wrapped = (
        "2026-07-29T10:27:17Z [7,612ms] [Error] "
        "[omni.kit.app._impl] [py stderr]: "
        f"{progress_line}"
    )

    rendered = format_subprocess_output_line(wrapped)

    assert rendered.startswith("[PROGRESS] [##########----------] 50.0%")
    assert rendered.endswith("(asset 2/4)")
    assert "[Error]" not in rendered
    assert PROGRESS_PREFIX not in rendered


def test_subprocess_line_formatter_warns_for_malformed_kit_progress() -> None:
    wrapped = (
        "2026-07-29T10:27:17Z [7,612ms] [Error] "
        "[omni.kit.app._impl] [py stderr]: "
        f"{PROGRESS_PREFIX}{{not-json"
    )

    rendered = format_subprocess_output_line(wrapped)

    assert rendered.startswith("[WARNING] malformed asset progress line")
    assert wrapped in rendered


@pytest.mark.parametrize(
    "line",
    [
        (
            "2026-07-29T10:27:17Z [7,612ms] [Error] "
            "[omni.kit.app._impl] [py stderr]: ordinary Kit error"
        ),
        (
            "untrusted prefix [Error] [omni.kit.app._impl] [py stderr]: "
            f"{PROGRESS_PREFIX}{{}}"
        ),
        (
            "junk 2026-07-29T10:27:17Z [7,612ms] [Error] "
            "[omni.kit.app._impl] [py stderr]: "
            f"{PROGRESS_PREFIX}{{}}"
        ),
        (
            "2026-07-29T10:27:17Z [7,612ms] [Error] "
            "[other.extension] [py stderr]: "
            f"{PROGRESS_PREFIX}{{}}"
        ),
        (
            "2026-07-29T10:27:17Z [7,612ms] [Error] "
            "[omni.kit.app._impl] [py stderr]: junk "
            f"{PROGRESS_PREFIX}{{}}"
        ),
    ],
)
def test_subprocess_line_formatter_preserves_untrusted_kit_like_lines(
    line: str,
) -> None:
    assert format_subprocess_output_line(line) == line


def test_run_command_translates_only_protocol_lines() -> None:
    progress_line = encode_progress_event(_event())
    child_code = (
        "print('ordinary child log')\n"
        f"print({progress_line!r})\n"
        "print('finished')\n"
    )
    messages: list[str] = []

    run_command([sys.executable, "-c", child_code], log_cb=messages.append)

    assert messages[1] == "ordinary child log"
    assert messages[2].startswith("[PROGRESS] [##########----------] 50.0%")
    assert messages[2].endswith("(asset 2/4)")
    assert messages[3] == "finished"


def test_run_command_disables_unmanaged_user_site_packages() -> None:
    messages: list[str] = []

    run_command(
        [
            sys.executable,
            "-c",
            "import os, site; print(os.environ['PYTHONNOUSERSITE']); "
            "print(site.ENABLE_USER_SITE)",
        ],
        log_cb=messages.append,
    )

    assert messages[1:] == ["1", "False"]


def test_run_command_translates_kit_wrapped_progress_without_false_error() -> None:
    progress_line = encode_progress_event(_event())
    kit_prefix = (
        "2026-07-29T10:27:17Z [7,612ms] [Error] "
        "[omni.kit.app._impl] [py stderr]: "
    )
    child_code = (
        f"print({(kit_prefix + progress_line)!r})\n"
        f"print({(kit_prefix + 'ordinary Kit error')!r})\n"
    )
    messages: list[str] = []

    run_command([sys.executable, "-c", child_code], log_cb=messages.append)

    assert messages[1].startswith("[PROGRESS] [##########----------] 50.0%")
    assert messages[2] == kit_prefix + "ordinary Kit error"


def test_run_command_rejects_uncaught_traceback_with_zero_exit_code() -> None:
    child_code = (
        "prefix = '2026-07-29 [Error] [omni.kit.app._impl] [py stderr]: '\n"
        "print(prefix + 'Traceback (most recent call last):')\n"
        "print(prefix + '  File \"apply.py\", line 7, in <module>')\n"
        "print('    raise ValueError(\"invalid plan\")')\n"
        "print(prefix + 'ValueError: invalid plan')\n"
    )

    with pytest.raises(
        RuntimeError,
        match="uncaught Python traceback despite exit code 0",
    ):
        run_command([sys.executable, "-c", child_code])


def test_run_command_rejects_unwrapped_uncaught_traceback_with_zero_exit_code() -> None:
    child_code = (
        "print('Traceback (most recent call last):')\n"
        "print('  File \"worker.py\", line 3, in <module>')\n"
        "print('RuntimeError: worker failed')\n"
    )

    with pytest.raises(
        RuntimeError,
        match="uncaught Python traceback despite exit code 0",
    ):
        run_command([sys.executable, "-c", child_code])


def test_run_command_allows_kit_handled_optional_extension_traceback() -> None:
    child_code = (
        "print('Traceback (most recent call last):')\n"
        "print('  File \"extension.py\", line 9, in startup')\n"
        "print('ImportError: optional extension dependency is unavailable')\n"
        "print('2026-08-31 [Error] [carb.scripting-python.plugin] "
        "Exception: Extension python module: optional.extension failed')\n"
        "print('[1.000s] app ready')\n"
    )

    run_command([sys.executable, "-c", child_code])


def test_run_command_ignores_previous_crash_warning_and_incomplete_traceback() -> None:
    child_code = (
        "print('[Warning] Previous crash detected; report may contain: "
        "Traceback (most recent call last):')\n"
        "print('Traceback (most recent call last):')\n"
        "print('[Warning] Previous crash dump is unavailable')\n"
    )

    run_command([sys.executable, "-c", child_code])


def test_run_command_preserves_nonzero_exit_code_error_for_traceback() -> None:
    child_code = (
        "import sys\n"
        "print('Traceback (most recent call last):')\n"
        "print('RuntimeError: child failed')\n"
        "sys.exit(7)\n"
    )

    with pytest.raises(RuntimeError, match="failed with exit code 7"):
        run_command([sys.executable, "-c", child_code])


def _patch_manual_cad_jobs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    visual: bool,
    fail_physics: bool = False,
) -> None:
    cad_root = tmp_path / "cad"
    converted = cad_root / "asset" / "asset.usd"
    converted.parent.mkdir(parents=True)
    converted.write_text("cad", encoding="utf-8")
    look = tmp_path / "visual" / "asset_look.usda"

    monkeypatch.setattr(
        manual_cad,
        "run_cad_to_usd_job",
        lambda **_kwargs: {
            "out_dir": str(cad_root),
            "cad_files": ["asset.stp"],
            "usd_files": [str(converted)],
        },
    )

    def physics(**kwargs: object) -> dict[str, object]:
        if fail_physics:
            raise RuntimeError("physics failed")
        output = Path(str(kwargs["out_dir"])) / "asset_phys.usd"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("physics", encoding="utf-8")
        return {}

    def assign_visual(**_kwargs: object) -> dict[str, object]:
        look.parent.mkdir(parents=True, exist_ok=True)
        look.write_text("look", encoding="utf-8")
        return {
            "effective_usd": str(look),
            "output_dir": str(look.parent),
            "rendered_registry": "registry.json",
            "apply_report": "apply.json",
            "visual_quality_gate_status": "PASS",
        }

    def collect(**kwargs: object) -> dict[str, object]:
        source = Path(str(kwargs["folder"]))
        output = Path(str(kwargs["out_dir"])) / source.stem / source.name
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("collected", encoding="utf-8")
        return {}

    monkeypatch.setattr(manual_cad, "run_add_physics_job", physics)
    monkeypatch.setattr(manual_cad, "run_collect_job", collect)
    if visual:
        monkeypatch.setattr(
            manual_cad, "run_assign_visual_materials_job", assign_visual
        )
        monkeypatch.setattr(
            manual_cad,
            "run_validate_visual_material_delivery_job",
            lambda **_kwargs: {"overall_pass": True},
        )
        monkeypatch.setattr(
            manual_cad,
            "run_final_visual_acceptance_job",
            lambda **_kwargs: {
                "state": "COMPLETED",
                "completion_allowed": True,
            },
        )


def _progress_stage_states(messages: list[str]) -> list[tuple[str, str]]:
    result: list[tuple[str, str]] = []
    for message in messages:
        if not message.startswith("[PROGRESS]"):
            continue
        tokens = message.split()
        scope_stage_index = next(
            index
            for index, token in enumerate(tokens)
            if token.startswith("manual_cad/")
        )
        result.append(
            (
                tokens[scope_stage_index].split("/", 1)[1],
                tokens[scope_stage_index + 1].lower(),
            )
        )
    return result


def test_manual_cad_visual_path_reports_six_real_stages(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_manual_cad_jobs(monkeypatch, tmp_path, visual=True)
    messages: list[str] = []

    manual_cad.run_manual_cad_workflow(
        input_path="asset.stp",
        intermediate_output_dir=str(tmp_path / "intermediate"),
        final_output_dir=str(tmp_path / "final"),
        auto_visual_materials=True,
        log_cb=messages.append,
    )

    stages = [
        "cad",
        "physics",
        "visual",
        "collect",
        "delivery_validation",
        "final_acceptance",
    ]
    assert _progress_stage_states(messages) == [
        item
        for stage in stages
        for item in ((stage, "start"), (stage, "complete"))
    ]
    progress_messages = [
        message for message in messages if message.startswith("[PROGRESS]")
    ]
    assert progress_messages[0].endswith("(stage 0/6)")
    assert progress_messages[-1].endswith("(stage 6/6)")


def test_manual_cad_nonvisual_path_reports_only_three_real_stages(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_manual_cad_jobs(monkeypatch, tmp_path, visual=False)
    messages: list[str] = []

    manual_cad.run_manual_cad_workflow(
        input_path="asset.stp",
        intermediate_output_dir=str(tmp_path / "intermediate"),
        final_output_dir=str(tmp_path / "final"),
        log_cb=messages.append,
    )

    assert _progress_stage_states(messages) == [
        ("cad", "start"),
        ("cad", "complete"),
        ("physics", "start"),
        ("physics", "complete"),
        ("collect", "start"),
        ("collect", "complete"),
    ]
    progress_messages = [
        message for message in messages if message.startswith("[PROGRESS]")
    ]
    assert all("/3" in message for message in progress_messages)
    assert not any("visual" in message for message in progress_messages)


def test_manual_cad_reports_failed_stage_without_false_completion(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_manual_cad_jobs(
        monkeypatch,
        tmp_path,
        visual=False,
        fail_physics=True,
    )
    messages: list[str] = []

    with pytest.raises(RuntimeError, match="physics failed"):
        manual_cad.run_manual_cad_workflow(
            input_path="asset.stp",
            intermediate_output_dir=str(tmp_path / "intermediate"),
            final_output_dir=str(tmp_path / "final"),
            log_cb=messages.append,
        )

    assert _progress_stage_states(messages)[-2:] == [
        ("physics", "start"),
        ("physics", "failed"),
    ]
