from __future__ import annotations

import json

import pytest

from qwen_material_pipeline.core.progress import (
    PROGRESS_PREFIX,
    emit_progress,
    format_progress_event,
    parse_progress_line,
    progress_event,
    report_progress,
)


def test_emit_progress_writes_one_flushed_stderr_protocol_line(
    capsys: pytest.CaptureFixture[str],
) -> None:
    event = emit_progress(
        scope="qwen_material_pipeline",
        stage="primary_mapping",
        state="update",
        current=2,
        total=3,
        unit="batches",
        detail="B02 completed",
    )

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.endswith("\n")
    assert captured.err.count("\n") == 1
    assert captured.err.startswith(PROGRESS_PREFIX)
    assert parse_progress_line(captured.err) == event
    assert json.loads(captured.err[len(PROGRESS_PREFIX) :]) == event


def test_uncounted_stage_keeps_null_counts_and_string_detail() -> None:
    event = progress_event(
        scope="qwen_material_pipeline",
        stage="mvinverse",
        state="complete",
        detail="MVInverse completed",
    )

    assert event["current"] is None
    assert event["total"] is None
    assert event["unit"] is None
    assert event["detail"] == "MVInverse completed"


def test_counted_progress_requires_positive_total_and_valid_state() -> None:
    with pytest.raises(ValueError, match="total > 0"):
        progress_event(
            scope="qwen_material_pipeline",
            stage="mapping",
            state="start",
            current=0,
            total=0,
            unit="batches",
        )
    with pytest.raises(ValueError, match="state"):
        progress_event(
            scope="qwen_material_pipeline",
            stage="mapping",
            state="progress",
        )
    with pytest.raises(ValueError, match="detail"):
        progress_event(
            scope="qwen_material_pipeline",
            stage="mapping",
            state="complete",
            detail=None,  # type: ignore[arg-type]
        )


def test_optional_callback_receives_event_without_writing_console(
    capsys: pytest.CaptureFixture[str],
) -> None:
    events: list[dict[str, object]] = []
    event = report_progress(
        events.append,
        scope="qwen_material_pipeline",
        stage="spatial_render_decode",
        state="start",
        current=0,
        total=2,
        unit="views",
    )

    assert events == [event]
    assert capsys.readouterr() == ("", "")


def test_parser_rejects_non_protocol_and_malformed_protocol_lines() -> None:
    assert parse_progress_line("ordinary diagnostic\n") is None
    assert parse_progress_line(PROGRESS_PREFIX + "{}\n") is None
    assert (
        parse_progress_line(
            PROGRESS_PREFIX + '{"schema_version":"asset-pipeline-progress/v1",'
            '"scope":"x","stage":"y","state":"update",'
            '"current":2,"total":1,"unit":"parts","detail":""}\n'
        )
        is None
    )
    assert (
        parse_progress_line(
            PROGRESS_PREFIX + '{"schema_version":"asset-pipeline-progress/v1",'
            '"scope":"x","stage":"y","state":"complete",'
            '"current":1,"total":1,"unit":"parts","detail":"",'
            '"unexpected":true}\n'
        )
        is None
    )


def test_parser_accepts_isaac_kit_stderr_wrapper() -> None:
    event = progress_event(
        scope="qwen_material_pipeline",
        stage="face_topology",
        state="update",
        current=12,
        total=596,
        unit="parts",
        detail="face topology part P0012 completed",
    )
    wrapped = (
        "2026-07-31T06:54:00Z [12,345ms] [Error] "
        "[omni.kit.app._impl] [py stderr]: "
        + format_progress_event(event)
        + "\n"
    )

    assert parse_progress_line(wrapped) == event
