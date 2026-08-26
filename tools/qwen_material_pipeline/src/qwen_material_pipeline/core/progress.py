"""Machine-readable progress events for long-running asset pipeline stages."""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Callable, Mapping
from typing import Any, TextIO


PROGRESS_PREFIX = "@@ASSET_PROGRESS "
PROGRESS_SCHEMA_VERSION = "asset-pipeline-progress/v1"
SCHEMA_VERSION = PROGRESS_SCHEMA_VERSION
PROGRESS_STATES = frozenset({"start", "update", "complete", "retry", "failed"})
_KIT_PROGRESS_WRAPPER_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T[^\r\n ]+"
    r"(?: \[[^\r\n]*\])*"
    r" \[Error\] \[omni\.kit\.app\._impl\] "
    r"\[py stderr\]: (?P<payload>[^\r\n]*)$"
)
_EVENT_FIELDS = (
    "schema_version",
    "scope",
    "stage",
    "state",
    "current",
    "total",
    "unit",
    "detail",
)

ProgressEvent = dict[str, Any]
ProgressCallback = Callable[[ProgressEvent], None]


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def validate_progress_event(event: Mapping[str, Any]) -> ProgressEvent:
    """Validate and normalize one v1 event without retaining extra fields."""

    if not isinstance(event, Mapping):
        raise TypeError("progress event must be an object")
    missing = [field for field in _EVENT_FIELDS if field not in event]
    if missing:
        raise ValueError(f"progress event is missing fields: {missing}")
    extra = sorted(set(event) - set(_EVENT_FIELDS))
    if extra:
        raise ValueError(f"progress event has unknown fields: {extra}")
    if event["schema_version"] != PROGRESS_SCHEMA_VERSION:
        raise ValueError("progress event schema_version is unsupported")

    scope = _nonempty_string(event["scope"], "scope")
    stage = _nonempty_string(event["stage"], "stage")
    state = event["state"]
    if state not in PROGRESS_STATES:
        raise ValueError(f"state must be one of {sorted(PROGRESS_STATES)}")

    current = event["current"]
    total = event["total"]
    unit = event["unit"]
    if current is None or total is None:
        if current is not None or total is not None or unit is not None:
            raise ValueError(
                "uncounted progress requires current, total, and unit to be null"
            )
    else:
        if (
            isinstance(current, bool)
            or not isinstance(current, int)
            or isinstance(total, bool)
            or not isinstance(total, int)
        ):
            raise ValueError("counted progress requires integer current and total")
        if total <= 0 or current < 0 or current > total:
            raise ValueError(
                "counted progress requires total > 0 and 0 <= current <= total"
            )
        unit = _nonempty_string(unit, "unit")

    detail = event["detail"]
    if not isinstance(detail, str):
        raise ValueError("detail must be a string")

    return {
        "schema_version": PROGRESS_SCHEMA_VERSION,
        "scope": scope,
        "stage": stage,
        "state": state,
        "current": current,
        "total": total,
        "unit": unit,
        "detail": detail,
    }


def progress_event(
    *,
    scope: str,
    stage: str,
    state: str,
    current: int | None = None,
    total: int | None = None,
    unit: str | None = None,
    detail: str = "",
) -> ProgressEvent:
    """Construct one validated progress event."""

    return validate_progress_event(
        {
            "schema_version": PROGRESS_SCHEMA_VERSION,
            "scope": scope,
            "stage": stage,
            "state": state,
            "current": current,
            "total": total,
            "unit": unit,
            "detail": detail,
        }
    )


def format_progress_event(event: Mapping[str, Any]) -> str:
    """Serialize one validated event as a complete protocol line."""

    normalized = validate_progress_event(event)
    return PROGRESS_PREFIX + json.dumps(
        normalized,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )


def parse_progress_line(line: str) -> ProgressEvent | None:
    """Return a validated protocol event, or ``None`` for any unsafe line."""

    if not isinstance(line, str):
        return None
    candidate = line.rstrip("\r\n")
    wrapper = _KIT_PROGRESS_WRAPPER_RE.fullmatch(candidate)
    if wrapper is not None:
        candidate = wrapper.group("payload")
    if not candidate.startswith(PROGRESS_PREFIX):
        return None
    try:
        value = json.loads(candidate[len(PROGRESS_PREFIX) :])
        return validate_progress_event(value)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def emit_progress_event(
    event: Mapping[str, Any],
    *,
    stream: TextIO | None = None,
) -> ProgressEvent:
    """Write one existing event to stderr (or an explicit stream) and flush."""

    normalized = validate_progress_event(event)
    destination = sys.stderr if stream is None else stream
    destination.write(format_progress_event(normalized) + "\n")
    destination.flush()
    return normalized


def emit_progress(
    *,
    scope: str,
    stage: str,
    state: str,
    current: int | None = None,
    total: int | None = None,
    unit: str | None = None,
    detail: str = "",
    stream: TextIO | None = None,
) -> ProgressEvent:
    """Construct and immediately emit one progress event."""

    event = progress_event(
        scope=scope,
        stage=stage,
        state=state,
        current=current,
        total=total,
        unit=unit,
        detail=detail,
    )
    return emit_progress_event(event, stream=stream)


def report_progress(
    callback: ProgressCallback | None,
    *,
    scope: str,
    stage: str,
    state: str,
    current: int | None = None,
    total: int | None = None,
    unit: str | None = None,
    detail: str = "",
) -> ProgressEvent:
    """Build an event and deliver it when an optional callback is configured."""

    event = progress_event(
        scope=scope,
        stage=stage,
        state=state,
        current=current,
        total=total,
        unit=unit,
        detail=detail,
    )
    if callback is not None:
        callback(event)
    return event
