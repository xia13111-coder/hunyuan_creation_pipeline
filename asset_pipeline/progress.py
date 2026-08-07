"""Versioned progress events and human-readable progress rendering."""

from __future__ import annotations

import json
import re
from typing import Callable, Mapping


PROGRESS_PREFIX = "@@ASSET_PROGRESS "
PROGRESS_SCHEMA_VERSION = "asset-pipeline-progress/v1"
PROGRESS_STATES = frozenset({"start", "update", "complete", "retry", "failed"})
DEFAULT_BAR_WIDTH = 20

_KIT_PYTHON_STDERR_MARKER = "[py stderr]: "
_KIT_PROGRESS_WRAPPER_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T[^\r\n ]+"
    r"(?: \[[^\r\n]*\])*"
    r" \[Error\] \[omni\.kit\.app\._impl\] "
    + re.escape(_KIT_PYTHON_STDERR_MARKER)
    + r"(?P<payload>[^\r\n]*)$"
)

ProgressEvent = dict[str, object]
ProgressLogCallback = Callable[[str], None] | None

_EVENT_FIELDS = frozenset(
    {
        "schema_version",
        "scope",
        "stage",
        "state",
        "current",
        "total",
        "unit",
        "detail",
    }
)


class ProgressProtocolError(ValueError):
    """Raised when a marked progress event violates the v1 protocol."""


def validate_progress_event(event: object) -> ProgressEvent:
    """Validate and return a copy of one v1 progress event.

    Counted events use integer ``current``/``total`` values and a non-empty
    ``unit``. Stages that cannot report a meaningful count use JSON ``null``
    for all three fields.
    """

    if not isinstance(event, Mapping):
        raise ProgressProtocolError("progress event must be a JSON object")

    fields = frozenset(event)
    missing = _EVENT_FIELDS - fields
    extra = fields - _EVENT_FIELDS
    if missing:
        raise ProgressProtocolError(
            f"progress event is missing fields: {', '.join(sorted(missing))}"
        )
    if extra:
        raise ProgressProtocolError(
            "progress event has unknown fields: "
            + ", ".join(sorted(str(name) for name in extra))
        )
    if event["schema_version"] != PROGRESS_SCHEMA_VERSION:
        raise ProgressProtocolError("unsupported progress schema_version")

    for name in ("scope", "stage"):
        value = event[name]
        if not isinstance(value, str) or not value.strip():
            raise ProgressProtocolError(f"progress {name} must be a non-empty string")

    state = event["state"]
    if not isinstance(state, str) or state not in PROGRESS_STATES:
        raise ProgressProtocolError(
            "progress state must be one of: "
            + ", ".join(sorted(PROGRESS_STATES))
        )

    detail = event["detail"]
    if not isinstance(detail, str):
        raise ProgressProtocolError("progress detail must be a string")

    current = event["current"]
    total = event["total"]
    unit = event["unit"]
    count_values = (current, total, unit)
    if any(value is None for value in count_values):
        if not all(value is None for value in count_values):
            raise ProgressProtocolError(
                "progress current, total, and unit must all be null or all be set"
            )
    else:
        if isinstance(current, bool) or not isinstance(current, int):
            raise ProgressProtocolError("progress current must be an integer")
        if isinstance(total, bool) or not isinstance(total, int):
            raise ProgressProtocolError("progress total must be an integer")
        if total <= 0:
            raise ProgressProtocolError("progress total must be greater than zero")
        if current < 0 or current > total:
            raise ProgressProtocolError(
                "progress current must be between zero and total"
            )
        if not isinstance(unit, str) or not unit.strip():
            raise ProgressProtocolError("progress unit must be a non-empty string")

    return {name: event[name] for name in _EVENT_FIELDS}


def make_progress_event(
    *,
    scope: str,
    stage: str,
    state: str,
    current: int | None,
    total: int | None,
    unit: str | None = None,
    detail: str = "",
) -> ProgressEvent:
    """Build a validated v1 progress event."""

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


def encode_progress_event(event: object) -> str:
    """Serialize an event for transport over a subprocess output stream."""

    validated = validate_progress_event(event)
    return PROGRESS_PREFIX + json.dumps(
        validated,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def parse_progress_line(line: str) -> ProgressEvent | None:
    """Parse a marked subprocess line.

    Non-protocol lines return ``None``. A line with the marker but malformed
    JSON or an invalid event raises :class:`ProgressProtocolError`; stream
    consumers should catch it and preserve or warn about the original line.
    """

    if not line.startswith(PROGRESS_PREFIX):
        return None
    payload = line[len(PROGRESS_PREFIX) :]
    try:
        event = json.loads(payload)
    except (TypeError, ValueError, RecursionError) as exc:
        raise ProgressProtocolError(f"invalid progress JSON: {exc}") from exc
    return validate_progress_event(event)


def format_progress_event(
    event: object,
    *,
    bar_width: int = DEFAULT_BAR_WIDTH,
) -> str:
    """Render a validated event as a stable, ASCII-only log line."""

    validated = validate_progress_event(event)
    scope_stage = f"{validated['scope']}/{validated['stage']}"
    state = str(validated["state"]).upper()
    detail = str(validated["detail"]).strip()
    current = validated["current"]
    total = validated["total"]
    unit = validated["unit"]

    if current is None:
        parts = [f"[PROGRESS] {scope_stage} {state}"]
        if detail:
            parts.append(detail)
        return " ".join(parts)

    if isinstance(bar_width, bool) or not isinstance(bar_width, int) or bar_width <= 0:
        raise ValueError("bar_width must be a positive integer")
    assert isinstance(current, int)
    assert isinstance(total, int)
    completed = (current * bar_width * 2 + total) // (total * 2)
    completed = min(bar_width, max(0, completed))
    percentage_tenths = (current * 1000 * 2 + total) // (total * 2)
    bar = "#" * completed + "-" * (bar_width - completed)
    parts = [
        f"[PROGRESS] [{bar}] "
        f"{percentage_tenths // 10}.{percentage_tenths % 10}%",
        scope_stage,
        state,
    ]
    if detail:
        parts.append(detail)
    parts.append(f"({unit} {current}/{total})")
    return " ".join(parts)


def format_subprocess_output_line(line: str) -> str:
    """Translate a valid marked line and safely retain malformed output."""

    protocol_line = line
    if not line.startswith(PROGRESS_PREFIX):
        kit_match = _KIT_PROGRESS_WRAPPER_RE.fullmatch(line)
        if kit_match is None:
            return line
        payload = kit_match.group("payload")
        if not payload.startswith(PROGRESS_PREFIX):
            return line
        protocol_line = payload

    try:
        event = parse_progress_line(protocol_line)
    except ProgressProtocolError as exc:
        return f"[WARNING] malformed asset progress line ({exc}): {line}"
    if event is None:
        return line
    return format_progress_event(event)


def emit_progress(
    log_cb: ProgressLogCallback,
    *,
    scope: str,
    stage: str,
    state: str,
    current: int | None,
    total: int | None,
    unit: str | None = None,
    detail: str = "",
) -> None:
    """Emit one root-process event directly as a human-readable progress line."""

    event = make_progress_event(
        scope=scope,
        stage=stage,
        state=state,
        current=current,
        total=total,
        unit=unit,
        detail=detail,
    )
    if log_cb is not None:
        log_cb(format_progress_event(event))


__all__ = [
    "DEFAULT_BAR_WIDTH",
    "PROGRESS_PREFIX",
    "PROGRESS_SCHEMA_VERSION",
    "PROGRESS_STATES",
    "ProgressEvent",
    "ProgressProtocolError",
    "emit_progress",
    "encode_progress_event",
    "format_progress_event",
    "format_subprocess_output_line",
    "make_progress_event",
    "parse_progress_line",
    "validate_progress_event",
]
