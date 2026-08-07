"""Convert verified inverse-rendering evidence into renderer parameters.

This module is deliberately independent of USD and ML runtimes.  MVInverse
predicts display-space base colour plus metallic/roughness evidence; material
application consumes only values that passed the evidence layer's fail-closed
``auto`` decision.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


MVINVERSE_EVIDENCE_SCHEMA = "qwen-mvinverse-pbr-evidence/v1"


def srgb_channel_to_linear(channel: float) -> float:
    """Convert one normalized IEC sRGB channel to linear RGB."""

    value = float(channel)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"sRGB channel must be finite and in [0,1], got {channel!r}")
    if value <= 0.04045:
        return value / 12.92
    return ((value + 0.055) / 1.055) ** 2.4


def srgb_to_linear(color: Sequence[float]) -> list[float]:
    """Convert a three-channel normalized sRGB colour to linear RGB."""

    if isinstance(color, (str, bytes)) or len(color) != 3:
        raise ValueError("base_color_srgb must contain exactly three channels")
    return [srgb_channel_to_linear(channel) for channel in color]


def _unit_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number in [0,1]")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{label} must be finite and in [0,1]")
    return number


def mvinverse_paint_parameters(
    evidence: Mapping[str, Any], group_id: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return MDL paint parameters and an audit record for one eligible group.

    A missing, duplicated, rejected, or incomplete group raises ``ValueError``.
    This prevents a workflow from silently falling back to stale constants when
    inverse-rendering evidence is unavailable.
    """

    if evidence.get("schema_version") != MVINVERSE_EVIDENCE_SCHEMA:
        raise ValueError(
            f"unsupported MVInverse evidence schema: {evidence.get('schema_version')!r}"
        )
    groups = evidence.get("groups")
    if not isinstance(groups, list):
        raise ValueError("MVInverse evidence has no groups array")
    matches = [
        group
        for group in groups
        if isinstance(group, Mapping) and group.get("group_id") == group_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"MVInverse group {group_id!r} must occur exactly once, found {len(matches)}"
        )
    group = matches[0]
    suggestion = group.get("suggestion")
    if not isinstance(suggestion, Mapping):
        raise ValueError(f"MVInverse group {group_id!r} has no suggestion")
    if (
        suggestion.get("decision") != "auto"
        or suggestion.get("auto_parameter_eligible") is not True
    ):
        raise ValueError(
            f"MVInverse group {group_id!r} is not eligible for automatic parameters"
        )

    raw_color = suggestion.get("base_color_srgb")
    if not isinstance(raw_color, Sequence) or isinstance(raw_color, (str, bytes)):
        raise ValueError(f"MVInverse group {group_id!r} has invalid base_color_srgb")
    color_srgb = [_unit_float(value, "base_color_srgb") for value in raw_color]
    color_linear = srgb_to_linear(color_srgb)
    metallic = _unit_float(suggestion.get("metallic"), "metallic")
    roughness = _unit_float(suggestion.get("roughness"), "roughness")

    parameters = {
        "paint_color": color_linear,
        "paint_roughness": roughness,
        "paint_roughness_variation": 0.0,
        "dirt_weight": 0.0,
        "wash_weight": 0.0,
        "paint_stroke_normal_strength": 0.0,
        "uneven_normal_strength": 0.0,
        "enable_rust_damage": False,
    }
    audit = {
        "group_id": group_id,
        "decision": "auto",
        "base_color_srgb": color_srgb,
        "base_color_linear": color_linear,
        "metallic": metallic,
        "roughness": roughness,
        "reason_codes": list(suggestion.get("reason_codes") or []),
        "warning_codes": list(suggestion.get("warning_codes") or []),
    }
    return parameters, audit
