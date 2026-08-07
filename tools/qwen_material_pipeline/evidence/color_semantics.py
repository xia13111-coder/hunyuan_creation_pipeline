"""Shared deterministic colour semantics for palette evidence and fusion.

The palette schema intentionally uses a small set of human-readable colour
labels.  Some boundaries in that set describe illumination changes rather
than materially different appearances: cyan highlights are commonly called
blue by Qwen, while an orange surface in shadow is classified as brown.

Keep those equivalences in one data-only module so pixel verification and
multiview fusion cannot silently disagree about them.
"""

from __future__ import annotations

import colorsys


_EVIDENCE_FAMILIES = (
    frozenset({"white", "silver", "gray"}),
    frozenset({"blue", "cyan"}),
    frozenset({"orange", "brown"}),
)

_EVIDENCE_LABELS = {
    label: family for family in _EVIDENCE_FAMILIES for label in family
}

_FUSION_CANONICAL_LABELS = {
    "blue": "blue",
    "cyan": "blue",
    "orange": "orange",
    "brown": "orange",
}


def normalized_color_label(value: str) -> str:
    """Return one normalized palette colour label."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError("colour label must be a non-empty string")
    return value.strip().casefold()


def evidence_color_labels(value: str) -> frozenset[str]:
    """Return labels that are pixel-equivalent to ``value``.

    Neutral colours retain the existing white/silver/gray shadow family.
    Blue/cyan and orange/brown are joined because their schema boundary is a
    narrow hue or value threshold that routinely moves under illumination.
    Black deliberately remains independent.
    """

    label = normalized_color_label(value)
    return _EVIDENCE_LABELS.get(label, frozenset({label}))


def fusion_color_label(value: str) -> str:
    """Return a stable cross-view colour key and canonical display label."""

    label = normalized_color_label(value)
    return _FUSION_CANONICAL_LABELS.get(label, label)


def pixel_color_label(red: int, green: int, blue: int) -> str:
    """Classify one 8-bit sRGB pixel into the palette's coarse colour set."""

    maximum = max(red, green, blue)
    minimum = min(red, green, blue)
    delta = maximum - minimum
    value = maximum / 255.0
    saturation = 0.0 if maximum == 0 else delta / maximum
    if maximum < 55:
        return "black"
    if saturation < 0.14:
        if minimum > 215:
            return "white"
        if value > 0.68:
            return "silver"
        return "gray"
    hue = colorsys.rgb_to_hsv(
        red / 255.0,
        green / 255.0,
        blue / 255.0,
    )[0] * 360.0
    if hue < 15.0 or hue >= 345.0:
        return "red"
    if hue < 45.0:
        return "brown" if value < 0.55 else "orange"
    if hue < 70.0:
        return "yellow"
    if hue < 170.0:
        return "green"
    if hue < 200.0:
        return "cyan"
    if hue < 260.0:
        return "blue"
    return "pink"


__all__ = [
    "evidence_color_labels",
    "fusion_color_label",
    "normalized_color_label",
    "pixel_color_label",
]
