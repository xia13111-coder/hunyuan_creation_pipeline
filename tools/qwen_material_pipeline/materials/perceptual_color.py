"""Perceptual colour comparisons used by Part-ID material selection.

The old retrieval score normalized an RGB Euclidean distance by the diagonal
of the complete RGB cube.  That made most ordinary dark colours look highly
similar to one another.  This module converts display-space sRGB to CIE Lab
and uses CIEDE2000 so a score describes perceptual colour agreement instead of
distance inside an arbitrarily large cube.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def _finite_unit_triplet(color: Sequence[float]) -> tuple[float, float, float]:
    if isinstance(color, (str, bytes)) or len(color) != 3:
        raise ValueError("sRGB colour must contain exactly three channels")
    values = tuple(float(value) for value in color)
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
        raise ValueError("sRGB colour channels must be finite values in [0,1]")
    return values


def srgb_to_lab(color: Sequence[float]) -> tuple[float, float, float]:
    """Convert normalized IEC sRGB to CIE Lab using a D65 reference white."""

    red, green, blue = _finite_unit_triplet(color)

    def linearize(value: float) -> float:
        if value <= 0.04045:
            return value / 12.92
        return ((value + 0.055) / 1.055) ** 2.4

    red, green, blue = (linearize(value) for value in (red, green, blue))
    x = (0.4124564 * red + 0.3575761 * green + 0.1804375 * blue) / 0.95047
    y = 0.2126729 * red + 0.7151522 * green + 0.0721750 * blue
    z = (0.0193339 * red + 0.1191920 * green + 0.9503041 * blue) / 1.08883

    def pivot(value: float) -> float:
        delta = 6.0 / 29.0
        if value > delta**3:
            return value ** (1.0 / 3.0)
        return value / (3.0 * delta**2) + 4.0 / 29.0

    fx, fy, fz = pivot(x), pivot(y), pivot(z)
    return 116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)


def delta_e_ciede2000(
    first_lab: Sequence[float],
    second_lab: Sequence[float],
) -> float:
    """Return the CIEDE2000 colour difference for two CIE Lab triplets."""

    if len(first_lab) != 3 or len(second_lab) != 3:
        raise ValueError("Lab colours must contain exactly three channels")
    l1, a1, b1 = (float(value) for value in first_lab)
    l2, a2, b2 = (float(value) for value in second_lab)
    if any(
        not math.isfinite(value)
        for value in (l1, a1, b1, l2, a2, b2)
    ):
        raise ValueError("Lab channels must be finite")

    c1 = math.hypot(a1, b1)
    c2 = math.hypot(a2, b2)
    mean_c = (c1 + c2) / 2.0
    mean_c_seventh = mean_c**7
    g = 0.5 * (
        1.0
        - math.sqrt(
            mean_c_seventh / (mean_c_seventh + 25.0**7)
            if mean_c_seventh
            else 0.0
        )
    )
    a1_prime = (1.0 + g) * a1
    a2_prime = (1.0 + g) * a2
    c1_prime = math.hypot(a1_prime, b1)
    c2_prime = math.hypot(a2_prime, b2)

    def hue_degrees(a_value: float, b_value: float) -> float:
        if a_value == 0.0 and b_value == 0.0:
            return 0.0
        return math.degrees(math.atan2(b_value, a_value)) % 360.0

    h1_prime = hue_degrees(a1_prime, b1)
    h2_prime = hue_degrees(a2_prime, b2)
    delta_l_prime = l2 - l1
    delta_c_prime = c2_prime - c1_prime
    if c1_prime * c2_prime == 0.0:
        delta_h_degrees = 0.0
    else:
        raw_delta_h = h2_prime - h1_prime
        if abs(raw_delta_h) <= 180.0:
            delta_h_degrees = raw_delta_h
        elif raw_delta_h > 180.0:
            delta_h_degrees = raw_delta_h - 360.0
        else:
            delta_h_degrees = raw_delta_h + 360.0
    delta_h_prime = 2.0 * math.sqrt(c1_prime * c2_prime) * math.sin(
        math.radians(delta_h_degrees / 2.0)
    )

    mean_l_prime = (l1 + l2) / 2.0
    mean_c_prime = (c1_prime + c2_prime) / 2.0
    if c1_prime * c2_prime == 0.0:
        mean_h_prime = h1_prime + h2_prime
    elif abs(h1_prime - h2_prime) <= 180.0:
        mean_h_prime = (h1_prime + h2_prime) / 2.0
    elif h1_prime + h2_prime < 360.0:
        mean_h_prime = (h1_prime + h2_prime + 360.0) / 2.0
    else:
        mean_h_prime = (h1_prime + h2_prime - 360.0) / 2.0

    t = (
        1.0
        - 0.17 * math.cos(math.radians(mean_h_prime - 30.0))
        + 0.24 * math.cos(math.radians(2.0 * mean_h_prime))
        + 0.32 * math.cos(math.radians(3.0 * mean_h_prime + 6.0))
        - 0.20 * math.cos(math.radians(4.0 * mean_h_prime - 63.0))
    )
    delta_theta = 30.0 * math.exp(
        -((mean_h_prime - 275.0) / 25.0) ** 2
    )
    c_ratio = mean_c_prime**7 / (mean_c_prime**7 + 25.0**7)
    r_c = 2.0 * math.sqrt(c_ratio)
    s_l = 1.0 + (
        0.015 * (mean_l_prime - 50.0) ** 2
        / math.sqrt(20.0 + (mean_l_prime - 50.0) ** 2)
    )
    s_c = 1.0 + 0.045 * mean_c_prime
    s_h = 1.0 + 0.015 * mean_c_prime * t
    r_t = -math.sin(math.radians(2.0 * delta_theta)) * r_c
    l_term = delta_l_prime / s_l
    c_term = delta_c_prime / s_c
    h_term = delta_h_prime / s_h
    return math.sqrt(
        l_term**2 + c_term**2 + h_term**2 + r_t * c_term * h_term
    )


def srgb_delta_e(first: Sequence[float], second: Sequence[float]) -> float:
    """Return CIEDE2000 directly for two normalized sRGB colours."""

    return delta_e_ciede2000(srgb_to_lab(first), srgb_to_lab(second))


def perceptual_similarity(
    first: Sequence[float],
    second: Sequence[float],
    *,
    scale: float = 12.0,
) -> float:
    """Map CIEDE2000 to a bounded similarity with no high neutral baseline."""

    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("perceptual similarity scale must be positive")
    return math.exp(-srgb_delta_e(first, second) / scale)
