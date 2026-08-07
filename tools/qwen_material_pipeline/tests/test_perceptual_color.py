from __future__ import annotations

import pytest

from qwen_material_pipeline.materials.perceptual_color import (
    delta_e_ciede2000,
    perceptual_similarity,
    srgb_delta_e,
)


def test_ciede2000_matches_published_reference_pair() -> None:
    # Sharma et al. CIEDE2000 supplementary test pair 1.
    assert delta_e_ciede2000(
        [50.0, 2.6772, -79.7751],
        [50.0, 0.0, -82.7485],
    ) == pytest.approx(2.0425, abs=1e-4)


def test_gray_is_not_given_a_high_green_similarity() -> None:
    green = [0.204, 0.459, 0.247]
    gray = [0.396, 0.396, 0.396]
    assert srgb_delta_e(green, gray) > 20.0
    assert perceptual_similarity(green, gray) < 0.15
    assert perceptual_similarity(green, green) == pytest.approx(1.0)
