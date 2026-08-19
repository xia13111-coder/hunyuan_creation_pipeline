from __future__ import annotations

import numpy as np
import pytest

from qwen_material_pipeline.segmentation.entityseg_regions import (
    _cad_location_agreement,
    _expanded_crop,
    _select_candidate,
)


def test_expanded_crop_is_resolution_bounded() -> None:
    assert _expanded_crop(
        [0, 0, 100, 100],
        width=200,
        height=100,
        context_fraction=0.2,
    ) == (0, 0, 64, 64)


def test_cad_location_agreement_is_normalized_by_part_scale() -> None:
    seed = np.zeros((80, 80), dtype=bool)
    candidate = np.zeros_like(seed)
    seed[20:40, 20:40] = True
    candidate[22:42, 22:42] = True

    metrics = _cad_location_agreement(candidate, seed)

    assert metrics["cad_centroid_distance_pixels"] == np.sqrt(8.0)
    assert metrics["cad_centroid_distance_normalized"] == 0.1
    assert metrics["cad_direct_intersection_pixels"] == 18 * 18


def test_selection_rejects_neighboring_identical_component() -> None:
    source = np.zeros((100, 120, 3), dtype=np.uint8)
    seed = np.zeros((100, 120), dtype=bool)
    correct = np.zeros_like(seed)
    identical_neighbor = np.zeros_like(seed)
    seed[30:60, 40:55] = True
    correct[30:60, 40:55] = True
    identical_neighbor[30:60, 60:75] = True

    selected, audit = _select_candidate(
        [
            {
                "source": "cad_local_crop",
                "prediction_index": 0,
                "model_score": 0.99,
                "mask": identical_neighbor,
            },
            {
                "source": "cad_local_crop",
                "prediction_index": 1,
                "model_score": 0.60,
                "mask": correct,
            },
        ],
        seed=seed,
        source_image=source,
        minimum_shape_iou=0.5,
        minimum_area_agreement=0.5,
        maximum_centroid_distance=0.15,
    )

    assert selected is not None
    assert selected["prediction_index"] == 1
    neighbor = next(row for row in audit if row["prediction_index"] == 0)
    assert neighbor["accepted"] is False
    assert "cad_centroid_too_far_from_registered_part" in neighbor["reason_codes"]


def test_selection_does_not_move_cad_template_per_candidate() -> None:
    source = np.zeros((100, 120, 3), dtype=np.uint8)
    seed = np.zeros((100, 120), dtype=bool)
    shifted = np.zeros_like(seed)
    seed[30:60, 40:60] = True
    shifted[32:62, 42:62] = True

    selected, audit = _select_candidate(
        [
            {
                "source": "cad_local_crop",
                "prediction_index": 0,
                "model_score": 0.8,
                "mask": shifted,
            }
        ],
        seed=seed,
        source_image=source,
        minimum_shape_iou=0.5,
        minimum_area_agreement=0.5,
        maximum_centroid_distance=0.15,
        box=[250, 200, 700, 800],
    )

    assert selected is not None
    row = audit[0]
    assert row["cad_template_alignment"]["translation_xy_pixels"] == [0.0, 0.0]
    assert row["cad_template_alignment"]["part_local_translation_xy_pixels"] == [
        0.0,
        0.0,
    ]
    assert row["cad_template_alignment"]["candidate_centroid_residual_xy_pixels"] == [
        2.0,
        2.0,
    ]
    assert row["cad_template_alignment"]["per_mesh_pose_change_allowed"] is False
    assert row["cad_direct_iou"] == pytest.approx(0.72413793)
    assert row["registered_cad_centroid_distance_normalized"] > 0.0


def test_selection_uses_one_shared_view_translation_for_every_candidate() -> None:
    source = np.zeros((100, 120, 3), dtype=np.uint8)
    seed = np.zeros((100, 120), dtype=bool)
    shifted = np.zeros_like(seed)
    seed[30:60, 40:60] = True
    shifted[33:63, 44:64] = True
    shared = {
        "translation_xy_pixels": [4.0, 3.0],
        "maximum_translation_xy_pixels": [12, 12],
        "estimation_mode": (
            "whole_workpiece_foreground_to_visible_cad_union_integer_translation"
        ),
        "part_specific_translation_allowed": False,
        "cad_union_pixels": 600,
    }

    selected, audit = _select_candidate(
        [
            {
                "source": "cad_local_crop",
                "prediction_index": 0,
                "model_score": 0.8,
                "mask": shifted,
            }
        ],
        seed=seed,
        source_image=source,
        minimum_shape_iou=0.5,
        minimum_area_agreement=0.5,
        maximum_centroid_distance=0.15,
        box=[250, 200, 700, 800],
        view_shared_alignment=shared,
    )

    assert selected is not None
    alignment = audit[0]["cad_template_alignment"]
    assert alignment["translation_xy_pixels"] == [4.0, 3.0]
    assert alignment["part_local_translation_xy_pixels"] == [0.0, 0.0]
    assert alignment["part_specific_translation_allowed"] is False
    assert alignment["per_mesh_pose_change_allowed"] is False
