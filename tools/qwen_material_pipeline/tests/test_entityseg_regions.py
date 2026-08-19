from __future__ import annotations

import numpy as np
import pytest

from qwen_material_pipeline.segmentation.entityseg_regions import (
    _cad_location_agreement,
    _expanded_crop,
    _select_candidate,
)


def test_expanded_crop_is_resolution_bounded() -> None:
    assert (
        _expanded_crop(
            [0, 0, 100, 100],
            width=200,
            height=100,
            context_fraction=0.2,
        )
        == (0, 0, 64, 64)
    )


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

    selected, audit, repair = _select_candidate(
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
    assert repair is None
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

    selected, audit, _repair = _select_candidate(
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

    selected, audit, _repair = _select_candidate(
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


def test_oversized_local_entity_can_repair_only_an_enclosed_cad_hole() -> None:
    source = np.zeros((100, 120, 3), dtype=np.uint8)
    amodal = np.zeros((100, 120), dtype=bool)
    amodal[20:80, 25:95] = True
    visible = amodal.copy()
    visible[45:55, 50:70] = False
    oversized = np.zeros_like(visible)
    oversized[15:85, 15:110] = True

    selected, audit, repair = _select_candidate(
        [
            {
                "source": "cad_local_crop",
                "prediction_index": 0,
                "model_score": 0.9,
                "mask": oversized,
            }
        ],
        seed=visible,
        amodal=amodal,
        source_image=source,
        minimum_shape_iou=0.5,
        minimum_area_agreement=0.5,
        maximum_centroid_distance=0.15,
    )

    assert selected is None
    assert repair is not None
    assert repair["internal_repair_eligible"] is True
    assert repair["internal_repair"]["entity_supported_enclosed_hole_count"] == 1
    assert repair["internal_repair"]["entity_supported_enclosed_hole_pixels"] > 0
    assert audit[0]["accepted"] is False


def test_entity_cannot_repair_occlusion_connected_to_mesh_exterior() -> None:
    source = np.zeros((100, 120, 3), dtype=np.uint8)
    amodal = np.zeros((100, 120), dtype=bool)
    amodal[20:80, 25:95] = True
    visible = amodal.copy()
    visible[45:55, 60:95] = False
    oversized = np.zeros_like(visible)
    oversized[15:85, 15:110] = True

    _selected, audit, repair = _select_candidate(
        [
            {
                "source": "cad_local_crop",
                "prediction_index": 0,
                "model_score": 0.9,
                "mask": oversized,
            }
        ],
        seed=visible,
        amodal=amodal,
        source_image=source,
        minimum_shape_iou=0.5,
        minimum_area_agreement=0.5,
        maximum_centroid_distance=0.15,
    )

    assert repair is None
    assert (
        "no_entity_supported_bounded_cad_internal_gap"
        in audit[0]["internal_repair_rejection_reasons"]
    )


def test_entity_can_repair_scale_bounded_narrow_cad_internal_gap() -> None:
    source = np.zeros((100, 120, 3), dtype=np.uint8)
    amodal = np.zeros((100, 120), dtype=bool)
    amodal[20:80, 25:95] = True
    visible = amodal.copy()
    # A long but thin CAD slit reaches the projected boundary. It is not a
    # topological hole, yet it closes at the automatically derived part scale.
    visible[48:52, 55:95] = False
    oversized = np.zeros_like(visible)
    oversized[15:85, 15:110] = True

    selected, audit, repair = _select_candidate(
        [
            {
                "source": "cad_local_crop",
                "prediction_index": 0,
                "model_score": 0.9,
                "mask": oversized,
            }
        ],
        seed=visible,
        amodal=amodal,
        source_image=source,
        minimum_shape_iou=0.5,
        minimum_area_agreement=0.5,
        maximum_centroid_distance=0.15,
    )

    assert selected is None
    assert repair is not None
    assert repair["internal_repair_eligible"] is True
    assert repair["internal_repair"]["enclosed_cad_hole_count"] == 0
    assert repair["internal_repair"]["narrow_cad_internal_gap_pixels"] >= 152
    assert (
        repair["internal_repair"]["automatic_internal_gap_closing_radius_pixels"] == 2
    )
    assert audit[0]["accepted"] is False
