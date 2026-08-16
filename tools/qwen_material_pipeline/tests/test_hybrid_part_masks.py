from __future__ import annotations

import numpy as np

from qwen_material_pipeline.segmentation.hybrid_part_masks import (
    _connected_component_count,
    _entity_rejection_reasons,
    _sam_aligned_cad_seed,
    _trim_entity_to_cad_support,
)


def _safe_metrics() -> dict[str, float | int]:
    return {
        "connected_component_count": 1,
        "entity_to_cad_area_ratio": 1.2,
        "entity_cad_direct_iou": 0.7,
        "entity_cad_shape_iou": 0.8,
        "entity_cad_centroid_distance": 0.05,
        "entity_edge_support": 0.95,
        "entity_edge_improvement": 0.12,
        "sam_entity_overlap_smaller": 0.8,
    }


def test_safe_entity_boundary_can_replace_sam3() -> None:
    assert _entity_rejection_reasons(_safe_metrics(), sam_accepted=True) == []


def test_entity_only_candidate_does_not_require_sam3_comparison() -> None:
    metrics = _safe_metrics()
    metrics["entity_edge_improvement"] = -1.0
    metrics["sam_entity_overlap_smaller"] = 0.0

    assert _entity_rejection_reasons(metrics, sam_accepted=False) == []


def test_merged_entity_and_non_improving_boundary_are_rejected() -> None:
    metrics = _safe_metrics()
    metrics["connected_component_count"] = 2
    metrics["entity_to_cad_area_ratio"] = 1.9
    metrics["entity_edge_improvement"] = 0.0

    reasons = _entity_rejection_reasons(metrics, sam_accepted=True)

    assert "entity_mask_is_not_one_connected_component" in reasons
    assert "entity_mask_area_exceeds_cad_bound" in reasons
    assert "entity_boundary_does_not_improve_over_sam3" in reasons


def test_entity_may_disagree_with_sam_only_when_cad_direct_overlap_is_strong() -> None:
    metrics = _safe_metrics()
    metrics["sam_entity_overlap_smaller"] = 0.3
    metrics["entity_cad_direct_iou"] = 0.59
    assert "entity_disagrees_with_both_sam3_and_cad_location" in (
        _entity_rejection_reasons(metrics, sam_accepted=True)
    )

    metrics["entity_cad_direct_iou"] = 0.61
    assert "entity_disagrees_with_both_sam3_and_cad_location" not in (
        _entity_rejection_reasons(metrics, sam_accepted=True)
    )


def test_connected_component_count_ignores_tiny_specks() -> None:
    mask = np.zeros((40, 40), dtype=bool)
    mask[5:15, 5:15] = True
    mask[30, 30] = True

    assert _connected_component_count(mask) == 1


def test_cad_support_trim_removes_touching_adjacent_part() -> None:
    seed = np.zeros((80, 100), dtype=bool)
    entity = np.zeros_like(seed)
    seed[20:60, 20:60] = True
    entity[18:62, 18:62] = True
    entity[45:65, 62:90] = True

    trimmed, audit = _trim_entity_to_cad_support(entity, seed)

    assert np.all(trimmed <= entity)
    assert not np.any(trimmed[:, 70:])
    assert audit["retained_entity_fraction"] < 1.0
    assert audit["final_to_cad_area_ratio"] <= 1.25


def test_cad_support_trim_preserves_boundary_already_inside_bound() -> None:
    seed = np.zeros((80, 100), dtype=bool)
    entity = np.zeros_like(seed)
    seed[20:60, 20:60] = True
    entity[19:61, 19:61] = True

    trimmed, audit = _trim_entity_to_cad_support(entity, seed)

    assert np.array_equal(trimmed, entity)
    assert audit["retained_entity_fraction"] == 1.0


def test_sam_specific_cad_area_bound_is_stricter_than_entity_default() -> None:
    seed = np.zeros((80, 100), dtype=bool)
    candidate = np.zeros_like(seed)
    seed[20:60, 20:60] = True
    candidate[17:63, 17:63] = True

    default_trimmed, _default_audit = _trim_entity_to_cad_support(candidate, seed)
    sam_trimmed, sam_audit = _trim_entity_to_cad_support(
        candidate,
        seed,
        maximum_final_to_cad_area_ratio=1.15,
    )

    assert np.count_nonzero(sam_trimmed) < np.count_nonzero(default_trimmed)
    assert sam_audit["final_to_cad_area_ratio"] <= 1.15


def test_hybrid_replays_sam_shared_camera_template_translation() -> None:
    seed = np.zeros((20, 30), dtype=bool)
    seed[5:10, 7:12] = True
    aligned, audit = _sam_aligned_cad_seed(
        seed,
        {
            "box_audits": [
                {
                    "shape_point_refinement": {
                        "accepted": True,
                        "prompt_audit": {"translation_xy_pixels": [3.0, 2.0]},
                    }
                }
            ]
        },
    )

    expected = np.zeros_like(seed)
    expected[7:12, 10:15] = True
    assert np.array_equal(aligned, expected)
    assert audit["per_mesh_pose_change_allowed"] is False
