from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from qwen_material_pipeline.evidence.part_id_projection import (
    _load_part_id_refinement_manifest,
)
from qwen_material_pipeline.segmentation.hybrid_part_masks import (
    build_hybrid_masks,
    _connected_component_count,
    _entity_aligned_cad_seed,
    _entity_rejection_reasons,
    _iterative_shape_guided_refinement,
    _sam_aligned_cad_seed,
    _trim_entity_to_cad_support,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def test_iterative_refinement_removes_current_view_occluder_and_snaps_edges() -> None:
    image = np.zeros((96, 128, 3), dtype=np.uint8)
    image[20:76, 24:104] = (30, 130, 40)
    # A light occluding bar crosses a green CAD panel in the current view.
    image[42:50, 60:112] = (210, 210, 210)
    amodal = np.zeros((96, 128), dtype=bool)
    amodal[20:76, 24:104] = True
    visible = amodal.copy()
    visible[42:50, 60:104] = False
    sam = amodal.copy()
    sam[40:53, 58:108] = True
    sam[38:41, 92:101] = True

    refined, audit, _support = _iterative_shape_guided_refinement(
        image=image,
        visible_seed=visible,
        amodal_seed=amodal,
        candidate_masks=[("sam3", sam)],
        primary_candidate_source="sam3",
    )

    known_occluded = amodal & ~visible
    assert np.count_nonzero(refined & known_occluded) == 0
    assert audit["known_occluded_primary_candidate_pixels_removed"] > 0
    assert audit["selected_iteration"] > 0
    assert (
        audit["final_metrics"]["image_edge_support"]
        > audit["initial_metrics"]["image_edge_support"]
    )
    assert not np.array_equal(refined, sam)


def test_iterative_refinement_jointly_consumes_safe_sam_and_entity_priors() -> None:
    image = np.zeros((80, 100, 3), dtype=np.uint8)
    image[18:62, 20:80] = (80, 140, 90)
    visible = np.zeros((80, 100), dtype=bool)
    visible[18:62, 20:80] = True
    sam = np.zeros_like(visible)
    sam[17:61, 19:79] = True
    entity = np.zeros_like(visible)
    entity[19:63, 21:81] = True

    refined, audit, _support = _iterative_shape_guided_refinement(
        image=image,
        visible_seed=visible,
        amodal_seed=visible,
        candidate_masks=[("entityseg", entity), ("sam3", sam)],
        primary_candidate_source="entityseg",
    )

    assert np.any(refined)
    assert audit["candidate_sources"] == ["entityseg", "sam3"]
    assert audit["prior_candidate_role"] == "probable_foreground_initialization_only"
    assert audit["method"] == "iterative_visible_mesh_edge_optimization"


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


def test_hybrid_replays_selected_entity_bounded_camera_translation() -> None:
    seed = np.zeros((20, 30), dtype=bool)
    seed[5:10, 7:12] = True
    aligned, audit = _entity_aligned_cad_seed(
        seed,
        {
            "selected_candidate": {
                "cad_template_alignment": {
                    "translation_xy_pixels": [-2.0, 3.0],
                    "per_mesh_pose_change_allowed": False,
                }
            }
        },
    )

    expected = np.zeros_like(seed)
    expected[8:13, 5:10] = True
    assert np.array_equal(aligned, expected)
    assert audit["source"] == "entityseg_selected_candidate_bounded_camera_residual"
    assert audit["per_mesh_pose_change_allowed"] is False


def test_hybrid_rejects_entity_alignment_that_changes_one_mesh_pose() -> None:
    with pytest.raises(ValueError, match="translation is malformed"):
        _entity_aligned_cad_seed(
            np.ones((4, 4), dtype=bool),
            {
                "selected_candidate": {
                    "cad_template_alignment": {
                        "translation_xy_pixels": [1.0, 2.0],
                        "per_mesh_pose_change_allowed": True,
                    }
                }
            },
        )


def test_hybrid_manifest_is_directly_consumable_as_part_id_evidence(
    tmp_path: Path,
) -> None:
    source = np.zeros((48, 64, 3), dtype=np.uint8)
    source[12:32, 20:40] = 180
    seed = np.zeros((48, 64), dtype=np.uint8)
    seed[12:32, 20:40] = 255
    source_path = tmp_path / "source.png"
    seed_path = tmp_path / "seed.png"
    sam_mask_path = tmp_path / "sam.png"
    assert cv2.imwrite(str(source_path), source)
    assert cv2.imwrite(str(seed_path), seed)
    assert cv2.imwrite(str(sam_mask_path), seed)
    shared = {
        "translation_xy_pixels": [0.0, 0.0],
        "maximum_translation_xy_pixels": [4, 4],
        "estimation_mode": (
            "whole_workpiece_foreground_to_visible_cad_union_integer_translation"
        ),
        "part_specific_translation_allowed": False,
        "cad_union_pixels": 400,
    }
    shape_candidate = {
        "candidate_index": 0,
        "accepted": True,
        "cad_shape_seed_pixels": 400,
        "cad_shape_iou": 1.0,
        "cad_shape_area_agreement": 1.0,
        "cad_shape_location_invariant": True,
    }
    sam = {
        "schema_version": "qwen-sam3-region-result/v1",
        "request": {"path": str(tmp_path / "request.json"), "sha256": "a" * 64},
        "policy": {},
        "records": [
            {
                "view_id": "front",
                "group_id": "P0001",
                "source_image": str(source_path),
                "source_image_sha256": _sha256(source_path),
                "view_shared_alignment": shared,
                "accepted": True,
                "mask": {"path": str(sam_mask_path), "sha256": _sha256(sam_mask_path)},
                "cad_projection_seed": {
                    "path": str(seed_path),
                    "sha256": _sha256(seed_path),
                },
                "cad_amodal_template": None,
                "box_audits": [
                    {
                        "accepted": True,
                        "selected_candidate_index": 0,
                        "shape_point_refinement": {
                            "accepted": True,
                            "prompt_audit": {
                                "translation_xy_pixels": [0.0, 0.0],
                                "part_local_translation_xy_pixels": [0.0, 0.0],
                                "part_specific_translation_allowed": False,
                                "per_mesh_pose_change_allowed": False,
                            },
                        },
                        "candidates": [shape_candidate],
                    }
                ],
            }
        ],
    }
    entity = {
        "schema_version": "qwen-entityseg-region-result/v1",
        "request": dict(sam["request"]),
        "policy": {},
        "records": [
            {
                "view_id": "front",
                "group_id": "P0001",
                "source_image": str(source_path),
                "source_image_sha256": _sha256(source_path),
                "view_shared_alignment": shared,
                "accepted": False,
                "selected_candidate": None,
                "mask": None,
                "cad_projection_seed": {
                    "path": str(seed_path),
                    "sha256": _sha256(seed_path),
                },
                "cad_amodal_template": None,
            }
        ],
    }
    sam_path = tmp_path / "sam-manifest.json"
    entity_path = tmp_path / "entity-manifest.json"
    sam_path.write_text(json.dumps(sam), encoding="utf-8")
    entity_path.write_text(json.dumps(entity), encoding="utf-8")
    hybrid_dir = tmp_path / "hybrid"

    result = build_hybrid_masks(
        sam_manifest_path=sam_path,
        entity_manifest_path=entity_path,
        output_dir=hybrid_dir,
    )
    manifest_path = hybrid_dir / "manifest.json"
    loaded, owner, accepted, records = _load_part_id_refinement_manifest(
        manifest_path,
        hybrid=True,
    )

    assert loaded == result
    assert owner == manifest_path
    assert set(records) == {("front", "P0001")}
    selected = accepted[("front", "P0001")]
    assert selected["selected_source"] == "shape_guided_iterative"
    assert selected["primary_candidate_source"] == "sam3"
    assert selected["candidate_sources"] == ["sam3"]
    assert selected["view_shared_alignment"] == shared
    assert selected["shape_candidate"]["cad_shape_iou"] == 1.0
    assert result["records"][0]["cad_projection_seed"]["sha256"] == _sha256(seed_path)
    assert result["records"][0]["iterative_refinement"]["method"] == (
        "iterative_visible_mesh_edge_optimization"
    )
