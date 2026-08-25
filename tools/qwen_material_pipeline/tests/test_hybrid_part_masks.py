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
    _best_model_domain_shape_agreement,
    _connected_component_count,
    _entity_aligned_cad_seed,
    _entity_rejection_reasons,
    _filamentary_shape_audit,
    _full_resolution_similarity_fallback,
    _iterative_shape_guided_refinement,
    _model_domain_shape_references,
    _part_color,
    _register_visible_template_to_photo,
    _sam_aligned_cad_seed,
    _trim_entity_to_cad_support,
)
from qwen_material_pipeline.segmentation.part_relation_guidance import (
    _infer_target_affine,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


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


def test_local_photo_registration_moves_only_the_2d_cad_mask_to_supported_edges() -> None:
    image = np.zeros((80, 100, 3), dtype=np.uint8)
    image[22:62, 32:72] = (80, 150, 90)
    visible = np.zeros((80, 100), dtype=bool)
    visible[20:60, 30:70] = True
    candidate = np.zeros_like(visible)
    candidate[22:62, 32:72] = True

    registered, registered_amodal, audit = _register_visible_template_to_photo(
        image=image,
        visible_seed=visible,
        amodal_seed=visible,
        candidate_masks=[candidate],
    )

    assert audit["accepted"] is True
    assert audit["translation_xy_pixels"] == [2, 2]
    assert audit["rotation_degrees"] == pytest.approx(0.0)
    assert audit["selected_metrics"]["registration_score"] > (
        audit["initial_metrics"]["registration_score"]
    )
    assert np.array_equal(registered, candidate)
    assert np.array_equal(registered_amodal, candidate)
    assert audit["transformed_object"] == "reference_view_segmentation_template_only"
    assert audit["cad_mesh_transform_changed"] is False
    assert audit["assembly_camera_changed"] is False
    assert audit["per_mesh_pose_change_allowed"] is False


def test_local_photo_registration_keeps_zero_transform_when_already_aligned() -> None:
    image = np.zeros((80, 100, 3), dtype=np.uint8)
    image[20:60, 30:70] = (80, 150, 90)
    visible = np.zeros((80, 100), dtype=bool)
    visible[20:60, 30:70] = True

    registered, registered_amodal, audit = _register_visible_template_to_photo(
        image=image,
        visible_seed=visible,
        amodal_seed=visible,
        candidate_masks=[visible],
    )

    assert audit["accepted"] is False
    assert audit["translation_xy_pixels"] == [0, 0]
    assert audit["rotation_degrees"] == pytest.approx(0.0)
    assert np.array_equal(registered, visible)
    assert np.array_equal(registered_amodal, visible)


def test_iterative_refinement_selects_registered_branch_only_after_final_pareto() -> None:
    image = np.zeros((80, 100, 3), dtype=np.uint8)
    image[22:62, 32:72] = (80, 150, 90)
    visible = np.zeros((80, 100), dtype=bool)
    visible[20:60, 30:70] = True
    candidate = np.zeros_like(visible)
    candidate[22:62, 32:72] = True
    model_shape = np.ones((40, 40), dtype=bool)

    refined, audit, _support = _iterative_shape_guided_refinement(
        image=image,
        visible_seed=visible,
        amodal_seed=None,
        model_visible_shape=model_shape,
        candidate_masks=[("sam3", candidate)],
        primary_candidate_source="sam3",
    )

    registration = audit["reference_space_local_registration"]
    assert registration["accepted"] is True
    assert registration["selected_final_branch"] == ("bounded_2d_local_registration")
    assert registration["final_selection_rejection_reasons"] == []
    assert np.array_equal(refined, candidate)
    for metric in (
        "image_edge_support",
        "model_domain_shape_score",
        "mean_prior_candidate_iou",
    ):
        assert registration["registered_final_metrics"][metric] >= (
            registration["zero_transform_final_metrics"][metric]
        )


def test_iterative_refinement_keeps_zero_branch_without_registration_evidence() -> None:
    image = np.zeros((80, 100, 3), dtype=np.uint8)
    image[20:60, 30:70] = (80, 150, 90)
    visible = np.zeros((80, 100), dtype=bool)
    visible[20:60, 30:70] = True

    refined, audit, _support = _iterative_shape_guided_refinement(
        image=image,
        visible_seed=visible,
        amodal_seed=None,
        model_visible_shape=np.ones((40, 40), dtype=bool),
        candidate_masks=[("sam3", visible)],
        primary_candidate_source="sam3",
    )

    registration = audit["reference_space_local_registration"]
    assert registration["accepted"] is False
    assert registration["selected_final_branch"] == "zero_transform_baseline"
    assert registration["translation_xy_pixels"] == [0, 0]
    assert registration["rotation_degrees"] == pytest.approx(0.0)
    assert np.array_equal(refined, visible)


def test_relation_cad_fallback_can_leave_its_self_derived_support_mask() -> None:
    image = np.zeros((80, 100, 3), dtype=np.uint8)
    image[22:62, 32:72] = (80, 150, 90)
    relation_seed = np.zeros((80, 100), dtype=bool)
    relation_seed[20:60, 30:70] = True

    refined, audit, _support = _iterative_shape_guided_refinement(
        image=image,
        visible_seed=relation_seed,
        amodal_seed=None,
        model_visible_shape=np.ones((40, 40), dtype=bool),
        candidate_masks=[("relation_cad_location_fallback", relation_seed)],
        primary_candidate_source="relation_cad_location_fallback",
        complete_target_shape_variants=True,
    )

    registration = audit["model_domain_photo_registration"]
    assert registration["accepted"] is True
    assert registration["selected_final_branch"] == (
        "registered_model_image_shape_proposal"
    )
    assert registration["location_prior_only"] is True
    assert registration["candidate_support_role"] == (
        "location_hint_not_search_or_acceptance_floor"
    )
    assert registration["proposal_candidate_support"] < (
        registration["original_cad_candidate_support_floor"]
    )
    assert registration["proposal_photo_assembly_score"] > (
        registration["baseline_photo_assembly_score"]
    )
    assert registration["selection_rejection_reasons"] == []
    assert np.array_equal(refined, image[:, :, 0] > 0)


def test_filamentary_relation_fallback_uses_photo_centerline_not_its_old_mask() -> None:
    image = np.full((96, 128, 3), (35, 105, 45), dtype=np.uint8)
    target_points = np.asarray(
        [(28, 34), (28, 58), (42, 70), (84, 70), (98, 58)], np.int32
    )
    cv2.polylines(image, [target_points], False, (205, 210, 205), 5, cv2.LINE_AA)

    relation_seed = np.zeros(image.shape[:2], dtype=np.uint8)
    old_points = target_points + np.asarray((0, -5), dtype=np.int32)
    cv2.polylines(relation_seed, [old_points], False, 1, 5, cv2.LINE_8)
    model_shape = relation_seed[20:76, 20:106] > 0

    geometry = _filamentary_shape_audit(model_shape)
    assert geometry["filamentary"] is True

    refined, audit, _support = _iterative_shape_guided_refinement(
        image=image,
        visible_seed=relation_seed > 0,
        amodal_seed=None,
        model_visible_shape=model_shape,
        candidate_masks=[("relation_cad_location_fallback", relation_seed > 0)],
        primary_candidate_source="relation_cad_location_fallback",
        complete_target_shape_variants=True,
    )

    registration = audit["model_domain_photo_registration"]
    assert registration["accepted"] is True
    assert registration["registration_evidence_mode"] == (
        "filamentary_bright_achromatic_ridge_boundary_and_assembly"
    )
    assert registration["photo_evidence_metric"] == (
        "filamentary_bright_achromatic_ridge_boundary_geometric_mean"
    )
    assert registration["proposal_photo_evidence"] > (
        registration["baseline_photo_evidence"]
    )
    assert registration["local_translation_xy_pixels"][1] > 0
    assert registration["cad_mesh_transform_changed"] is False
    assert registration["assembly_camera_changed"] is False
    assert not np.array_equal(refined, relation_seed > 0)


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


def test_model_domain_shape_selects_the_corresponding_local_component() -> None:
    candidate = np.zeros((80, 100), dtype=bool)
    candidate[20:60, 20:60] = True
    matching_component = np.zeros((120, 140), dtype=bool)
    matching_component[30:90, 20:80] = True
    unrelated_component = np.zeros_like(matching_component)
    unrelated_component[10:110, 110:116] = True
    union = matching_component | unrelated_component

    agreement = _best_model_domain_shape_agreement(
        candidate,
        [union, matching_component, unrelated_component],
    )

    assert agreement["model_shape_variant_index"] == 1
    assert agreement["model_shape_variant_count"] == 3
    assert agreement["cad_shape_iou"] > 0.95


def test_model_domain_reference_keeps_shape_on_model_image(tmp_path: Path) -> None:
    model_rgb = np.full((32, 40, 3), 120, dtype=np.uint8)
    part_ids = np.full((32, 40, 3), 28, dtype=np.uint8)
    red, green, blue = _part_color("P0001")
    part_ids[6:18, 5:15] = (blue, green, red)
    part_ids[22:26, 30:36] = (blue, green, red)
    red_2, green_2, blue_2 = _part_color("P0002")
    part_ids[8:20, 22:28] = (blue_2, green_2, red_2)
    complete = np.zeros((32, 40), dtype=np.uint8)
    complete[5:19, 4:16] = 255
    complete[21:27, 29:37] = 255
    local_reference = np.zeros((32, 40), dtype=np.uint8)
    local_reference[6:18, 5:15] = 255
    model_rgb_path = tmp_path / "model.png"
    part_ids_path = tmp_path / "part_ids.png"
    complete_path = tmp_path / "complete.png"
    local_reference_path = tmp_path / "local.png"
    for path, image in (
        (model_rgb_path, model_rgb),
        (part_ids_path, part_ids),
        (complete_path, complete),
        (local_reference_path, local_reference),
    ):
        assert cv2.imwrite(str(path), image)
    registry = {
        "render_set": {
            "views": [
                {
                    "view_id": "front",
                    "rgb": str(model_rgb_path),
                    "part_ids": str(part_ids_path),
                    "visible_parts": [
                        {"part_id": "P0001", "pixels": 144},
                        {"part_id": "P0002", "pixels": 72},
                    ],
                }
            ]
        }
    }
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    spatial = {
        "view_alignments": [
            {
                "reference_view_id": "front",
                "bbox_affine": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                "ecc_warp": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            }
        ]
    }
    spatial_path = tmp_path / "spatial.json"
    spatial_path.write_text(json.dumps(spatial), encoding="utf-8")
    manifest = {
        "schema_version": "qwen-cad-amodal-part-templates/v1",
        "inputs": {
            "registry": {"path": str(registry_path), "sha256": _sha256(registry_path)},
            "spatial_report": {
                "path": str(spatial_path),
                "sha256": _sha256(spatial_path),
            },
        },
        "records": [
            {
                "view_id": "front",
                "part_id": "P0001",
                "render_view_id": "front",
                "quarter_turns_ccw": 0,
                "raw_amodal_mask": {
                    "path": str(complete_path),
                    "sha256": _sha256(complete_path),
                },
                "aligned_amodal_mask": {"sha256": "a" * 64},
                "modal_visibility_mask": {
                    "path": str(local_reference_path),
                    "sha256": _sha256(local_reference_path),
                },
            }
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    references, _document = _model_domain_shape_references(
        manifest_path=manifest_path,
        expected_keys={("front", "P0001")},
    )

    reference = references[("front", "P0001")]
    assert reference["audit"]["coordinate_domain"] == "cad_model_render_image"
    assert reference["audit"]["model_selected_component_indices"] == [1]
    assert np.count_nonzero(reference["display_visible_shape"]) == 120
    assert len(reference["visible_shape"]) == 1
    assert np.count_nonzero(reference["assembly_neighbor_context"]) > 0
    assert reference["audit"]["assembly_context_role"] == (
        "preserve_target_position_relative_to_other_visible_cad_parts"
    )


def test_tiny_model_mask_registration_fallback_stays_full_resolution() -> None:
    source = np.zeros((512, 512), dtype=bool)
    source[240:242, 300:308] = True
    target = np.zeros((443, 582), dtype=bool)
    target[210:213, 355:366] = True

    registered, audit = _full_resolution_similarity_fallback(source, target)

    assert np.count_nonzero(registered) > 0
    assert audit["method"] == "full_resolution_tiny_mask_similarity_fallback"
    assert audit["downsampling_applied"] is False
    assert np.asarray(audit["affine_2x3"]).shape == (2, 3)


def test_hybrid_replays_sam_shared_camera_template_translation() -> None:
    seed = np.zeros((20, 30), dtype=bool)
    seed[5:10, 7:12] = True
    aligned, audit = _sam_aligned_cad_seed(
        seed,
        {
            "view_shared_alignment": {
                "translation_xy_pixels": [3.0, 2.0],
                "part_specific_translation_allowed": False,
            },
            "box_audits": [
                {
                    "shape_point_refinement": {
                        "accepted": True,
                        "prompt_audit": {"translation_xy_pixels": [3.0, 2.0]},
                    }
                }
            ],
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
            "view_shared_alignment": {
                "translation_xy_pixels": [-2.0, 3.0],
                "part_specific_translation_allowed": False,
            },
            "selected_candidate": {
                "cad_template_alignment": {
                    "translation_xy_pixels": [-2.0, 3.0],
                    "per_mesh_pose_change_allowed": False,
                }
            },
        },
    )

    expected = np.zeros_like(seed)
    expected[8:13, 5:10] = True
    assert np.array_equal(aligned, expected)
    assert audit["source"] == "entityseg_view_shared_camera_projection"
    assert audit["per_mesh_pose_change_allowed"] is False


def test_hybrid_rejects_entity_alignment_that_changes_one_mesh_pose() -> None:
    with pytest.raises(
        ValueError, match="candidate and shared CAD translations differ"
    ):
        _entity_aligned_cad_seed(
            np.ones((4, 4), dtype=bool),
            {
                "view_shared_alignment": {
                    "translation_xy_pixels": [1.0, 2.0],
                    "part_specific_translation_allowed": False,
                },
                "selected_candidate": {
                    "cad_template_alignment": {
                        "translation_xy_pixels": [1.0, 2.0],
                        "per_mesh_pose_change_allowed": True,
                    }
                },
            },
        )


def test_rejected_candidates_still_replay_the_view_shared_camera_translation() -> None:
    seed = np.zeros((20, 30), dtype=bool)
    seed[5:10, 7:12] = True
    shared = {
        "translation_xy_pixels": [-1.0, 2.0],
        "part_specific_translation_allowed": False,
    }

    sam_aligned, _sam_audit = _sam_aligned_cad_seed(
        seed,
        {"view_shared_alignment": shared, "box_audits": [], "accepted": False},
    )
    entity_aligned, _entity_audit = _entity_aligned_cad_seed(
        seed,
        {
            "view_shared_alignment": shared,
            "selected_candidate": None,
            "accepted": False,
        },
    )

    expected = np.zeros_like(seed)
    expected[7:12, 6:11] = True
    assert np.array_equal(sam_aligned, expected)
    assert np.array_equal(entity_aligned, expected)


def test_relation_location_excludes_the_targets_own_bad_first_pass_mask() -> None:
    target = np.zeros((120, 160), dtype=bool)
    target[45:55, 72:88] = True
    linear = np.asarray([[1.1, -0.2], [0.2, 1.1]], dtype=np.float64)
    translation = np.asarray([14.0, -7.0], dtype=np.float64)
    model_points = {
        "P0001": np.asarray([20.0, 20.0]),
        "P0002": np.asarray([130.0, 22.0]),
        "P0003": np.asarray([25.0, 95.0]),
        "P0004": np.asarray([135.0, 98.0]),
        "P0099": np.asarray([80.0, 50.0]),
    }
    observations = []
    for part_id, point in model_points.items():
        photo = linear @ point + translation
        if part_id == "P0099":
            photo = np.asarray([5.0, 115.0])  # deliberately wrong target mask
        observations.append(
            {
                "part_id": part_id,
                "model_centroid_xy": point,
                "photo_centroid_xy": photo,
                "photo_bbox_diagonal": 20.0,
                "mask_pixels": 400,
                "quality": 1.0,
            }
        )

    affine, audit = _infer_target_affine(
        target_part_id="P0099",
        target_shape=target,
        observations=observations,
    )
    inferred = affine[:, :2] @ np.asarray([79.5, 49.5]) + affine[:, 2]
    expected = linear @ np.asarray([79.5, 49.5]) + translation

    assert np.linalg.norm(inferred - expected) < 1e-6
    assert audit["target_first_pass_mask_used"] is False
    assert "P0099" not in audit["inlier_anchor_part_ids"]
    assert audit["inlier_anchor_count"] == 4


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
    request_document = {"schema_version": "synthetic-request/v1"}
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(request_document), encoding="utf-8")
    request_binding = {
        "path": str(request_path),
        "sha256": _sha256(request_path),
        "document_sha256": _canonical_sha256(request_document),
    }
    sam = {
        "schema_version": "qwen-sam3-region-result/v1",
        "request": request_binding,
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
