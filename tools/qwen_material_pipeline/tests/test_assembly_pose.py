from __future__ import annotations

import numpy as np
import pytest

from qwen_material_pipeline.evidence.assembly_pose import (
    _largest_matching_residual_component,
    _candidate_assembly_subtrees,
    _mask_metrics,
    _regional_overlap_metrics,
    _regional_support_bounds,
    _select_pose_winner,
    candidate_translations,
    discover_nested_residual_xform,
    discover_assembly_candidate,
    search_nested_screen_translation,
    solve_world_translation_from_render_responses,
)


def test_discovers_largest_generic_coherent_assembly() -> None:
    report = {
        "views": [
            {
                "reference_view_id": "side",
                "final": {
                    "assembly_residual_clusters": [
                        {
                            "assembly_subtree": "/Asset/Small",
                            "part_count": 2,
                            "part_ids": ["P1", "P2"],
                            "projected_pixels": 50,
                            "median_residual_px": 20,
                        "residual_direction_coherence": 1.0,
                            "minimum_inside_reference_ratio": 0.0,
                            "classification": "assembly_state_or_geometry_mismatch",
                        },
                        {
                            "assembly_subtree": "/Asset/Large",
                            "part_count": 3,
                            "part_ids": ["P3", "P4", "P5"],
                            "projected_pixels": 500,
                            "median_residual_px": 18,
                            "residual_direction_coherence": 0.9,
                            "minimum_inside_reference_ratio": 0.0,
                            "classification": "assembly_state_or_geometry_mismatch",
                        },
                    ]
                },
            }
        ]
    }

    assert discover_assembly_candidate(report=report, reference_view_id="side")[
        "assembly_subtree"
    ] == "/Asset/Large"


def test_rejects_noncoherent_or_single_part_residual() -> None:
    report = {
        "views": [
            {
                "reference_view_id": "side",
                "final": {
                    "assembly_residual_clusters": [
                        {
                            "assembly_subtree": "/Asset/Part",
                            "part_count": 1,
                            "projected_pixels": 500,
                            "residual_direction_coherence": 1.0,
                            "minimum_inside_reference_ratio": 0.0,
                            "classification": "assembly_state_or_geometry_mismatch",
                        }
                    ]
                },
            }
        ]
    }

    with pytest.raises(ValueError, match="No coherent assembly residual"):
        discover_assembly_candidate(report=report, reference_view_id="side")


def test_residual_component_matches_area_and_distance() -> None:
    cluster = np.zeros((100, 100), np.uint8)
    cluster[10:30, 10:30] = 1
    residual = np.zeros_like(cluster)
    residual[40:60, 10:30] = 1
    residual[70:72, 70:72] = 1

    selected = _largest_matching_residual_component(
        reference_only=residual, cluster=cluster
    )

    assert int(selected.sum()) == 400
    assert selected[50, 15] == 1


def test_local_mask_objective_rewards_actual_overlap() -> None:
    target = np.zeros((80, 80), np.uint8)
    target[20:40, 20:40] = 1
    missed = np.zeros_like(target)
    missed[45:65, 20:40] = 1
    aligned = target.copy()

    assert _mask_metrics(target, aligned)["objective"] > _mask_metrics(
        target, missed
    )["objective"]
    assert _mask_metrics(target, aligned)["iou"] == 1.0


def test_candidate_grid_is_finite_and_bounded_around_seed() -> None:
    values = candidate_translations([0.02, -0.01, 0.0], [0.0, 0.0, -0.1])

    assert len(values) == 9
    assert all(np.isfinite(raw).all() for raw in values)
    assert max(np.linalg.norm(raw) for raw in values) < 0.2


def test_candidate_subtrees_include_one_bounded_parent() -> None:
    registry = {
        "parts": [
            {
                "part_id": f"P{index:04d}",
                "prim_path": (
                    f"/Asset/Mechanism/Branch{index % 3}/Mesh{index}"
                ),
            }
            for index in range(12)
        ]
        + [
            {
                "part_id": f"Q{index:04d}",
                "prim_path": f"/Asset/Other/Mesh{index}",
            }
            for index in range(48)
        ]
    }

    candidates = _candidate_assembly_subtrees(
        registry=registry,
        detected_subtree="/Asset/Mechanism/Branch0",
    )

    assert [raw[0] for raw in candidates] == [
        "/Asset/Mechanism/Branch0",
        "/Asset/Mechanism",
    ]
    assert len(candidates[0][1]) == 4
    assert len(candidates[1][1]) == 12


def test_regional_support_scores_all_nearby_sibling_residuals() -> None:
    reference = np.zeros((100, 100), np.uint8)
    reference[50:70, 30:50] = 1
    reference[35:45, 35:45] = 1
    cluster = np.zeros_like(reference)
    cluster[20:40, 30:50] = 1
    target = np.zeros_like(reference)
    target[50:70, 30:50] = 1
    leaf_only = reference.copy()
    leaf_only[35:45, 35:45] = 0

    bounds = _regional_support_bounds(
        registered_cluster=cluster,
        target_component=target,
    )
    incomplete = _regional_overlap_metrics(
        reference_mask=reference,
        registered_foreground=leaf_only,
        support_bounds_xyxy=bounds,
    )
    complete = _regional_overlap_metrics(
        reference_mask=reference,
        registered_foreground=reference,
        support_bounds_xyxy=bounds,
    )

    assert complete["iou"] == 1.0
    assert incomplete["reference_only_pixels"] == 100
    assert complete["mismatch_pixels"] == 0


def test_regional_residual_selects_complete_parent_over_better_leaf_anchor() -> None:
    baseline_local = {"objective": 0.1, "centroid_error_px": 30.0}
    baseline_regional = {"iou": 0.3}
    common = {
        "translation_norm": 0.1,
        "global": {
            "projection_iou": 0.92,
            "fixed_geometry_consensus_residual_px": 1.0,
        },
    }
    leaf = {
        **common,
        "candidate_id": "leaf",
        "assembly_subtree": "/Asset/Mechanism/Branch",
        "subtree_member_part_ids": ["P1", "P2"],
        "local": {"objective": 0.8, "centroid_error_px": 1.0},
        "regional": {"iou": 0.65},
    }
    parent = {
        **common,
        "candidate_id": "parent",
        "assembly_subtree": "/Asset/Mechanism",
        "subtree_member_part_ids": ["P1", "P2", "P3", "P4"],
        "local": {"objective": 0.7, "centroid_error_px": 2.0},
        "regional": {"iou": 0.78},
    }

    winner = _select_pose_winner(
        candidates=[leaf, parent],
        baseline_local=baseline_local,
        baseline_regional=baseline_regional,
        baseline_projection_iou=0.91,
        baseline_fixed_by_subtree={
            "/Asset/Mechanism/Branch": 1.0,
            "/Asset/Mechanism": 1.0,
        },
    )

    assert winner["candidate_id"] == "parent"


def test_discovers_nested_residual_without_part_id_or_asset_name() -> None:
    audit = {
        "cad_part_residual_attribution": [
            {
                "part_id": "arbitrary-a",
                "prim_path": "/Asset/Mechanism/Lever/Mesh",
                "projected_pixels": 50,
                "cad_only_pixels": 30,
                "inside_reference_ratio": 0.4,
            },
            {
                "part_id": "arbitrary-b",
                "prim_path": "/Asset/Other/Mesh",
                "projected_pixels": 100,
                "cad_only_pixels": 100,
                "inside_reference_ratio": 0.0,
            },
        ]
    }

    selected = discover_nested_residual_xform(
        residual_audit=audit, parent_subtree="/Asset/Mechanism"
    )

    assert selected["part_id"] == "arbitrary-a"
    assert selected["assembly_subtree"] == "/Asset/Mechanism/Lever"


def test_nested_screen_search_uses_full_regional_mismatch() -> None:
    reference = np.zeros((80, 80), np.uint8)
    reference[20:60, 40:60] = 1
    reference[45:48, 20:40] = 1
    part = np.zeros_like(reference)
    part[30:33, 15:35] = 1
    foreground = reference.copy()
    foreground[45:48, 20:40] = 0
    foreground |= part

    result = search_nested_screen_translation(
        reference_mask=reference,
        registered_foreground=foreground,
        registered_part=part,
        support_bounds_xyxy=[10, 10, 70, 70],
        maximum_shift_px=20,
    )

    assert result["requested_integer_translation_px"] == [5, 15]
    assert result["regional_mismatch_pixels"] == 0


def test_render_response_jacobian_lifts_screen_shift_to_world() -> None:
    result = solve_world_translation_from_render_responses(
        target_screen_translation_px=[7.0, -14.0],
        probe_world_translations=[[0.1, 0.0, 0.0], [0.0, 0.0, 0.2]],
        measured_screen_translations_px=[[10.0, 2.0], [4.0, -20.0]],
    )

    coefficients = np.asarray(result["probe_coefficients"])
    assert np.allclose(
        np.asarray([[10.0, 4.0], [2.0, -20.0]]) @ coefficients,
        [7.0, -14.0],
    )
    assert np.allclose(
        result["world_translation"],
        coefficients @ np.asarray([[0.1, 0.0, 0.0], [0.0, 0.0, 0.2]]),
    )
