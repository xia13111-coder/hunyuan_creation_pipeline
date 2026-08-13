from __future__ import annotations

import pytest

from qwen_material_pipeline.evidence.whole_asset_pose import (
    phase_one_pose_candidates,
    registered_asset_root_path,
    registry_bounds,
    select_whole_asset_pose,
)


def _registry() -> dict:
    return {
        "parts": [
            {
                "part_id": "P1",
                "prim_path": "/Wrapper/Workpiece/A/Mesh",
                "world_bbox": [[-1.0, -2.0, -3.0], [0.0, 1.0, 2.0]],
            },
            {
                "part_id": "P2",
                "prim_path": "/Wrapper/Workpiece/B/Mesh",
                "world_bbox": [[0.0, -1.0, -2.0], [3.0, 4.0, 5.0]],
            },
        ]
    }


def test_whole_asset_root_and_bounds_cover_every_registered_part() -> None:
    registry = _registry()

    assert registered_asset_root_path(registry) == "/Wrapper/Workpiece"
    assert registry_bounds(registry) == {
        "minimum": [-1.0, -2.0, -3.0],
        "maximum": [3.0, 4.0, 5.0],
        "center": [1.0, 1.0, 1.0],
        "diagonal": pytest.approx((4.0**2 + 6.0**2 + 8.0**2) ** 0.5),
    }


def test_phase_one_candidates_are_only_whole_asset_translation_or_rotation() -> None:
    candidates = phase_one_pose_candidates(
        asset_root="/Wrapper/Workpiece", asset_diagonal=10.0
    )

    assert len(candidates) == 12
    assert len({raw["candidate_id"] for raw in candidates}) == 12
    assert all(raw["asset_root_prim_path"] == "/Wrapper/Workpiece" for raw in candidates)
    assert all(raw["pivot"] == "asset_bounds_center" for raw in candidates)
    assert all(
        (sum(value != 0.0 for value in raw["world_translation"]) == 1)
        != (
            sum(
                value != 0.0
                for value in raw["world_rotation_rotvec_degrees"]
            )
            == 1
        )
        for raw in candidates
    )


def test_joint_selector_accepts_only_multiview_nonregressive_gain() -> None:
    baseline = {
        "aggregate": {
            "mean_projection_iou": 0.8,
            "mean_mismatch_over_union": 0.2,
        }
    }
    bad_single_view = {
        "candidate_id": "bad",
        "aggregate": {
            "mean_projection_iou": 0.83,
            "mean_mismatch_over_union": 0.17,
            "mean_boundary_p95_px": 5.0,
        },
        "worst_view_iou_regression": 0.01,
    }
    good = {
        "candidate_id": "good",
        "aggregate": {
            "mean_projection_iou": 0.81,
            "mean_mismatch_over_union": 0.19,
            "mean_boundary_p95_px": 6.0,
        },
        "worst_view_iou_regression": 0.002,
    }

    decision, winner = select_whole_asset_pose(
        baseline=baseline, candidates=[bad_single_view, good]
    )

    assert decision == "OPTIMIZED"
    assert winner is good


def test_joint_selector_returns_noop_instead_of_overfitting_one_view() -> None:
    baseline = {
        "aggregate": {
            "mean_projection_iou": 0.8,
            "mean_mismatch_over_union": 0.2,
        }
    }
    candidate = {
        "candidate_id": "side_only",
        "aggregate": {
            "mean_projection_iou": 0.82,
            "mean_mismatch_over_union": 0.18,
            "mean_boundary_p95_px": 5.0,
        },
        "worst_view_iou_regression": 0.02,
    }

    decision, winner = select_whole_asset_pose(
        baseline=baseline, candidates=[candidate]
    )

    assert decision == "NO_OP"
    assert winner is None


def test_whole_asset_root_rejects_duplicate_or_disjoint_registered_parts() -> None:
    duplicate = _registry()
    duplicate["parts"][1]["prim_path"] = duplicate["parts"][0]["prim_path"]
    with pytest.raises(ValueError, match="unique"):
        registered_asset_root_path(duplicate)

    disjoint = _registry()
    disjoint["parts"][1]["prim_path"] = "/Other/Mesh"
    with pytest.raises(ValueError, match="share"):
        registered_asset_root_path(disjoint)
