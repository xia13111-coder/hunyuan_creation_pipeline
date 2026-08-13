from __future__ import annotations

import copy

import numpy as np
import pytest

from qwen_material_pipeline.evidence.pose_model_camera_seed import (
    MODEL_REVISION,
    baseline_camera_spec,
    pose_to_camera_spec,
    select_verified_seed,
    validate_proposals,
)


def _baseline() -> dict:
    return {
        "view_id": "side",
        "analysis_direction": [0.0, -1.0, 0.0],
        "analysis_up_axis": [0.0, 0.0, 1.0],
        "focal_length_mm": 44.55,
        "distance_multiplier": 1.92,
        "target_offset_u": -0.04,
        "target_offset_v": -0.003,
        "projection_mode": "perspective",
        "orthographic_span_multiplier": 2.0,
        "calibration": {"old": "sealed"},
    }


def _proposals() -> dict:
    return {
        "schema_version": "qwen-rigid-pose-model-proposals/v1",
        "model": {"name": "GigaPose", "repository_revision": MODEL_REVISION},
        "views": [
            {
                "view_id": "side",
                "candidates": [
                    {
                        "rank": 1,
                        "model_score": 0.8,
                        "object_to_camera_rotation": [
                            [1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0],
                        ],
                        "object_to_camera_translation": [0.0, 0.0, 2.0],
                        "template_view_id": 7,
                    }
                ],
            }
        ],
    }


def test_validates_exact_top_k_view_contract() -> None:
    assert validate_proposals(_proposals(), expected_view_ids=["side"])["side"][0][
        "rank"
    ] == 1
    missing = _proposals()
    missing["views"][0]["view_id"] = "front"
    with pytest.raises(ValueError, match="exactly match"):
        validate_proposals(missing, expected_view_ids=["side"])


@pytest.mark.parametrize(
    "update, message",
    [
        (("model", "repository_revision"), "unsealed"),
        (("views", 0, "candidates", 0, "object_to_camera_translation"), "translation"),
        (("views", 0, "candidates", 0, "object_to_camera_rotation"), "orthonormal"),
    ],
)
def test_proposal_contract_fails_closed(update, message) -> None:
    document = _proposals()
    cursor = document
    for key in update[:-1]:
        cursor = cursor[key]
    if update[-1] == "repository_revision":
        cursor[update[-1]] = "different"
    elif update[-1] == "object_to_camera_translation":
        cursor[update[-1]] = [0.0, 0.0, -1.0]
    else:
        cursor[update[-1]] = [[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    with pytest.raises(ValueError, match=message):
        validate_proposals(document, expected_view_ids=["side"])


def test_pose_is_inverted_into_one_camera_without_mesh_transform() -> None:
    candidate = _proposals()["views"][0]["candidates"][0]
    spec = pose_to_camera_spec(
        baseline=_baseline(), view_id="side", candidate=candidate
    )
    assert spec["view_id"] == "pose_side_01"
    assert spec["analysis_direction"] == pytest.approx([0.0, 0.0, -1.0])
    assert spec["analysis_up_axis"] == pytest.approx([0.0, -1.0, 0.0])
    assert spec["focal_length_mm"] == 44.55
    assert spec["distance_multiplier"] == 1.92
    assert "world_translation" not in spec
    assert "part_id" not in spec


def test_baseline_is_the_only_anchor_and_learned_pose_is_advisory() -> None:
    baseline = baseline_camera_spec(baseline=_baseline(), view_id="side")
    learned = pose_to_camera_spec(
        baseline=_baseline(),
        view_id="side",
        candidate=_proposals()["views"][0]["candidates"][0],
    )
    assert baseline["calibration"]["frame_anchor"] is True
    assert learned["calibration"]["frame_anchor"] is False


def _score(rank: int, *, objective: float, iou: float, boundary: float) -> dict:
    return {
        "view_id": f"candidate_{rank}",
        "score": objective,
        "projection_iou": iou,
        "boundary_p95_px": boundary,
        "rigid_consensus_valid": True,
        "rigid_consensus_score": 0.8,
        "calibration": {"proposal_rank": rank},
    }


def test_true_render_gate_accepts_only_nonregressive_objective_gain() -> None:
    baseline = _score(0, objective=0.70, iou=0.87, boundary=18.0)
    model_high_confidence_but_bad = _score(
        1, objective=0.90, iou=0.84, boundary=22.0
    )
    verified = _score(2, objective=0.73, iou=0.871, boundary=17.4)

    decision, winner = select_verified_seed(
        [baseline, model_high_confidence_but_bad, verified]
    )
    assert decision == "POSE_MODEL_ACCEPTED"
    assert winner["calibration"]["proposal_rank"] == 2


def test_true_render_gate_retains_baseline_when_model_does_not_help() -> None:
    baseline = _score(0, objective=0.70, iou=0.87, boundary=18.0)
    bad = _score(1, objective=0.80, iou=0.84, boundary=18.0)
    decision, winner = select_verified_seed([baseline, bad])
    assert decision == "BASELINE_RETAINED"
    assert winner is not bad
