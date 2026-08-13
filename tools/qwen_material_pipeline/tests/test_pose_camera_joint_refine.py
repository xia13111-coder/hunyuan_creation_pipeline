from __future__ import annotations

import numpy as np
import pytest

from qwen_material_pipeline.evidence.pose_camera_joint_refine import (
    BASELINE_SCALES,
    PARAMETER_NAMES,
    full_resolution_spec,
    joint_candidate_spec,
    joint_directions,
    select_final_candidate,
)


def _seed() -> dict:
    return {
        "analysis_direction": [0.0, -1.0, 0.0],
        "analysis_up_axis": [0.0, 0.0, 1.0],
        "focal_length_mm": 45.0,
        "distance_multiplier": 2.15,
        "target_offset_u": 0.0,
        "target_offset_v": 0.0,
        "principal_point_u": 0.0,
        "principal_point_v": 0.0,
        "radial_distortion_k1": 0.0,
        "radial_distortion_k2": 0.0,
        "projection_mode": "perspective",
    }


def _record(
    *,
    view_id: str,
    kind: str,
    iou: float,
    boundary: float,
    score: float,
    anchor: bool = False,
    consensus: bool = True,
) -> dict:
    return {
        "view_id": view_id,
        "projection_iou": iou,
        "boundary_p95_px": boundary,
        "score": score,
        "rigid_consensus_valid": consensus,
        "calibration": {
            "frame_anchor": anchor,
            "lineage": {"kind": kind},
        },
    }


def test_joint_directions_are_dense_deterministic_and_antithetic() -> None:
    first = joint_directions(count=4, sequence_offset=7)
    second = joint_directions(count=4, sequence_offset=7)
    assert len(first) == 4
    assert all(raw.shape == (len(PARAMETER_NAMES),) for raw in first)
    assert all(np.count_nonzero(np.abs(raw) > 0.0) == len(PARAMETER_NAMES) for raw in first)
    assert np.allclose(first[0], -first[1])
    assert np.allclose(first[2], -first[3])
    assert all(np.array_equal(a, b) for a, b in zip(first, second, strict=True))


def test_joint_candidate_couples_whole_pose_and_intrinsics() -> None:
    vector = joint_directions(count=1)[0]
    result = joint_candidate_spec(
        seed=_seed(),
        reference_id="side",
        candidate_id="candidate",
        vector=vector,
        scales=BASELINE_SCALES,
        round_index=1,
        lineage={"kind": "sealed_baseline", "rank": 0},
    )
    assert result["analysis_direction"] != _seed()["analysis_direction"]
    assert result["analysis_up_axis"] != _seed()["analysis_up_axis"]
    assert result["distance_multiplier"] != 2.15
    assert result["target_offset_u"] != 0.0
    assert result["focal_length_mm"] != 45.0
    assert result["principal_point_u"] != 0.0
    assert result["radial_distortion_k1"] != 0.0
    calibration = result["calibration"]
    assert calibration["whole_asset_se3_equivalent_only"] is True
    assert calibration["camera_intrinsics_jointly_optimized"] is True
    assert calibration["per_mesh_or_subtree_transform_applied"] is False
    assert "prim_path" not in result
    assert "mesh" not in result


def test_exact_joint_candidate_preserves_seed_camera() -> None:
    result = joint_candidate_spec(
        seed=_seed(),
        reference_id="front",
        candidate_id="baseline",
        vector=None,
        scales=BASELINE_SCALES,
        round_index=1,
        lineage={"kind": "sealed_baseline", "rank": 0},
        frame_anchor=True,
    )
    assert np.allclose(result["analysis_direction"], _seed()["analysis_direction"])
    assert np.allclose(result["analysis_up_axis"], _seed()["analysis_up_axis"])
    assert result["focal_length_mm"] == 45.0
    assert result["distance_multiplier"] == 2.15
    assert result["calibration"]["exact_start"] is True
    assert result["calibration"]["frame_anchor"] is True


def test_exact_top_camera_does_not_acquire_false_polar_rotation() -> None:
    seed = {
        **_seed(),
        "analysis_direction": [2.5e-17, 5.5e-17, 1.0],
        "analysis_up_axis": [1.0, 0.0, -2.5e-17],
    }
    result = joint_candidate_spec(
        seed=seed,
        reference_id="top",
        candidate_id="baseline_top",
        vector=None,
        scales=BASELINE_SCALES,
        round_index=1,
        lineage={"kind": "sealed_baseline", "rank": 0},
        frame_anchor=True,
    )
    assert np.allclose(result["analysis_direction"], seed["analysis_direction"], atol=1e-15)
    assert np.allclose(result["analysis_up_axis"], seed["analysis_up_axis"], atol=1e-15)


def test_select_final_candidate_accepts_only_nonregressive_improvement() -> None:
    baseline = _record(
        view_id="baseline",
        kind="sealed_baseline",
        iou=0.88,
        boundary=17.5,
        score=0.61,
        anchor=True,
    )
    better = _record(
        view_id="better",
        kind="gigapose",
        iou=0.881,
        boundary=17.4,
        score=0.63,
    )
    decision, selected = select_final_candidate([baseline, better])
    assert decision == "JOINT_REFINEMENT_ACCEPTED"
    assert selected["view_id"] == "better"


@pytest.mark.parametrize(
    "candidate",
    [
        _record(view_id="bad_iou", kind="gigapose", iou=0.877, boundary=17.4, score=0.70),
        _record(view_id="bad_boundary", kind="gigapose", iou=0.881, boundary=18.1, score=0.70),
        _record(view_id="no_gain", kind="gigapose", iou=0.881, boundary=17.4, score=0.6105),
        _record(view_id="no_consensus", kind="gigapose", iou=0.881, boundary=17.4, score=0.70, consensus=False),
    ],
)
def test_select_final_candidate_falls_back_on_regression(candidate: dict) -> None:
    baseline = _record(
        view_id="baseline",
        kind="sealed_baseline",
        iou=0.88,
        boundary=17.5,
        score=0.61,
        anchor=True,
    )
    decision, selected = select_final_candidate([baseline, candidate])
    assert decision == "BASELINE_RETAINED"
    assert selected["view_id"] == "baseline"


def test_select_final_candidate_requires_one_anchor() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        select_final_candidate([])


def test_full_resolution_spec_discards_search_pixel_anchor() -> None:
    score = {
        **_seed(),
        "view_id": "search",
        "calibration": {
            "reference_view_id": "side",
            "lineage": {"kind": "sealed_baseline"},
            "frame_anchor": True,
            "frame_anchor_affine": [[1.0, 0.0, 12.0], [0.0, 1.0, -4.0]],
        },
    }
    result = full_resolution_spec(score, view_id="final_side", mark_anchor=True)
    assert result["calibration"]["frame_anchor"] is True
    assert "frame_anchor_affine" not in result["calibration"]
    assert result["calibration"]["lineage"] == {"kind": "sealed_baseline"}
