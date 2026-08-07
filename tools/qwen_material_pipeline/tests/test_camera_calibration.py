from __future__ import annotations

import math
import json
import os
import subprocess
from pathlib import Path

import cv2
import numpy as np

import qwen_material_pipeline.evidence.camera_calibration as camera_calibration
from qwen_material_pipeline.evidence.camera_calibration import (
    CAMERA_PHASES,
    _angles,
    _alignment_candidate_sort_key,
    _boundary_metrics,
    _candidate_specs,
    _component_balanced_reference_metrics,
    _direction,
    _deterministic_part_id_foreground,
    _global_finalists,
    _merge_registry,
    _part_balanced_structure_metrics,
    _reference_masks,
    _residual_components,
    _seal_full_resolution_winners,
    _seed_by_view_specs,
    _silhouette_coverage_metrics,
    _spatial_balanced_reference_metrics,
)
from qwen_material_pipeline.evidence.spatial import _part_color


def test_camera_render_retries_cleanly_after_transient_isaac_startup_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output = tmp_path / "renders"
    output.mkdir()
    (output / "stale.txt").write_text("partial", encoding="utf-8")
    calls = 0

    def fake_run(command, *, check, env):
        nonlocal calls
        assert check is True
        assert Path(env["PYTHONPATH"].split(os.pathsep)[0]) == (
            Path(camera_calibration.__file__).resolve().parents[2]
        )
        calls += 1
        if calls == 1:
            raise subprocess.CalledProcessError(1, command)
        destination = Path(command[command.index("--output-dir") + 1])
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "part_registry.rendered.json").write_text(
            "{}", encoding="utf-8"
        )

    monkeypatch.setattr(camera_calibration.subprocess, "run", fake_run)
    monkeypatch.setattr(camera_calibration.time, "sleep", lambda _seconds: None)

    rendered = camera_calibration._run_render(
        isaac_python=tmp_path / "isaac-python.sh",
        registry=tmp_path / "registry.json",
        output_dir=output,
        view_specs=tmp_path / "views.json",
        resolution=512,
        rt_subframes=4,
        analysis_up_axis="z",
        analysis_front_axis="-y",
    )

    assert calls == 2
    assert rendered == (output / "part_registry.rendered.json").resolve()
    assert not (output / "stale.txt").exists()


def test_camera_phase_checkpoint_reuse_requires_exact_candidate_specs(
    tmp_path: Path,
) -> None:
    specs = {
        "schema_version": "qwen-camera-view-specs/v1",
        "views": [{"view_id": "cal_front_micro_000"}],
    }
    specs_path = tmp_path / "micro_view_specs.json"
    scores_path = tmp_path / "micro_scores.json"
    specs_path.write_text(json.dumps(specs), encoding="utf-8")
    scores_path.write_text(
        json.dumps(
            {
                "schema_version": camera_calibration.SCHEMA_VERSION,
                "reference_view_id": "front",
                "phase": "micro",
                "winner": {"view_id": "cal_front_micro_000"},
                "candidates": [{"view_id": "cal_front_micro_000"}],
            }
        ),
        encoding="utf-8",
    )

    reused = camera_calibration._completed_phase(
        specs_path=specs_path,
        scores_path=scores_path,
        expected_specs=specs,
        reference_id="front",
        phase="micro",
    )
    assert reused is not None
    assert reused[0]["view_id"] == "cal_front_micro_000"

    changed = {**specs, "views": [{"view_id": "cal_front_micro_001"}]}
    assert (
        camera_calibration._completed_phase(
            specs_path=specs_path,
            scores_path=scores_path,
            expected_specs=changed,
            reference_id="front",
            phase="micro",
        )
        is None
    )


def test_direction_angle_round_trip_is_continuous() -> None:
    for azimuth, elevation in ((0.0, 15.0), (137.25, 23.5), (359.5, 81.0)):
        recovered_azimuth, recovered_elevation = _angles(_direction(azimuth, elevation))
        assert math.isclose(recovered_azimuth, azimuth, abs_tol=1e-9)
        assert math.isclose(recovered_elevation, elevation, abs_tol=1e-9)


def test_fine_camera_candidates_preserve_one_whole_asset_camera() -> None:
    specs = _candidate_specs(
        reference_id="iso",
        seed={
            "analysis_direction": _direction(135.0, 15.0),
            "focal_length_mm": 45.0,
            "distance_multiplier": 2.15,
        },
        phase="fine",
    )

    assert len(specs["views"]) == 27
    assert all(
        view["calibration"]["reference_view_id"] == "iso" for view in specs["views"]
    )
    assert all("part" not in view for view in specs["views"])


def test_settle_camera_candidates_expand_distance_and_reduce_angle_step() -> None:
    specs = _candidate_specs(
        reference_id="top",
        seed={
            "analysis_direction": _direction(180.0, 90.0),
            "analysis_up_axis": [1.0, 0.0, 0.0],
            "focal_length_mm": 45.0,
            "distance_multiplier": 2.45,
        },
        phase="settle",
    )

    distances = {view["calibration"]["distance_multiplier"] for view in specs["views"]}
    azimuths = {view["calibration"]["azimuth_degrees"] for view in specs["views"]}
    assert len(specs["views"]) == 27
    assert distances == {1.85, 2.45, 3.05}
    assert azimuths == {178.5, 180.0, 181.5}


def test_lens_camera_candidates_optimize_focal_length_and_distance() -> None:
    specs = _candidate_specs(
        reference_id="front",
        seed={
            "analysis_direction": _direction(0.0, 10.0),
            "analysis_up_axis": [0.0, 0.0, 1.0],
            "focal_length_mm": 48.0,
            "distance_multiplier": 2.2,
        },
        phase="lens",
    )

    assert len(specs["views"]) == 25
    assert {view["focal_length_mm"] for view in specs["views"]} == {
        38.4,
        43.2,
        48.0,
        52.8,
        60.0,
    }
    assert {view["distance_multiplier"] for view in specs["views"]} == {
        1.6,
        1.9,
        2.2,
        2.5,
        2.8,
    }


def test_micro_camera_candidates_refine_pose_and_distance() -> None:
    specs = _candidate_specs(
        reference_id="front",
        seed={
            "analysis_direction": _direction(12.0, 8.0),
            "analysis_up_axis": [0.0, 0.0, 1.0],
            "focal_length_mm": 48.0,
            "distance_multiplier": 2.2,
        },
        phase="micro",
    )

    assert len(specs["views"]) == 27
    assert {view["distance_multiplier"] for view in specs["views"]} == {2.05, 2.2, 2.35}
    assert {view["focal_length_mm"] for view in specs["views"]} == {48.0}


def test_camera_phases_finish_with_sub_tenth_degree_refinement() -> None:
    assert CAMERA_PHASES.index("fine") < CAMERA_PHASES.index("perspective")
    assert CAMERA_PHASES.index("perspective") < CAMERA_PHASES.index("component_pose")
    assert CAMERA_PHASES.index("component_pose") < CAMERA_PHASES.index(
        "perspective_recheck"
    )
    assert CAMERA_PHASES.index("perspective_recheck") < CAMERA_PHASES.index(
        "component_pose_recheck"
    )
    assert CAMERA_PHASES.index("component_pose_recheck") < CAMERA_PHASES.index(
        "orthographic"
    )
    assert CAMERA_PHASES.index("orthographic") < CAMERA_PHASES.index("settle")
    assert CAMERA_PHASES.index("component_pose_recheck") < CAMERA_PHASES.index(
        "settle"
    )
    assert CAMERA_PHASES.index("perspective") < CAMERA_PHASES.index("settle")
    assert CAMERA_PHASES.index("settle") < CAMERA_PHASES.index("micro")
    assert CAMERA_PHASES.index("target") < CAMERA_PHASES.index("lens_micro")
    assert CAMERA_PHASES[-4:] == (
        "nano",
        "target_micro",
        "pico",
        "target_pico",
    )


def test_lens_micro_candidates_refine_perspective_ratio() -> None:
    specs = _candidate_specs(
        reference_id="iso",
        seed={
            "analysis_direction": _direction(120.0, 24.0),
            "analysis_up_axis": [0.0, 0.0, 1.0],
            "focal_length_mm": 50.0,
            "distance_multiplier": 2.4,
        },
        phase="lens_micro",
    )

    assert len(specs["views"]) == 25
    assert {view["focal_length_mm"] for view in specs["views"]} == {
        49.0,
        49.5,
        50.0,
        50.5,
        51.0,
    }
    assert {view["distance_multiplier"] for view in specs["views"]} == {
        2.32,
        2.36,
        2.4,
        2.44,
        2.48,
    }


def test_lens_micro_preserves_weak_perspective_focal_range() -> None:
    specs = _candidate_specs(
        reference_id="side",
        seed={
            "analysis_direction": _direction(0.0, -4.0),
            "analysis_up_axis": [0.0, 0.0, 1.0],
            "focal_length_mm": 891.0,
            "distance_multiplier": 38.7,
        },
        phase="lens_micro",
    )

    assert {view["focal_length_mm"] for view in specs["views"]} == {
        873.18,
        882.09,
        891.0,
        899.91,
        908.82,
    }


def test_perspective_phase_searches_weak_perspective_distance() -> None:
    specs = _candidate_specs(
        reference_id="top",
        seed={
            "analysis_direction": _direction(165.0, 90.0),
            "analysis_up_axis": [1.0, 0.0, 0.0],
            "focal_length_mm": 45.0,
            "distance_multiplier": 2.75,
        },
        phase="perspective",
    )

    assert len(specs["views"]) == 11
    assert {view["distance_multiplier"] for view in specs["views"]} == {
        1.65,
        2.2,
        2.75,
        4.125,
        5.5,
        8.25,
        11.0,
        16.5,
        22.0,
        33.0,
        44.0,
    }
    assert {
        (view["distance_multiplier"], view["focal_length_mm"])
        for view in specs["views"]
    } == {
        (1.65, 27.0),
        (2.2, 36.0),
        (2.75, 45.0),
        (4.125, 67.5),
        (5.5, 90.0),
        (8.25, 135.0),
        (11.0, 180.0),
        (16.5, 270.0),
        (22.0, 360.0),
        (33.0, 540.0),
        (44.0, 720.0),
    }


def test_orthographic_phase_compares_true_projection_without_editing_asset() -> None:
    specs = _candidate_specs(
        reference_id="side",
        seed={
            "analysis_direction": _direction(1.25, -1.6),
            "analysis_up_axis": [0.0, 0.0, 1.0],
            "focal_length_mm": 891.0,
            "distance_multiplier": 38.5,
            "projection_mode": "perspective",
        },
        phase="orthographic",
    )

    assert len(specs["views"]) == 18
    assert {view["projection_mode"] for view in specs["views"]} == {
        "perspective",
        "orthographic",
    }
    assert sum(
        view["calibration"]["projection_mode"] == "perspective"
        for view in specs["views"]
    ) == 9
    assert all("part" not in view for view in specs["views"])


def test_component_pose_reopens_bounded_rigid_camera_neighborhood() -> None:
    specs = _candidate_specs(
        reference_id="side",
        seed={
            "analysis_direction": _direction(1.0, 8.0),
            "analysis_up_axis": [0.0, 0.0, 1.0],
            "focal_length_mm": 74.25,
            "distance_multiplier": 3.225,
        },
        phase="component_pose",
    )

    assert len(specs["views"]) == 25
    assert {view["calibration"]["azimuth_degrees"] for view in specs["views"]} == {
        349.0,
        355.0,
        1.0,
        7.0,
        13.0,
    }
    assert {view["calibration"]["elevation_degrees"] for view in specs["views"]} == {
        -4.0,
        2.0,
        8.0,
        14.0,
        20.0,
    }


def test_alternating_recheck_resolves_perspective_then_pose_again() -> None:
    seed = {
        "analysis_direction": _direction(1.0, -4.0),
        "analysis_up_axis": [0.0, 0.0, 1.0],
        "focal_length_mm": 75.0,
        "distance_multiplier": 3.25,
    }
    perspective = _candidate_specs(
        reference_id="side",
        seed=seed,
        phase="perspective_recheck",
    )
    pose = _candidate_specs(
        reference_id="side",
        seed=seed,
        phase="component_pose_recheck",
    )

    assert len(perspective["views"]) == 11
    assert len(pose["views"]) == 25
    assert {
        view["calibration"]["azimuth_degrees"] for view in pose["views"]
    } == {355.0, 358.0, 1.0, 4.0, 7.0}
    assert {
        view["calibration"]["elevation_degrees"] for view in pose["views"]
    } == {-10.0, -7.0, -4.0, -1.0, 2.0}


def test_nano_and_pico_candidates_monotonically_reduce_pose_step() -> None:
    seed = {
        "analysis_direction": _direction(120.0, 24.0),
        "analysis_up_axis": [0.0, 0.0, 1.0],
        "focal_length_mm": 50.0,
        "distance_multiplier": 2.4,
    }

    nano = _candidate_specs(
        reference_id="iso",
        seed=seed,
        phase="nano",
    )
    pico = _candidate_specs(
        reference_id="iso",
        seed=seed,
        phase="pico",
    )

    assert len(nano["views"]) == 27
    assert len(pico["views"]) == 27
    nano_azimuths = {view["calibration"]["azimuth_degrees"] for view in nano["views"]}
    pico_azimuths = {view["calibration"]["azimuth_degrees"] for view in pico["views"]}
    assert nano_azimuths == {119.75, 120.0, 120.25}
    assert pico_azimuths == {119.95, 120.0, 120.05}


def test_target_candidates_complete_rigid_camera_extrinsics() -> None:
    seed = {
        "analysis_direction": _direction(120.0, 24.0),
        "analysis_up_axis": [0.0, 0.0, 1.0],
        "focal_length_mm": 50.0,
        "distance_multiplier": 2.4,
        "target_offset_u": 0.02,
        "target_offset_v": -0.03,
    }

    target = _candidate_specs(
        reference_id="iso",
        seed=seed,
        phase="target",
    )
    target_micro = _candidate_specs(
        reference_id="iso",
        seed=seed,
        phase="target_micro",
    )
    target_pico = _candidate_specs(
        reference_id="iso",
        seed=seed,
        phase="target_pico",
    )

    assert len(target["views"]) == 25
    assert len(target_micro["views"]) == 9
    assert len(target_pico["views"]) == 9
    assert {view["target_offset_u"] for view in target["views"]} == {
        -0.06,
        -0.02,
        0.02,
        0.06,
        0.1,
    }
    assert {view["target_offset_v"] for view in target_micro["views"]} == {
        -0.04,
        -0.03,
        -0.02,
    }
    assert {view["target_offset_u"] for view in target_pico["views"]} == {
        0.017,
        0.02,
        0.023,
    }
    assert {view["calibration"]["azimuth_degrees"] for view in target["views"]} == {
        120.0
    }
    assert {view["calibration"]["elevation_degrees"] for view in target["views"]} == {
        24.0
    }


def test_continuous_view_specs_can_resume_camera_only_refinement() -> None:
    seeds = _seed_by_view_specs(
        {
            "schema_version": "qwen-camera-view-specs/v1",
            "views": [
                {
                    "view_id": "front",
                    "analysis_direction": [1.0, 0.0, 0.0],
                    "analysis_up_axis": [0.0, 0.0, 1.0],
                    "focal_length_mm": 52.0,
                    "distance_multiplier": 2.8,
                    "target_offset_u": 0.04,
                    "target_offset_v": -0.02,
                }
            ],
        }
    )

    assert seeds["front"]["focal_length_mm"] == 52.0
    assert seeds["front"]["distance_multiplier"] == 2.8
    assert seeds["front"]["target_offset_u"] == 0.04
    assert seeds["front"]["target_offset_v"] == -0.02


def test_full_resolution_finalists_are_global_across_phases() -> None:
    def candidate(name: str, score: float, distance: float) -> dict:
        return {
            "view_id": name,
            "score": score,
            "projection_iou": score,
            "boundary_p95_px": 10.0 - score,
            "analysis_direction": [1.0, 0.0, 0.0],
            "analysis_up_axis": [0.0, 0.0, 1.0],
            "focal_length_mm": 45.0 * distance,
            "distance_multiplier": distance,
            "target_offset_u": 0.0,
            "target_offset_v": 0.0,
        }

    finalists = _global_finalists(
        [
            candidate("fine_best", 0.95, 2.0),
            candidate("pico_best", 0.91, 3.0),
            candidate("fine_duplicate", 0.90, 2.0),
        ],
        count=2,
    )

    assert [item["view_id"] for item in finalists] == [
        "fine_best",
        "pico_best",
    ]


def test_alignment_gate_precedes_weighted_structure_score() -> None:
    candidates = [
        {
            "view_id": "weighted_but_misaligned",
            "score": 0.99,
            "projection_iou": 0.94,
            "boundary_p95_px": 8.0,
        },
        {
            "view_id": "gate_pass",
            "score": 0.80,
            "projection_iou": 0.975,
            "boundary_p95_px": 2.5,
        },
    ]

    ranked = sorted(candidates, key=_alignment_candidate_sort_key)

    assert ranked[0]["view_id"] == "gate_pass"


def test_incomplete_foreground_reports_recall_separately_from_precision() -> None:
    reference = np.zeros((64, 64), dtype=np.uint8)
    reference[16:48, 16:40] = 255
    rendered = np.zeros_like(reference)
    rendered[16:48, 16:52] = 255

    metrics = _silhouette_coverage_metrics(reference, rendered)

    assert metrics["target_recall"] == 1.0
    assert metrics["rendered_precision"] < 1.0
    assert metrics["silhouette_f_score"] < 1.0


def test_detached_reference_components_have_equal_registration_weight() -> None:
    reference = np.zeros((80, 80), dtype=np.uint8)
    reference[10:60, 10:60] = 255
    reference[68:73, 68:73] = 255
    reference[59:70, 59] = 255
    reference[69, 59:69] = 255
    rendered = np.zeros_like(reference)
    rendered[10:60, 10:60] = 255

    metrics = _component_balanced_reference_metrics(reference, rendered)

    assert metrics["reference_component_count"] == 2
    assert metrics["reference_component_macro_recall"] == 0.5
    assert metrics["reference_component_min_recall"] == 0.0


def test_spatial_reference_cells_expose_attached_local_misalignment() -> None:
    reference = np.zeros((90, 90), dtype=np.uint8)
    reference[20:80, 20:70] = 255
    reference[5:20, 55:65] = 255
    rendered = np.zeros_like(reference)
    rendered[20:80, 20:70] = 255

    metrics = _spatial_balanced_reference_metrics(reference, rendered)

    assert metrics["reference_spatial_cell_count"] >= 4
    assert metrics["reference_spatial_macro_recall"] < 1.0
    assert metrics["reference_spatial_min_recall"] < 0.6
    assert any(
        cell["row"] == 0 and cell["recall"] < 0.6
        for cell in metrics["reference_spatial_cells"]
    )


def test_part_structure_objective_balances_small_medium_and_large_parts() -> None:
    image = np.zeros((160, 160, 3), dtype=np.uint8)
    ids = np.zeros_like(image)
    foreground = np.zeros((160, 160), dtype=np.uint8)
    rectangles = {
        "P0001": (12, 20, 20, 28),  # 64 pixels: small
        "P0002": (48, 20, 68, 40),  # 400 pixels: medium
        "P0003": (88, 20, 128, 50),  # 1200 pixels: large
    }
    for part_id, (left, top, right, bottom) in rectangles.items():
        red, green, blue = _part_color(part_id)
        ids[top:bottom, left:right] = (blue, green, red)
        image[top:bottom, left:right] = 255
        foreground[top:bottom, left:right] = 255

    metrics = _part_balanced_structure_metrics(
        ids=ids,
        parts=[{"part_id": part_id} for part_id in rectangles],
        affine=np.asarray(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float32,
        ),
        reference_image=image,
        reference_mask=foreground,
    )

    assert metrics["structure_part_count"] == 3
    assert {
        name: values["part_count"]
        for name, values in metrics["structure_size_strata"].items()
    } == {"small": 1, "medium": 1, "large": 1}
    assert metrics["structure_score"] > 0.7


def test_boundary_metric_detects_pixel_shift() -> None:
    target = np.zeros((128, 128), dtype=np.uint8)
    cv2.rectangle(target, (24, 24), (104, 104), 255, -1)
    shifted = np.zeros_like(target)
    cv2.rectangle(shifted, (30, 24), (110, 104), 255, -1)

    exact = _boundary_metrics(target, target)
    offset = _boundary_metrics(target, shifted)

    assert exact["boundary_p95_px"] == 0.0
    assert offset["boundary_p95_px"] > 0.0


def test_camera_foreground_depends_only_on_stable_part_ids() -> None:
    ids = np.full((32, 32, 3), 28, dtype=np.uint8)
    red, green, blue = _part_color("P0001")
    ids[8:24, 10:22] = (blue, green, red)
    colors = [np.asarray((blue, green, red), dtype=np.uint8)]

    first = _deterministic_part_id_foreground(ids, colors)
    second = _deterministic_part_id_foreground(ids.copy(), colors)

    assert np.array_equal(first, second)
    assert np.count_nonzero(first) == 16 * 12
    assert np.all(first[:8] == 0)


def test_residual_components_report_largest_regions_first() -> None:
    residual = np.zeros((100, 120), dtype=np.uint8)
    cv2.rectangle(residual, (5, 5), (14, 14), 255, -1)
    cv2.rectangle(residual, (40, 30), (69, 49), 255, -1)

    components = _residual_components(residual)

    assert len(components) == 2
    assert components[0]["area_pixels"] == 600
    assert components[0]["bbox_xywh"] == [40, 30, 30, 20]
    assert components[1]["area_pixels"] == 100


def test_reference_masks_accept_human_annotation_before_palette_stage(
    tmp_path: Path,
) -> None:
    mask_path = tmp_path / "front.png"
    mask = np.zeros((32, 48), dtype=np.uint8)
    mask[4:28, 8:40] = 255
    assert cv2.imwrite(str(mask_path), mask)
    manifest_path = tmp_path / "annotations.json"
    manifest_path.write_text(
        json.dumps(
            {
                "source_views": [
                    {
                        "id": "front",
                        "confirmed_mask": {"path": mask_path.name},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    loaded = _reference_masks(manifest_path)

    assert set(loaded) == {"front"}
    assert np.array_equal(loaded["front"][0], mask)


def test_calibrated_registry_removes_discrete_seed_bank(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "baseline.json"
    calibrated = tmp_path / "calibrated.json"
    output = tmp_path / "merged.json"
    baseline.write_text(
        json.dumps(
            {
                "schema_version": "qwen-material-parts/v1",
                "parts": [{"part_id": "P0001"}],
                "render_set": {
                    "views": [
                        {"view_id": "front"},
                        {"view_id": "pose_a045_e035"},
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    calibrated.write_text(
        json.dumps(
            {
                "schema_version": "qwen-material-parts/v1",
                "parts": [{"part_id": "P0001"}],
                "render_set": {
                    "views": [
                        {
                            "view_id": "photo_01",
                            "camera_calibration": {"reference_view_id": "photo_01"},
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )

    _merge_registry(
        baseline_path=baseline,
        calibrated_path=calibrated,
        output_path=output,
        reference_ids={"photo_01"},
    )
    merged = json.loads(output.read_text(encoding="utf-8"))

    assert [view["view_id"] for view in merged["render_set"]["views"]] == ["photo_01"]
    assert merged["render_set"]["continuous_camera_calibration"] is True
    assert merged["render_set"]["calibration_source_view_count"] == 2


def test_full_resolution_winner_is_sealed_without_rerender(
    tmp_path: Path,
) -> None:
    rendered = tmp_path / "finalists.json"
    output = tmp_path / "sealed" / "part_registry.rendered.json"
    rendered.write_text(
        json.dumps(
            {
                "schema_version": "qwen-material-parts/v1",
                "parts": [{"part_id": "P0001"}],
                "render_set": {
                    "views": [
                        {
                            "view_id": "rerank_front_02",
                            "rgb": "/tmp/already_rendered.png",
                            "part_ids": "/tmp/already_rendered_ids.png",
                            "camera_calibration": {
                                "reference_view_id": "front",
                                "phase": "full_resolution_rerank",
                            },
                        },
                        {
                            "view_id": "rerank_front_03",
                            "camera_calibration": {
                                "reference_view_id": "front",
                                "phase": "full_resolution_rerank",
                            },
                        },
                    ]
                },
            }
        ),
        encoding="utf-8",
    )

    _seal_full_resolution_winners(
        rendered_path=rendered,
        winners={"front": {"view_id": "rerank_front_02"}},
        output_path=output,
    )
    sealed = json.loads(output.read_text(encoding="utf-8"))

    assert [view["view_id"] for view in sealed["render_set"]["views"]] == [
        "front"
    ]
    assert sealed["render_set"]["views"][0]["rgb"] == "/tmp/already_rendered.png"
    assert sealed["render_set"]["views"][0]["sealed_source_view_id"] == (
        "rerank_front_02"
    )
    assert sealed["render_set"]["sealed_full_resolution_winners"] is True
