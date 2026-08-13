from __future__ import annotations

import math
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import qwen_material_pipeline.evidence.camera_calibration as camera_calibration
from qwen_material_pipeline.evidence.camera_calibration import (
    CAMERA_PHASES,
    _angles,
    _alignment_candidate_sort_key,
    _boundary_metrics,
    _candidate_specs,
    _classify_multiview_residuals,
    _component_balanced_reference_metrics,
    _constrained_frame_projection,
    _direction,
    _deterministic_part_id_foreground,
    _global_finalists,
    _merge_registry,
    _part_balanced_structure_metrics,
    _robust_part_consensus,
    _reference_masks,
    _residual_components,
    _score_candidates,
    _select_alignment_candidate,
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


def test_inprocess_camera_render_cleans_partial_output_and_never_retries_failure(
    tmp_path: Path,
) -> None:
    output = tmp_path / "renders"
    output.mkdir()
    (output / "stale.txt").write_text("partial", encoding="utf-8")
    calls = 0

    def fail_once(**kwargs) -> Path:
        nonlocal calls
        calls += 1
        assert not (kwargs["output_dir"] / "stale.txt").exists()
        kwargs["output_dir"].mkdir(parents=True)
        (kwargs["output_dir"] / "partial.txt").write_text(
            "poisoned", encoding="utf-8"
        )
        raise ValueError("render failed")

    with pytest.raises(RuntimeError, match="refusing to reuse this Isaac session"):
        camera_calibration._run_render(
            isaac_python=tmp_path / "isaac-python.sh",
            registry=tmp_path / "registry.json",
            output_dir=output,
            view_specs=tmp_path / "views.json",
            resolution=256,
            rt_subframes=2,
            analysis_up_axis="z",
            analysis_front_axis="-y",
            render_runner=fail_once,
        )

    assert calls == 1


def test_inprocess_renderer_adapter_preserves_camera_render_contract(
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def fake_render_part_views(**kwargs):
        captured.update(kwargs)
        output = Path(kwargs["output_dir"]) / "part_registry.rendered.json"
        output.parent.mkdir(parents=True)
        output.write_text("{}", encoding="utf-8")
        return {"output_registry": str(output)}

    runner = camera_calibration._make_inprocess_render_runner(
        render_part_views=fake_render_part_views,
        axis_vectors={"z": (0.0, 0.0, 1.0), "-y": (0.0, -1.0, 0.0)},
    )
    output_dir = tmp_path / "renders"
    output = runner(
        registry=tmp_path / "registry.json",
        output_dir=output_dir,
        view_specs=tmp_path / "views.json",
        resolution=384,
        rt_subframes=3,
        analysis_up_axis="z",
        analysis_front_axis="-y",
    )

    assert output == output_dir / "part_registry.rendered.json"
    assert captured == {
        "registry_path": tmp_path / "registry.json",
        "output_dir": output_dir,
        "resolution": 384,
        "view_names": None,
        "rt_subframes": 3,
        "analysis_up_axis": (0.0, 0.0, 1.0),
        "analysis_front_axis": (0.0, -1.0, 0.0),
        "lighting_profile": "geometry",
        "showcase": False,
        "generate_part_evidence": False,
        "custom_view_specs_path": tmp_path / "views.json",
    }


def test_inprocess_backend_starts_and_closes_one_app_for_multiple_batches(
    tmp_path: Path,
    monkeypatch,
) -> None:
    events: list[str] = []
    render_calls: list[Path] = []

    class FakeApp:
        def close(self) -> None:
            events.append("close")

    def app_factory(config):
        assert config == {"headless": True, "create_new_stage": False}
        events.append("start")
        return FakeApp()

    def fake_render_part_views(**kwargs):
        render_calls.append(Path(kwargs["output_dir"]))
        output = Path(kwargs["output_dir"]) / "part_registry.rendered.json"
        output.parent.mkdir(parents=True)
        output.write_text("{}", encoding="utf-8")
        return {"output_registry": str(output)}

    def load_render_module():
        assert events == ["start"]
        events.append("import-render")
        return SimpleNamespace(
            render_part_views=fake_render_part_views,
            AXIS_VECTORS={"z": (0.0, 0.0, 1.0), "-y": (0.0, -1.0, 0.0)},
        )

    def fake_calibrate(_args, *, render_runner):
        for name in ("phase_a", "phase_b"):
            render_runner(
                registry=tmp_path / "registry.json",
                output_dir=tmp_path / name,
                view_specs=tmp_path / f"{name}.json",
                resolution=256,
                rt_subframes=2,
                analysis_up_axis="z",
                analysis_front_axis="-y",
            )
        return {"views": []}

    monkeypatch.setattr(camera_calibration, "_calibrate_from_args", fake_calibrate)
    exit_code, report = camera_calibration._run_inprocess_backend(
        SimpleNamespace(),
        simulation_app_factory=app_factory,
        render_module_loader=load_render_module,
    )

    assert exit_code == 0
    assert report == {"views": []}
    assert render_calls == [tmp_path / "phase_a", tmp_path / "phase_b"]
    assert events == ["start", "import-render", "close"]


def test_inprocess_budget_rotation_writes_checkpoint_marker_and_closes_app(
    tmp_path: Path,
    monkeypatch,
) -> None:
    events: list[str] = []

    class FakeApp:
        def close(self) -> None:
            events.append("close")

    def fake_calibrate(_args, *, render_runner):
        assert callable(render_runner)
        checkpoint = tmp_path / "camera" / "front" / "nano_scores.json"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_text("{}", encoding="utf-8")
        raise camera_calibration._RenderBatchBudgetReached(
            "sealed front/nano",
            checkpoint=checkpoint,
        )

    monkeypatch.setattr(camera_calibration, "_calibrate_from_args", fake_calibrate)
    exit_code, report = camera_calibration._run_inprocess_backend(
        SimpleNamespace(output_dir=tmp_path / "camera", max_new_render_batches=8),
        simulation_app_factory=lambda _config: FakeApp(),
        render_module_loader=lambda: SimpleNamespace(
            render_part_views=lambda **_kwargs: {},
            AXIS_VECTORS={},
        ),
    )

    assert exit_code == 0
    assert report is None
    assert events == ["close"]
    marker = camera_calibration._read_object(
        tmp_path / "camera" / camera_calibration.SUPERVISOR_ROTATION_MARKER
    )
    assert marker["schema_version"] == (
        camera_calibration.SUPERVISOR_ROTATION_SCHEMA_VERSION
    )
    assert marker["checkpoint"].endswith("front/nano_scores.json")


def test_supervisor_rotates_budget_sessions_and_retries_true_failures(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    isaac_python = tmp_path / "python.sh"
    isaac_python.write_text("#!/bin/sh\n", encoding="utf-8")
    args = SimpleNamespace(
        registry=tmp_path / "registry.json",
        reference_manifest=tmp_path / "references.json",
        spatial_mapping=None,
        initial_view_specs=None,
        search_phases=None,
        isaac_python=isaac_python,
        output_dir=tmp_path / "camera",
        reference_ids=None,
        search_resolution=256,
        final_resolution=512,
        rt_subframes=2,
        analysis_up_axis="z",
        analysis_front_axis="-y",
        max_new_render_batches=None,
    )
    outcomes = iter(("failure", "rotation", "failure", "failure", "complete"))
    commands: list[list[str]] = []
    sleeps: list[float] = []

    def fake_run(command, *, check, env):
        assert check is False
        assert Path(env["PYTHONPATH"].split(os.pathsep)[0]) == (
            Path(camera_calibration.__file__).resolve().parents[2]
        )
        commands.append(list(command))
        outcome = next(outcomes)
        if outcome == "rotation":
            session_batch_limit = int(
                command[command.index("--max-new-render-batches") + 1]
            )
            checkpoint = args.output_dir / "front" / "nano_scores.json"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_text("{}", encoding="utf-8")
            camera_calibration._write_object(
                args.output_dir / camera_calibration.SUPERVISOR_ROTATION_MARKER,
                {
                    "schema_version": (
                        camera_calibration.SUPERVISOR_ROTATION_SCHEMA_VERSION
                    ),
                    "output_dir": str(args.output_dir.resolve()),
                    "checkpoint": str(checkpoint.resolve()),
                    "checkpoint_sha256": camera_calibration._sha256_file(checkpoint),
                    "max_new_render_batches": session_batch_limit,
                },
            )
            return SimpleNamespace(returncode=0)
        if outcome == "complete":
            args.output_dir.mkdir(parents=True, exist_ok=True)
            (args.output_dir / "camera_calibration_report.json").write_text(
                "{}", encoding="utf-8"
            )
            return SimpleNamespace(returncode=0)
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(camera_calibration.subprocess, "run", fake_run)
    monkeypatch.setattr(
        camera_calibration.time, "sleep", lambda seconds: sleeps.append(seconds)
    )

    assert camera_calibration._run_supervisor_backend(args) == 0
    assert len(commands) == 5
    assert commands[0][0] == str(isaac_python.resolve())
    assert commands[0][commands[0].index("--render-backend") + 1] == "inprocess"
    assert [
        command[command.index("--max-new-render-batches") + 1]
        for command in commands
    ] == ["2", "1", "1", "1", "1"]
    assert sleeps == [
        camera_calibration.RENDER_RETRY_DELAY_SECONDS,
        camera_calibration.RENDER_RETRY_DELAY_SECONDS,
        camera_calibration.RENDER_RETRY_DELAY_SECONDS,
    ]
    assert capsys.readouterr().out.count(
        "reducing all remaining Isaac sessions to one new render batch"
    ) == 1


def test_supervisor_keeps_batch_two_after_normal_budget_rotation(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    isaac_python = tmp_path / "python.sh"
    isaac_python.write_text("#!/bin/sh\n", encoding="utf-8")
    output_dir = tmp_path / "camera"
    args = SimpleNamespace(
        registry=tmp_path / "registry.json",
        reference_manifest=tmp_path / "references.json",
        spatial_mapping=None,
        initial_view_specs=None,
        search_phases=None,
        isaac_python=isaac_python,
        output_dir=output_dir,
        reference_ids=None,
        search_resolution=256,
        final_resolution=512,
        rt_subframes=2,
        analysis_up_axis="z",
        analysis_front_axis="-y",
        max_new_render_batches=None,
    )
    outcomes = iter(("rotation", "complete"))
    limits: list[int] = []

    def fake_run(command, *, check, env):
        assert check is False
        assert env["PYTHONPATH"]
        session_batch_limit = int(
            command[command.index("--max-new-render-batches") + 1]
        )
        limits.append(session_batch_limit)
        outcome = next(outcomes)
        output_dir.mkdir(parents=True, exist_ok=True)
        if outcome == "rotation":
            checkpoint = output_dir / "front" / "nano_scores.json"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_text("{}", encoding="utf-8")
            camera_calibration._write_object(
                output_dir / camera_calibration.SUPERVISOR_ROTATION_MARKER,
                {
                    "schema_version": (
                        camera_calibration.SUPERVISOR_ROTATION_SCHEMA_VERSION
                    ),
                    "output_dir": str(output_dir.resolve()),
                    "checkpoint": str(checkpoint.resolve()),
                    "checkpoint_sha256": camera_calibration._sha256_file(checkpoint),
                    "max_new_render_batches": session_batch_limit,
                },
            )
        else:
            (output_dir / "camera_calibration_report.json").write_text(
                "{}", encoding="utf-8"
            )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(camera_calibration.subprocess, "run", fake_run)

    assert camera_calibration._run_supervisor_backend(args) == 0
    assert limits == [2, 2]
    assert "reducing all remaining Isaac sessions" not in capsys.readouterr().out


def test_supervisor_completes_in_limit_one_session_after_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    isaac_python = tmp_path / "python.sh"
    isaac_python.write_text("#!/bin/sh\n", encoding="utf-8")
    output_dir = tmp_path / "camera"
    args = SimpleNamespace(
        registry=tmp_path / "registry.json",
        reference_manifest=tmp_path / "references.json",
        spatial_mapping=None,
        initial_view_specs=None,
        search_phases=None,
        isaac_python=isaac_python,
        output_dir=output_dir,
        reference_ids=None,
        search_resolution=256,
        final_resolution=512,
        rt_subframes=2,
        analysis_up_axis="z",
        analysis_front_axis="-y",
        max_new_render_batches=None,
    )
    return_codes = iter((1, 0))
    limits: list[int] = []

    def fake_run(command, *, check, env):
        assert check is False
        assert env["PYTHONPATH"]
        limits.append(
            int(command[command.index("--max-new-render-batches") + 1])
        )
        return_code = next(return_codes)
        if return_code == 0:
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "camera_calibration_report.json").write_text(
                "{}", encoding="utf-8"
            )
        return SimpleNamespace(returncode=return_code)

    monkeypatch.setattr(camera_calibration.subprocess, "run", fake_run)
    monkeypatch.setattr(camera_calibration.time, "sleep", lambda _seconds: None)

    assert camera_calibration._run_supervisor_backend(args) == 0
    assert limits == [2, 1]


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
                "winner": {
                    "view_id": "cal_front_micro_000",
                    "objective_version": camera_calibration.CAMERA_OBJECTIVE_VERSION,
                },
                "candidates": [
                    {
                        "view_id": "cal_front_micro_000",
                        "objective_version": (
                            camera_calibration.CAMERA_OBJECTIVE_VERSION
                        ),
                    }
                ],
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

    score_document = json.loads(scores_path.read_text(encoding="utf-8"))
    score_document["view_specs_sha256"] = "0" * 64
    scores_path.write_text(json.dumps(score_document), encoding="utf-8")
    assert (
        camera_calibration._completed_phase(
            specs_path=specs_path,
            scores_path=scores_path,
            expected_specs=specs,
            reference_id="front",
            phase="micro",
        )
        is None
    )

    # Legacy checkpoints created before the hash field remain reusable only
    # while their complete specs and candidate IDs still match exactly.
    score_document.pop("view_specs_sha256")
    scores_path.write_text(json.dumps(score_document), encoding="utf-8")

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


def test_first_stage_camera_candidates_add_roll_principal_point_and_distortion() -> None:
    seed = {
        "analysis_direction": _direction(15.0, 12.0),
        "analysis_up_axis": [0.0, 0.0, 1.0],
        "focal_length_mm": 45.0,
        "distance_multiplier": 2.15,
    }
    roll = _candidate_specs(reference_id="side", seed=seed, phase="roll")
    principal = _candidate_specs(
        reference_id="side", seed=seed, phase="principal_point"
    )
    radial = _candidate_specs(
        reference_id="side", seed=seed, phase="radial_distortion"
    )

    assert len(roll["views"]) == 5
    assert {row["roll_degrees"] for row in roll["views"]} == {
        -6.0,
        -3.0,
        0.0,
        3.0,
        6.0,
    }
    assert len(principal["views"]) == 25
    assert {row["principal_point_u"] for row in principal["views"]} == {
        -0.08,
        -0.04,
        0.0,
        0.04,
        0.08,
    }
    assert len(radial["views"]) == 15
    assert {row["radial_distortion_k1"] for row in radial["views"]} == {
        -0.16,
        -0.08,
        0.0,
        0.08,
        0.16,
    }
    assert sum(
        row["calibration"]["frame_anchor"] is True for row in radial["views"]
    ) == 1


def test_constrained_frame_projection_cannot_hide_large_scale_error() -> None:
    reference = np.zeros((256, 256), dtype=np.uint8)
    reference[72:184, 76:180] = 255
    source = np.zeros_like(reference)
    source[96:160, 98:158] = 255

    projection = _constrained_frame_projection(
        reference,
        source,
        anchor_affine=np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
    )
    audit = projection["ecc_transform_audit"]

    assert projection["projection_iou"] < 0.50
    assert audit["residual_scale"] <= 1.015
    assert abs(audit["residual_rotation_degrees"]) <= 1.0
    assert max(abs(value) for value in audit["residual_translation_ratio_xy"]) <= 0.02


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
    assert CAMERA_PHASES.index("fine") < CAMERA_PHASES.index("roll")
    assert CAMERA_PHASES.index("roll") < CAMERA_PHASES.index("principal_point")
    assert CAMERA_PHASES.index("principal_point") < CAMERA_PHASES.index(
        "radial_distortion"
    )
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
        "frame_micro",
        "radial_distortion_micro",
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


def test_phase_winner_can_be_retained_ahead_of_regressive_global_score() -> None:
    trusted_winner = {
        "view_id": "trusted_winner",
        "score": 0.65,
        "projection_iou": 0.91,
        "boundary_p95_px": 13.0,
        "analysis_direction": [1.0, 0.0, 0.0],
        "analysis_up_axis": [0.0, 0.0, 1.0],
        "focal_length_mm": 45.0,
        "distance_multiplier": 2.0,
    }
    regressive = {
        **trusted_winner,
        "view_id": "regressive_structure_score",
        "score": 0.80,
        "projection_iou": 0.80,
        "boundary_p95_px": 10.0,
        "distance_multiplier": 3.0,
    }

    finalists = _global_finalists(
        [trusted_winner, regressive],
        count=1,
        required=(trusted_winner,),
    )

    assert [item["view_id"] for item in finalists] == ["trusted_winner"]


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


def test_incomplete_alignment_candidates_use_complete_objective() -> None:
    candidates = [
        {
            "view_id": "lower_boundary_but_weaker_complete_evidence",
            "score": 0.70,
            "projection_iou": 0.80,
            "boundary_p95_px": 20.0,
        },
        {
            "view_id": "stronger_complete_evidence",
            "score": 0.82,
            "projection_iou": 0.85,
            "boundary_p95_px": 23.0,
        },
    ]

    ranked = sorted(candidates, key=_alignment_candidate_sort_key)

    assert ranked[0]["view_id"] == "stronger_complete_evidence"


def test_phase_selection_keeps_incomplete_candidate_inside_incumbent_trust_region(
) -> None:
    records = [
        {
            "view_id": "incumbent",
            "score": 0.70,
            "projection_iou": 0.88,
            "boundary_p95_px": 17.5,
            "complete_alignment_candidate": False,
            "calibration": {"frame_anchor": True},
        },
        {
            "view_id": "tempting_but_drifted",
            "score": 0.90,
            "projection_iou": 0.85,
            "boundary_p95_px": 18.0,
            "complete_alignment_candidate": False,
            "calibration": {"frame_anchor": False},
        },
        {
            "view_id": "trusted_improvement",
            "score": 0.80,
            "projection_iou": 0.877,
            "boundary_p95_px": 18.0,
            "complete_alignment_candidate": False,
            "calibration": {"frame_anchor": False},
        },
    ]

    winner = _select_alignment_candidate(records)

    assert winner["view_id"] == "trusted_improvement"


def test_phase_selection_does_not_replace_valid_rigid_consensus_with_invalid_one(
) -> None:
    records = [
        {
            "view_id": "incumbent",
            "score": 0.70,
            "projection_iou": 0.90,
            "boundary_p95_px": 12.0,
            "complete_alignment_candidate": False,
            "rigid_consensus_valid": True,
            "rigid_consensus_score": 0.80,
            "calibration": {"frame_anchor": True},
        },
        {
            "view_id": "invalid_consensus_but_high_score",
            "score": 0.99,
            "projection_iou": 0.901,
            "boundary_p95_px": 12.0,
            "complete_alignment_candidate": False,
            "rigid_consensus_valid": False,
            "rigid_consensus_score": 0.0,
            "calibration": {"frame_anchor": False},
        },
    ]

    winner = _select_alignment_candidate(records)

    assert winner["view_id"] == "incumbent"


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


def test_robust_rigid_consensus_rejects_detached_assembly_without_part_rules() -> None:
    reference = np.zeros((180, 180), dtype=np.uint8)
    image = np.zeros((180, 180, 3), dtype=np.uint8)
    ids = np.zeros_like(image)
    parts = []
    rectangles = {
        "P0001": (50, 30, 95, 150),
        "P0002": (95, 30, 130, 150),
        "P0003": (60, 20, 85, 30),
        # Two sibling Parts are rendered at the wrong assembly state.  Their
        # photographed counterpart is 35 pixels lower.
        "P0004": (18, 112, 38, 132),
        "P0005": (38, 118, 50, 126),
    }
    for part_id, (left, top, right, bottom) in rectangles.items():
        red, green, blue = _part_color(part_id)
        ids[top:bottom, left:right] = (blue, green, red)
        parts.append(
            {
                "part_id": part_id,
                "prim_path": (
                    f"/Asset/Rigid/{part_id}/Mesh"
                    if part_id <= "P0003"
                    else f"/Asset/Accessory/Joint/{part_id}/Mesh"
                ),
            }
        )
    reference[30:150, 50:130] = 255
    reference[20:30, 60:85] = 255
    reference[147:167, 18:38] = 255
    reference[153:161, 38:50] = 255
    image[reference > 0] = 255

    consensus = _robust_part_consensus(
        ids=ids,
        parts=parts,
        affine=np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], np.float32),
        reference_image=image,
        reference_mask=reference,
    )

    assert consensus["rigid_consensus_valid"] is True
    assert {"P0001", "P0002", "P0003"}.issubset(
        consensus["rigid_consensus_inlier_part_ids"]
    )
    assert {"P0004", "P0005"}.issubset(
        consensus["rigid_consensus_outlier_part_ids"]
    )
    cluster = next(
        item
        for item in consensus["assembly_residual_clusters"]
        if item["part_ids"] == ["P0004", "P0005"]
    )
    assert cluster["classification"] == "assembly_state_or_geometry_mismatch"
    assert cluster["residual_direction_coherence"] >= 0.7
    assert np.linalg.norm(cluster["median_residual_vector_px"]) > 5.0


def test_robust_rigid_consensus_marks_edge_free_internal_part_indeterminate() -> None:
    reference = np.zeros((160, 160), dtype=np.uint8)
    reference[20:140, 20:140] = 255
    image = np.dstack([reference] * 3)
    ids = np.zeros_like(image)
    parts = []
    rectangles = {
        "P0001": (20, 20, 70, 140),
        "P0002": (70, 20, 140, 80),
        "P0003": (70, 80, 140, 140),
        # This internal panel has no corresponding edge in the photograph.
        # It must not be called a rigid outlier merely because its CAD boundary
        # crosses a uniform photographed surface.
        "P0004": (80, 50, 120, 110),
    }
    for part_id, (left, top, right, bottom) in rectangles.items():
        red, green, blue = _part_color(part_id)
        ids[top:bottom, left:right] = (blue, green, red)
        parts.append(
            {
                "part_id": part_id,
                "prim_path": f"/Asset/Assembly/{part_id}/Mesh",
            }
        )

    consensus = _robust_part_consensus(
        ids=ids,
        parts=parts,
        affine=np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], np.float32),
        reference_image=image,
        reference_mask=reference,
    )

    assert consensus["rigid_consensus_valid"] is True
    assert "P0004" in consensus["rigid_consensus_indeterminate_part_ids"]
    assert "P0004" not in consensus["rigid_consensus_outlier_part_ids"]
    row = next(
        item
        for item in consensus["rigid_consensus_part_residuals"]
        if item["part_id"] == "P0004"
    )
    assert row["consensus_observable"] is False
    assert row["rigid_inlier"] is None


def test_robust_rigid_consensus_fails_open_to_phase1_for_sparse_assets() -> None:
    reference = np.zeros((80, 80), dtype=np.uint8)
    reference[20:60, 20:60] = 255
    image = np.dstack([reference] * 3)
    ids = np.zeros_like(image)
    red, green, blue = _part_color("P0001")
    ids[20:60, 20:60] = (blue, green, red)

    consensus = _robust_part_consensus(
        ids=ids,
        parts=[{"part_id": "P0001", "prim_path": "/Asset/Part/Mesh"}],
        affine=np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], np.float32),
        reference_image=image,
        reference_mask=reference,
    )

    assert consensus["rigid_consensus_valid"] is False
    assert consensus["rigid_consensus_reason"] == "insufficient_visible_parts"


def test_multiview_residual_classification_requires_repeated_rejection() -> None:
    def score(*, inliers: list[str], rows: list[tuple[str, str]]) -> dict:
        return {
            "rigid_consensus_inlier_part_ids": inliers,
            "rigid_consensus_part_residuals": [
                {
                    "part_id": part_id,
                    "assembly_subtree": subtree,
                    "residual_px": 12.0,
                    "inside_reference_ratio": 0.4,
                }
                for part_id, subtree in rows
            ],
        }

    diagnosis = _classify_multiview_residuals(
        {
            "front": score(
                inliers=["P0001", "P0003"],
                rows=[
                    ("P0001", "/Asset/Rigid"),
                    ("P0002", "/Asset/Accessory"),
                    ("P0003", "/Asset/Rigid"),
                ],
            ),
            "side": score(
                inliers=["P0001"],
                rows=[
                    ("P0001", "/Asset/Rigid"),
                    ("P0002", "/Asset/Accessory"),
                    ("P0003", "/Asset/Rigid"),
                ],
            ),
        }
    )

    assert diagnosis["persistent_mismatch_part_ids"] == ["P0002"]
    assert diagnosis["view_local_mismatch_part_ids"] == ["P0003"]


def test_fixed_rigid_anchor_scores_exact_sealed_parts_and_penalizes_missing() -> None:
    from qwen_material_pipeline.evidence.camera_calibration import (
        _fixed_rigid_anchor_metrics,
    )

    observations = [
        {
            "part_id": part_id,
            "stratum": stratum,
            "projected_pixels": pixels,
            "residual_px": residual,
            "inside_reference_ratio": inside,
        }
        for part_id, stratum, pixels, residual, inside in (
            ("P1", "small", 64, 2.0, 0.9),
            ("P2", "medium", 256, 3.0, 0.8),
            ("P3", "large", 1024, 4.0, 0.7),
        )
    ]

    complete = _fixed_rigid_anchor_metrics(
        observations=observations,
        expected_part_ids=["P1", "P2", "P3"],
        image_shape=(512, 512),
    )
    missing = _fixed_rigid_anchor_metrics(
        observations=observations[:2],
        expected_part_ids=["P1", "P2", "P3"],
        image_shape=(512, 512),
    )

    assert complete["fixed_anchor_valid"] is True
    assert complete["fixed_anchor_coverage"] == 1.0
    assert missing["fixed_anchor_coverage"] == pytest.approx(2 / 3)
    assert missing["fixed_anchor_score"] < complete["fixed_anchor_score"]


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


def test_camera_scores_preserve_each_candidates_own_calibration_metadata(
    tmp_path: Path,
) -> None:
    red, green, blue = _part_color("P0001")
    ids = np.zeros((64, 64, 3), dtype=np.uint8)
    ids[12:52, 14:50] = (blue, green, red)
    first_ids = tmp_path / "first.png"
    second_ids = tmp_path / "second.png"
    assert cv2.imwrite(str(first_ids), ids)
    assert cv2.imwrite(str(second_ids), ids)
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "parts": [{"part_id": "P0001"}],
                "render_set": {
                    "views": [
                        {
                            "view_id": "candidate_a",
                            "part_ids_raw": str(first_ids),
                            "camera_calibration": {
                                "reference_view_id": "side",
                                "candidate_marker": "a",
                            },
                        },
                        {
                            "view_id": "candidate_b",
                            "part_ids_raw": str(second_ids),
                            "camera_calibration": {
                                "reference_view_id": "side",
                                "candidate_marker": "b",
                            },
                        },
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    reference = np.zeros((64, 64), dtype=np.uint8)
    reference[12:52, 14:50] = 255

    _winner, candidates = _score_candidates(
        reference_id="side",
        reference_mask=reference,
        reference_image=np.zeros((64, 64, 3), dtype=np.uint8),
        registry_path=registry,
    )

    by_id = {candidate["view_id"]: candidate for candidate in candidates}
    assert by_id["candidate_a"]["calibration"]["candidate_marker"] == "a"
    assert by_id["candidate_b"]["calibration"]["candidate_marker"] == "b"
    assert by_id["candidate_a"]["calibration"]["frame_anchor_affine"] == (
        by_id["candidate_b"]["calibration"]["frame_anchor_affine"]
    )


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
        winners={
            "front": {
                "view_id": "rerank_front_02",
                "whole_asset_similarity": {
                    "ecc_transform_audit": {
                        "anchor_affine": [[1.2, 0.0, 3.0], [0.0, 1.2, 4.0]]
                    }
                },
            }
        },
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
    assert sealed["render_set"]["views"][0]["camera_calibration"][
        "frame_anchor_affine"
    ] == [[1.2, 0.0, 3.0], [0.0, 1.2, 4.0]]
    assert sealed["render_set"]["sealed_full_resolution_winners"] is True
