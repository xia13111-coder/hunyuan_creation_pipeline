from __future__ import annotations

import io
import inspect
import json
import math

import pytest

from qwen_material_pipeline.core.progress import emit_progress_event, parse_progress_line
from qwen_material_pipeline.usd.render import (
    EVIDENCE_LIGHTING_PROFILES,
    RENDER_CAPTURE_PROGRESS_STAGE,
    RENDER_VIEWS_PROGRESS_STAGE,
    VIEW_DIRECTIONS,
    VIEW_PRESETS,
    _analysis_basis,
    _analysis_to_world,
    _camera_up_axis,
    _clear_render_evidence,
    _counted_progress_items,
    _cross,
    _highlighted_context_crop,
    _isolated_target_crop,
    _load_custom_view_specs,
    _normalize,
    _resolve_view_directions,
    _showcase_scene_spec,
    _simulation_app_launch_config,
    render_part_views,
)


def _assert_vector_close(
    actual: tuple[float, float, float], expected: tuple[float, float, float]
) -> None:
    assert len(actual) == len(expected)
    for actual_value, expected_value in zip(actual, expected, strict=True):
        assert math.isclose(actual_value, expected_value, abs_tol=1e-9)


def test_default_z_up_and_negative_y_front_preserve_canonical_directions() -> None:
    basis = _analysis_basis((0.0, 0.0, 1.0), (0.0, -1.0, 0.0))

    _assert_vector_close(
        _analysis_to_world(VIEW_DIRECTIONS["front"], basis),
        VIEW_DIRECTIONS["front"],
    )
    _assert_vector_close(
        _analysis_to_world(VIEW_DIRECTIONS["iso"], basis),
        VIEW_DIRECTIONS["iso"],
    )


def test_x_up_remaps_entire_canonical_camera_frame() -> None:
    basis = _analysis_basis((1.0, 0.0, 0.0), (0.0, -1.0, 0.0))

    _assert_vector_close(
        _analysis_to_world(VIEW_DIRECTIONS["iso"], basis),
        (0.55, -0.8, -0.8),
    )
    _assert_vector_close(
        _analysis_to_world(VIEW_DIRECTIONS["top"], basis),
        (1.0, 0.0, 0.0),
    )


def test_parallel_analysis_front_and_up_are_rejected() -> None:
    with pytest.raises(ValueError, match="front axis cannot be parallel"):
        _analysis_basis((1.0, 0.0, 0.0), (-1.0, 0.0, 0.0))


def test_old_render_evidence_is_cleared_before_rerender() -> None:
    registry = {
        "parts": [
            {
                "part_id": "P0001",
                "renders": [{"view_id": "old"}],
                "isolated_evidence": {"path": "/tmp/stale.png"},
            },
            {"part_id": "P0002"},
        ],
        "render_set": {"views": [{"view_id": "old"}]},
    }

    _clear_render_evidence(registry)

    assert [part["renders"] for part in registry["parts"]] == [[], []]
    assert all("isolated_evidence" not in part for part in registry["parts"])
    assert "render_set" not in registry


def test_requested_physical_up_is_preserved_for_orthogonal_view() -> None:
    assert _camera_up_axis((0.0, 0.0, 1.0), (1.0, 0.0, 0.0)) == (
        1.0,
        0.0,
        0.0,
    )


def test_parallel_physical_up_uses_stable_fallback() -> None:
    direction = _normalize((1.0, 0.0, 0.15))
    camera_up = _camera_up_axis(direction, (1.0, 0.0, 0.0))
    dot = sum(direction[index] * camera_up[index] for index in range(3))
    assert math.isclose(dot, 0.0, abs_tol=1e-9)
    assert camera_up == (0.0, 1.0, 0.0)


def test_zero_up_axis_is_rejected() -> None:
    with pytest.raises(ValueError, match="cannot be zero"):
        _camera_up_axis((0.0, 0.0, 1.0), (0.0, 0.0, 0.0))


def test_custom_view_specs_support_continuous_camera_parameters(tmp_path) -> None:
    path = tmp_path / "views.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "qwen-camera-view-specs/v1",
                "views": [
                    {
                        "view_id": "calibrated_iso",
                        "analysis_direction": [1.0, 1.0, 0.4],
                        "analysis_up_axis": [0.0, 0.0, 1.0],
                        "focal_length_mm": 52.0,
                        "distance_multiplier": 2.8,
                        "target_offset_u": 0.12,
                        "target_offset_v": -0.08,
                        "projection_mode": "orthographic",
                        "orthographic_span_multiplier": 1.8,
                        "calibration": {"reference_view_id": "iso"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    specs = _load_custom_view_specs(path)

    assert list(specs) == ["calibrated_iso"]
    assert math.isclose(
        sum(value * value for value in specs["calibrated_iso"]["analysis_direction"]),
        1.0,
    )
    assert specs["calibrated_iso"]["focal_length_mm"] == 52.0
    assert specs["calibrated_iso"]["distance_multiplier"] == 2.8
    assert specs["calibrated_iso"]["target_offset_u"] == 0.12
    assert specs["calibrated_iso"]["target_offset_v"] == -0.08
    assert specs["calibrated_iso"]["projection_mode"] == "orthographic"
    assert specs["calibrated_iso"]["orthographic_span_multiplier"] == 1.8


def test_custom_view_specs_reject_parallel_up_axis(tmp_path) -> None:
    path = tmp_path / "views.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "qwen-camera-view-specs/v1",
                "views": [
                    {
                        "view_id": "bad",
                        "analysis_direction": [0.0, 0.0, 1.0],
                        "analysis_up_axis": [0.0, 0.0, 2.0],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="parallel"):
        _load_custom_view_specs(path)


def test_highlighted_crop_preserves_target_and_marks_geometry_only() -> None:
    import numpy as np
    from PIL import Image

    pixels = np.full((80, 100, 3), (180, 200, 220), dtype=np.uint8)
    mask = np.zeros((80, 100), dtype=bool)
    mask[30:50, 40:60] = True
    crop = _highlighted_context_crop(Image.fromarray(pixels), mask, "P0042")

    assert crop is not None
    assert crop.width > 20
    assert crop.height > 20
    crop_pixels = np.asarray(crop)
    assert (crop_pixels[..., 0] > crop_pixels[..., 1] * 2).any()


def test_isolated_target_crop_keeps_raw_count_and_neutralizes_cad_color() -> None:
    import numpy as np
    from PIL import Image

    pixels = np.full((40, 50, 3), (12, 190, 245), dtype=np.uint8)
    mask = np.zeros((40, 50), dtype=bool)
    mask[17:20, 22:26] = True

    result = _isolated_target_crop(
        Image.fromarray(pixels),
        mask,
        "P0042",
        "front",
        canvas_size=160,
        target_long_edge=96,
    )

    assert result is not None
    crop, metadata = result
    assert crop.size == (160, 198)
    assert metadata["source_visible_pixels"] == 12
    assert metadata["normalized_visible_pixels"] > 12
    assert metadata["material_neutralized"] is True
    assert metadata["background_removed"] is True
    crop_pixels = np.asarray(crop)
    neutral_target = crop_pixels[60:150, 40:120]
    non_outline = neutral_target[
        np.max(neutral_target, axis=2) - np.min(neutral_target, axis=2) < 3
    ]
    assert len(non_outline) > 0


def test_showcase_scene_places_ground_below_x_up_asset() -> None:
    basis = _analysis_basis((1.0, 0.0, 0.0), (0.0, -1.0, 0.0))
    spec = _showcase_scene_spec(
        minimum=(-2.0, -1.0, -0.5),
        maximum=(3.0, 1.0, 0.5),
        center=(0.5, 0.0, 0.0),
        diagonal=math.sqrt(30.0),
        basis=basis,
    )

    assert spec["ground_normal"] == (1.0, 0.0, 0.0)
    ground_heights = {round(point[0], 12) for point in spec["ground_corners"]}
    assert len(ground_heights) == 1
    assert next(iter(ground_heights)) < -2.0
    assert len(spec["lights"]) == 3
    assert all(light["radius"] > 0 for light in spec["lights"])


def test_showcase_ground_winding_matches_requested_up() -> None:
    basis = _analysis_basis((0.0, 0.0, 1.0), (0.0, -1.0, 0.0))
    spec = _showcase_scene_spec(
        minimum=(-1.0, -1.0, -1.0),
        maximum=(1.0, 1.0, 1.0),
        center=(0.0, 0.0, 0.0),
        diagonal=math.sqrt(12.0),
        basis=basis,
    )
    p0, p1, p2, _ = spec["ground_corners"]
    first_edge = tuple(p1[index] - p0[index] for index in range(3))
    second_edge = tuple(p2[index] - p1[index] for index in range(3))

    _assert_vector_close(_normalize(_cross(first_edge, second_edge)), (0.0, 0.0, 1.0))


def test_showcase_is_strictly_opt_in() -> None:
    parameter = inspect.signature(render_part_views).parameters["showcase"]

    assert parameter.default is False


def test_part_evidence_is_default_on_but_can_be_disabled_for_sealed_replay() -> None:
    parameter = inspect.signature(render_part_views).parameters[
        "generate_part_evidence"
    ]

    assert parameter.default is True


def test_material_neutral_lighting_is_explicitly_available() -> None:
    parameter = inspect.signature(render_part_views).parameters["lighting_profile"]

    assert parameter.default == "geometry"
    assert EVIDENCE_LIGHTING_PROFILES == ("geometry", "material-neutral")


@pytest.mark.parametrize("rt_subframes", [0, -1, True, 1.5])
def test_rt_subframes_must_be_a_positive_integer(
    tmp_path, rt_subframes: object
) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        render_part_views(
            registry_path=tmp_path / "unused.json",
            output_dir=tmp_path / "renders",
            rt_subframes=rt_subframes,  # type: ignore[arg-type]
        )


def test_renderer_startup_does_not_create_an_unused_usd_stage() -> None:
    assert _simulation_app_launch_config() == {
        "headless": True,
        "create_new_stage": False,
    }


def test_pose_bank_expands_to_deterministic_upper_hemisphere_views() -> None:
    directions, presets = _resolve_view_directions(["pose-bank-26"])

    assert presets == ["pose-bank-26"]
    assert len(directions) == 26
    assert tuple(directions) == VIEW_PRESETS["pose-bank-26"]
    assert set(VIEW_DIRECTIONS) <= set(directions)
    assert all(_normalize(direction)[2] >= 0.0 for direction in directions.values())


def test_pose_bank_can_be_combined_without_duplicate_views() -> None:
    directions, presets = _resolve_view_directions(
        ["front", "pose-bank-26", "front"]
    )

    assert presets == ["pose-bank-26"]
    assert len(directions) == 26


def test_render_progress_is_machine_readable_and_counts_completed_work() -> None:
    stream = io.StringIO()

    captured = list(
        _counted_progress_items(
            (1, 2),
            progress_callback=lambda event: emit_progress_event(event, stream=stream),
            stage=RENDER_CAPTURE_PROGRESS_STAGE,
            unit="steps",
        )
    )
    viewed = list(
        _counted_progress_items(
            ("front", "side", "top"),
            progress_callback=lambda event: emit_progress_event(event, stream=stream),
            stage=RENDER_VIEWS_PROGRESS_STAGE,
            unit="views",
            detail=lambda view: f"Rendered view {view}",
        )
    )

    assert captured == [1, 2]
    assert viewed == ["front", "side", "top"]
    events = [
        parse_progress_line(line)
        for line in stream.getvalue().splitlines(keepends=True)
    ]
    assert all(event is not None for event in events)
    assert [
        (
            event["stage"],
            event["state"],
            event["current"],
            event["total"],
            event["unit"],
        )
        for event in events
        if event is not None
    ] == [
        ("render_capture", "start", 0, 2, "steps"),
        ("render_capture", "update", 1, 2, "steps"),
        ("render_capture", "update", 2, 2, "steps"),
        ("render_capture", "complete", 2, 2, "steps"),
        ("render_views", "start", 0, 3, "views"),
        ("render_views", "update", 1, 3, "views"),
        ("render_views", "update", 2, 3, "views"),
        ("render_views", "update", 3, 3, "views"),
        ("render_views", "complete", 3, 3, "views"),
    ]


def test_render_progress_is_optional_and_failed_work_never_completes() -> None:
    assert list(
        _counted_progress_items(
            ("front",),
            progress_callback=None,
            stage=RENDER_VIEWS_PROGRESS_STAGE,
            unit="views",
        )
    ) == ["front"]

    events: list[dict[str, object]] = []
    with pytest.raises(RuntimeError, match="synthetic render failure"):
        for view in _counted_progress_items(
            ("front", "side"),
            progress_callback=events.append,
            stage=RENDER_VIEWS_PROGRESS_STAGE,
            unit="views",
        ):
            if view == "side":
                raise RuntimeError("synthetic render failure")

    assert [
        (event["state"], event["current"], event["total"]) for event in events
    ] == [
        ("start", 0, 2),
        ("update", 1, 2),
    ]
