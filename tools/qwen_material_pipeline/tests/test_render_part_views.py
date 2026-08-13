from __future__ import annotations

import io
import inspect
import json
import math
from collections.abc import Callable

import cv2
import numpy as np
import pytest

from qwen_material_pipeline.core.progress import emit_progress_event, parse_progress_line
from qwen_material_pipeline.usd import render as render_module
from qwen_material_pipeline.usd.render import (
    EVIDENCE_LIGHTING_PROFILES,
    RENDER_CAPTURE_PROGRESS_STAGE,
    RENDER_VIEWS_PROGRESS_STAGE,
    VIEW_DIRECTIONS,
    VIEW_PRESETS,
    _analysis_basis,
    _analysis_to_world,
    _camera_up_axis,
    _apply_radial_distortion,
    _clear_render_evidence,
    _counted_progress_items,
    _cross,
    _highlighted_context_crop,
    _isolated_target_crop,
    _load_assembly_pose_overrides,
    _load_whole_asset_pose_override,
    _lowest_common_part_xform_path,
    _load_custom_view_specs,
    _normalize,
    _RenderCleanupError,
    _RenderLifecycle,
    _resolve_view_directions,
    _showcase_scene_spec,
    _simulation_app_launch_config,
    render_part_views,
)


class _FakePrim:
    def __init__(self, type_name: str, *, instance_proxy: bool = False) -> None:
        self._type_name = type_name
        self._instance_proxy = instance_proxy

    def __bool__(self) -> bool:
        return bool(self._type_name)

    def GetTypeName(self) -> str:
        return self._type_name

    def IsInstanceProxy(self) -> bool:
        return self._instance_proxy


class _FakeStage:
    def __init__(self, prims: dict[str, _FakePrim]) -> None:
        self._prims = prims

    def GetPrimAtPath(self, path: str) -> _FakePrim:
        return self._prims.get(path, _FakePrim(""))


def test_assembly_pose_overrides_require_rigid_subtree_translation(tmp_path) -> None:
    path = tmp_path / "overrides.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "qwen-assembly-pose-overrides/v1",
                "overrides": [
                    {
                        "prim_path": "/Asset/Accessory",
                        "world_translation": [0.1, -0.2, 0.3],
                    }
                ],
            }
        )
    )

    assert _load_assembly_pose_overrides(path) == [
        {
            "prim_path": "/Asset/Accessory",
            "world_translation": [0.1, -0.2, 0.3],
        }
    ]


@pytest.mark.parametrize(
    "override",
    [
        {"prim_path": "relative", "world_translation": [0.0, 0.0, 0.0]},
        {"prim_path": "/Asset", "world_translation": [0.0, float("nan"), 0.0]},
        {"prim_path": "/Asset", "world_translation": [0.0, 0.0]},
    ],
)
def test_assembly_pose_overrides_fail_closed(tmp_path, override) -> None:
    path = tmp_path / "invalid.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "qwen-assembly-pose-overrides/v1",
                "overrides": [override],
            }
        )
    )

    with pytest.raises(ValueError, match="Assembly pose override"):
        _load_assembly_pose_overrides(path)


def test_whole_asset_pose_requires_one_finite_se3_about_asset_center(
    tmp_path,
) -> None:
    path = tmp_path / "whole_asset_pose.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "qwen-whole-asset-pose-override/v1",
                "asset_root_prim_path": "/Asset/RegisteredRoot",
                "world_translation": [0.1, -0.2, 0.3],
                "world_rotation_rotvec_degrees": [1.0, 2.0, -3.0],
                "pivot": "asset_bounds_center",
            }
        )
    )

    assert _load_whole_asset_pose_override(path) == {
        "asset_root_prim_path": "/Asset/RegisteredRoot",
        "world_translation": [0.1, -0.2, 0.3],
        "world_rotation_rotvec_degrees": [1.0, 2.0, -3.0],
        "pivot": "asset_bounds_center",
    }


@pytest.mark.parametrize(
    "update, message",
    [
        ({"asset_root_prim_path": "Asset/Part"}, "absolute asset root"),
        ({"world_translation": [0.0, 0.0]}, "finite translation"),
        (
            {"world_rotation_rotvec_degrees": [0.0, float("nan"), 0.0]},
            "finite rotation vector",
        ),
        ({"pivot": "world_origin"}, "asset bounds center"),
    ],
)
def test_whole_asset_pose_fails_closed(tmp_path, update, message) -> None:
    document = {
        "schema_version": "qwen-whole-asset-pose-override/v1",
        "asset_root_prim_path": "/Asset",
        "world_translation": [0.0, 0.0, 0.0],
        "world_rotation_rotvec_degrees": [0.0, 0.0, 0.0],
        "pivot": "asset_bounds_center",
    }
    document.update(update)
    path = tmp_path / "invalid_whole_asset_pose.json"
    path.write_text(json.dumps(document))

    with pytest.raises(ValueError, match=message):
        _load_whole_asset_pose_override(path)


def test_whole_asset_pose_root_is_lowest_common_registered_xform() -> None:
    stage = _FakeStage(
        {
            "/Wrapper": _FakePrim("Xform"),
            "/Wrapper/Asset": _FakePrim("Xform"),
            "/Wrapper/Asset/BranchA": _FakePrim("Xform"),
            "/Wrapper/Asset/BranchA/Mesh": _FakePrim("Mesh"),
            "/Wrapper/Asset/BranchB": _FakePrim("Xform"),
            "/Wrapper/Asset/BranchB/Mesh": _FakePrim("Mesh"),
        }
    )

    assert _lowest_common_part_xform_path(
        stage=stage,
        part_paths=[
            "/Wrapper/Asset/BranchA/Mesh",
            "/Wrapper/Asset/BranchB/Mesh",
        ],
    ) == "/Wrapper/Asset"


def test_whole_asset_pose_root_rejects_duplicate_or_unrooted_parts() -> None:
    stage = _FakeStage({"/Asset": _FakePrim("Xform")})

    with pytest.raises(ValueError, match="unique"):
        _lowest_common_part_xform_path(
            stage=stage, part_paths=["/Asset/Mesh", "/Asset/Mesh"]
        )
    with pytest.raises(ValueError, match="share"):
        _lowest_common_part_xform_path(
            stage=stage, part_paths=["/Asset/Mesh", "/Other/Mesh"]
        )


class _FakeContext:
    def __init__(
        self,
        events: list[str],
        name: str,
        *,
        close_result: object = True,
        close_after_updates: int = 0,
    ) -> None:
        self.events = events
        self.name = name
        self.close_result = close_result
        self.close_after_updates = close_after_updates
        self._stage: object | None = object()
        self._close_requested = False
        self._post_close_updates = 0

    def close_stage(self) -> object:
        self.events.append(f"{self.name}.close")
        if self.close_result is True:
            self._close_requested = True
            if self.close_after_updates == 0:
                self._stage = None
        return self.close_result

    def get_stage(self) -> object | None:
        return self._stage

    def advance_kit_update(self) -> None:
        if not self._close_requested or self._stage is None:
            return
        self._post_close_updates += 1
        if self._post_close_updates >= self.close_after_updates:
            self._stage = None


class _FakeKitApp:
    def __init__(
        self,
        events: list[str],
        *,
        on_update: Callable[[], None] | None = None,
        update_error: BaseException | None = None,
    ) -> None:
        self.events = events
        self.on_update = on_update
        self.update_error = update_error

    def update(self) -> None:
        self.events.append("kit.update")
        if self.update_error is not None:
            raise self.update_error
        if self.on_update is not None:
            self.on_update()


class _FakeRenderProduct:
    def __init__(self, events: list[str], name: str) -> None:
        self.events = events
        self.name = name
        self.destroy_calls = 0

    def destroy(self) -> None:
        self.destroy_calls += 1
        self.events.append(f"{self.name}.destroy")


class _FakeAnnotator:
    def __init__(
        self,
        events: list[str],
        name: str,
        *,
        attach_error: BaseException | None = None,
    ) -> None:
        self.events = events
        self.name = name
        self.attach_error = attach_error

    def attach(self, render_product: _FakeRenderProduct) -> None:
        self.events.append(f"{self.name}.attach:{render_product.name}")
        if self.attach_error is not None:
            raise self.attach_error

    def detach(self, render_product: _FakeRenderProduct) -> None:
        self.events.append(f"{self.name}.detach:{render_product.name}")


class _FakeOrchestrator:
    def __init__(
        self,
        events: list[str],
        *,
        wait_error: BaseException | None = None,
    ) -> None:
        self.events = events
        self.wait_error = wait_error
        self.state = "STOPPED"

    def step(self, **kwargs: object) -> None:
        assert self.state == "STOPPED"
        self.events.append(f"orchestrator.step:{kwargs['batch']}")
        self.state = "STARTED"

    def stop(self) -> None:
        self.events.append("orchestrator.stop")
        self.state = "STOPPING"

    def wait_until_complete(self) -> None:
        self.events.append("orchestrator.wait")
        if self.wait_error is not None:
            raise self.wait_error
        self.state = "STOPPED"


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
                        "roll_degrees": 2.5,
                        "principal_point_u": 0.03,
                        "principal_point_v": -0.02,
                        "radial_distortion_k1": 0.08,
                        "radial_distortion_k2": -0.01,
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
    assert specs["calibrated_iso"]["roll_degrees"] == 2.5
    assert specs["calibrated_iso"]["principal_point_u"] == 0.03
    assert specs["calibrated_iso"]["principal_point_v"] == -0.02
    assert specs["calibrated_iso"]["radial_distortion_k1"] == 0.08
    assert specs["calibrated_iso"]["radial_distortion_k2"] == -0.01
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


def test_radial_distortion_is_identity_at_zero_and_moves_outer_pixels() -> None:
    pixels = np.zeros((101, 101), dtype=np.uint16)
    pixels[50, 85] = 7

    identity = _apply_radial_distortion(
        pixels,
        k1=0.0,
        k2=0.0,
        interpolation=cv2.INTER_NEAREST,
    )
    distorted = _apply_radial_distortion(
        pixels,
        k1=0.20,
        k2=0.0,
        interpolation=cv2.INTER_NEAREST,
    )

    assert np.array_equal(identity, pixels)
    assert int(np.count_nonzero(distorted == 7)) >= 1
    assert int(np.where(distorted == 7)[1].max()) > 85


def test_radial_distortion_preserves_replicator_uint32_semantic_ids() -> None:
    pixels = np.zeros((101, 101), dtype=np.uint32)
    pixels[50, 85] = np.uint32(2_000_000_007)

    distorted = _apply_radial_distortion(
        pixels,
        k1=0.20,
        k2=0.0,
        interpolation=cv2.INTER_NEAREST,
    )

    assert distorted.dtype == np.uint32
    assert 2_000_000_007 in distorted


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


def test_render_lifecycle_is_fully_stopped_before_sequential_invocations(
    monkeypatch,
) -> None:
    events: list[str] = []
    orchestrator = _FakeOrchestrator(events)
    invocation = 0
    products: list[_FakeRenderProduct] = []

    def fake_render_once(**kwargs: object) -> dict[str, object]:
        nonlocal invocation
        invocation += 1
        lifecycle = kwargs["_lifecycle"]
        assert isinstance(lifecycle, _RenderLifecycle)
        context = _FakeContext(events, f"context{invocation}")
        product = _FakeRenderProduct(events, f"product{invocation}")
        products.append(product)
        annotator = _FakeAnnotator(events, f"annotator{invocation}")
        lifecycle.register_open_stage(context)
        lifecycle.attach_annotator(annotator, product)
        lifecycle.step(orchestrator, batch=invocation)
        return {"invocation": invocation}

    monkeypatch.setattr(render_module, "_render_part_views_once", fake_render_once)
    monkeypatch.setattr(
        render_module,
        "_get_kit_app",
        lambda: _FakeKitApp(events),
    )

    first = render_module.render_part_views(
        registry_path="unused.json", output_dir="unused"
    )
    second = render_module.render_part_views(
        registry_path="unused.json", output_dir="unused"
    )

    assert first == {"invocation": 1}
    assert second == {"invocation": 2}
    assert orchestrator.state == "STOPPED"
    assert [product.destroy_calls for product in products] == [0, 0]
    assert events == [
        "annotator1.attach:product1",
        "orchestrator.step:1",
        "orchestrator.stop",
        "orchestrator.wait",
        "annotator1.detach:product1",
        "context1.close",
        "kit.update",
        "kit.update",
        "annotator2.attach:product2",
        "orchestrator.step:2",
        "orchestrator.stop",
        "orchestrator.wait",
        "annotator2.detach:product2",
        "context2.close",
        "kit.update",
        "kit.update",
    ]


def test_render_body_exception_still_releases_every_resource(monkeypatch) -> None:
    events: list[str] = []
    orchestrator = _FakeOrchestrator(events)
    products: list[_FakeRenderProduct] = []

    def failing_render_once(**kwargs: object) -> dict[str, object]:
        lifecycle = kwargs["_lifecycle"]
        assert isinstance(lifecycle, _RenderLifecycle)
        context = _FakeContext(events, "context")
        product = _FakeRenderProduct(events, "product")
        products.append(product)
        annotator = _FakeAnnotator(events, "annotator")
        lifecycle.register_open_stage(context)
        lifecycle.attach_annotator(annotator, product)
        lifecycle.step(orchestrator, batch="failure")
        raise ValueError("synthetic render body failure")

    monkeypatch.setattr(render_module, "_render_part_views_once", failing_render_once)
    monkeypatch.setattr(
        render_module,
        "_get_kit_app",
        lambda: _FakeKitApp(events),
    )

    with pytest.raises(ValueError, match="synthetic render body failure"):
        render_module.render_part_views(
            registry_path="unused.json", output_dir="unused"
        )

    assert products[0].destroy_calls == 0
    assert events[-6:] == [
        "orchestrator.stop",
        "orchestrator.wait",
        "annotator.detach:product",
        "context.close",
        "kit.update",
        "kit.update",
    ]


def test_partial_annotator_attach_is_registered_before_failure(monkeypatch) -> None:
    events: list[str] = []
    products: list[_FakeRenderProduct] = []

    def failing_attach_once(**kwargs: object) -> dict[str, object]:
        lifecycle = kwargs["_lifecycle"]
        assert isinstance(lifecycle, _RenderLifecycle)
        context = _FakeContext(events, "context")
        product = _FakeRenderProduct(events, "product")
        products.append(product)
        annotator = _FakeAnnotator(
            events,
            "annotator",
            attach_error=RuntimeError("synthetic attach failure"),
        )
        lifecycle.register_open_stage(context)
        lifecycle.attach_annotator(annotator, product)
        raise AssertionError("unreachable")

    monkeypatch.setattr(render_module, "_render_part_views_once", failing_attach_once)
    monkeypatch.setattr(
        render_module,
        "_get_kit_app",
        lambda: _FakeKitApp(events),
    )

    with pytest.raises(RuntimeError, match="synthetic attach failure"):
        render_module.render_part_views(
            registry_path="unused.json", output_dir="unused"
        )

    assert products[0].destroy_calls == 0
    assert events == [
        "annotator.attach:product",
        "annotator.detach:product",
        "context.close",
        "kit.update",
        "kit.update",
    ]


def test_wait_failure_still_detaches_closes_and_drains_stage(monkeypatch) -> None:
    events: list[str] = []
    orchestrator = _FakeOrchestrator(
        events,
        wait_error=RuntimeError("synthetic wait failure"),
    )
    products: list[_FakeRenderProduct] = []

    def successful_render_once(**kwargs: object) -> dict[str, object]:
        lifecycle = kwargs["_lifecycle"]
        assert isinstance(lifecycle, _RenderLifecycle)
        context = _FakeContext(events, "context")
        product = _FakeRenderProduct(events, "product")
        products.append(product)
        annotator = _FakeAnnotator(events, "annotator")
        lifecycle.register_open_stage(context)
        lifecycle.attach_annotator(annotator, product)
        lifecycle.step(orchestrator, batch="wait-failure")
        return {"rendered": True}

    monkeypatch.setattr(
        render_module, "_render_part_views_once", successful_render_once
    )
    monkeypatch.setattr(
        render_module,
        "_get_kit_app",
        lambda: _FakeKitApp(events),
    )

    with pytest.raises(_RenderCleanupError, match="synthetic wait failure") as raised:
        render_module.render_part_views(
            registry_path="unused.json", output_dir="unused"
        )

    assert [operation for operation, _ in raised.value.failures] == [
        "orchestrator.wait_until_complete"
    ]
    assert products[0].destroy_calls == 0
    assert events[-4:] == [
        "annotator.detach:product",
        "context.close",
        "kit.update",
        "kit.update",
    ]


def test_close_stage_must_return_literal_true() -> None:
    events: list[str] = []
    lifecycle = _RenderLifecycle()
    lifecycle.register_open_stage(_FakeContext(events, "context", close_result=None))

    with pytest.raises(_RenderCleanupError, match="did not return True"):
        lifecycle.cleanup()

    assert events == ["context.close"]


def test_stage_close_is_drained_until_empty_after_minimum_updates(monkeypatch) -> None:
    events: list[str] = []
    context = _FakeContext(events, "context", close_after_updates=3)
    app = _FakeKitApp(events, on_update=context.advance_kit_update)
    lifecycle = _RenderLifecycle()
    lifecycle.register_open_stage(context)
    monkeypatch.setattr(render_module, "_get_kit_app", lambda: app)

    lifecycle.cleanup()

    assert context.get_stage() is None
    assert events == [
        "context.close",
        "kit.update",
        "kit.update",
        "kit.update",
    ]


def test_post_close_update_failure_is_fail_closed(monkeypatch) -> None:
    events: list[str] = []
    context = _FakeContext(events, "context")
    app = _FakeKitApp(
        events,
        update_error=RuntimeError("synthetic Kit update failure"),
    )
    lifecycle = _RenderLifecycle()
    lifecycle.register_open_stage(context)
    monkeypatch.setattr(render_module, "_get_kit_app", lambda: app)

    with pytest.raises(
        _RenderCleanupError, match="synthetic Kit update failure"
    ) as raised:
        lifecycle.cleanup()

    assert [operation for operation, _ in raised.value.failures] == [
        "usd_context.await_stage_closed"
    ]
    assert events == ["context.close", "kit.update"]


def test_post_close_stage_timeout_is_fail_closed(monkeypatch) -> None:
    events: list[str] = []
    context = _FakeContext(events, "context", close_after_updates=100)
    app = _FakeKitApp(events, on_update=context.advance_kit_update)
    lifecycle = _RenderLifecycle()
    lifecycle.register_open_stage(context)
    monkeypatch.setattr(render_module, "POST_CLOSE_MAX_KIT_UPDATES", 3)
    monkeypatch.setattr(render_module, "_get_kit_app", lambda: app)

    with pytest.raises(_RenderCleanupError, match="remained open") as raised:
        lifecycle.cleanup()

    assert [operation for operation, _ in raised.value.failures] == [
        "usd_context.await_stage_closed"
    ]
    assert events == [
        "context.close",
        "kit.update",
        "kit.update",
        "kit.update",
    ]


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
