from __future__ import annotations

import io
import json
import math
from pathlib import Path

import pytest

import qwen_material_pipeline.evidence.face_regions as face_region_evidence
from qwen_material_pipeline.core.progress import emit_progress_event, parse_progress_line
from qwen_material_pipeline.evidence.face_regions import (
    ProjectionMesh,
    _inclusive_ranges,
    _normal_coherent_components,
    _rasterize_region_labels,
    _region_colors,
    analyze_mesh_topology,
)


def test_surface_regions_do_not_accumulate_transitive_normal_drift() -> None:
    angle = math.radians(30.0)
    normals = [
        (0.0, 0.0, 1.0),
        (math.sin(angle), 0.0, math.cos(angle)),
        (math.sin(2.0 * angle), 0.0, math.cos(2.0 * angle)),
    ]

    assert _normal_coherent_components(
        [{1}, {0, 2}, {1}],
        normals,
        math.cos(math.radians(35.0)),
    ) == [[0, 1], [2]]


def test_face_ranges_are_deterministic_and_inclusive() -> None:
    assert _inclusive_ranges([8, 2, 3, 4, 10]) == [[2, 4], [8, 8], [10, 10]]
    with pytest.raises(ValueError, match="unique"):
        _inclusive_ranges([1, 1])


def test_coplanar_triangles_form_one_geometric_patch() -> None:
    evidence, faces, patch_by_face = analyze_mesh_topology(
        points_world=[(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)],
        face_vertex_counts=[3, 3],
        face_vertex_indices=[0, 1, 2, 0, 2, 3],
    )

    assert faces == [(0, 1, 2), (0, 2, 3)]
    assert patch_by_face == [0, 0]
    assert evidence["raw_topology_component_count"] == 1
    assert evidence["welded_topology_component_count"] == 1
    assert evidence["surface_patch_count"] == 1
    patch = evidence["surface_patches"][0]
    assert patch["face_indices"] == [0, 1]
    assert patch["face_ranges"] == [[0, 1]]
    assert patch["candidate_kind"] == "unclassified_geometric_surface_patch"
    assert math.isclose(patch["area_world"], 1.0)
    assert patch["mean_normal_world"] == [0.0, 0.0, 1.0]


def test_coordinate_weld_repairs_duplicate_cad_seam_vertices() -> None:
    evidence, _, patch_by_face = analyze_mesh_topology(
        points_world=[
            (0, 0, 0),
            (1, 0, 0),
            (1, 1, 0),
            (0, 0, 0),
            (1, 1, 0),
            (0, 1, 0),
        ],
        face_vertex_counts=[3, 3],
        face_vertex_indices=[0, 1, 2, 3, 4, 5],
    )

    assert evidence["raw_topology_component_count"] == 2
    assert evidence["welded_topology_component_count"] == 1
    assert evidence["surface_patch_count"] == 1
    assert evidence["weld_audit"]["merged_point_count"] == 2
    assert patch_by_face == [0, 0]


def test_crease_angle_splits_right_angle_faces() -> None:
    evidence, _, patch_by_face = analyze_mesh_topology(
        points_world=[(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)],
        face_vertex_counts=[3, 3],
        face_vertex_indices=[0, 1, 2, 1, 0, 3],
        crease_angle_degrees=35.0,
    )

    assert evidence["welded_topology_component_count"] == 1
    assert evidence["surface_patch_count"] == 2
    assert patch_by_face == [0, 1]
    assert evidence["surface_patch_adjacency_count"] == 1
    adjacency = evidence["surface_patches"][0]["adjacent_patches"][0]
    assert adjacency["region_id"] == "R0002"
    assert adjacency["shared_edge_count"] == 1
    assert math.isclose(adjacency["mean_dihedral_degrees"], 90.0)


@pytest.mark.parametrize(
    ("counts", "indices", "message"),
    [
        ([3], [0, 1], "exactly cover"),
        ([3], [0, 1, 3], "outside"),
        ([3], [0, 1, 1], "repeated"),
        ([2], [0, 1], "at least three"),
    ],
)
def test_invalid_topology_fails_closed(
    counts: list[int], indices: list[int], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        analyze_mesh_topology(
            points_world=[(0, 0, 0), (1, 0, 0), (0, 1, 0)],
            face_vertex_counts=counts,
            face_vertex_indices=indices,
        )


def test_overlarge_weld_tolerance_fails_if_a_face_collapses() -> None:
    with pytest.raises(ValueError, match="collapses a nondegenerate face"):
        analyze_mesh_topology(
            points_world=[(0, 0, 0), (0.001, 0, 0), (0, 1, 0)],
            face_vertex_counts=[3],
            face_vertex_indices=[0, 1, 2],
            weld_tolerance_ratio=0.1,
        )


def test_projection_z_buffer_keeps_nearest_region() -> None:
    far = ProjectionMesh(
        part_id="P0001",
        points_world=[(-0.2, -0.2, 3), (0.2, -0.2, 3), (0, 0.2, 3)],
        face_vertices=[(0, 1, 2)],
        face_labels=[1],
    )
    near = ProjectionMesh(
        part_id="P0002",
        points_world=[(-0.2, -0.2, 2), (0.2, -0.2, 2), (0, 0.2, 2)],
        face_vertices=[(0, 1, 2)],
        face_labels=[2],
    )

    labels = _rasterize_region_labels(
        [far, near],
        camera_position=(0, 0, 0),
        world_direction=(0, 0, -1),
        camera_up_axis=(0, 1, 0),
        width=64,
        height=64,
        focal_length_mm=45.0,
        horizontal_aperture_mm=20.955,
    )

    assert labels[32, 32] == 2
    assert 1 not in set(labels.ravel())


def test_orthographic_projection_uses_recorded_span_instead_of_focal_length() -> None:
    mesh = ProjectionMesh(
        part_id="P0001",
        points_world=[(-0.5, -0.5, 2), (0.5, -0.5, 2), (0.0, 0.5, 2)],
        face_vertices=[(0, 1, 2)],
        face_labels=[1],
    )

    first = _rasterize_region_labels(
        [mesh],
        camera_position=(0, 0, 0),
        world_direction=(0, 0, -1),
        camera_up_axis=(0, 1, 0),
        width=64,
        height=64,
        focal_length_mm=20.0,
        horizontal_aperture_mm=20.955,
        projection_mode="orthographic",
        orthographic_span_multiplier=2.0,
        asset_diagonal=2.0,
    )
    second = _rasterize_region_labels(
        [mesh],
        camera_position=(0, 0, -3),
        world_direction=(0, 0, -1),
        camera_up_axis=(0, 1, 0),
        width=64,
        height=64,
        focal_length_mm=900.0,
        horizontal_aperture_mm=20.955,
        projection_mode="orthographic",
        orthographic_span_multiplier=2.0,
        asset_diagonal=2.0,
    )

    assert first.tolist() == second.tolist()
    assert int((first > 0).sum()) > 0


def test_projection_camera_frame_prefers_recorded_look_at_target() -> None:
    _, _, _, forward = face_region_evidence._camera_frame(
        (0, 0, 0),
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 2),
    )

    assert forward == (0.0, 0.0, 1.0)


def test_region_colors_are_deterministic_and_unique() -> None:
    first = _region_colors(["P0001:R0001", "P0001:R0002", "P0002:R0001"])
    second = _region_colors(["P0001:R0001", "P0001:R0002", "P0002:R0001"])

    assert first == second
    assert len(set(first.values())) == 3


def test_atomic_output_cleans_temporary_directory_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = tmp_path / "registry.json"
    registry.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "evidence"

    def fail_loader(**_: object) -> None:
        raise ValueError("synthetic validation failure")

    monkeypatch.setattr(face_region_evidence, "_load_registered_meshes", fail_loader)
    with pytest.raises(ValueError, match="synthetic validation failure"):
        face_region_evidence.build_face_region_evidence(
            registry_path=registry,
            output_dir=output,
        )

    assert not output.exists()
    assert not list(tmp_path.glob(".evidence.tmp-*"))


def _stub_face_region_inputs(
    *,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    view_ids: list[str],
) -> tuple[Path, Path, Path]:
    registry = tmp_path / "registry.json"
    registry.write_text("{}\n", encoding="utf-8")
    rendered_registry = tmp_path / "rendered_registry.json"
    rendered_registry.write_text("{}\n", encoding="utf-8")
    asset = tmp_path / "asset.usda"
    asset.write_text("#usda 1.0\n", encoding="utf-8")

    manifest = {
        "part_count": 1,
        "face_count": 1,
        "welded_topology_component_count": 1,
        "surface_patch_count": 1,
        "parts": [
            {
                "part_id": "P0001",
                "prim_path": "/Mesh",
                "face_count": 1,
                "raw_topology_component_count": 1,
                "welded_topology_component_count": 1,
                "surface_patch_count": 1,
            }
        ],
    }
    mesh = ProjectionMesh(
        part_id="P0001",
        points_world=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
        face_vertices=[(0, 1, 2)],
        face_labels=[1],
    )
    metadata = {
        "label_metadata": {1: {"part_id": "P0001", "region_id": "R0001"}},
        "meters_per_unit": 1.0,
    }
    monkeypatch.setattr(
        face_region_evidence,
        "_load_registered_meshes",
        lambda **_: (manifest, [mesh], asset, face_region_evidence._sha256(asset), metadata),
    )
    monkeypatch.setattr(
        face_region_evidence,
        "_load_projection_views",
        lambda **_: (
            [
                {
                    "view_id": view_id,
                    "projection_resolution": (16, 16),
                    "camera_position": (0.0, 0.0, 1.0),
                    "world_direction": (0.0, 0.0, -1.0),
                    "camera_up_axis": (0.0, 1.0, 0.0),
                    "rgb_path": tmp_path / f"{view_id}.png",
                    "visible_parts": [],
                }
                for view_id in view_ids
            ],
            {
                "source_resolution": [16, 16],
                "projection_resolution": [16, 16],
            },
        ),
    )
    monkeypatch.setattr(
        face_region_evidence,
        "_rasterize_region_labels",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        face_region_evidence,
        "_write_projection_images",
        lambda **kwargs: {"view_id": kwargs["view_id"]},
    )
    return registry, rendered_registry, asset


def test_projection_progress_is_machine_readable_and_updates_after_every_view(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry, rendered_registry, _ = _stub_face_region_inputs(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        view_ids=["front", "side", "top"],
    )
    stream = io.StringIO()

    face_region_evidence.build_face_region_evidence(
        registry_path=registry,
        output_dir=tmp_path / "evidence",
        rendered_registry_path=rendered_registry,
        progress_callback=lambda event: emit_progress_event(event, stream=stream),
    )

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
        ("face_regions/views", "start", 0, 3, "views"),
        ("face_regions/views", "update", 1, 3, "views"),
        ("face_regions/views", "update", 2, 3, "views"),
        ("face_regions/views", "update", 3, 3, "views"),
        ("face_regions/views", "complete", 3, 3, "views"),
    ]


def test_projection_uses_each_registered_view_focal_length(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry, rendered_registry, _ = _stub_face_region_inputs(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        view_ids=["front", "side"],
    )
    original_loader = face_region_evidence._load_projection_views
    views, source = original_loader(
        rendered_registry_path=rendered_registry,
        asset_path=tmp_path / "asset.usda",
        requested_views=None,
        projection_max_size=512,
    )
    views[0]["focal_length_mm"] = 45.0
    views[1]["focal_length_mm"] = 60.0
    monkeypatch.setattr(
        face_region_evidence,
        "_load_projection_views",
        lambda **_: (views, source),
    )
    used_focals: list[float] = []

    def capture_focal(*_args: object, **kwargs: object) -> object:
        used_focals.append(float(kwargs["focal_length_mm"]))
        return object()

    monkeypatch.setattr(
        face_region_evidence,
        "_rasterize_region_labels",
        capture_focal,
    )

    face_region_evidence.build_face_region_evidence(
        registry_path=registry,
        output_dir=tmp_path / "evidence",
        rendered_registry_path=rendered_registry,
    )
    report = json.loads(
        (tmp_path / "evidence" / "manifest.json").read_text(encoding="utf-8")
    )

    assert used_focals == [45.0, 60.0]
    assert [
        view["focal_length_mm"]
        for view in report["projection_contract"]["views"]
    ] == [45.0, 60.0]


def test_projection_forwards_recorded_camera_projection_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry, rendered_registry, _ = _stub_face_region_inputs(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        view_ids=["front"],
    )
    original_loader = face_region_evidence._load_projection_views
    views, source = original_loader(
        rendered_registry_path=rendered_registry,
        asset_path=tmp_path / "asset.usda",
        requested_views=None,
        projection_max_size=512,
    )
    views[0].update(
        {
            "camera_projection_mode": "orthographic",
            "camera_orthographic_span_multiplier": 1.75,
            "camera_look_at_target": [0.25, 0.0, 0.0],
        }
    )
    monkeypatch.setattr(
        face_region_evidence,
        "_load_projection_views",
        lambda **_: (views, source),
    )
    captured: dict[str, object] = {}

    def capture_contract(*_args: object, **kwargs: object) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(
        face_region_evidence,
        "_rasterize_region_labels",
        capture_contract,
    )

    face_region_evidence.build_face_region_evidence(
        registry_path=registry,
        output_dir=tmp_path / "evidence",
        rendered_registry_path=rendered_registry,
    )
    report = json.loads(
        (tmp_path / "evidence" / "manifest.json").read_text(encoding="utf-8")
    )

    assert captured["projection_mode"] == "orthographic"
    assert captured["orthographic_span_multiplier"] == 1.75
    assert captured["camera_look_at_target"] == [0.25, 0.0, 0.0]
    assert captured["asset_diagonal"] == pytest.approx(math.sqrt(2.0))
    assert report["projection_contract"]["views"][0]["projection_mode"] == (
        "orthographic"
    )


def test_projection_progress_does_not_advance_a_failed_view(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry, rendered_registry, _ = _stub_face_region_inputs(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        view_ids=["front", "side"],
    )
    events: list[dict[str, object]] = []
    completed_writes = 0

    def fail_second_write(**_: object) -> dict[str, str]:
        nonlocal completed_writes
        completed_writes += 1
        if completed_writes == 2:
            raise RuntimeError("synthetic projection failure")
        return {"view_id": "front"}

    monkeypatch.setattr(
        face_region_evidence,
        "_write_projection_images",
        fail_second_write,
    )

    with pytest.raises(RuntimeError, match="synthetic projection failure"):
        face_region_evidence.build_face_region_evidence(
            registry_path=registry,
            output_dir=tmp_path / "evidence",
            rendered_registry_path=rendered_registry,
            progress_callback=events.append,
        )

    assert [
        (event["state"], event["current"], event["total"])
        for event in events
    ] == [
        ("start", 0, 2),
        ("update", 1, 2),
    ]


def test_existing_output_directory_is_never_overwritten(tmp_path: Path) -> None:
    registry = tmp_path / "registry.json"
    registry.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "evidence"
    output.mkdir()

    with pytest.raises(FileExistsError, match="already exists"):
        face_region_evidence.build_face_region_evidence(
            registry_path=registry,
            output_dir=output,
        )
