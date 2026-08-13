from __future__ import annotations

import json
from pathlib import Path

import pytest

from qwen_material_pipeline.scripts.evaluate_multiview_assembly_pose import (
    _camera_contract,
    _camera_sources,
)


def _write(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _report(
    *, tmp_path: Path, view_id: str, registry: Path, manifest: Path
) -> Path:
    specs = _write(
        tmp_path / f"{view_id}_specs.json",
        {
            "schema_version": "qwen-camera-view-specs/v1",
            "views": [
                {
                    "view_id": view_id,
                    "calibration": {"reference_view_id": view_id},
                }
            ],
        },
    )
    return _write(
        tmp_path / f"{view_id}_report.json",
        {
            "source_registry": str(registry),
            "reference_manifest": str(manifest),
            "final_view_specs": str(specs),
            "views": [
                {
                    "reference_view_id": view_id,
                    "final": {
                        "projection_iou": 0.9,
                        "boundary_p95_px": 5.0,
                        "whole_asset_similarity": {
                            "bbox_affine": [[1.0, 0.0, 2.0], [0.0, 1.0, 3.0]]
                        },
                    },
                }
            ],
        },
    )


def test_multiview_camera_contract_binds_each_sealed_affine(tmp_path: Path) -> None:
    source = {
        "asset_usd": str(_write(tmp_path / "asset.usda", {})),
        "parts": [
            {"part_id": "P1", "prim_path": "/Asset/Assembly/Mesh1"},
            {"part_id": "P2", "prim_path": "/Asset/Assembly/Mesh2"},
        ],
    }
    registry = _write(tmp_path / "registry.json", source)
    manifest = _write(tmp_path / "manifest.json", {})
    reports = {
        view_id: _report(
            tmp_path=tmp_path,
            view_id=view_id,
            registry=registry,
            manifest=manifest,
        )
        for view_id in ("front", "side")
    }

    provenance, actual_manifest, views = _camera_contract(
        camera_reports=reports,
        source_registry=source,
    )

    assert actual_manifest == manifest
    assert set(provenance) == {"front", "side"}
    assert [row["view_id"] for row in views] == ["front", "side"]
    assert all(
        row["calibration"]["frame_anchor_affine"]
        == [[1.0, 0.0, 2.0], [0.0, 1.0, 3.0]]
        for row in views
    )


def test_multiview_camera_contract_rejects_another_part_hierarchy(
    tmp_path: Path,
) -> None:
    source = {
        "asset_usd": str(_write(tmp_path / "asset.usda", {})),
        "parts": [{"part_id": "P1", "prim_path": "/Asset/Mesh1"}],
    }
    wrong_registry = _write(
        tmp_path / "wrong_registry.json",
        {
            "asset_usd": source["asset_usd"],
            "parts": [{"part_id": "P1", "prim_path": "/Other/Mesh1"}],
        },
    )
    manifest = _write(tmp_path / "manifest.json", {})
    report = _report(
        tmp_path=tmp_path,
        view_id="side",
        registry=wrong_registry,
        manifest=manifest,
    )

    with pytest.raises(ValueError, match="another Part hierarchy"):
        _camera_contract(camera_reports={"side": report}, source_registry=source)


def test_multiview_camera_arguments_require_unique_multiple_views(
    tmp_path: Path,
) -> None:
    report = _write(tmp_path / "report.json", {})

    with pytest.raises(ValueError, match="at least two"):
        _camera_sources([f"side={report}"])
    with pytest.raises(ValueError, match="Duplicate"):
        _camera_sources([f"side={report}", f"side={report}"])
