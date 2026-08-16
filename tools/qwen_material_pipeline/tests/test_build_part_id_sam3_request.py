from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from qwen_material_pipeline.scripts.build_part_id_sam3_request import (
    _sha256,
    build_request,
)


def _write_image(path: Path, mode: str, *, tiny: bool = False) -> None:
    image = Image.new(mode, (32, 24), 0)
    if mode == "L":
        y_range = range(8, 10) if tiny else range(8, 16)
        x_range = range(10, 13) if tiny else range(10, 20)
        for y in y_range:
            for x in x_range:
                image.putpixel((x, y), 255)
    image.save(path)


def test_build_request_refines_every_visible_part_view(tmp_path: Path) -> None:
    observations = []
    for view_id in ("front", "top"):
        image = tmp_path / f"{view_id}.png"
        mask = tmp_path / f"{view_id}-mask.png"
        _write_image(image, "RGB")
        _write_image(mask, "L")
        observations.append(
            {
                "view_id": view_id,
                "image": str(image),
                "mask": str(mask),
                "selected_for_material_inference": view_id == "front",
            }
        )
    evidence = tmp_path / "evidence.json"
    evidence.write_text(
        json.dumps(
            {
                "parts": [
                    {
                        "part_id": "P0001",
                        "status": "observed",
                        "observations": observations,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    request = build_request(evidence)

    assert len(request["regions"]) == 2
    assert {
        (region["view_id"], region["group_id"])
        for region in request["regions"]
    } == {("front", "P0001"), ("top", "P0001")}
    assert all("cad_projection_seed" in region for region in request["regions"])


def test_build_request_accepts_six_pixel_chromatic_rescue(tmp_path: Path) -> None:
    image = tmp_path / "top.png"
    mask = tmp_path / "top-mask.png"
    _write_image(image, "RGB")
    _write_image(mask, "L", tiny=True)
    evidence = tmp_path / "evidence.json"
    evidence.write_text(
        json.dumps(
            {
                "parts": [
                    {
                        "part_id": "P0002",
                        "status": "observed",
                        "observations": [
                            {
                                "view_id": "top",
                                "image": str(image),
                                "mask": str(mask),
                                "selected_for_material_inference": True,
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    request = build_request(evidence)

    assert len(request["regions"]) == 1
    assert (
        request["regions"][0]["cad_projection_seed"][
            "projected_mask_pixels"
        ]
        == 6
    )


def test_build_request_binds_complete_mesh_shape_separately_from_visibility(
    tmp_path: Path,
) -> None:
    image = tmp_path / "front.png"
    modal = tmp_path / "front-modal.png"
    amodal = tmp_path / "front-amodal.png"
    _write_image(image, "RGB")
    _write_image(modal, "L")
    _write_image(amodal, "L")
    evidence = tmp_path / "evidence.json"
    evidence.write_text(
        json.dumps(
            {
                "parts": [
                    {
                        "part_id": "P0003",
                        "status": "observed",
                        "observations": [
                            {
                                "view_id": "front",
                                "image": str(image),
                                "mask": str(modal),
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "amodal.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "qwen-cad-amodal-part-templates/v1",
                "inputs": {
                    "part_id_evidence": {
                        "path": str(evidence.resolve()),
                        "sha256": _sha256(evidence.resolve()),
                    }
                },
                "records": [
                    {
                        "view_id": "front",
                        "part_id": "P0003",
                        "mesh_prim_path": "/Asset/P0003/Mesh",
                        "render_view_id": "front",
                        "projection_contract": {
                            "whole_asset_camera_unchanged": True,
                            "whole_asset_transform_unchanged": True,
                            "per_mesh_pose_change_allowed": False,
                            "other_mesh_occlusion_disabled_for_shape_only": True,
                        },
                        "aligned_amodal_mask": {
                            "path": str(amodal.resolve()),
                            "sha256": _sha256(amodal.resolve()),
                            "mask_size": [32, 24],
                            "mask_pixels": 80,
                            "bbox_pixels": [10, 8, 20, 16],
                        },
                    }
                ],
                "integrity": {"result_sha256": "a" * 64},
            }
        ),
        encoding="utf-8",
    )

    request = build_request(evidence, amodal_templates_path=manifest)

    region = request["regions"][0]
    assert region["cad_projection_seed"]["path"] == str(modal.resolve())
    assert region["cad_amodal_template"]["path"] == str(amodal.resolve())
    assert (
        region["cad_amodal_template"]["projection_contract"]
        ["per_mesh_pose_change_allowed"]
        is False
    )
    assert request["prompt_authority"] == (
        "whole_asset_visible_part_id_location_plus_isolated_mesh_shape"
    )
