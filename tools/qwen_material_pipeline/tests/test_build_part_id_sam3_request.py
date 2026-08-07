from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from qwen_material_pipeline.scripts.build_part_id_sam3_request import (
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
