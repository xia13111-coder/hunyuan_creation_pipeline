from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from qwen_material_pipeline.evidence.palette import (
    filter_palette_by_image_evidence,
)
from qwen_material_pipeline.evidence.palette_augmentation import (
    augment_palette_with_detected_accents,
)


def _green_palette() -> dict:
    return {
        "schema_version": "qwen-material-palette/v1",
        "source_view_id": "front",
        "groups": [
            {
                "group_id": "G01",
                "family_hint": "metal",
                "base_color": "green",
                "finish_hint": "painted",
                "visual_description": "green machine enclosure",
                "boxes": [[150, 150, 850, 850]],
                "confidence": 0.95,
            }
        ],
    }


def test_connected_accent_recovery_adds_tube_and_caps_but_not_grid(
    tmp_path: Path,
) -> None:
    path = tmp_path / "reference.png"
    image = Image.new("RGB", (256, 256), "black")
    draw = ImageDraw.Draw(image)
    draw.line((0, 30, 255, 30), fill=(110, 30, 25), width=1)
    draw.line((20, 0, 20, 255), fill=(110, 30, 25), width=1)
    draw.rectangle((45, 45, 215, 215), fill=(30, 130, 55))
    draw.line((80, 60, 95, 145), fill=(150, 82, 35), width=6)
    draw.rectangle((155, 65, 166, 80), fill=(20, 145, 215))
    draw.rectangle((172, 65, 183, 80), fill=(20, 145, 215))
    image.save(path)

    augmented, audit = augment_palette_with_detected_accents(
        _green_palette(),
        path,
    )

    added_colors = {
        group["base_color"]
        for group in augmented["groups"]
        if group["group_id"] in audit["added_group_ids"]
    }
    assert added_colors == {"orange", "blue"}

    filtered, _filter_audit = filter_palette_by_image_evidence(augmented, path)
    assert {group["base_color"] for group in filtered["groups"]} == {
        "green",
        "orange",
        "blue",
    }


def test_isolated_chromatic_noise_is_not_promoted(tmp_path: Path) -> None:
    path = tmp_path / "noise.png"
    image = Image.new("RGB", (256, 256), "black")
    draw = ImageDraw.Draw(image)
    draw.rectangle((45, 45, 215, 215), fill=(30, 130, 55))
    for x, y in ((70, 70), (100, 90), (130, 110), (160, 130)):
        draw.point((x, y), fill=(150, 82, 35))
    image.save(path)

    augmented, audit = augment_palette_with_detected_accents(
        _green_palette(),
        path,
    )

    assert audit["added_group_ids"] == []
    assert [group["base_color"] for group in augmented["groups"]] == ["green"]


def test_accent_recovery_is_not_limited_to_fixture_colors(tmp_path: Path) -> None:
    path = tmp_path / "chromatic-accents.png"
    image = Image.new("RGB", (256, 256), "black")
    draw = ImageDraw.Draw(image)
    draw.rectangle((45, 45, 215, 215), fill=(30, 130, 55))
    draw.rectangle((70, 70, 90, 90), fill=(210, 35, 45))
    draw.rectangle((110, 70, 130, 90), fill=(225, 190, 25))
    draw.rectangle((150, 70, 170, 90), fill=(205, 45, 180))
    image.save(path)

    augmented, audit = augment_palette_with_detected_accents(
        _green_palette(),
        path,
    )

    added = [
        group
        for group in augmented["groups"]
        if group["group_id"] in audit["added_group_ids"]
    ]
    assert {group["base_color"] for group in added} == {"red", "yellow", "pink"}
    assert {(group["family_hint"], group["finish_hint"]) for group in added} == {
        ("other", "other")
    }


def test_light_neutral_modules_are_recovered_without_promoting_grid_lines(
    tmp_path: Path,
) -> None:
    path = tmp_path / "light-neutral-modules.png"
    image = Image.new("RGB", (256, 256), "black")
    draw = ImageDraw.Draw(image)
    draw.line((0, 24, 255, 24), fill=(230, 230, 230), width=1)
    draw.line((24, 0, 24, 255), fill=(230, 230, 230), width=1)
    draw.rectangle((45, 45, 215, 215), fill=(30, 130, 55))
    draw.rectangle((72, 82, 92, 126), fill=(232, 235, 238))
    draw.rectangle((105, 82, 125, 126), fill=(210, 216, 222))
    image.save(path)

    augmented, audit = augment_palette_with_detected_accents(
        _green_palette(),
        path,
    )

    added = [
        group
        for group in augmented["groups"]
        if group["group_id"] in audit["added_group_ids"]
    ]
    assert [group["base_color"] for group in added] == ["white"]
    assert len(added[0]["boxes"]) == 2
    white_audit = next(
        item for item in audit["components"] if item["base_color"] == "white"
    )
    assert white_audit["rejected_component_counts"]["touches_image_border"] >= 1


def test_existing_white_group_prevents_duplicate_light_neutral_group(
    tmp_path: Path,
) -> None:
    path = tmp_path / "existing-white.png"
    image = Image.new("RGB", (256, 256), "black")
    draw = ImageDraw.Draw(image)
    draw.rectangle((45, 45, 215, 215), fill=(30, 130, 55))
    draw.rectangle((72, 82, 92, 126), fill=(232, 235, 238))
    image.save(path)
    palette = _green_palette()
    palette["groups"].append(
        {
            "group_id": "G02",
            "family_hint": "other",
            "base_color": "white",
            "finish_hint": "other",
            "visual_description": "white control module",
            "boxes": [[280, 320, 380, 520]],
            "confidence": 0.8,
        }
    )

    augmented, audit = augment_palette_with_detected_accents(palette, path)

    assert audit["added_group_ids"] == []
    assert [group["base_color"] for group in augmented["groups"]] == [
        "green",
        "white",
    ]


def test_dark_foreground_recovery_requires_mask_and_rejects_black_background(
    tmp_path: Path,
) -> None:
    path = tmp_path / "dark-on-black.png"
    mask_path = tmp_path / "foreground-mask.png"
    image = Image.new("RGB", (256, 256), "black")
    draw = ImageDraw.Draw(image)
    draw.rectangle((45, 45, 215, 215), fill=(30, 130, 55))
    draw.rectangle((72, 82, 130, 126), fill=(16, 18, 17))
    image.save(path)
    mask = Image.new("L", image.size, 0)
    ImageDraw.Draw(mask).rectangle((45, 45, 215, 215), fill=255)
    mask.save(mask_path)

    without_mask, without_mask_audit = augment_palette_with_detected_accents(
        _green_palette(),
        path,
    )
    assert "black" not in {
        group["base_color"] for group in without_mask["groups"]
    }
    black_without_mask = next(
        item
        for item in without_mask_audit["components"]
        if item["base_color"] == "black"
    )
    assert black_without_mask["decision"] == "foreground_mask_required"

    augmented, audit = augment_palette_with_detected_accents(
        _green_palette(),
        path,
        mask_path=mask_path,
    )
    added = [
        group
        for group in augmented["groups"]
        if group["group_id"] in audit["added_group_ids"]
    ]
    assert [group["base_color"] for group in added] == ["black"]
    assert added[0]["boxes"] == [[273, 312, 520, 504]]
    assert audit["mask"] == str(mask_path.resolve())
    assert audit["masked_dark_recovery_enabled"] is True

    filtered, filter_audit = filter_palette_by_image_evidence(
        augmented,
        path,
        mask_path=mask_path,
    )
    assert {group["base_color"] for group in filtered["groups"]} == {
        "green",
        "black",
    }
    black_group_id = added[0]["group_id"]
    black_filter = next(
        item
        for item in filter_audit["groups"]
        if item["group_id"] == black_group_id
    )
    assert black_filter["accepted"] is True
    assert black_filter["boxes"][0]["foreground_method"] == "mask"


def test_masked_rust_completion_recovers_uncovered_dark_brown_region(
    tmp_path: Path,
) -> None:
    path = tmp_path / "rust-completion.png"
    mask_path = tmp_path / "foreground-mask.png"
    image = Image.new("RGB", (256, 256), "black")
    draw = ImageDraw.Draw(image)
    draw.rectangle((35, 35, 220, 220), fill=(30, 120, 50))
    draw.line((55, 55, 70, 115), fill=(150, 82, 35), width=8)
    draw.rectangle((125, 105, 205, 155), fill=(58, 48, 47))
    image.save(path)
    mask = Image.new("L", image.size, 0)
    ImageDraw.Draw(mask).rectangle((35, 35, 220, 220), fill=255)
    mask.save(mask_path)

    palette = _green_palette()
    palette["groups"].append(
        {
            "group_id": "G02",
            "family_hint": "metal",
            "base_color": "orange",
            "finish_hint": "bare",
            "visual_description": "copper tube",
            "boxes": [[195, 195, 290, 475]],
            "confidence": 0.9,
        }
    )

    augmented, audit = augment_palette_with_detected_accents(
        palette,
        path,
        mask_path=mask_path,
    )

    added = [
        group
        for group in augmented["groups"]
        if group["group_id"] in audit["added_group_ids"]
    ]
    assert [group["base_color"] for group in added] == ["orange"]
    assert added[0]["family_hint"] == "other"
    assert any(box[0] >= 450 for box in added[0]["boxes"])
    assert audit["masked_low_saturation_rust_recovery_enabled"] is True
