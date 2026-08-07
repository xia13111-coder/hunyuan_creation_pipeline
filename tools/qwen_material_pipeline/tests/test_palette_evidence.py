from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from qwen_material_pipeline.core.staged_analysis import StagedAnalysisError
from qwen_material_pipeline.evidence.palette import (
    filter_palette_by_image_evidence,
)


def _palette() -> dict:
    return {
        "schema_version": "qwen-material-palette/v1",
        "source_view_id": "ref_test",
        "groups": [
            {
                "group_id": "G01",
                "family_hint": "metal",
                "base_color": "green",
                "finish_hint": "painted",
                "visual_description": "green painted body",
                "boxes": [[200, 200, 500, 800]],
                "confidence": 0.98,
            },
            {
                "group_id": "G02",
                "family_hint": "metal",
                "base_color": "gray",
                "finish_hint": "painted",
                "visual_description": "invented gray background region",
                "boxes": [[700, 100, 950, 300]],
                "confidence": 0.95,
            },
            {
                "group_id": "G03",
                "family_hint": "metal",
                "base_color": "black",
                "finish_hint": "painted",
                "visual_description": "wrong color over green body",
                "boxes": [[200, 200, 500, 800]],
                "confidence": 0.95,
            },
        ],
    }


def test_palette_evidence_rejects_background_and_wrong_color(tmp_path: Path) -> None:
    image_path = tmp_path / "reference.png"
    image = Image.new("RGB", (200, 200), (0, 0, 0))
    ImageDraw.Draw(image).rectangle((40, 40, 100, 160), fill=(25, 130, 50))
    image.save(image_path)
    filtered, audit = filter_palette_by_image_evidence(_palette(), image_path)
    assert [group["group_id"] for group in filtered["groups"]] == ["G01"]
    assert audit["accepted_group_ids"] == ["G01"]
    assert audit["rejected_group_ids"] == ["G02", "G03"]
    assert filtered["groups"][0]["confidence"] < 0.85


def _single_group_palette(color: str, box: list[int], *, group_id: str = "G01") -> dict:
    return {
        "schema_version": "qwen-material-palette/v1",
        "source_view_id": "ref_test",
        "groups": [
            {
                "group_id": group_id,
                "family_hint": "metal",
                "base_color": color,
                "finish_hint": "painted",
                "visual_description": f"{color} painted part",
                "boxes": [box],
                "confidence": 0.98,
            }
        ],
    }


def test_black_material_survives_black_background_filter(tmp_path: Path) -> None:
    image_path = tmp_path / "black_part.png"
    image = Image.new("RGB", (200, 200), (0, 0, 0))
    # This subtly lit black part is closer than the normal 28-RGB-distance
    # foreground cutoff, which used to remove every one of its pixels.
    ImageDraw.Draw(image).rectangle((40, 40, 120, 160), fill=(10, 10, 10))
    image.save(image_path)

    filtered, audit = filter_palette_by_image_evidence(
        _single_group_palette("black", [180, 180, 650, 850]), image_path
    )

    assert [group["group_id"] for group in filtered["groups"]] == ["G01"]
    box_audit = audit["groups"][0]["boxes"][0]
    assert box_audit["foreground_method"] == "dark_structure"
    assert box_audit["effective_background_distance"] < 28.0
    assert box_audit["black_structure_supported"] is True
    assert box_audit["color_match"] == 1.0
    assert box_audit["matching_pixel_count"] > 0
    assert box_audit["representative_srgb"] == [10, 10, 10]


def test_pure_black_background_does_not_count_as_black_material(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "black_background.png"
    image = Image.new("RGB", (200, 200), (0, 0, 0))
    ImageDraw.Draw(image).rectangle((20, 40, 80, 160), fill=(25, 130, 50))
    image.save(image_path)
    palette = {
        "schema_version": "qwen-material-palette/v1",
        "source_view_id": "ref_test",
        "groups": [
            _single_group_palette("green", [80, 180, 430, 850])["groups"][0],
            _single_group_palette("black", [600, 100, 950, 900], group_id="G02")[
                "groups"
            ][0],
        ],
    }

    filtered, audit = filter_palette_by_image_evidence(palette, image_path)

    assert [group["group_id"] for group in filtered["groups"]] == ["G01"]
    black_audit = audit["groups"][1]["boxes"][0]
    assert black_audit["foreground_pixels"] == 0
    assert black_audit["black_structure_supported"] is False
    assert "missing_black_structure" in black_audit["rejection_reasons"]


def test_white_evidence_accepts_neutral_shadow_pixels(tmp_path: Path) -> None:
    image_path = tmp_path / "shaded_white.png"
    image = Image.new("RGB", (200, 200), (0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rectangle((40, 40, 120, 80), fill=(118, 121, 120))
    draw.rectangle((40, 81, 120, 120), fill=(165, 168, 167))
    draw.rectangle((40, 121, 120, 160), fill=(205, 207, 206))
    image.save(image_path)

    filtered, audit = filter_palette_by_image_evidence(
        _single_group_palette("white", [180, 180, 650, 850]), image_path
    )

    assert [group["group_id"] for group in filtered["groups"]] == ["G01"]
    box_audit = audit["groups"][0]["boxes"][0]
    assert box_audit["color_match"] == 1.0
    assert set(box_audit["foreground_color_counts"]) == {"gray", "silver"}
    assert box_audit["accepted_color_labels"] == ["gray", "silver", "white"]
    assert box_audit["matching_pixel_count"] == box_audit["foreground_pixels"]
    assert box_audit["representative_srgb"] == [165, 168, 167]


def test_explicit_mask_can_separate_exact_black_part_from_black_background(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "indistinguishable_black.png"
    mask_path = tmp_path / "foreground_mask.png"
    Image.new("RGB", (200, 200), (0, 0, 0)).save(image_path)
    mask = Image.new("L", (200, 200), 0)
    ImageDraw.Draw(mask).rectangle((40, 40, 120, 160), fill=255)
    mask.save(mask_path)

    filtered, audit = filter_palette_by_image_evidence(
        _single_group_palette("black", [180, 180, 650, 850]),
        image_path,
        mask_path=mask_path,
    )

    assert [group["group_id"] for group in filtered["groups"]] == ["G01"]
    assert audit["mask"] == str(mask_path.resolve())
    assert audit["mask_source"] == "explicit"
    assert audit["mask_channel"] == "luminance"
    box_audit = audit["groups"][0]["boxes"][0]
    assert box_audit["foreground_method"] == "mask"
    assert box_audit["black_structure_supported"] is True


def test_mixed_box_accepts_connected_achromatic_surface(tmp_path: Path) -> None:
    image_path = tmp_path / "mixed_white_and_green.png"
    image = Image.new("RGB", (200, 200), (0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rectangle((30, 30, 170, 170), fill=(25, 130, 50))
    draw.rectangle((50, 55, 120, 145), fill=(205, 208, 210))
    image.save(image_path)

    filtered, audit = filter_palette_by_image_evidence(
        _single_group_palette("white", [120, 120, 880, 880]), image_path
    )

    assert [group["group_id"] for group in filtered["groups"]] == ["G01"]
    box_audit = audit["groups"][0]["boxes"][0]
    assert box_audit["color_match"] < audit["minimum_color_match"]
    assert box_audit["mixed_achromatic_supported"] is True
    assert box_audit["largest_matching_structure_coverage"] >= 0.02


def test_blue_evidence_accepts_cyan_highlight_pixels(tmp_path: Path) -> None:
    image_path = tmp_path / "cyan_lit_blue_part.png"
    image = Image.new("RGB", (200, 200), (0, 0, 0))
    ImageDraw.Draw(image).rectangle((40, 40, 120, 160), fill=(20, 175, 205))
    image.save(image_path)

    filtered, audit = filter_palette_by_image_evidence(
        _single_group_palette("blue", [180, 180, 650, 850]), image_path
    )

    assert [group["group_id"] for group in filtered["groups"]] == ["G01"]
    box_audit = audit["groups"][0]["boxes"][0]
    assert box_audit["foreground_color_counts"] == {"cyan": 9801}
    assert box_audit["accepted_color_labels"] == ["blue", "cyan"]
    assert box_audit["matching_pixel_count"] == box_audit["foreground_pixels"]
    assert box_audit["color_match"] == 1.0


def test_orange_evidence_accepts_brown_shadow_pixels(tmp_path: Path) -> None:
    image_path = tmp_path / "shadowed_orange_part.png"
    image = Image.new("RGB", (200, 200), (0, 0, 0))
    ImageDraw.Draw(image).rectangle((40, 40, 120, 160), fill=(125, 70, 25))
    image.save(image_path)

    filtered, audit = filter_palette_by_image_evidence(
        _single_group_palette("orange", [180, 180, 650, 850]), image_path
    )

    assert [group["group_id"] for group in filtered["groups"]] == ["G01"]
    box_audit = audit["groups"][0]["boxes"][0]
    assert box_audit["foreground_color_counts"] == {"brown": 9801}
    assert box_audit["accepted_color_labels"] == ["brown", "orange"]
    assert box_audit["matching_pixel_count"] == box_audit["foreground_pixels"]
    assert box_audit["color_match"] == 1.0


def test_connected_small_chromatic_accent_survives_broad_citation(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "copper_tube_on_green_body.png"
    image = Image.new("RGB", (200, 200), (0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rectangle((30, 30, 170, 170), fill=(25, 130, 50))
    draw.rectangle((55, 45, 64, 145), fill=(135, 78, 38))
    image.save(image_path)

    filtered, audit = filter_palette_by_image_evidence(
        _single_group_palette("orange", [120, 120, 880, 880]), image_path
    )

    assert [group["group_id"] for group in filtered["groups"]] == ["G01"]
    assert filtered["groups"][0]["confidence"] >= 0.60
    box_audit = audit["groups"][0]["boxes"][0]
    assert box_audit["color_match"] < audit["minimum_color_match"]
    assert box_audit["mixed_chromatic_supported"] is True
    assert box_audit["largest_matching_structure_coverage"] >= 0.02


def test_orange_evidence_does_not_accept_red_pixels(tmp_path: Path) -> None:
    image_path = tmp_path / "red_part.png"
    image = Image.new("RGB", (200, 200), (0, 0, 0))
    ImageDraw.Draw(image).rectangle((40, 40, 120, 160), fill=(190, 25, 25))
    image.save(image_path)

    try:
        filter_palette_by_image_evidence(
            _single_group_palette("orange", [180, 180, 650, 850]), image_path
        )
    except StagedAnalysisError as error:
        assert "all Qwen palette groups failed" in str(error)
    else:
        raise AssertionError("red pixels must not verify an orange palette group")
