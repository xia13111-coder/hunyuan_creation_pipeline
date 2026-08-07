from __future__ import annotations

from pathlib import Path

import pytest

from qwen_material_pipeline.materials.mdl_similarity import (
    extract_mdl_appearance_profile,
    extract_thumbnail_appearance_profile,
    mvinverse_similarity_terms,
)


def test_extracts_exact_export_color_and_pbr_defaults(tmp_path: Path) -> None:
    mdl = tmp_path / "library.mdl"
    mdl.write_text(
        """
export material Blue_Opaque(*)
 = Template(
    diffuse_tint: color(0.04f, 0.18f, 0.84f),
    roughness: 0.31f,
    metallic_constant: 0.0f);

export material Red_Opaque(*)
 = Template(
    diffuse_tint: color(0.64f, 0.03f, 0.03f),
    roughness: 0.12f,
    metallic_constant: 0.0f);
""",
        encoding="utf-8",
    )

    profile = extract_mdl_appearance_profile(mdl, "Blue_Opaque")

    assert profile is not None
    assert profile["color_parameter"] == "diffuse_tint"
    assert profile["base_color_linear"] == pytest.approx([0.04, 0.18, 0.84])
    assert profile["roughness"] == pytest.approx(0.31)
    assert profile["metallic"] == 0.0


def test_mvinverse_similarity_prefers_closer_numeric_profile() -> None:
    evidence = {
        "albedo": {"median": [0.2, 0.65, 0.9]},
        "roughness": {"median": 0.4},
        "metallic": {"median": 0.05},
    }
    close = {
        "base_color_srgb": [0.22, 0.64, 0.88],
        "roughness": 0.38,
        "metallic": 0.0,
    }
    far = {
        "base_color_srgb": [0.8, 0.1, 0.1],
        "roughness": 0.05,
        "metallic": 1.0,
    }

    close_score, close_terms = mvinverse_similarity_terms(close, evidence)
    far_score, _far_terms = mvinverse_similarity_terms(far, evidence)

    assert close_score > far_score
    assert close_terms == [
        "mvinverse_color",
        "mvinverse_roughness",
        "mvinverse_metallic",
    ]


def test_extracts_anodized_export_defaults(tmp_path: Path) -> None:
    mdl = tmp_path / "anodized.mdl"
    mdl.write_text(
        """
export material Aluminum_Anodized_Grass_Green(*)
 = Aluminum_Anodized(
    color_1: color(0.039546f, 0.381326f, 0.039546f),
    anodization_roughness: 0.77f);
""",
        encoding="utf-8",
    )

    profile = extract_mdl_appearance_profile(
        mdl,
        "Aluminum_Anodized_Grass_Green",
    )

    assert profile is not None
    assert profile["color_parameter"] == "color_1"
    assert profile["base_color_linear"] == pytest.approx(
        [0.039546, 0.381326, 0.039546]
    )
    assert profile["roughness_parameter"] == "anodization_roughness"
    assert profile["roughness"] == pytest.approx(0.77)


def test_extracts_catalog_thumbnail_central_material_color(tmp_path: Path) -> None:
    pillow = pytest.importorskip("PIL.Image")
    image = pillow.new("RGB", (128, 128), (12, 12, 12))
    for y in range(128):
        for x in range(128):
            if (x - 63.5) ** 2 + (y - 63.5) ** 2 <= 52**2:
                image.putpixel((x, y), (48, 142, 51))
    thumbnail = tmp_path / "green.png"
    image.save(thumbnail)

    profile = extract_thumbnail_appearance_profile(thumbnail)

    assert profile is not None
    assert profile["source"] == "nvidia_thumbnail_central_disk_median/v1"
    assert profile["base_color_srgb"] == pytest.approx(
        [48 / 255, 142 / 255, 51 / 255]
    )
    assert profile["chromatic_sample_preferred"] is True


def test_fixed_default_similarity_makes_perceptual_color_decisive() -> None:
    evidence = {
        "albedo": {"median": [0.22941176, 0.53529412, 0.19803921]},
        "roughness": {"median": 0.42745098},
    }
    olive = {
        "base_color_srgb": [0.28627464, 0.31372583, 0.11764688],
        "roughness": 0.32,
    }
    reference_green = {
        "base_color_srgb": [0.36078415, 0.48235258, 0.18823586],
        "roughness": 0.42,
    }

    normal_olive, _ = mvinverse_similarity_terms(olive, evidence)
    normal_green, _ = mvinverse_similarity_terms(reference_green, evidence)
    fixed_olive, _ = mvinverse_similarity_terms(
        olive,
        evidence,
        fixed_defaults_required=True,
    )
    fixed_green, _ = mvinverse_similarity_terms(
        reference_green,
        evidence,
        fixed_defaults_required=True,
    )

    assert normal_green > normal_olive
    assert fixed_green > fixed_olive + 250.0


def test_texture_driven_channels_do_not_use_unrelated_omnipbr_defaults(
    tmp_path: Path,
) -> None:
    mdl = tmp_path / "textured.mdl"
    mdl.write_text(
        """
export material Dark_Treatment(*)
 = OmniPBR(
    diffuse_texture: texture_2d("./dark_base.png"),
    diffuse_tint: color(1.0f, 1.0f, 1.0f),
    reflection_roughness_constant: 0.5f,
    metallic_constant: 0.0f,
    enable_ORM_texture: true,
    ORM_texture: texture_2d("./dark_orm.png"));
""",
        encoding="utf-8",
    )

    profile = extract_mdl_appearance_profile(mdl, "Dark_Treatment")

    assert profile is not None
    assert profile["base_color_srgb"] is None
    assert profile["roughness"] is None
    assert profile["metallic"] is None
    assert profile["texture_driven_channels"] == [
        "albedo",
        "roughness",
        "metallic",
    ]
    score, terms = mvinverse_similarity_terms(
        profile,
        {
            "albedo": {"median": [0.02, 0.03, 0.02]},
            "roughness": {"median": 0.65},
            "metallic": {"median": 0.8},
        },
    )
    assert score == 0.0
    assert terms == []
