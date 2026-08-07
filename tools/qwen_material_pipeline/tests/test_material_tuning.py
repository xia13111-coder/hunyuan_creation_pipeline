from __future__ import annotations

import pytest

from qwen_material_pipeline.materials.tuning import (
    PLASTIC_ABS,
    tune_selected_material_color_from_mvinverse,
    tune_selected_material_from_mvinverse,
)
from qwen_material_pipeline.usd.material_common import normalize_material_parameters


BLUE_POLYCARBONATE = (
    "mdl:vMaterials_2/Plastic/Polycarbonate_Opaque.mdl#Polycarbonate_Blue"
)


def _evidence() -> dict:
    return {
        "group_id": "G05",
        "surface_class": "dielectric",
        "contributing_view_ids": ["front", "iso"],
        "metallic": {"median": 0.08},
        "suggestion": {
            "decision": "auto",
            "auto_parameter_eligible": True,
            "base_color_srgb": [0.2, 0.55, 0.9],
            "metallic": 0.0,
            "roughness": 0.41,
        },
    }


def test_polycarbonate_tuning_preserves_selected_mdl_and_authors_minimal_delta() -> None:
    parameters, audit = tune_selected_material_from_mvinverse(
        _evidence(),
        group_id="G05",
        material_id=BLUE_POLYCARBONATE,
    )

    assert parameters["diffuse_tint"] == pytest.approx(
        [0.03310477, 0.26327341, 0.78741229]
    )
    assert parameters["roughness"] == pytest.approx(0.41)
    assert set(parameters) == {"diffuse_tint", "roughness"}
    assert audit["material_id"] == BLUE_POLYCARBONATE
    assert audit["tuning_profile_id"] == "nvidia_polycarbonate_opaque"
    assert (
        audit["color_parameter_semantics"] == "absolute_linear_color"
    )
    assert normalize_material_parameters(BLUE_POLYCARBONATE, parameters)


def test_abs_tuning_can_author_color_roughness_and_dielectric_metallic() -> None:
    parameters, _audit = tune_selected_material_from_mvinverse(
        _evidence(),
        group_id="G05",
        material_id=PLASTIC_ABS,
    )

    assert set(parameters) == {
        "diffuse_tint",
        "reflection_roughness_constant",
        "metallic_constant",
    }
    assert parameters["metallic_constant"] == 0.0
    assert normalize_material_parameters(PLASTIC_ABS, parameters)


def test_base_paint_tuning_preserves_preset_and_authors_bounded_delta() -> None:
    material_id = "mdl:Miscellaneous/Paint_Matte.mdl#Paint_Matte"

    parameters, audit = tune_selected_material_from_mvinverse(
        _evidence(),
        group_id="G05",
        material_id=material_id,
    )

    assert set(parameters) == {"diffuse_tint", "reflection_roughness_constant"}
    assert audit["tuning_profile_id"] == "nvidia_base_paint_omnipbr"
    assert normalize_material_parameters(material_id, parameters)


def test_materials_root_base_namespace_uses_same_tuning_contract() -> None:
    material_id = "mdl:Base/Miscellaneous/Paint_Matte.mdl#Paint_Matte"

    parameters, audit = tune_selected_material_from_mvinverse(
        _evidence(),
        group_id="G05",
        material_id=material_id,
    )

    assert set(parameters) == {"diffuse_tint", "reflection_roughness_constant"}
    assert audit["tuning_profile_id"] == "nvidia_base_paint_omnipbr"
    assert normalize_material_parameters(material_id, parameters)


def test_base_metal_tuning_can_author_mvinverse_metallicity() -> None:
    material_id = "mdl:Metals/Steel_Stainless.mdl#Steel_Stainless"
    evidence = _evidence()
    evidence["surface_class"] = "metal"
    evidence["metallic"]["median"] = 0.91
    evidence["suggestion"]["metallic"] = 0.94

    parameters, audit = tune_selected_material_from_mvinverse(
        evidence,
        group_id="G05",
        material_id=material_id,
    )

    assert parameters["metallic_constant"] == pytest.approx(0.94)
    assert audit["tuning_profile_id"] == "nvidia_base_metal_omnipbr"
    assert normalize_material_parameters(material_id, parameters)


def test_tuning_fails_closed_without_multiview_auto_evidence() -> None:
    evidence = _evidence()
    evidence["contributing_view_ids"] = ["front"]

    with pytest.raises(ValueError, match="distinct multi-view"):
        tune_selected_material_from_mvinverse(
            evidence,
            group_id="G05",
            material_id=BLUE_POLYCARBONATE,
        )


def test_corroborated_single_view_tuning_authors_color_only() -> None:
    material_id = "mdl:Miscellaneous/Paint_Matte.mdl#Paint_Matte"
    evidence = {
        "group_id": "G07",
        "surface_class": "dielectric",
        "contributing_view_ids": ["top"],
        "albedo": {"median": [0.55, 0.35, 0.19]},
        "metallic": {"median": 0.16},
        "roughness": {"median": 0.38},
        "suggestion": {
            "decision": "preserve",
            "auto_parameter_eligible": False,
            "reason_codes": ["insufficient_distinct_views"],
        },
    }

    parameters, audit = tune_selected_material_color_from_mvinverse(
        evidence,
        group_id="G07",
        material_id=material_id,
    )

    assert set(parameters) == {"diffuse_tint"}
    assert audit["parameterization_mode"] == (
        "multiview_palette_corroborated_color_only"
    )
    assert audit["authored_metallic"] is None
    assert normalize_material_parameters(material_id, parameters)


def test_neutral_diffuse_tint_retains_absolute_value() -> None:
    evidence = _evidence()
    evidence["suggestion"]["base_color_srgb"] = [0.2, 0.2, 0.2]

    parameters, audit = tune_selected_material_from_mvinverse(
        evidence,
        group_id="G05",
        material_id=PLASTIC_ABS,
    )

    assert parameters["diffuse_tint"] == pytest.approx(
        [0.03310477, 0.03310477, 0.03310477]
    )
    assert audit["color_parameter_semantics"] == "absolute_linear_color"


def test_untextured_chromatic_diffuse_tint_retains_absolute_value() -> None:
    evidence = _evidence()

    parameters, audit = tune_selected_material_from_mvinverse(
        evidence,
        group_id="G05",
        material_id=PLASTIC_ABS,
    )

    assert parameters["diffuse_tint"] == pytest.approx(
        [0.03310477, 0.26327341, 0.78741229]
    )
    assert audit["color_parameter_semantics"] == "absolute_linear_color"


def test_color_only_tuning_rejects_non_sampling_failures() -> None:
    evidence = {
        "group_id": "G01",
        "surface_class": "dielectric",
        "contributing_view_ids": ["front"],
        "albedo": {"median": [0.05, 0.05, 0.05]},
        "metallic": {"median": 0.7},
        "roughness": {"median": 0.3},
        "suggestion": {
            "decision": "preserve",
            "auto_parameter_eligible": False,
            "reason_codes": ["dielectric_metallicity_conflict"],
        },
    }

    with pytest.raises(ValueError, match="not eligible for corroborated color"):
        tune_selected_material_color_from_mvinverse(
            evidence,
            group_id="G01",
            material_id="mdl:Miscellaneous/Paint_Matte.mdl#Paint_Matte",
        )
