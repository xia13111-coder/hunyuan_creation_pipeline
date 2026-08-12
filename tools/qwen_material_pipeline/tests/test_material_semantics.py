from __future__ import annotations

import pytest

from qwen_material_pipeline.materials.semantics import (
    MaterialSemanticsError,
    catalog_matches_part_semantics,
    infer_catalog_surface_semantics,
    normalize_part_material_semantics,
    semantics_from_legacy_surface,
    validate_selection_status,
)


def test_painted_metal_separates_substrate_from_visible_layer() -> None:
    semantics = semantics_from_legacy_surface("painted_metal", confidence=0.82)

    assert semantics["substrate"] == "metal"
    assert semantics["surface_treatment"] == "paint"
    assert semantics["optical_behavior"] == "opaque"
    assert semantics["physical_source"] == "vision_inference"


def test_green_glass_is_not_compatible_with_painted_metal() -> None:
    target = semantics_from_legacy_surface("painted_metal")
    green_glass = infer_catalog_surface_semantics(
        family="glass",
        tokens={"green", "glass"},
    )
    paint = infer_catalog_surface_semantics(
        family="paint",
        tokens={"paint", "matte"},
    )

    assert not catalog_matches_part_semantics(green_glass, target)
    assert catalog_matches_part_semantics(paint, target)


def test_ambiguous_physical_evidence_cannot_authorize_catalog_selection() -> None:
    target = normalize_part_material_semantics(
        {
            "substrate": "metal",
            "surface_treatment": "paint",
            "optical_behavior": "opaque",
            "finish": "unknown",
            "physical_source": "vision_inference",
            "evidence_status": "ambiguous",
            "confidence": 0.52,
        }
    )
    paint = infer_catalog_surface_semantics(
        family="paint",
        tokens={"paint", "matte"},
    )

    assert not catalog_matches_part_semantics(paint, target)


@pytest.mark.parametrize(
    "status",
    [
        "selected",
        "catalog_gap",
        "unknown",
        "unobserved",
        "safe_fallback_unverified",
    ],
)
def test_selection_status_vocabulary(status: str) -> None:
    assert validate_selection_status(status) == status


def test_invalid_semantics_fail_closed() -> None:
    with pytest.raises(MaterialSemanticsError, match="substrate"):
        normalize_part_material_semantics(
            {
                "substrate": "painted_metal",
                "surface_treatment": "paint",
                "optical_behavior": "opaque",
            }
        )


@pytest.mark.parametrize(
    "semantics, message",
    [
        (
            {
                "substrate": "metal",
                "surface_treatment": "emissive",
                "optical_behavior": "opaque",
                "evidence_status": "observed",
            },
            "emissive",
        ),
        (
            {
                "substrate": "metal",
                "surface_treatment": "bare",
                "optical_behavior": "transparent",
                "evidence_status": "observed",
            },
            "transparent",
        ),
        (
            {
                "substrate": "unknown",
                "surface_treatment": "paint",
                "optical_behavior": "opaque",
                "evidence_status": "observed",
            },
            "substrate",
        ),
    ],
)
def test_impossible_cross_field_semantics_fail_closed(
    semantics: dict,
    message: str,
) -> None:
    with pytest.raises(MaterialSemanticsError, match=message):
        normalize_part_material_semantics(semantics)
