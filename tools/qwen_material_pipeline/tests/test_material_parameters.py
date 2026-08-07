from __future__ import annotations

import pytest

from qwen_material_pipeline.materials.parameters import (
    mvinverse_paint_parameters,
    srgb_to_linear,
)


def _evidence(*, decision: str = "auto", eligible: bool = True):
    return {
        "schema_version": "qwen-mvinverse-pbr-evidence/v1",
        "groups": [
            {
                "group_id": "G01",
                "suggestion": {
                    "decision": decision,
                    "auto_parameter_eligible": eligible,
                    "base_color_srgb": [0.22941176, 0.53529412, 0.19607843],
                    "metallic": 0.0,
                    "roughness": 0.42745098,
                    "reason_codes": ["multi_view_evidence_sufficient"],
                    "warning_codes": [],
                },
            }
        ],
    }


def test_srgb_to_linear_matches_renderer_values() -> None:
    assert srgb_to_linear([0.22941176, 0.53529412, 0.19607843]) == pytest.approx(
        [0.043019783112744646, 0.24817520774901805, 0.03189603265453956]
    )


def test_mvinverse_parameters_include_roughness_and_metallic_audit() -> None:
    parameters, audit = mvinverse_paint_parameters(_evidence(), "G01")

    assert parameters["paint_color"] == pytest.approx(
        [0.043019783112744646, 0.24817520774901805, 0.03189603265453956]
    )
    assert parameters["paint_roughness"] == pytest.approx(0.42745098)
    assert audit["metallic"] == 0.0
    assert audit["group_id"] == "G01"


@pytest.mark.parametrize(
    ("decision", "eligible"),
    [("preserve", False), ("auto", False), ("preserve", True)],
)
def test_non_auto_evidence_fails_closed(decision: str, eligible: bool) -> None:
    with pytest.raises(ValueError, match="not eligible"):
        mvinverse_paint_parameters(
            _evidence(decision=decision, eligible=eligible), "G01"
        )


def test_missing_group_fails_closed() -> None:
    with pytest.raises(ValueError, match="exactly once"):
        mvinverse_paint_parameters(_evidence(), "G03")
