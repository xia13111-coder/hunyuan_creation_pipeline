from __future__ import annotations

import pytest

from qwen_material_pipeline.materials.exact_group_rebind import (
    ExactGroupRebindError,
    rebind_exact_material_cohort,
)


SOURCE = "mdl:Paint/Green.mdl#Army"
TARGET = "mdl:Plastic/Green.mdl#Leaf"


def _catalog() -> dict:
    return {
        "materials": [
            {"material_id": SOURCE},
            {"material_id": TARGET},
        ]
    }


def _plan() -> dict:
    return {
        "schema_version": "1.0",
        "provenance": {"asset_sha256": "asset"},
        "assignments": [
            {
                "part_id": "P0001",
                "material_id": SOURCE,
                "parameters": {},
                "face_subsets": [
                    {
                        "subset_name": "green",
                        "material_id": SOURCE,
                        "parameters": {},
                    }
                ],
            },
            {"part_id": "P0002", "material_id": TARGET},
        ],
    }


def test_rebinds_whole_parts_and_subsets_without_parameters() -> None:
    output, audit = rebind_exact_material_cohort(
        plan=_plan(),
        catalog=_catalog(),
        source_material_ids={SOURCE},
        target_material_id=TARGET,
        group_id="G06",
    )

    assignment = output["assignments"][0]
    assert assignment["material_id"] == TARGET
    assert assignment["face_subsets"][0]["material_id"] == TARGET
    assert assignment["provenance"]["canonical_group_id"] == "G06"
    assert "provenance" not in assignment["face_subsets"][0]
    assert audit["part_change_count"] == 1
    assert audit["face_subset_change_count"] == 1
    assert audit["parameter_write_count"] == 0


def test_rejects_parameterized_source_plan() -> None:
    plan = _plan()
    plan["assignments"][0]["parameters"] = {"roughness": 0.2}
    with pytest.raises(ExactGroupRebindError, match="modifies MDL parameters"):
        rebind_exact_material_cohort(
            plan=plan,
            catalog=_catalog(),
            source_material_ids={SOURCE},
            target_material_id=TARGET,
            group_id="G06",
        )


def test_rejects_target_outside_catalog() -> None:
    with pytest.raises(ExactGroupRebindError, match="absent from the catalog"):
        rebind_exact_material_cohort(
            plan=_plan(),
            catalog=_catalog(),
            source_material_ids={SOURCE},
            target_material_id="mdl:Missing/Green.mdl#Green",
            group_id="G06",
        )
