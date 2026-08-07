from qwen_material_pipeline.materials.immutable_defaults import (
    rebind_verified_plan_provenance,
    resolve_plan_to_immutable_defaults,
)


def _catalog():
    return {
        "materials": [
            {
                "material_id": "mdl:v/Steel.mdl#Steel",
                "mdl_path": "v/Steel.mdl",
                "sub_identifier": "Steel",
                "colors": [],
                "display_name": "Steel",
            },
            {
                "material_id": "mdl:v/Steel.mdl#Steel_Army_Green",
                "mdl_path": "v/Steel.mdl",
                "sub_identifier": "Steel_Army_Green",
                "colors": ["green"],
                "display_name": "Steel Army Green",
            },
            {
                "material_id": "mdl:v/Steel.mdl#Steel_Arcadia_Green",
                "mdl_path": "v/Steel.mdl",
                "sub_identifier": "Steel_Arcadia_Green",
                "colors": ["green"],
                "display_name": "Steel Arcadia Green",
            },
        ]
    }


def test_resolves_parameterized_record_to_primary_exact_export():
    plan = {
        "schema_version": "1.0",
        "assignments": [
            {
                "part_id": "P0001",
                "material_id": "mdl:v/Steel.mdl#Steel",
                "semantic": "green painted steel",
                "parameters": {"paint_color": [0.1, 0.4, 0.1]},
                "face_subsets": [],
            }
        ],
    }
    output, audit = resolve_plan_to_immutable_defaults(
        plan=plan,
        catalog=_catalog(),
        primary_candidate_ids={"mdl:v/Steel.mdl#Steel_Army_Green"},
    )
    assignment = output["assignments"][0]
    assert assignment["material_id"] == "mdl:v/Steel.mdl#Steel_Army_Green"
    assert "parameters" not in assignment
    assert audit["changed_record_count"] == 1
    assert audit["parameter_write_count"] == 0
    assert audit["parameters_remaining"] is False


def test_keeps_existing_parameter_free_exact_export():
    plan = {
        "assignments": [
            {
                "part_id": "P0001",
                "material_id": "mdl:v/Steel.mdl#Steel_Army_Green",
                "semantic": "green steel",
            }
        ]
    }
    output, audit = resolve_plan_to_immutable_defaults(
        plan=plan,
        catalog=_catalog(),
    )
    assert output["assignments"][0]["material_id"].endswith("#Steel_Army_Green")
    assert audit["changed_record_count"] == 0


def test_rebinds_only_equivalent_complete_plan_provenance():
    source = {
        "provenance": {"asset_sha256": "asset", "registry_sha256": "old"},
        "assignments": [{"part_id": "P0001"}],
    }
    target = {
        "provenance": {"asset_sha256": "asset", "registry_sha256": "new"},
        "assignments": [{"part_id": "P0001"}],
    }
    rebind_verified_plan_provenance(plan=source, target_plan=target)
    assert source["provenance"]["registry_sha256"] == "new"
