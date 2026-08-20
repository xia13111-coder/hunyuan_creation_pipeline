from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from asset_pipeline.visual_materials.config import canonical_sha256
from asset_pipeline.visual_materials.corresponding_color import (
    corresponding_material_eligibility,
    rebind_part_id_audit_for_corresponding_color,
    validate_corresponding_color_result,
)
from asset_pipeline.visual_materials.references import sha256_file


def _write(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return path


def _sealed(value: dict) -> dict:
    output = copy.deepcopy(value)
    output["integrity"] = {"document_sha256": canonical_sha256(output)}
    return output


def _fixture(tmp_path: Path) -> dict[str, Path | dict]:
    inputs = tmp_path / "inputs"
    result = tmp_path / "material_identity_color"
    final = result / "final_selected"
    material_root = tmp_path / "materials"
    material_root.mkdir()
    source_plan_document = {
        "schema_version": "1.0",
        "assignment_unit": "part_id",
        "palette_fusion_used": False,
        "part_material_groups_used": False,
        "assignments": [
            {
                "part_id": "P1",
                "material_id": "mdl:Miscellaneous/Paint_Matte.mdl#Paint_Matte",
                "status": "auto",
                "provenance": {"assignment_unit": "part_id"},
            },
            {
                "part_id": "P2",
                "material_id": "mdl:Copper.mdl#Copper",
                "status": "auto",
                "provenance": {"assignment_unit": "part_id"},
            },
        ],
    }
    selected_plan_document = copy.deepcopy(source_plan_document)
    selected_plan_document["assignments"][0]["parameters"] = {
        "diffuse_tint": [0.2, 0.4, 0.1]
    }
    source_plan = _write(inputs / "source_plan.json", source_plan_document)
    qwen = _write(
        inputs / "qwen.json",
        _sealed(
            {
                "assignment_unit": "part_id",
            "selections": [
                {
                    "part_id": "P1",
                    "material_id": "mdl:Miscellaneous/Paint_Matte.mdl#Paint_Matte",
                    "match_type": "CORRESPONDING_MATERIAL",
                },
                {
                    "part_id": "P2",
                    "material_id": "mdl:Copper.mdl#Copper",
                    "match_type": "EXACT_LIBRARY_MATCH",
                },
            ],
                "component_identity_consensus": {"components": []},
            }
        ),
    )
    evidence = _write(inputs / "evidence.json", {})
    spatial = _write(inputs / "spatial.json", {})
    catalog = _write(inputs / "catalog.json", {})
    registry = _write(inputs / "registry.json", {})
    view_specs = _write(inputs / "view_specs.json", {})
    reference = _write(
        inputs / "reference.json",
        {"source_views": [{"id": "front"}, {"id": "iso"}]},
    )
    asset = inputs / "asset.usda"
    asset.write_text("#usda 1.0\n", encoding="utf-8")
    isaac = inputs / "python.sh"
    isaac.write_text("#!/bin/sh\n", encoding="utf-8")

    selected_plan = _write(
        final / "part_id_material_plan.color.selected.json",
        selected_plan_document,
    )
    selection_audit_document = _sealed(
        {
            "schema_version": (
                "qwen-corresponding-material-color-render-selection-audit/v2"
            ),
            "status": "PASS",
            "source_plan_sha256": canonical_sha256(source_plan_document),
            "output_plan_sha256": canonical_sha256(selected_plan_document),
            "summary": {
                "colour_scope_count": 1,
                "parameterized_part_count": 1,
                "material_identity_change_count": 0,
                "local_quality_gate_status": "PASS",
            },
            "selections": [
                {
                    "scope_id": "PART:P1",
                    "member_part_ids": ["P1"],
                    "selected_candidate_id": "iteration_01",
                    "local_quality_gate": {"status": "PASS"},
                }
            ],
        }
    )
    selection_audit = _write(
        final / "corresponding_material_color_selection_audit.json",
        selection_audit_document,
    )
    look = final / "material_look.usda"
    look.write_text("#usda 1.0\n", encoding="utf-8")
    apply_report = _write(final / "apply_report.json", {"applied_count": 2})
    output_registry = _write(final / "part_registry.json", {})
    rendered_registry = _write(final / "renders" / "part_registry.rendered.json", {})
    quality_document = {
        "aggregate": {
            "status": "PASS",
            "reference_view_count": 2,
            "comparable_view_count": 2,
        },
        "views": [
            {"reference_view_id": "front", "status": "PASS"},
            {"reference_view_id": "iso", "status": "PASS"},
        ],
    }
    quality = _write(final / "reference_render_comparison.json", quality_document)
    input_paths = {
        "source_plan": source_plan,
        "qwen_choices": qwen,
        "part_id_evidence": evidence,
        "spatial_mapping_report": spatial,
        "asset_usd": asset,
        "catalog": catalog,
        "registry": registry,
        "view_specs": view_specs,
        "reference_manifest": reference,
        "isaac_python": isaac,
    }
    output_paths = {
        "selected_plan": selected_plan,
        "selection_audit": selection_audit,
        "asset": look,
        "apply_report": apply_report,
        "registry": output_registry,
        "rendered_registry": rendered_registry,
        "quality_report": quality,
    }
    manifest = _write(
        result / "workflow_manifest.json",
        {
            "schema_version": "qwen-corresponding-material-color-workflow/v3",
            "workflow_state": "COMPLETE",
            "quality_status": "PASS",
            "policy": {
                "material_identity_mutation_allowed": False,
                "same_component_shares_material_and_colour": True,
                "actual_cad_render_selection": True,
                "local_part_scope_quality_gate": True,
                "optimization_mode": "adaptive_per_scope",
            },
            "inputs": {
                **{
                    label: {"path": str(path.resolve()), "sha256": sha256_file(path)}
                    for label, path in input_paths.items()
                },
                "material_root": {"path": str(material_root.resolve())},
            },
            "outputs": {
                label: {"path": str(path.resolve()), "sha256": sha256_file(path)}
                for label, path in output_paths.items()
            },
            "candidates": [{"candidate_id": "iteration_00"}],
        },
    )
    return {
        "manifest": manifest,
        "material_root": material_root,
        "source_plan_document": source_plan_document,
        "selected_plan_document": selected_plan_document,
        "selection_audit_document": selection_audit_document,
        **input_paths,
        "selected_plan": selected_plan,
        "selection_audit": selection_audit,
        "look_usd": look,
        "apply_report": apply_report,
        "registry_output": output_registry,
        "rendered_registry": rendered_registry,
        "quality_report": quality,
    }


def test_mainline_validator_accepts_same_identity_corresponding_colour(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    result = validate_corresponding_color_result(
        manifest_path=paths["manifest"],
        source_plan_path=paths["source_plan"],
        qwen_choices_path=paths["qwen_choices"],
        part_id_evidence_path=paths["part_id_evidence"],
        spatial_mapping_report_path=paths["spatial_mapping_report"],
        asset_usd_path=paths["asset_usd"],
        catalog_path=paths["catalog"],
        registry_path=paths["registry"],
        material_root_path=paths["material_root"],
        view_specs_path=paths["view_specs"],
        reference_manifest_path=paths["reference_manifest"],
        isaac_python_path=paths["isaac_python"],
        selected_plan_path=paths["selected_plan"],
        selection_audit_path=paths["selection_audit"],
        look_usd_path=paths["look_usd"],
        apply_report_path=paths["apply_report"],
        registry_output_path=paths["registry_output"],
        rendered_registry_path=paths["rendered_registry"],
        quality_report_path=paths["quality_report"],
    )
    assert result.applied_count == 2
    assert result.selected_plan["assignments"][1].get("parameters") is None


def test_mainline_validator_rejects_parameter_on_exact_preset(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    selected = json.loads(paths["selected_plan"].read_text(encoding="utf-8"))
    selected["assignments"][1]["parameters"] = {"diffuse_tint": [1, 0, 0]}
    _write(paths["selected_plan"], selected)
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    manifest["outputs"]["selected_plan"]["sha256"] = sha256_file(paths["selected_plan"])
    _write(paths["manifest"], manifest)
    with pytest.raises(RuntimeError, match="non-corresponding"):
        validate_corresponding_color_result(
            manifest_path=paths["manifest"],
            source_plan_path=paths["source_plan"],
            qwen_choices_path=paths["qwen_choices"],
            part_id_evidence_path=paths["part_id_evidence"],
            spatial_mapping_report_path=paths["spatial_mapping_report"],
            asset_usd_path=paths["asset_usd"],
            catalog_path=paths["catalog"],
            registry_path=paths["registry"],
            material_root_path=paths["material_root"],
            view_specs_path=paths["view_specs"],
            reference_manifest_path=paths["reference_manifest"],
            isaac_python_path=paths["isaac_python"],
            selected_plan_path=paths["selected_plan"],
            selection_audit_path=paths["selection_audit"],
            look_usd_path=paths["look_usd"],
            apply_report_path=paths["apply_report"],
            registry_output_path=paths["registry_output"],
            rendered_registry_path=paths["rendered_registry"],
            quality_report_path=paths["quality_report"],
        )


def test_part_id_audit_rebinds_only_colour_and_final_plan_hash(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    source_plan = paths["source_plan_document"]
    final_plan = paths["selected_plan_document"]
    source_audit = _sealed(
        {
            "schema_version": "qwen-part-id-material-plan-audit/v1",
            "output_plan_sha256": canonical_sha256(source_plan),
            "parts": [
                {
                    "part_id": "P1",
                    "material_id": "mdl:Miscellaneous/Paint_Matte.mdl#Paint_Matte",
                },
                {"part_id": "P2", "material_id": "mdl:Copper.mdl#Copper"},
            ],
            "summary": {"part_count": 2, "color_parameterized_count": 0},
        }
    )
    rebound = rebind_part_id_audit_for_corresponding_color(
        source_audit=source_audit,
        source_plan=source_plan,
        final_plan=final_plan,
        selection_audit=paths["selection_audit_document"],
    )
    assert rebound["output_plan_sha256"] == canonical_sha256(final_plan)
    assert rebound["summary"]["color_parameterized_count"] == 1
    assert "corresponding_material_color_calibration" in rebound["parts"][0]
    assert "corresponding_material_color_calibration" not in rebound["parts"][1]
