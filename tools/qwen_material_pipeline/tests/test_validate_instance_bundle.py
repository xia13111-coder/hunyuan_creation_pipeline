from __future__ import annotations

import ast
import importlib
import json
import shutil
from pathlib import Path

import pytest

from qwen_material_pipeline.usd.apply_instances import (
    SCHEMA_VERSION as APPLY_SCHEMA_VERSION,
)
from qwen_material_pipeline.usd.material_common import sha256_file
from qwen_material_pipeline.usd.stage_state import canonical_sha256
from qwen_material_pipeline.usd.validate_instances import (
    _is_at_or_below,
    validate_instance_bundle,
)


@pytest.mark.parametrize(
    "module_name",
    (
        "qwen_material_pipeline.usd.apply_instances",
        "qwen_material_pipeline.usd.validate_instances",
    ),
)
def test_instance_modules_do_not_import_private_cross_module_symbols(
    module_name: str,
) -> None:
    module = importlib.import_module(module_name)
    module_path = Path(module.__file__).resolve()
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    private_imports = [
        (node.module, alias.name)
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
        if alias.name.startswith("_")
    ]

    assert private_imports == []


def test_is_at_or_below_requires_a_usd_path_boundary() -> None:
    root = "/Asset/QwenInstanceLooks"
    assert _is_at_or_below(root, root)
    assert _is_at_or_below(root + "/Steel", root)
    assert not _is_at_or_below(root + "Backup/Steel", root)
    assert not _is_at_or_below(None, root)


def test_instance_bundle_validator_covers_instance_occurrences(tmp_path: Path) -> None:
    pytest.importorskip("pxr")
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt

    from qwen_material_pipeline.usd.instances import (
        create_editable_instance_layer,
    )
    from qwen_material_pipeline.usd.registry import build_part_registry

    source = tmp_path / "source.usda"
    stage = Usd.Stage.CreateNew(str(source))
    root = UsdGeom.Xform.Define(stage, "/Asset").GetPrim()
    stage.SetDefaultPrim(root)
    stage.CreateClassPrim("/Asset/_Part")
    mesh = UsdGeom.Mesh.Define(stage, "/Asset/_Part/Mesh")
    mesh.CreatePointsAttr(
        Vt.Vec3fArray(
            [
                Gf.Vec3f(0.0, 0.0, 0.0),
                Gf.Vec3f(1.0, 0.0, 0.0),
                Gf.Vec3f(0.0, 1.0, 0.0),
            ]
        )
    )
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3]))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray([0, 1, 2]))
    for index, name in enumerate(("A", "B")):
        occurrence = UsdGeom.Xform.Define(stage, f"/Asset/{name}")
        occurrence.GetPrim().GetReferences().AddInternalReference("/Asset/_Part")
        occurrence.GetPrim().SetInstanceable(True)
        occurrence.AddTranslateOp().Set(Gf.Vec3d(index * 2.0, 0.0, 0.0))
    stage.GetRootLayer().Save()
    stage = None

    bridge = tmp_path / "editable.usda"
    bridge_report = create_editable_instance_layer(source_usd=source, output_usd=bridge)
    registry_document = build_part_registry(bridge)
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps(registry_document), encoding="utf-8")

    original_mdl = tmp_path / "Test.mdl"
    original_mdl.write_text(
        "mdl 1.7;\nexport material Test() = material();\n", encoding="utf-8"
    )
    bundle = tmp_path / "bundle"
    bundled_source = bundle / "SubUSDs" / "source.usda"
    bundled_mdl = bundle / "SubUSDs" / "materials" / "Test.mdl"
    bundled_source.parent.mkdir(parents=True)
    bundled_mdl.parent.mkdir(parents=True)
    shutil.copy2(source, bundled_source)
    shutil.copy2(original_mdl, bundled_mdl)

    final = bundle / "look.usda"
    stage = Usd.Stage.CreateNew(str(final))
    root = stage.DefinePrim("/Asset", "Xform")
    root.GetReferences().AddReference("./SubUSDs/source.usda", "/Asset")
    stage.SetDefaultPrim(root)
    for name in ("A", "B"):
        stage.OverridePrim(f"/Asset/{name}").SetInstanceable(False)
    UsdGeom.Scope.Define(stage, "/Asset/QwenInstanceLooks")
    material = UsdShade.Material.Define(stage, "/Asset/QwenInstanceLooks/Test")
    shader = UsdShade.Shader.Define(stage, "/Asset/QwenInstanceLooks/Test/Shader")
    shader.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
    shader.SetSourceAsset(Sdf.AssetPath("./SubUSDs/materials/Test.mdl"), "mdl")
    shader.SetSourceAssetSubIdentifier("Test", "mdl")
    shader_output = shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
    shader_output.SetRenderType("material")
    material.CreateSurfaceOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
    for part in registry_document["parts"]:
        prim = stage.OverridePrim(part["prim_path"])
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            material,
            UsdShade.Tokens.weakerThanDescendants,
            UsdShade.Tokens.allPurpose,
        )
    stage.GetRootLayer().Save()
    stage = None

    mapping = {
        "version": 1,
        "file_records": [
            {
                "source_url": str(source.resolve()),
                "target_url": "./SubUSDs/source.usda",
            },
            {
                "source_url": str(original_mdl.resolve()),
                "target_url": "./SubUSDs/materials/Test.mdl",
            },
        ],
    }
    (bundle / ".collect.mapping.json").write_text(json.dumps(mapping), encoding="utf-8")

    source_sha256 = sha256_file(source)
    material_record = {
        "material_id": "mdl:test#Test",
        "material_prim_path": "/Asset/QwenInstanceLooks/Test",
        "mdl_path": str(original_mdl.resolve()),
        "subidentifier": "Test",
        "parameters": {},
    }
    applied = [
        {
            "part_id": part["part_id"],
            "prim_path": part["prim_path"],
            "face_count": part["face_count"],
            **material_record,
            "parent_binding_preserved": False,
            "parent_binding_relationship_authored": True,
            "source_subset_paths_rebound": [],
            "face_subsets": [],
        }
        for part in registry_document["parts"]
    ]
    report_document = {
        "schema_version": APPLY_SCHEMA_VERSION,
        "source_usd": str(source.resolve()),
        "source_sha256": source_sha256,
        "occurrence_registry_asset_sha256": bridge_report["output_sha256"],
        "occurrence_registry_sha256": canonical_sha256(registry_document),
        "plan_provenance": {
            "asset_sha256": source_sha256,
            "registry_sha256": canonical_sha256(registry_document),
        },
        "deinstanced_prim_count": 2,
        "mesh_occurrence_count": 2,
        "point_occurrence_count": 6,
        "face_occurrence_count": 2,
        "covered_face_occurrence_count": 2,
        "source_subset_occurrence_count": 0,
        "face_subset_count": 0,
        "applied_count": 2,
        "parent_binding_preserved_count": 0,
        "applied": applied,
        "materials": [
            {
                **material_record,
                "mdl_sha256": sha256_file(original_mdl),
            }
        ],
        "mdl_dependencies": [
            {
                "source_path": str(original_mdl.resolve()),
                "relative_to_material_root": "Test.mdl",
                "sha256": sha256_file(original_mdl),
            }
        ],
        "validation": {
            "explicit_physics_prim_count_before": 0,
            "explicit_physics_prim_count_after": 0,
        },
    }
    apply_report = tmp_path / "apply_report.json"
    apply_report.write_text(json.dumps(report_document), encoding="utf-8")

    result = validate_instance_bundle(
        source_usd=source,
        collected_root_usd=final,
        registry_path=registry,
        apply_report_path=apply_report,
        bundle_root=bundle,
    )

    assert result["status"] == "PASS"
    assert result["summary"]["failure_count"] == 0
    assert result["verified_contract"] == {
        "instance_aware_source_traversal": True,
        "mesh_occurrence_count": 2,
        "point_occurrence_count": 6,
        "face_occurrence_count": 2,
        "covered_face_occurrence_count": 2,
        "source_subset_occurrence_count": 0,
        "source_subset_face_occurrence_count": 0,
        "explicit_physics_prim_count": 0,
        "new_look_scope": "/Asset/QwenInstanceLooks",
        "authored_material_prim_count": 1,
        "declared_mdl_dependency_count": 1,
        "runtime_texture_reference_count": 0,
        "usd_asset_dependency_count": 1,
        "source_unchanged": True,
    }
