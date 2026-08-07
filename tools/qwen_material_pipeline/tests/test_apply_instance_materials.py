from __future__ import annotations

import json
from pathlib import Path

import pytest

from qwen_material_pipeline.usd.apply_instances import (
    _validate_plan_provenance,
)
from qwen_material_pipeline.usd.stage_state import canonical_sha256


GENERIC_PAINT = "mdl:vMaterials_2/Metal/Steel_Painted.mdl#Steel_Painted"
BLACK_PAINT = "mdl:vMaterials_2/Metal/Steel_Painted.mdl#Steel_Painted_Black"


def test_canonical_registry_digest_is_key_order_independent() -> None:
    left = {"parts": [{"part_id": "P0001", "prim_path": "/资产/Mesh"}], "z": 2}
    right = {"z": 2, "parts": [{"prim_path": "/资产/Mesh", "part_id": "P0001"}]}
    assert canonical_sha256(left) == canonical_sha256(right)


def test_plan_provenance_fails_closed() -> None:
    valid = {
        "provenance": {
            "asset_sha256": "asset",
            "registry_sha256": "registry",
        }
    }
    assert _validate_plan_provenance(
        valid, source_sha256="asset", registry_sha256="registry"
    ) == {"asset_sha256": "asset", "registry_sha256": "registry"}

    with pytest.raises(ValueError, match="provenance object"):
        _validate_plan_provenance({}, source_sha256="asset", registry_sha256="registry")
    with pytest.raises(ValueError, match="asset_sha256"):
        _validate_plan_provenance(
            {
                "provenance": {
                    "asset_sha256": "wrong",
                    "registry_sha256": "registry",
                }
            },
            source_sha256="asset",
            registry_sha256="registry",
        )
    with pytest.raises(ValueError, match="registry_sha256"):
        _validate_plan_provenance(
            {
                "provenance": {
                    "asset_sha256": "asset",
                    "registry_sha256": "wrong",
                }
            },
            source_sha256="asset",
            registry_sha256="registry",
        )


def test_instance_application_reuses_source_subsets_and_preserves_geometry(
    tmp_path: Path,
) -> None:
    pytest.importorskip("pxr")
    from pxr import Gf, Usd, UsdGeom, UsdShade, Vt

    from qwen_material_pipeline.usd.apply_instances import (
        apply_instance_materials,
    )
    from qwen_material_pipeline.usd.instances import (
        create_editable_instance_layer,
    )
    from qwen_material_pipeline.usd.material_common import sha256_file
    from qwen_material_pipeline.usd.registry import build_part_registry
    from qwen_material_pipeline.usd.stage_state import material_binding_path

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
                Gf.Vec3f(1.0, 1.0, 0.0),
                Gf.Vec3f(0.0, 1.0, 0.0),
            ]
        )
    )
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3, 3]))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray([0, 1, 2, 0, 2, 3]))
    subset = UsdGeom.Subset.CreateGeomSubset(
        mesh,
        "Surface",
        UsdGeom.Tokens.face,
        Vt.IntArray([0, 1]),
        UsdShade.Tokens.materialBind,
        UsdGeom.Tokens.nonOverlapping,
    )
    original = UsdShade.Material.Define(stage, "/Asset/Looks/Original")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(original)
    UsdShade.MaterialBindingAPI.Apply(subset.GetPrim()).Bind(original)
    stage.CreateClassPrim("/Asset/_PlainPart")
    plain_mesh = UsdGeom.Mesh.Define(stage, "/Asset/_PlainPart/Mesh")
    plain_mesh.CreatePointsAttr(mesh.GetPointsAttr().Get())
    plain_mesh.CreateFaceVertexCountsAttr(mesh.GetFaceVertexCountsAttr().Get())
    plain_mesh.CreateFaceVertexIndicesAttr(mesh.GetFaceVertexIndicesAttr().Get())
    UsdShade.MaterialBindingAPI.Apply(plain_mesh.GetPrim()).Bind(original)
    for index, name in enumerate(("A", "B")):
        occurrence = UsdGeom.Xform.Define(stage, f"/Asset/{name}")
        occurrence.GetPrim().GetReferences().AddInternalReference("/Asset/_Part")
        occurrence.GetPrim().SetInstanceable(True)
        occurrence.AddTranslateOp().Set(Gf.Vec3d(index * 2.0, 0.0, 0.0))
    occurrence = UsdGeom.Xform.Define(stage, "/Asset/C")
    occurrence.GetPrim().GetReferences().AddInternalReference("/Asset/_PlainPart")
    occurrence.GetPrim().SetInstanceable(True)
    occurrence.AddTranslateOp().Set(Gf.Vec3d(4.0, 0.0, 0.0))
    stage.GetRootLayer().Save()
    stage = None
    source_sha256 = sha256_file(source)

    bridge = tmp_path / "editable.usda"
    bridge_report = create_editable_instance_layer(source_usd=source, output_usd=bridge)
    assert bridge_report["deinstanced_prim_count"] == 3
    registry_document = build_part_registry(bridge)
    assert registry_document["part_count"] == 3
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps(registry_document), encoding="utf-8")

    material_root = tmp_path / "materials"
    material_root.mkdir()
    mdl = material_root / "Steel_Painted.mdl"
    mdl.write_text("mdl 1.7;\n", encoding="utf-8")
    catalog_document = {
        "materials": [
            {
                "material_id": GENERIC_PAINT,
                "mdl_path": str(mdl),
                "sub_identifier": "Steel_Painted",
            },
            {
                "material_id": BLACK_PAINT,
                "mdl_path": str(mdl),
                "sub_identifier": "Steel_Painted_Black",
            },
        ]
    }
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps(catalog_document), encoding="utf-8")

    assignments = []
    for part in registry_document["parts"]:
        assignment = {
            "part_id": part["part_id"],
            "material_id": GENERIC_PAINT,
            "parameters": {
                "paint_color": [0.02, 0.2, 0.04],
                "paint_roughness": 0.4,
            },
            "confidence": 0.99,
            "status": "approved",
        }
        if part["prim_path"] == "/Asset/A/Mesh":
            assignment.update(
                {
                    "preserve_parent_material_binding": True,
                    "face_subsets": [
                        {
                            "subset_name": "Surface",
                            "material_id": BLACK_PAINT,
                            "face_indices": [0, 1],
                        }
                    ],
                }
            )
        elif part["prim_path"] == "/Asset/C/Mesh":
            assignment.update(
                {
                    "preserve_parent_material_binding": True,
                    "face_subsets": [
                        {
                            "subset_name": "AutoMaterial_Green",
                            "material_id": BLACK_PAINT,
                            "face_indices": [0],
                        }
                    ],
                }
            )
        assignments.append(assignment)
    plan_document = {
        "schema_version": "1.0",
        "provenance": {
            "asset_sha256": source_sha256,
            "registry_sha256": canonical_sha256(registry_document),
        },
        "assignments": assignments,
    }
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(plan_document), encoding="utf-8")

    output = tmp_path / "look.usda"
    report = apply_instance_materials(
        source_usd=source,
        catalog_path=catalog,
        registry_path=registry,
        plan_path=plan,
        output_usd=output,
        material_root=material_root,
    )

    assert report["deinstanced_prim_count"] == 3
    assert report["mesh_occurrence_count"] == 3
    assert report["point_occurrence_count"] == 12
    assert report["face_occurrence_count"] == 6
    assert report["covered_face_occurrence_count"] == 6
    assert report["source_subset_occurrence_count"] == 2
    assert report["authored_face_subset_count"] == 1
    assert report["verified_subset_binding_count"] == 3
    assert report["planned_face_subset_override_count"] == 2
    assert report["parent_binding_preserved_count"] == 2
    assert report["validation"]["explicit_physics_prim_count_before"] == 0
    assert report["validation"]["explicit_physics_prim_count_after"] == 0
    assert sha256_file(source) == source_sha256

    composed = Usd.Stage.Open(str(output), load=Usd.Stage.LoadAll)
    assert sum(prim.IsInstance() for prim in composed.Traverse()) == 0
    assert material_binding_path(composed.GetPrimAtPath("/Asset/A/Mesh")) == (
        "/Asset/Looks/Original"
    )
    assert material_binding_path(
        composed.GetPrimAtPath("/Asset/A/Mesh/Surface")
    ).startswith("/Asset/QwenInstanceLooks/")
    b_material = material_binding_path(composed.GetPrimAtPath("/Asset/B/Mesh"))
    assert b_material.startswith("/Asset/QwenInstanceLooks/")
    assert (
        material_binding_path(composed.GetPrimAtPath("/Asset/B/Mesh/Surface"))
        == b_material
    )
    assert (
        material_binding_path(composed.GetPrimAtPath("/Asset/C/Mesh"))
        == "/Asset/Looks/Original"
    )
    assert material_binding_path(
        composed.GetPrimAtPath("/Asset/C/Mesh/AutoMaterial_Green")
    ).startswith("/Asset/QwenInstanceLooks/")
