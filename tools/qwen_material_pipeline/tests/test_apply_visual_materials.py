from __future__ import annotations

import pytest

from qwen_material_pipeline.usd.apply import (
    _match_existing_material_subsets,
    _propagate_parent_assignment_to_existing_subsets,
)
from qwen_material_pipeline.usd.material_common import (
    json_material_parameters,
    material_instance_key,
    normalize_face_subsets,
    normalize_material_parameters,
    preserve_parent_material_binding,
)


GENERIC_PAINT = "mdl:vMaterials_2/Metal/Steel_Painted.mdl#Steel_Painted"
BLACK_PAINT = "mdl:vMaterials_2/Metal/Steel_Painted.mdl#Steel_Painted_Black"


def test_existing_material_subsets_can_only_be_rebound_with_exact_topology() -> None:
    planned = [
        {"subset_name": "Cover", "face_indices": (4, 1)},
        {"subset_name": "Fasteners", "face_indices": (2, 3)},
    ]
    existing = [
        {
            "subset_name": "Fasteners",
            "subset_prim_path": "/Asset/Mesh/Fasteners",
            "element_type": "face",
            "family_name": "materialBind",
            "face_indices": [2, 3],
        },
        {
            "subset_name": "Cover",
            "subset_prim_path": "/Asset/Mesh/Cover",
            "element_type": "face",
            "family_name": "materialBind",
            "face_indices": [4, 1],
        },
    ]

    assert _match_existing_material_subsets("P0001", planned, existing) == 2
    assert planned[0]["source_subset_rebind"] is True
    assert planned[0]["source_subset_prim_path"] == "/Asset/Mesh/Cover"
    assert planned[1]["source_subset_prim_path"] == "/Asset/Mesh/Fasteners"


def test_whole_mesh_parent_assignment_is_propagated_to_source_subsets() -> None:
    parameters = {
        "paint_color": (0.01, 0.2, 0.04),
        "paint_roughness": 0.32,
    }
    existing = [
        {
            "subset_name": "Cover",
            "subset_prim_path": "/Asset/Mesh/Cover",
            "element_type": "face",
            "family_name": "materialBind",
            "face_indices": [4, 1],
        },
        {
            "subset_name": "Fasteners",
            "subset_prim_path": "/Asset/Mesh/Fasteners",
            "element_type": "face",
            "family_name": "materialBind",
            "face_indices": [2, 3],
        },
    ]

    propagated = _propagate_parent_assignment_to_existing_subsets(
        "P0001",
        existing,
        material_id=GENERIC_PAINT,
        parameters=parameters,
    )

    assert [item["subset_name"] for item in propagated] == ["Cover", "Fasteners"]
    assert [item["face_indices"] for item in propagated] == [(4, 1), (2, 3)]
    assert all(item["material_id"] == GENERIC_PAINT for item in propagated)
    assert all(item["parameters"] == parameters for item in propagated)
    assert all(item["source_subset_rebind"] is True for item in propagated)
    assert all(item["explicit_plan_override"] is False for item in propagated)
    assert all(item["parent_assignment_propagated"] is True for item in propagated)

    # The synthesized records must not share a mutable parameter dictionary
    # with the parent assignment or with each other.
    assert propagated[0]["parameters"] is not parameters
    assert propagated[0]["parameters"] is not propagated[1]["parameters"]


@pytest.mark.parametrize(
    ("existing", "message"),
    [
        (
            [
                {
                    "subset_name": "Other",
                    "subset_prim_path": "/Asset/Mesh/Other",
                    "element_type": "face",
                    "family_name": "materialBind",
                    "face_indices": [0],
                }
            ],
            "membership does not exactly match",
        ),
        (
            [
                {
                    "subset_name": "Cover",
                    "subset_prim_path": "/Asset/Mesh/Cover",
                    "element_type": "face",
                    "family_name": "materialBind",
                    "face_indices": [1],
                }
            ],
            "indices differ",
        ),
        (
            [
                {
                    "subset_name": "Cover",
                    "subset_prim_path": "/Asset/Mesh/Cover",
                    "element_type": "point",
                    "family_name": "materialBind",
                    "face_indices": [0],
                }
            ],
            "not a face subset",
        ),
        (
            [
                {
                    "subset_name": "Cover",
                    "subset_prim_path": "/Asset/Mesh/Cover",
                    "element_type": "face",
                    "family_name": "selection",
                    "face_indices": [0],
                }
            ],
            "not in the materialBind family",
        ),
    ],
)
def test_existing_material_subset_rebind_fails_closed(
    existing: list[dict], message: str
) -> None:
    planned = [{"subset_name": "Cover", "face_indices": (0,)}]
    with pytest.raises(ValueError, match=message):
        _match_existing_material_subsets("P0001", planned, existing)


def test_subset_only_parent_binding_policy_is_explicit_and_requires_subsets() -> None:
    assignment = {
        "material_id": GENERIC_PAINT,
        "parameters": {"paint_color": [0.01, 0.2, 0.04]},
        "preserve_parent_material_binding": True,
    }
    assert (
        preserve_parent_material_binding("P0024", assignment, has_face_subsets=True)
        is True
    )
    assert (
        preserve_parent_material_binding(
            "P0024", {"material_id": GENERIC_PAINT}, has_face_subsets=False
        )
        is False
    )

    with pytest.raises(ValueError, match="requires face_subsets"):
        preserve_parent_material_binding("P0024", assignment, has_face_subsets=False)
    with pytest.raises(ValueError, match="must be a boolean"):
        preserve_parent_material_binding(
            "P0024",
            {"preserve_parent_material_binding": 1},
            has_face_subsets=True,
        )


def test_generic_painted_steel_accepts_bounded_linear_color() -> None:
    value = normalize_material_parameters(
        GENERIC_PAINT, {"paint_color": [0.01, 0.2, 0.04]}
    )
    assert value == {"paint_color": (0.01, 0.2, 0.04)}


def test_generic_painted_steel_accepts_strict_clean_paint_controls() -> None:
    value = normalize_material_parameters(
        GENERIC_PAINT,
        {
            "paint_roughness": 0.32,
            "paint_roughness_variation": 0.04,
            "dirt_weight": 0,
            "wash_weight": 0.0,
            "paint_stroke_normal_strength": 0.05,
            "uneven_normal_strength": 0.02,
            "enable_rust_damage": False,
        },
    )
    assert value == {
        "paint_roughness": 0.32,
        "paint_roughness_variation": 0.04,
        "dirt_weight": 0.0,
        "wash_weight": 0.0,
        "paint_stroke_normal_strength": 0.05,
        "uneven_normal_strength": 0.02,
        "enable_rust_damage": False,
    }
    assert json_material_parameters(value) == value


@pytest.mark.parametrize(
    "parameters",
    [
        {"invented": [0.1, 0.2, 0.3]},
        {"paint_color": [0.1, 0.2]},
        {"paint_color": [-0.1, 0.2, 0.3]},
        {"paint_color": [0.1, float("nan"), 0.3]},
        {"paint_roughness": True},
        {"paint_roughness": "0.3"},
        {"paint_roughness": -0.01},
        {"paint_roughness": 1.01},
        {"paint_roughness": float("nan")},
        {"paint_roughness": float("inf")},
        {"enable_rust_damage": 0},
        {"enable_rust_damage": 1},
        {"enable_rust_damage": "false"},
    ],
)
def test_parameters_fail_closed(parameters: dict) -> None:
    with pytest.raises(ValueError):
        normalize_material_parameters(GENERIC_PAINT, parameters)


def test_parameterized_material_instances_have_distinct_keys() -> None:
    first = material_instance_key(GENERIC_PAINT, {"paint_color": (0.01, 0.2, 0.04)})
    second = material_instance_key(GENERIC_PAINT, {"paint_color": (0.2, 0.01, 0.04)})
    assert first != second

    matte = material_instance_key(
        GENERIC_PAINT,
        {"paint_roughness": 0.32, "enable_rust_damage": False},
    )
    glossy = material_instance_key(
        GENERIC_PAINT,
        {"paint_roughness": 0.12, "enable_rust_damage": False},
    )
    rusty = material_instance_key(
        GENERIC_PAINT,
        {"paint_roughness": 0.32, "enable_rust_damage": True},
    )
    assert len({matte, glossy, rusty}) == 3


def test_face_subsets_accept_bounded_non_overlapping_faces_and_parameters() -> None:
    value = normalize_face_subsets(
        "P0024",
        [
            {
                "subset_name": "green_panels",
                "material_id": GENERIC_PAINT,
                "parameters": {"paint_color": [0.02, 0.2, 0.04]},
                "semantic": "green painted panels",
                "face_indices": [4, 1],
            },
            {
                "subset_name": "black_controls",
                "material_id": BLACK_PAINT,
                "face_indices": [2, 3],
            },
        ],
        allowed_material_ids={GENERIC_PAINT, BLACK_PAINT},
        face_count=5,
    )
    assert value[0]["face_indices"] == (4, 1)
    assert value[0]["parameters"] == {"paint_color": (0.02, 0.2, 0.04)}
    assert value[1]["parameters"] == {}


@pytest.mark.parametrize(
    ("face_subsets", "message"),
    [
        ([], "non-empty list"),
        (
            [
                {
                    "subset_name": "bad/name",
                    "material_id": BLACK_PAINT,
                    "face_indices": [0],
                }
            ],
            "unsafe subset_name",
        ),
        (
            [
                {
                    "subset_name": "black",
                    "material_id": "NOT_ALLOWED",
                    "face_indices": [0],
                }
            ],
            "unknown material_id",
        ),
        (
            [
                {
                    "subset_name": "black",
                    "material_id": BLACK_PAINT,
                    "face_indices": [0, 0],
                }
            ],
            "must be unique",
        ),
        (
            [
                {
                    "subset_name": "black",
                    "material_id": BLACK_PAINT,
                    "face_indices": [-1],
                }
            ],
            "out of range",
        ),
        (
            [
                {
                    "subset_name": "black",
                    "material_id": BLACK_PAINT,
                    "face_indices": [5],
                }
            ],
            "out of range",
        ),
        (
            [
                {
                    "subset_name": "black",
                    "material_id": BLACK_PAINT,
                    "face_indices": [True],
                }
            ],
            "only integers",
        ),
    ],
)
def test_face_subsets_fail_closed(face_subsets: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        normalize_face_subsets(
            "P0024",
            face_subsets,
            allowed_material_ids={GENERIC_PAINT, BLACK_PAINT},
            face_count=5,
        )


def test_face_subsets_cannot_overlap() -> None:
    with pytest.raises(ValueError, match="overlap.*3"):
        normalize_face_subsets(
            "P0024",
            [
                {
                    "subset_name": "first",
                    "material_id": BLACK_PAINT,
                    "face_indices": [1, 3],
                },
                {
                    "subset_name": "second",
                    "material_id": BLACK_PAINT,
                    "face_indices": [3, 4],
                },
            ],
            allowed_material_ids={BLACK_PAINT},
            face_count=5,
        )


def test_face_subsets_reject_duplicate_names_and_unapproved_parameters() -> None:
    with pytest.raises(ValueError, match="Duplicate subset_name"):
        normalize_face_subsets(
            "P0024",
            [
                {
                    "subset_name": "same",
                    "material_id": BLACK_PAINT,
                    "face_indices": [0],
                },
                {
                    "subset_name": "same",
                    "material_id": BLACK_PAINT,
                    "face_indices": [1],
                },
            ],
            allowed_material_ids={BLACK_PAINT},
            face_count=2,
        )
    with pytest.raises(ValueError, match="Unsupported material parameters"):
        normalize_face_subsets(
            "P0024",
            [
                {
                    "subset_name": "black",
                    "material_id": BLACK_PAINT,
                    "parameters": {"enable_rust_damage": True},
                    "face_indices": [0],
                }
            ],
            allowed_material_ids={BLACK_PAINT},
            face_count=1,
        )


def test_subset_only_application_preserves_parent_binding_and_authors_subset(
    tmp_path,
) -> None:
    pytest.importorskip("pxr")
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt

    from qwen_material_pipeline.usd.apply import (
        apply_visual_materials,
    )
    from qwen_material_pipeline.usd.material_common import sha256_file

    source = tmp_path / "source.usda"
    stage = Usd.Stage.CreateNew(str(source))
    root = UsdGeom.Xform.Define(stage, "/Asset").GetPrim()
    stage.SetDefaultPrim(root)
    mesh = UsdGeom.Mesh.Define(stage, "/Asset/Mesh")
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
    original = UsdShade.Material.Define(stage, "/Asset/Looks/Original")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(original)
    stage.GetRootLayer().Save()
    stage = None

    material_root = tmp_path / "materials"
    material_root.mkdir()
    mdl = material_root / "Steel_Painted.mdl"
    mdl.write_text("mdl 1.7;\n", encoding="utf-8")

    def write_json(path, value) -> None:
        import json

        path.write_text(json.dumps(value), encoding="utf-8")

    catalog = tmp_path / "catalog.json"
    write_json(
        catalog,
        {
            "materials": [
                {
                    "material_id": GENERIC_PAINT,
                    "mdl_path": str(mdl),
                    "sub_identifier": "Steel_Painted",
                }
            ]
        },
    )
    registry = tmp_path / "registry.json"
    write_json(
        registry,
        {
            "asset_usd": str(source.resolve()),
            "asset_sha256": sha256_file(source),
            "parts": [{"part_id": "P0001", "prim_path": "/Asset/Mesh"}],
        },
    )
    plan = tmp_path / "plan.json"
    write_json(
        plan,
        {
            "schema_version": "1.0",
            "assignments": [
                {
                    "part_id": "P0001",
                    "material_id": GENERIC_PAINT,
                    "parameters": {"paint_color": [0.01, 0.2, 0.04]},
                    "confidence": 0.99,
                    "status": "approved",
                    "preserve_parent_material_binding": True,
                    "face_subsets": [
                        {
                            "subset_name": "proven_green_face",
                            "material_id": GENERIC_PAINT,
                            "parameters": {"paint_color": [0.01, 0.2, 0.04]},
                            "face_indices": [0],
                        }
                    ],
                }
            ],
        },
    )

    output = tmp_path / "look.usda"
    report = apply_visual_materials(
        asset_usd=source,
        catalog_path=catalog,
        registry_path=registry,
        plan_path=plan,
        output_usd=output,
        material_root=material_root,
    )

    assert report["parent_binding_preserved_count"] == 1
    assert report["mesh_occurrence_count"] == 1
    assert report["point_occurrence_count"] == 4
    assert report["face_occurrence_count"] == 2
    assert report["covered_face_occurrence_count"] == 2
    assert report["applied"][0]["parent_binding_preserved"] is True
    assert report["applied"][0]["parent_binding_relationship_authored"] is False
    assert "material_prim_path" not in report["applied"][0]
    composed = Usd.Stage.Open(str(output), load=Usd.Stage.LoadAll)
    composed_mesh = composed.GetPrimAtPath("/Asset/Mesh")
    bound, _relationship = UsdShade.MaterialBindingAPI(
        composed_mesh
    ).ComputeBoundMaterial(materialPurpose=UsdShade.Tokens.allPurpose)
    assert bound.GetPath().pathString == "/Asset/Looks/Original"
    assert (
        composed.GetRootLayer().GetPropertyAtPath(
            Sdf.Path("/Asset/Mesh.material:binding")
        )
        is None
    )
    subset = composed.GetPrimAtPath("/Asset/Mesh/proven_green_face")
    subset_bound, _relationship = UsdShade.MaterialBindingAPI(
        subset
    ).ComputeBoundMaterial(materialPurpose=UsdShade.Tokens.allPurpose)
    assert subset_bound.GetPath().pathString.startswith("/Asset/QwenLooks/")

    # Replacing the all-purpose binding must not be rejected as a physics
    # mutation.  USD's ComputeBoundMaterial("physics") falls back to this
    # visual binding when no explicit physics relationship exists.
    replacement_plan = tmp_path / "replacement_plan.json"
    write_json(
        replacement_plan,
        {
            "schema_version": "1.0",
            "assignments": [
                {
                    "part_id": "P0001",
                    "material_id": GENERIC_PAINT,
                    "parameters": {"paint_color": [0.01, 0.2, 0.04]},
                    "confidence": 0.99,
                    "status": "approved",
                }
            ],
        },
    )
    replacement_output = tmp_path / "replacement_look.usda"
    replacement_report = apply_visual_materials(
        asset_usd=source,
        catalog_path=catalog,
        registry_path=registry,
        plan_path=replacement_plan,
        output_usd=replacement_output,
        material_root=material_root,
    )

    assert replacement_report["applied_count"] == 1
    replacement_stage = Usd.Stage.Open(str(replacement_output), load=Usd.Stage.LoadAll)
    replacement_mesh = replacement_stage.GetPrimAtPath("/Asset/Mesh")
    replacement_bound, _relationship = UsdShade.MaterialBindingAPI(
        replacement_mesh
    ).ComputeBoundMaterial(materialPurpose=UsdShade.Tokens.allPurpose)
    assert replacement_bound.GetPath().pathString.startswith("/Asset/QwenLooks/")
    assert not replacement_mesh.GetRelationship("material:binding:physics").GetTargets()


def test_whole_mesh_application_rebinds_existing_subsets_without_topology_edits(
    tmp_path,
) -> None:
    pytest.importorskip("pxr")
    import json

    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt

    from qwen_material_pipeline.usd.apply import apply_visual_materials
    from qwen_material_pipeline.usd.material_common import sha256_file

    source = tmp_path / "source-with-subsets.usda"
    stage = Usd.Stage.CreateNew(str(source))
    root = UsdGeom.Xform.Define(stage, "/Asset").GetPrim()
    stage.SetDefaultPrim(root)
    mesh = UsdGeom.Mesh.Define(stage, "/Asset/Mesh")
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
    cover = UsdGeom.Subset.CreateGeomSubset(
        mesh,
        "Cover",
        UsdGeom.Tokens.face,
        Vt.IntArray([0]),
        UsdShade.Tokens.materialBind,
        UsdGeom.Tokens.partition,
    )
    fasteners = UsdGeom.Subset.CreateGeomSubset(
        mesh,
        "Fasteners",
        UsdGeom.Tokens.face,
        Vt.IntArray([1]),
        UsdShade.Tokens.materialBind,
        UsdGeom.Tokens.partition,
    )
    original_parent = UsdShade.Material.Define(stage, "/Asset/Looks/OriginalParent")
    original_cover = UsdShade.Material.Define(stage, "/Asset/Looks/OriginalCover")
    original_fasteners = UsdShade.Material.Define(
        stage, "/Asset/Looks/OriginalFasteners"
    )
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(original_parent)
    UsdShade.MaterialBindingAPI.Apply(cover.GetPrim()).Bind(original_cover)
    UsdShade.MaterialBindingAPI.Apply(fasteners.GetPrim()).Bind(original_fasteners)
    stage.GetRootLayer().Save()
    stage = None

    material_root = tmp_path / "materials-with-subsets"
    material_root.mkdir()
    mdl = material_root / "Steel_Painted.mdl"
    mdl.write_text("mdl 1.7;\n", encoding="utf-8")
    catalog = tmp_path / "catalog-with-subsets.json"
    catalog.write_text(
        json.dumps(
            {
                "materials": [
                    {
                        "material_id": GENERIC_PAINT,
                        "mdl_path": str(mdl),
                        "sub_identifier": "Steel_Painted",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    registry = tmp_path / "registry-with-subsets.json"
    registry.write_text(
        json.dumps(
            {
                "asset_usd": str(source.resolve()),
                "asset_sha256": sha256_file(source),
                "parts": [{"part_id": "P0001", "prim_path": "/Asset/Mesh"}],
            }
        ),
        encoding="utf-8",
    )
    plan = tmp_path / "plan-with-subsets.json"
    plan.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "assignments": [
                    {
                        "part_id": "P0001",
                        "material_id": GENERIC_PAINT,
                        "parameters": {
                            "paint_color": [0.01, 0.2, 0.04],
                            "paint_roughness": 0.32,
                        },
                        "confidence": 0.99,
                        "status": "approved",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    output = tmp_path / "look-with-subsets.usda"
    report = apply_visual_materials(
        asset_usd=source,
        catalog_path=catalog,
        registry_path=registry,
        plan_path=plan,
        output_usd=output,
        material_root=material_root,
    )

    assert report["existing_subset_rebind_count"] == 2
    assert report["parent_assignment_subset_rebind_count"] == 2
    assert report["planned_face_subset_override_count"] == 0
    assert report["verified_subset_binding_count"] == 2
    assert (
        report["validation"][
            "whole_mesh_parent_material_propagated_to_existing_subsets"
        ]
        is True
    )
    record = report["applied"][0]
    assert record["source_subset_paths_rebound"] == [
        "/Asset/Mesh/Cover",
        "/Asset/Mesh/Fasteners",
    ]
    assert record["parent_assignment_subset_rebind_count"] == 2
    assert all(
        subset["explicit_plan_override"] is False
        and subset["parent_assignment_propagated"] is True
        and subset["material_id"] == record["material_id"]
        and subset["material_prim_path"] == record["material_prim_path"]
        and subset["parameters"] == record["parameters"]
        for subset in record["face_subsets"]
    )

    composed = Usd.Stage.Open(str(output), load=Usd.Stage.LoadAll)
    composed_mesh = UsdGeom.Mesh(composed.GetPrimAtPath("/Asset/Mesh"))
    assert (
        UsdGeom.Subset.GetFamilyType(
            composed_mesh, UsdShade.Tokens.materialBind
        )
        == UsdGeom.Tokens.partition
    )
    expected_indices = {
        "/Asset/Mesh/Cover": [0],
        "/Asset/Mesh/Fasteners": [1],
    }
    for subset_path, face_indices in expected_indices.items():
        subset_prim = composed.GetPrimAtPath(subset_path)
        subset = UsdGeom.Subset(subset_prim)
        assert list(subset.GetIndicesAttr().Get()) == face_indices
        assert subset.GetElementTypeAttr().Get() == UsdGeom.Tokens.face
        assert subset.GetFamilyNameAttr().Get() == UsdShade.Tokens.materialBind
        bound, _relationship = UsdShade.MaterialBindingAPI(
            subset_prim
        ).ComputeBoundMaterial(materialPurpose=UsdShade.Tokens.allPurpose)
        assert bound.GetPath().pathString == record["material_prim_path"]
        assert [
            target.pathString
            for target in subset_prim.GetRelationship(
                "material:binding"
            ).GetTargets()
        ] == [record["material_prim_path"]]
        for property_name in ("indices", "elementType", "familyName"):
            assert (
                composed.GetRootLayer().GetPropertyAtPath(
                    Sdf.Path(subset_path).AppendProperty(property_name)
                )
                is None
            )
    assert (
        composed.GetRootLayer().GetPropertyAtPath(
            Sdf.Path("/Asset/Mesh").AppendProperty(
                "subsetFamily:materialBind:familyType"
            )
        )
        is None
    )


def test_source_visual_preserve_is_exact_material_noop(tmp_path) -> None:
    pytest.importorskip("pxr")
    import json

    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt

    from qwen_material_pipeline.usd.apply import apply_visual_materials
    from qwen_material_pipeline.usd.material_common import (
        POLICY_EXACT_COVER_MODE,
        POLICY_FALLBACK_CONFIDENCE_BASIS,
        SOURCE_VISUAL_PRESERVE_ACTION,
        canonical_sha256,
        sha256_file,
        source_visual_binding_sha256,
    )

    source = tmp_path / "source-preserve.usda"
    stage = Usd.Stage.CreateNew(str(source))
    root = UsdGeom.Xform.Define(stage, "/Asset").GetPrim()
    stage.SetDefaultPrim(root)
    mesh = UsdGeom.Mesh.Define(stage, "/Asset/Mesh")
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
    subset = UsdGeom.Subset.CreateGeomSubset(
        mesh,
        "AuthoredSurface",
        UsdGeom.Tokens.face,
        Vt.IntArray([0]),
        UsdShade.Tokens.materialBind,
        UsdGeom.Tokens.nonOverlapping,
    )
    original = UsdShade.Material.Define(stage, "/Asset/Looks/Original")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(original)
    UsdShade.MaterialBindingAPI.Apply(subset.GetPrim()).Bind(original)
    stage.GetRootLayer().Save()
    stage = None

    registry_document = {
        "schema_version": "qwen-material-parts/v1",
        "asset_usd": str(source.resolve()),
        "asset_sha256": sha256_file(source),
        "part_count": 1,
        "parts": [
            {
                "part_id": "P0001",
                "prim_path": "/Asset/Mesh",
                "existing_visual_material": "/Asset/Looks/Original",
            }
        ],
    }
    registry = tmp_path / "registry-preserve.json"
    registry.write_text(json.dumps(registry_document), encoding="utf-8")
    binding_digest = source_visual_binding_sha256(
        part_id="P0001",
        prim_path="/Asset/Mesh",
        material_prim_path="/Asset/Looks/Original",
    )
    plan_document = {
        "schema_version": "1.0",
        "provenance": {
            "mode": POLICY_EXACT_COVER_MODE,
            "registry_asset_sha256": registry_document["asset_sha256"],
            "registry_sha256": canonical_sha256(registry_document),
        },
        "assignments": [
            {
                "part_id": "P0001",
                # Deliberately absent from the catalog: no-op application must
                # not resolve or author the compatibility fallback material.
                "material_id": "mdl:unused#Neutral",
                "confidence": 0.0,
                "evidence_views": [],
                "status": "policy_fallback",
                "apply_action": SOURCE_VISUAL_PRESERVE_ACTION,
                "source_visual_material_prim_path": "/Asset/Looks/Original",
                "source_visual_material_binding_sha256": binding_digest,
                "provenance": {
                    "tier": "source_visual_preserve",
                    "reason_codes": [
                        "SOURCE_VISUAL_MATERIAL_PRESENT",
                        "SOURCE_VISUAL_BINDING_HASH_BOUND",
                        "PRESERVE_SOURCE_VISUAL_NOOP",
                    ],
                    "output_confidence_basis": POLICY_FALLBACK_CONFIDENCE_BASIS,
                    "sources": [],
                },
            }
        ],
    }
    plan = tmp_path / "plan-preserve.json"
    plan.write_text(json.dumps(plan_document), encoding="utf-8")
    material_root = tmp_path / "materials-preserve"
    material_root.mkdir()
    placeholder = material_root / "Placeholder.mdl"
    placeholder.write_text("mdl 1.7;\n", encoding="utf-8")
    catalog = tmp_path / "catalog-preserve.json"
    catalog.write_text(
        json.dumps(
            {
                "materials": [
                    {
                        "material_id": "mdl:placeholder#Material",
                        "mdl_path": str(placeholder),
                        "sub_identifier": "Material",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    output = tmp_path / "look-preserve.usda"
    report = apply_visual_materials(
        asset_usd=source,
        catalog_path=catalog,
        registry_path=registry,
        plan_path=plan,
        output_usd=output,
        material_root=material_root,
        include_policy_fallback=True,
    )

    assert report["source_visual_preserve_count"] == 1
    assert report["source_visual_preserved_subset_count"] == 1
    record = report["applied"][0]
    assert record["source_visual_preserved"] is True
    assert record["source_visual_material_binding_sha256"] == binding_digest
    assert record["face_subsets"] == []
    composed = Usd.Stage.Open(str(output), load=Usd.Stage.LoadAll)
    assert (
        UsdShade.MaterialBindingAPI(composed.GetPrimAtPath("/Asset/Mesh"))
        .ComputeBoundMaterial(materialPurpose=UsdShade.Tokens.allPurpose)[0]
        .GetPath()
        .pathString
        == "/Asset/Looks/Original"
    )
    assert (
        composed.GetRootLayer().GetPropertyAtPath(
            Sdf.Path("/Asset/Mesh.material:binding")
        )
        is None
    )
    assert (
        composed.GetRootLayer().GetPropertyAtPath(
            Sdf.Path("/Asset/Mesh/AuthoredSurface.material:binding")
        )
        is None
    )
