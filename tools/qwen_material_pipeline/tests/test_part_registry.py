from __future__ import annotations

import pytest

from qwen_material_pipeline.usd.registry import (
    _finite_color,
    _finite_scalar,
    _preview_material_properties,
    _validated_material_bind_face_subset_records,
    geometry_content_sha256,
    source_appearance_sha256,
    source_material_bind_subset_sha256,
    source_subset_layout_sha256,
)


class _Prim:
    def IsValid(self) -> bool:
        return True


class _Path:
    pathString = "/Looks/Diffuse/PreviewSurface"


class _Attribute:
    def __init__(self, value):
        self._value = value

    def Get(self):
        return self._value


class _Input(_Attribute):
    pass


class _Shader:
    def __init__(self) -> None:
        self._inputs = {
            "diffuseColor": (0.1, 0.2, 0.3),
            "metallic": 0.5,
            "roughness": 0.45,
            "opacity": 1.0,
        }

    def GetPrim(self) -> _Prim:
        return _Prim()

    def GetIdAttr(self) -> _Attribute:
        return _Attribute("UsdPreviewSurface")

    def GetPath(self) -> _Path:
        return _Path()

    def GetInput(self, name: str):
        value = self._inputs.get(name)
        return _Input(value) if value is not None else None


class _Material:
    def ComputeSurfaceSource(self):
        return _Shader(), "surface", "universal"


def test_preview_material_properties_preserve_weak_cad_values() -> None:
    assert _preview_material_properties(_Material()) == {
        "shader_path": "/Looks/Diffuse/PreviewSurface",
        "shader_id": "UsdPreviewSurface",
        "diffuseColor": [0.1, 0.2, 0.3],
        "metallic": 0.5,
        "roughness": 0.45,
        "opacity": 1.0,
    }


def test_preview_material_properties_fail_closed_for_missing_source() -> None:
    class Missing:
        def ComputeSurfaceSource(self):
            return None, "", ""

    assert _preview_material_properties(Missing()) is None


def test_finite_helpers_reject_non_finite_or_wrong_shape() -> None:
    assert _finite_scalar(float("nan")) is None
    assert _finite_scalar(float("inf")) is None
    assert _finite_scalar("0.25") == 0.25
    assert _finite_color((0.1, 0.2)) is None
    assert _finite_color((0.1, float("nan"), 0.3)) is None
    assert _finite_color((0.1, 0.2, 0.3, 1.0)) == [0.1, 0.2, 0.3, 1.0]


def test_geometry_content_hash_is_path_free_and_translation_invariant() -> None:
    common = {
        "face_vertex_counts": [3],
        "face_vertex_indices": [0, 1, 2],
        "orientation": "rightHanded",
        "subdivision_scheme": "none",
    }
    original = geometry_content_sha256(
        points=[(0, 0, 0), (1, 0, 0), (0, 1, 0)],
        **common,
    )
    translated = geometry_content_sha256(
        points=[(17, -4, 9), (18, -4, 9), (17, -3, 9)],
        **common,
    )
    different_topology = geometry_content_sha256(
        points=[(0, 0, 0), (1, 0, 0), (0, 1, 0)],
        face_vertex_counts=[3],
        face_vertex_indices=[0, 2, 1],
        orientation="rightHanded",
        subdivision_scheme="none",
    )

    assert translated == original
    assert different_topology != original


def test_source_appearance_hash_ignores_shader_prim_path_only() -> None:
    first = {
        "shader_path": "/Assembly/A/Looks/Paint/Shader",
        "shader_id": "UsdPreviewSurface",
        "diffuseColor": [0.1, 0.2, 0.3],
        "roughness": 0.5,
    }
    relocated = dict(first, shader_path="/Assembly/B/Looks/Paint/Shader")
    changed = dict(first, roughness=0.25)

    assert source_appearance_sha256(relocated) == source_appearance_sha256(first)
    assert source_appearance_sha256(changed) != source_appearance_sha256(first)
    assert source_appearance_sha256(None) != source_appearance_sha256(first)


def test_source_subset_layout_hash_is_path_free_but_face_exact() -> None:
    paint = _source_subset("Paint", [0, 1])
    zinc = _source_subset("Zinc", [2, 3])
    appearance = {
        paint["visual_material_prim_path"]: "paint-sha",
        zinc["visual_material_prim_path"]: "zinc-sha",
    }
    first = source_subset_layout_sha256(
        records=[paint, zinc],
        appearance_hash_by_material_path=appearance,
    )

    relocated_paint = dict(
        paint,
        subset_name="Coating",
        subset_prim_path="/Other/Mesh/Coating",
        visual_material_prim_path="/Other/Looks/Coating",
        binding_targets=["/Other/Looks/Coating"],
    )
    relocated_zinc = dict(
        zinc,
        subset_name="Metal",
        subset_prim_path="/Other/Mesh/Metal",
        visual_material_prim_path="/Other/Looks/Metal",
        binding_targets=["/Other/Looks/Metal"],
    )
    relocated = source_subset_layout_sha256(
        records=[relocated_zinc, relocated_paint],
        appearance_hash_by_material_path={
            "/Other/Looks/Coating": "paint-sha",
            "/Other/Looks/Metal": "zinc-sha",
        },
    )
    changed_faces = source_subset_layout_sha256(
        records=[dict(paint, face_indices=[0, 2]), dict(zinc, face_indices=[1, 3])],
        appearance_hash_by_material_path=appearance,
    )

    assert relocated == first
    assert changed_faces != first


def _source_subset(name: str, indices: list[int]) -> dict:
    return {
        "subset_name": name,
        "subset_prim_path": f"/Asset/Part/Mesh/{name}",
        "family_name": "materialBind",
        "family_type": "nonOverlapping",
        "element_type": "face",
        "face_indices": indices,
        "visual_material_prim_path": f"/Asset/Looks/{name}",
        "binding_relationship_name": "material:binding",
        "binding_targets": [f"/Asset/Looks/{name}"],
    }


def test_source_material_subsets_are_validated_sorted_and_hash_bound() -> None:
    records = _validated_material_bind_face_subset_records(
        part_id="P0001",
        prim_path="/Asset/Part/Mesh",
        face_count=6,
        records=[
            _source_subset("Zinc", [4, 5]),
            _source_subset("Paint", [0, 1, 2, 3]),
        ],
        verify_hashes=False,
    )

    assert [item["subset_name"] for item in records] == ["Paint", "Zinc"]
    for record in records:
        assert record["source_subset_binding_sha256"] == (
            source_material_bind_subset_sha256(
                part_id="P0001",
                prim_path="/Asset/Part/Mesh",
                subset_record=record,
            )
        )
    assert (
        _validated_material_bind_face_subset_records(
            part_id="P0001",
            prim_path="/Asset/Part/Mesh",
            face_count=6,
            records=records,
            verify_hashes=True,
        )
        == records
    )


def test_source_material_subset_rejects_overlap_and_out_of_range() -> None:
    with pytest.raises(RuntimeError, match="overlap"):
        _validated_material_bind_face_subset_records(
            part_id="P0001",
            prim_path="/Asset/Part/Mesh",
            face_count=6,
            records=[
                _source_subset("Paint", [0, 1]),
                _source_subset("Zinc", [1, 2]),
            ],
            verify_hashes=False,
        )
    with pytest.raises(RuntimeError, match="outside"):
        _validated_material_bind_face_subset_records(
            part_id="P0001",
            prim_path="/Asset/Part/Mesh",
            face_count=6,
            records=[_source_subset("Paint", [6])],
            verify_hashes=False,
        )


def test_source_material_subset_rejects_invalid_family_and_tampered_hash() -> None:
    invalid_family = _source_subset("Paint", [0])
    invalid_family["family_type"] = "notAUsdFamilyType"
    with pytest.raises(RuntimeError, match="invalid materialBind family type"):
        _validated_material_bind_face_subset_records(
            part_id="P0001",
            prim_path="/Asset/Part/Mesh",
            face_count=6,
            records=[invalid_family],
            verify_hashes=False,
        )

    unrestricted = _source_subset("Paint", [0])
    unrestricted["family_type"] = "unrestricted"
    assert _validated_material_bind_face_subset_records(
        part_id="P0001",
        prim_path="/Asset/Part/Mesh",
        face_count=6,
        records=[unrestricted],
        verify_hashes=False,
    )[0]["family_type"] == "unrestricted"

    record = _validated_material_bind_face_subset_records(
        part_id="P0001",
        prim_path="/Asset/Part/Mesh",
        face_count=6,
        records=[_source_subset("Paint", [0])],
        verify_hashes=False,
    )[0]
    record["face_indices"] = [1]
    with pytest.raises(RuntimeError, match="invalid source subset hash"):
        _validated_material_bind_face_subset_records(
            part_id="P0001",
            prim_path="/Asset/Part/Mesh",
            face_count=6,
            records=[record],
            verify_hashes=True,
        )
