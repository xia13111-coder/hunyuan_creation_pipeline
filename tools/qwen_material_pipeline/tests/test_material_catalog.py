from __future__ import annotations

import json
from pathlib import Path

import pytest

from qwen_material_pipeline.materials.catalog import (
    CATALOG_SCHEMA_VERSION,
    LEGACY_CATALOG_SCHEMA_VERSION,
    MaterialCatalog,
    MaterialCatalogError,
    MaterialPathError,
    build_catalog,
    search_materials,
)


METAL_MDL = r"""
mdl 1.6;

const string DESCRIPTION =
    "Painted industrial steel "
    "for machinery";

export material Steel_Painted_Black_Matte(*)
[[
    ::anno::display_name("Painted Steel - Black Matte"),
    ::anno::description(DESCRIPTION),
    ::anno::key_words(string[]("metal", "steel", "black", "matte", "painted"))
]] = material();

// export material Commented_Out(*)
export material Steel_Painted_Blue_Glossy(*)
[[
    ::anno::display_name("Painted Steel - Blue Glossy"),
    ::anno::description("Blue glossy machine paint"),
    ::anno::key_words(string[]("metal", "steel", "blue", "glossy"))
]] = material();
"""


PLASTIC_MDL = r"""
mdl 1.4;

export material ABS_Red(*)
[[
    display_name("ABS Red"),
    description("Smooth red engineering plastic"),
    key_words(string[]("plastic", "ABS", "red", "smooth"))
]] = material();
"""


def _write_fixture(root: Path) -> None:
    metal = root / "vMaterials_2" / "Metal"
    plastic = root / "Base" / "Plastic"
    metal.mkdir(parents=True)
    plastic.mkdir(parents=True)
    (metal / "Steel_Painted.mdl").write_text(METAL_MDL, encoding="utf-8")
    (plastic / "ABS.mdl").write_text(PLASTIC_MDL, encoding="utf-8")

    thumbnails = metal / ".thumbs" / "256x256"
    thumbnails.mkdir(parents=True)
    (thumbnails / "Steel_Painted.mdl@Steel_Painted_Black_Matte.png").write_bytes(
        b"not-a-real-png"
    )
    # Module-level fallback used when there is no per-export preview.
    (thumbnails / "Steel_Painted.mdl.png").write_bytes(b"not-a-real-png")


def test_scan_parses_exports_annotations_and_thumbnails(tmp_path: Path) -> None:
    root = tmp_path / "Materials"
    _write_fixture(root)

    catalog = MaterialCatalog.scan(root)

    assert len(catalog.materials) == 3
    black = catalog.get(
        "mdl:vMaterials_2/Metal/Steel_Painted.mdl#Steel_Painted_Black_Matte"
    )
    assert black.family == "metal"
    assert black.display_name == "Painted Steel - Black Matte"
    assert black.description == "Painted industrial steel for machinery"
    assert black.keywords == ("metal", "steel", "black", "matte", "painted")
    assert black.colors == ("black",)
    assert {"matte", "painted"} <= set(black.finishes)
    assert black.thumbnail_path == (
        "vMaterials_2/Metal/.thumbs/256x256/"
        "Steel_Painted.mdl@Steel_Painted_Black_Matte.png"
    )

    blue = catalog.get(
        "mdl:vMaterials_2/Metal/Steel_Painted.mdl#Steel_Painted_Blue_Glossy"
    )
    assert blue.thumbnail_path.endswith("Steel_Painted.mdl.png")
    assert all(record.sub_identifier != "Commented_Out" for record in catalog.materials)


def test_plural_library_directory_is_normalized_to_family(tmp_path: Path) -> None:
    root = tmp_path / "Materials"
    metals = root / "Base" / "Metals"
    metals.mkdir(parents=True)
    (metals / "Steel.mdl").write_text(
        "export material Steel(*) = material();", encoding="utf-8"
    )

    catalog = MaterialCatalog.scan(root)

    assert catalog.materials[0].family == "metal"
    assert catalog.search(family="metal")[0]["sub_identifier"] == "Steel"


def test_catalog_json_is_stable_and_loaded_against_explicit_root(
    tmp_path: Path,
) -> None:
    first_root = tmp_path / "one" / "Materials"
    second_root = tmp_path / "two" / "Materials"
    _write_fixture(first_root)
    _write_fixture(second_root)
    output = tmp_path / "catalog.json"

    first = build_catalog(first_root, output)
    second = MaterialCatalog.scan(second_root)
    loaded = MaterialCatalog.load(output, material_root=first_root)

    assert [item.material_id for item in first.materials] == [
        item.material_id for item in second.materials
    ]
    assert loaded.to_dict()["materials"] == first.to_dict()["materials"]
    document = json.loads(output.read_text(encoding="utf-8"))
    assert document["schema_version"] == CATALOG_SCHEMA_VERSION == 2
    assert document["surface_semantics_schema_version"] == (
        "qwen-catalog-surface-semantics/v1"
    )
    assert document["material_count"] == 3
    assert all("surface_semantics" in item for item in document["materials"])


def test_v1_catalog_load_is_enriched_with_surface_semantics(tmp_path: Path) -> None:
    root = tmp_path / "Materials"
    _write_fixture(root)
    catalog = MaterialCatalog.scan(root)
    document = catalog.to_dict()
    document["schema_version"] = LEGACY_CATALOG_SCHEMA_VERSION
    document.pop("surface_semantics_schema_version")
    for record in document["materials"]:
        record.pop("surface_semantics")
    path = tmp_path / "legacy_catalog.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    loaded = MaterialCatalog.load(path, material_root=root)

    assert loaded.to_dict()["schema_version"] == CATALOG_SCHEMA_VERSION
    assert all(record.surface_semantics for record in loaded.materials)


def test_full_allowlist_exactly_contains_every_catalog_export(tmp_path: Path) -> None:
    root = tmp_path / "Materials"
    _write_fixture(root)

    catalog = MaterialCatalog.scan(root)
    allowlist = catalog.to_full_allowlist_dict()

    assert allowlist["scope"] == "catalog_exact"
    assert allowlist["material_count"] == len(catalog.materials) == 3
    assert allowlist["material_ids"] == [
        record.material_id for record in catalog.materials
    ]


def test_miscellaneous_paint_is_classified_as_paint(tmp_path: Path) -> None:
    root = tmp_path / "Base"
    paint = root / "Miscellaneous"
    paint.mkdir(parents=True)
    (paint / "Paint_Matte.mdl").write_text(
        "export material Paint_Matte(*) = material();",
        encoding="utf-8",
    )

    catalog = MaterialCatalog.scan(root)

    assert catalog.materials[0].family == "paint"
    assert catalog.materials[0].surface_semantics == {
        "schema_version": "qwen-catalog-surface-semantics/v1",
        "compatible_substrates": ["metal", "polymer", "wood"],
        "surface_treatment": "paint",
        "optical_behavior": "opaque",
        "finish": "matte",
        "inference_source": "nvidia_path_name_and_authored_defaults/v1",
        "confidence": "high",
    }


def test_structured_top_k_search_and_chinese_aliases(tmp_path: Path) -> None:
    root = tmp_path / "Materials"
    _write_fixture(root)
    catalog = MaterialCatalog.scan(root)

    results = search_materials(
        catalog,
        family="steel",
        color="black",
        finish="matte",
        query="industrial painted",
        top_k=2,
    )
    assert [item["sub_identifier"] for item in results] == ["Steel_Painted_Black_Matte"]
    assert results[0]["matched_fields"] == ["family", "color", "finish", "query"]

    chinese = catalog.search(family="金属", color="蓝色", finish="亮光", top_k=1)
    assert chinese[0]["sub_identifier"] == "Steel_Painted_Blue_Glossy"


def test_unknown_id_and_invalid_top_k_are_rejected(tmp_path: Path) -> None:
    root = tmp_path / "Materials"
    _write_fixture(root)
    catalog = MaterialCatalog.scan(root)

    with pytest.raises(MaterialCatalogError, match="unknown material ID"):
        catalog.resolve_material("mdl:outside.mdl#Anything")
    with pytest.raises(MaterialCatalogError, match="positive integer"):
        catalog.search(top_k=0)


def test_tampered_catalog_cannot_escape_whitelisted_root(tmp_path: Path) -> None:
    root = tmp_path / "Materials"
    _write_fixture(root)
    catalog_path = tmp_path / "catalog.json"
    catalog = build_catalog(root, catalog_path)
    document = catalog.to_dict()
    document["material_root"] = "/"
    record = document["materials"][0]
    record["mdl_path"] = "../outside.mdl"
    record["material_id"] = "mdl:../outside.mdl#ABS_Red"
    catalog_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(MaterialPathError):
        MaterialCatalog.load(catalog_path, material_root=root)


def test_symlinked_mdl_outside_root_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "Materials"
    root.mkdir()
    outside = tmp_path / "outside.mdl"
    outside.write_text("export material Outside(*) = material();", encoding="utf-8")
    (root / "escape.mdl").symlink_to(outside)

    with pytest.raises(MaterialPathError, match="unsafe MDL path"):
        MaterialCatalog.scan(root)
