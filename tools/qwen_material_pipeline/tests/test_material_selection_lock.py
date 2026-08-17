from __future__ import annotations

import copy
from pathlib import Path

import pytest

from qwen_material_pipeline.materials.catalog import MaterialCatalog
from qwen_material_pipeline.materials.selection_lock import (
    MaterialSelectionLockError,
    build_material_selection_lock,
    validate_material_selection_lock,
)


def _fixture(tmp_path: Path) -> tuple[Path, Path, dict]:
    root = tmp_path / "Materials"
    mdl = root / "Base" / "Metals" / "Steel.mdl"
    mdl.parent.mkdir(parents=True)
    mdl.write_text(
        "mdl 1.6;\nexport material Steel(*) = material();\n",
        encoding="utf-8",
    )
    catalog_path = tmp_path / "catalog.json"
    MaterialCatalog.scan(root).save(catalog_path)
    plan = {
        "schema_version": "1.0",
        "assignments": [
            {
                "part_id": "P0001",
                "material_id": "mdl:Base/Metals/Steel.mdl#Steel",
                "status": "auto",
                "confidence": 0.99,
                "evidence_views": ["front", "side"],
            }
        ],
    }
    return root, catalog_path, plan


def test_lock_accepts_the_exact_selected_mdl_plan(tmp_path: Path) -> None:
    root, catalog, plan = _fixture(tmp_path)
    lock = build_material_selection_lock(
        plan=plan,
        catalog_path=catalog,
        material_root=root,
    )

    verified = validate_material_selection_lock(
        lock=lock,
        plan=plan,
        catalog_path=catalog,
        material_root=root,
    )

    assert verified == lock
    assert lock["post_selection_operations"]["write_parameters"] is False
    assert lock["selected_mdl_modules"][0]["mdl_path"] == ("Base/Metals/Steel.mdl")


@pytest.mark.parametrize("field", ["material_id", "parameters", "face_subsets"])
def test_lock_rejects_any_assignment_mutation(
    tmp_path: Path,
    field: str,
) -> None:
    root, catalog, plan = _fixture(tmp_path)
    lock = build_material_selection_lock(
        plan=plan,
        catalog_path=catalog,
        material_root=root,
    )
    changed = copy.deepcopy(plan)
    if field == "material_id":
        changed["assignments"][0][field] = "mdl:Base/Metals/Other.mdl#Other"
    elif field == "parameters":
        changed["assignments"][0][field] = {"roughness": 0.2}
    else:
        changed["assignments"][0][field] = [
            {
                "subset_name": "changed",
                "material_id": "mdl:Base/Metals/Steel.mdl#Steel",
                "face_indices": [0],
            }
        ]

    with pytest.raises(MaterialSelectionLockError):
        validate_material_selection_lock(
            lock=lock,
            plan=changed,
            catalog_path=catalog,
            material_root=root,
        )


def test_lock_rejects_a_changed_source_mdl(tmp_path: Path) -> None:
    root, catalog, plan = _fixture(tmp_path)
    lock = build_material_selection_lock(
        plan=plan,
        catalog_path=catalog,
        material_root=root,
    )
    (root / "Base" / "Metals" / "Steel.mdl").write_text(
        "mdl 1.6;\nexport material Steel(*) = material();\n// changed\n",
        encoding="utf-8",
    )

    with pytest.raises(MaterialSelectionLockError, match="changed after selection"):
        validate_material_selection_lock(
            lock=lock,
            plan=plan,
            catalog_path=catalog,
            material_root=root,
        )


def test_lock_cannot_be_created_for_parameter_overrides(tmp_path: Path) -> None:
    root, catalog, plan = _fixture(tmp_path)
    plan["assignments"][0]["parameters"] = {"roughness": 0.2}

    with pytest.raises(MaterialSelectionLockError, match="library-default"):
        build_material_selection_lock(
            plan=plan,
            catalog_path=catalog,
            material_root=root,
        )


def test_lock_seals_reviewed_same_mdl_colour_parameters(tmp_path: Path) -> None:
    root, catalog, plan = _fixture(tmp_path)
    plan["assignments"][0]["parameters"] = {"diffuse_tint": [0.2, 0.4, 0.1]}
    lock = build_material_selection_lock(
        plan=plan,
        catalog_path=catalog,
        material_root=root,
        allow_reviewed_color_parameters=True,
    )
    assert lock["reviewed_color_parameters_locked"] is True
    assert lock["selected_mdl_library_defaults_required"] is False
    assert (
        validate_material_selection_lock(
            lock=lock,
            plan=plan,
            catalog_path=catalog,
            material_root=root,
        )
        == lock
    )

    changed = copy.deepcopy(plan)
    changed["assignments"][0]["parameters"]["diffuse_tint"][0] = 0.3
    with pytest.raises(MaterialSelectionLockError, match="changed after selection"):
        validate_material_selection_lock(
            lock=lock,
            plan=changed,
            catalog_path=catalog,
            material_root=root,
        )


def test_reviewed_colour_lock_rejects_unreviewed_parameter(tmp_path: Path) -> None:
    root, catalog, plan = _fixture(tmp_path)
    plan["assignments"][0]["parameters"] = {"not_a_real_input": 0.5}
    with pytest.raises(MaterialSelectionLockError, match="unreviewed"):
        build_material_selection_lock(
            plan=plan,
            catalog_path=catalog,
            material_root=root,
            allow_reviewed_color_parameters=True,
        )
