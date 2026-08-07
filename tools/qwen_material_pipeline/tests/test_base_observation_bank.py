from __future__ import annotations

import json
from pathlib import Path

import pytest

from qwen_material_pipeline.materials.base_observation_bank import (
    BaseObservationBankError,
    _parse_omnipbr_defaults,
    initialize_bank,
    resolve_base_root,
)


def _write_material(path: Path, name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""
mdl 1.4;
export material {name}(*) = ::OmniPBR::OmniPBR(
    diffuse_color_constant: color(0.1f, 0.2f, 0.3f),
    reflection_roughness_constant: 0.42f,
    metallic_constant: 0.75f,
    enable_ORM_texture: false,
    diffuse_texture: texture_2d("./{name}.png"),
    normalmap_texture: texture_2d());
""",
        encoding="utf-8",
    )


def test_base_root_accepts_collection_or_parent_and_rejects_vmaterials(
    tmp_path: Path,
) -> None:
    materials = tmp_path / "NVIDIA" / "Materials"
    base = materials / "Base"
    vmaterials = materials / "vMaterials_2"
    base.mkdir(parents=True)
    vmaterials.mkdir()

    assert resolve_base_root(materials) == base
    assert resolve_base_root(base) == base
    with pytest.raises(BaseObservationBankError, match="forbidden"):
        resolve_base_root(vmaterials)


def test_initialize_bank_is_exactly_base_only(tmp_path: Path) -> None:
    materials = tmp_path / "NVIDIA" / "Materials"
    _write_material(materials / "Base" / "Metals" / "BaseSteel.mdl", "BaseSteel")
    _write_material(
        materials / "vMaterials_2" / "Metal" / "Forbidden.mdl", "Forbidden"
    )
    output = tmp_path / "bank"

    catalog, report = initialize_bank(
        material_root=materials,
        output_dir=output,
    )

    assert [record.sub_identifier for record in catalog.materials] == ["BaseSteel"]
    assert report["scope"] == "nvidia_base"
    assert report["material_count"] == 1
    assert report["forbidden_vmaterials_2_count"] == 0
    allowlist = json.loads((output / "allowlist.json").read_text(encoding="utf-8"))
    assert allowlist["material_count"] == 1
    assert all("vMaterials_2" not in value for value in allowlist["material_ids"])


def test_parse_omnipbr_defaults_preserves_authored_pbr_values() -> None:
    parsed = _parse_omnipbr_defaults(
        """
        diffuse_color_constant: color(0.1f, 0.2f, 0.3f),
        reflection_roughness_constant: 0.42f,
        metallic_constant: 0.75f,
        enable_ORM_texture: true,
        diffuse_texture: texture_2d("./steel.png"),
        normalmap_texture: texture_2d()
        """
    )

    assert parsed["diffuse_color_constant"] == [0.1, 0.2, 0.3]
    assert parsed["reflection_roughness_constant"] == 0.42
    assert parsed["metallic_constant"] == 0.75
    assert parsed["enable_ORM_texture"] is True
    assert parsed["diffuse_texture"] == "./steel.png"
    assert parsed["normalmap_texture"] is None
