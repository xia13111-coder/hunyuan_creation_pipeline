#!/usr/bin/env python3
"""Bind one visual MDL material to every Mesh and material face subset in a USD.

The operation is intentionally separate from ``add_physics.py``: this script
authors only all-purpose visual bindings and verifies that Physics-purpose
bindings and Mesh attributes are unchanged.  By default the MDL document and
its relative texture dependencies are copied beside the USD so the resulting
asset does not depend on the original absolute material-library path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any


_EXPORT_MATERIAL_RE = re.compile(r"\bexport\s+material\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(")
_TEXTURE_ASSET_RE = re.compile(r'\btexture_(?:1d|2d|3d|cube|ptex)\s*\(\s*"([^"]+)"')


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _atomic_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copy2(source, temporary)
        temporary.chmod(0o644)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _material_exports(mdl_path: Path) -> list[str]:
    text = mdl_path.read_text(encoding="utf-8", errors="strict")
    return sorted(set(_EXPORT_MATERIAL_RE.findall(text)))


def _relative_texture_dependencies(mdl_path: Path) -> list[tuple[Path, Path]]:
    text = mdl_path.read_text(encoding="utf-8", errors="strict")
    source_root = mdl_path.parent.resolve()
    dependencies: list[tuple[Path, Path]] = []
    seen: set[Path] = set()
    for authored_value in _TEXTURE_ASSET_RE.findall(text):
        if "://" in authored_value:
            raise ValueError(
                f"Remote texture dependency is not supported: {authored_value}"
            )
        relative = Path(authored_value)
        if relative.is_absolute():
            raise ValueError(
                f"Absolute texture dependency is not portable: {authored_value}"
            )
        source = (source_root / relative).resolve(strict=True)
        if not _is_inside(source, source_root):
            raise ValueError(
                f"Texture dependency escapes the MDL directory: {authored_value}"
            )
        normalized_relative = source.relative_to(source_root)
        if normalized_relative not in seen:
            dependencies.append((source, normalized_relative))
            seen.add(normalized_relative)
    return sorted(dependencies, key=lambda item: item[1].as_posix())


def _attribute_snapshot(prim: Any) -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    for attribute in prim.GetAttributes():
        time_samples = attribute.GetTimeSamples()
        snapshot[attribute.GetName()] = {
            "default": repr(attribute.Get()),
            "samples": [
                [sample, repr(attribute.Get(sample))] for sample in time_samples
            ],
        }
    return snapshot


def _relationship_snapshot(relationship: Any) -> dict[str, Any]:
    from pxr import UsdShade

    result: dict[str, Any] = {
        "targets": [target.pathString for target in relationship.GetTargets()]
    }
    if relationship.GetName().startswith("material:binding"):
        result["strength"] = str(
            UsdShade.MaterialBindingAPI.GetMaterialBindingStrength(relationship)
        )
    return result


def _physics_snapshot(stage: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for prim in stage.TraverseAll():
        attributes = {
            attribute.GetName(): {
                "default": repr(attribute.Get()),
                "samples": [
                    [sample, repr(attribute.Get(sample))]
                    for sample in attribute.GetTimeSamples()
                ],
            }
            for attribute in prim.GetAttributes()
            if attribute.GetName().startswith(("physics:", "physx"))
        }
        relationships = {
            relationship.GetName(): _relationship_snapshot(relationship)
            for relationship in prim.GetRelationships()
            if relationship.GetName().startswith(("physics:", "physx"))
            or relationship.GetName() == "material:binding:physics"
        }
        schemas = sorted(
            schema
            for schema in prim.GetAppliedSchemas()
            if "physics" in schema.casefold() or "physx" in schema.casefold()
        )
        if attributes or relationships or schemas:
            result[prim.GetPath().pathString] = {
                "attributes": attributes,
                "relationships": relationships,
                "schemas": schemas,
            }
    return result


def _mesh_snapshot(stage: Any) -> dict[str, Any]:
    from pxr import UsdGeom

    return {
        prim.GetPath().pathString: _attribute_snapshot(prim)
        for prim in stage.TraverseAll()
        if prim.IsA(UsdGeom.Mesh)
    }


def _material_subsets(stage: Any) -> list[Any]:
    from pxr import UsdGeom, UsdShade

    return [
        prim
        for prim in stage.TraverseAll()
        if prim.IsA(UsdGeom.Subset)
        and UsdGeom.Subset(prim).GetFamilyNameAttr().Get()
        == UsdShade.Tokens.materialBind
    ]


def _binding_path(prim: Any) -> str | None:
    from pxr import UsdShade

    material, _relationship = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial(
        materialPurpose=UsdShade.Tokens.allPurpose
    )
    if not material or not material.GetPrim().IsValid():
        return None
    return material.GetPath().pathString


def _resolve_subidentifier(mdl_path: Path, requested: str | None) -> str:
    exports = _material_exports(mdl_path)
    if requested:
        if requested not in exports:
            raise ValueError(
                f"MDL has no exported material {requested!r}; available: {exports}"
            )
        return requested
    if len(exports) != 1:
        raise ValueError(
            "MDL subidentifier is ambiguous; pass --subidentifier. "
            f"Available: {exports}"
        )
    return exports[0]


def apply_uniform_mdl(
    *,
    usd_path: str | Path,
    mdl_path: str | Path,
    subidentifier: str | None = None,
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Apply one bundled MDL to every Mesh and materialBind GeomSubset."""

    from pxr import Sdf, Usd, UsdGeom, UsdShade

    usd = Path(usd_path).expanduser().resolve(strict=True)
    mdl = Path(mdl_path).expanduser().resolve(strict=True)
    if usd.suffix.casefold() not in {".usd", ".usda", ".usdc"}:
        raise ValueError(f"Unsupported USD extension: {usd}")
    if mdl.suffix.casefold() != ".mdl":
        raise ValueError(f"Expected an MDL document: {mdl}")

    chosen_subidentifier = _resolve_subidentifier(mdl, subidentifier)
    texture_dependencies = _relative_texture_dependencies(mdl)
    before_sha256 = _sha256(usd)

    backup = usd.with_name(f"{usd.name}.before_uniform_mdl.bak")
    if not backup.exists():
        _atomic_copy(usd, backup)
    backup_sha256 = _sha256(backup)

    material_dir = usd.parent / "materials"
    bundled_mdl = material_dir / mdl.name
    _atomic_copy(mdl, bundled_mdl)
    copied_dependencies: list[dict[str, str]] = []
    for source, relative in texture_dependencies:
        destination = material_dir / relative
        _atomic_copy(source, destination)
        copied_dependencies.append(
            {
                "source": str(source),
                "bundled": str(destination.relative_to(usd.parent)),
                "sha256": _sha256(destination),
            }
        )

    temporary = usd.with_name(f".{usd.stem}.uniform_mdl.tmp{usd.suffix}")
    temporary.unlink(missing_ok=True)
    shutil.copy2(usd, temporary)
    try:
        stage = Usd.Stage.Open(str(temporary), load=Usd.Stage.LoadAll)
        if stage is None:
            raise RuntimeError(f"Unable to open USD stage: {temporary}")
        default_prim = stage.GetDefaultPrim()
        if not default_prim or not default_prim.IsValid():
            raise ValueError(f"USD has no valid default prim: {usd}")

        meshes = [prim for prim in stage.TraverseAll() if prim.IsA(UsdGeom.Mesh)]
        if not meshes:
            raise ValueError(f"USD contains no Mesh prims: {usd}")
        subsets = _material_subsets(stage)
        mesh_paths_before = [prim.GetPath().pathString for prim in meshes]
        subset_paths_before = [prim.GetPath().pathString for prim in subsets]
        mesh_state_before = _mesh_snapshot(stage)
        physics_state_before = _physics_snapshot(stage)
        previous_visual_materials = sorted(
            {path for prim in meshes + subsets if (path := _binding_path(prim))}
        )

        looks_path = default_prim.GetPath().AppendChild("UniformVisualMaterials")
        UsdGeom.Scope.Define(stage, looks_path)
        material_path = looks_path.AppendChild(chosen_subidentifier)
        material = UsdShade.Material.Define(stage, material_path)
        shader = UsdShade.Shader.Define(stage, material_path.AppendChild("Shader"))
        shader.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
        authored_mdl_path = f"./materials/{mdl.name}"
        shader.SetSourceAsset(Sdf.AssetPath(authored_mdl_path), "mdl")
        shader.SetSourceAssetSubIdentifier(chosen_subidentifier, "mdl")
        shader_output = shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
        shader_output.SetRenderType("material")
        connectable = shader.ConnectableAPI()
        material.CreateSurfaceOutput("mdl").ConnectToSource(connectable, "out")
        material.CreateVolumeOutput("mdl").ConnectToSource(connectable, "out")
        material.CreateDisplacementOutput("mdl").ConnectToSource(connectable, "out")

        for prim in meshes + subsets:
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                material,
                UsdShade.Tokens.strongerThanDescendants,
                UsdShade.Tokens.allPurpose,
            )

        stage.GetRootLayer().Save()
        del stage

        verification_stage = Usd.Stage.Open(str(temporary), load=Usd.Stage.LoadAll)
        if verification_stage is None:
            raise RuntimeError(f"Unable to reopen authored USD: {temporary}")
        verified_meshes = [
            prim for prim in verification_stage.TraverseAll() if prim.IsA(UsdGeom.Mesh)
        ]
        verified_subsets = _material_subsets(verification_stage)
        verified_material = verification_stage.GetPrimAtPath(material_path)
        if not verified_material or not verified_material.IsA(UsdShade.Material):
            raise RuntimeError(f"Authored material is missing: {material_path}")

        material_schema = UsdShade.Material(verified_material)
        verified_shader, _name, _type = material_schema.ComputeSurfaceSource("mdl")
        if not verified_shader or not verified_shader.GetPrim().IsValid():
            raise RuntimeError("Authored MDL surface shader does not resolve")
        source_asset = verified_shader.GetSourceAsset("mdl")
        if source_asset is None or source_asset.path != authored_mdl_path:
            raise RuntimeError(
                f"Unexpected authored MDL path: {source_asset.path if source_asset else None}"
            )
        if verified_shader.GetSourceAssetSubIdentifier("mdl") != chosen_subidentifier:
            raise RuntimeError("Authored MDL subidentifier changed")
        resolved_mdl = Path(source_asset.resolvedPath).resolve(strict=False)
        if not source_asset.resolvedPath or resolved_mdl != bundled_mdl:
            raise RuntimeError(
                f"Bundled MDL does not resolve correctly: {source_asset}"
            )

        if [prim.GetPath().pathString for prim in verified_meshes] != mesh_paths_before:
            raise RuntimeError("Mesh prim paths changed while applying visual material")
        if [
            prim.GetPath().pathString for prim in verified_subsets
        ] != subset_paths_before:
            raise RuntimeError(
                "materialBind GeomSubset paths changed while applying visual material"
            )
        if _mesh_snapshot(verification_stage) != mesh_state_before:
            raise RuntimeError(
                "Mesh geometry or authored attributes changed while applying material"
            )
        if _physics_snapshot(verification_stage) != physics_state_before:
            raise RuntimeError(
                "Physics state changed while applying the visual material"
            )

        expected_material_path = material_path.pathString
        wrong_mesh_bindings = [
            prim.GetPath().pathString
            for prim in verified_meshes
            if _binding_path(prim) != expected_material_path
        ]
        wrong_subset_bindings = [
            prim.GetPath().pathString
            for prim in verified_subsets
            if _binding_path(prim) != expected_material_path
        ]
        if wrong_mesh_bindings or wrong_subset_bindings:
            raise RuntimeError(
                "Uniform visual material did not cover every Mesh/subset: "
                f"meshes={wrong_mesh_bindings[:3]}, subsets={wrong_subset_bindings[:3]}"
            )
        del verification_stage

        os.replace(temporary, usd)
    finally:
        temporary.unlink(missing_ok=True)

    report = {
        "schema_version": "uniform-mdl-binding/v1",
        "status": "PASS",
        "usd": str(usd),
        "backup": str(backup),
        "backup_sha256": backup_sha256,
        "usd_sha256_before": before_sha256,
        "usd_sha256_after": _sha256(usd),
        "source_mdl": str(mdl),
        "source_mdl_sha256": _sha256(mdl),
        "bundled_mdl": str(bundled_mdl.relative_to(usd.parent)),
        "bundled_mdl_sha256": _sha256(bundled_mdl),
        "subidentifier": chosen_subidentifier,
        "material_prim_path": material_path.pathString,
        "shader_prim_path": material_path.AppendChild("Shader").pathString,
        "mesh_count": len(mesh_paths_before),
        "material_subset_count": len(subset_paths_before),
        "mesh_binding_count": len(mesh_paths_before),
        "material_subset_binding_count": len(subset_paths_before),
        "previous_visual_material_target_count": len(previous_visual_materials),
        "all_meshes_bound": True,
        "all_material_subsets_bound": True,
        "mesh_attributes_unchanged": True,
        "physics_state_unchanged": True,
        "copied_texture_dependencies": copied_dependencies,
    }
    if report_path is not None:
        report_file = Path(report_path).expanduser().resolve()
        report_file.parent.mkdir(parents=True, exist_ok=True)
        report_file.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        report["report"] = str(report_file)
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Bind one bundled visual MDL to every Mesh and material face subset "
            "without changing Physics-purpose bindings"
        )
    )
    parser.add_argument("--usd", required=True, help="USD file to update in place")
    parser.add_argument("--mdl", required=True, help="Source MDL document")
    parser.add_argument(
        "--subidentifier",
        help="Exported MDL material name; inferred only when exactly one exists",
    )
    parser.add_argument("--report", help="Optional JSON verification report")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    report = apply_uniform_mdl(
        usd_path=args.usd,
        mdl_path=args.mdl,
        subidentifier=args.subidentifier,
        report_path=args.report,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
