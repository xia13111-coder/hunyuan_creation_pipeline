#!/usr/bin/env python3
"""Fail-closed visual-material validation after Physics and USD collection.

This validator intentionally permits Physics/PhysX additions.  It verifies the
orthogonal contract: every planned Mesh still resolves the authored allPurpose
MDL material in the Look, Physics, and collected stages, and collected MDL
dependencies remain inside the delivery directory.
"""

from __future__ import annotations

import argparse
import json
import math
import traceback
from pathlib import Path
from typing import Any, Mapping

from qwen_material_pipeline.usd.material_common import (
    source_visual_binding_sha256,
)


SCHEMA_VERSION = "qwen-material-delivery-validation/v1"


def _start_isaac_if_needed(headless: bool = True):
    try:
        from pxr import Usd  # noqa: F401

        return None
    except ImportError:
        try:
            from isaacsim import SimulationApp
        except ImportError as exc:
            raise RuntimeError(
                "pxr is unavailable. Run this command with Isaac Sim python.sh."
            ) from exc
        return SimulationApp({"headless": headless})


def _load_object(path: str | Path, label: str) -> tuple[Path, dict[str, Any]]:
    resolved = Path(path).expanduser().resolve(strict=True)
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {resolved}")
    return resolved, value


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
        return True
    except ValueError:
        return False


def _value_matches(actual: Any, expected: Any) -> bool:
    if type(expected) is bool:
        return type(actual) is bool and actual is expected
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        try:
            return math.isclose(
                float(actual), float(expected), rel_tol=1e-6, abs_tol=1e-7
            )
        except (TypeError, ValueError):
            return False
    if isinstance(expected, list) and len(expected) == 3:
        try:
            return all(
                math.isclose(float(found), float(wanted), rel_tol=1e-6, abs_tol=1e-7)
                for found, wanted in zip(actual, expected, strict=True)
            )
        except (TypeError, ValueError):
            return False
    return actual == expected


def _material_record(record: Mapping[str, Any], label: str) -> dict[str, Any]:
    required = ("material_prim_path", "subidentifier", "parameters")
    if any(field not in record for field in required):
        raise ValueError(f"{label} is missing an authored material contract")
    material_path = record["material_prim_path"]
    subidentifier = record["subidentifier"]
    parameters = record["parameters"]
    if not isinstance(material_path, str) or not material_path.startswith("/"):
        raise ValueError(f"{label}.material_prim_path is invalid")
    if not isinstance(subidentifier, str) or not subidentifier:
        raise ValueError(f"{label}.subidentifier is invalid")
    if not isinstance(parameters, dict):
        raise ValueError(f"{label}.parameters must be an object")
    return {
        "material_prim_path": material_path,
        "subidentifier": subidentifier,
        "parameters": parameters,
    }


def _binding_contracts(
    registry: dict[str, Any], apply_report: dict[str, Any]
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    raw_parts = registry.get("parts")
    raw_applied = apply_report.get("applied")
    if not isinstance(raw_parts, list) or not raw_parts:
        raise ValueError("Registry contains no parts")
    if not isinstance(raw_applied, list) or not raw_applied:
        raise ValueError("Apply report contains no applied records")

    registry_by_id: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(raw_parts):
        if not isinstance(raw, dict):
            raise ValueError(f"Registry part {index} is invalid")
        part_id = raw.get("part_id")
        prim_path = raw.get("prim_path")
        if not isinstance(part_id, str) or not isinstance(prim_path, str):
            raise ValueError(f"Registry part {index} has invalid identity")
        if part_id in registry_by_id:
            raise ValueError(f"Registry duplicates part_id: {part_id}")
        registry_by_id[part_id] = raw

    applied_by_id: dict[str, dict[str, Any]] = {}
    subset_contracts: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_applied):
        if not isinstance(raw, dict):
            raise ValueError(f"Apply record {index} is invalid")
        part_id = raw.get("part_id")
        prim_path = raw.get("prim_path")
        registered = registry_by_id.get(part_id)
        if registered is None or registered.get("prim_path") != prim_path:
            raise ValueError(f"Apply record does not match registry: {part_id}")
        if part_id in applied_by_id:
            raise ValueError(f"Apply report duplicates part_id: {part_id}")
        preserved = raw.get("parent_binding_preserved", False)
        if type(preserved) is not bool:
            raise ValueError(f"Invalid parent-binding policy: {part_id}")
        source_visual_preserved = raw.get("source_visual_preserved", False)
        if type(source_visual_preserved) is not bool:
            raise ValueError(f"Invalid source-visual preserve policy: {part_id}")
        if source_visual_preserved and not preserved:
            raise ValueError(
                f"Source-visual preserve must preserve the parent binding: {part_id}"
            )
        source_visual_path = None
        if source_visual_preserved:
            source_visual_path = raw.get("source_visual_material_prim_path")
            registry_source_path = registered.get("existing_visual_material")
            if (
                not isinstance(source_visual_path, str)
                or source_visual_path != registry_source_path
            ):
                raise ValueError(
                    f"Source-visual preserve binding differs from registry: {part_id}"
                )
            expected_digest = source_visual_binding_sha256(
                part_id=str(part_id),
                prim_path=str(prim_path),
                material_prim_path=source_visual_path,
            )
            if raw.get("source_visual_material_binding_sha256") != expected_digest:
                raise ValueError(
                    f"Source-visual preserve binding hash is invalid: {part_id}"
                )
        parent_contract = None if preserved else _material_record(raw, str(part_id))
        subsets = raw.get("face_subsets", [])
        if not isinstance(subsets, list):
            raise ValueError(f"face_subsets must be an array: {part_id}")
        for subset_index, subset in enumerate(subsets):
            if not isinstance(subset, dict):
                raise ValueError(f"Invalid face subset: {part_id}[{subset_index}]")
            subset_path = subset.get("subset_prim_path")
            if not isinstance(subset_path, str) or not subset_path.startswith("/"):
                raise ValueError(f"Invalid subset prim path: {part_id}[{subset_index}]")
            subset_contracts.append(
                {
                    "part_id": part_id,
                    "prim_path": subset_path,
                    **_material_record(subset, f"{part_id}[{subset_index}]"),
                }
            )
        source_visual_subsets = raw.get("source_visual_subset_bindings", [])
        if not isinstance(source_visual_subsets, list):
            raise ValueError(
                f"source_visual_subset_bindings must be an array: {part_id}"
            )
        normalized_source_subsets: list[dict[str, str]] = []
        for subset_index, subset in enumerate(source_visual_subsets):
            if not source_visual_preserved or not isinstance(subset, Mapping):
                raise ValueError(
                    f"Invalid source-visual subset: {part_id}[{subset_index}]"
                )
            subset_path = subset.get("subset_prim_path")
            subset_material_path = subset.get(
                "source_visual_material_prim_path"
            )
            if (
                not isinstance(subset_path, str)
                or not subset_path.startswith("/")
                or not isinstance(subset_material_path, str)
                or not subset_material_path.startswith("/")
            ):
                raise ValueError(
                    f"Invalid source-visual subset contract: "
                    f"{part_id}[{subset_index}]"
                )
            normalized_source_subsets.append(
                {
                    "subset_prim_path": subset_path,
                    "source_visual_material_prim_path": subset_material_path,
                }
            )
        applied_by_id[str(part_id)] = {
            "part_id": part_id,
            "prim_path": prim_path,
            "parent_binding_preserved": preserved,
            "source_visual_preserved": source_visual_preserved,
            "source_visual_material_prim_path": source_visual_path,
            "source_visual_subset_bindings": normalized_source_subsets,
            "parent_material": parent_contract,
        }

    expected_ids = set(registry_by_id)
    if set(applied_by_id) != expected_ids:
        raise ValueError(
            "Apply report does not exactly cover the registry: "
            f"missing={sorted(expected_ids - set(applied_by_id))[:20]}"
        )
    if apply_report.get("applied_count") != len(expected_ids):
        raise ValueError("Apply report applied_count is not exact coverage")
    if apply_report.get("covered_face_occurrence_count") is not None and (
        apply_report.get("covered_face_occurrence_count")
        != apply_report.get("face_occurrence_count")
    ):
        raise ValueError("Apply report does not cover every occurrence face")
    return applied_by_id, subset_contracts


def _resolved_binding(prim):
    from pxr import UsdShade

    material, _relationship = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial(
        materialPurpose=UsdShade.Tokens.allPurpose
    )
    return material if material and material.GetPrim().IsValid() else None


def _check_material(
    *,
    material,
    expected: Mapping[str, Any],
    stage_label: str,
    binding_label: str,
    bundle_root: Path,
    collected: bool,
    failures: list[dict[str, Any]],
) -> None:
    material_path = material.GetPath().pathString if material else None
    if material_path != expected["material_prim_path"]:
        failures.append(
            {
                "stage": stage_label,
                "binding": binding_label,
                "reason": "material_path",
                "expected": expected["material_prim_path"],
                "actual": material_path,
            }
        )
        return
    shader, _source_name, _source_type = material.ComputeSurfaceSource("mdl")
    if not shader or not shader.GetPrim().IsValid():
        failures.append(
            {"stage": stage_label, "binding": binding_label, "reason": "mdl_shader"}
        )
        return
    actual_subidentifier = shader.GetSourceAssetSubIdentifier("mdl")
    if actual_subidentifier != expected["subidentifier"]:
        failures.append(
            {
                "stage": stage_label,
                "binding": binding_label,
                "reason": "subidentifier",
                "expected": expected["subidentifier"],
                "actual": actual_subidentifier,
            }
        )
    for name, wanted in expected["parameters"].items():
        shader_input = shader.GetInput(name)
        actual = shader_input.Get() if shader_input else None
        if not _value_matches(actual, wanted):
            failures.append(
                {
                    "stage": stage_label,
                    "binding": binding_label,
                    "reason": f"parameter:{name}",
                    "expected": wanted,
                    "actual": repr(actual),
                }
            )
    source_asset = shader.GetSourceAsset("mdl")
    resolved = Path(source_asset.resolvedPath).resolve(strict=False)
    if not source_asset.resolvedPath or not resolved.is_file():
        failures.append(
            {
                "stage": stage_label,
                "binding": binding_label,
                "reason": "unresolved_mdl",
                "authored_path": source_asset.path,
                "resolved_path": source_asset.resolvedPath,
            }
        )
    elif collected and not _inside(resolved, bundle_root):
        failures.append(
            {
                "stage": stage_label,
                "binding": binding_label,
                "reason": "mdl_outside_bundle",
                "resolved_path": str(resolved),
            }
        )


def validate_visual_material_delivery(
    *,
    look_usd: str | Path,
    physics_usd: str | Path,
    collected_root_usd: str | Path,
    registry_path: str | Path,
    apply_report_path: str | Path,
    bundle_root: str | Path,
) -> dict[str, Any]:
    from pxr import Usd, UsdGeom

    look_path = Path(look_usd).expanduser().resolve(strict=True)
    physics_path = Path(physics_usd).expanduser().resolve(strict=True)
    collected_path = Path(collected_root_usd).expanduser().resolve(strict=True)
    bundle_path = Path(bundle_root).expanduser().resolve(strict=True)
    if not bundle_path.is_dir() or not _inside(collected_path, bundle_path):
        raise ValueError("Collected USD must be inside its bundle root")
    registry_file, registry = _load_object(registry_path, "registry")
    apply_file, apply_report = _load_object(apply_report_path, "apply report")
    applied, subsets = _binding_contracts(registry, apply_report)

    stages = {
        "look": look_path,
        "physics": physics_path,
        "collected": collected_path,
    }
    failures: list[dict[str, Any]] = []
    metrics: dict[str, Any] = {
        "expected_mesh_count": len(applied),
        "expected_subset_binding_count": len(subsets),
    }
    expected_mesh_paths = {record["prim_path"] for record in applied.values()}
    for stage_label, stage_path in stages.items():
        stage = Usd.Stage.Open(str(stage_path), load=Usd.Stage.LoadAll)
        if stage is None:
            raise RuntimeError(f"Unable to open {stage_label} USD: {stage_path}")
        mesh_paths = {
            prim.GetPath().pathString
            for prim in stage.Traverse()
            if prim.IsA(UsdGeom.Mesh)
        }
        metrics[f"{stage_label}_mesh_count"] = len(mesh_paths)
        if mesh_paths != expected_mesh_paths:
            failures.append(
                {
                    "stage": stage_label,
                    "reason": "mesh_exact_cover",
                    "missing": sorted(expected_mesh_paths - mesh_paths)[:20],
                    "unexpected": sorted(mesh_paths - expected_mesh_paths)[:20],
                }
            )
        for record in applied.values():
            prim = stage.GetPrimAtPath(record["prim_path"])
            material = _resolved_binding(prim) if prim and prim.IsValid() else None
            if material is None:
                failures.append(
                    {
                        "stage": stage_label,
                        "binding": record["part_id"],
                        "reason": "missing_parent_visual_binding",
                    }
                )
                continue
            contract = record["parent_material"]
            if contract is not None:
                _check_material(
                    material=material,
                    expected=contract,
                    stage_label=stage_label,
                    binding_label=str(record["part_id"]),
                    bundle_root=bundle_path,
                    collected=stage_label == "collected",
                    failures=failures,
                )
            elif record["source_visual_preserved"]:
                actual_path = material.GetPath().pathString
                expected_path = record["source_visual_material_prim_path"]
                if actual_path != expected_path:
                    failures.append(
                        {
                            "stage": stage_label,
                            "binding": record["part_id"],
                            "reason": "source_visual_binding_changed",
                            "expected": expected_path,
                            "actual": actual_path,
                        }
                    )
                for subset in record["source_visual_subset_bindings"]:
                    subset_prim = stage.GetPrimAtPath(
                        subset["subset_prim_path"]
                    )
                    subset_material = (
                        _resolved_binding(subset_prim)
                        if subset_prim and subset_prim.IsValid()
                        else None
                    )
                    actual_subset_path = (
                        subset_material.GetPath().pathString
                        if subset_material
                        else None
                    )
                    expected_subset_path = subset[
                        "source_visual_material_prim_path"
                    ]
                    if actual_subset_path != expected_subset_path:
                        failures.append(
                            {
                                "stage": stage_label,
                                "binding": subset["subset_prim_path"],
                                "reason": (
                                    "source_visual_subset_binding_changed"
                                ),
                                "expected": expected_subset_path,
                                "actual": actual_subset_path,
                            }
                        )
        for subset in subsets:
            prim = stage.GetPrimAtPath(subset["prim_path"])
            material = _resolved_binding(prim) if prim and prim.IsValid() else None
            if material is None:
                failures.append(
                    {
                        "stage": stage_label,
                        "binding": subset["prim_path"],
                        "reason": "missing_subset_visual_binding",
                    }
                )
                continue
            _check_material(
                material=material,
                expected=subset,
                stage_label=stage_label,
                binding_label=subset["prim_path"],
                bundle_root=bundle_path,
                collected=stage_label == "collected",
                failures=failures,
            )

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS" if not failures else "FAIL",
        "overall_pass": not failures,
        "inputs": {
            "look_usd": str(look_path),
            "physics_usd": str(physics_path),
            "collected_root_usd": str(collected_path),
            "registry": str(registry_file),
            "apply_report": str(apply_file),
            "bundle_root": str(bundle_path),
        },
        "metrics": metrics,
        "failure_count": len(failures),
        "failures": failures,
    }


def _write_report(report: dict[str, Any], output: str | Path) -> Path:
    path = Path(output).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate visual materials after Physics and USD collection"
    )
    parser.add_argument("--look-usd", required=True)
    parser.add_argument("--physics-usd", required=True)
    parser.add_argument("--collected-root-usd", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--apply-report", required=True)
    parser.add_argument("--bundle-root", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    app = None
    try:
        app = _start_isaac_if_needed()
        report = validate_visual_material_delivery(
            look_usd=args.look_usd,
            physics_usd=args.physics_usd,
            collected_root_usd=args.collected_root_usd,
            registry_path=args.registry,
            apply_report_path=args.apply_report,
            bundle_root=args.bundle_root,
        )
        output = _write_report(report, args.output)
        print(
            json.dumps(
                {
                    "overall_pass": report["overall_pass"],
                    "output": str(output),
                    "metrics": report["metrics"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return 0 if report["overall_pass"] else 1
    except Exception:
        traceback.print_exc()
        return 1
    finally:
        if app is not None:
            app.close()


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "SCHEMA_VERSION",
    "parse_args",
    "validate_visual_material_delivery",
]
