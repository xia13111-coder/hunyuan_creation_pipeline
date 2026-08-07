#!/usr/bin/env python3
"""Shared implementation for collected USD bundle validation.

Both regular and instance-aware validators consume the public audit and bundle
inspection API in this module.  It never mutates either the source or collected
stage.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import re
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from qwen_material_pipeline.usd.material_common import (
    source_visual_binding_sha256,
)


SCHEMA_VERSION = "qwen-material-final-bundle-validation/v1"
GEOMETRY_ATTRIBUTES = (
    "points",
    "faceVertexCounts",
    "faceVertexIndices",
    "holeIndices",
    "orientation",
    "subdivisionScheme",
)
CHECK_LABELS = {
    "inputs": "Input identity and report integrity",
    "registry": "Part registry exact Mesh coverage",
    "geometry": "Exact Mesh geometry and topology",
    "transforms": "Exact local and world transforms",
    "physics": "Exact Physics/PhysX state",
    "visual_bindings": "Applied and preserved visual bindings",
    "mdl_materials": "MDL sources, subidentifiers, and parameters",
    "geom_subsets": "Material GeomSubset definitions and bindings",
    "usd_dependencies": "Collected USD dependencies",
    "mdl_textures": "Collected MDL runtime texture dependencies",
}

_TEXTURE_2D_RE = re.compile(r'\btexture_2d\s*\(\s*"((?:\\.|[^"\\])*)"', re.MULTILINE)
_THUMBNAIL_RE = re.compile(
    r'(?:::)?anno::thumbnail\s*\(\s*"((?:\\.|[^"\\])*)"', re.MULTILINE
)
_TILE_TOKENS = ("<UDIM>", "<UVTILE>", "<UVTILE0>", "<UVTILE1>")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _is_inside(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
        return True
    except ValueError:
        return False


def _decode_mdl_string(value: str) -> str:
    """Decode the two escapes relevant to local MDL asset paths."""

    return value.replace(r"\"", '"').replace(r"\\", "\\")


def _strip_mdl_comments(text: str) -> str:
    """Remove MDL comments while preserving string literals and newlines."""

    result: list[str] = []
    index = 0
    state = "normal"
    while index < len(text):
        char = text[index]
        following = text[index + 1] if index + 1 < len(text) else ""
        if state == "normal":
            if char == '"':
                state = "string"
                result.append(char)
                index += 1
            elif char == "/" and following == "/":
                state = "line_comment"
                result.extend((" ", " "))
                index += 2
            elif char == "/" and following == "*":
                state = "block_comment"
                result.extend((" ", " "))
                index += 2
            else:
                result.append(char)
                index += 1
        elif state == "string":
            result.append(char)
            index += 1
            if char == "\\" and index < len(text):
                result.append(text[index])
                index += 1
            elif char == '"':
                state = "normal"
        elif state == "line_comment":
            if char in "\r\n":
                result.append(char)
                state = "normal"
            else:
                result.append(" ")
            index += 1
        else:  # block_comment
            if char == "*" and following == "/":
                result.extend((" ", " "))
                index += 2
                state = "normal"
            else:
                result.append(char if char in "\r\n" else " ")
                index += 1
    return "".join(result)


def _scan_mdl_document(text: str) -> tuple[list[str], list[str]]:
    uncommented = _strip_mdl_comments(text)
    textures = sorted(
        {_decode_mdl_string(match) for match in _TEXTURE_2D_RE.findall(uncommented)}
    )
    thumbnails = sorted(
        {_decode_mdl_string(match) for match in _THUMBNAIL_RE.findall(uncommented)}
    )
    return textures, thumbnails


def _local_asset_candidates(asset: str, owner: Path) -> list[Path]:
    """Resolve a literal local asset, including common tiled-texture tokens."""

    if "://" in asset:
        return []
    candidate = Path(asset).expanduser()
    if not candidate.is_absolute():
        candidate = owner.parent / candidate
    raw = str(candidate)
    if any(token in raw for token in _TILE_TOKENS):
        pattern = raw
        for token in _TILE_TOKENS:
            pattern = pattern.replace(token, "*")
        return sorted(Path(item).resolve(strict=False) for item in glob.glob(pattern))
    return [candidate.resolve(strict=False)]


@dataclass
class _Check:
    label: str
    failures: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def status(self) -> str:
        return "FAIL" if self.failures else "PASS"


class _Audit:
    def __init__(self) -> None:
        self.checks = {
            check_id: _Check(label) for check_id, label in CHECK_LABELS.items()
        }
        self.warnings: list[dict[str, Any]] = []

    def fail(
        self,
        check_id: str,
        message: str,
        *,
        context: dict[str, Any] | None = None,
    ) -> None:
        failure: dict[str, Any] = {"message": message}
        if context:
            failure["context"] = context
        self.checks[check_id].failures.append(failure)

    def require(
        self,
        check_id: str,
        condition: bool,
        message: str,
        *,
        context: dict[str, Any] | None = None,
    ) -> bool:
        if not condition:
            self.fail(check_id, message, context=context)
        return condition

    def metric(self, check_id: str, name: str, value: Any) -> None:
        self.checks[check_id].metrics[name] = value

    def warn(self, message: str, *, context: dict[str, Any] | None = None) -> None:
        warning: dict[str, Any] = {"message": message}
        if context:
            warning["context"] = context
        self.warnings.append(warning)

    def to_report(self, inputs: dict[str, Any]) -> dict[str, Any]:
        checks = [
            {
                "id": check_id,
                "label": check.label,
                "status": check.status,
                "metrics": check.metrics,
                "failures": check.failures,
            }
            for check_id, check in self.checks.items()
        ]
        failure_count = sum(len(check.failures) for check in self.checks.values())
        passed = sum(check.status == "PASS" for check in self.checks.values())
        overall_pass = failure_count == 0
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "PASS" if overall_pass else "FAIL",
            "overall_pass": overall_pass,
            "inputs": inputs,
            "summary": {
                "check_count": len(checks),
                "passed_check_count": passed,
                "failed_check_count": len(checks) - passed,
                "failure_count": failure_count,
                "warning_count": len(self.warnings),
            },
            "checks": checks,
            "warnings": self.warnings,
        }


def _attribute_signature(attribute: Any) -> dict[str, Any]:
    samples = list(attribute.GetTimeSamples())
    return {
        "type": str(attribute.GetTypeName()),
        "default": repr(attribute.Get()),
        "time_samples": samples,
        "sample_values": [repr(attribute.Get(time)) for time in samples],
        "connections": [path.pathString for path in attribute.GetConnections()],
        "metadata": {
            key: repr(value)
            for key, value in sorted(attribute.GetAllMetadata().items())
        },
    }


def _relationship_signature(relationship: Any) -> dict[str, Any]:
    return {
        "targets": [path.pathString for path in relationship.GetTargets()],
        "metadata": {
            key: repr(value)
            for key, value in sorted(relationship.GetAllMetadata().items())
        },
    }


def _geometry_signature(prim: Any, name: str) -> dict[str, Any]:
    return _attribute_signature(prim.GetAttribute(name))


def _matrix_values(matrix: Any) -> tuple[float, ...]:
    return tuple(float(matrix[row][column]) for row in range(4) for column in range(4))


def _physics_state(stage: Any) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for prim in stage.Traverse():
        attributes = {
            attribute.GetName(): _attribute_signature(attribute)
            for attribute in prim.GetAttributes()
            if attribute.GetName().startswith(("physics:", "physx"))
        }
        relationships = {
            relationship.GetName(): _relationship_signature(relationship)
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


def _material_binding_path(prim: Any, purpose: str) -> str | None:
    from pxr import UsdShade

    material, _relationship = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial(
        materialPurpose=purpose
    )
    return material.GetPath().pathString if material else None


def _direct_visual_binding_targets(prim: Any) -> list[str]:
    """Return the parent Prim's direct all-purpose visual binding targets."""

    relationship = prim.GetRelationship("material:binding")
    if not relationship or not relationship.IsValid():
        return []
    return [target.pathString for target in relationship.GetTargets()]


def _registry_parts(
    registry: dict[str, Any], audit: _Audit
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    raw_parts = registry.get("parts")
    if not isinstance(raw_parts, list):
        audit.fail("registry", "Registry must contain a parts list")
        return {}, {}
    by_id: dict[str, dict[str, Any]] = {}
    by_path: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(raw_parts):
        if not isinstance(raw, dict):
            audit.fail(
                "registry",
                "Registry part must be an object",
                context={"index": index},
            )
            continue
        part_id = raw.get("part_id")
        prim_path = raw.get("prim_path")
        if not isinstance(part_id, str) or not isinstance(prim_path, str):
            audit.fail(
                "registry",
                "Registry part requires string part_id and prim_path",
                context={"index": index},
            )
            continue
        if part_id in by_id:
            audit.fail(
                "registry", "Duplicate registry part_id", context={"part_id": part_id}
            )
            continue
        if prim_path in by_path:
            audit.fail(
                "registry",
                "Duplicate registry Mesh path",
                context={"prim_path": prim_path},
            )
            continue
        by_id[part_id] = raw
        by_path[prim_path] = raw
    audit.metric("registry", "registry_part_count", len(by_id))
    return by_id, by_path


def _collect_mapping(bundle_root: Path, audit: _Audit) -> dict[str, Path]:
    mapping_path = bundle_root / ".collect.mapping.json"
    if not mapping_path.is_file():
        audit.warn(
            "Collect mapping is absent; MDL identity will use the strict filename fallback",
            context={"expected": str(mapping_path)},
        )
        audit.metric("usd_dependencies", "collect_mapping_present", False)
        return {}
    audit.metric("usd_dependencies", "collect_mapping_present", True)
    document = _load_json_object(mapping_path)
    records = document.get("file_records")
    if not isinstance(records, list):
        audit.fail("usd_dependencies", "Collect mapping has no file_records list")
        return {}
    result: dict[str, Path] = {}
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            audit.fail(
                "usd_dependencies",
                "Collect mapping record must be an object",
                context={"index": index},
            )
            continue
        source = record.get("source_url")
        target = record.get("target_url")
        if not isinstance(source, str) or not isinstance(target, str):
            audit.fail(
                "usd_dependencies",
                "Collect mapping record requires source_url and target_url",
                context={"index": index},
            )
            continue
        if "://" in target:
            audit.fail(
                "usd_dependencies",
                "Collect target is not a local bundle path",
                context={"target_url": target},
            )
            continue
        target_path = Path(target).expanduser()
        if not target_path.is_absolute():
            target_path = bundle_root / target_path
        target_path = target_path.resolve(strict=False)
        if not _is_inside(target_path, bundle_root):
            audit.fail(
                "usd_dependencies",
                "Collect mapping target escapes the bundle",
                context={"target_url": target, "resolved": str(target_path)},
            )
        if not target_path.is_file():
            audit.fail(
                "usd_dependencies",
                "Collect mapping target is missing",
                context={"target_url": target, "resolved": str(target_path)},
            )
        source_key = source
        if "://" not in source:
            source_key = str(Path(source).expanduser().resolve(strict=False))
        if source_key in result and result[source_key] != target_path:
            audit.fail(
                "usd_dependencies",
                "Collect mapping has conflicting targets",
                context={"source_url": source},
            )
        result[source_key] = target_path
    audit.metric("usd_dependencies", "collect_mapping_record_count", len(records))
    audit.metric("usd_dependencies", "valid_collect_mapping_source_count", len(result))
    return result


def _report_records(
    report: dict[str, Any],
    registry_by_id: dict[str, dict[str, Any]],
    audit: _Audit,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    raw_applied = report.get("applied")
    if not isinstance(raw_applied, list):
        audit.fail("inputs", "Apply report must contain an applied list")
        return {}, {}
    applied_by_path: dict[str, dict[str, Any]] = {}
    materials: dict[str, dict[str, Any]] = {}

    def add_material(record: dict[str, Any], label: str) -> None:
        required = ("material_prim_path", "mdl_path", "subidentifier", "parameters")
        if any(key not in record for key in required):
            audit.fail(
                "mdl_materials",
                "Apply report material record is incomplete",
                context={"record": label},
            )
            return
        material_path = record.get("material_prim_path")
        if not isinstance(material_path, str):
            audit.fail(
                "mdl_materials",
                "material_prim_path must be a string",
                context={"record": label},
            )
            return
        canonical = {
            "material_id": record.get("material_id"),
            "mdl_path": record.get("mdl_path"),
            "subidentifier": record.get("subidentifier"),
            "parameters": record.get("parameters"),
        }
        existing = materials.get(material_path)
        if existing is not None and existing["canonical"] != canonical:
            audit.fail(
                "mdl_materials",
                "One material prim has conflicting apply-report definitions",
                context={"material_prim_path": material_path, "record": label},
            )
            return
        materials[material_path] = {
            "canonical": canonical,
            "record": record,
            "labels": (existing or {}).get("labels", []) + [label],
        }

    seen_parts: set[str] = set()
    preserved_parent_count = 0
    source_visual_preserve_count = 0
    source_visual_preserved_subset_count = 0
    for index, record in enumerate(raw_applied):
        if not isinstance(record, dict):
            audit.fail(
                "inputs",
                "Apply report entry must be an object",
                context={"index": index},
            )
            continue
        part_id = record.get("part_id")
        prim_path = record.get("prim_path")
        if not isinstance(part_id, str) or not isinstance(prim_path, str):
            audit.fail(
                "inputs",
                "Apply report entry needs string part_id and prim_path",
                context={"index": index},
            )
            continue
        if part_id in seen_parts or prim_path in applied_by_path:
            audit.fail(
                "inputs",
                "Apply report has a duplicate applied part or Mesh path",
                context={"part_id": part_id, "prim_path": prim_path},
            )
            continue
        seen_parts.add(part_id)
        parent_binding_preserved = record.get("parent_binding_preserved", False)
        if type(parent_binding_preserved) is not bool:
            audit.fail(
                "inputs",
                "parent_binding_preserved must be a boolean",
                context={"part_id": part_id},
            )
            parent_binding_preserved = False
        source_visual_preserved = record.get("source_visual_preserved", False)
        if type(source_visual_preserved) is not bool:
            audit.fail(
                "inputs",
                "source_visual_preserved must be a boolean",
                context={"part_id": part_id},
            )
            source_visual_preserved = False
        if source_visual_preserved and not parent_binding_preserved:
            audit.fail(
                "inputs",
                "source_visual_preserved requires parent_binding_preserved",
                context={"part_id": part_id},
            )
        registered = registry_by_id.get(part_id)
        if registered is None or registered.get("prim_path") != prim_path:
            audit.fail(
                "inputs",
                "Apply report part does not match the registry",
                context={"part_id": part_id, "prim_path": prim_path},
            )
        applied_by_path[prim_path] = record
        if parent_binding_preserved:
            preserved_parent_count += 1
            if "source_visual_material_prim_path" not in record or not isinstance(
                record.get("source_visual_material_prim_path"), (str, type(None))
            ):
                audit.fail(
                    "inputs",
                    "Subset-only apply report must record the source parent binding",
                    context={"part_id": part_id},
                )
            if record.get("parent_binding_relationship_authored") is not False:
                audit.fail(
                    "inputs",
                    "Subset-only apply report must attest that no parent binding "
                    "relationship was authored",
                    context={"part_id": part_id},
                )
            unexpected_parent_material_fields = sorted(
                field
                for field in (
                    "material_id",
                    "mdl_path",
                    "subidentifier",
                    "material_prim_path",
                    "parameters",
                )
                if field in record
            )
            if unexpected_parent_material_fields:
                audit.fail(
                    "inputs",
                    "Subset-only apply report ambiguously contains parent material fields",
                    context={
                        "part_id": part_id,
                        "fields": unexpected_parent_material_fields,
                    },
                )
            if source_visual_preserved:
                source_visual_preserve_count += 1
                source_path = record.get("source_visual_material_prim_path")
                registry_source_path = (
                    registered.get("existing_visual_material")
                    if isinstance(registered, dict)
                    else None
                )
                audit.require(
                    "inputs",
                    isinstance(source_path, str)
                    and source_path == registry_source_path,
                    "Source-preserve binding differs from the registry",
                    context={
                        "part_id": part_id,
                        "reported": source_path,
                        "registry": registry_source_path,
                    },
                )
                if isinstance(source_path, str):
                    expected_digest = source_visual_binding_sha256(
                        part_id=part_id,
                        prim_path=prim_path,
                        material_prim_path=source_path,
                    )
                    audit.require(
                        "inputs",
                        record.get("source_visual_material_binding_sha256")
                        == expected_digest,
                        "Source-preserve binding SHA-256 is invalid",
                        context={"part_id": part_id},
                    )
        else:
            add_material(record, part_id)
        subsets = record.get("face_subsets", [])
        if not isinstance(subsets, list):
            audit.fail(
                "geom_subsets",
                "face_subsets must be a list",
                context={"part_id": part_id},
            )
            continue
        for subset_index, subset in enumerate(subsets):
            if not isinstance(subset, dict):
                audit.fail(
                    "geom_subsets",
                    "Face subset report entry must be an object",
                    context={"part_id": part_id, "index": subset_index},
                )
                continue
            add_material(subset, f"{part_id}.face_subsets[{subset_index}]")
        source_subsets = record.get("source_visual_subset_bindings", [])
        if not isinstance(source_subsets, list):
            audit.fail(
                "geom_subsets",
                "source_visual_subset_bindings must be a list",
                context={"part_id": part_id},
            )
            source_subsets = []
        for subset_index, subset in enumerate(source_subsets):
            if (
                not isinstance(subset, dict)
                or not isinstance(subset.get("subset_prim_path"), str)
                or not isinstance(
                    subset.get("source_visual_material_prim_path"), str
                )
                or not isinstance(subset.get("face_indices"), list)
            ):
                audit.fail(
                    "geom_subsets",
                    "Source-preserve subset contract is invalid",
                    context={"part_id": part_id, "index": subset_index},
                )
        source_visual_preserved_subset_count += len(source_subsets)
        if parent_binding_preserved and not subsets and not source_visual_preserved:
            audit.fail(
                "geom_subsets",
                "Preserved parent binding requires at least one face subset",
                context={"part_id": part_id},
            )

    audit.require(
        "inputs",
        report.get("applied_count") == len(raw_applied),
        "Apply report applied_count does not match applied records",
        context={
            "declared": report.get("applied_count"),
            "actual": len(raw_applied),
        },
    )
    audit.require(
        "inputs",
        report.get("parent_binding_preserved_count", 0) == preserved_parent_count,
        "Apply report parent_binding_preserved_count does not match records",
        context={
            "declared": report.get("parent_binding_preserved_count", 0),
            "actual": preserved_parent_count,
        },
    )
    audit.require(
        "inputs",
        report.get("source_visual_preserve_count", 0)
        == source_visual_preserve_count,
        "Apply report source_visual_preserve_count does not match records",
        context={
            "declared": report.get("source_visual_preserve_count", 0),
            "actual": source_visual_preserve_count,
        },
    )
    audit.require(
        "inputs",
        report.get("source_visual_preserved_subset_count", 0)
        == source_visual_preserved_subset_count,
        "Apply report source-preserved subset count does not match records",
        context={
            "declared": report.get("source_visual_preserved_subset_count", 0),
            "actual": source_visual_preserved_subset_count,
        },
    )
    declared_subset_count = report.get("face_subset_count")
    actual_subset_count = sum(
        len(record.get("face_subsets", []))
        for record in raw_applied
        if isinstance(record, dict) and isinstance(record.get("face_subsets", []), list)
    )
    audit.require(
        "inputs",
        declared_subset_count == actual_subset_count,
        "Apply report face_subset_count does not match subset records",
        context={"declared": declared_subset_count, "actual": actual_subset_count},
    )
    audit.metric(
        "visual_bindings",
        "expected_applied_binding_count",
        len(applied_by_path) - preserved_parent_count,
    )
    audit.metric(
        "visual_bindings",
        "expected_preserved_parent_binding_count",
        preserved_parent_count,
    )
    audit.metric(
        "visual_bindings",
        "expected_source_visual_preserve_count",
        source_visual_preserve_count,
    )
    audit.metric("mdl_materials", "expected_material_prim_count", len(materials))
    audit.metric("geom_subsets", "expected_face_subset_count", actual_subset_count)
    return applied_by_path, materials


def _expected_mdl_target(
    original: str,
    actual: Path,
    collect_mapping: dict[str, Path],
    audit: _Audit,
    material_path: str,
) -> Path | None:
    original_key = original
    if "://" not in original:
        original_key = str(Path(original).expanduser().resolve(strict=False))
    if collect_mapping:
        target = collect_mapping.get(original_key)
        if target is None:
            audit.fail(
                "mdl_materials",
                "Original MDL source is absent from the collect mapping",
                context={"material_prim_path": material_path, "mdl_path": original},
            )
        return target
    if Path(original).name != actual.name:
        audit.fail(
            "mdl_materials",
            "Collected MDL filename does not match the apply report",
            context={
                "material_prim_path": material_path,
                "expected_filename": Path(original).name,
                "actual": str(actual),
            },
        )
        return None
    return actual


def _parameter_matches(actual: Any, expected: Any) -> bool:
    if type(expected) is bool:
        return type(actual) is bool and actual is expected
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        return (
            actual is not None
            and not isinstance(actual, bool)
            and math.isclose(float(actual), float(expected), rel_tol=1e-6, abs_tol=1e-7)
        )
    if isinstance(expected, list) and len(expected) == 3 and actual is not None:
        try:
            return all(
                math.isclose(float(found), float(wanted), rel_tol=1e-6, abs_tol=1e-7)
                for found, wanted in zip(actual, expected, strict=True)
            )
        except (TypeError, ValueError):
            return False
    return repr(actual) == repr(expected)


def _verify_materials(
    stage: Any,
    expected_materials: dict[str, dict[str, Any]],
    collect_mapping: dict[str, Path],
    bundle_root: Path,
    audit: _Audit,
) -> None:
    from pxr import Sdf, UsdShade

    verified = 0
    for material_path, definition in sorted(expected_materials.items()):
        record = definition["record"]
        prim = stage.GetPrimAtPath(material_path)
        if not prim or not prim.IsValid() or not prim.IsA(UsdShade.Material):
            audit.fail(
                "mdl_materials",
                "Expected material prim is missing or has the wrong type",
                context={"material_prim_path": material_path},
            )
            continue
        material = UsdShade.Material(prim)
        shader, _source_name, _source_type = material.ComputeSurfaceSource("mdl")
        if not shader or not shader.GetPrim().IsValid():
            audit.fail(
                "mdl_materials",
                "Material has no resolvable MDL surface source",
                context={"material_prim_path": material_path},
            )
            continue
        source_asset = shader.GetSourceAsset("mdl")
        if source_asset is None or not source_asset.path:
            audit.fail(
                "mdl_materials",
                "MDL shader has no source asset",
                context={"material_prim_path": material_path},
            )
            continue
        resolved_asset = Path(source_asset.resolvedPath).resolve(strict=False)
        audit.require(
            "mdl_materials",
            bool(source_asset.resolvedPath) and resolved_asset.is_file(),
            "MDL source asset does not resolve to a file",
            context={
                "material_prim_path": material_path,
                "authored_path": source_asset.path,
                "resolved_path": source_asset.resolvedPath,
            },
        )
        audit.require(
            "mdl_materials",
            _is_inside(resolved_asset, bundle_root),
            "MDL source asset escapes the collected bundle",
            context={"material_prim_path": material_path, "path": str(resolved_asset)},
        )
        original_mdl = record.get("mdl_path")
        if not isinstance(original_mdl, str):
            audit.fail(
                "mdl_materials",
                "Apply report mdl_path must be a string",
                context={"material_prim_path": material_path},
            )
        else:
            expected_target = _expected_mdl_target(
                original_mdl,
                resolved_asset,
                collect_mapping,
                audit,
                material_path,
            )
            if expected_target is not None:
                audit.require(
                    "mdl_materials",
                    resolved_asset == expected_target.resolve(strict=False),
                    "Collected MDL source differs from the collect mapping",
                    context={
                        "material_prim_path": material_path,
                        "expected": str(expected_target),
                        "actual": str(resolved_asset),
                    },
                )
        audit.require(
            "mdl_materials",
            shader.GetSourceAssetSubIdentifier("mdl") == record.get("subidentifier"),
            "MDL source subidentifier differs from the apply report",
            context={
                "material_prim_path": material_path,
                "expected": record.get("subidentifier"),
                "actual": shader.GetSourceAssetSubIdentifier("mdl"),
            },
        )
        parameters = record.get("parameters")
        if not isinstance(parameters, dict):
            audit.fail(
                "mdl_materials",
                "Apply report parameters must be an object",
                context={"material_prim_path": material_path},
            )
            parameters = {}
        actual_inputs = {
            shader_input.GetBaseName(): shader_input
            for shader_input in shader.GetInputs()
        }
        audit.require(
            "mdl_materials",
            set(actual_inputs) == set(parameters),
            "Authored MDL input names differ from the apply report",
            context={
                "material_prim_path": material_path,
                "expected": sorted(parameters),
                "actual": sorted(actual_inputs),
            },
        )
        for name, expected in sorted(parameters.items()):
            shader_input = actual_inputs.get(name)
            if shader_input is None:
                continue
            expected_type = None
            if type(expected) is bool:
                expected_type = Sdf.ValueTypeNames.Bool
            elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
                expected_type = Sdf.ValueTypeNames.Float
            elif isinstance(expected, list) and len(expected) == 3:
                expected_type = Sdf.ValueTypeNames.Color3f
            if expected_type is not None:
                audit.require(
                    "mdl_materials",
                    shader_input.GetTypeName() == expected_type,
                    "MDL parameter USD type differs from the apply report value type",
                    context={
                        "material_prim_path": material_path,
                        "parameter": name,
                        "expected": str(expected_type),
                        "actual": str(shader_input.GetTypeName()),
                    },
                )
            actual = shader_input.Get()
            audit.require(
                "mdl_materials",
                _parameter_matches(actual, expected),
                "MDL parameter value differs from the apply report",
                context={
                    "material_prim_path": material_path,
                    "parameter": name,
                    "expected": expected,
                    "actual": repr(actual),
                },
            )
        verified += 1
    audit.metric("mdl_materials", "verified_material_prim_count", verified)


def _verify_subsets(
    stage: Any,
    applied_by_path: dict[str, dict[str, Any]],
    registered_mesh_paths: Iterable[str],
    audit: _Audit,
) -> None:
    from pxr import UsdGeom, UsdShade

    expected_by_mesh: dict[str, dict[str, dict[str, Any]]] = {}
    for mesh_path in registered_mesh_paths:
        record = applied_by_path.get(mesh_path, {})
        subsets: dict[str, dict[str, Any]] = {}
        for subset in record.get("face_subsets", []):
            if not isinstance(subset, dict):
                continue
            subset_path = subset.get("subset_prim_path")
            if not isinstance(subset_path, str):
                audit.fail(
                    "geom_subsets",
                    "Subset report entry has no string subset_prim_path",
                    context={"mesh_path": mesh_path},
                )
                continue
            if subset_path in subsets:
                audit.fail(
                    "geom_subsets",
                    "Apply report has a duplicate subset path",
                    context={"subset_prim_path": subset_path},
                )
            subsets[subset_path] = subset
        for subset in record.get("source_visual_subset_bindings", []):
            if not isinstance(subset, dict):
                continue
            subset_path = subset.get("subset_prim_path")
            if not isinstance(subset_path, str):
                audit.fail(
                    "geom_subsets",
                    "Source-preserve subset has no string subset_prim_path",
                    context={"mesh_path": mesh_path},
                )
                continue
            if subset_path in subsets:
                audit.fail(
                    "geom_subsets",
                    "Apply report has a duplicate authored/preserved subset path",
                    context={"subset_prim_path": subset_path},
                )
            subsets[subset_path] = {
                **subset,
                "_source_visual_preserved": True,
            }
        expected_by_mesh[mesh_path] = subsets

    actual_total = 0
    verified_total = 0
    for mesh_path, expected in sorted(expected_by_mesh.items()):
        mesh_prim = stage.GetPrimAtPath(mesh_path)
        if not mesh_prim or not mesh_prim.IsValid():
            continue
        mesh = UsdGeom.Mesh(mesh_prim)
        actual = [
            UsdGeom.Subset(child)
            for child in mesh_prim.GetChildren()
            if child.IsA(UsdGeom.Subset)
            and UsdGeom.Subset(child).GetFamilyNameAttr().Get()
            == UsdShade.Tokens.materialBind
        ]
        actual_by_path = {subset.GetPath().pathString: subset for subset in actual}
        actual_total += len(actual_by_path)
        audit.require(
            "geom_subsets",
            set(actual_by_path) == set(expected),
            "Collected materialBind face subsets differ from the apply report",
            context={
                "mesh_path": mesh_path,
                "expected": sorted(expected),
                "actual": sorted(actual_by_path),
            },
        )
        if expected:
            family_type = UsdGeom.Subset.GetFamilyType(
                mesh, UsdShade.Tokens.materialBind
            )
            audit.require(
                "geom_subsets",
                family_type == UsdGeom.Tokens.nonOverlapping,
                "materialBind subset family is not nonOverlapping",
                context={"mesh_path": mesh_path, "actual": str(family_type)},
            )
            valid, reason = UsdGeom.Subset.ValidateFamily(
                mesh, UsdGeom.Tokens.face, UsdShade.Tokens.materialBind
            )
            audit.require(
                "geom_subsets",
                bool(valid),
                "materialBind subset family is invalid",
                context={"mesh_path": mesh_path, "reason": str(reason)},
            )
        for subset_path, record in sorted(expected.items()):
            subset = actual_by_path.get(subset_path)
            if subset is None:
                continue
            audit.require(
                "geom_subsets",
                subset.GetElementTypeAttr().Get() == UsdGeom.Tokens.face,
                "GeomSubset element type is not face",
                context={"subset_prim_path": subset_path},
            )
            audit.require(
                "geom_subsets",
                subset.GetFamilyNameAttr().Get() == UsdShade.Tokens.materialBind,
                "GeomSubset family is not materialBind",
                context={"subset_prim_path": subset_path},
            )
            expected_indices = record.get("face_indices")
            actual_indices = list(subset.GetIndicesAttr().Get() or [])
            audit.require(
                "geom_subsets",
                isinstance(expected_indices, list)
                and actual_indices == expected_indices,
                "GeomSubset face indices differ from the apply report",
                context={
                    "subset_prim_path": subset_path,
                    "expected": expected_indices,
                    "actual": actual_indices,
                },
            )
            binding = _material_binding_path(
                subset.GetPrim(), UsdShade.Tokens.allPurpose
            )
            expected_binding = (
                record.get("source_visual_material_prim_path")
                if record.get("_source_visual_preserved") is True
                else record.get("material_prim_path")
            )
            audit.require(
                "geom_subsets",
                binding == expected_binding,
                "GeomSubset visual binding differs from the apply report",
                context={
                    "subset_prim_path": subset_path,
                    "expected": expected_binding,
                    "actual": binding,
                },
            )
            verified_total += 1
    audit.metric("geom_subsets", "actual_material_face_subset_count", actual_total)
    audit.metric("geom_subsets", "verified_face_subset_count", verified_total)


def _verify_usd_dependencies(
    collected_root: Path, bundle_root: Path, audit: _Audit
) -> None:
    from pxr import UsdUtils

    layers, assets, unresolved = UsdUtils.ComputeAllDependencies(str(collected_root))
    audit.metric("usd_dependencies", "usd_layer_count", len(layers))
    audit.metric("usd_dependencies", "usd_asset_dependency_count", len(assets))
    audit.metric("usd_dependencies", "unresolved_dependency_count", len(unresolved))
    if unresolved:
        audit.fail(
            "usd_dependencies",
            "USD dependencies are unresolved",
            context={"unresolved": sorted(str(item) for item in unresolved)},
        )

    resolved_paths: list[Path] = []
    for layer in layers:
        raw = layer.realPath or layer.identifier
        if not raw or "://" in raw:
            audit.fail(
                "usd_dependencies",
                "Collected USD layer is not a local path",
                context={"identifier": str(raw)},
            )
            continue
        path = Path(raw).expanduser().resolve(strict=False)
        resolved_paths.append(path)
    for raw in assets:
        value = str(raw)
        if not value or "://" in value:
            audit.fail(
                "usd_dependencies",
                "Collected USD asset dependency is not a local path",
                context={"asset": value},
            )
            continue
        resolved_paths.append(Path(value).expanduser().resolve(strict=False))

    for path in sorted(set(resolved_paths)):
        audit.require(
            "usd_dependencies",
            path.is_file(),
            "Collected USD dependency is missing",
            context={"path": str(path)},
        )
        audit.require(
            "usd_dependencies",
            _is_inside(path, bundle_root),
            "Collected USD dependency escapes the bundle",
            context={"path": str(path), "bundle_root": str(bundle_root)},
        )
    audit.metric(
        "usd_dependencies", "resolved_dependency_path_count", len(set(resolved_paths))
    )


def _verify_mdl_textures(bundle_root: Path, audit: _Audit) -> None:
    mdl_files = sorted(bundle_root.rglob("*.mdl"))
    runtime_records: list[dict[str, str]] = []
    thumbnail_records: list[dict[str, str]] = []
    missing_optional = 0
    for mdl_path in mdl_files:
        if not _is_inside(mdl_path, bundle_root):
            audit.fail(
                "mdl_textures",
                "Collected MDL file escapes the bundle",
                context={"path": str(mdl_path)},
            )
            continue
        text = mdl_path.read_text(encoding="utf-8", errors="replace")
        textures, thumbnails = _scan_mdl_document(text)
        for authored in textures:
            candidates = _local_asset_candidates(authored, mdl_path)
            record = {"mdl": str(mdl_path), "authored_path": authored}
            runtime_records.append(record)
            if not candidates:
                audit.fail(
                    "mdl_textures",
                    "MDL runtime texture is not a local bundle path",
                    context=record,
                )
                continue
            if not any(candidate.is_file() for candidate in candidates):
                audit.fail(
                    "mdl_textures",
                    "MDL runtime texture target is missing",
                    context={**record, "resolved": [str(item) for item in candidates]},
                )
            outside = [
                str(item) for item in candidates if not _is_inside(item, bundle_root)
            ]
            if outside:
                audit.fail(
                    "mdl_textures",
                    "MDL runtime texture target escapes the bundle",
                    context={**record, "resolved_outside": outside},
                )
        for authored in thumbnails:
            thumbnail_records.append({"mdl": str(mdl_path), "authored_path": authored})
            candidates = _local_asset_candidates(authored, mdl_path)
            if not candidates or not any(
                candidate.is_file() for candidate in candidates
            ):
                missing_optional += 1

    audit.metric("mdl_textures", "mdl_file_count", len(mdl_files))
    audit.metric(
        "mdl_textures", "runtime_texture_reference_count", len(runtime_records)
    )
    audit.metric(
        "mdl_textures", "thumbnail_metadata_reference_count", len(thumbnail_records)
    )
    audit.metric("mdl_textures", "missing_optional_thumbnail_count", missing_optional)
    if missing_optional:
        audit.warn(
            "Optional MDL thumbnail metadata targets were not collected",
            context={"missing_count": missing_optional},
        )


def validate_final_bundle(
    *,
    source_usd: str | Path,
    collected_root_usd: str | Path,
    registry_path: str | Path,
    apply_report_path: str | Path,
    bundle_root: str | Path,
) -> dict[str, Any]:
    from pxr import Usd, UsdGeom, UsdShade

    source_path = Path(source_usd).expanduser().resolve(strict=True)
    collected_path = Path(collected_root_usd).expanduser().resolve(strict=True)
    registry_file = Path(registry_path).expanduser().resolve(strict=True)
    apply_file = Path(apply_report_path).expanduser().resolve(strict=True)
    bundle_path = Path(bundle_root).expanduser().resolve(strict=True)
    if not bundle_path.is_dir():
        raise ValueError(f"Bundle root is not a directory: {bundle_path}")

    audit = _Audit()
    audit.require(
        "inputs",
        _is_inside(collected_path, bundle_path),
        "Collected root USD is outside the declared bundle root",
        context={
            "collected_root_usd": str(collected_path),
            "bundle_root": str(bundle_path),
        },
    )
    registry = _load_json_object(registry_file)
    apply_report = _load_json_object(apply_file)
    source_hash = _sha256_file(source_path)
    registry_hash = registry.get("asset_sha256")
    apply_source_hash = apply_report.get("source_sha256")
    audit.require(
        "inputs",
        registry_hash == source_hash,
        "Registry SHA-256 does not match the source USD",
        context={"expected": source_hash, "registry": registry_hash},
    )
    audit.require(
        "inputs",
        apply_source_hash == source_hash,
        "Apply report SHA-256 does not match the source USD",
        context={"expected": source_hash, "apply_report": apply_source_hash},
    )
    registry_asset = registry.get("asset_usd")
    if isinstance(registry_asset, str):
        audit.require(
            "inputs",
            Path(registry_asset).expanduser().resolve(strict=False) == source_path,
            "Registry asset path does not identify the source USD",
            context={"registry_asset": registry_asset, "source_usd": str(source_path)},
        )
    else:
        audit.fail("inputs", "Registry has no string asset_usd")
    report_source = apply_report.get("source_usd")
    if isinstance(report_source, str):
        audit.require(
            "inputs",
            Path(report_source).expanduser().resolve(strict=False) == source_path,
            "Apply report source path does not identify the source USD",
            context={"report_source": report_source, "source_usd": str(source_path)},
        )
    else:
        audit.fail("inputs", "Apply report has no string source_usd")

    source_stage = Usd.Stage.Open(str(source_path), load=Usd.Stage.LoadAll)
    collected_stage = Usd.Stage.Open(str(collected_path), load=Usd.Stage.LoadAll)
    if source_stage is None:
        raise RuntimeError(f"Unable to open source USD: {source_path}")
    if collected_stage is None:
        raise RuntimeError(f"Unable to open collected root USD: {collected_path}")

    registry_by_id, registry_by_path = _registry_parts(registry, audit)
    source_mesh_paths = {
        prim.GetPath().pathString
        for prim in source_stage.Traverse()
        if prim.IsA(UsdGeom.Mesh)
    }
    collected_mesh_paths = {
        prim.GetPath().pathString
        for prim in collected_stage.Traverse()
        if prim.IsA(UsdGeom.Mesh)
    }
    registered_paths = set(registry_by_path)
    audit.require(
        "registry",
        registered_paths == source_mesh_paths,
        "Registry does not exactly cover source Mesh paths",
        context={
            "unregistered_source_meshes": sorted(source_mesh_paths - registered_paths),
            "registry_paths_not_source_meshes": sorted(
                registered_paths - source_mesh_paths
            ),
        },
    )
    audit.require(
        "registry",
        collected_mesh_paths == source_mesh_paths,
        "Collected stage Mesh paths differ from the source USD",
        context={
            "missing_collected_meshes": sorted(
                source_mesh_paths - collected_mesh_paths
            ),
            "unexpected_collected_meshes": sorted(
                collected_mesh_paths - source_mesh_paths
            ),
        },
    )
    audit.require(
        "registry",
        registry.get("part_count") == len(registry_by_id),
        "Registry part_count does not match valid unique parts",
        context={"declared": registry.get("part_count"), "actual": len(registry_by_id)},
    )

    valid_mesh_paths: list[str] = []
    for path, part in sorted(registry_by_path.items()):
        source_prim = source_stage.GetPrimAtPath(path)
        collected_prim = collected_stage.GetPrimAtPath(path)
        if (
            not source_prim
            or not source_prim.IsValid()
            or not source_prim.IsA(UsdGeom.Mesh)
        ):
            audit.fail(
                "registry",
                "Registered source prim is not a Mesh",
                context={"path": path},
            )
            continue
        if (
            not collected_prim
            or not collected_prim.IsValid()
            or not collected_prim.IsA(UsdGeom.Mesh)
        ):
            audit.fail(
                "registry",
                "Registered Mesh is missing from collected stage",
                context={"path": path},
            )
            continue
        source_mesh = UsdGeom.Mesh(source_prim)
        point_count = len(source_mesh.GetPointsAttr().Get() or [])
        face_count = len(source_mesh.GetFaceVertexCountsAttr().Get() or [])
        audit.require(
            "registry",
            part.get("point_count") == point_count,
            "Registry point_count differs from source Mesh",
            context={
                "path": path,
                "declared": part.get("point_count"),
                "actual": point_count,
            },
        )
        audit.require(
            "registry",
            part.get("face_count") == face_count,
            "Registry face_count differs from source Mesh",
            context={
                "path": path,
                "declared": part.get("face_count"),
                "actual": face_count,
            },
        )
        valid_mesh_paths.append(path)
    audit.metric("registry", "source_mesh_count", len(source_mesh_paths))
    audit.metric("registry", "collected_mesh_count", len(collected_mesh_paths))
    audit.metric("registry", "collected_registered_mesh_count", len(valid_mesh_paths))

    changed_geometry: list[dict[str, str]] = []
    totals = {
        "point_count": 0,
        "face_count": 0,
        "face_vertex_index_count": 0,
        "hole_count": 0,
    }
    for path in valid_mesh_paths:
        source_prim = source_stage.GetPrimAtPath(path)
        collected_prim = collected_stage.GetPrimAtPath(path)
        for name in GEOMETRY_ATTRIBUTES:
            if _geometry_signature(source_prim, name) != _geometry_signature(
                collected_prim, name
            ):
                changed_geometry.append({"prim_path": path, "attribute": name})
        source_mesh = UsdGeom.Mesh(source_prim)
        totals["point_count"] += len(source_mesh.GetPointsAttr().Get() or [])
        totals["face_count"] += len(source_mesh.GetFaceVertexCountsAttr().Get() or [])
        totals["face_vertex_index_count"] += len(
            source_mesh.GetFaceVertexIndicesAttr().Get() or []
        )
        totals["hole_count"] += len(source_mesh.GetHoleIndicesAttr().Get() or [])
    if changed_geometry:
        audit.fail(
            "geometry",
            "Mesh geometry or topology differs from the source USD",
            context={"changed": changed_geometry},
        )
    audit.metric("geometry", "mesh_count", len(valid_mesh_paths))
    audit.metric(
        "geometry",
        "attribute_comparison_count",
        len(valid_mesh_paths) * len(GEOMETRY_ATTRIBUTES),
    )
    for name, value in totals.items():
        audit.metric("geometry", name, value)

    local_changes: list[dict[str, Any]] = []
    world_changes: list[dict[str, Any]] = []
    transform_comparisons = 0
    for path in valid_mesh_paths:
        source_prim = source_stage.GetPrimAtPath(path)
        collected_prim = collected_stage.GetPrimAtPath(path)
        source_xform = UsdGeom.Xformable(source_prim)
        collected_xform = UsdGeom.Xformable(collected_prim)
        source_times = list(source_xform.GetTimeSamples())
        collected_times = list(collected_xform.GetTimeSamples())
        if source_times != collected_times:
            local_changes.append(
                {
                    "prim_path": path,
                    "reason": "time_samples",
                    "source": source_times,
                    "collected": collected_times,
                }
            )
        if source_xform.GetResetXformStack() != collected_xform.GetResetXformStack():
            local_changes.append({"prim_path": path, "reason": "reset_xform_stack"})
        times: list[float | None] = [None] + sorted(
            set(source_times) | set(collected_times)
        )
        for sample in times:
            time_code = (
                Usd.TimeCode.Default() if sample is None else Usd.TimeCode(sample)
            )
            source_local = _matrix_values(
                source_xform.GetLocalTransformation(time_code)
            )
            collected_local = _matrix_values(
                collected_xform.GetLocalTransformation(time_code)
            )
            if source_local != collected_local:
                local_changes.append({"prim_path": path, "time": sample})
            source_cache = UsdGeom.XformCache(time_code)
            collected_cache = UsdGeom.XformCache(time_code)
            source_world = _matrix_values(
                source_cache.GetLocalToWorldTransform(source_prim)
            )
            collected_world = _matrix_values(
                collected_cache.GetLocalToWorldTransform(collected_prim)
            )
            if source_world != collected_world:
                world_changes.append({"prim_path": path, "time": sample})
            transform_comparisons += 1
    if local_changes:
        audit.fail(
            "transforms",
            "Local transforms differ from the source USD",
            context={"changed": local_changes},
        )
    if world_changes:
        audit.fail(
            "transforms",
            "World transforms differ from the source USD",
            context={"changed": world_changes},
        )
    audit.metric("transforms", "mesh_count", len(valid_mesh_paths))
    audit.metric("transforms", "time_evaluation_count", transform_comparisons)

    source_physics = _physics_state(source_stage)
    collected_physics = _physics_state(collected_stage)
    if source_physics != collected_physics:
        source_keys = set(source_physics)
        collected_keys = set(collected_physics)
        changed = sorted(
            path
            for path in source_keys & collected_keys
            if source_physics[path] != collected_physics[path]
        )
        audit.fail(
            "physics",
            "Physics/PhysX schemas, attributes, or relationships differ",
            context={
                "removed_paths": sorted(source_keys - collected_keys),
                "added_paths": sorted(collected_keys - source_keys),
                "changed_paths": changed,
            },
        )
    physics_binding_changes: list[dict[str, Any]] = []
    for path in valid_mesh_paths:
        source_binding = _material_binding_path(
            source_stage.GetPrimAtPath(path), "physics"
        )
        collected_binding = _material_binding_path(
            collected_stage.GetPrimAtPath(path), "physics"
        )
        if source_binding != collected_binding:
            physics_binding_changes.append(
                {
                    "prim_path": path,
                    "source": source_binding,
                    "collected": collected_binding,
                }
            )
    if physics_binding_changes:
        audit.fail(
            "physics",
            "Computed physics material bindings differ",
            context={"changed": physics_binding_changes},
        )
    audit.metric("physics", "physics_prim_count", len(source_physics))
    audit.metric(
        "physics",
        "physics_attribute_count",
        sum(len(item["attributes"]) for item in source_physics.values()),
    )
    audit.metric(
        "physics",
        "physics_relationship_count",
        sum(len(item["relationships"]) for item in source_physics.values()),
    )
    audit.metric(
        "physics",
        "physics_schema_instance_count",
        sum(len(item["schemas"]) for item in source_physics.values()),
    )
    audit.metric("physics", "physics_binding_count", len(valid_mesh_paths))

    applied_by_path, expected_materials = _report_records(
        apply_report, registry_by_id, audit
    )
    collected_mapping = _collect_mapping(bundle_path, audit)
    verified_applied = 0
    preserved = 0
    verified_subset_only_parents = 0
    for path in valid_mesh_paths:
        collected_prim = collected_stage.GetPrimAtPath(path)
        actual = _material_binding_path(collected_prim, UsdShade.Tokens.allPurpose)
        record = applied_by_path.get(path)
        if record is not None:
            if record.get("parent_binding_preserved", False):
                source_prim = source_stage.GetPrimAtPath(path)
                source_actual = _material_binding_path(
                    source_prim, UsdShade.Tokens.allPurpose
                )
                audit.require(
                    "visual_bindings",
                    record.get("source_visual_material_prim_path") == source_actual,
                    "Apply report source parent binding differs from source USD",
                    context={
                        "prim_path": path,
                        "reported": record.get("source_visual_material_prim_path"),
                        "source": source_actual,
                    },
                )
                audit.require(
                    "visual_bindings",
                    actual == source_actual,
                    "Subset-only assignment changed the parent visual binding",
                    context={
                        "prim_path": path,
                        "expected": source_actual,
                        "actual": actual,
                    },
                )
                source_direct = _direct_visual_binding_targets(source_prim)
                collected_direct = _direct_visual_binding_targets(collected_prim)
                audit.require(
                    "visual_bindings",
                    collected_direct == source_direct,
                    "Subset-only assignment changed the direct parent binding relationship",
                    context={
                        "prim_path": path,
                        "expected": source_direct,
                        "actual": collected_direct,
                    },
                )
                verified_subset_only_parents += 1
                preserved += 1
                continue
            expected = record.get("material_prim_path")
            audit.require(
                "visual_bindings",
                actual == expected,
                "Applied visual material binding differs from the apply report",
                context={"prim_path": path, "expected": expected, "actual": actual},
            )
            direct_targets = [
                target.pathString
                for target in collected_prim.GetRelationship(
                    "material:binding"
                ).GetTargets()
            ]
            audit.require(
                "visual_bindings",
                direct_targets == [expected],
                "Direct visual material relationship differs from the apply report",
                context={
                    "prim_path": path,
                    "expected": [expected],
                    "actual": direct_targets,
                },
            )
            verified_applied += 1
        else:
            source_actual = _material_binding_path(
                source_stage.GetPrimAtPath(path), UsdShade.Tokens.allPurpose
            )
            audit.require(
                "visual_bindings",
                actual == source_actual,
                "Unapplied visual material binding was not preserved",
                context={
                    "prim_path": path,
                    "expected": source_actual,
                    "actual": actual,
                },
            )
            preserved += 1
    audit.metric("visual_bindings", "verified_applied_binding_count", verified_applied)
    audit.metric("visual_bindings", "verified_preserved_binding_count", preserved)
    audit.metric(
        "visual_bindings",
        "verified_subset_only_parent_binding_count",
        verified_subset_only_parents,
    )

    _verify_materials(
        collected_stage, expected_materials, collected_mapping, bundle_path, audit
    )
    _verify_subsets(collected_stage, applied_by_path, valid_mesh_paths, audit)
    _verify_usd_dependencies(collected_path, bundle_path, audit)
    _verify_mdl_textures(bundle_path, audit)

    inputs = {
        "source_usd": str(source_path),
        "source_sha256": source_hash,
        "collected_root_usd": str(collected_path),
        "collected_root_sha256": _sha256_file(collected_path),
        "registry": str(registry_file),
        "registry_sha256": _sha256_file(registry_file),
        "apply_report": str(apply_file),
        "apply_report_sha256": _sha256_file(apply_file),
        "bundle_root": str(bundle_path),
    }
    return audit.to_report(inputs)


def _start_isaac_if_needed() -> Any:
    try:
        from pxr import Usd  # noqa: F401

        return None
    except ImportError:
        try:
            from isaacsim import SimulationApp
        except ImportError as exc:
            raise RuntimeError(
                "pxr is unavailable. Run this validator with Isaac Sim python.sh."
            ) from exc
        return SimulationApp({"headless": True})


# Public shared API.  The aliases preserve the thoroughly tested implementation
# while making the dependency direction explicit: entry-point validators import
# this contract instead of reaching into one another's private namespace.
Audit = _Audit
collect_mapping = _collect_mapping
is_inside = _is_inside
load_json_object = _load_json_object
local_asset_candidates = _local_asset_candidates
parameter_matches = _parameter_matches
report_records = _report_records
scan_mdl_document = _scan_mdl_document
start_isaac_if_needed = _start_isaac_if_needed
strip_mdl_comments = _strip_mdl_comments
verify_materials = _verify_materials
verify_mdl_textures = _verify_mdl_textures
verify_usd_dependencies = _verify_usd_dependencies


def _write_report(report: dict[str, Any], output: str | Path) -> Path:
    output_path = Path(output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(output_path)
    return output_path


__all__ = [
    "CHECK_LABELS",
    "SCHEMA_VERSION",
    "Audit",
    "collect_mapping",
    "is_inside",
    "load_json_object",
    "local_asset_candidates",
    "main",
    "parameter_matches",
    "parse_args",
    "report_records",
    "scan_mdl_document",
    "start_isaac_if_needed",
    "strip_mdl_comments",
    "validate_final_bundle",
    "verify_materials",
    "verify_mdl_textures",
    "verify_usd_dependencies",
]


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate an exact, self-contained collected Qwen material USD bundle"
    )
    parser.add_argument("--source-usd", required=True)
    parser.add_argument("--collected-root-usd", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--apply-report", required=True)
    parser.add_argument("--bundle-root", required=True)
    parser.add_argument("--output", required=True, help="Machine-readable JSON report")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    app = None
    try:
        app = _start_isaac_if_needed()
        report = validate_final_bundle(
            source_usd=args.source_usd,
            collected_root_usd=args.collected_root_usd,
            registry_path=args.registry,
            apply_report_path=args.apply_report,
            bundle_root=args.bundle_root,
        )
        output = _write_report(report, args.output)
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "overall_pass": report["overall_pass"],
                    "output": str(output),
                    "summary": report["summary"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return 0 if report["overall_pass"] else 1
    except Exception as exc:
        traceback.print_exc()
        fatal_report = {
            "schema_version": SCHEMA_VERSION,
            "status": "FAIL",
            "overall_pass": False,
            "inputs": {
                "source_usd": str(Path(args.source_usd).expanduser()),
                "collected_root_usd": str(Path(args.collected_root_usd).expanduser()),
                "registry": str(Path(args.registry).expanduser()),
                "apply_report": str(Path(args.apply_report).expanduser()),
                "bundle_root": str(Path(args.bundle_root).expanduser()),
            },
            "summary": {
                "check_count": len(CHECK_LABELS),
                "passed_check_count": 0,
                "failed_check_count": len(CHECK_LABELS),
                "failure_count": 1,
                "warning_count": 0,
            },
            "checks": [],
            "warnings": [],
            "fatal_error": {"type": type(exc).__name__, "message": str(exc)},
        }
        try:
            output = _write_report(fatal_report, args.output)
            print(
                json.dumps(
                    {"status": "FAIL", "overall_pass": False, "output": str(output)},
                    ensure_ascii=False,
                ),
                flush=True,
            )
        except Exception:
            traceback.print_exc()
        return 2
    finally:
        if app is not None:
            app.close()


if __name__ == "__main__":
    raise SystemExit(main())
