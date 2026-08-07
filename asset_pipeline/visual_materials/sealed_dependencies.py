"""Fail-closed verification for sealed NVIDIA MDL replay dependencies.

A bundled material project is only reproducible when the material catalog
still resolves to the exact MDL modules, runtime resources, and Isaac MDL
helper modules used by the accepted result.  This module intentionally does
not start Isaac Sim.  It validates a project-owned lock manifest against the
files that a later render/apply stage would consume.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any


LOCK_SCHEMA_VERSION = "qwen-sealed-material-dependency-lock/v1"
VERIFICATION_SCHEMA_VERSION = "qwen-sealed-material-dependency-verification/v1"

_EXPORT_MATERIAL_RE = re.compile(
    r"\bexport\s+material\s+([A-Za-z_][A-Za-z0-9_]*)\s*\("
)
_RUNTIME_ASSET_RE = re.compile(
    r'\btexture_(?:1d|2d|3d|cube|ptex)\s*\(\s*"((?:\\.|[^"\\])*)"',
    re.MULTILINE,
)
_TILED_ASSET_TOKENS = ("<UDIM>", "<UVTILE>", "<UVTILE0>", "<UVTILE1>")
_SEALED_EXACT_PARAMETER_POLICY = {
    "mode": "sealed-template-exact-parameters/v1",
    "template_hash_binds_parameters": True,
    "library_modules_immutable": True,
    "post_selection_parameter_mutation": False,
    "library_defaults_required": False,
}
_LIBRARY_DEFAULT_PARAMETER_POLICY = {
    "mode": "library-default-selected-mdl/v1",
    "template_hash_binds_parameters": True,
    "library_modules_immutable": True,
    "post_selection_parameter_mutation": False,
    "library_defaults_required": True,
}
_ALLOWED_HISTORICAL_PARAMETER_POLICIES = (
    _SEALED_EXACT_PARAMETER_POLICY,
    _LIBRARY_DEFAULT_PARAMETER_POLICY,
)


class SealedDependencyLockError(ValueError):
    """Raised when a sealed dependency contract cannot be verified."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SealedDependencyLockError(
            f"Unable to read {label}: {path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise SealedDependencyLockError(f"{label} must be a JSON object: {path}")
    return value


def _strip_mdl_comments(text: str) -> str:
    """Remove MDL comments while preserving strings and line positions."""

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
        else:
            if char == "*" and following == "/":
                result.extend((" ", " "))
                index += 2
                state = "normal"
            else:
                result.append(char if char in "\r\n" else " ")
                index += 1
    return "".join(result)


def _decode_mdl_string(value: str) -> str:
    return value.replace(r"\"", '"').replace(r"\\", "\\")


def _runtime_asset_literals(mdl_path: Path) -> set[str]:
    text = mdl_path.read_text(encoding="utf-8", errors="strict")
    uncommented = _strip_mdl_comments(text)
    return {
        _decode_mdl_string(value)
        for value in _RUNTIME_ASSET_RE.findall(uncommented)
    }


def _exported_materials(mdl_path: Path) -> set[str]:
    text = mdl_path.read_text(encoding="utf-8", errors="strict")
    return set(_EXPORT_MATERIAL_RE.findall(_strip_mdl_comments(text)))


def _relative_value(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise SealedDependencyLockError(f"{label} must be a non-empty string")
    if "://" in value:
        raise SealedDependencyLockError(f"{label} cannot be a remote URI: {value}")
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts:
        raise SealedDependencyLockError(
            f"{label} must remain inside its declared root: {value}"
        )
    normalized = pure.as_posix()
    if normalized in {"", "."}:
        raise SealedDependencyLockError(f"{label} is not a file path")
    return normalized


def _rooted_file(root: Path, relative: Any, label: str) -> Path:
    normalized = _relative_value(relative, label)
    root_resolved = root.expanduser().resolve(strict=True)
    try:
        path = (root_resolved / normalized).resolve(strict=True)
        path.relative_to(root_resolved)
    except (OSError, RuntimeError, ValueError) as exc:
        raise SealedDependencyLockError(
            f"{label} escapes its declared root or is missing: {normalized}"
        ) from exc
    if not path.is_file():
        raise SealedDependencyLockError(f"{label} is not a file: {path}")
    return path


def _require_sha256(record: Mapping[str, Any], path: Path, label: str) -> str:
    expected = record.get("sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise SealedDependencyLockError(f"{label}.sha256 is invalid")
    actual = _sha256_file(path)
    if actual != expected:
        raise SealedDependencyLockError(
            f"{label} content changed: {path} expected={expected} actual={actual}"
        )
    return actual


def _records(value: Any, label: str) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        raise SealedDependencyLockError(f"{label} must be an array")
    records: list[Mapping[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise SealedDependencyLockError(f"{label}[{index}] must be an object")
        records.append(item)
    return records


def _catalog_contract(
    *,
    catalog_path: Path,
    selected_records: list[Mapping[str, Any]],
    module_records: list[Mapping[str, Any]],
) -> tuple[dict[str, Mapping[str, Any]], dict[str, Mapping[str, Any]]]:
    catalog = _read_object(catalog_path, "sealed material catalog")
    raw_materials = _records(catalog.get("materials"), "catalog.materials")
    if catalog.get("material_count") != len(raw_materials):
        raise SealedDependencyLockError("catalog.material_count is inconsistent")

    catalog_by_id: dict[str, Mapping[str, Any]] = {}
    for index, item in enumerate(raw_materials):
        material_id = item.get("material_id")
        if (
            not isinstance(material_id, str)
            or not material_id
            or material_id in catalog_by_id
        ):
            raise SealedDependencyLockError(
                f"catalog.materials[{index}].material_id is invalid or duplicated"
            )
        catalog_by_id[material_id] = item

    selected_by_id: dict[str, Mapping[str, Any]] = {}
    for index, item in enumerate(selected_records):
        material_id = item.get("material_id")
        if (
            not isinstance(material_id, str)
            or not material_id
            or material_id in selected_by_id
        ):
            raise SealedDependencyLockError(
                "dependency_lock.selected_materials"
                f"[{index}].material_id is invalid or duplicated"
            )
        selected_by_id[material_id] = item
    if set(selected_by_id) != set(catalog_by_id):
        raise SealedDependencyLockError(
            "dependency lock selected materials do not exactly cover the catalog"
        )
    for material_id, locked in selected_by_id.items():
        catalog_item = catalog_by_id[material_id]
        expected_pair = (
            catalog_item.get("mdl_path"),
            catalog_item.get("sub_identifier"),
        )
        locked_pair = (locked.get("mdl_path"), locked.get("sub_identifier"))
        if locked_pair != expected_pair:
            raise SealedDependencyLockError(
                "dependency lock material path/subidentifier differs from catalog: "
                f"{material_id}"
            )

    module_by_path: dict[str, Mapping[str, Any]] = {}
    for index, item in enumerate(module_records):
        mdl_path = _relative_value(
            item.get("mdl_path"),
            f"dependency_lock.top_level_modules[{index}].mdl_path",
        )
        if mdl_path in module_by_path:
            raise SealedDependencyLockError(
                f"dependency lock duplicates top-level MDL module: {mdl_path}"
            )
        module_by_path[mdl_path] = item
    catalog_modules = {
        _relative_value(
            item.get("mdl_path"),
            f"catalog material {material_id}.mdl_path",
        )
        for material_id, item in catalog_by_id.items()
    }
    if set(module_by_path) != catalog_modules:
        raise SealedDependencyLockError(
            "dependency lock top-level modules do not exactly cover the catalog"
        )
    return selected_by_id, module_by_path


def verify_sealed_dependency_lock(
    *,
    lock_path: str | Path,
    expected_lock_sha256: str,
    catalog_path: str | Path,
    material_root: str | Path,
    isaac_root: str | Path,
    expected_asset_id: str | None = None,
) -> dict[str, Any]:
    """Verify every file needed by one sealed MDL replay.

    The returned object is safe to persist as audit evidence.  Verification is
    deliberately repeated immediately before material application so a file
    changed after project matching cannot enter the authored Look.
    """

    lock_file = Path(lock_path).expanduser().resolve(strict=True)
    if (
        not isinstance(expected_lock_sha256, str)
        or len(expected_lock_sha256) != 64
        or _sha256_file(lock_file) != expected_lock_sha256
    ):
        raise SealedDependencyLockError(
            "sealed dependency lock manifest hash changed"
        )
    lock = _read_object(lock_file, "sealed dependency lock")
    if lock.get("schema_version") != LOCK_SCHEMA_VERSION:
        raise SealedDependencyLockError(
            "sealed dependency lock has an unsupported schema"
        )
    if expected_asset_id is not None and lock.get("asset_id") != expected_asset_id:
        raise SealedDependencyLockError(
            "sealed dependency lock asset_id differs from the project"
        )

    material_root_path = Path(material_root).expanduser().resolve(strict=True)
    isaac_root_path = Path(isaac_root).expanduser().resolve(strict=True)
    catalog_file = Path(catalog_path).expanduser().resolve(strict=True)
    selected_records = _records(
        lock.get("selected_materials"),
        "dependency_lock.selected_materials",
    )
    module_records = _records(
        lock.get("top_level_modules"),
        "dependency_lock.top_level_modules",
    )
    selected_by_id, module_by_path = _catalog_contract(
        catalog_path=catalog_file,
        selected_records=selected_records,
        module_records=module_records,
    )

    module_files: dict[str, Path] = {}
    module_exports: dict[str, set[str]] = {}
    top_level_bytes = 0
    for mdl_path, record in module_by_path.items():
        module_file = _rooted_file(
            material_root_path,
            mdl_path,
            f"top-level MDL {mdl_path}",
        )
        if module_file.suffix.casefold() != ".mdl":
            raise SealedDependencyLockError(
                f"top-level material dependency is not MDL: {mdl_path}"
            )
        _require_sha256(record, module_file, f"top-level MDL {mdl_path}")
        module_files[mdl_path] = module_file
        module_exports[mdl_path] = _exported_materials(module_file)
        top_level_bytes += module_file.stat().st_size

    for material_id, record in selected_by_id.items():
        mdl_path = str(record["mdl_path"])
        sub_identifier = record.get("sub_identifier")
        if (
            not isinstance(sub_identifier, str)
            or sub_identifier not in module_exports[mdl_path]
        ):
            raise SealedDependencyLockError(
                "sealed material subidentifier is not exported by its MDL: "
                f"{material_id}#{sub_identifier}"
            )

    resource_records = _records(
        lock.get("runtime_resources"),
        "dependency_lock.runtime_resources",
    )
    resource_by_identity: dict[tuple[str, str], Mapping[str, Any]] = {}
    for index, record in enumerate(resource_records):
        owner = _relative_value(
            record.get("owner_mdl_path"),
            f"dependency_lock.runtime_resources[{index}].owner_mdl_path",
        )
        if owner not in module_files:
            raise SealedDependencyLockError(
                f"runtime resource owner is not a locked top-level MDL: {owner}"
            )
        authored = record.get("authored_path")
        if not isinstance(authored, str) or not authored:
            raise SealedDependencyLockError(
                f"dependency_lock.runtime_resources[{index}].authored_path is invalid"
            )
        if (
            "://" in authored
            or Path(authored).is_absolute()
            or any(token in authored for token in _TILED_ASSET_TOKENS)
        ):
            raise SealedDependencyLockError(
                f"sealed runtime resource path is not a fixed local file: {authored}"
            )
        identity = (owner, authored)
        if identity in resource_by_identity:
            raise SealedDependencyLockError(
                f"dependency lock duplicates runtime resource: {identity}"
            )
        resource_by_identity[identity] = record

    parsed_resource_identities = {
        (owner, authored)
        for owner, module_file in module_files.items()
        for authored in _runtime_asset_literals(module_file)
    }
    if set(resource_by_identity) != parsed_resource_identities:
        raise SealedDependencyLockError(
            "dependency lock runtime resources do not exactly match active "
            "literal MDL resources"
        )

    runtime_resource_bytes = 0
    for (owner, authored), record in resource_by_identity.items():
        owner_file = module_files[owner]
        try:
            authored_file = (owner_file.parent / authored).resolve(strict=True)
            authored_file.relative_to(material_root_path)
        except (OSError, RuntimeError, ValueError) as exc:
            raise SealedDependencyLockError(
                f"runtime resource escapes the NVIDIA material root: {authored}"
            ) from exc
        resolved_file = _rooted_file(
            material_root_path,
            record.get("resolved_path"),
            f"runtime resource {owner}:{authored}",
        )
        if authored_file != resolved_file:
            raise SealedDependencyLockError(
                "runtime resource resolved path differs from its MDL literal: "
                f"{owner}:{authored}"
            )
        _require_sha256(
            record,
            resolved_file,
            f"runtime resource {owner}:{authored}",
        )
        runtime_resource_bytes += resolved_file.stat().st_size

    helper_records = _records(
        lock.get("isaac_helper_modules"),
        "dependency_lock.isaac_helper_modules",
    )
    helper_names: set[str] = set()
    helper_paths: set[str] = set()
    helper_module_bytes = 0
    for index, record in enumerate(helper_records):
        module_name = record.get("module_name")
        relative = _relative_value(
            record.get("relative_to_isaac_root"),
            (
                "dependency_lock.isaac_helper_modules"
                f"[{index}].relative_to_isaac_root"
            ),
        )
        if (
            not isinstance(module_name, str)
            or not module_name.startswith("::")
            or module_name in helper_names
            or relative in helper_paths
        ):
            raise SealedDependencyLockError(
                "Isaac helper module identity is invalid or duplicated"
            )
        helper_file = _rooted_file(
            isaac_root_path,
            relative,
            f"Isaac helper module {module_name}",
        )
        if helper_file.suffix.casefold() != ".mdl":
            raise SealedDependencyLockError(
                f"Isaac helper dependency is not MDL: {relative}"
            )
        _require_sha256(
            record,
            helper_file,
            f"Isaac helper module {module_name}",
        )
        helper_names.add(module_name)
        helper_paths.add(relative)
        helper_module_bytes += helper_file.stat().st_size

    runtime = lock.get("isaac_runtime")
    if not isinstance(runtime, Mapping):
        raise SealedDependencyLockError(
            "dependency_lock.isaac_runtime must be an object"
        )
    version_file = _rooted_file(
        isaac_root_path,
        runtime.get("version_file"),
        "Isaac VERSION contract",
    )
    _require_sha256(runtime, version_file, "Isaac VERSION contract")
    version = version_file.read_text(encoding="utf-8").strip()
    if runtime.get("version") != version:
        raise SealedDependencyLockError("Isaac VERSION content changed")

    summary = lock.get("summary")
    expected_summary = {
        "selected_material_count": len(selected_by_id),
        "top_level_module_count": len(module_files),
        "runtime_resource_count": len(resource_by_identity),
        "isaac_helper_module_count": len(helper_names),
    }
    if not isinstance(summary, Mapping) or any(
        summary.get(key) != value for key, value in expected_summary.items()
    ):
        raise SealedDependencyLockError(
            "sealed dependency lock summary is inconsistent"
        )
    historical_parameter_policy = lock.get("historical_parameter_policy")
    if historical_parameter_policy not in _ALLOWED_HISTORICAL_PARAMETER_POLICIES:
        raise SealedDependencyLockError(
            "sealed historical parameter policy is invalid"
        )

    return {
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "status": "PASS",
        "dependency_lock_verified": True,
        "lock_path": str(lock_file),
        "lock_sha256": expected_lock_sha256,
        "catalog_path": str(catalog_file),
        "material_root": str(material_root_path),
        "isaac_root": str(isaac_root_path),
        "isaac_version": version,
        **expected_summary,
        "top_level_module_bytes": top_level_bytes,
        "runtime_resource_bytes": runtime_resource_bytes,
        "isaac_helper_module_bytes": helper_module_bytes,
        "historical_parameter_policy": historical_parameter_policy,
    }


__all__ = [
    "LOCK_SCHEMA_VERSION",
    "SealedDependencyLockError",
    "VERIFICATION_SCHEMA_VERSION",
    "verify_sealed_dependency_lock",
]
