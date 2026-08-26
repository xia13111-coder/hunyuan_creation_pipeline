"""Seal selected NVIDIA MDL assignments against post-selection mutation.

The lock is intentionally narrower than a whole material-plan hash: occurrence
plans may add provenance needed by the instance-aware USD writer, but their
ordered assignment objects must remain byte-for-byte equivalent under
canonical JSON serialization.  Every selected MDL module is also content
hashed beneath the caller-controlled NVIDIA material root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .catalog import MaterialCatalog
from .tuning import parameter_policy_for_material
from ..usd.material_common import normalize_material_parameters


SCHEMA_VERSION = "qwen-selected-mdl-lock/v1"
POLICY = "selected_mdl_identity_library_defaults_and_subsets_immutable/v1"
REVIEWED_COLOR_POLICY = (
    "selected_mdl_identity_reviewed_color_parameters_and_subsets_immutable/v1"
)


class MaterialSelectionLockError(ValueError):
    """Raised when a selected MDL plan or source module changed."""


def _canonical_sha256(value: Any) -> str:
    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise MaterialSelectionLockError(
            f"material selection is not canonical JSON: {exc}"
        ) from exc
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assignments(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    if plan.get("schema_version") != "1.0":
        raise MaterialSelectionLockError("material plan schema_version must be '1.0'")
    raw = plan.get("assignments")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or not raw:
        raise MaterialSelectionLockError(
            "material plan must contain non-empty assignments"
        )
    result: list[dict[str, Any]] = []
    part_ids: set[str] = set()
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise MaterialSelectionLockError(
                f"material assignment {index} must be an object"
            )
        assignment = dict(item)
        part_id = assignment.get("part_id")
        if not isinstance(part_id, str) or not part_id or part_id in part_ids:
            raise MaterialSelectionLockError(
                f"material assignment {index} has an invalid/duplicate part_id"
            )
        part_ids.add(part_id)
        result.append(assignment)
    return result


def _material_ids(assignments: Sequence[Mapping[str, Any]]) -> list[str]:
    result: set[str] = set()
    for assignment in assignments:
        material_id = assignment.get("material_id")
        if material_id is not None:
            if not isinstance(material_id, str) or not material_id:
                raise MaterialSelectionLockError("assignment material_id is invalid")
            result.add(material_id)
        raw_subsets = assignment.get("face_subsets", [])
        if not isinstance(raw_subsets, Sequence) or isinstance(
            raw_subsets, (str, bytes)
        ):
            raise MaterialSelectionLockError("assignment face_subsets is invalid")
        for subset in raw_subsets:
            if not isinstance(subset, Mapping):
                raise MaterialSelectionLockError("face subset must be an object")
            subset_material_id = subset.get("material_id")
            if not isinstance(subset_material_id, str) or not subset_material_id:
                raise MaterialSelectionLockError("face subset material_id is invalid")
            result.add(subset_material_id)
    return sorted(result)


def _reject_parameter_overrides(
    assignments: Sequence[Mapping[str, Any]],
) -> None:
    for assignment in assignments:
        part_id = str(assignment["part_id"])
        parameters = assignment.get("parameters")
        if parameters is not None and (
            not isinstance(parameters, Mapping) or bool(parameters)
        ):
            raise MaterialSelectionLockError(
                f"selected MDL must use library-default parameters: {part_id}"
            )
        for subset in assignment.get("face_subsets", []):
            subset_parameters = subset.get("parameters")
            if subset_parameters is not None and (
                not isinstance(subset_parameters, Mapping) or bool(subset_parameters)
            ):
                raise MaterialSelectionLockError(
                    "selected face-subset MDL must use library-default "
                    f"parameters: {part_id}/{subset.get('subset_name')}"
                )


def _validate_reviewed_color_parameters(
    assignments: Sequence[Mapping[str, Any]],
) -> None:
    for assignment in assignments:
        part_id = str(assignment["part_id"])
        material_id = assignment.get("material_id")
        parameters = assignment.get("parameters")
        if parameters not in (None, {}):
            if not isinstance(material_id, str) or not isinstance(parameters, Mapping):
                raise MaterialSelectionLockError(
                    f"reviewed colour parameters are invalid: {part_id}"
                )
            policy = parameter_policy_for_material(material_id)
            if not policy or set(parameters) - set(policy):
                raise MaterialSelectionLockError(
                    f"selected MDL has unreviewed colour parameters: {part_id}"
                )
            try:
                normalize_material_parameters(material_id, dict(parameters))
            except ValueError as exc:
                raise MaterialSelectionLockError(
                    f"selected MDL colour parameters are invalid: {part_id}"
                ) from exc
        for subset in assignment.get("face_subsets", []):
            if subset.get("parameters") not in (None, {}):
                raise MaterialSelectionLockError(
                    "reviewed colour calibration cannot author face-subset "
                    f"parameters: {part_id}/{subset.get('subset_name')}"
                )


def build_material_selection_lock(
    *,
    plan: Mapping[str, Any],
    catalog_path: str | Path,
    material_root: str | Path,
    allow_reviewed_color_parameters: bool = False,
) -> dict[str, Any]:
    """Build a deterministic lock for one completed MDL selection."""

    assignments = _assignments(plan)
    if allow_reviewed_color_parameters:
        _validate_reviewed_color_parameters(assignments)
    else:
        _reject_parameter_overrides(assignments)
    catalog_file = Path(catalog_path).expanduser().resolve(strict=True)
    root = Path(material_root).expanduser().resolve(strict=True)
    catalog = MaterialCatalog.load(catalog_file, material_root=root)
    modules: dict[str, dict[str, Any]] = {}
    material_records: list[dict[str, Any]] = []
    for material_id in _material_ids(assignments):
        try:
            mdl_file, sub_identifier = catalog.resolve_material(material_id)
        except (KeyError, OSError, ValueError) as exc:
            raise MaterialSelectionLockError(
                f"selected material cannot be resolved: {material_id}"
            ) from exc
        relative = mdl_file.relative_to(root).as_posix()
        digest = _sha256_file(mdl_file)
        modules.setdefault(
            relative,
            {
                "mdl_path": relative,
                "mdl_sha256": digest,
            },
        )
        material_records.append(
            {
                "material_id": material_id,
                "mdl_path": relative,
                "sub_identifier": sub_identifier,
                "mdl_sha256": digest,
            }
        )

    lock: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "policy": REVIEWED_COLOR_POLICY if allow_reviewed_color_parameters else POLICY,
        "catalog_sha256": _sha256_file(catalog_file),
        "assignment_count": len(assignments),
        "assignments_sha256": _canonical_sha256(assignments),
        "assignments": [
            {
                "part_id": assignment["part_id"],
                "assignment_sha256": _canonical_sha256(assignment),
                "material_id": assignment.get("material_id"),
                "parameters_locked": "parameters" in assignment,
                "face_subsets_locked": "face_subsets" in assignment,
            }
            for assignment in assignments
        ],
        "selected_materials": material_records,
        "selected_mdl_modules": [modules[path] for path in sorted(modules)],
        "post_selection_operations": {
            "replace_material_id": False,
            "write_parameters": False,
            "add_or_change_face_subsets": False,
            "appearance_optimization": False,
            "quality_repair_material_mutation": False,
            "camera_pose_and_quality_measurement_only": True,
        },
        "selected_mdl_library_defaults_required": (not allow_reviewed_color_parameters),
        "reviewed_color_parameters_locked": allow_reviewed_color_parameters,
    }
    lock["integrity"] = {
        "lock_sha256": _canonical_sha256(lock),
    }
    return lock


def validate_material_selection_lock(
    *,
    lock: Mapping[str, Any],
    plan: Mapping[str, Any],
    catalog_path: str | Path,
    material_root: str | Path,
) -> dict[str, Any]:
    """Recompute and validate a selected-MDL lock, returning a safe copy."""

    policy = lock.get("policy")
    if lock.get("schema_version") != SCHEMA_VERSION or policy not in {
        POLICY,
        REVIEWED_COLOR_POLICY,
    }:
        raise MaterialSelectionLockError(
            "material selection lock has an unsupported schema or policy"
        )
    raw_integrity = lock.get("integrity")
    if not isinstance(raw_integrity, Mapping):
        raise MaterialSelectionLockError("material selection lock lacks integrity")
    unsigned = {key: value for key, value in lock.items() if key != "integrity"}
    if raw_integrity.get("lock_sha256") != _canonical_sha256(unsigned):
        raise MaterialSelectionLockError(
            "material selection lock integrity check failed"
        )
    expected = build_material_selection_lock(
        plan=plan,
        catalog_path=catalog_path,
        material_root=material_root,
        allow_reviewed_color_parameters=(policy == REVIEWED_COLOR_POLICY),
    )
    if dict(lock) != expected:
        raise MaterialSelectionLockError(
            "selected MDL identity, parameters, face subsets, catalog, or "
            "source module changed after selection"
        )
    return expected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--material-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    plan = json.loads(args.plan.expanduser().resolve(strict=True).read_text("utf-8"))
    if not isinstance(plan, Mapping):
        raise MaterialSelectionLockError("plan must contain a JSON object")
    lock = build_material_selection_lock(
        plan=plan,
        catalog_path=args.catalog,
        material_root=args.material_root,
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(lock, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(
        json.dumps(
            {
                "status": "LOCKED",
                "output": str(output),
                "assignment_count": lock["assignment_count"],
                "selected_material_count": len(lock["selected_materials"]),
                "lock_sha256": lock["integrity"]["lock_sha256"],
            },
            ensure_ascii=False,
        )
    )
    return 0


__all__ = [
    "MaterialSelectionLockError",
    "POLICY",
    "REVIEWED_COLOR_POLICY",
    "SCHEMA_VERSION",
    "build_material_selection_lock",
    "validate_material_selection_lock",
]


if __name__ == "__main__":
    raise SystemExit(main())
