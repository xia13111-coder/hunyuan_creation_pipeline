"""Create an immutable exact-MDL candidate by rebinding one visual cohort.

This command never authors material parameters.  It only replaces exact
``material_id`` identities already present in a complete instance plan,
including face-subset identities, and records an auditable cohort delta.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


class ExactGroupRebindError(ValueError):
    """Raised when an immutable cohort rebind is unsafe or incomplete."""


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require_default_parameters(record: Mapping[str, Any], label: str) -> None:
    parameters = record.get("parameters")
    if parameters is not None and (
        not isinstance(parameters, Mapping) or bool(parameters)
    ):
        raise ExactGroupRebindError(f"{label} modifies MDL parameters")


def rebind_exact_material_cohort(
    *,
    plan: Mapping[str, Any],
    catalog: Mapping[str, Any],
    source_material_ids: set[str],
    target_material_id: str,
    group_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Replace one exact-material cohort without changing any MDL parameter."""

    if plan.get("schema_version") != "1.0":
        raise ExactGroupRebindError("plan schema_version must be 1.0")
    if not source_material_ids or any(
        not value.startswith("mdl:") for value in source_material_ids
    ):
        raise ExactGroupRebindError("source_material_ids must be exact MDL IDs")
    if not target_material_id.startswith("mdl:"):
        raise ExactGroupRebindError("target_material_id must be an exact MDL ID")
    if not group_id:
        raise ExactGroupRebindError("group_id must not be empty")

    catalog_ids = {
        str(item["material_id"])
        for item in catalog.get("materials", [])
        if isinstance(item, Mapping)
        and isinstance(item.get("material_id"), str)
    }
    missing = sorted((source_material_ids | {target_material_id}) - catalog_ids)
    if missing:
        raise ExactGroupRebindError(
            "material identities are absent from the catalog: "
            + ", ".join(missing)
        )

    output = copy.deepcopy(dict(plan))
    assignments = output.get("assignments")
    if not isinstance(assignments, Sequence) or isinstance(
        assignments, (str, bytes)
    ):
        raise ExactGroupRebindError("plan.assignments must be an array")

    part_changes: list[dict[str, Any]] = []
    subset_changes: list[dict[str, Any]] = []
    seen_parts: set[str] = set()
    for index, assignment in enumerate(assignments):
        if not isinstance(assignment, dict):
            raise ExactGroupRebindError(f"assignments[{index}] must be an object")
        part_id = assignment.get("part_id")
        if (
            not isinstance(part_id, str)
            or not part_id
            or part_id in seen_parts
        ):
            raise ExactGroupRebindError(f"assignments[{index}] has invalid part_id")
        seen_parts.add(part_id)
        _require_default_parameters(assignment, part_id)

        old_material_id = assignment.get("material_id")
        if old_material_id in source_material_ids:
            assignment["material_id"] = target_material_id
            provenance = assignment.setdefault("provenance", {})
            if not isinstance(provenance, dict):
                raise ExactGroupRebindError(f"{part_id}.provenance must be an object")
            provenance["canonical_group_id"] = group_id
            part_changes.append(
                {
                    "part_id": part_id,
                    "old_material_id": old_material_id,
                    "new_material_id": target_material_id,
                }
            )

        raw_subsets = assignment.get("face_subsets", [])
        if not isinstance(raw_subsets, Sequence) or isinstance(
            raw_subsets, (str, bytes)
        ):
            raise ExactGroupRebindError(f"{part_id}.face_subsets must be an array")
        for subset_index, subset in enumerate(raw_subsets):
            if not isinstance(subset, dict):
                raise ExactGroupRebindError(
                    f"{part_id}.face_subsets[{subset_index}] must be an object"
                )
            _require_default_parameters(
                subset,
                f"{part_id}.face_subsets[{subset_index}]",
            )
            old_subset_material_id = subset.get("material_id")
            if old_subset_material_id not in source_material_ids:
                continue
            subset["material_id"] = target_material_id
            subset_changes.append(
                {
                    "part_id": part_id,
                    "subset_name": subset.get("subset_name"),
                    "old_material_id": old_subset_material_id,
                    "new_material_id": target_material_id,
                }
            )

    if not part_changes and not subset_changes:
        raise ExactGroupRebindError("source cohort does not occur in the plan")

    plan_provenance = output.setdefault("provenance", {})
    if not isinstance(plan_provenance, dict):
        raise ExactGroupRebindError("plan.provenance must be an object")
    plan_provenance["immutable_mdl_after_selection"] = True
    plan_provenance["exact_group_rebind"] = {
        "group_id": group_id,
        "source_plan_sha256": _canonical_sha256(plan),
        "source_material_ids": sorted(source_material_ids),
        "target_material_id": target_material_id,
        "changed_part_ids": sorted(
            {change["part_id"] for change in part_changes + subset_changes}
        ),
        "parameters_locked_to_library_defaults": True,
    }
    audit = {
        "schema_version": "qwen-exact-group-rebind/v1",
        "status": "RESOLVED",
        "group_id": group_id,
        "source_plan_sha256": _canonical_sha256(plan),
        "output_plan_sha256": _canonical_sha256(output),
        "source_material_ids": sorted(source_material_ids),
        "target_material_id": target_material_id,
        "part_change_count": len(part_changes),
        "face_subset_change_count": len(subset_changes),
        "parameter_write_count": 0,
        "parameters_locked_to_library_defaults": True,
        "part_changes": part_changes,
        "face_subset_changes": subset_changes,
    }
    return output, audit


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.expanduser().resolve(strict=True).read_text("utf-8"))
    if not isinstance(value, dict):
        raise ExactGroupRebindError(f"{path} must contain a JSON object")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    output = path.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--source-material-id", action="append", required=True)
    parser.add_argument("--target-material-id", required=True)
    parser.add_argument("--group-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()

    output, audit = rebind_exact_material_cohort(
        plan=_read_json(args.plan),
        catalog=_read_json(args.catalog),
        source_material_ids=set(args.source_material_id),
        target_material_id=args.target_material_id,
        group_id=args.group_id,
    )
    _write_json(args.output, output)
    _write_json(args.audit, audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
