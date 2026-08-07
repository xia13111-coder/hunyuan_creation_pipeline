#!/usr/bin/env python3
"""Resolve human material-review decisions without mutating model output.

The staged Qwen result is treated as immutable evidence.  A separate review
document either approves/overrides a suggestion or explicitly preserves the
source material.  The resulting material plan contains only human-approved
bindings and can be passed to :mod:`apply_visual_materials` safely.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any


REVIEW_SCHEMA_VERSION = "qwen-material-review/v1"
MATERIAL_PLAN_SCHEMA_VERSION = "1.0"
DECISIONS = frozenset({"approve", "override", "preserve_existing", "reject"})
PACKAGE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_WHITELIST = PACKAGE_DIR / "configs" / "materials" / "industrial_whitelist.json"
SAFE_SUBSET_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
MAX_EXPANDED_FACE_INDICES = 1_000_000


def _read_object(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve(strict=True)
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {resolved}")
    return value


def _write_object(path: str | Path, value: Mapping[str, Any]) -> Path:
    resolved = Path(path).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved.with_name(resolved.name + ".tmp")
    temporary.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(resolved)
    return resolved


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve(strict=True).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _catalog_ids(document: Mapping[str, Any]) -> set[str]:
    records = document.get("materials")
    if not isinstance(records, list):
        raise ValueError("Catalog must contain a materials list")
    identifiers: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"catalog.materials[{index}] must be an object")
        material_id = record.get("material_id")
        if not isinstance(material_id, str) or not material_id:
            raise ValueError(f"catalog.materials[{index}] has invalid material_id")
        if material_id in identifiers:
            raise ValueError(f"Catalog contains duplicate material_id: {material_id}")
        identifiers.add(material_id)
    return identifiers


def _whitelist_ids(document: Mapping[str, Any], catalog_ids: set[str]) -> set[str]:
    """Validate the review whitelist and return its catalog-bounded IDs."""

    schema_version = document.get("schema_version")
    if isinstance(schema_version, bool) or schema_version != 1:
        raise ValueError("Whitelist schema_version must be 1")
    records = document.get("material_ids")
    if not isinstance(records, list) or not records:
        raise ValueError("Whitelist must contain a non-empty material_ids list")
    if any(not isinstance(item, str) or not item for item in records):
        raise ValueError("Whitelist material_ids must be non-empty strings")
    identifiers = set(records)
    if len(identifiers) != len(records):
        raise ValueError("Whitelist contains duplicate material_ids")
    unknown = sorted(identifiers - catalog_ids)
    if unknown:
        raise ValueError(
            f"Whitelist contains material_ids absent from catalog: {unknown}"
        )
    bounded = identifiers & catalog_ids
    if not bounded:
        raise ValueError("Whitelist and catalog have no material_ids in common")
    return bounded


def _review_face_subsets(
    value: Any,
    *,
    part_id: str,
    allowed_material_ids: set[str],
) -> list[dict[str, Any]]:
    """Validate and defensively copy human-authored face subset decisions."""

    if not isinstance(value, list) or not value:
        raise ValueError(f"face_subsets for {part_id} must be a non-empty list")
    allowed_fields = {
        "subset_name",
        "material_id",
        "parameters",
        "semantic",
        "face_indices",
        "face_ranges",
    }
    seen_names: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(value):
        label = f"face_subsets[{index}] for {part_id}"
        if not isinstance(raw, Mapping):
            raise ValueError(f"{label} must be an object")
        unexpected = set(raw) - allowed_fields
        if unexpected:
            raise ValueError(f"{label} has unexpected fields: {sorted(unexpected)}")

        subset_name = raw.get("subset_name")
        if not isinstance(subset_name, str) or not SAFE_SUBSET_NAME.fullmatch(
            subset_name
        ):
            raise ValueError(f"{label} has unsafe subset_name: {subset_name!r}")
        if subset_name in seen_names:
            raise ValueError(f"Duplicate subset_name for {part_id}: {subset_name}")
        seen_names.add(subset_name)

        material_id = raw.get("material_id")
        if not isinstance(material_id, str) or material_id not in allowed_material_ids:
            raise ValueError(f"{label} has unknown material_id: {material_id!r}")

        has_indices = "face_indices" in raw
        has_ranges = "face_ranges" in raw
        if has_indices == has_ranges:
            raise ValueError(
                f"{label} must contain exactly one of face_indices or face_ranges"
            )
        if has_indices:
            face_indices = raw["face_indices"]
            if not isinstance(face_indices, list) or not face_indices:
                raise ValueError(f"{label}.face_indices must be a non-empty list")
            if any(
                isinstance(face_index, bool) or not isinstance(face_index, int)
                for face_index in face_indices
            ):
                raise ValueError(f"{label}.face_indices must contain only integers")
            if any(face_index < 0 for face_index in face_indices):
                raise ValueError(f"{label}.face_indices must be non-negative")
            face_indices = list(face_indices)
        else:
            face_ranges = raw["face_ranges"]
            if not isinstance(face_ranges, list) or not face_ranges:
                raise ValueError(f"{label}.face_ranges must be a non-empty list")
            face_indices = []
            for range_index, face_range in enumerate(face_ranges):
                range_label = f"{label}.face_ranges[{range_index}]"
                if (
                    not isinstance(face_range, list)
                    or len(face_range) != 2
                    or any(
                        isinstance(bound, bool) or not isinstance(bound, int)
                        for bound in face_range
                    )
                ):
                    raise ValueError(
                        f"{range_label} must be an inclusive [start, end] integer pair"
                    )
                start, end = face_range
                if start < 0 or end < 0:
                    raise ValueError(f"{range_label} bounds must be non-negative")
                if end < start:
                    raise ValueError(f"{range_label} end must be >= start")
                expanded_count = end - start + 1
                if len(face_indices) + expanded_count > MAX_EXPANDED_FACE_INDICES:
                    raise ValueError(
                        f"{label} expands beyond {MAX_EXPANDED_FACE_INDICES} faces"
                    )
                face_indices.extend(range(start, end + 1))
        if len(set(face_indices)) != len(face_indices):
            raise ValueError(f"{label}.face_indices/face_ranges must be unique faces")

        subset: dict[str, Any] = {
            "subset_name": subset_name,
            "material_id": material_id,
            "face_indices": list(face_indices),
        }
        if "parameters" in raw:
            parameters = raw["parameters"]
            if not isinstance(parameters, Mapping):
                raise ValueError(f"{label}.parameters must be an object")
            subset["parameters"] = dict(parameters)
        if "semantic" in raw:
            semantic = raw["semantic"]
            if not isinstance(semantic, str) or not semantic.strip():
                raise ValueError(f"{label}.semantic must be a non-empty string")
            subset["semantic"] = semantic
        normalized.append(subset)
    return normalized


def _registry_ids(document: Mapping[str, Any]) -> set[str]:
    records = document.get("parts")
    if not isinstance(records, list) or not records:
        raise ValueError("Registry must contain a non-empty parts list")
    identifiers: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"registry.parts[{index}] must be an object")
        part_id = record.get("part_id")
        if not isinstance(part_id, str) or not part_id:
            raise ValueError(f"registry.parts[{index}] has invalid part_id")
        if part_id in identifiers:
            raise ValueError(f"Registry contains duplicate part_id: {part_id}")
        identifiers.add(part_id)
    return identifiers


def resolve_review_decisions(
    staged_result: Mapping[str, Any],
    review: Mapping[str, Any],
    *,
    source_result_sha256: str,
    allowed_material_ids: set[str],
    expected_part_ids: set[str] | None = None,
    require_complete: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return an approved plan and an auditable review report."""

    if review.get("schema_version") != REVIEW_SCHEMA_VERSION:
        raise ValueError(f"review.schema_version must be {REVIEW_SCHEMA_VERSION!r}")
    if review.get("source_result_sha256") != source_result_sha256:
        raise ValueError("Review source_result_sha256 does not match staged result")

    material_plan = staged_result.get("material_plan")
    if not isinstance(material_plan, Mapping):
        raise ValueError("Staged result has no material_plan object")
    raw_assignments = material_plan.get("assignments")
    raw_unknown = staged_result.get("unknown_parts")
    if not isinstance(raw_assignments, list) or not isinstance(raw_unknown, list):
        raise ValueError("Staged result assignments/unknown_parts must be lists")

    suggestions: dict[str, dict[str, Any]] = {}
    all_parts: set[str] = set()
    for index, assignment in enumerate(raw_assignments):
        if not isinstance(assignment, dict):
            raise ValueError(f"material_plan.assignments[{index}] must be an object")
        part_id = assignment.get("part_id")
        if not isinstance(part_id, str) or not part_id or part_id in all_parts:
            raise ValueError(f"Invalid or duplicate staged part_id: {part_id!r}")
        all_parts.add(part_id)
        suggestions[part_id] = dict(assignment)
    for index, record in enumerate(raw_unknown):
        if not isinstance(record, Mapping):
            raise ValueError(f"unknown_parts[{index}] must be an object")
        part_id = record.get("part_id")
        if not isinstance(part_id, str) or not part_id or part_id in all_parts:
            raise ValueError(f"Invalid or duplicate unknown part_id: {part_id!r}")
        all_parts.add(part_id)

    if expected_part_ids is not None and all_parts != expected_part_ids:
        missing = sorted(expected_part_ids - all_parts)
        unexpected = sorted(all_parts - expected_part_ids)
        raise ValueError(
            "Staged result does not exactly cover the registry; "
            f"missing={missing}, unexpected={unexpected}"
        )

    raw_decisions = review.get("decisions")
    if not isinstance(raw_decisions, list):
        raise ValueError("review.decisions must be a list")
    decisions_by_part: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(raw_decisions):
        if not isinstance(raw, dict):
            raise ValueError(f"review.decisions[{index}] must be an object")
        allowed_fields = {
            "part_id",
            "decision",
            "material_id",
            "semantic",
            "note",
            "parameters",
            "face_subsets",
            "preserve_parent_material_binding",
        }
        unexpected = set(raw) - allowed_fields
        if unexpected:
            raise ValueError(
                f"review.decisions[{index}] has unexpected fields: {sorted(unexpected)}"
            )
        part_id = raw.get("part_id")
        decision = raw.get("decision")
        if not isinstance(part_id, str) or part_id not in all_parts:
            raise ValueError(f"Unknown reviewed part_id: {part_id!r}")
        if part_id in decisions_by_part:
            raise ValueError(f"Duplicate review decision for part_id: {part_id}")
        if decision not in DECISIONS:
            raise ValueError(
                f"Invalid decision for {part_id}: {decision!r}; allowed={sorted(DECISIONS)}"
            )
        decisions_by_part[part_id] = dict(raw)

    if require_complete and set(decisions_by_part) != all_parts:
        missing = sorted(all_parts - set(decisions_by_part))
        raise ValueError(f"Review does not cover every part; missing={missing}")

    approved: list[dict[str, Any]] = []
    preserved: list[str] = []
    rejected: list[str] = []
    for part_id in sorted(decisions_by_part):
        decision = decisions_by_part[part_id]
        action = decision["decision"]
        suggestion = suggestions.get(part_id)
        if action == "approve":
            if suggestion is None:
                raise ValueError(
                    f"Cannot approve {part_id}: no model suggestion exists"
                )
            material_id = suggestion.get("material_id")
            semantic = suggestion.get("semantic", "human approved model suggestion")
            evidence_views = suggestion.get("evidence_views", [])
            confidence = suggestion.get("confidence", 0.0)
        elif action == "override":
            material_id = decision.get("material_id")
            semantic = decision.get("semantic", "human material override")
            evidence_views = (
                suggestion.get("evidence_views", []) if suggestion is not None else []
            )
            confidence = 1.0
        elif action == "preserve_existing":
            if "face_subsets" in decision:
                raise ValueError(
                    f"face_subsets are not allowed with preserve_existing: {part_id}"
                )
            if "preserve_parent_material_binding" in decision:
                raise ValueError(
                    "preserve_parent_material_binding is not allowed with "
                    f"preserve_existing: {part_id}"
                )
            preserved.append(part_id)
            continue
        else:
            if "face_subsets" in decision:
                raise ValueError(f"face_subsets are not allowed with reject: {part_id}")
            if "preserve_parent_material_binding" in decision:
                raise ValueError(
                    "preserve_parent_material_binding is not allowed with "
                    f"reject: {part_id}"
                )
            rejected.append(part_id)
            continue

        if not isinstance(material_id, str) or material_id not in allowed_material_ids:
            raise ValueError(
                f"Decision for {part_id} has unknown material_id: {material_id!r}"
            )
        assignment: dict[str, Any] = {
            "part_id": part_id,
            "material_id": material_id,
            "semantic": str(semantic),
            "confidence": float(confidence),
            "evidence_views": (
                list(evidence_views) if isinstance(evidence_views, list) else []
            ),
            "status": "approved",
        }
        parameters = decision.get("parameters")
        if parameters is not None:
            if not isinstance(parameters, dict):
                raise ValueError(f"Decision parameters for {part_id} must be an object")
            assignment["parameters"] = dict(parameters)
        if "face_subsets" in decision:
            assignment["face_subsets"] = _review_face_subsets(
                decision["face_subsets"],
                part_id=part_id,
                allowed_material_ids=allowed_material_ids,
            )
        if "preserve_parent_material_binding" in decision:
            preserve_parent = decision["preserve_parent_material_binding"]
            if type(preserve_parent) is not bool:
                raise ValueError(
                    f"preserve_parent_material_binding for {part_id} must be a boolean"
                )
            if preserve_parent and not assignment.get("face_subsets"):
                raise ValueError(
                    "preserve_parent_material_binding for "
                    f"{part_id} requires face_subsets"
                )
            assignment["preserve_parent_material_binding"] = preserve_parent
        approved.append(assignment)

    plan = {
        "schema_version": MATERIAL_PLAN_SCHEMA_VERSION,
        "assignments": approved,
    }
    report = {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "source_result_sha256": source_result_sha256,
        "part_count": len(all_parts),
        "decision_count": len(decisions_by_part),
        "approved_count": len(approved),
        "preserve_existing_count": len(preserved),
        "rejected_count": len(rejected),
        "unreviewed_count": len(all_parts - set(decisions_by_part)),
        "approved_parts": [item["part_id"] for item in approved],
        "face_subset_count": sum(
            len(item.get("face_subsets", [])) for item in approved
        ),
        "face_subset_parts": [
            item["part_id"] for item in approved if item.get("face_subsets")
        ],
        "preserved_parts": preserved,
        "rejected_parts": rejected,
        "complete": set(decisions_by_part) == all_parts,
    }
    return plan, report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staged-result", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--whitelist", type=Path, default=DEFAULT_WHITELIST)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output-plan", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    staged_path = args.staged_result.expanduser().resolve(strict=True)
    staged = _read_object(staged_path)
    review = _read_object(args.review)
    catalog = _read_object(args.catalog)
    catalog_ids = _catalog_ids(catalog)
    whitelist = _read_object(args.whitelist)
    registry = _read_object(args.registry)
    plan, report = resolve_review_decisions(
        staged,
        review,
        source_result_sha256=sha256_file(staged_path),
        allowed_material_ids=_whitelist_ids(whitelist, catalog_ids),
        expected_part_ids=_registry_ids(registry),
        require_complete=not args.allow_partial,
    )
    _write_object(args.output_plan, plan)
    _write_object(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
