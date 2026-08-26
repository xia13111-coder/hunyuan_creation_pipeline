"""Build a complete, hash-bound material plan from reviewed topology evidence.

This command is intentionally separate from the fail-closed unattended gate.
It consumes an already reviewed full-part plan, replaces selected painted
material parameters with eligible MVInverse group suggestions, and fills only
the explicitly listed structural fallbacks.  The output must cover every part
in the source registry exactly once.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from qwen_material_pipeline.mvinverse.evidence import (
    MVInverseEvidenceError,
    validate_mvinverse_evidence,
)


POLICY_SCHEMA_VERSION = "qwen-complete-material-policy/v1"
REPORT_SCHEMA_VERSION = "qwen-complete-material-plan-report/v1"
MVINVERSE_SCHEMA_VERSION = "qwen-mvinverse-pbr-evidence/v1"
MATERIAL_PLAN_SCHEMA_VERSION = "1.0"
GENERIC_STEEL_PAINTED = "mdl:vMaterials_2/Metal/Steel_Painted.mdl#Steel_Painted"


class CompleteMaterialPlanError(ValueError):
    """Raised when a complete material plan cannot be proven deterministic."""


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CompleteMaterialPlanError(f"{label} must be an object")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise CompleteMaterialPlanError(f"{label} must be an array")
    return value


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CompleteMaterialPlanError(f"{label} must be a non-empty string")
    return value.strip()


def _unit(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise CompleteMaterialPlanError(f"{label} must be a finite number from 0 to 1")
    return float(value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_document(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _srgb_channel_to_linear(value: float) -> float:
    return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4


def _registry_parts(registry: Mapping[str, Any]) -> tuple[list[str], str]:
    parts = _sequence(registry.get("parts"), "registry.parts")
    part_ids: list[str] = []
    for index, raw_part in enumerate(parts):
        part = _mapping(raw_part, f"registry.parts[{index}]")
        part_ids.append(
            _string(part.get("part_id"), f"registry.parts[{index}].part_id")
        )
    if len(set(part_ids)) != len(part_ids):
        raise CompleteMaterialPlanError("registry contains duplicate part_id values")
    asset_sha256 = _string(registry.get("asset_sha256"), "registry.asset_sha256")
    return part_ids, asset_sha256


def _evidence_groups(evidence: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    if evidence.get("schema_version") != MVINVERSE_SCHEMA_VERSION:
        raise CompleteMaterialPlanError(
            "MVInverse evidence has an unsupported schema_version"
        )
    groups: dict[str, Mapping[str, Any]] = {}
    for index, raw_group in enumerate(
        _sequence(evidence.get("groups"), "mvinverse_evidence.groups")
    ):
        group = _mapping(raw_group, f"mvinverse_evidence.groups[{index}]")
        group_id = _string(group.get("group_id"), f"evidence group[{index}].group_id")
        if group_id in groups:
            raise CompleteMaterialPlanError(
                f"duplicate MVInverse evidence group: {group_id}"
            )
        groups[group_id] = group
    return groups


def _paint_parameters(
    group: Mapping[str, Any], *, group_id: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    suggestion = _mapping(
        group.get("suggestion"), f"evidence group {group_id}.suggestion"
    )
    if (
        suggestion.get("decision") != "auto"
        or suggestion.get("auto_parameter_eligible") is not True
    ):
        raise CompleteMaterialPlanError(
            f"MVInverse group {group_id} is not eligible for automatic parameters"
        )
    if group.get("surface_class") != "dielectric":
        raise CompleteMaterialPlanError(
            f"MVInverse group {group_id} is not a dielectric painted surface"
        )
    color_srgb_raw = _sequence(
        suggestion.get("base_color_srgb"),
        f"evidence group {group_id}.suggestion.base_color_srgb",
    )
    if len(color_srgb_raw) != 3:
        raise CompleteMaterialPlanError(
            f"MVInverse group {group_id} base_color_srgb must have three channels"
        )
    color_srgb = [
        _unit(channel, f"evidence group {group_id}.base_color_srgb[{index}]")
        for index, channel in enumerate(color_srgb_raw)
    ]
    authored_metallic = _unit(
        suggestion.get("metallic"), f"evidence group {group_id}.metallic"
    )
    if authored_metallic != 0.0:
        raise CompleteMaterialPlanError(
            f"painted dielectric group {group_id} must author metallic=0"
        )
    roughness = _unit(
        suggestion.get("roughness"), f"evidence group {group_id}.roughness"
    )
    observed_metallic = _unit(
        _mapping(
            group.get("metallic"), f"evidence group {group_id}.metallic stats"
        ).get("median"),
        f"evidence group {group_id}.metallic.median",
    )
    contributing_view_ids = [
        _string(value, f"evidence group {group_id}.contributing_view_ids")
        for value in _sequence(
            group.get("contributing_view_ids", []),
            f"evidence group {group_id}.contributing_view_ids",
        )
    ]
    if len(contributing_view_ids) < 2 or len(set(contributing_view_ids)) != len(
        contributing_view_ids
    ):
        raise CompleteMaterialPlanError(
            f"MVInverse group {group_id} needs distinct multi-view evidence"
        )
    parameters = {
        "paint_color": [_srgb_channel_to_linear(value) for value in color_srgb],
        "paint_roughness": roughness,
        "paint_roughness_variation": 0.0,
        "dirt_weight": 0.0,
        "wash_weight": 0.0,
        "paint_stroke_normal_strength": 0.0,
        "uneven_normal_strength": 0.0,
        "enable_rust_damage": False,
    }
    audit = {
        "group_id": group_id,
        "contributing_view_ids": sorted(contributing_view_ids),
        "base_color_srgb": color_srgb,
        "paint_color_linear": parameters["paint_color"],
        "observed_metallic": observed_metallic,
        "authored_metallic": authored_metallic,
        "paint_roughness": roughness,
    }
    return parameters, audit


def build_complete_material_plan(
    *,
    base_plan: Mapping[str, Any],
    registry: Mapping[str, Any],
    mvinverse_evidence: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return an exact-cover material plan and its deterministic audit report."""

    if base_plan.get("schema_version") != MATERIAL_PLAN_SCHEMA_VERSION:
        raise CompleteMaterialPlanError(
            "base material plan schema_version must be '1.0'"
        )
    if policy.get("schema_version") != POLICY_SCHEMA_VERSION:
        raise CompleteMaterialPlanError(
            "completion policy has an unsupported schema_version"
        )

    registry_part_ids, registry_asset_sha256 = _registry_parts(registry)
    expected_asset_sha256 = _string(
        policy.get("expected_asset_sha256"), "policy.expected_asset_sha256"
    )
    if registry_asset_sha256 != expected_asset_sha256:
        raise CompleteMaterialPlanError(
            "completion policy asset SHA-256 does not match the registry"
        )

    assignments_by_part: dict[str, dict[str, Any]] = {}
    for index, raw_assignment in enumerate(
        _sequence(base_plan.get("assignments"), "base_plan.assignments")
    ):
        assignment = copy.deepcopy(
            dict(_mapping(raw_assignment, f"base_plan.assignments[{index}]"))
        )
        part_id = _string(assignment.get("part_id"), f"assignment[{index}].part_id")
        if part_id in assignments_by_part:
            raise CompleteMaterialPlanError(f"duplicate base assignment: {part_id}")
        if part_id not in registry_part_ids:
            raise CompleteMaterialPlanError(
                f"base assignment is not in registry: {part_id}"
            )
        assignments_by_part[part_id] = assignment

    mvinverse_validation: dict[str, Any]
    try:
        verified_mvinverse_evidence = validate_mvinverse_evidence(
            mvinverse_evidence
        )
    except MVInverseEvidenceError as exc:
        # Exact coverage is the primary contract.  A caller that supplies only
        # a projection of the MVInverse report (for example forged ``groups``)
        # must not be able to author shader parameters, but it also must not
        # prevent the reviewed/fallback plan from being completed.
        verified_mvinverse_evidence = None
        mvinverse_validation = {
            "state": "rejected_fail_closed",
            "reason_code": "MVINVERSE_EVIDENCE_STRICT_VALIDATION_FAILED",
            "detail": str(exc),
        }
    else:
        mvinverse_validation = {
            "state": "verified",
            "reason_code": "MVINVERSE_EVIDENCE_STRICT_VALIDATION_PASSED",
        }
    groups = (
        _evidence_groups(verified_mvinverse_evidence)
        if verified_mvinverse_evidence is not None
        else {}
    )
    parameterized_parts: list[str] = []
    group_audits: list[dict[str, Any]] = []
    skipped_parameterized_groups: list[dict[str, Any]] = []
    seen_parameterized_parts: set[str] = set()
    for index, raw_spec in enumerate(
        _sequence(policy.get("parameterized_groups"), "policy.parameterized_groups")
    ):
        spec = _mapping(raw_spec, f"policy.parameterized_groups[{index}]")
        group_id = _string(
            spec.get("group_id"), f"parameterized_groups[{index}].group_id"
        )
        material_id = _string(
            spec.get("material_id"), f"parameterized_groups[{index}].material_id"
        )
        if material_id != GENERIC_STEEL_PAINTED:
            raise CompleteMaterialPlanError(
                f"MVInverse paint parameters require {GENERIC_STEEL_PAINTED}"
            )
        if verified_mvinverse_evidence is None:
            skipped_parameterized_groups.append(
                {
                    "group_id": group_id,
                    "material_id": material_id,
                    "reason_code": "MVINVERSE_EVIDENCE_STRICT_VALIDATION_FAILED",
                }
            )
            continue
        group = groups.get(group_id)
        if group is None:
            raise CompleteMaterialPlanError(
                f"missing MVInverse evidence group: {group_id}"
            )
        parameters, group_audit = _paint_parameters(group, group_id=group_id)
        part_ids = [
            _string(value, f"parameterized_groups[{index}].part_ids")
            for value in _sequence(
                spec.get("part_ids"), f"parameterized_groups[{index}].part_ids"
            )
        ]
        if not part_ids or len(set(part_ids)) != len(part_ids):
            raise CompleteMaterialPlanError(
                f"parameterized group {group_id} needs unique part_ids"
            )
        for part_id in part_ids:
            if part_id in seen_parameterized_parts:
                raise CompleteMaterialPlanError(
                    f"part is parameterized by more than one group: {part_id}"
                )
            seen_parameterized_parts.add(part_id)
            assignment = assignments_by_part.get(part_id)
            if assignment is None:
                raise CompleteMaterialPlanError(
                    f"parameterized part has no reviewed base assignment: {part_id}"
                )
            if assignment.get("material_id") != material_id:
                raise CompleteMaterialPlanError(
                    f"material mismatch for {part_id}: expected {material_id}"
                )
            assignment["parameters"] = copy.deepcopy(parameters)
            assignment["evidence_views"] = list(group_audit["contributing_view_ids"])
            parameterized_parts.append(part_id)
        group_audit["part_ids"] = sorted(part_ids)
        group_audits.append(group_audit)

    fallback_parts: list[str] = []
    for index, raw_assignment in enumerate(
        _sequence(policy.get("fallback_assignments", []), "policy.fallback_assignments")
    ):
        assignment = copy.deepcopy(
            dict(_mapping(raw_assignment, f"policy.fallback_assignments[{index}]"))
        )
        part_id = _string(
            assignment.get("part_id"), f"policy.fallback_assignments[{index}].part_id"
        )
        if part_id not in registry_part_ids:
            raise CompleteMaterialPlanError(
                f"fallback part is not in registry: {part_id}"
            )
        if part_id in assignments_by_part:
            raise CompleteMaterialPlanError(
                f"fallback would overwrite an existing assignment: {part_id}"
            )
        if assignment.get("status") not in {"approved", "auto"}:
            raise CompleteMaterialPlanError(
                f"fallback assignment for {part_id} must be approved or auto"
            )
        _unit(assignment.get("confidence"), f"fallback {part_id}.confidence")
        _string(assignment.get("material_id"), f"fallback {part_id}.material_id")
        _string(assignment.get("semantic"), f"fallback {part_id}.semantic")
        _sequence(
            assignment.get("evidence_views", []), f"fallback {part_id}.evidence_views"
        )
        assignments_by_part[part_id] = assignment
        fallback_parts.append(part_id)

    registry_part_set = set(registry_part_ids)
    assignment_part_set = set(assignments_by_part)
    if assignment_part_set != registry_part_set:
        missing = sorted(registry_part_set - assignment_part_set)
        unexpected = sorted(assignment_part_set - registry_part_set)
        raise CompleteMaterialPlanError(
            f"complete plan does not exactly cover registry; missing={missing}, "
            f"unexpected={unexpected}"
        )

    assignments = [
        assignments_by_part[part_id] for part_id in sorted(registry_part_ids)
    ]
    plan = {
        "schema_version": MATERIAL_PLAN_SCHEMA_VERSION,
        "assignments": assignments,
    }
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "asset_sha256": registry_asset_sha256,
        "inputs": {
            "base_plan_sha256": _sha256_document(base_plan),
            "registry_sha256": _sha256_document(registry),
            "mvinverse_evidence_sha256": _sha256_document(mvinverse_evidence),
            "policy_sha256": _sha256_document(policy),
        },
        "summary": {
            "registry_part_count": len(registry_part_ids),
            "base_assignment_count": len(
                _sequence(base_plan.get("assignments"), "base_plan.assignments")
            ),
            "parameterized_part_count": len(parameterized_parts),
            "fallback_assignment_count": len(fallback_parts),
            "output_assignment_count": len(assignments),
            "face_subset_count": sum(
                len(assignment.get("face_subsets", [])) for assignment in assignments
            ),
            "all_registry_parts_assigned": True,
        },
        "parameterized_groups": sorted(group_audits, key=lambda item: item["group_id"]),
        "skipped_parameterized_groups": sorted(
            skipped_parameterized_groups,
            key=lambda item: (item["group_id"], item["material_id"]),
        ),
        "mvinverse_validation": mvinverse_validation,
        "parameterized_parts": sorted(parameterized_parts),
        "fallback_parts": sorted(fallback_parts),
        "output_plan_sha256": _sha256_document(plan),
    }
    return plan, report


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CompleteMaterialPlanError(f"unable to read JSON {path}: {exc}") from exc
    return _mapping(value, str(path))


def _write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve()
    if path.exists():
        raise CompleteMaterialPlanError(
            f"refusing to overwrite existing output: {path}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-plan", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--mvinverse-evidence", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output-plan", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    base_plan = _read_json(args.base_plan.expanduser().resolve())
    registry = _read_json(args.registry.expanduser().resolve())
    evidence = _read_json(args.mvinverse_evidence.expanduser().resolve())
    policy = _read_json(args.policy.expanduser().resolve())

    asset_path = (
        Path(_string(registry.get("asset_usd"), "registry.asset_usd"))
        .expanduser()
        .resolve()
    )
    if not asset_path.is_file():
        raise CompleteMaterialPlanError(f"registry asset does not exist: {asset_path}")
    actual_asset_sha256 = _sha256_file(asset_path)
    if actual_asset_sha256 != registry.get("asset_sha256"):
        raise CompleteMaterialPlanError(
            "registry asset SHA-256 does not match the current source asset"
        )

    plan, report = build_complete_material_plan(
        base_plan=base_plan,
        registry=registry,
        mvinverse_evidence=evidence,
        policy=policy,
    )
    report = dict(report)
    report["paths"] = {
        "asset_usd": str(asset_path),
        "base_plan": str(args.base_plan.expanduser().resolve()),
        "registry": str(args.registry.expanduser().resolve()),
        "mvinverse_evidence": str(args.mvinverse_evidence.expanduser().resolve()),
        "policy": str(args.policy.expanduser().resolve()),
        "output_plan": str(args.output_plan.expanduser().resolve()),
    }
    _write_json_new(args.output_plan, plan)
    _write_json_new(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
