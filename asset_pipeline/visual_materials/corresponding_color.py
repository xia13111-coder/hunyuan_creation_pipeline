"""Validate the identity-preserving actual-CAD colour-calibration stage."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qwen_material_pipeline.materials.corresponding_material_color import (
    CorrespondingMaterialColorError,
    reviewed_corresponding_material_partitions,
    source_accepted_color_selection_tiers,
)

from .config import canonical_sha256, read_object
from .references import sha256_file


WORKFLOW_SCHEMA_VERSION = "qwen-corresponding-material-color-workflow/v3"
SELECTION_AUDIT_SCHEMA_VERSION = (
    "qwen-corresponding-material-color-render-selection-audit/v2"
)


@dataclass(frozen=True)
class CorrespondingColorResult:
    manifest: dict[str, Any]
    selected_plan: dict[str, Any]
    selection_audit: dict[str, Any]
    quality_report: dict[str, Any]
    applied_count: int


def _assignments(document: Mapping[str, Any], label: str) -> dict[str, dict[str, Any]]:
    raw = document.get("assignments")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or not raw:
        raise RuntimeError(f"{label} has no material assignments")
    output: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(raw):
        part_id = item.get("part_id") if isinstance(item, Mapping) else None
        if not isinstance(part_id, str) or not part_id or part_id in output:
            raise RuntimeError(f"{label} has invalid Part-ID at index {index}")
        output[part_id] = dict(item)
    return output


def _sealed_document(document: Mapping[str, Any], label: str) -> None:
    integrity = document.get("integrity")
    unsigned = dict(document)
    unsigned.pop("integrity", None)
    if not isinstance(integrity, Mapping) or integrity.get(
        "document_sha256"
    ) != canonical_sha256(unsigned):
        raise RuntimeError(f"{label} failed its integrity seal")


def _bound_file(
    record: object,
    *,
    expected: Path,
    label: str,
) -> None:
    if not isinstance(record, Mapping):
        raise RuntimeError(f"colour workflow manifest lacks {label}")
    resolved = expected.expanduser().resolve(strict=True)
    if record.get("path") != str(resolved):
        raise RuntimeError(f"colour workflow {label} path is not the sealed output")
    if record.get("sha256") != sha256_file(resolved):
        raise RuntimeError(f"colour workflow {label} hash changed")


def _selection_tiers(
    qwen_choices: Mapping[str, Any], source_plan: Mapping[str, Any]
) -> dict[str, str]:
    try:
        tiers, _ = source_accepted_color_selection_tiers(
            source_plan=source_plan,
            qwen_choices=qwen_choices,
        )
    except CorrespondingMaterialColorError as exc:
        raise RuntimeError(str(exc)) from exc
    return tiers


def corresponding_material_part_ids(
    qwen_choices: Mapping[str, Any],
    source_plan: Mapping[str, Any],
) -> tuple[str, ...]:
    """Return the sealed Part IDs eligible for the second-stage colour pass."""

    eligible, _ = corresponding_material_eligibility(
        qwen_choices=qwen_choices,
        source_plan=source_plan,
    )
    return eligible


def corresponding_material_eligibility(
    qwen_choices: Mapping[str, Any],
    source_plan: Mapping[str, Any],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return tunable and preset-preserved corresponding Part IDs."""

    try:
        return reviewed_corresponding_material_partitions(
            source_plan=source_plan,
            qwen_choices=qwen_choices,
        )
    except CorrespondingMaterialColorError as exc:
        raise RuntimeError(str(exc)) from exc


def validate_corresponding_color_result(
    *,
    manifest_path: Path,
    source_plan_path: Path,
    qwen_choices_path: Path,
    part_id_evidence_path: Path,
    spatial_mapping_report_path: Path,
    asset_usd_path: Path,
    catalog_path: Path,
    registry_path: Path,
    material_root_path: Path,
    view_specs_path: Path,
    reference_manifest_path: Path,
    isaac_python_path: Path,
    selected_plan_path: Path,
    selection_audit_path: Path,
    look_usd_path: Path,
    apply_report_path: Path,
    registry_output_path: Path,
    rendered_registry_path: Path,
    quality_report_path: Path,
) -> CorrespondingColorResult:
    """Replay the stage boundary before downstream publication can use it."""

    manifest = read_object(manifest_path, "corresponding colour workflow manifest")
    if (
        manifest.get("schema_version") != WORKFLOW_SCHEMA_VERSION
        or manifest.get("workflow_state") != "COMPLETE"
    ):
        raise RuntimeError("corresponding colour workflow did not complete")
    policy = manifest.get("policy")
    if (
        not isinstance(policy, Mapping)
        or policy.get("material_identity_mutation_allowed") is not False
        or policy.get("same_component_shares_material_and_colour") is not True
        or policy.get("actual_cad_render_selection") is not True
        or policy.get("local_part_scope_quality_gate") is not True
        or policy.get("optimization_mode") != "adaptive_per_scope"
    ):
        raise RuntimeError(
            "corresponding colour workflow policy is not production-safe"
        )

    expected_inputs = {
        "source_plan": source_plan_path,
        "qwen_choices": qwen_choices_path,
        "part_id_evidence": part_id_evidence_path,
        "spatial_mapping_report": spatial_mapping_report_path,
        "asset_usd": asset_usd_path,
        "catalog": catalog_path,
        "registry": registry_path,
        "view_specs": view_specs_path,
        "reference_manifest": reference_manifest_path,
        "isaac_python": isaac_python_path,
    }
    inputs = manifest.get("inputs")
    if not isinstance(inputs, Mapping):
        raise RuntimeError("corresponding colour workflow has no sealed inputs")
    for label, path in expected_inputs.items():
        _bound_file(inputs.get(label), expected=path, label=f"input {label}")
    material_root = material_root_path.expanduser().resolve(strict=True)
    material_root_record = inputs.get("material_root")
    if not isinstance(material_root_record, Mapping) or material_root_record.get(
        "path"
    ) != str(material_root):
        raise RuntimeError("corresponding colour workflow material root changed")

    outputs = manifest.get("outputs")
    if not isinstance(outputs, Mapping):
        raise RuntimeError("corresponding colour workflow has no sealed outputs")
    for label, path in {
        "selected_plan": selected_plan_path,
        "selection_audit": selection_audit_path,
        "asset": look_usd_path,
        "apply_report": apply_report_path,
        "registry": registry_output_path,
        "rendered_registry": rendered_registry_path,
        "quality_report": quality_report_path,
    }.items():
        _bound_file(outputs.get(label), expected=path, label=f"output {label}")

    source_plan = read_object(source_plan_path, "identity-fixed material plan")
    selected_plan = read_object(selected_plan_path, "colour-selected material plan")
    source_assignments = _assignments(source_plan, "identity-fixed material plan")
    selected_assignments = _assignments(selected_plan, "colour-selected material plan")
    if set(source_assignments) != set(selected_assignments):
        raise RuntimeError("colour calibration changed the Part-ID exact cover")
    qwen_choices = read_object(qwen_choices_path, "material-identity Qwen choices")
    tiers = _selection_tiers(qwen_choices, source_plan)
    eligible_corresponding_ids, preserved_corresponding_ids = (
        corresponding_material_eligibility(qwen_choices, source_plan)
    )
    eligible_corresponding = set(eligible_corresponding_ids)
    preserved_corresponding = set(preserved_corresponding_ids)
    if not set(tiers) <= set(source_assignments):
        raise RuntimeError("material selections are outside the identity plan")
    parameterized: set[str] = set()
    for part_id, source_assignment in source_assignments.items():
        selected_assignment = selected_assignments[part_id]
        if selected_assignment.get("material_id") != source_assignment.get(
            "material_id"
        ):
            raise RuntimeError(f"colour calibration changed MDL identity for {part_id}")
        if source_assignment.get("parameters") not in (None, {}):
            raise RuntimeError(
                f"identity-fixed source plan already has parameters for {part_id}"
            )
        parameters = selected_assignment.get("parameters")
        if parameters not in (None, {}):
            if not isinstance(parameters, Mapping):
                raise RuntimeError(f"colour parameters for {part_id} are invalid")
            if part_id not in eligible_corresponding:
                raise RuntimeError(
                    "colour calibration parameterized a non-corresponding or "
                    "unreviewed-interface Part-ID: "
                    f"{part_id}"
                )
            parameterized.add(part_id)
        elif part_id in eligible_corresponding:
            raise RuntimeError(
                f"corresponding material {part_id} was not colour calibrated"
            )
        elif (
            part_id in preserved_corresponding
            and selected_assignment != source_assignment
        ):
            raise RuntimeError(
                f"unsupported corresponding material {part_id} was not preserved "
                "unchanged"
            )

    audit = read_object(selection_audit_path, "colour render-selection audit")
    if audit.get("schema_version") != SELECTION_AUDIT_SCHEMA_VERSION:
        raise RuntimeError("colour render-selection audit has an unsupported schema")
    _sealed_document(audit, "colour render-selection audit")
    if (
        audit.get("status") != "PASS"
        or audit.get("source_plan_sha256") != canonical_sha256(source_plan)
        or audit.get("output_plan_sha256") != canonical_sha256(selected_plan)
    ):
        raise RuntimeError("colour render-selection audit is not bound to the plans")
    summary = audit.get("summary")
    if (
        not isinstance(summary, Mapping)
        or summary.get("parameterized_part_count") != len(parameterized)
        or summary.get("material_identity_change_count") != 0
        or summary.get("local_quality_gate_status") != "PASS"
    ):
        raise RuntimeError("colour render-selection summary is inconsistent")
    selections = audit.get("selections")
    if (
        not isinstance(selections, list)
        or len(selections) != summary.get("colour_scope_count")
        or any(
            not isinstance(row, Mapping)
            or not isinstance(row.get("local_quality_gate"), Mapping)
            or row["local_quality_gate"].get("status") != "PASS"
            for row in selections
        )
    ):
        raise RuntimeError("colour render-selection local quality gate is incomplete")

    apply_report = read_object(apply_report_path, "colour-selected apply report")
    applied_count = apply_report.get("applied_count")
    if (
        isinstance(applied_count, bool)
        or not isinstance(applied_count, int)
        or applied_count != len(selected_assignments)
    ):
        raise RuntimeError("colour-selected apply report changed exact coverage")

    reference_manifest = read_object(
        reference_manifest_path, "colour-calibration reference manifest"
    )
    raw_source_views = reference_manifest.get("source_views")
    expected_views = (
        {
            str(row.get("id"))
            for row in raw_source_views
            if isinstance(row, Mapping) and isinstance(row.get("id"), str)
        }
        if isinstance(raw_source_views, list)
        else set()
    )
    quality = read_object(quality_report_path, "colour-calibration quality report")
    raw_quality_views = quality.get("views")
    actual_views = (
        {
            str(row.get("reference_view_id"))
            for row in raw_quality_views
            if isinstance(row, Mapping)
            and isinstance(row.get("reference_view_id"), str)
        }
        if isinstance(raw_quality_views, list)
        else set()
    )
    aggregate = quality.get("aggregate")
    if (
        not expected_views
        or actual_views != expected_views
        or not isinstance(aggregate, Mapping)
        or aggregate.get("reference_view_count") != len(expected_views)
        or aggregate.get("comparable_view_count") != len(expected_views)
        or manifest.get("quality_status") != aggregate.get("status")
    ):
        raise RuntimeError("colour calibration did not compare every reference view")

    return CorrespondingColorResult(
        manifest=manifest,
        selected_plan=selected_plan,
        selection_audit=audit,
        quality_report=quality,
        applied_count=applied_count,
    )


def rebind_part_id_audit_for_corresponding_color(
    *,
    source_audit: Mapping[str, Any],
    source_plan: Mapping[str, Any],
    final_plan: Mapping[str, Any],
    selection_audit: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind the Part-ID publication audit to the calibrated same-ID plan."""

    _sealed_document(source_audit, "source Part-ID material audit")
    _sealed_document(selection_audit, "colour render-selection audit")
    source_hash = canonical_sha256(source_plan)
    final_hash = canonical_sha256(final_plan)
    if (
        source_audit.get("output_plan_sha256") != source_hash
        or selection_audit.get("source_plan_sha256") != source_hash
        or selection_audit.get("output_plan_sha256") != final_hash
    ):
        raise RuntimeError("colour calibration audits are not bound to the plans")
    source_assignments = _assignments(source_plan, "identity-fixed material plan")
    final_assignments = _assignments(final_plan, "colour-selected material plan")
    if set(source_assignments) != set(final_assignments):
        raise RuntimeError("colour calibration changed Part-ID coverage")

    raw_rows = source_audit.get("parts")
    if not isinstance(raw_rows, list):
        raise RuntimeError("source Part-ID material audit has no rows")
    rows = {
        str(row.get("part_id")): copy.deepcopy(dict(row))
        for row in raw_rows
        if isinstance(row, Mapping) and isinstance(row.get("part_id"), str)
    }
    if len(rows) != len(raw_rows) or set(rows) != set(source_assignments):
        raise RuntimeError("source Part-ID material audit does not exactly cover plan")

    calibrated_ids: set[str] = set()
    selections = selection_audit.get("selections")
    if not isinstance(selections, list):
        raise RuntimeError("colour render-selection audit has no selections")
    for selection in selections:
        if not isinstance(selection, Mapping):
            raise RuntimeError("colour render-selection row is invalid")
        scope_id = selection.get("scope_id")
        candidate_id = selection.get("selected_candidate_id")
        member_ids = selection.get("member_part_ids")
        if (
            not isinstance(scope_id, str)
            or not isinstance(candidate_id, str)
            or not isinstance(member_ids, list)
            or not member_ids
        ):
            raise RuntimeError("colour render-selection row is incomplete")
        for raw_part_id in member_ids:
            if not isinstance(raw_part_id, str) or raw_part_id in calibrated_ids:
                raise RuntimeError("colour scopes overlap or contain invalid Part IDs")
            part_id = raw_part_id
            source_assignment = source_assignments.get(part_id)
            final_assignment = final_assignments.get(part_id)
            if source_assignment is None or final_assignment is None:
                raise RuntimeError("colour scope is outside the Part-ID plan")
            if source_assignment.get("material_id") != final_assignment.get(
                "material_id"
            ) or final_assignment.get("parameters") in (None, {}):
                raise RuntimeError("colour scope changed identity or lacks parameters")
            row = rows[part_id]
            if row.get("material_id") != source_assignment.get("material_id"):
                raise RuntimeError("Part-ID audit material differs from source plan")
            row["mdl_color_parameterization"] = {
                "status": "render_calibrated_corresponding_material",
                "material_id": final_assignment.get("material_id"),
                "selected_candidate_id": candidate_id,
                "parameters_applied": True,
                "scope_id": scope_id,
            }
            row["corresponding_material_color_calibration"] = {
                "scope_id": scope_id,
                "selected_candidate_id": candidate_id,
                "material_id_unchanged": True,
                "parameters_sha256": canonical_sha256(
                    final_assignment.get("parameters")
                ),
                "selection_audit_sha256": canonical_sha256(selection_audit),
            }
            calibrated_ids.add(part_id)

    final_parameterized = {
        part_id
        for part_id, assignment in final_assignments.items()
        if assignment.get("parameters") not in (None, {})
    }
    if calibrated_ids != final_parameterized:
        raise RuntimeError("colour calibration audit does not cover all parameters")

    output = copy.deepcopy(dict(source_audit))
    output.pop("integrity", None)
    output["parts"] = [rows[str(row["part_id"])] for row in raw_rows]
    output["output_plan_sha256"] = final_hash
    summary = output.get("summary")
    if not isinstance(summary, dict):
        raise RuntimeError("source Part-ID material audit has no summary")
    summary["color_parameterized_count"] = len(calibrated_ids)
    output["corresponding_material_color_calibration"] = {
        "schema_version": "asset-pipeline-corresponding-color-audit-binding/v1",
        "source_plan_sha256": source_hash,
        "final_plan_sha256": final_hash,
        "selection_audit_sha256": canonical_sha256(selection_audit),
        "parameterized_part_count": len(calibrated_ids),
        "material_identity_change_count": 0,
    }
    output["integrity"] = {"document_sha256": canonical_sha256(output)}
    return output


__all__ = [
    "CorrespondingColorResult",
    "corresponding_material_eligibility",
    "corresponding_material_part_ids",
    "rebind_part_id_audit_for_corresponding_color",
    "validate_corresponding_color_result",
]
