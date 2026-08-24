"""Validate the identity-preserving actual-CAD colour-calibration stage."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

from qwen_material_pipeline.materials.corresponding_material_color import (
    CorrespondingMaterialColorError,
    reviewed_corresponding_material_partitions,
    source_accepted_color_selection_tiers,
)
from qwen_material_pipeline.materials.corresponding_material_color_selection import (
    MINIMUM_COMPONENT_MEMBER_APPEARANCE_SCORE,
    MINIMUM_SCORABLE_COMPONENT_MEMBER_PIXELS,
    MINIMUM_SCORABLE_SCOPE_PIXELS,
    MINIMUM_SCOPE_APPEARANCE_SCORE,
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


def _validated_local_quality_audit(
    audit: Mapping[str, Any],
    *,
    expected_parameterized_part_ids: set[str],
) -> tuple[list[str], list[str]]:
    """Replay fixed local-quality policy and return the scopes needing review."""

    summary = audit.get("summary")
    selections = audit.get("selections")
    if not isinstance(summary, Mapping) or not isinstance(selections, list):
        raise RuntimeError("colour render-selection local quality audit is missing")
    if (
        len(selections) != summary.get("colour_scope_count")
        or summary.get("minimum_scorable_scope_pixels")
        != MINIMUM_SCORABLE_SCOPE_PIXELS
        or summary.get("minimum_scope_appearance_score")
        != MINIMUM_SCOPE_APPEARANCE_SCORE
        or summary.get("minimum_scorable_component_member_pixels")
        != MINIMUM_SCORABLE_COMPONENT_MEMBER_PIXELS
        or summary.get("minimum_component_member_appearance_score")
        != MINIMUM_COMPONENT_MEMBER_APPEARANCE_SCORE
    ):
        raise RuntimeError("colour render-selection quality policy changed")

    seen_scopes: set[str] = set()
    seen_parts: set[str] = set()
    review_scope_ids: list[str] = []
    review_part_ids: set[str] = set()
    for raw_selection in selections:
        if not isinstance(raw_selection, Mapping):
            raise RuntimeError("colour render-selection row is invalid")
        scope_id = raw_selection.get("scope_id")
        member_ids = raw_selection.get("member_part_ids")
        gate = raw_selection.get("local_quality_gate")
        if (
            not isinstance(scope_id, str)
            or not scope_id
            or scope_id in seen_scopes
            or not isinstance(member_ids, list)
            or not member_ids
            or any(not isinstance(value, str) or not value for value in member_ids)
            or len(set(member_ids)) != len(member_ids)
            or seen_parts.intersection(member_ids)
            or not isinstance(gate, Mapping)
        ):
            raise RuntimeError("colour render-selection scope contract is invalid")
        seen_scopes.add(scope_id)
        seen_parts.update(member_ids)

        pixels = gate.get("comparison_pixel_count")
        appearance = gate.get("appearance_score")
        if (
            isinstance(pixels, bool)
            or not isinstance(pixels, int)
            or pixels < 0
            or isinstance(appearance, bool)
            or not isinstance(appearance, (int, float))
            or not math.isfinite(float(appearance))
            or gate.get("scope_evaluated")
            is not (pixels >= MINIMUM_SCORABLE_SCOPE_PIXELS)
            or gate.get("minimum_scorable_scope_pixels")
            != MINIMUM_SCORABLE_SCOPE_PIXELS
            or gate.get("minimum_scope_appearance_score")
            != MINIMUM_SCOPE_APPEARANCE_SCORE
            or gate.get("minimum_scorable_component_member_pixels")
            != MINIMUM_SCORABLE_COMPONENT_MEMBER_PIXELS
            or gate.get("minimum_component_member_appearance_score")
            != MINIMUM_COMPONENT_MEMBER_APPEARANCE_SCORE
        ):
            raise RuntimeError("colour render-selection scope quality is malformed")

        failures: list[str] = []
        if (
            pixels >= MINIMUM_SCORABLE_SCOPE_PIXELS
            and float(appearance) < MINIMUM_SCOPE_APPEARANCE_SCORE
        ):
            failures.append("scope_appearance_below_floor")
        raw_members = gate.get("member_scores")
        if not isinstance(raw_members, list):
            raise RuntimeError("colour render-selection member quality is malformed")
        validated_member_ids: list[str] = []
        for raw_member in raw_members:
            if not isinstance(raw_member, Mapping):
                raise RuntimeError(
                    "colour render-selection member quality is malformed"
                )
            part_id = raw_member.get("part_id")
            member_pixels = raw_member.get("comparison_pixel_count")
            member_appearance = raw_member.get("appearance_score")
            if (
                not isinstance(part_id, str)
                or part_id in validated_member_ids
                or isinstance(member_pixels, bool)
                or not isinstance(member_pixels, int)
                or member_pixels < 0
                or isinstance(member_appearance, bool)
                or not isinstance(member_appearance, (int, float))
                or not math.isfinite(float(member_appearance))
            ):
                raise RuntimeError(
                    "colour render-selection member quality is malformed"
                )
            evaluated = member_pixels >= MINIMUM_SCORABLE_COMPONENT_MEMBER_PIXELS
            passed = (
                not evaluated
                or float(member_appearance)
                >= MINIMUM_COMPONENT_MEMBER_APPEARANCE_SCORE
            )
            if (
                raw_member.get("evaluated") is not evaluated
                or raw_member.get("passed") is not passed
            ):
                raise RuntimeError(
                    "colour render-selection member quality decision changed"
                )
            if not passed:
                failures.append(f"component_member_below_floor:{part_id}")
            validated_member_ids.append(part_id)
        if len(member_ids) > 1 and set(validated_member_ids) != set(member_ids):
            raise RuntimeError(
                "component local quality does not exactly cover its members"
            )
        expected_status = "FAIL" if failures else "PASS"
        if (
            gate.get("status") != expected_status
            or gate.get("failure_reasons") != failures
            or raw_selection.get("selected_appearance_score") != appearance
        ):
            raise RuntimeError("colour render-selection quality decision changed")
        if expected_status == "FAIL":
            review_scope_ids.append(scope_id)
            review_part_ids.update(member_ids)

    if seen_parts != expected_parameterized_part_ids:
        raise RuntimeError(
            "colour render-selection scopes do not cover all parameterized parts"
        )
    review_scope_ids.sort()
    sorted_review_parts = sorted(review_part_ids)
    expected_local_status = "REVIEW" if review_scope_ids else "PASS"
    if (
        audit.get("status") != expected_local_status
        or summary.get("local_quality_gate_status") != expected_local_status
        or summary.get("local_quality_review_scope_count") != len(review_scope_ids)
        or summary.get("local_quality_review_part_count") != len(sorted_review_parts)
        or summary.get("local_quality_review_scope_ids") != review_scope_ids
        or summary.get("local_quality_review_part_ids") != sorted_review_parts
        or summary.get("local_quality_pass_scope_count")
        != len(selections) - len(review_scope_ids)
        or summary.get("quality_rejection_behavior")
        != "retain_best_rendered_candidate_and_continue_with_review"
    ):
        raise RuntimeError("colour render-selection review status is inconsistent")
    return review_scope_ids, sorted_review_parts


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
        or policy.get("local_quality_rejection_behavior")
        != "retain_best_rendered_candidate_and_continue_with_review"
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
        audit.get("status") not in {"PASS", "REVIEW"}
        or audit.get("source_plan_sha256") != canonical_sha256(source_plan)
        or audit.get("output_plan_sha256") != canonical_sha256(selected_plan)
    ):
        raise RuntimeError("colour render-selection audit is not bound to the plans")
    summary = audit.get("summary")
    if (
        not isinstance(summary, Mapping)
        or summary.get("parameterized_part_count") != len(parameterized)
        or summary.get("material_identity_change_count") != 0
        or summary.get("local_quality_gate_status") not in {"PASS", "REVIEW"}
        or summary.get("quality_rejection_behavior")
        != "retain_best_rendered_candidate_and_continue_with_review"
    ):
        raise RuntimeError("colour render-selection summary is inconsistent")
    review_scope_ids, _review_part_ids = _validated_local_quality_audit(
        audit,
        expected_parameterized_part_ids=parameterized,
    )
    expected_local_status = "REVIEW" if review_scope_ids else "PASS"
    if manifest.get("local_quality_status") != expected_local_status:
        raise RuntimeError("colour render-selection review status is inconsistent")

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
    review_scope_ids: list[str] = []
    review_part_ids: set[str] = set()
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
        local_quality_gate = selection.get("local_quality_gate")
        if (
            not isinstance(local_quality_gate, Mapping)
            or local_quality_gate.get("status") not in {"PASS", "FAIL"}
        ):
            raise RuntimeError("colour render-selection row lacks local quality")
        local_quality_status = str(local_quality_gate["status"])
        if local_quality_status == "FAIL":
            review_scope_ids.append(scope_id)
            review_part_ids.update(str(value) for value in member_ids)
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
                "local_quality_gate_status": local_quality_status,
                "best_available_result_retained": local_quality_status == "FAIL",
            }
            row["corresponding_material_color_calibration"] = {
                "scope_id": scope_id,
                "selected_candidate_id": candidate_id,
                "material_id_unchanged": True,
                "parameters_sha256": canonical_sha256(
                    final_assignment.get("parameters")
                ),
                "selection_audit_sha256": canonical_sha256(selection_audit),
                "local_quality_gate": copy.deepcopy(dict(local_quality_gate)),
            }
            calibrated_ids.add(part_id)

    final_parameterized = {
        part_id
        for part_id, assignment in final_assignments.items()
        if assignment.get("parameters") not in (None, {})
    }
    if calibrated_ids != final_parameterized:
        raise RuntimeError("colour calibration audit does not cover all parameters")
    validated_review_scopes, validated_review_parts = (
        _validated_local_quality_audit(
            selection_audit,
            expected_parameterized_part_ids=final_parameterized,
        )
    )
    if (
        sorted(review_scope_ids) != validated_review_scopes
        or sorted(review_part_ids) != validated_review_parts
    ):
        raise RuntimeError("colour review binding differs from selection audit")

    output = copy.deepcopy(dict(source_audit))
    output.pop("integrity", None)
    output["parts"] = [rows[str(row["part_id"])] for row in raw_rows]
    output["output_plan_sha256"] = final_hash
    summary = output.get("summary")
    if not isinstance(summary, dict):
        raise RuntimeError("source Part-ID material audit has no summary")
    summary["color_parameterized_count"] = len(calibrated_ids)
    summary["color_review_required_scope_count"] = len(review_scope_ids)
    summary["color_review_required_part_count"] = len(review_part_ids)
    output["corresponding_material_color_calibration"] = {
        "schema_version": "asset-pipeline-corresponding-color-audit-binding/v1",
        "source_plan_sha256": source_hash,
        "final_plan_sha256": final_hash,
        "selection_audit_sha256": canonical_sha256(selection_audit),
        "parameterized_part_count": len(calibrated_ids),
        "material_identity_change_count": 0,
        "local_quality_status": "REVIEW" if review_scope_ids else "PASS",
        "review_required_scope_ids": sorted(review_scope_ids),
        "review_required_part_ids": sorted(review_part_ids),
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
