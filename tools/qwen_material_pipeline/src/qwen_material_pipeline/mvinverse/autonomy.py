"""Fail-closed helpers for consuming MVInverse evidence without a human step.

MVInverse predicts image-space PBR observations, not a USD material.  This
module only performs two narrow, deterministic joins:

* expand independently evaluated palette-group material choices to the
  existing per-part view-evidence contract; and
* tune an already-approved, explicitly supported NVIDIA MDL export when
  multi-view PBR evidence passes the MVInverse evidence policy.

It never promotes a review/unknown assignment and never invents a fallback
material for a part that lacks evidence.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from qwen_material_pipeline.mvinverse.evidence import (
    MVInverseEvidenceError,
    validate_mvinverse_evidence,
)
from qwen_material_pipeline.materials.tuning import (
    tune_selected_material_from_mvinverse,
    tuning_profile_for_material,
)


VIEW_EVIDENCE_SCHEMA_VERSION = "qwen-material-view-evidence/v1"
AUTONOMY_SCHEMA_VERSION = "qwen-mvinverse-autonomous-plan/v1"
MATERIAL_PLAN_SCHEMA_VERSION = "1.0"
MVINVERSE_EVIDENCE_SCHEMA_VERSION = "qwen-mvinverse-pbr-evidence/v1"

STEEL_PAINTED_MODULE_PREFIX = "mdl:Miscellaneous/Paint_Matte.mdl#"
GENERIC_STEEL_PAINTED = STEEL_PAINTED_MODULE_PREFIX + "Paint_Matte"


class MVInverseAutonomyError(ValueError):
    """Raised when evidence cannot be joined without guessing."""


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MVInverseAutonomyError(f"{label} must be an object")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise MVInverseAutonomyError(f"{label} must be an array")
    return value


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MVInverseAutonomyError(f"{label} must be a non-empty string")
    return value.strip()


def _unit(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise MVInverseAutonomyError(f"{label} must be a finite number from 0 to 1")
    return float(value)


def _part_group_map(batches: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    result: dict[str, str] = {}
    for batch_index, raw_batch in enumerate(batches):
        batch = _mapping(raw_batch, f"batches[{batch_index}]")
        mappings = _sequence(batch.get("mappings"), f"batches[{batch_index}].mappings")
        for mapping_index, raw_mapping in enumerate(mappings):
            mapping = _mapping(
                raw_mapping,
                f"batches[{batch_index}].mappings[{mapping_index}]",
            )
            part_id = _string(mapping.get("part_id"), "mapping.part_id")
            if part_id in result:
                raise MVInverseAutonomyError(
                    f"duplicate part-to-group mapping for {part_id}"
                )
            status = mapping.get("status")
            group_id = mapping.get("group_id")
            if status == "unknown" or group_id is None:
                continue
            result[part_id] = _string(group_id, f"mapping {part_id}.group_id")
    return result


def build_part_view_evidence(
    *,
    batches: Sequence[Mapping[str, Any]],
    group_view_choices: Mapping[str, Sequence[Mapping[str, Any]]],
    mapping_votes: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Expand real per-view group choices into strict per-part evidence.

    ``group_view_choices`` must contain choices actually made from separate
    original-image crops.  Forward/reverse prompts over one crop must not be
    supplied as two views.

    When ``mapping_votes`` is supplied, a group-level material choice counts
    for a part only when the independent Qwen call in that same source view
    mapped the part to that canonical group.  This prevents a material choice
    for (for example) the white group from being broadcast to every part whose
    primary-view mapping happened to be white.
    """

    part_groups = _part_group_map(batches)
    canonical_choices: dict[str, list[dict[str, Any]]] = {}
    for raw_group_id, raw_choices in group_view_choices.items():
        group_id = _string(raw_group_id, "group_view_choices group_id")
        choices = _sequence(raw_choices, f"group_view_choices[{group_id}]")
        seen_views: set[str] = set()
        canonical: list[dict[str, Any]] = []
        for index, raw_choice in enumerate(choices):
            choice = _mapping(raw_choice, f"group_view_choices[{group_id}][{index}]")
            view_id = _string(choice.get("view_id"), f"choice {group_id}.view_id")
            if view_id in seen_views:
                raise MVInverseAutonomyError(
                    f"duplicate source view for palette group {group_id}: {view_id}"
                )
            seen_views.add(view_id)
            material_id = _string(
                choice.get("material_id"), f"choice {group_id}/{view_id}.material_id"
            )
            confidence = _unit(
                choice.get("confidence"), f"choice {group_id}/{view_id}.confidence"
            )
            margin = choice.get("candidate_margin")
            canonical.append(
                {
                    "view_id": view_id,
                    "material_id": material_id,
                    "confidence": confidence,
                    "candidate_margin": (
                        _unit(
                            margin,
                            f"choice {group_id}/{view_id}.candidate_margin",
                        )
                        if margin is not None
                        else None
                    ),
                }
            )
        canonical_choices[group_id] = sorted(
            canonical, key=lambda item: item["view_id"]
        )

    predictions: list[dict[str, Any]] = []
    if mapping_votes is None:
        for part_id, group_id in sorted(part_groups.items()):
            for choice in canonical_choices.get(group_id, []):
                predictions.append({"part_id": part_id, **choice})
    else:
        choices_by_group_view = {
            (group_id, choice["view_id"]): choice
            for group_id, choices in canonical_choices.items()
            for choice in choices
        }
        seen_part_views: set[tuple[str, str]] = set()
        for index, raw_vote in enumerate(
            _sequence(mapping_votes, "mapping_votes")
        ):
            vote = _mapping(raw_vote, f"mapping_votes[{index}]")
            part_id = _string(
                vote.get("part_id"), f"mapping_votes[{index}].part_id"
            )
            view_id = _string(
                vote.get("view_id"), f"mapping_votes[{index}].view_id"
            )
            identity = (part_id, view_id)
            if identity in seen_part_views:
                raise MVInverseAutonomyError(
                    f"duplicate part/view mapping vote: {part_id}/{view_id}"
                )
            seen_part_views.add(identity)
            if part_id not in part_groups:
                continue
            status = vote.get("status")
            if status == "unknown":
                continue
            if status not in {"matched", "review"}:
                raise MVInverseAutonomyError(
                    f"mapping vote has invalid status: {part_id}/{view_id}"
                )
            group_id = vote.get("canonical_group_id")
            if group_id is None:
                continue
            group_id = _string(
                group_id,
                f"mapping_votes[{index}].canonical_group_id",
            )
            if group_id != part_groups[part_id]:
                continue
            choice = choices_by_group_view.get((group_id, view_id))
            if choice is None:
                continue
            vote_confidence = _unit(
                vote.get("confidence"),
                f"mapping_votes[{index}].confidence",
            )
            predictions.append(
                {
                    "part_id": part_id,
                    **choice,
                    "confidence": min(choice["confidence"], vote_confidence),
                }
            )
    return {
        "schema_version": VIEW_EVIDENCE_SCHEMA_VERSION,
        "predictions": sorted(
            predictions, key=lambda item: (item["part_id"], item["view_id"])
        ),
    }


def _evidence_groups(document: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    if document.get("schema_version") != MVINVERSE_EVIDENCE_SCHEMA_VERSION:
        raise MVInverseAutonomyError(
            "MVInverse evidence has an unsupported schema_version"
        )
    groups = _sequence(document.get("groups"), "mvinverse_evidence.groups")
    result: dict[str, Mapping[str, Any]] = {}
    for index, raw_group in enumerate(groups):
        group = _mapping(raw_group, f"mvinverse_evidence.groups[{index}]")
        group_id = _string(group.get("group_id"), f"evidence group[{index}].group_id")
        if group_id in result:
            raise MVInverseAutonomyError(
                f"duplicate MVInverse evidence group: {group_id}"
            )
        result[group_id] = group
    return result


def _palette_groups(document: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    groups = _sequence(document.get("groups"), "palette.groups")
    result: dict[str, Mapping[str, Any]] = {}
    for index, raw_group in enumerate(groups):
        group = _mapping(raw_group, f"palette.groups[{index}]")
        group_id = _string(group.get("group_id"), f"palette group[{index}].group_id")
        if group_id in result:
            raise MVInverseAutonomyError(f"duplicate palette group: {group_id}")
        result[group_id] = group
    return result


def parameterize_auto_material_plan(
    *,
    auto_material_plan: Mapping[str, Any],
    batches: Sequence[Mapping[str, Any]],
    palette: Mapping[str, Any],
    mvinverse_evidence: Mapping[str, Any],
    allowed_material_ids: Iterable[str],
    part_group_overrides: Mapping[str, str] | None = None,
    maximum_dielectric_metallic: float = 0.35,
    allow_parameter_writes: bool = True,
) -> dict[str, Any]:
    """Apply high-confidence PBR scalars to supported already-auto MDLs."""

    if auto_material_plan.get("schema_version") != MATERIAL_PLAN_SCHEMA_VERSION:
        raise MVInverseAutonomyError(
            "auto material plan has unsupported schema_version"
        )
    try:
        verified_evidence = validate_mvinverse_evidence(mvinverse_evidence)
    except MVInverseEvidenceError as exc:
        raise MVInverseAutonomyError(
            f"MVInverse evidence failed strict validation: {exc}"
        ) from exc
    maximum_dielectric_metallic = _unit(
        maximum_dielectric_metallic, "maximum_dielectric_metallic"
    )
    if not isinstance(allow_parameter_writes, bool):
        raise MVInverseAutonomyError("allow_parameter_writes must be boolean")
    allowed = {_string(value, "allowed_material_ids") for value in allowed_material_ids}
    part_groups = _part_group_map(batches)
    overrides: dict[str, str] = {}
    if part_group_overrides is not None:
        if not isinstance(part_group_overrides, Mapping):
            raise MVInverseAutonomyError("part_group_overrides must be an object")
        for raw_part_id, raw_group_id in part_group_overrides.items():
            part_id = _string(raw_part_id, "part_group_overrides part_id")
            group_id = _string(raw_group_id, f"part_group_overrides[{part_id}]")
            overrides[part_id] = group_id
    palette_groups = _palette_groups(palette)
    evidence_groups = _evidence_groups(verified_evidence)
    assignments = _sequence(
        auto_material_plan.get("assignments"), "auto_material_plan.assignments"
    )
    assignment_part_ids = {
        _string(
            _mapping(raw_assignment, f"assignments[{index}]").get("part_id"),
            f"assignment[{index}].part_id",
        )
        for index, raw_assignment in enumerate(assignments)
    }
    unknown_overrides = sorted(set(overrides) - assignment_part_ids)
    if unknown_overrides:
        raise MVInverseAutonomyError(
            "part_group_overrides contains parts outside the auto plan: "
            f"{unknown_overrides}"
        )
    part_groups.update(overrides)

    output: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    seen_parts: set[str] = set()
    for index, raw_assignment in enumerate(assignments):
        assignment = dict(_mapping(raw_assignment, f"assignments[{index}]"))
        part_id = _string(assignment.get("part_id"), f"assignment[{index}].part_id")
        if part_id in seen_parts:
            raise MVInverseAutonomyError(f"duplicate auto assignment: {part_id}")
        seen_parts.add(part_id)
        if assignment.get("status") != "auto":
            raise MVInverseAutonomyError(
                f"auto material plan contains non-auto assignment: {part_id}"
            )
        material_id = _string(
            assignment.get("material_id"), f"assignment {part_id}.material_id"
        )
        if material_id not in allowed:
            raise MVInverseAutonomyError(
                f"assignment {part_id} material is outside the whitelist"
            )
        reason = "material_has_no_bounded_tuning_profile"
        applied = False
        pbr_observation: dict[str, Any] | None = None
        group_id = part_groups.get(part_id)
        if not allow_parameter_writes:
            reason = "selected_mdl_library_defaults_locked"
        elif group_id is None:
            reason = "part_group_mapping_unavailable"
        elif group_id not in palette_groups:
            reason = "palette_group_unavailable"
        elif tuning_profile_for_material(material_id) is None:
            reason = "material_has_no_bounded_tuning_profile"
        elif group_id not in evidence_groups:
            reason = "mvinverse_group_evidence_unavailable"
        else:
            evidence_group = evidence_groups[group_id]
            pbr_observation = {
                "contributing_view_ids": list(
                    evidence_group.get("contributing_view_ids", [])
                ),
                "base_color_srgb_median": (
                    evidence_group.get("albedo", {}).get("median")
                    if isinstance(evidence_group.get("albedo"), Mapping)
                    else None
                ),
                "metallic_median": (
                    evidence_group.get("metallic", {}).get("median")
                    if isinstance(evidence_group.get("metallic"), Mapping)
                    else None
                ),
                "roughness_median": (
                    evidence_group.get("roughness", {}).get("median")
                    if isinstance(evidence_group.get("roughness"), Mapping)
                    else None
                ),
            }
            suggestion = evidence_group.get("suggestion")
            predicted_metallic = (
                suggestion.get("metallic") if isinstance(suggestion, Mapping) else None
            )
            if (
                isinstance(predicted_metallic, (int, float))
                and not isinstance(predicted_metallic, bool)
                and float(predicted_metallic) > maximum_dielectric_metallic
            ):
                reason = "metallic_contradicts_dielectric_paint"
            else:
                try:
                    parameters, tuning_audit = tune_selected_material_from_mvinverse(
                        evidence_group,
                        group_id=group_id,
                        material_id=material_id,
                    )
                except ValueError as exc:
                    reason = "mvinverse_parameter_gate_preserved"
                    pbr_observation["parameterization_error"] = str(exc)
                else:
                    assignment["parameters"] = parameters
                    existing_views = assignment.get("evidence_views", [])
                    existing_views = [
                        _string(value, f"assignment {part_id}.evidence_views")
                        for value in _sequence(
                            existing_views, f"assignment {part_id}.evidence_views"
                        )
                    ]
                    assignment["evidence_views"] = sorted(
                        set(existing_views)
                        | set(tuning_audit["contributing_view_ids"])
                    )
                    reason = "mvinverse_selected_mdl_parameters_applied"
                    applied = True
        output.append(assignment)
        audit.append(
            {
                "part_id": part_id,
                "group_id": group_id,
                "source_material_id": material_id,
                "output_material_id": assignment["material_id"],
                "parameterized": applied,
                "reason_code": reason,
                "mvinverse_observation": pbr_observation,
            }
        )

    parameterized_count = sum(item["parameterized"] for item in audit)
    state = "READY_TO_APPLY" if output else "COMPLETED_SAFE_NOOP"
    summary: dict[str, Any] = {
        "auto_assignment_count": len(output),
        "parameterized_assignment_count": parameterized_count,
        "unchanged_auto_assignment_count": len(output) - parameterized_count,
    }
    if not allow_parameter_writes:
        summary.update(
            {
                "parameter_writes_enabled": False,
                "selected_mdl_library_defaults_locked": True,
            }
        )
    return {
        "schema_version": AUTONOMY_SCHEMA_VERSION,
        "state": state,
        "fail_closed": True,
        "summary": summary,
        "material_plan": {
            "schema_version": MATERIAL_PLAN_SCHEMA_VERSION,
            "assignments": output,
        },
        "decisions": audit,
    }


__all__ = [
    "AUTONOMY_SCHEMA_VERSION",
    "GENERIC_STEEL_PAINTED",
    "MVInverseAutonomyError",
    "build_part_view_evidence",
    "parameterize_auto_material_plan",
]
