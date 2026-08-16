"""Add photo-derived colour only to corresponding-material assignments.

This is the deliberately narrow second stage of library-first material
selection.  Exact library matches keep the authored preset unchanged.  A
``CORRESPONDING_MATERIAL`` keeps its already-selected MDL identity and receives
only colour inputs exposed by a reviewed tuning profile.  Members of one
photo-supported material component share one robust target colour.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .perceptual_color import srgb_delta_e
from .parameters import srgb_to_linear
from .tuning import (
    parameter_policy_for_material,
    tuning_profile_for_material,
)
from ..usd.material_common import canonical_sha256, normalize_material_parameters


SCHEMA_VERSION = "qwen-corresponding-material-color/v1"
AUDIT_SCHEMA_VERSION = "qwen-corresponding-material-color-audit/v1"
EXACT_LIBRARY_MATCH = "EXACT_LIBRARY_MATCH"
CORRESPONDING_MATERIAL = "CORRESPONDING_MATERIAL"


class CorrespondingMaterialColorError(ValueError):
    """Raised when the bounded colour-stage contract is violated."""


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CorrespondingMaterialColorError(f"{label} must be an object")
    return value


def _array(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise CorrespondingMaterialColorError(f"{label} must be an array")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CorrespondingMaterialColorError(f"{label} must be a non-empty string")
    return value.strip()


def _unit_color(value: Any, label: str) -> list[float]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 3
    ):
        raise CorrespondingMaterialColorError(
            f"{label} must contain exactly three sRGB channels"
        )
    color = [float(channel) for channel in value]
    if any(not math.isfinite(channel) or not 0.0 <= channel <= 1.0 for channel in color):
        raise CorrespondingMaterialColorError(
            f"{label} channels must be finite values in [0,1]"
        )
    return color


def _positive_gain(value: Any) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.1 <= float(value) <= 8.0
    ):
        raise CorrespondingMaterialColorError(
            "linear_intensity_gain must be a finite number from 0.1 to 8.0"
        )
    return float(value)


def _render_calibrated_color_parameters(
    *,
    profile: Any,
    target_srgb: Sequence[float],
    linear_intensity_gain: float,
) -> tuple[dict[str, list[float]], dict[str, Any]]:
    """Keep target chroma and absolute intensity available for render search.

    NVIDIA Base ``diffuse_tint`` often multiplies a material's native response.
    The older adapter normalized its strongest chromatic channel to one, which
    irreversibly discarded photo luminance before any real-CAD render.  This
    bounded adapter instead preserves the target's absolute linear magnitude
    and exposes one audited scalar gain for actual-render calibration.
    """

    gain = _positive_gain(linear_intensity_gain)
    validated_srgb = _unit_color(target_srgb, "target_srgb")
    target_linear = [float(value) for value in srgb_to_linear(validated_srgb)]
    authored = [min(1.0, max(0.0, value * gain)) for value in target_linear]
    parameters = {name: list(authored) for name in profile.color_parameters}
    return parameters, {
        "base_color_srgb": validated_srgb,
        "base_color_linear": target_linear,
        "authored_color_linear": authored,
        "color_parameter_semantics": (
            "render_calibrated_absolute_linear_color_gain"
        ),
        "linear_intensity_gain": gain,
        "clipped_channel_count": sum(
            value * gain > 1.0 for value in target_linear
        ),
    }


def _unique_by_part(rows: Any, label: str) -> dict[str, Mapping[str, Any]]:
    output: dict[str, Mapping[str, Any]] = {}
    for index, raw in enumerate(_array(rows, label)):
        row = _mapping(raw, f"{label}[{index}]")
        part_id = _text(row.get("part_id"), f"{label}[{index}].part_id")
        if part_id in output:
            raise CorrespondingMaterialColorError(
                f"{label} contains duplicate Part ID {part_id}"
            )
        output[part_id] = row
    return output


def _verify_document_integrity(document: Mapping[str, Any], label: str) -> None:
    integrity = _mapping(document.get("integrity"), f"{label}.integrity")
    expected = _text(
        integrity.get("document_sha256"), f"{label}.integrity.document_sha256"
    )
    unsigned = dict(document)
    unsigned.pop("integrity", None)
    if canonical_sha256(unsigned) != expected:
        raise CorrespondingMaterialColorError(f"{label} integrity mismatch")


def _selected_observation(part: Mapping[str, Any], part_id: str) -> Mapping[str, Any]:
    view_id = _text(
        part.get("selected_observation_view_id"),
        f"evidence {part_id}.selected_observation_view_id",
    )
    matches = [
        _mapping(row, f"evidence {part_id}.observation")
        for row in _array(part.get("observations"), f"evidence {part_id}.observations")
        if isinstance(row, Mapping) and row.get("view_id") == view_id
    ]
    if len(matches) != 1 or matches[0].get("selected_for_material_inference") is not True:
        raise CorrespondingMaterialColorError(
            f"evidence {part_id} must have one selected material observation"
        )
    return matches[0]


def _color_evidence(part: Mapping[str, Any], part_id: str) -> dict[str, Any]:
    if part.get("status") != "observed":
        raise CorrespondingMaterialColorError(
            f"corresponding material {part_id} must have observed photo evidence"
        )
    descriptor = _mapping(part.get("descriptor"), f"evidence {part_id}.descriptor")
    robust = _mapping(
        descriptor.get("robust_color_evidence"),
        f"evidence {part_id}.descriptor.robust_color_evidence",
    )
    if robust.get("method") != "cielab_medoid_fixed_radius":
        raise CorrespondingMaterialColorError(
            f"evidence {part_id} does not use the sealed robust colour estimator"
        )
    target = _unit_color(
        robust.get("robust_reference_srgb"),
        f"evidence {part_id}.robust_reference_srgb",
    )
    sample_count = robust.get("sample_count")
    inlier_fraction = robust.get("inlier_fraction")
    if (
        isinstance(sample_count, bool)
        or not isinstance(sample_count, int)
        or sample_count <= 0
        or isinstance(inlier_fraction, bool)
        or not isinstance(inlier_fraction, (int, float))
        or not math.isfinite(float(inlier_fraction))
        or not 0.0 < float(inlier_fraction) <= 1.0
    ):
        raise CorrespondingMaterialColorError(
            f"evidence {part_id} has invalid robust colour support"
        )
    observation = _selected_observation(part, part_id)
    evidence_weight = observation.get("camera_alignment_evidence_weight")
    if (
        isinstance(evidence_weight, bool)
        or not isinstance(evidence_weight, (int, float))
        or not math.isfinite(float(evidence_weight))
        or not 0.0 < float(evidence_weight) <= 1.0
    ):
        raise CorrespondingMaterialColorError(
            f"evidence {part_id} has invalid camera evidence weight"
        )
    # Square-root support prevents one very large panel from erasing the
    # independent observations of smaller members of the same coating group.
    weight = math.sqrt(float(sample_count)) * float(inlier_fraction) * float(
        evidence_weight
    )
    return {
        "part_id": part_id,
        "view_id": observation.get("view_id"),
        "target_srgb": target,
        "sample_count": sample_count,
        "inlier_fraction": float(inlier_fraction),
        "camera_alignment_evidence_weight": float(evidence_weight),
        "medoid_weight": weight,
    }


def _weighted_medoid(entries: Sequence[Mapping[str, Any]]) -> tuple[list[float], str]:
    if not entries:
        raise CorrespondingMaterialColorError("colour scope has no evidence")
    ranked: list[tuple[float, str, list[float]]] = []
    for candidate in entries:
        candidate_color = _unit_color(candidate.get("target_srgb"), "candidate colour")
        cost = 0.0
        for other in entries:
            cost += srgb_delta_e(candidate_color, other["target_srgb"]) * float(
                other["medoid_weight"]
            )
        ranked.append((cost, str(candidate["part_id"]), candidate_color))
    _, part_id, color = min(ranked, key=lambda row: (row[0], row[1]))
    return color, part_id


def build_corresponding_material_color_plan(
    *,
    source_plan: Mapping[str, Any],
    qwen_choices: Mapping[str, Any],
    part_id_evidence: Mapping[str, Any],
    linear_intensity_gain: float = 1.0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a plan/audit pair for the bounded second-stage colour pass."""

    calibrated_gain = _positive_gain(linear_intensity_gain)

    if source_plan.get("assignment_unit") != "part_id":
        raise CorrespondingMaterialColorError("source plan must use Part-ID assignments")
    if qwen_choices.get("assignment_unit") != "part_id":
        raise CorrespondingMaterialColorError("Qwen choices must use Part-ID assignments")
    if part_id_evidence.get("assignment_unit") != "part_id":
        raise CorrespondingMaterialColorError("evidence must use Part-ID assignments")
    _verify_document_integrity(qwen_choices, "qwen_choices")
    _verify_document_integrity(part_id_evidence, "part_id_evidence")

    assignments = _unique_by_part(source_plan.get("assignments"), "assignments")
    selections = _unique_by_part(qwen_choices.get("selections"), "selections")
    evidence = _unique_by_part(part_id_evidence.get("parts"), "evidence.parts")
    if not selections:
        raise CorrespondingMaterialColorError("Qwen choices contain no selections")

    declared_match_types: dict[str, str] = {}
    for part_id, selection in selections.items():
        if part_id not in assignments:
            raise CorrespondingMaterialColorError(
                f"selection {part_id} is missing from the source plan"
            )
        assignment = assignments[part_id]
        if selection.get("material_id") != assignment.get("material_id"):
            raise CorrespondingMaterialColorError(
                f"selection/source material mismatch for {part_id}"
            )
        if assignment.get("parameters") not in (None, {}):
            raise CorrespondingMaterialColorError(
                f"source assignment {part_id} already contains parameters"
            )
        match_type = selection.get("match_type")
        if match_type not in {EXACT_LIBRARY_MATCH, CORRESPONDING_MATERIAL}:
            raise CorrespondingMaterialColorError(
                f"selection {part_id} has unsupported match_type {match_type!r}"
            )
        declared_match_types[part_id] = str(match_type)

    consensus = _mapping(
        qwen_choices.get("component_identity_consensus"),
        "component_identity_consensus",
    )
    component_rows = _array(
        consensus.get("components"), "component_identity_consensus.components"
    )
    component_by_part: dict[str, Mapping[str, Any]] = {}
    for index, raw in enumerate(component_rows):
        component = _mapping(raw, f"component[{index}]")
        component_id = _text(component.get("component_id"), f"component[{index}].id")
        member_ids = [
            _text(value, f"component {component_id}.member_part_ids")
            for value in _array(
                component.get("member_part_ids"),
                f"component {component_id}.member_part_ids",
            )
        ]
        if not member_ids or len(member_ids) != len(set(member_ids)):
            raise CorrespondingMaterialColorError(
                f"component {component_id} has invalid members"
            )
        missing_selection_ids = sorted(set(member_ids) - set(selections))
        missing_assignment_ids = sorted(set(member_ids) - set(assignments))
        if missing_selection_ids or missing_assignment_ids:
            raise CorrespondingMaterialColorError(
                f"component {component_id} has members outside the sealed plans: "
                f"missing selections={missing_selection_ids}, "
                f"missing assignments={missing_assignment_ids}"
            )
        for part_id in member_ids:
            if part_id in component_by_part:
                raise CorrespondingMaterialColorError(
                    f"Part ID {part_id} occurs in multiple material components"
                )
            component_by_part[part_id] = component
        member_types = {selections[part_id].get("match_type") for part_id in member_ids}
        member_materials = {
            assignments[part_id].get("material_id") for part_id in member_ids
        }
        if component.get("match_type") not in member_types or len(member_types) != 1:
            raise CorrespondingMaterialColorError(
                f"component {component_id} mixes library match types"
            )
        if (
            len(member_materials) != 1
            or component.get("selected_material_id") not in member_materials
        ):
            raise CorrespondingMaterialColorError(
                f"component {component_id} does not share one selected MDL"
            )

    # Component identity consensus and exact colour-preset confirmation are
    # separate contracts.  A repeated-role consensus may legitimately choose
    # one exact *material identity* from conflicting members, but that does not
    # prove the authored colour preset matches the whole photo component.  In
    # that explicit case retain the selected MDL and make the whole component
    # eligible for the bounded colour pass.  A protected exact preset, by
    # contrast, remains immutable for every member of its material component.
    effective_match_types = dict(declared_match_types)
    for component in component_rows:
        component = _mapping(component, "component")
        if component.get("match_type") != EXACT_LIBRARY_MATCH:
            continue
        if component.get("consensus_mode") != "REPEATED_ROLE_JOINT_CONSENSUS":
            continue
        for part_id in component.get("member_part_ids", []):
            effective_match_types[str(part_id)] = CORRESPONDING_MATERIAL

    exact_ids = {
        part_id
        for part_id, match_type in effective_match_types.items()
        if match_type == EXACT_LIBRARY_MATCH
    }
    corresponding_ids = {
        part_id
        for part_id, match_type in effective_match_types.items()
        if match_type == CORRESPONDING_MATERIAL
    }
    if not corresponding_ids:
        raise CorrespondingMaterialColorError("there are no corresponding materials")

    evidence_by_id = {
        part_id: _color_evidence(evidence[part_id], part_id)
        for part_id in sorted(corresponding_ids)
        if part_id in evidence
    }
    if set(evidence_by_id) != corresponding_ids:
        raise CorrespondingMaterialColorError(
            "every corresponding material must have sealed Part-ID colour evidence"
        )

    scope_members: dict[str, list[str]] = {}
    for part_id in sorted(corresponding_ids):
        component = component_by_part.get(part_id)
        if component is None:
            scope_id = f"PART:{part_id}"
            scope_members[scope_id] = [part_id]
            continue
        component_id = str(component["component_id"])
        scope_id = f"COMPONENT:{component_id}"
        members = sorted(str(value) for value in component["member_part_ids"])
        if not set(members) <= corresponding_ids:
            raise CorrespondingMaterialColorError(
                f"corresponding component {component_id} contains an exact library match"
            )
        scope_members[scope_id] = members

    output = copy.deepcopy(dict(source_plan))
    output_assignments = _unique_by_part(output.get("assignments"), "output.assignments")
    scope_audits: list[dict[str, Any]] = []
    for scope_id, member_ids in sorted(scope_members.items()):
        materials = {str(output_assignments[part_id]["material_id"]) for part_id in member_ids}
        if len(materials) != 1:
            raise CorrespondingMaterialColorError(
                f"colour scope {scope_id} does not share one selected MDL"
            )
        material_id = next(iter(materials))
        profile = tuning_profile_for_material(material_id)
        if profile is None:
            raise CorrespondingMaterialColorError(
                f"selected material {material_id} has no reviewed colour interface"
            )
        contributors = [evidence_by_id[part_id] for part_id in member_ids]
        target_srgb, medoid_part_id = _weighted_medoid(contributors)
        parameters, color_audit = _render_calibrated_color_parameters(
            profile=profile,
            target_srgb=target_srgb,
            linear_intensity_gain=calibrated_gain,
        )
        normalized = normalize_material_parameters(material_id, parameters)
        if set(normalized) != set(parameters) or set(parameters) - set(
            parameter_policy_for_material(material_id)
        ):
            raise CorrespondingMaterialColorError(
                f"generated parameters for {scope_id} violate the reviewed policy"
            )
        record = {
            "schema_version": SCHEMA_VERSION,
            "scope_id": scope_id,
            "member_part_ids": member_ids,
            "material_id": material_id,
            "material_id_unchanged": True,
            "target_authority": "sealed_photo_evidence_weighted_cielab_medoid",
            "target_srgb": target_srgb,
            "medoid_part_id": medoid_part_id,
            "tuning_profile_id": profile.profile_id,
            "parameters": parameters,
            "color_parameter_audit": color_audit,
            "contributors": contributors,
        }
        record["record_sha256"] = canonical_sha256(record)
        for part_id in member_ids:
            assignment = output_assignments[part_id]
            assignment["parameters"] = copy.deepcopy(parameters)
            provenance = dict(_mapping(assignment.get("provenance"), f"{part_id}.provenance"))
            provenance["corresponding_material_color"] = {
                "schema_version": SCHEMA_VERSION,
                "scope_id": scope_id,
                "record_sha256": record["record_sha256"],
                "source_material_id": material_id,
                "material_id_unchanged": True,
                "parameter_names": sorted(parameters),
            }
            assignment["provenance"] = provenance
        scope_audits.append(record)

    for part_id in exact_ids:
        if output_assignments[part_id].get("parameters") not in (None, {}):
            raise CorrespondingMaterialColorError(
                f"exact library match {part_id} was unexpectedly parameterized"
            )
    if any(
        assignment.get("material_id") != assignments[part_id].get("material_id")
        for part_id, assignment in output_assignments.items()
    ):
        raise CorrespondingMaterialColorError("the colour stage changed an MDL identity")

    source_plan_sha256 = canonical_sha256(source_plan)
    plan_provenance = dict(_mapping(output.get("provenance"), "plan.provenance"))
    plan_provenance["corresponding_material_color"] = {
        "schema_version": SCHEMA_VERSION,
        "source_plan_sha256": source_plan_sha256,
        "qwen_choices_sha256": canonical_sha256(qwen_choices),
        "part_id_evidence_sha256": canonical_sha256(part_id_evidence),
        "exact_library_matches_preserved": len(exact_ids),
        "corresponding_materials_parameterized": len(corresponding_ids),
        "colour_scope_count": len(scope_audits),
        "linear_intensity_gain": calibrated_gain,
        "material_identity_changes": 0,
    }
    output["provenance"] = plan_provenance

    audit_unsigned = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "status": "PASS",
        "source_plan_sha256": source_plan_sha256,
        "output_plan_sha256": canonical_sha256(output),
        "qwen_choices_sha256": canonical_sha256(qwen_choices),
        "part_id_evidence_sha256": canonical_sha256(part_id_evidence),
        "policy": {
            "parameterize_match_type": CORRESPONDING_MATERIAL,
            "preserve_match_type": EXACT_LIBRARY_MATCH,
            "selected_mdl_identity_immutable": True,
            "reviewed_colour_interfaces_only": True,
            "same_photo_material_component_shares_colour": True,
            "absolute_photo_luminance_preserved_before_render_calibration": True,
            "linear_intensity_gain": calibrated_gain,
        },
        "summary": {
            "selection_count": len(selections),
            "exact_library_match_count": len(exact_ids),
            "corresponding_material_count": len(corresponding_ids),
            "parameterized_part_count": len(corresponding_ids),
            "colour_scope_count": len(scope_audits),
            "shared_component_scope_count": sum(
                scope["scope_id"].startswith("COMPONENT:") for scope in scope_audits
            ),
            "independent_scope_count": sum(
                scope["scope_id"].startswith("PART:") for scope in scope_audits
            ),
            "material_identity_change_count": 0,
        },
        "scopes": scope_audits,
    }
    audit = {
        **audit_unsigned,
        "integrity": {"document_sha256": canonical_sha256(audit_unsigned)},
    }
    return output, audit


def _read_object(path: Path) -> dict[str, Any]:
    with path.expanduser().resolve(strict=True).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise CorrespondingMaterialColorError(f"JSON root must be an object: {path}")
    return value


def _write_object(path: Path, value: Mapping[str, Any]) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-plan", type=Path, required=True)
    parser.add_argument("--qwen-choices", type=Path, required=True)
    parser.add_argument("--part-id-evidence", type=Path, required=True)
    parser.add_argument(
        "--linear-intensity-gain",
        type=float,
        default=1.0,
        help=(
            "bounded multiplier applied to the absolute linear photo colour; "
            "select this value with registered real-CAD renders"
        ),
    )
    parser.add_argument("--output-plan", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output, audit = build_corresponding_material_color_plan(
        source_plan=_read_object(args.source_plan),
        qwen_choices=_read_object(args.qwen_choices),
        part_id_evidence=_read_object(args.part_id_evidence),
        linear_intensity_gain=args.linear_intensity_gain,
    )
    _write_object(args.output_plan, output)
    _write_object(args.audit, audit)
    print(
        "Corresponding-material colour plan complete: "
        f"{audit['summary']['parameterized_part_count']} parts / "
        f"{audit['summary']['colour_scope_count']} colour scopes"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
