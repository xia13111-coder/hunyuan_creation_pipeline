"""Render-confirm dominant assembly cohort membership before MDL selection.

Quality repair may identify a visually coherent assembly subtree from a small
set of strict spatial anchors.  Geometry can justify proposing that the same
material extends to sibling parts, but it cannot prove the visible result.
This module turns that proposal into a deterministic two-candidate contract:

``M0``
    Keep the strict anchors and restore every expanded member to its exact
    pre-repair NVIDIA MDL material.

``M1``
    Keep the complete expanded cohort on the proposed exact NVIDIA MDL.

The selector is deliberately conservative.  It accepts M1 only from complete,
hash-bound, group-local render comparisons with at least two independently
trusted reference views.  Any malformed or incomplete evidence restores M0.
Material identity selection happens later, after membership has been frozen.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any


COHORT_SCHEMA_VERSION = "qwen-dominant-assembly-cohort/v1"
ROUND_SCHEMA_VERSION = "qwen-dominant-assembly-membership-round/v1"
SELECTION_SCHEMA_VERSION = "qwen-dominant-assembly-membership-selection/v1"

CANDIDATE_KIND_DOMINANT_ASSEMBLY = "dominant_assembly"
CANDIDATE_KIND_RARE_SOURCE_IDENTITY_PAIR = "rare_source_identity_pair"
RARE_SOURCE_IDENTITY_PAIR_POLICY = "single_strict_anchor_bounded_signature_sibling/v1"

M0_CANDIDATE = "M0_STRICT_ANCHORS_ONLY"
M1_CANDIDATE = "M1_EXPANDED_COHORT"

DEFAULT_MINIMUM_APPEARANCE_IMPROVEMENT = 0.015
DEFAULT_MINIMUM_WINNER_MARGIN = 0.01
DEFAULT_MINIMUM_TARGET_ERROR_IMPROVEMENT = 0.02
DEFAULT_MAXIMUM_TARGET_ERROR_REGRESSION = 0.03
DEFAULT_MAXIMUM_NON_TARGET_EXCESS_REGRESSION = 0.02
DEFAULT_MAXIMUM_NON_TARGET_CHROMATIC_EXCESS = 0.05

_QUALITY_RANK = {"FAIL": 0, "REVIEW": 1, "PASS": 2}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED_INPUT_HASHES = frozenset(
    {
        "baseline_plan_sha256",
        "quality_report_sha256",
        "palette_fusion_sha256",
        "spatial_report_sha256",
        "spatial_gate_audit_sha256",
        "mapping_consensus_sha256",
        "geometry_risk_sha256",
        "group_materials_sha256",
        "registry_sha256",
    }
)


class MembershipTournamentError(ValueError):
    """Raised when cohort membership cannot be safely render-confirmed."""


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
        raise MembershipTournamentError(
            f"document is not canonical JSON: {exc}"
        ) from exc
    return hashlib.sha256(payload).hexdigest()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MembershipTournamentError(f"{label} must be an object")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise MembershipTournamentError(f"{label} must be an array")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise MembershipTournamentError(f"{label} must be non-empty text")
    return value


def _sha256(value: Any, label: str) -> str:
    text = _text(value, label)
    if _SHA256.fullmatch(text) is None:
        raise MembershipTournamentError(f"{label} must be a lowercase SHA256")
    return text


def _sorted_unique_texts(value: Any, label: str) -> list[str]:
    values = _sequence(value, label)
    result = [_text(item, f"{label}[{index}]") for index, item in enumerate(values)]
    if result != sorted(set(result)):
        raise MembershipTournamentError(f"{label} must be sorted and unique")
    return result


def _unit(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise MembershipTournamentError(f"{label} must be a finite number from 0 to 1")
    return float(value)


def _assignments(
    plan: Mapping[str, Any],
    label: str,
) -> dict[str, dict[str, Any]]:
    if plan.get("schema_version") != "1.0":
        raise MembershipTournamentError(f"{label} has an unsupported schema")
    raw_assignments = _sequence(plan.get("assignments"), f"{label}.assignments")
    if not raw_assignments:
        raise MembershipTournamentError(f"{label} has no assignments")
    result: dict[str, dict[str, Any]] = {}
    for index, raw_assignment in enumerate(raw_assignments):
        assignment = _mapping(
            raw_assignment,
            f"{label}.assignments[{index}]",
        )
        part_id = _text(
            assignment.get("part_id"),
            f"{label}.assignments[{index}].part_id",
        )
        material_id = _text(
            assignment.get("material_id"),
            f"{label}.assignments[{index}].material_id",
        )
        if part_id in result:
            raise MembershipTournamentError(f"{label} repeats part {part_id}")
        if not material_id.startswith("mdl:"):
            raise MembershipTournamentError(
                f"{label}/{part_id} does not use an exact NVIDIA MDL"
            )
        if assignment.get("parameters"):
            raise MembershipTournamentError(
                f"{label}/{part_id} writes material parameters"
            )
        raw_subsets = assignment.get("face_subsets", [])
        for subset_index, raw_subset in enumerate(
            _sequence(raw_subsets, f"{label}/{part_id}.face_subsets")
        ):
            subset = _mapping(
                raw_subset,
                f"{label}/{part_id}.face_subsets[{subset_index}]",
            )
            if subset.get("parameters"):
                raise MembershipTournamentError(
                    f"{label}/{part_id} face subset writes material parameters"
                )
        result[part_id] = copy.deepcopy(dict(assignment))
    return result


def _cohort_record(
    assignment: Mapping[str, Any],
    *,
    part_id: str,
) -> Mapping[str, Any] | None:
    provenance = assignment.get("provenance")
    if provenance is None:
        return None
    provenance = _mapping(provenance, f"assignment {part_id}.provenance")
    raw = provenance.get("dominant_assembly_cohort")
    if raw is None:
        return None
    return _mapping(
        raw,
        f"assignment {part_id}.provenance.dominant_assembly_cohort",
    )


def _shared_cohort_fields(
    record: Mapping[str, Any],
    *,
    part_id: str,
) -> dict[str, Any]:
    label = f"dominant cohort member {part_id}"
    if record.get("schema_version") != COHORT_SCHEMA_VERSION:
        raise MembershipTournamentError(f"{label} has an unsupported schema")
    candidate_kind = _text(
        record.get("candidate_kind"),
        f"{label}.candidate_kind",
    )
    if candidate_kind not in {
        CANDIDATE_KIND_DOMINANT_ASSEMBLY,
        CANDIDATE_KIND_RARE_SOURCE_IDENTITY_PAIR,
    }:
        raise MembershipTournamentError(
            f"{label}.candidate_kind is unsupported: {candidate_kind!r}"
        )
    raw_proposal_policy = record.get("proposal_policy")
    proposal_policy = (
        None
        if raw_proposal_policy is None
        else _text(raw_proposal_policy, f"{label}.proposal_policy")
    )
    if (
        candidate_kind == CANDIDATE_KIND_RARE_SOURCE_IDENTITY_PAIR
        and proposal_policy != RARE_SOURCE_IDENTITY_PAIR_POLICY
    ):
        raise MembershipTournamentError(
            f"{label} rare pair requires proposal policy "
            f"{RARE_SOURCE_IDENTITY_PAIR_POLICY!r}"
        )
    cohort_id = _sha256(record.get("cohort_id"), f"{label}.cohort_id")
    contract_sha256 = _sha256(
        record.get("contract_sha256"),
        f"{label}.contract_sha256",
    )
    canonical_group_id = _text(
        record.get("canonical_group_id"),
        f"{label}.canonical_group_id",
    )
    assembly_path = _text(record.get("assembly_path"), f"{label}.assembly_path")
    signature_sha256 = _sha256(
        record.get("source_visual_stable_properties_signature_sha256"),
        f"{label}.source_visual_stable_properties_signature_sha256",
    )
    anchor_part_ids = _sorted_unique_texts(
        record.get("anchor_part_ids"),
        f"{label}.anchor_part_ids",
    )
    expanded_part_ids = _sorted_unique_texts(
        record.get("expanded_member_part_ids"),
        f"{label}.expanded_member_part_ids",
    )
    cohort_part_ids = _sorted_unique_texts(
        record.get("cohort_part_ids"),
        f"{label}.cohort_part_ids",
    )
    if (
        not anchor_part_ids
        or not expanded_part_ids
        or set(anchor_part_ids).intersection(expanded_part_ids)
        or sorted([*anchor_part_ids, *expanded_part_ids]) != cohort_part_ids
    ):
        raise MembershipTournamentError(
            f"{label} must partition cohort_part_ids into anchors and expansions"
        )
    anchor_view_ids = _sorted_unique_texts(
        record.get("anchor_supporting_view_ids"),
        f"{label}.anchor_supporting_view_ids",
    )
    minimum_anchor_view_count = (
        1 if candidate_kind == CANDIDATE_KIND_RARE_SOURCE_IDENTITY_PAIR else 2
    )
    if len(anchor_view_ids) < minimum_anchor_view_count:
        raise MembershipTournamentError(
            f"{label} candidate kind {candidate_kind!r} requires at least "
            f"{minimum_anchor_view_count} anchor-supporting view(s)"
        )
    anchor_child_branches = _sorted_unique_texts(
        record.get("anchor_child_branches"),
        f"{label}.anchor_child_branches",
    )
    if candidate_kind == CANDIDATE_KIND_RARE_SOURCE_IDENTITY_PAIR and (
        len(anchor_part_ids) != 1
        or len(expanded_part_ids) != 1
        or len(cohort_part_ids) != 2
        or len(anchor_child_branches) != 1
    ):
        raise MembershipTournamentError(
            f"{label} rare pair must contain exactly one anchor, one bounded "
            "sibling, and one direct-child anchor branch"
        )
    input_hashes = _mapping(record.get("input_hashes"), f"{label}.input_hashes")
    missing_hashes = sorted(_REQUIRED_INPUT_HASHES - set(input_hashes))
    if missing_hashes:
        raise MembershipTournamentError(f"{label}.input_hashes lacks {missing_hashes}")
    normalized_hashes = {
        str(key): _sha256(value, f"{label}.input_hashes[{key}]")
        for key, value in sorted(input_hashes.items())
    }
    if record.get("membership_status") != ("PROVISIONAL_PENDING_GROUP_TOURNAMENT"):
        raise MembershipTournamentError(
            f"{label} is not pending the required render tournament"
        )
    return {
        "schema_version": COHORT_SCHEMA_VERSION,
        "candidate_kind": candidate_kind,
        "proposal_policy": proposal_policy,
        "minimum_anchor_supporting_view_count": minimum_anchor_view_count,
        "cohort_id": cohort_id,
        "contract_sha256": contract_sha256,
        "canonical_group_id": canonical_group_id,
        "assembly_path": assembly_path,
        "source_visual_stable_properties_signature_sha256": signature_sha256,
        "anchor_part_ids": anchor_part_ids,
        "anchor_supporting_view_ids": anchor_view_ids,
        "anchor_child_branches": anchor_child_branches,
        "cohort_part_ids": cohort_part_ids,
        "expanded_member_part_ids": expanded_part_ids,
        "membership_status": "PROVISIONAL_PENDING_GROUP_TOURNAMENT",
        "input_hashes": normalized_hashes,
    }


def _target_view_maps(
    palette_fusion: Mapping[str, Any],
    canonical_group_id: str,
) -> dict[str, str]:
    raw_maps = _mapping(
        palette_fusion.get("view_group_id_maps"),
        "palette_fusion.view_group_id_maps",
    )
    local_ids_by_view: dict[str, str] = {}
    for raw_view_id, raw_group_map in sorted(raw_maps.items()):
        view_id = _text(raw_view_id, "palette fusion view ID")
        group_map = _mapping(
            raw_group_map,
            f"palette_fusion.view_group_id_maps[{view_id}]",
        )
        matches = sorted(
            _text(local_group_id, f"{view_id} local group ID")
            for local_group_id, mapped_group_id in group_map.items()
            if mapped_group_id == canonical_group_id
        )
        if len(matches) > 1:
            raise MembershipTournamentError(
                f"view {view_id} maps multiple local groups to {canonical_group_id}"
            )
        if matches:
            local_ids_by_view[view_id] = matches[0]
    if len(local_ids_by_view) < 2:
        raise MembershipTournamentError(
            f"canonical group {canonical_group_id} lacks two independent "
            "view_group_id_maps entries"
        )

    canonical_palette = _mapping(
        palette_fusion.get("canonical_palette"),
        "palette_fusion.canonical_palette",
    )
    raw_groups = _sequence(
        canonical_palette.get("groups"),
        "palette_fusion.canonical_palette.groups",
    )
    matches = [
        group
        for group in raw_groups
        if isinstance(group, Mapping) and group.get("group_id") == canonical_group_id
    ]
    if len(matches) != 1:
        raise MembershipTournamentError(
            f"canonical group {canonical_group_id} must occur exactly once"
        )
    sources = _sequence(
        matches[0].get("sources"),
        f"canonical group {canonical_group_id}.sources",
    )
    source_pairs: set[tuple[str, str]] = set()
    for index, raw_source in enumerate(sources):
        source = _mapping(
            raw_source,
            f"canonical group {canonical_group_id}.sources[{index}]",
        )
        view_id = _text(source.get("view_id"), f"group source {index}.view_id")
        local_group_id = _text(
            source.get("local_group_id"),
            f"group source {index}.local_group_id",
        )
        mapped = local_ids_by_view.get(view_id)
        if mapped != local_group_id:
            raise MembershipTournamentError(
                "canonical group sources and palette_fusion.view_group_id_maps "
                f"disagree for {view_id}:{local_group_id}"
            )
        source_pairs.add((view_id, local_group_id))
    if len({view_id for view_id, _ in source_pairs}) < 2:
        raise MembershipTournamentError(
            f"canonical group {canonical_group_id} lacks two source views"
        )
    return {view_id: local_group_id for view_id, local_group_id in sorted(source_pairs)}


def discover_dominant_assembly_cohorts(
    *,
    plan: Mapping[str, Any],
    palette_fusion: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Validate and freeze all provisional cohort contracts in a plan."""

    assignments = _assignments(plan, "dominant cohort source plan")
    members_by_cohort: dict[str, dict[str, Mapping[str, Any]]] = {}
    for part_id, assignment in assignments.items():
        record = _cohort_record(assignment, part_id=part_id)
        if record is None:
            continue
        cohort_id = _sha256(
            record.get("cohort_id"),
            f"dominant cohort member {part_id}.cohort_id",
        )
        members_by_cohort.setdefault(cohort_id, {})[part_id] = record

    cohorts: list[dict[str, Any]] = []
    for cohort_id, raw_members in sorted(members_by_cohort.items()):
        first_part_id = sorted(raw_members)[0]
        shared = _shared_cohort_fields(
            raw_members[first_part_id],
            part_id=first_part_id,
        )
        expected_part_ids = shared["cohort_part_ids"]
        if sorted(raw_members) != expected_part_ids:
            missing = sorted(set(expected_part_ids) - set(raw_members))
            unexpected = sorted(set(raw_members) - set(expected_part_ids))
            raise MembershipTournamentError(
                f"cohort {cohort_id} member provenance coverage differs from "
                f"cohort_part_ids: missing={missing}, unexpected={unexpected}"
            )

        member_baseline_material_ids: dict[str, str] = {}
        current_material_ids: set[str] = set()
        anchor_parts = set(shared["anchor_part_ids"])
        expanded_parts = set(shared["expanded_member_part_ids"])
        for part_id in expected_part_ids:
            record = raw_members[part_id]
            if _shared_cohort_fields(record, part_id=part_id) != shared:
                raise MembershipTournamentError(
                    f"cohort {cohort_id} has divergent member contracts"
                )
            role = _text(
                record.get("member_role"),
                f"dominant cohort member {part_id}.member_role",
            )
            expected_role = (
                "strict_spatial_anchor"
                if part_id in anchor_parts
                else "expanded_member"
            )
            if role != expected_role or (
                part_id not in anchor_parts and part_id not in expanded_parts
            ):
                raise MembershipTournamentError(
                    f"cohort {cohort_id} member {part_id} has role {role!r}, "
                    f"expected {expected_role!r}"
                )
            baseline_material_id = _text(
                record.get("baseline_material_id"),
                f"dominant cohort member {part_id}.baseline_material_id",
            )
            if not baseline_material_id.startswith("mdl:"):
                raise MembershipTournamentError(
                    f"cohort {cohort_id} member {part_id} baseline is not exact MDL"
                )
            member_baseline_material_ids[part_id] = baseline_material_id
            current_material_ids.add(assignments[part_id]["material_id"])
        if len(current_material_ids) != 1:
            raise MembershipTournamentError(
                f"cohort {cohort_id} M1 members do not share one exact MDL"
            )
        local_ids_by_view = _target_view_maps(
            palette_fusion,
            str(shared["canonical_group_id"]),
        )
        cohorts.append(
            {
                **copy.deepcopy(shared),
                "member_baseline_material_ids": dict(
                    sorted(member_baseline_material_ids.items())
                ),
                "expanded_material_id": next(iter(current_material_ids)),
                "target_local_group_ids_by_view": local_ids_by_view,
                "reference_view_ids": sorted(local_ids_by_view),
            }
        )
    return cohorts


def build_membership_candidate_plans(
    *,
    source_plan: Mapping[str, Any],
    palette_fusion: Mapping[str, Any],
    cohort_id: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build hash-bound M0/M1 candidates for one provisional cohort."""

    cohort_id = _sha256(cohort_id, "cohort_id")
    cohorts = {
        str(cohort["cohort_id"]): cohort
        for cohort in discover_dominant_assembly_cohorts(
            plan=source_plan,
            palette_fusion=palette_fusion,
        )
    }
    if cohort_id not in cohorts:
        raise MembershipTournamentError(f"unknown provisional cohort {cohort_id}")
    cohort = cohorts[cohort_id]
    source_plan_sha256 = _canonical_sha256(source_plan)

    m0_plan = copy.deepcopy(dict(source_plan))
    m0_assignments = _assignments(m0_plan, "M0 candidate plan")
    material_changes: list[dict[str, str]] = []
    for part_id in cohort["expanded_member_part_ids"]:
        baseline_material_id = cohort["member_baseline_material_ids"][part_id]
        previous_material_id = m0_assignments[part_id]["material_id"]
        m0_assignments[part_id]["material_id"] = baseline_material_id
        if previous_material_id != baseline_material_id:
            material_changes.append(
                {
                    "part_id": part_id,
                    "from_material_id": previous_material_id,
                    "to_material_id": baseline_material_id,
                }
            )
    if not material_changes:
        raise MembershipTournamentError(
            f"cohort {cohort_id} M0 does not restore any expanded member"
        )
    m0_plan["assignments"] = [
        m0_assignments[str(assignment["part_id"])]
        for assignment in _sequence(
            m0_plan.get("assignments"),
            "M0 candidate plan.assignments",
        )
    ]
    _assignments(m0_plan, "sealed M0 candidate plan")
    m1_plan = copy.deepcopy(dict(source_plan))

    prefix = cohort_id[:12]
    candidates = [
        {
            "candidate_id": f"membership_{prefix}_m0",
            "membership_mode": M0_CANDIDATE,
            "is_baseline": True,
            "plan": m0_plan,
            "plan_sha256": _canonical_sha256(m0_plan),
            "material_changes": material_changes,
        },
        {
            "candidate_id": f"membership_{prefix}_m1",
            "membership_mode": M1_CANDIDATE,
            "is_baseline": False,
            "plan": m1_plan,
            "plan_sha256": _canonical_sha256(m1_plan),
            "material_changes": [],
        },
    ]
    contract = {
        "schema_version": ROUND_SCHEMA_VERSION,
        "status": "PLANNED",
        "cohort_schema_version": COHORT_SCHEMA_VERSION,
        "cohort_id": cohort_id,
        "source_cohort_contract_sha256": cohort["contract_sha256"],
        "candidate_kind": cohort["candidate_kind"],
        "proposal_policy": cohort["proposal_policy"],
        "minimum_anchor_supporting_view_count": cohort[
            "minimum_anchor_supporting_view_count"
        ],
        "canonical_group_id": cohort["canonical_group_id"],
        "assembly_path": cohort["assembly_path"],
        "source_plan_sha256": source_plan_sha256,
        "palette_fusion_sha256": _canonical_sha256(palette_fusion),
        "source_input_hashes": copy.deepcopy(cohort["input_hashes"]),
        "anchor_part_ids": copy.deepcopy(cohort["anchor_part_ids"]),
        "expanded_member_part_ids": copy.deepcopy(cohort["expanded_member_part_ids"]),
        "cohort_part_ids": copy.deepcopy(cohort["cohort_part_ids"]),
        "member_baseline_material_ids": copy.deepcopy(
            cohort["member_baseline_material_ids"]
        ),
        "expanded_material_id": cohort["expanded_material_id"],
        "target_local_group_ids_by_view": copy.deepcopy(
            cohort["target_local_group_ids_by_view"]
        ),
        "reference_view_ids": copy.deepcopy(cohort["reference_view_ids"]),
        "candidate_plan_sha256": {
            candidate["membership_mode"]: candidate["plan_sha256"]
            for candidate in candidates
        },
        "candidate_ids": {
            candidate["membership_mode"]: candidate["candidate_id"]
            for candidate in candidates
        },
        "policy": {
            "quality_tier_order": ["FAIL", "REVIEW", "PASS"],
            "minimum_appearance_improvement_same_tier": (
                DEFAULT_MINIMUM_APPEARANCE_IMPROVEMENT
            ),
            "minimum_winner_margin_same_tier": DEFAULT_MINIMUM_WINNER_MARGIN,
            "minimum_target_error_improvement": (
                DEFAULT_MINIMUM_TARGET_ERROR_IMPROVEMENT
            ),
            "minimum_improved_independent_view_count": 2,
            "maximum_target_error_regression": (
                DEFAULT_MAXIMUM_TARGET_ERROR_REGRESSION
            ),
            "maximum_non_target_chromatic_excess_regression": (
                DEFAULT_MAXIMUM_NON_TARGET_EXCESS_REGRESSION
            ),
            "maximum_non_target_chromatic_excess": (
                DEFAULT_MAXIMUM_NON_TARGET_CHROMATIC_EXCESS
            ),
            "rejection_candidate": M0_CANDIDATE,
            "local_to_canonical_mapping_source": ("palette_fusion.view_group_id_maps"),
        },
        "membership_frozen_before_exact_mdl_tournament": True,
        "parameters_locked_to_library_defaults": True,
    }
    contract["round_contract_sha256"] = _canonical_sha256(contract)
    return candidates, contract


def _validate_round_contract(
    contract: Mapping[str, Any],
    *,
    palette_fusion: Mapping[str, Any],
) -> dict[str, Any]:
    if contract.get("schema_version") != ROUND_SCHEMA_VERSION:
        raise MembershipTournamentError("membership round has an unsupported schema")
    candidate_kind = _text(
        contract.get("candidate_kind"),
        "membership round candidate_kind",
    )
    if candidate_kind not in {
        CANDIDATE_KIND_DOMINANT_ASSEMBLY,
        CANDIDATE_KIND_RARE_SOURCE_IDENTITY_PAIR,
    }:
        raise MembershipTournamentError(
            "membership round candidate_kind is unsupported"
        )
    expected_anchor_view_minimum = (
        1 if candidate_kind == CANDIDATE_KIND_RARE_SOURCE_IDENTITY_PAIR else 2
    )
    if contract.get("minimum_anchor_supporting_view_count") != (
        expected_anchor_view_minimum
    ):
        raise MembershipTournamentError(
            "membership round weakens its lane-specific anchor view minimum"
        )
    if (
        candidate_kind == CANDIDATE_KIND_RARE_SOURCE_IDENTITY_PAIR
        and contract.get("proposal_policy") != RARE_SOURCE_IDENTITY_PAIR_POLICY
    ):
        raise MembershipTournamentError(
            "rare pair membership round has an invalid proposal policy"
        )
    raw_hash = _sha256(
        contract.get("round_contract_sha256"),
        "round_contract_sha256",
    )
    hash_payload = dict(contract)
    hash_payload.pop("round_contract_sha256", None)
    if _canonical_sha256(hash_payload) != raw_hash:
        raise MembershipTournamentError("membership round contract hash mismatch")
    if contract.get("palette_fusion_sha256") != _canonical_sha256(palette_fusion):
        raise MembershipTournamentError("membership round palette fusion hash mismatch")
    group_id = _text(contract.get("canonical_group_id"), "canonical_group_id")
    expected_maps = _target_view_maps(palette_fusion, group_id)
    if contract.get("target_local_group_ids_by_view") != expected_maps:
        raise MembershipTournamentError(
            "membership round does not preserve palette_fusion.view_group_id_maps"
        )
    if contract.get("reference_view_ids") != sorted(expected_maps):
        raise MembershipTournamentError(
            "membership round reference view set is inconsistent"
        )
    policy = _mapping(contract.get("policy"), "membership round policy")
    expected_policy = {
        "quality_tier_order": ["FAIL", "REVIEW", "PASS"],
        "minimum_appearance_improvement_same_tier": (
            DEFAULT_MINIMUM_APPEARANCE_IMPROVEMENT
        ),
        "minimum_winner_margin_same_tier": DEFAULT_MINIMUM_WINNER_MARGIN,
        "minimum_target_error_improvement": (DEFAULT_MINIMUM_TARGET_ERROR_IMPROVEMENT),
        "minimum_improved_independent_view_count": 2,
        "maximum_target_error_regression": (DEFAULT_MAXIMUM_TARGET_ERROR_REGRESSION),
        "maximum_non_target_chromatic_excess_regression": (
            DEFAULT_MAXIMUM_NON_TARGET_EXCESS_REGRESSION
        ),
        "maximum_non_target_chromatic_excess": (
            DEFAULT_MAXIMUM_NON_TARGET_CHROMATIC_EXCESS
        ),
        "rejection_candidate": M0_CANDIDATE,
        "local_to_canonical_mapping_source": "palette_fusion.view_group_id_maps",
    }
    if policy != expected_policy:
        raise MembershipTournamentError(
            "membership round weakens the fixed acceptance policy"
        )
    if (
        contract.get("membership_frozen_before_exact_mdl_tournament") is not True
        or contract.get("parameters_locked_to_library_defaults") is not True
    ):
        raise MembershipTournamentError(
            "membership round does not freeze membership at library defaults"
        )
    return dict(contract)


def _quality_evidence(
    quality_report: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
    candidate_id: str,
) -> dict[str, Any]:
    if quality_report.get("schema_version") != ("qwen-reference-render-comparison/v1"):
        raise MembershipTournamentError(
            f"candidate {candidate_id} quality report schema is invalid"
        )
    inputs = _mapping(
        quality_report.get("inputs"),
        f"candidate {candidate_id}.quality.inputs",
    )
    scope = _mapping(
        inputs.get("comparison_scope"),
        f"candidate {candidate_id}.quality.inputs.comparison_scope",
    )
    expected_parts = contract["cohort_part_ids"]
    if (
        scope.get("mode") != "canonical_group_local"
        or scope.get("target_group_id") != contract["canonical_group_id"]
        or scope.get("target_part_ids") != expected_parts
        or scope.get("reference_view_ids") != contract["reference_view_ids"]
    ):
        raise MembershipTournamentError(
            f"candidate {candidate_id} quality scope is not the frozen cohort"
        )

    aggregate = _mapping(
        quality_report.get("aggregate"),
        f"candidate {candidate_id}.quality.aggregate",
    )
    aggregate_status = _text(
        aggregate.get("status"),
        f"candidate {candidate_id}.aggregate.status",
    )
    if aggregate_status not in _QUALITY_RANK:
        raise MembershipTournamentError(
            f"candidate {candidate_id} aggregate status is not comparable"
        )
    appearance_score = _unit(
        aggregate.get("material_appearance_score"),
        f"candidate {candidate_id}.aggregate.material_appearance_score",
    )

    raw_views = _sequence(
        quality_report.get("views"),
        f"candidate {candidate_id}.quality.views",
    )
    views: dict[str, dict[str, Any]] = {}
    target_maps = _mapping(
        contract.get("target_local_group_ids_by_view"),
        "membership target local group maps",
    )
    for index, raw_view in enumerate(raw_views):
        view = _mapping(raw_view, f"candidate {candidate_id}.views[{index}]")
        view_id = _text(
            view.get("reference_view_id"),
            f"candidate {candidate_id}.views[{index}].reference_view_id",
        )
        if view_id in views:
            raise MembershipTournamentError(
                f"candidate {candidate_id} repeats view {view_id}"
            )
        if view_id not in target_maps:
            raise MembershipTournamentError(
                f"candidate {candidate_id} has unexpected view {view_id}"
            )
        status = _text(
            view.get("status"),
            f"candidate {candidate_id}.{view_id}.status",
        )
        if status not in _QUALITY_RANK:
            raise MembershipTournamentError(
                f"candidate {candidate_id}.{view_id} status is not comparable"
            )
        trusted = _mapping(
            _mapping(
                view.get("reference"),
                f"candidate {candidate_id}.{view_id}.reference",
            ).get("trusted_evidence"),
            f"candidate {candidate_id}.{view_id}.trusted_evidence",
        )
        local_group_id = _text(
            target_maps[view_id],
            f"membership target local group for {view_id}",
        )
        if (
            trusted.get("usable") is not True
            or trusted.get("target_group_filter_applied") is not True
            or trusted.get("target_local_group_id") != local_group_id
        ):
            raise MembershipTournamentError(
                f"candidate {candidate_id}.{view_id} lacks trusted target evidence"
            )
        color = _mapping(
            view.get("material_color"),
            f"candidate {candidate_id}.{view_id}.material_color",
        )
        group_recall = _mapping(
            color.get("trusted_evidence_group_recall"),
            f"candidate {candidate_id}.{view_id}.trusted_evidence_group_recall",
        )
        groups = _sequence(
            group_recall.get("groups"),
            f"candidate {candidate_id}.{view_id}.group_recall.groups",
        )
        group_matches = [
            group
            for group in groups
            if isinstance(group, Mapping) and group.get("group_id") == local_group_id
        ]
        if len(group_matches) != 1:
            raise MembershipTournamentError(
                f"candidate {candidate_id}.{view_id} lacks one local target group"
            )
        group = group_matches[0]
        reference_share = _unit(
            group.get("reference_color_share"),
            f"candidate {candidate_id}.{view_id}.reference_color_share",
        )
        render_share = _unit(
            group.get("observed_render_share"),
            f"candidate {candidate_id}.{view_id}.observed_render_share",
        )
        target_bins = set(
            _sorted_unique_texts(
                group.get("render_color_bins"),
                f"candidate {candidate_id}.{view_id}.render_color_bins",
            )
        )
        if not target_bins:
            raise MembershipTournamentError(
                f"candidate {candidate_id}.{view_id} has no target color bins"
            )
        chromatic = _mapping(
            color.get("unreferenced_render_chromatic_mass"),
            f"candidate {candidate_id}.{view_id}.unreferenced chromatic mass",
        )
        bins = _sequence(
            chromatic.get("bins"),
            f"candidate {candidate_id}.{view_id}.chromatic bins",
        )
        non_target_excess = 0.0
        for bin_index, raw_bin in enumerate(bins):
            bin_record = _mapping(
                raw_bin,
                f"candidate {candidate_id}.{view_id}.chromatic bins[{bin_index}]",
            )
            color_bin = _text(
                bin_record.get("color_bin"),
                f"candidate {candidate_id}.{view_id}.chromatic color_bin",
            )
            excess = _unit(
                bin_record.get(
                    "effective_excess_share",
                    bin_record.get("excess_share"),
                ),
                (
                    f"candidate {candidate_id}.{view_id}.{color_bin}."
                    "effective_excess_share"
                ),
            )
            if color_bin not in target_bins:
                non_target_excess += excess
        non_target_excess = min(1.0, non_target_excess)
        views[view_id] = {
            "status": status,
            "status_rank": _QUALITY_RANK[status],
            "target_local_group_id": local_group_id,
            "target_canonical_group_id": contract["canonical_group_id"],
            "mapping_source": "palette_fusion.view_group_id_maps",
            "reference_target_color_share": reference_share,
            "render_target_color_share": render_share,
            "target_color_share_absolute_error": abs(reference_share - render_share),
            "non_target_chromatic_excess": non_target_excess,
            "material_appearance_score": _unit(
                view.get("material_appearance_score"),
                f"candidate {candidate_id}.{view_id}.material_appearance_score",
            ),
        }
    if sorted(views) != contract["reference_view_ids"]:
        raise MembershipTournamentError(
            f"candidate {candidate_id} does not cover every trusted reference view"
        )
    return {
        "aggregate_status": aggregate_status,
        "aggregate_status_rank": _QUALITY_RANK[aggregate_status],
        "material_appearance_score": appearance_score,
        "views": views,
    }


def select_membership_candidate(
    *,
    contract: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    palette_fusion: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Select M1 only when complete render evidence clears every fixed gate.

    ``candidates`` must contain exactly two bundles with ``candidate_id``,
    ``plan`` and ``quality_report``.  Any contract or evidence error is raised;
    a valid but insufficient M1 result deterministically returns M0.
    """

    validated_contract = _validate_round_contract(
        contract,
        palette_fusion=palette_fusion,
    )
    raw_candidate_hashes = _mapping(
        validated_contract.get("candidate_plan_sha256"),
        "candidate_plan_sha256",
    )
    raw_candidate_ids = _mapping(
        validated_contract.get("candidate_ids"),
        "candidate_ids",
    )
    bundles: dict[str, dict[str, Any]] = {}
    for index, raw_bundle in enumerate(_sequence(candidates, "candidates")):
        bundle = _mapping(raw_bundle, f"candidates[{index}]")
        candidate_id = _text(
            bundle.get("candidate_id"),
            f"candidates[{index}].candidate_id",
        )
        matching_modes = [
            mode
            for mode, expected_id in raw_candidate_ids.items()
            if expected_id == candidate_id
        ]
        if len(matching_modes) != 1:
            raise MembershipTournamentError(
                f"unexpected membership candidate {candidate_id}"
            )
        mode = matching_modes[0]
        if mode in bundles:
            raise MembershipTournamentError(
                f"membership candidate mode {mode} is duplicated"
            )
        plan = _mapping(bundle.get("plan"), f"candidate {candidate_id}.plan")
        expected_hash = _sha256(
            raw_candidate_hashes.get(mode),
            f"candidate_plan_sha256[{mode}]",
        )
        if _canonical_sha256(plan) != expected_hash:
            raise MembershipTournamentError(
                f"candidate {candidate_id} plan hash mismatch"
            )
        raw_quality_report = bundle.get("quality_report")
        evidence_error: str | None = None
        if isinstance(raw_quality_report, Mapping):
            quality_report = raw_quality_report
            quality_report_sha256 = _canonical_sha256(quality_report)
            try:
                evidence = _quality_evidence(
                    quality_report,
                    contract=validated_contract,
                    candidate_id=candidate_id,
                )
            except MembershipTournamentError as exc:
                evidence = None
                evidence_error = str(exc)
        else:
            quality_report_sha256 = None
            evidence = None
            evidence_error = (
                f"candidate {candidate_id}.quality_report must be an object"
            )
        bundles[mode] = {
            "candidate_id": candidate_id,
            "plan": copy.deepcopy(dict(plan)),
            "plan_sha256": expected_hash,
            "quality_report_sha256": quality_report_sha256,
            "evidence": evidence,
            "evidence_error": evidence_error,
        }
    if set(bundles) != {M0_CANDIDATE, M1_CANDIDATE}:
        raise MembershipTournamentError(
            "membership tournament requires exactly M0 and M1"
        )

    m0 = bundles[M0_CANDIDATE]
    m1 = bundles[M1_CANDIDATE]
    evidence_errors = {
        mode: str(bundle["evidence_error"])
        for mode, bundle in sorted(bundles.items())
        if bundle["evidence_error"] is not None
    }
    if evidence_errors:
        output = copy.deepcopy(m0["plan"])
        provenance = output.setdefault("provenance", {})
        if not isinstance(provenance, dict):
            raise MembershipTournamentError(
                "M0 membership plan provenance must be an object"
            )
        history = provenance.setdefault(
            "dominant_assembly_membership_tournaments",
            [],
        )
        if not isinstance(history, list):
            raise MembershipTournamentError(
                "membership tournament history must be an array"
            )
        history.append(
            {
                "schema_version": SELECTION_SCHEMA_VERSION,
                "cohort_id": validated_contract["cohort_id"],
                "source_cohort_contract_sha256": validated_contract[
                    "source_cohort_contract_sha256"
                ],
                "candidate_kind": validated_contract["candidate_kind"],
                "proposal_policy": validated_contract["proposal_policy"],
                "round_contract_sha256": validated_contract["round_contract_sha256"],
                "canonical_group_id": validated_contract["canonical_group_id"],
                "selected_membership_mode": M0_CANDIDATE,
                "selected_candidate_id": m0["candidate_id"],
                "selected_candidate_plan_sha256": m0["plan_sha256"],
                "membership_status": ("RENDER_EVIDENCE_INVALID_M0_RESTORED"),
                "included_part_ids": validated_contract["anchor_part_ids"],
                "excluded_expanded_part_ids": validated_contract[
                    "expanded_member_part_ids"
                ],
                "membership_frozen_before_exact_mdl_tournament": True,
                "parameters_locked_to_library_defaults": True,
            }
        )
        return output, {
            "schema_version": SELECTION_SCHEMA_VERSION,
            "status": "REJECTED_EXPANSION_RESTORED_M0",
            "cohort_id": validated_contract["cohort_id"],
            "canonical_group_id": validated_contract["canonical_group_id"],
            "candidate_kind": validated_contract["candidate_kind"],
            "proposal_policy": validated_contract["proposal_policy"],
            "round_contract_sha256": validated_contract["round_contract_sha256"],
            "m0_candidate_id": m0["candidate_id"],
            "m1_candidate_id": m1["candidate_id"],
            "m0_plan_sha256": m0["plan_sha256"],
            "m1_plan_sha256": m1["plan_sha256"],
            "m0_quality_report_sha256": m0["quality_report_sha256"],
            "m1_quality_report_sha256": m1["quality_report_sha256"],
            "m0_aggregate_status": None,
            "m1_aggregate_status": None,
            "m0_material_appearance_score": None,
            "m1_material_appearance_score": None,
            "appearance_score_improvement": None,
            "quality_tier_gate_passed": False,
            "tier_improved": False,
            "same_tier_score_gate_passed": False,
            "improved_independent_view_count": 0,
            "independent_view_gate_passed": False,
            "no_view_regression_gate_passed": False,
            "views": [],
            "reason_codes": ["CANDIDATE_RENDER_EVIDENCE_INVALID"],
            "evidence_errors": evidence_errors,
            "selected_membership_mode": M0_CANDIDATE,
            "selected_candidate_id": m0["candidate_id"],
            "selected_candidate_plan_sha256": m0["plan_sha256"],
            "output_plan_sha256": _canonical_sha256(output),
            "rejection_restores_m0": True,
            "membership_frozen_before_exact_mdl_tournament": True,
            "parameters_locked_to_library_defaults": True,
        }
    m0_evidence = m0["evidence"]
    m1_evidence = m1["evidence"]
    assert isinstance(m0_evidence, Mapping)
    assert isinstance(m1_evidence, Mapping)
    m0_rank = int(m0_evidence["aggregate_status_rank"])
    m1_rank = int(m1_evidence["aggregate_status_rank"])
    appearance_delta = float(m1_evidence["material_appearance_score"]) - float(
        m0_evidence["material_appearance_score"]
    )
    tier_improved = m1_rank > m0_rank
    same_tier_score_gate = (
        m1_rank == m0_rank
        and appearance_delta >= DEFAULT_MINIMUM_APPEARANCE_IMPROVEMENT
        and appearance_delta >= DEFAULT_MINIMUM_WINNER_MARGIN
    )
    quality_gate = tier_improved or same_tier_score_gate

    view_records: list[dict[str, Any]] = []
    improved_view_count = 0
    regression_reason_codes: set[str] = set()
    for view_id in validated_contract["reference_view_ids"]:
        m0_view = m0_evidence["views"][view_id]
        m1_view = m1_evidence["views"][view_id]
        target_error_delta = float(
            m1_view["target_color_share_absolute_error"]
        ) - float(m0_view["target_color_share_absolute_error"])
        target_error_improvement = -target_error_delta
        excess_delta = float(m1_view["non_target_chromatic_excess"]) - float(
            m0_view["non_target_chromatic_excess"]
        )
        improved = target_error_improvement >= DEFAULT_MINIMUM_TARGET_ERROR_IMPROVEMENT
        if improved:
            improved_view_count += 1
        reason_codes: list[str] = []
        if int(m1_view["status_rank"]) < int(m0_view["status_rank"]):
            reason_codes.append("VIEW_STATUS_REGRESSION")
        if target_error_delta > DEFAULT_MAXIMUM_TARGET_ERROR_REGRESSION:
            reason_codes.append("TARGET_COLOR_SHARE_ERROR_REGRESSION")
        if excess_delta > DEFAULT_MAXIMUM_NON_TARGET_EXCESS_REGRESSION:
            reason_codes.append("NON_TARGET_CHROMATIC_EXCESS_REGRESSION")
        if (
            float(m1_view["non_target_chromatic_excess"])
            > DEFAULT_MAXIMUM_NON_TARGET_CHROMATIC_EXCESS
        ):
            reason_codes.append("NON_TARGET_CHROMATIC_EXCESS_LIMIT_EXCEEDED")
        regression_reason_codes.update(reason_codes)
        view_records.append(
            {
                "reference_view_id": view_id,
                "target_local_group_id": m0_view["target_local_group_id"],
                "target_canonical_group_id": (validated_contract["canonical_group_id"]),
                "mapping_source": "palette_fusion.view_group_id_maps",
                "m0_status": m0_view["status"],
                "m1_status": m1_view["status"],
                "m0_target_color_share_absolute_error": (
                    m0_view["target_color_share_absolute_error"]
                ),
                "m1_target_color_share_absolute_error": (
                    m1_view["target_color_share_absolute_error"]
                ),
                "target_color_share_error_improvement": (target_error_improvement),
                "target_error_improvement_gate_passed": improved,
                "m0_non_target_chromatic_excess": (
                    m0_view["non_target_chromatic_excess"]
                ),
                "m1_non_target_chromatic_excess": (
                    m1_view["non_target_chromatic_excess"]
                ),
                "non_target_chromatic_excess_delta": excess_delta,
                "reason_codes": reason_codes,
            }
        )
    independent_view_gate = improved_view_count >= 2
    no_regression_gate = not regression_reason_codes
    accept_m1 = quality_gate and independent_view_gate and no_regression_gate

    decision_reasons: list[str] = []
    if not quality_gate:
        if m1_rank < m0_rank:
            decision_reasons.append("M1_AGGREGATE_QUALITY_TIER_REGRESSED")
        elif m1_rank == m0_rank:
            decision_reasons.append("M1_SAME_TIER_APPEARANCE_IMPROVEMENT_INSUFFICIENT")
        else:
            decision_reasons.append("M1_QUALITY_GATE_FAILED")
    if not independent_view_gate:
        decision_reasons.append("M1_TARGET_COLOR_SHARE_DID_NOT_IMPROVE_IN_TWO_VIEWS")
    decision_reasons.extend(sorted(regression_reason_codes))
    selected_mode = M1_CANDIDATE if accept_m1 else M0_CANDIDATE
    selected = bundles[selected_mode]
    output = copy.deepcopy(selected["plan"])
    provenance = output.setdefault("provenance", {})
    if not isinstance(provenance, dict):
        raise MembershipTournamentError(
            "selected membership plan provenance must be an object"
        )
    history = provenance.setdefault(
        "dominant_assembly_membership_tournaments",
        [],
    )
    if not isinstance(history, list):
        raise MembershipTournamentError(
            "membership tournament history must be an array"
        )
    selection_summary = {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "cohort_id": validated_contract["cohort_id"],
        "source_cohort_contract_sha256": validated_contract[
            "source_cohort_contract_sha256"
        ],
        "candidate_kind": validated_contract["candidate_kind"],
        "proposal_policy": validated_contract["proposal_policy"],
        "round_contract_sha256": validated_contract["round_contract_sha256"],
        "canonical_group_id": validated_contract["canonical_group_id"],
        "selected_membership_mode": selected_mode,
        "selected_candidate_id": selected["candidate_id"],
        "selected_candidate_plan_sha256": selected["plan_sha256"],
        "membership_status": (
            "RENDER_CONFIRMED_EXPANDED_COHORT"
            if accept_m1
            else "RENDER_REJECTED_EXPANSION_M0_RESTORED"
        ),
        "included_part_ids": (
            validated_contract["cohort_part_ids"]
            if accept_m1
            else validated_contract["anchor_part_ids"]
        ),
        "excluded_expanded_part_ids": (
            [] if accept_m1 else validated_contract["expanded_member_part_ids"]
        ),
        "membership_frozen_before_exact_mdl_tournament": True,
        "parameters_locked_to_library_defaults": True,
    }
    history.append(selection_summary)
    output_sha256 = _canonical_sha256(output)
    audit = {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "status": (
            "ACCEPTED_EXPANDED_COHORT"
            if accept_m1
            else "REJECTED_EXPANSION_RESTORED_M0"
        ),
        "cohort_id": validated_contract["cohort_id"],
        "canonical_group_id": validated_contract["canonical_group_id"],
        "candidate_kind": validated_contract["candidate_kind"],
        "proposal_policy": validated_contract["proposal_policy"],
        "round_contract_sha256": validated_contract["round_contract_sha256"],
        "m0_candidate_id": m0["candidate_id"],
        "m1_candidate_id": m1["candidate_id"],
        "m0_plan_sha256": m0["plan_sha256"],
        "m1_plan_sha256": m1["plan_sha256"],
        "m0_quality_report_sha256": m0["quality_report_sha256"],
        "m1_quality_report_sha256": m1["quality_report_sha256"],
        "m0_aggregate_status": m0_evidence["aggregate_status"],
        "m1_aggregate_status": m1_evidence["aggregate_status"],
        "m0_material_appearance_score": m0_evidence["material_appearance_score"],
        "m1_material_appearance_score": m1_evidence["material_appearance_score"],
        "appearance_score_improvement": appearance_delta,
        "quality_tier_gate_passed": quality_gate,
        "tier_improved": tier_improved,
        "same_tier_score_gate_passed": same_tier_score_gate,
        "improved_independent_view_count": improved_view_count,
        "independent_view_gate_passed": independent_view_gate,
        "no_view_regression_gate_passed": no_regression_gate,
        "views": view_records,
        "reason_codes": decision_reasons,
        "selected_membership_mode": selected_mode,
        "selected_candidate_id": selected["candidate_id"],
        "selected_candidate_plan_sha256": selected["plan_sha256"],
        "output_plan_sha256": output_sha256,
        "rejection_restores_m0": not accept_m1,
        "membership_frozen_before_exact_mdl_tournament": True,
        "parameters_locked_to_library_defaults": True,
    }
    return output, audit


def membership_exclusions_by_group(
    plan: Mapping[str, Any],
) -> dict[str, set[str]]:
    """Return expanded parts that an M0 decision bars from later MDL rounds.

    The multi-group exact-MDL queue calls this after visual-group annotation so
    a rejected expansion cannot silently re-enter when the canonical group is
    rebound to another MDL identity.
    """

    provenance = plan.get("provenance")
    if provenance is None:
        return {}
    provenance = _mapping(provenance, "plan.provenance")
    raw_history = provenance.get("dominant_assembly_membership_tournaments")
    if raw_history is None:
        return {}
    history = _sequence(
        raw_history,
        "plan.provenance.dominant_assembly_membership_tournaments",
    )
    exclusions: dict[str, set[str]] = {}
    seen_cohort_ids: set[str] = set()
    for index, raw_selection in enumerate(history):
        selection = _mapping(
            raw_selection,
            f"membership tournament selection[{index}]",
        )
        if selection.get("schema_version") != SELECTION_SCHEMA_VERSION:
            raise MembershipTournamentError(
                f"membership tournament selection[{index}] schema is invalid"
            )
        cohort_id = _sha256(
            selection.get("cohort_id"),
            f"membership tournament selection[{index}].cohort_id",
        )
        if cohort_id in seen_cohort_ids:
            raise MembershipTournamentError(
                f"membership cohort {cohort_id} was frozen more than once"
            )
        seen_cohort_ids.add(cohort_id)
        if (
            selection.get("membership_frozen_before_exact_mdl_tournament") is not True
            or selection.get("parameters_locked_to_library_defaults") is not True
        ):
            raise MembershipTournamentError(
                f"membership cohort {cohort_id} selection is not frozen"
            )
        mode = selection.get("selected_membership_mode")
        group_id = _text(
            selection.get("canonical_group_id"),
            f"membership cohort {cohort_id}.canonical_group_id",
        )
        excluded = _sorted_unique_texts(
            selection.get("excluded_expanded_part_ids"),
            f"membership cohort {cohort_id}.excluded_expanded_part_ids",
        )
        if mode == M1_CANDIDATE:
            if excluded:
                raise MembershipTournamentError(
                    f"accepted cohort {cohort_id} unexpectedly excludes members"
                )
        elif mode == M0_CANDIDATE:
            if not excluded:
                raise MembershipTournamentError(
                    f"rejected cohort {cohort_id} lacks restored M0 members"
                )
            exclusions.setdefault(group_id, set()).update(excluded)
        else:
            raise MembershipTournamentError(
                f"membership cohort {cohort_id} has invalid selected mode"
            )
    return exclusions


__all__ = [
    "CANDIDATE_KIND_DOMINANT_ASSEMBLY",
    "CANDIDATE_KIND_RARE_SOURCE_IDENTITY_PAIR",
    "COHORT_SCHEMA_VERSION",
    "DEFAULT_MAXIMUM_NON_TARGET_CHROMATIC_EXCESS",
    "DEFAULT_MAXIMUM_NON_TARGET_EXCESS_REGRESSION",
    "DEFAULT_MAXIMUM_TARGET_ERROR_REGRESSION",
    "DEFAULT_MINIMUM_APPEARANCE_IMPROVEMENT",
    "DEFAULT_MINIMUM_TARGET_ERROR_IMPROVEMENT",
    "DEFAULT_MINIMUM_WINNER_MARGIN",
    "M0_CANDIDATE",
    "M1_CANDIDATE",
    "MembershipTournamentError",
    "RARE_SOURCE_IDENTITY_PAIR_POLICY",
    "ROUND_SCHEMA_VERSION",
    "SELECTION_SCHEMA_VERSION",
    "build_membership_candidate_plans",
    "discover_dominant_assembly_cohorts",
    "membership_exclusions_by_group",
    "select_membership_candidate",
]
