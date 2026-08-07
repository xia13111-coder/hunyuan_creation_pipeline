"""Hash-bound quality-repair authorization and provenance validation."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from typing import Any

from ..config import canonical_sha256
from ..policy_contract import (
    NEUTRAL_FALLBACK_TIERS,
    POLICY_FALLBACK_CONFIDENCE_BASIS,
    POLICY_FALLBACK_STATUS,
    POLICY_PLAN_MODE,
)
from ..policy_exact_cover import _require_exact_int
from .constants import (
    MATERIAL_SELECTION_OBJECTIVE_SEMANTIC,
    MATERIAL_SELECTION_OBJECTIVE_VISUAL,
    QUALITY_REPAIR_ANCHORED_SINGLE_VIEW_LANE,
    QUALITY_REPAIR_BOUNDED_SIGNATURE_SIBLING_COHORT_LANE,
    QUALITY_REPAIR_BOUNDED_SIGNATURE_SIBLING_REASON_CODES,
    QUALITY_REPAIR_DARK_FOREGROUND_MAX_SINGLE_CONTRIBUTION_FACTOR,
    QUALITY_REPAIR_DARK_FOREGROUND_REASON_CODES,
    QUALITY_REPAIR_DARK_FOREGROUND_RESIDUAL_LANE,
    QUALITY_REPAIR_DOMINANT_ASSEMBLY_COHORT_LANE,
    QUALITY_REPAIR_DOMINANT_ASSEMBLY_REASON_CODES,
    QUALITY_REPAIR_DOMINANT_RESIDUAL_SINGLE_VIEW_LANE,
    QUALITY_REPAIR_LOCALIZATION_LANES,
    QUALITY_REPAIR_MIN_PROVISIONAL_MATERIAL_CONFIDENCE,
    QUALITY_REPAIR_MULTIVIEW_DARK_IDENTITY_LANE,
    QUALITY_REPAIR_MULTIVIEW_SEMANTIC_REVIEW_LANE,
    QUALITY_REPAIR_PLAN_MODE,
    QUALITY_REPAIR_PROVENANCE_FIELD,
    QUALITY_REPAIR_PROVISIONAL_MATERIAL_BASIS,
    QUALITY_REPAIR_PROVISIONAL_REASON_CODES,
    QUALITY_REPAIR_REASON_CODES,
    QUALITY_REPAIR_REPEATED_GEOMETRY_DARK_REASON_CODES,
    QUALITY_REPAIR_REPEATED_GEOMETRY_DARK_RESIDUAL_LANE,
    QUALITY_REPAIR_REPORT_SCHEMA_VERSION,
    QUALITY_REPAIR_SEMANTIC_SINGLE_VIEW_LANE,
    QUALITY_REPAIR_SOURCE_IDENTITY_COHORT_CONSENSUS_LANE,
    QUALITY_REPAIR_SOURCE_IDENTITY_LANE,
    QUALITY_REPAIR_SOURCE_IDENTITY_MAX_ASSEMBLY_COHORT_SIZE,
    QUALITY_REPAIR_SOURCE_IDENTITY_MAX_REGISTRY_FRACTION,
    QUALITY_REPAIR_SOURCE_IDENTITY_MIN_SIGNATURE_COUNT,
    QUALITY_REPAIR_SPATIAL_ANCHOR_SINGLE_VIEW_LANE,
)
from .diagnostics import (
    _quality_dark_alignment,
    _quality_dark_matched_semantic_conflict,
    _quality_dark_reference_evidence,
    _quality_dark_spatial_observation,
    _quality_dark_spatial_part,
    _quality_dark_support_from_report,
    _quality_repeated_source_signature,
    _recompute_quality_multiview_dark_identity,
    _validate_quality_dark_foreground_diagnostic,
    _validate_quality_multiview_semantic_review_support,
    _validate_quality_semantic_review_override,
    _validate_repeated_geometry_dark_cohorts,
)
from .metrics import (
    _independent_quality_spatial_anchor_view_ids,
    _quality_sha256,
    _quality_unit,
    _validate_quality_dominant_mass,
)
from qwen_material_pipeline.materials.membership_tournament import (
    MembershipTournamentError,
    discover_dominant_assembly_cohorts,
)
from qwen_material_pipeline.materials.tuning import (
    tune_selected_material_from_mvinverse,
    tuning_profile_for_material,
)


def _expected_repair_mvinverse_delta(
    *,
    evidence: dict[str, Any],
    group_id: str,
    source_material_id: str,
    allowed_material_ids: set[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Rebuild the only MVInverse parameter delta allowed during QA repair."""

    if (
        evidence.get("schema_version")
        != "qwen-mvinverse-pbr-evidence/v1"
        or not isinstance(evidence.get("inputs"), dict)
        or evidence["inputs"].get("integrity_verified") is not True
        or not isinstance(evidence.get("summary"), dict)
        or evidence["summary"].get("fail_closed") is not True
        or source_material_id not in allowed_material_ids
        or tuning_profile_for_material(source_material_id) is None
    ):
        raise RuntimeError("Quality-repair MVInverse trust boundary is invalid")
    raw_groups = evidence.get("groups")
    raw_views = evidence.get("views")
    summary = evidence.get("summary")
    if (
        not isinstance(raw_groups, list)
        or not isinstance(raw_views, list)
        or not isinstance(summary, dict)
    ):
        raise RuntimeError("Quality-repair MVInverse evidence is incomplete")
    group_ids = [
        item.get("group_id") if isinstance(item, dict) else None
        for item in raw_groups
    ]
    view_ids = [
        item.get("view_id") if isinstance(item, dict) else None
        for item in raw_views
    ]
    auto_group_count = sum(
        1
        for item in raw_groups
        if isinstance(item, dict)
        and isinstance(item.get("suggestion"), dict)
        and item["suggestion"].get("decision") == "auto"
        and item["suggestion"].get("auto_parameter_eligible") is True
    )
    if (
        any(not isinstance(value, str) or not value for value in group_ids)
        or len(group_ids) != len(set(group_ids))
        or any(not isinstance(value, str) or not value for value in view_ids)
        or len(view_ids) != len(set(view_ids))
        or summary.get("view_count") != len(view_ids)
        or summary.get("canonical_group_count") != len(group_ids)
        or summary.get("auto_parameter_group_count") != auto_group_count
        or summary.get("usd_modified") is not False
    ):
        raise RuntimeError("Quality-repair MVInverse evidence summary is invalid")
    matches = [
        item
        for item in raw_groups
        if isinstance(item, dict) and item.get("group_id") == group_id
    ]
    if len(matches) != 1:
        raise RuntimeError("Quality-repair MVInverse group join is not unique")
    group = matches[0]
    suggestion = group.get("suggestion")
    metallic_stats = group.get("metallic")
    contributing_views = group.get("contributing_view_ids")
    if (
        group.get("surface_class") != "dielectric"
        or not isinstance(suggestion, dict)
        or suggestion.get("decision") != "auto"
        or suggestion.get("auto_parameter_eligible") is not True
        or not isinstance(metallic_stats, dict)
        or not isinstance(contributing_views, list)
        or len(contributing_views) < 2
        or any(
            not isinstance(view_id, str) or not view_id
            for view_id in contributing_views
        )
        or len(contributing_views) != len(set(contributing_views))
        or not set(contributing_views) <= set(view_ids)
        or group.get("distinct_view_count") != len(contributing_views)
        or any(
            not isinstance(group.get(label), dict)
            or group[label].get("sample_count") != len(contributing_views)
            for label in ("albedo", "metallic", "roughness")
        )
    ):
        raise RuntimeError("Quality-repair MVInverse group is not eligible")

    try:
        parameters, parameter_audit = tune_selected_material_from_mvinverse(
            group,
            group_id=group_id,
            material_id=source_material_id,
        )
    except ValueError as exc:
        raise RuntimeError(
            "Quality-repair MVInverse parameter delta is invalid"
        ) from exc
    audit = {
        "source_material_id": source_material_id,
        "output_material_id": source_material_id,
        "group_id": parameter_audit["group_id"],
        "tuning_profile_id": parameter_audit["tuning_profile_id"],
        "parameterization_mode": parameter_audit["parameterization_mode"],
        "contributing_view_ids": list(
            parameter_audit["contributing_view_ids"]
        ),
        "base_color_srgb": list(parameter_audit["base_color_srgb"]),
        "base_color_linear": list(parameter_audit["base_color_linear"]),
        "observed_metallic": parameter_audit["observed_metallic"],
        "authored_metallic": parameter_audit["authored_metallic"],
        "roughness": parameter_audit["roughness"],
        "authored_parameter_names": list(
            parameter_audit["authored_parameter_names"]
        ),
        "reason_code": "MVINVERSE_AUTO_PARAMETER_ELIGIBLE",
    }
    return parameters, audit
def _validate_quality_repair_dominant_assembly_cohorts(
    *,
    plan: dict[str, Any],
    audit: dict[str, Any],
    palette_fusion: dict[str, Any],
    expected_input_hashes: Mapping[str, Any],
    changes_by_part: Mapping[str, dict[str, Any]],
    localization_lanes_by_part: Mapping[str, dict[str, str]],
) -> dict[str, dict[str, Any]]:
    """Validate hash-bound provisional cohorts before their render tournament."""

    cohort_lane_by_candidate_kind = {
        "dominant_assembly": QUALITY_REPAIR_DOMINANT_ASSEMBLY_COHORT_LANE,
        "rare_source_identity_pair": (
            QUALITY_REPAIR_BOUNDED_SIGNATURE_SIBLING_COHORT_LANE
        ),
    }
    lane_part_ids = {
        part_id
        for part_id, lane in localization_lanes_by_part.items()
        if lane["lane"] in set(cohort_lane_by_candidate_kind.values())
    }
    raw_cohorts = audit.get("dominant_assembly_cohorts", [])
    if not isinstance(raw_cohorts, list):
        raise RuntimeError("Quality-repair dominant assembly cohorts are invalid")
    if not lane_part_ids:
        if raw_cohorts:
            raise RuntimeError(
                "Quality-repair audit contains an unused dominant assembly cohort"
            )
        return {}

    try:
        discovered_cohorts = discover_dominant_assembly_cohorts(
            plan=plan,
            palette_fusion=palette_fusion,
        )
    except MembershipTournamentError as exc:
        raise RuntimeError(
            f"Quality-repair dominant assembly cohort contract is invalid: {exc}"
        ) from exc

    raw_by_id: dict[str, dict[str, Any]] = {}
    for raw_cohort in raw_cohorts:
        if not isinstance(raw_cohort, dict):
            raise RuntimeError(
                "Quality-repair dominant assembly cohort audit is invalid"
            )
        cohort_id = raw_cohort.get("cohort_id")
        contract_sha256 = raw_cohort.get("contract_sha256")
        unsigned_contract = copy.deepcopy(raw_cohort)
        unsigned_contract.pop("contract_sha256", None)
        if (
            not _quality_sha256(cohort_id)
            or cohort_id in raw_by_id
            or not _quality_sha256(contract_sha256)
            or canonical_sha256(unsigned_contract) != contract_sha256
        ):
            raise RuntimeError(
                "Quality-repair dominant assembly cohort hash is invalid"
            )
        raw_by_id[str(cohort_id)] = raw_cohort

    discovered_by_id = {
        str(cohort["cohort_id"]): cohort for cohort in discovered_cohorts
    }
    if (
        len(discovered_by_id) != len(discovered_cohorts)
        or set(raw_by_id) != set(discovered_by_id)
    ):
        raise RuntimeError(
            "Quality-repair dominant assembly cohort coverage is inconsistent"
        )

    validated_parts: dict[str, dict[str, Any]] = {}
    shared_fields = (
        "schema_version",
        "candidate_kind",
        "proposal_policy",
        "cohort_id",
        "contract_sha256",
        "canonical_group_id",
        "assembly_path",
        "source_visual_stable_properties_signature_sha256",
        "anchor_part_ids",
        "anchor_supporting_view_ids",
        "anchor_child_branches",
        "cohort_part_ids",
        "expanded_member_part_ids",
        "membership_status",
        "input_hashes",
    )
    for cohort_id, discovered in sorted(discovered_by_id.items()):
        raw_cohort = raw_by_id[cohort_id]
        candidate_kind = str(discovered["candidate_kind"])
        expected_lane = cohort_lane_by_candidate_kind.get(candidate_kind)
        raw_input_hashes = raw_cohort.get("input_hashes")
        if (
            expected_lane is None
            or any(
                raw_cohort.get(field) != discovered.get(field)
                for field in shared_fields
            )
            or raw_cohort.get("material_id")
            != discovered.get("expanded_material_id")
            or raw_cohort.get("accepted_part_ids")
            != discovered.get("cohort_part_ids")
            or not isinstance(raw_input_hashes, dict)
            or any(
                expected_input_hashes.get(key) != value
                for key, value in raw_input_hashes.items()
            )
        ):
            raise RuntimeError(
                "Quality-repair dominant assembly cohort audit does not match "
                "its sealed plan contract"
            )

        anchor_part_ids = set(discovered["anchor_part_ids"])
        cohort_part_ids = list(discovered["cohort_part_ids"])
        for part_id in cohort_part_ids:
            change = changes_by_part.get(part_id)
            lane = localization_lanes_by_part.get(part_id)
            member_role = (
                "strict_spatial_anchor"
                if part_id in anchor_part_ids
                else "expanded_member"
            )
            if (
                not isinstance(change, dict)
                or not isinstance(lane, dict)
                or lane.get("lane") != expected_lane
                or lane.get("canonical_group_id")
                != discovered.get("canonical_group_id")
                or change.get("canonical_group_id")
                != discovered.get("canonical_group_id")
                or change.get("material_id") != discovered.get("expanded_material_id")
                or change.get("dominant_assembly_cohort_id") != cohort_id
                or change.get("dominant_assembly_member_role") != member_role
                or part_id in validated_parts
            ):
                raise RuntimeError(
                    "Quality-repair dominant assembly cohort member is inconsistent"
                )
            expected_provenance = {
                field: copy.deepcopy(raw_cohort[field])
                for field in (
                    "schema_version",
                    "candidate_kind",
                    "cohort_id",
                    "contract_sha256",
                    "canonical_group_id",
                    "assembly_path",
                    "source_visual_stable_properties_signature_sha256",
                    "anchor_part_ids",
                    "anchor_supporting_view_ids",
                    "anchor_child_branches",
                    "cohort_part_ids",
                    "expanded_member_part_ids",
                    "membership_status",
                    "input_hashes",
                )
            }
            if isinstance(raw_cohort.get("proposal_policy"), str):
                expected_provenance["proposal_policy"] = raw_cohort[
                    "proposal_policy"
                ]
            expected_provenance.update(
                {
                    "member_role": member_role,
                    "baseline_material_id": change.get("old_material_id"),
                }
            )
            validated_parts[part_id] = {
                "cohort_id": cohort_id,
                "member_role": member_role,
                "provenance": expected_provenance,
            }

    if set(validated_parts) != lane_part_ids:
        raise RuntimeError(
            "Quality-repair dominant assembly cohorts do not cover their lanes"
        )
    return validated_parts


def _validate_quality_repair_bundle(
    *,
    plan: dict[str, Any],
    audit: dict[str, Any],
    baseline_plan: dict[str, Any],
    baseline_policy_audit: dict[str, Any],
    quality_report: dict[str, Any],
    palette_fusion: dict[str, Any],
    spatial_report: dict[str, Any],
    spatial_gate_audit: dict[str, Any],
    mapping_consensus: dict[str, Any],
    geometry_risk: dict[str, Any],
    group_materials: dict[str, Any],
    mvinverse_pbr_evidence: dict[str, Any],
    registry: dict[str, Any],
    whitelist: dict[str, Any],
    material_selection_objective: str = MATERIAL_SELECTION_OBJECTIVE_SEMANTIC,
) -> int:
    """Validate the repair compiler's complete trust boundary before Apply."""

    if audit.get("schema_version") != QUALITY_REPAIR_REPORT_SCHEMA_VERSION:
        raise RuntimeError("Quality-repair audit has an unsupported schema_version")
    summary = audit.get("summary")
    if not isinstance(summary, dict):
        raise RuntimeError("Quality-repair audit is missing its summary")
    changed_count = _require_exact_int(
        summary.get("changed_count"), "Quality-repair changed_count"
    )
    if (
        changed_count < 0
        or summary.get("exact_cover") is not True
        or summary.get("all_materials_in_whitelist") is not True
        or summary.get("maximum_orchestrator_retry_count") != 1
    ):
        raise RuntimeError("Quality-repair audit does not authorize one repair round")
    if changed_count == 0:
        if (
            summary.get("status") != "SAFE_NOOP"
            or summary.get("no_op") is not True
        ):
            raise RuntimeError("Quality-repair no-op audit is inconsistent")
    elif (
        summary.get("status") != "REPAIRED"
        or summary.get("no_op") is not False
    ):
        raise RuntimeError("Quality-repair audit does not authorize one repair round")

    if material_selection_objective not in {
        MATERIAL_SELECTION_OBJECTIVE_SEMANTIC,
        MATERIAL_SELECTION_OBJECTIVE_VISUAL,
    }:
        raise RuntimeError("Quality-repair material selection objective is invalid")
    visual_similarity_first = (
        material_selection_objective == MATERIAL_SELECTION_OBJECTIVE_VISUAL
    )
    if (
        audit.get(
            "material_selection_objective",
            MATERIAL_SELECTION_OBJECTIVE_SEMANTIC,
        )
        != material_selection_objective
    ):
        raise RuntimeError("Quality-repair material selection objective does not match")

    expected_hashes = {
        "baseline_plan_sha256": canonical_sha256(baseline_plan),
        "baseline_policy_audit_sha256": canonical_sha256(
            baseline_policy_audit
        ),
        "quality_report_sha256": canonical_sha256(quality_report),
        "palette_fusion_sha256": canonical_sha256(palette_fusion),
        "spatial_report_sha256": canonical_sha256(spatial_report),
        "spatial_gate_audit_sha256": canonical_sha256(spatial_gate_audit),
        "mapping_consensus_sha256": canonical_sha256(mapping_consensus),
        "geometry_risk_sha256": canonical_sha256(geometry_risk),
        "group_materials_sha256": canonical_sha256(group_materials),
        "mvinverse_pbr_evidence_sha256": canonical_sha256(
            mvinverse_pbr_evidence
        ),
        "registry_sha256": canonical_sha256(registry),
        "whitelist_sha256": canonical_sha256(whitelist),
    }
    if visual_similarity_first:
        expected_hashes["material_selection_objective"] = (
            material_selection_objective
        )
    if audit.get("input_hashes") != expected_hashes:
        raise RuntimeError("Quality-repair audit input hashes do not match")
    if audit.get("output_plan_sha256") != canonical_sha256(plan):
        raise RuntimeError("Quality-repair output plan hash does not match its audit")
    if changed_count == 0 and plan != baseline_plan:
        raise RuntimeError(
            "Quality-repair SAFE_NOOP must preserve the sealed baseline"
        )
    _validate_quality_dominant_mass(quality_report)

    baseline_provenance = baseline_plan.get("provenance")
    provenance = plan.get("provenance")
    if (
        plan.get("schema_version") != "1.0"
        or not isinstance(baseline_provenance, dict)
        or baseline_provenance.get("mode") != POLICY_PLAN_MODE
        or QUALITY_REPAIR_PROVENANCE_FIELD in baseline_provenance
        or not isinstance(provenance, dict)
    ):
        raise RuntimeError("Quality-repair plan has invalid exact-cover provenance")

    changes = audit.get("changes")
    if not isinstance(changes, list) or len(changes) != changed_count:
        raise RuntimeError("Quality-repair audit changes are inconsistent")
    changes_by_part: dict[str, dict[str, Any]] = {}
    changed_ids: list[str] = []
    for item in changes:
        if not isinstance(item, dict):
            raise RuntimeError("Quality-repair audit contains an invalid change")
        part_id = item.get("part_id")
        if (
            not isinstance(part_id, str)
            or not part_id
            or part_id in changes_by_part
        ):
            raise RuntimeError("Quality-repair changed part IDs are invalid")
        changes_by_part[part_id] = item
        changed_ids.append(part_id)
    sorted_changed_ids = sorted(changed_ids)
    if changed_ids != sorted_changed_ids or len(changes_by_part) != changed_count:
        raise RuntimeError("Quality-repair changed part IDs are invalid")

    raw_localization_lanes = audit.get("localization_lanes")
    if (
        not isinstance(raw_localization_lanes, list)
        or len(raw_localization_lanes) != changed_count
    ):
        raise RuntimeError("Quality-repair localization lanes are inconsistent")
    localization_lanes_by_part: dict[str, dict[str, str]] = {}
    for item in raw_localization_lanes:
        if not isinstance(item, dict) or set(item) != {
            "part_id",
            "canonical_group_id",
            "lane",
        }:
            raise RuntimeError("Quality-repair localization lane is invalid")
        part_id = item.get("part_id")
        canonical_group_id = item.get("canonical_group_id")
        lane = item.get("lane")
        if (
            not isinstance(part_id, str)
            or part_id not in changes_by_part
            or part_id in localization_lanes_by_part
            or not isinstance(canonical_group_id, str)
            or canonical_group_id
            != changes_by_part[part_id].get("canonical_group_id")
            or lane not in QUALITY_REPAIR_LOCALIZATION_LANES
        ):
            raise RuntimeError("Quality-repair localization lane is invalid")
        localization_lanes_by_part[part_id] = {
            "part_id": part_id,
            "canonical_group_id": canonical_group_id,
            "lane": str(lane),
        }
    if [item.get("part_id") for item in raw_localization_lanes] != sorted_changed_ids:
        raise RuntimeError("Quality-repair localization lanes are not deterministic")

    anchored_single_part_ids = {
        part_id
        for part_id, item in localization_lanes_by_part.items()
        if item["lane"] == QUALITY_REPAIR_ANCHORED_SINGLE_VIEW_LANE
    }
    multiview_semantic_review_part_ids = {
        part_id
        for part_id, item in localization_lanes_by_part.items()
        if item["lane"] == QUALITY_REPAIR_MULTIVIEW_SEMANTIC_REVIEW_LANE
    }
    dark_foreground_part_ids = {
        part_id
        for part_id, item in localization_lanes_by_part.items()
        if item["lane"] == QUALITY_REPAIR_DARK_FOREGROUND_RESIDUAL_LANE
    }
    multiview_dark_identity_part_ids = {
        part_id
        for part_id, item in localization_lanes_by_part.items()
        if item["lane"] == QUALITY_REPAIR_MULTIVIEW_DARK_IDENTITY_LANE
    }
    dark_residual_part_ids = (
        dark_foreground_part_ids | multiview_dark_identity_part_ids
    )
    repeated_geometry_dark_part_ids = {
        part_id
        for part_id, item in localization_lanes_by_part.items()
        if item["lane"] == QUALITY_REPAIR_REPEATED_GEOMETRY_DARK_RESIDUAL_LANE
    }
    source_identity_part_ids = {
        part_id
        for part_id, item in localization_lanes_by_part.items()
        if item["lane"]
        in {
            QUALITY_REPAIR_SOURCE_IDENTITY_LANE,
            QUALITY_REPAIR_SOURCE_IDENTITY_COHORT_CONSENSUS_LANE,
        }
    }
    group_diagnostics_by_id: dict[str, dict[str, Any]] = {}
    if (
        anchored_single_part_ids
        or multiview_semantic_review_part_ids
        or dark_residual_part_ids
        or repeated_geometry_dark_part_ids
        or source_identity_part_ids
    ):
        raw_group_diagnostics = audit.get("group_diagnostics")
        if not isinstance(raw_group_diagnostics, list):
            raise RuntimeError("Quality repair lacks canonical-group diagnostics")
        for diagnostic in raw_group_diagnostics:
            if not isinstance(diagnostic, dict):
                raise RuntimeError("Quality-repair group diagnostic is invalid")
            group_id = diagnostic.get("canonical_group_id")
            if (
                not isinstance(group_id, str)
                or not group_id
                or group_id in group_diagnostics_by_id
            ):
                raise RuntimeError("Quality-repair group diagnostic is invalid")
            group_diagnostics_by_id[group_id] = diagnostic

    expected_plan_provenance = copy.deepcopy(baseline_provenance)
    if changed_count:
        expected_plan_provenance[QUALITY_REPAIR_PROVENANCE_FIELD] = {
            "mode": QUALITY_REPAIR_PLAN_MODE,
            "input_hashes": expected_hashes,
            "changed_part_ids": sorted_changed_ids,
        }
    if provenance != expected_plan_provenance:
        raise RuntimeError("Quality-repair plan provenance is not the authorized delta")

    raw_registry_parts = registry.get("parts")
    raw_baseline_assignments = baseline_plan.get("assignments")
    raw_output_assignments = plan.get("assignments")
    allowed_material_ids = whitelist.get("material_ids")
    canonical_palette = palette_fusion.get("canonical_palette")
    raw_canonical_groups = (
        canonical_palette.get("groups")
        if isinstance(canonical_palette, dict)
        else None
    )
    raw_group_selections = group_materials.get("selections")
    if (
        not isinstance(raw_registry_parts, list)
        or not isinstance(raw_baseline_assignments, list)
        or not isinstance(raw_output_assignments, list)
        or not isinstance(allowed_material_ids, list)
        or any(not isinstance(value, str) for value in allowed_material_ids)
        or len(set(allowed_material_ids)) != len(allowed_material_ids)
        or not isinstance(raw_canonical_groups, list)
        or not isinstance(raw_group_selections, list)
    ):
        raise RuntimeError("Quality-repair exact-cover inputs are invalid")
    canonical_groups: dict[str, dict[str, Any]] = {}
    for group in raw_canonical_groups:
        if not isinstance(group, dict):
            raise RuntimeError("Quality-repair canonical palette is invalid")
        group_id = group.get("group_id")
        visual_description = group.get("visual_description")
        if (
            not isinstance(group_id, str)
            or not group_id
            or group_id in canonical_groups
            or not isinstance(visual_description, str)
            or not visual_description.strip()
        ):
            raise RuntimeError("Quality-repair canonical palette is invalid")
        canonical_groups[group_id] = group
    group_selections: dict[str, dict[str, Any]] = {}
    for selection in raw_group_selections:
        if not isinstance(selection, dict):
            raise RuntimeError("Quality-repair group-material selections are invalid")
        group_id = selection.get("group_id")
        if (
            not isinstance(group_id, str)
            or not group_id
        ):
            raise RuntimeError("Quality-repair group-material selections are invalid")
        group_selections[group_id] = selection

    dark_references: dict[str, dict[str, Any]] = {}
    dark_budget_by_group: dict[str, dict[str, Any]] = {}
    minimum_semantic_confidence = math.inf
    if dark_residual_part_ids:
        eligible_black_group_ids = sorted(
            group_id
            for group_id, group in canonical_groups.items()
            if str(group.get("base_color", "")).strip().casefold() == "black"
            and group.get("singleton") is not True
            and isinstance(group.get("distinct_view_count"), int)
            and not isinstance(group.get("distinct_view_count"), bool)
            and int(group["distinct_view_count"]) >= 2
        )
        dark_changed_group_ids = sorted(
            {
                localization_lanes_by_part[part_id]["canonical_group_id"]
                for part_id in dark_residual_part_ids
            }
        )
        if (
            len(eligible_black_group_ids) != 1
            or dark_changed_group_ids != eligible_black_group_ids
        ):
            raise RuntimeError(
                "Dark-foreground repair lacks one unique multiview black group"
            )
        dark_references = _quality_dark_reference_evidence(spatial_report)
        spatial_policy = spatial_report.get("policy")
        if not isinstance(spatial_policy, dict):
            raise RuntimeError("Dark-foreground repair lacks a spatial policy")
        minimum_semantic_confidence = _quality_unit(
            spatial_policy.get("minimum_semantic_confidence"),
            "Dark-foreground semantic confidence floor",
        )
        raw_dark_budgets = audit.get("dark_residual_budgets")
        if not isinstance(raw_dark_budgets, list):
            raise RuntimeError("Dark-foreground repair lacks a contribution budget")
        for budget in raw_dark_budgets:
            if not isinstance(budget, dict):
                raise RuntimeError("Dark-foreground repair budget is invalid")
            group_id = budget.get("canonical_group_id")
            if (
                not isinstance(group_id, str)
                or not group_id
                or group_id in dark_budget_by_group
            ):
                raise RuntimeError("Dark-foreground repair budget is invalid")
            dark_budget_by_group[group_id] = budget
        if not set(dark_changed_group_ids) <= set(dark_budget_by_group):
            raise RuntimeError(
                "Dark-foreground repair lacks its canonical-group budget"
            )

    registry_by_part = {
        item.get("part_id"): item
        for item in raw_registry_parts
        if isinstance(item, dict) and isinstance(item.get("part_id"), str)
    }
    registry_ids = set(registry_by_part)
    baseline_by_part = {
        item.get("part_id"): item
        for item in raw_baseline_assignments
        if isinstance(item, dict) and isinstance(item.get("part_id"), str)
    }
    output_by_part = {
        item.get("part_id"): item
        for item in raw_output_assignments
        if isinstance(item, dict) and isinstance(item.get("part_id"), str)
    }
    if (
        len(registry_ids) != len(raw_registry_parts)
        or set(baseline_by_part) != registry_ids
        or len(baseline_by_part) != len(raw_baseline_assignments)
        or set(output_by_part) != registry_ids
        or len(output_by_part) != len(raw_output_assignments)
        or [
            item.get("part_id") for item in raw_output_assignments
        ]
        != [item.get("part_id") for item in raw_baseline_assignments]
        or not set(sorted_changed_ids) <= registry_ids
    ):
        raise RuntimeError("Quality-repair plan is not an exact registry cover")

    validated_repeated_dark_parts = _validate_repeated_geometry_dark_cohorts(
        audit=audit,
        quality_report=quality_report,
        spatial_report=spatial_report,
        spatial_gate_audit=spatial_gate_audit,
        mapping_consensus=mapping_consensus,
        geometry_risk=geometry_risk,
        registry_parts=raw_registry_parts,
        canonical_groups=canonical_groups,
        group_diagnostics=group_diagnostics_by_id,
        changes_by_part=changes_by_part,
        localization_lanes_by_part=localization_lanes_by_part,
    )
    validated_dominant_assembly_parts = (
        _validate_quality_repair_dominant_assembly_cohorts(
            plan=plan,
            audit=audit,
            palette_fusion=palette_fusion,
            expected_input_hashes=expected_hashes,
            changes_by_part=changes_by_part,
            localization_lanes_by_part=localization_lanes_by_part,
        )
    )

    allowed_id_set = set(allowed_material_ids)
    parameterized_part_ids: list[str] = []
    validated_dark_changes: dict[str, dict[str, Any]] = {}

    def decision_for_part(
        document: dict[str, Any],
        part_id: str,
        *,
        label: str,
    ) -> dict[str, Any] | None:
        raw_decisions = document.get("decisions")
        if not isinstance(raw_decisions, list):
            raise RuntimeError(f"{label} decisions are invalid")
        matches = [
            item
            for item in raw_decisions
            if isinstance(item, dict) and item.get("part_id") == part_id
        ]
        if len(matches) > 1:
            raise RuntimeError(f"{label} decision is not unique for {part_id}")
        return matches[0] if matches else None

    def multiview_dark_record(
        part_id: str,
        canonical_group_id: str,
    ) -> tuple[dict[str, Any], list[str], list[str]]:
        """Recompute the multiview dark-identity evidence from source audits."""

        group_diagnostic = group_diagnostics_by_id.get(canonical_group_id)
        raw_dark_supports = (
            group_diagnostic.get("dark_residual_supporting_views")
            if isinstance(group_diagnostic, dict)
            else None
        )
        if (
            not isinstance(group_diagnostic, dict)
            or group_diagnostic.get("dark_residual_repairable") is not True
            or group_diagnostic.get("dark_residual_reason_codes") != []
            or not isinstance(raw_dark_supports, list)
            or len(raw_dark_supports) != 1
            or not isinstance(raw_dark_supports[0], dict)
        ):
            raise RuntimeError(
                "Multiview dark-identity change lacks one repairable QA residual"
            )
        diagnostic, support_views, semantic_override_views = (
            _recompute_quality_multiview_dark_identity(
                part_id=part_id,
                canonical_group_id=canonical_group_id,
                canonical_group=canonical_groups[canonical_group_id],
                spatial_part=_quality_dark_spatial_part(spatial_report, part_id),
                spatial_gate_decision=decision_for_part(
                    spatial_gate_audit,
                    part_id,
                    label="Spatial gate",
                ),
                mapping_decision=decision_for_part(
                mapping_consensus,
                    part_id,
                    label="Mapping consensus",
                ),
                dark_residual_support=raw_dark_supports[0],
                reference_evidence=dark_references,
                spatial_policy=spatial_policy,
                minimum_semantic_confidence=minimum_semantic_confidence,
            )
        )
        return diagnostic, support_views, semantic_override_views

    for part_id in sorted(registry_ids):
        baseline_assignment = baseline_by_part[part_id]
        output_assignment = output_by_part[part_id]
        if part_id not in changes_by_part:
            if output_assignment != baseline_assignment:
                raise RuntimeError(
                    f"Quality repair changed an unauthorized part: {part_id}"
                )
            continue

        baseline_item_provenance = baseline_assignment.get("provenance")
        baseline_confidence = baseline_assignment.get("confidence")
        if (
            baseline_assignment.get("status") != POLICY_FALLBACK_STATUS
            or isinstance(baseline_confidence, bool)
            or not isinstance(baseline_confidence, (int, float))
            or float(baseline_confidence) != 0.0
            or baseline_assignment.get("evidence_views") != []
            or not isinstance(baseline_item_provenance, dict)
            or baseline_item_provenance.get("tier") not in NEUTRAL_FALLBACK_TIERS
            or baseline_item_provenance.get("output_confidence_basis")
            != POLICY_FALLBACK_CONFIDENCE_BASIS
            or baseline_item_provenance.get("canonical_group_id") is not None
            or baseline_assignment.get("material_id") not in allowed_id_set
            or "parameters" in baseline_assignment
            or "face_subsets" in baseline_assignment
        ):
            raise RuntimeError(
                f"Quality-repair baseline is not neutral fallback for part {part_id}"
            )

        change = changes_by_part[part_id]
        lane = localization_lanes_by_part[part_id]["lane"]
        dominant_assembly_record = validated_dominant_assembly_parts.get(part_id)
        canonical_group_id = change.get("canonical_group_id")
        old_material_id = change.get("old_material_id")
        new_material_id = change.get("new_material_id")
        duplicated_material_id = change.get("material_id")
        if (
            not isinstance(canonical_group_id, str)
            or canonical_group_id not in canonical_groups
            or old_material_id != baseline_assignment.get("material_id")
            or not isinstance(new_material_id, str)
            or new_material_id == old_material_id
            or duplicated_material_id != new_material_id
        ):
            raise RuntimeError(
                f"Quality-repair audit material delta is invalid for part {part_id}"
            )
        selection = group_selections.get(canonical_group_id)
        parameterized = change.get("mvinverse_parameterized") is True
        confirmed_source_material_id = change.get(
            "confirmed_source_material_id"
        )
        provisional_material = (
            change.get("material_selection_basis")
            == QUALITY_REPAIR_PROVISIONAL_MATERIAL_BASIS
        )
        selection_confidence = (
            selection.get("confidence") if isinstance(selection, dict) else None
        )
        confirmed_material = (
            isinstance(selection, dict) and selection.get("confirmed") is True
        )
        high_confidence_provisional = (
            provisional_material
            and isinstance(selection, dict)
            and selection.get("confirmed") is False
            and isinstance(selection_confidence, (int, float))
            and not isinstance(selection_confidence, bool)
            and math.isfinite(float(selection_confidence))
            and QUALITY_REPAIR_MIN_PROVISIONAL_MATERIAL_CONFIDENCE
            <= float(selection_confidence)
            <= 1.0
            and selection.get("material_id") == new_material_id
            and not parameterized
        )
        if (
            selection is None
            or not (confirmed_material or high_confidence_provisional)
            or new_material_id not in allowed_id_set
        ):
            raise RuntimeError(
                "Quality-repair material is neither a confirmed selection nor "
                "a bounded high-confidence whitelist candidate "
                f"for part {part_id}"
            )
        if confirmed_material and provisional_material:
            raise RuntimeError(
                "Quality-repair confirmed material is mislabeled provisional "
                f"for part {part_id}"
            )

        support_fields = (
            "supporting_view_ids",
            "supporting_content_cluster_ids",
            "supporting_pose_cluster_ids",
        )
        exact_single_support = lane in {
            "exact_spatial_single_qa_view",
            QUALITY_REPAIR_SEMANTIC_SINGLE_VIEW_LANE,
            QUALITY_REPAIR_ANCHORED_SINGLE_VIEW_LANE,
            QUALITY_REPAIR_SPATIAL_ANCHOR_SINGLE_VIEW_LANE,
            QUALITY_REPAIR_DOMINANT_RESIDUAL_SINGLE_VIEW_LANE,
            QUALITY_REPAIR_DARK_FOREGROUND_RESIDUAL_LANE,
            QUALITY_REPAIR_REPEATED_GEOMETRY_DARK_RESIDUAL_LANE,
        }
        supports: dict[str, list[str]] = {}
        for field in support_fields:
            raw_support = change.get(field)
            if (
                not isinstance(raw_support, list)
                or (
                    len(raw_support) != 1
                    if exact_single_support
                    else len(raw_support) < 2
                )
                or any(not isinstance(value, str) or not value for value in raw_support)
                or raw_support != sorted(set(raw_support))
            ):
                raise RuntimeError(
                    f"Quality-repair audit {field} is invalid for part {part_id}"
                )
            supports[field] = raw_support

        if lane == QUALITY_REPAIR_MULTIVIEW_SEMANTIC_REVIEW_LANE:
            diagnostic = group_diagnostics_by_id.get(canonical_group_id)
            raw_deficit_supports = (
                diagnostic.get("supporting_views")
                if isinstance(diagnostic, dict)
                else None
            )
            if (
                not isinstance(diagnostic, dict)
                or diagnostic.get("repairable") is not True
                or not isinstance(raw_deficit_supports, list)
            ):
                raise RuntimeError(
                    "Semantic-review repair lacks a repairable QA group deficit"
                )
            deficit_view_ids = sorted(
                str(item["reference_view_id"])
                for item in raw_deficit_supports
                if isinstance(item, dict)
                and isinstance(item.get("reference_view_id"), str)
            )
            expected_semantic_support = (
                _validate_quality_multiview_semantic_review_support(
                    spatial_report=spatial_report,
                    spatial_gate_audit=spatial_gate_audit,
                    mapping_consensus=mapping_consensus,
                    part_id=part_id,
                    canonical_group_id=canonical_group_id,
                    deficit_view_ids=deficit_view_ids,
                )
            )
            if supports != expected_semantic_support:
                raise RuntimeError(
                    "Semantic-review repair support does not match trusted evidence"
                )

        spatial_anchor_view_ids: list[str] | None = None
        if lane == QUALITY_REPAIR_SPATIAL_ANCHOR_SINGLE_VIEW_LANE:
            expected_spatial_anchor_view_ids = (
                _independent_quality_spatial_anchor_view_ids(
                    spatial_report=spatial_report,
                    part_id=part_id,
                    canonical_group_id=canonical_group_id,
                    target_view_id=supports["supporting_view_ids"][0],
                )
            )
            raw_spatial_anchor_view_ids = change.get(
                "spatial_anchor_view_ids"
            )
            if (
                not expected_spatial_anchor_view_ids
                or raw_spatial_anchor_view_ids
                != expected_spatial_anchor_view_ids
            ):
                raise RuntimeError(
                    "Quality-repair single-view change lacks an independent "
                    f"spatial anchor for part {part_id}"
                )
            spatial_anchor_view_ids = expected_spatial_anchor_view_ids
        elif "spatial_anchor_view_ids" in change:
            raise RuntimeError(
                "Quality-repair spatial anchor is unauthorized "
                f"for part {part_id}"
            )

        anchor_part_ids: list[str] | None = None
        anchor_supporting_view_ids: list[str] | None = None
        if lane == QUALITY_REPAIR_ANCHORED_SINGLE_VIEW_LANE:
            canonical_group = canonical_groups[canonical_group_id]
            diagnostic = group_diagnostics_by_id.get(canonical_group_id)
            raw_deficit_supports = (
                diagnostic.get("supporting_views")
                if isinstance(diagnostic, dict)
                else None
            )
            if (
                canonical_group.get("singleton") is True
                or isinstance(canonical_group.get("distinct_view_count"), bool)
                or not isinstance(canonical_group.get("distinct_view_count"), int)
                or int(canonical_group["distinct_view_count"]) < 2
                or not isinstance(diagnostic, dict)
                or diagnostic.get("repairable") is not True
                or not isinstance(raw_deficit_supports, list)
            ):
                raise RuntimeError(
                    "Anchored quality repair lacks a repairable multiview deficit "
                    f"for part {part_id}"
                )
            deficit_view_ids: list[str] = []
            for raw_support in raw_deficit_supports:
                view_id = (
                    raw_support.get("reference_view_id")
                    if isinstance(raw_support, dict)
                    else None
                )
                if not isinstance(view_id, str) or not view_id:
                    raise RuntimeError(
                        "Anchored quality-repair deficit support is invalid"
                    )
                deficit_view_ids.append(view_id)
            deficit_view_ids = sorted(deficit_view_ids)
            if (
                len(deficit_view_ids) < 2
                or deficit_view_ids != sorted(set(deficit_view_ids))
                or not set(supports["supporting_view_ids"])
                < set(deficit_view_ids)
            ):
                raise RuntimeError(
                    "Anchored quality-repair single support is invalid "
                    f"for part {part_id}"
                )

            raw_anchor_part_ids = change.get("anchor_part_ids")
            raw_anchor_supporting_view_ids = change.get(
                "anchor_supporting_view_ids"
            )
            if (
                not isinstance(raw_anchor_part_ids, list)
                or not raw_anchor_part_ids
                or raw_anchor_part_ids != sorted(set(raw_anchor_part_ids))
                or any(
                    not isinstance(anchor_id, str) or not anchor_id
                    for anchor_id in raw_anchor_part_ids
                )
                or not isinstance(raw_anchor_supporting_view_ids, list)
                or raw_anchor_supporting_view_ids != deficit_view_ids
            ):
                raise RuntimeError(
                    "Anchored quality-repair audit fields are invalid "
                    f"for part {part_id}"
                )

            expected_anchor_part_ids = sorted(
                anchor_id
                for anchor_id, anchor_change in changes_by_part.items()
                if anchor_id != part_id
                and anchor_change.get("canonical_group_id") == canonical_group_id
                and localization_lanes_by_part[anchor_id]["lane"]
                in {
                    "stable_spatial_multiview",
                    "bounded_spatial_multiview",
                }
                and isinstance(anchor_change.get("supporting_view_ids"), list)
                and set(anchor_change["supporting_view_ids"])
                <= set(deficit_view_ids)
            )
            expected_anchor_view_ids = sorted(
                {
                    view_id
                    for anchor_id in expected_anchor_part_ids
                    for view_id in changes_by_part[anchor_id][
                        "supporting_view_ids"
                    ]
                }
            )
            if (
                raw_anchor_part_ids != expected_anchor_part_ids
                or not expected_anchor_part_ids
                or expected_anchor_view_ids != deficit_view_ids
            ):
                raise RuntimeError(
                    "Anchored quality repair lacks a complete existing multiview "
                    f"anchor for part {part_id}"
                )
            anchor_part_ids = raw_anchor_part_ids
            anchor_supporting_view_ids = raw_anchor_supporting_view_ids
        elif (
            "anchor_part_ids" in change
            or "anchor_supporting_view_ids" in change
        ):
            raise RuntimeError(
                f"Quality-repair anchor fields are unauthorized for part {part_id}"
            )

        source_identity_provenance: dict[str, Any] | None = None
        source_identity_fields = (
            "source_identity_anchor_part_ids",
            "source_identity_anchor_supporting_view_ids",
            "source_visual_stable_properties_signature_sha256",
            "source_identity_assembly_path",
            "source_identity_cohort_part_ids",
            "source_identity_signature_count",
        )
        source_identity_consensus_lane = (
            lane == QUALITY_REPAIR_SOURCE_IDENTITY_COHORT_CONSENSUS_LANE
        )
        if source_identity_consensus_lane:
            source_identity_fields = (
                *source_identity_fields,
                "source_identity_consensus_view_ids",
                "source_identity_consensus_content_cluster_ids",
                "source_identity_consensus_pose_cluster_ids",
                "source_identity_consensus_mode",
            )
        if lane in {
            QUALITY_REPAIR_SOURCE_IDENTITY_LANE,
            QUALITY_REPAIR_SOURCE_IDENTITY_COHORT_CONSENSUS_LANE,
        }:
            diagnostic = group_diagnostics_by_id.get(canonical_group_id)
            raw_deficit_supports = (
                diagnostic.get("supporting_views")
                if isinstance(diagnostic, dict)
                else None
            )
            if (
                not isinstance(diagnostic, dict)
                or diagnostic.get("repairable") is not True
                or not isinstance(raw_deficit_supports, list)
            ):
                raise RuntimeError(
                    "Source-identity repair lacks a repairable multiview deficit"
                )
            deficit_view_ids = sorted(
                str(item["reference_view_id"])
                for item in raw_deficit_supports
                if isinstance(item, dict)
                and isinstance(item.get("reference_view_id"), str)
            )
            if (
                len(deficit_view_ids) < 2
                or deficit_view_ids != sorted(set(deficit_view_ids))
                or len(supports["supporting_view_ids"]) != 1
                or not set(supports["supporting_view_ids"]) < set(deficit_view_ids)
            ):
                raise RuntimeError(
                    "Source-identity diagnostic support is not independently bounded"
                )

            candidate_part = registry_by_part[part_id]
            expected_signature = _quality_repeated_source_signature(candidate_part)
            candidate_parent = candidate_part.get("parent_path")
            if not isinstance(candidate_parent, str) or not candidate_parent:
                raise RuntimeError("Source-identity candidate lacks an assembly parent")
            expected_assembly_path = (
                candidate_parent.rsplit("/", 1)[0]
                if "/" in candidate_parent[1:]
                else "/"
            )

            source_signatures: dict[str, str] = {}
            for registry_part_id, registry_part in registry_by_part.items():
                try:
                    source_signatures[str(registry_part_id)] = (
                        _quality_repeated_source_signature(registry_part)
                    )
                except RuntimeError:
                    continue
            expected_signature_count = sum(
                signature == expected_signature
                for signature in source_signatures.values()
            )

            def assembly_path_for(candidate: dict[str, Any]) -> str | None:
                raw_parent = candidate.get("parent_path")
                if not isinstance(raw_parent, str) or not raw_parent:
                    return None
                return (
                    raw_parent.rsplit("/", 1)[0]
                    if "/" in raw_parent[1:]
                    else "/"
                )

            expected_cohort_part_ids = sorted(
                candidate_id
                for candidate_id, candidate in registry_by_part.items()
                if source_signatures.get(str(candidate_id)) == expected_signature
                and assembly_path_for(candidate) == expected_assembly_path
            )
            expected_anchor_part_ids = (
                sorted(set(expected_cohort_part_ids) - {part_id})
                if source_identity_consensus_lane
                else sorted(
                    candidate_id
                    for candidate_id, candidate_change in changes_by_part.items()
                    if candidate_id != part_id
                    and candidate_change.get("canonical_group_id")
                    == canonical_group_id
                    and localization_lanes_by_part[candidate_id]["lane"]
                    in {
                        "stable_spatial_multiview",
                        "bounded_spatial_multiview",
                        QUALITY_REPAIR_MULTIVIEW_SEMANTIC_REVIEW_LANE,
                    }
                    and candidate_id in expected_cohort_part_ids
                    and candidate_change.get("supporting_view_ids")
                    == deficit_view_ids
                )
            )
            maximum_signature_count = max(
                QUALITY_REPAIR_SOURCE_IDENTITY_MIN_SIGNATURE_COUNT,
                math.floor(
                    QUALITY_REPAIR_SOURCE_IDENTITY_MAX_REGISTRY_FRACTION
                    * len(registry_by_part)
                ),
            )
            if (
                not expected_anchor_part_ids
                or not (
                    QUALITY_REPAIR_SOURCE_IDENTITY_MIN_SIGNATURE_COUNT
                    <= expected_signature_count
                    <= maximum_signature_count
                )
                or not (
                    QUALITY_REPAIR_SOURCE_IDENTITY_MIN_SIGNATURE_COUNT
                    <= len(expected_cohort_part_ids)
                    <= QUALITY_REPAIR_SOURCE_IDENTITY_MAX_ASSEMBLY_COHORT_SIZE
                )
                or set(expected_cohort_part_ids)
                != {part_id, *expected_anchor_part_ids}
                or (
                    source_identity_consensus_lane
                    and any(
                        candidate_id not in changes_by_part
                        or changes_by_part[candidate_id].get(
                            "canonical_group_id"
                        )
                        != canonical_group_id
                        or localization_lanes_by_part[candidate_id]["lane"]
                        != QUALITY_REPAIR_SOURCE_IDENTITY_COHORT_CONSENSUS_LANE
                        for candidate_id in expected_cohort_part_ids
                    )
                )
            ):
                raise RuntimeError(
                    "Source-identity repair lacks one rare exact assembly cohort"
                )
            expected_anchor_supporting_view_ids = deficit_view_ids
            consensus_fields: dict[str, Any] = {}
            if source_identity_consensus_lane:
                raw_spatial_parts = spatial_report.get("parts")
                raw_reference_evidence = spatial_report.get("reference_evidence")
                spatial_policy = spatial_report.get("policy")
                if (
                    not isinstance(raw_spatial_parts, list)
                    or not isinstance(raw_reference_evidence, list)
                    or not isinstance(spatial_policy, dict)
                ):
                    raise RuntimeError(
                        "Source-identity cohort consensus lacks spatial evidence"
                    )
                spatial_parts_by_id = {
                    item.get("part_id"): item
                    for item in raw_spatial_parts
                    if isinstance(item, dict)
                    and isinstance(item.get("part_id"), str)
                }
                reference_evidence_by_id = {
                    item.get("view_id"): item
                    for item in raw_reference_evidence
                    if isinstance(item, dict)
                    and isinstance(item.get("view_id"), str)
                }
                source_identity_consensus_confidence = _quality_unit(
                    spatial_policy.get("minimum_semantic_conflict_confidence"),
                    "Source-identity cohort review confidence floor",
                )
                consensus_view_ids: set[str] = set()
                direct_consensus_view_ids: set[str] = set()
                conflict_group_ids: set[str] = set()
                for cohort_part_id in expected_cohort_part_ids:
                    spatial_part = spatial_parts_by_id.get(cohort_part_id)
                    raw_votes = (
                        spatial_part.get("semantic_votes")
                        if isinstance(spatial_part, dict)
                        else None
                    )
                    if not isinstance(raw_votes, list):
                        raise RuntimeError(
                            "Source-identity cohort member lacks semantic votes"
                        )
                    for vote in raw_votes:
                        if not isinstance(vote, dict):
                            continue
                        effective_confidence = vote.get("effective_confidence")
                        view_id = vote.get("view_id")
                        vote_group_id = vote.get("canonical_group_id")
                        if not (
                            vote.get("alignment_trusted") is True
                            and vote.get("unique_canonical_join") is True
                            and isinstance(view_id, str)
                            and view_id in reference_evidence_by_id
                            and isinstance(vote_group_id, str)
                            and isinstance(
                                effective_confidence, (int, float)
                            )
                            and not isinstance(effective_confidence, bool)
                            and math.isfinite(float(effective_confidence))
                            and float(effective_confidence)
                            >= source_identity_consensus_confidence
                        ):
                            continue
                        if (
                            vote_group_id == canonical_group_id
                            and vote.get("status") in {"matched", "review"}
                        ):
                            consensus_view_ids.add(view_id)
                            if vote.get("reason_code") == "direct_visual_match":
                                direct_consensus_view_ids.add(view_id)
                        elif (
                            vote_group_id != canonical_group_id
                            and vote.get("status") == "matched"
                        ):
                            conflict_group_ids.add(vote_group_id)
                consensus_content_cluster_ids = sorted(
                    {
                        reference_evidence_by_id[view_id].get(
                            "content_cluster_id"
                        )
                        for view_id in consensus_view_ids
                    }
                )
                consensus_pose_cluster_ids = sorted(
                    {
                        reference_evidence_by_id[view_id].get("pose_cluster_id")
                        for view_id in consensus_view_ids
                    }
                )
                independent_multiview_consensus = (
                    len(consensus_view_ids) >= 2
                    and len(consensus_content_cluster_ids) >= 2
                    and len(consensus_pose_cluster_ids) >= 2
                )
                visual_qa_bridge = (
                    visual_similarity_first
                    and len(deficit_view_ids) >= 2
                    and len(consensus_view_ids) >= 1
                    and len(direct_consensus_view_ids) >= 1
                    and len(consensus_content_cluster_ids) >= 1
                    and len(consensus_pose_cluster_ids) >= 1
                )
                if (
                    conflict_group_ids
                    or None in consensus_content_cluster_ids
                    or None in consensus_pose_cluster_ids
                    or not (
                        independent_multiview_consensus
                        or visual_qa_bridge
                    )
                ):
                    raise RuntimeError(
                        "Source-identity cohort lacks independently bounded "
                        "visual consensus"
                    )
                consensus_mode = (
                    "independent_multiview_semantic_consensus"
                    if independent_multiview_consensus
                    else "direct_visual_anchor_plus_multiview_qa_deficit"
                )
                expected_anchor_supporting_view_ids = sorted(
                    consensus_view_ids
                )
                consensus_fields = {
                    "source_identity_consensus_view_ids": sorted(
                        consensus_view_ids
                    ),
                    "source_identity_consensus_content_cluster_ids": (
                        consensus_content_cluster_ids
                    ),
                    "source_identity_consensus_pose_cluster_ids": (
                        consensus_pose_cluster_ids
                    ),
                    "source_identity_consensus_mode": consensus_mode,
                }
            expected_source_identity = {
                "source_identity_anchor_part_ids": expected_anchor_part_ids,
                "source_identity_anchor_supporting_view_ids": (
                    expected_anchor_supporting_view_ids
                ),
                "source_visual_stable_properties_signature_sha256": (
                    expected_signature
                ),
                "source_identity_assembly_path": expected_assembly_path,
                "source_identity_cohort_part_ids": expected_cohort_part_ids,
                "source_identity_signature_count": expected_signature_count,
                **consensus_fields,
            }
            if any(
                change.get(field) != expected_source_identity[field]
                for field in source_identity_fields
            ):
                raise RuntimeError(
                    "Source-identity repair audit does not match the registry"
                )
            source_identity_provenance = {
                "lane": lane,
                **expected_source_identity,
            }
        elif any(field in change for field in source_identity_fields):
            raise RuntimeError(
                f"Source-identity fields are unauthorized for part {part_id}"
            )

        dark_residual_support: dict[str, Any] | None = None
        dark_diagnostic_summary: dict[str, Any] | None = None
        dark_budget_fields: dict[str, int] | None = None
        multiview_dark_semantic_override_view_ids: list[str] | None = None
        if lane == QUALITY_REPAIR_MULTIVIEW_DARK_IDENTITY_LANE:
            group_diagnostic = group_diagnostics_by_id.get(canonical_group_id)
            raw_dark_supports = (
                group_diagnostic.get("dark_residual_supporting_views")
                if isinstance(group_diagnostic, dict)
                else None
            )
            if (
                not isinstance(raw_dark_supports, list)
                or len(raw_dark_supports) != 1
                or not isinstance(raw_dark_supports[0], dict)
            ):
                raise RuntimeError(
                    "Multiview dark-identity change lacks one QA residual"
                )
            global_support_view_id = raw_dark_supports[0].get(
                "reference_view_id"
            )
            spatial_reference = dark_references.get(global_support_view_id)
            if not isinstance(global_support_view_id, str) or not isinstance(
                spatial_reference, dict
            ):
                raise RuntimeError(
                    "Multiview dark-identity change lacks its trusted QA reference"
                )
            expected_dark_support = _quality_dark_support_from_report(
                quality_report=quality_report,
                view_id=global_support_view_id,
                canonical_group_id=canonical_group_id,
                spatial_reference=spatial_reference,
            )
            dark_residual_support = change.get("dark_residual_support")
            if (
                raw_dark_supports[0] != expected_dark_support
                or dark_residual_support != expected_dark_support
            ):
                raise RuntimeError(
                    "Multiview dark-identity residual does not match QA evidence"
                )
            (
                dark_diagnostic_summary,
                expected_support_view_ids,
                multiview_dark_semantic_override_view_ids,
            ) = multiview_dark_record(part_id, canonical_group_id)
            expected_support_records = [
                dark_references[view_id] for view_id in expected_support_view_ids
            ]
            expected_multiview_supports = {
                "supporting_view_ids": expected_support_view_ids,
                "supporting_content_cluster_ids": sorted(
                    {
                        str(record["content_cluster_id"])
                        for record in expected_support_records
                    }
                ),
                "supporting_pose_cluster_ids": sorted(
                    {
                        str(record["pose_cluster_id"])
                        for record in expected_support_records
                    }
                ),
            }
            if (
                supports != expected_multiview_supports
                or change.get("dark_foreground_diagnostic")
                != dark_diagnostic_summary
            ):
                raise RuntimeError(
                    "Multiview dark-identity diagnostic does not match spatial "
                    f"evidence for {part_id}"
                )
            dark_budget_fields = {
                field: _require_exact_int(
                    change.get(field),
                    f"Multiview dark-identity {field}",
                )
                for field in (
                    "budget_pixels",
                    "budget_limit_pixels",
                    "existing_contribution_pixels",
                    "estimated_contribution_pixels",
                    "selected_contribution_pixels",
                    "cumulative_contribution_pixels",
                )
            }
            estimated_contribution = _require_exact_int(
                dark_diagnostic_summary.get("estimated_contribution_pixels"),
                "Multiview dark-identity estimated contribution",
                minimum=1,
            )
            if (
                dark_budget_fields["budget_pixels"]
                != expected_dark_support["budget_pixels"]
                or dark_budget_fields["budget_limit_pixels"]
                != expected_dark_support["budget_limit_pixels"]
                or dark_budget_fields["estimated_contribution_pixels"]
                != estimated_contribution
                or dark_budget_fields["selected_contribution_pixels"]
                != estimated_contribution
                or estimated_contribution
                > math.floor(
                    QUALITY_REPAIR_DARK_FOREGROUND_MAX_SINGLE_CONTRIBUTION_FACTOR
                    * dark_budget_fields["budget_pixels"]
                )
                or dark_budget_fields["cumulative_contribution_pixels"]
                > dark_budget_fields["budget_limit_pixels"]
            ):
                raise RuntimeError(
                    "Multiview dark-identity change exceeds its contribution budget"
                )
            validated_dark_changes[part_id] = {
                "canonical_group_id": canonical_group_id,
                "support": dark_residual_support,
                "diagnostic": dark_diagnostic_summary,
                "budget": dark_budget_fields,
            }
        elif lane == QUALITY_REPAIR_DARK_FOREGROUND_RESIDUAL_LANE:
            group_diagnostic = group_diagnostics_by_id.get(canonical_group_id)
            raw_dark_supports = (
                group_diagnostic.get("dark_residual_supporting_views")
                if isinstance(group_diagnostic, dict)
                else None
            )
            if (
                not isinstance(group_diagnostic, dict)
                or group_diagnostic.get("dark_residual_repairable") is not True
                or group_diagnostic.get("dark_residual_reason_codes") != []
                or not isinstance(raw_dark_supports, list)
                or len(raw_dark_supports) != 1
                or not isinstance(raw_dark_supports[0], dict)
            ):
                raise RuntimeError(
                    "Dark-foreground change lacks one repairable QA residual"
                )
            support_view_id = supports["supporting_view_ids"][0]
            spatial_reference = dark_references.get(support_view_id)
            if not isinstance(spatial_reference, dict):
                raise RuntimeError(
                    "Dark-foreground change lacks its trusted reference"
                )
            expected_dark_support = _quality_dark_support_from_report(
                quality_report=quality_report,
                view_id=support_view_id,
                canonical_group_id=canonical_group_id,
                spatial_reference=spatial_reference,
            )
            dark_residual_support = change.get("dark_residual_support")
            if (
                raw_dark_supports[0] != expected_dark_support
                or dark_residual_support != expected_dark_support
                or supports["supporting_content_cluster_ids"]
                != [expected_dark_support["content_cluster_id"]]
                or supports["supporting_pose_cluster_ids"]
                != [expected_dark_support["pose_cluster_id"]]
            ):
                raise RuntimeError(
                    "Dark-foreground residual support does not match QA evidence"
                )
            spatial_part = _quality_dark_spatial_part(spatial_report, part_id)
            observation = _quality_dark_spatial_observation(
                spatial_part, support_view_id
            )
            raw_diagnostic = observation.get("dark_foreground_diagnostic")
            if not isinstance(raw_diagnostic, dict):
                raise RuntimeError(
                    "Dark-foreground spatial diagnostic is missing"
                )
            dark_diagnostic_summary = (
                _validate_quality_dark_foreground_diagnostic(
                    diagnostic=raw_diagnostic,
                    observation=observation,
                    alignment=_quality_dark_alignment(
                        spatial_report, support_view_id
                    ),
                    references=dark_references,
                    target_view_id=support_view_id,
                    canonical_group_id=canonical_group_id,
                )
            )
            if _quality_dark_matched_semantic_conflict(
                part_id=part_id,
                canonical_group_id=canonical_group_id,
                spatial_part=spatial_part,
                spatial_gate_audit=spatial_gate_audit,
                mapping_consensus=mapping_consensus,
                minimum_semantic_confidence=minimum_semantic_confidence,
            ):
                raise RuntimeError(
                    "Dark-foreground repair conflicts with a matched semantic group"
                )
            estimated_contribution = int(
                math.ceil(
                    dark_diagnostic_summary["projected_part_pixels"]
                    / expected_dark_support["render_foreground_pixels"]
                    * expected_dark_support["normalized_reference_pixels"]
                    * dark_diagnostic_summary["dark_signal_share"]
                )
            )
            dark_diagnostic_summary["estimated_contribution_pixels"] = (
                estimated_contribution
            )
            if (
                change.get("dark_foreground_diagnostic")
                != dark_diagnostic_summary
            ):
                raise RuntimeError(
                    "Dark-foreground change diagnostic does not match spatial evidence"
                )
            dark_budget_fields = {
                field: _require_exact_int(
                    change.get(field),
                    f"Dark-foreground {field}",
                )
                for field in (
                    "budget_pixels",
                    "budget_limit_pixels",
                    "existing_contribution_pixels",
                    "estimated_contribution_pixels",
                    "selected_contribution_pixels",
                    "cumulative_contribution_pixels",
                )
            }
            if (
                dark_budget_fields["budget_pixels"]
                != expected_dark_support["budget_pixels"]
                or dark_budget_fields["budget_limit_pixels"]
                != expected_dark_support["budget_limit_pixels"]
                or dark_budget_fields["estimated_contribution_pixels"]
                != estimated_contribution
                or dark_budget_fields["selected_contribution_pixels"]
                != estimated_contribution
                or estimated_contribution < 1
                or estimated_contribution
                > math.floor(
                    QUALITY_REPAIR_DARK_FOREGROUND_MAX_SINGLE_CONTRIBUTION_FACTOR
                    * dark_budget_fields["budget_pixels"]
                )
                or dark_budget_fields["cumulative_contribution_pixels"]
                > dark_budget_fields["budget_limit_pixels"]
            ):
                raise RuntimeError(
                    "Dark-foreground change exceeds its contribution budget"
                )
            validated_dark_changes[part_id] = {
                "canonical_group_id": canonical_group_id,
                "support": dark_residual_support,
                "diagnostic": dark_diagnostic_summary,
                "budget": dark_budget_fields,
            }
        elif any(
            field in change
            for field in (
                "dark_residual_support",
                "dark_foreground_diagnostic",
                "budget_pixels",
                "budget_limit_pixels",
                "existing_contribution_pixels",
                "estimated_contribution_pixels",
                "selected_contribution_pixels",
                "cumulative_contribution_pixels",
            )
        ):
            raise RuntimeError(
                f"Quality-repair dark residual fields are unauthorized for {part_id}"
            )

        source_material_id = selection.get("material_id")
        expected_parameters: dict[str, Any] | None = None
        expected_mvinverse: dict[str, Any] | None = None
        if lane in {
            QUALITY_REPAIR_DARK_FOREGROUND_RESIDUAL_LANE,
            QUALITY_REPAIR_MULTIVIEW_DARK_IDENTITY_LANE,
            QUALITY_REPAIR_REPEATED_GEOMETRY_DARK_RESIDUAL_LANE,
        } and (
                parameterized
                or "mvinverse_parameterized" in change
                or "confirmed_source_material_id" in change
        ):
            raise RuntimeError(
                "Dark-foreground quality repair cannot use MVInverse "
                f"parameterization for part {part_id}"
            )
        if parameterized:
            if (
                confirmed_source_material_id != source_material_id
                or new_material_id != source_material_id
                or not isinstance(source_material_id, str)
                or source_material_id not in allowed_id_set
            ):
                raise RuntimeError(
                    "Quality-repair MVInverse material substitution is invalid "
                    f"for part {part_id}"
                )
            expected_parameters, expected_mvinverse = (
                _expected_repair_mvinverse_delta(
                    evidence=mvinverse_pbr_evidence,
                    group_id=canonical_group_id,
                    source_material_id=source_material_id,
                    allowed_material_ids=allowed_id_set,
                )
            )
        elif (
            source_material_id != new_material_id
            or "confirmed_source_material_id" in change
            or "mvinverse_parameterized" in change
        ):
            raise RuntimeError(
                "Quality-repair fixed material delta is invalid "
                f"for part {part_id}"
            )

        expected_change = {
            "part_id": part_id,
            "canonical_group_id": canonical_group_id,
            "material_id": new_material_id,
            **supports,
            "old_material_id": old_material_id,
            "new_material_id": new_material_id,
        }
        if high_confidence_provisional:
            expected_change["material_selection_basis"] = (
                QUALITY_REPAIR_PROVISIONAL_MATERIAL_BASIS
            )
        if (
            anchor_part_ids is not None
            and anchor_supporting_view_ids is not None
        ):
            expected_change.update(
                {
                    "anchor_part_ids": anchor_part_ids,
                    "anchor_supporting_view_ids": anchor_supporting_view_ids,
                }
            )
        if spatial_anchor_view_ids is not None:
            expected_change["spatial_anchor_view_ids"] = spatial_anchor_view_ids
        if parameterized:
            expected_change.update(
                {
                    "confirmed_source_material_id": source_material_id,
                    "mvinverse_parameterized": True,
                }
            )
        if (
            dark_residual_support is not None
            and dark_diagnostic_summary is not None
            and dark_budget_fields is not None
        ):
            expected_change.update(
                {
                    "dark_residual_support": dark_residual_support,
                    "dark_foreground_diagnostic": dark_diagnostic_summary,
                    **dark_budget_fields,
                }
            )
        repeated_dark_record = validated_repeated_dark_parts.get(part_id)
        if lane == QUALITY_REPAIR_REPEATED_GEOMETRY_DARK_RESIDUAL_LANE:
            if repeated_dark_record is None:
                raise RuntimeError(
                    f"Repeated-dark cohort evidence is missing for part {part_id}"
                )
            expected_change.update(
                {
                    "repeated_geometry_dark_cohort_id": repeated_dark_record[
                        "cohort_id"
                    ],
                    "cohort_part_ids": repeated_dark_record["cohort_part_ids"],
                }
            )
        elif (
            "repeated_geometry_dark_cohort_id" in change
            or "cohort_part_ids" in change
        ):
            raise RuntimeError(
                f"Repeated-dark cohort fields are unauthorized for part {part_id}"
            )
        if dominant_assembly_record is not None:
            expected_change.update(
                {
                    "dominant_assembly_cohort_id": dominant_assembly_record[
                        "cohort_id"
                    ],
                    "dominant_assembly_member_role": dominant_assembly_record[
                        "member_role"
                    ],
                }
            )
        elif (
            "dominant_assembly_cohort_id" in change
            or "dominant_assembly_member_role" in change
        ):
            raise RuntimeError(
                f"Dominant assembly cohort fields are unauthorized for part {part_id}"
            )

        semantic_override_view_ids = change.get(
            "semantic_conflict_override_view_ids"
        )
        semantic_anchor_view_ids = change.get("semantic_anchor_view_ids")
        validated_semantic_review_override: dict[str, list[str]] | None = None
        if lane == QUALITY_REPAIR_DOMINANT_RESIDUAL_SINGLE_VIEW_LANE and (
            semantic_override_view_ids is not None
            or semantic_anchor_view_ids is not None
        ):
            if (
                not isinstance(semantic_override_view_ids, list)
                or semantic_override_view_ids
                != sorted(set(semantic_override_view_ids))
                or not semantic_override_view_ids
                or not isinstance(semantic_anchor_view_ids, list)
                or semantic_anchor_view_ids
                != sorted(set(semantic_anchor_view_ids))
                or not semantic_anchor_view_ids
            ):
                raise RuntimeError(
                    "Dominant quality-repair semantic-review override is invalid "
                    f"for part {part_id}"
                )
            expected_override_view_ids, expected_anchor_view_ids = (
                _validate_quality_semantic_review_override(
                    spatial_report=spatial_report,
                    spatial_gate_audit=spatial_gate_audit,
                    mapping_consensus=mapping_consensus,
                    part_id=part_id,
                    canonical_group_id=canonical_group_id,
                    target_view_id=supports["supporting_view_ids"][0],
                )
            )
            if (
                semantic_override_view_ids != expected_override_view_ids
                or semantic_anchor_view_ids != expected_anchor_view_ids
            ):
                raise RuntimeError(
                    "Dominant quality-repair semantic-review evidence is "
                    f"inconsistent for part {part_id}"
                )
            expected_change["semantic_conflict_override_view_ids"] = (
                semantic_override_view_ids
            )
            expected_change["semantic_anchor_view_ids"] = semantic_anchor_view_ids
            validated_semantic_review_override = {
                "conflict_view_ids": semantic_override_view_ids,
                "anchor_view_ids": semantic_anchor_view_ids,
            }
        elif lane == QUALITY_REPAIR_MULTIVIEW_DARK_IDENTITY_LANE:
            if semantic_anchor_view_ids is not None:
                raise RuntimeError(
                    "Multiview dark-identity repair cannot use a semantic anchor"
                )
            expected_override_view_ids = (
                multiview_dark_semantic_override_view_ids or []
            )
            if expected_override_view_ids:
                if semantic_override_view_ids != expected_override_view_ids:
                    raise RuntimeError(
                        "Multiview dark-identity semantic conflict audit is "
                        f"inconsistent for {part_id}"
                    )
                expected_change["semantic_conflict_override_view_ids"] = (
                    expected_override_view_ids
                )
            elif semantic_override_view_ids is not None:
                raise RuntimeError(
                    "Multiview dark-identity repair contains an unauthorized "
                    f"semantic conflict override for {part_id}"
                )
        elif semantic_anchor_view_ids is not None:
            raise RuntimeError(
                f"Quality-repair semantic anchor is unauthorized for part {part_id}"
            )
        elif semantic_override_view_ids is not None:
            if (
                lane
                not in {
                    "bounded_spatial_multiview",
                    QUALITY_REPAIR_SEMANTIC_SINGLE_VIEW_LANE,
                    QUALITY_REPAIR_ANCHORED_SINGLE_VIEW_LANE,
                    QUALITY_REPAIR_SPATIAL_ANCHOR_SINGLE_VIEW_LANE,
                    QUALITY_REPAIR_DARK_FOREGROUND_RESIDUAL_LANE,
                    QUALITY_REPAIR_REPEATED_GEOMETRY_DARK_RESIDUAL_LANE,
                }
                or not isinstance(semantic_override_view_ids, list)
                or not semantic_override_view_ids
                or semantic_override_view_ids
                != sorted(set(semantic_override_view_ids))
                or any(
                    not isinstance(view_id, str) or not view_id
                    for view_id in semantic_override_view_ids
                )
                or not set(semantic_override_view_ids)
                <= (
                    set(supports["supporting_view_ids"])
                    | set(spatial_anchor_view_ids or [])
                )
            ):
                raise RuntimeError(
                    "Quality-repair semantic-conflict override is invalid "
                    f"for part {part_id}"
                )
            expected_change["semantic_conflict_override_view_ids"] = (
                semantic_override_view_ids
            )

        if change != expected_change:
            raise RuntimeError(
                f"Quality-repair audit contains an unauthorized delta for part {part_id}"
            )

        if lane == QUALITY_REPAIR_REPEATED_GEOMETRY_DARK_RESIDUAL_LANE:
            expected_reason_codes = (
                QUALITY_REPAIR_REPEATED_GEOMETRY_DARK_REASON_CODES
            )
        elif lane in {
            QUALITY_REPAIR_DARK_FOREGROUND_RESIDUAL_LANE,
            QUALITY_REPAIR_MULTIVIEW_DARK_IDENTITY_LANE,
        }:
            expected_reason_codes = QUALITY_REPAIR_DARK_FOREGROUND_REASON_CODES
        elif dominant_assembly_record is not None:
            expected_reason_codes = QUALITY_REPAIR_DOMINANT_ASSEMBLY_REASON_CODES
        elif lane == QUALITY_REPAIR_BOUNDED_SIGNATURE_SIBLING_COHORT_LANE:
            expected_reason_codes = (
                QUALITY_REPAIR_BOUNDED_SIGNATURE_SIBLING_REASON_CODES
            )
        elif high_confidence_provisional:
            expected_reason_codes = QUALITY_REPAIR_PROVISIONAL_REASON_CODES
        else:
            expected_reason_codes = QUALITY_REPAIR_REASON_CODES

        expected_assignment = dict(baseline_assignment)
        expected_assignment.pop("apply_action", None)
        expected_assignment.pop("source_visual_material_prim_path", None)
        expected_assignment.pop("source_visual_material_binding_sha256", None)
        expected_assignment.update(
            {
                "material_id": new_material_id,
                "semantic": str(
                    canonical_groups[canonical_group_id]["visual_description"]
                ).strip(),
                "confidence": 0.0,
                "evidence_views": [],
                "status": POLICY_FALLBACK_STATUS,
                "provenance": {
                    "tier": "qa_repair_candidate",
                    "reason_codes": list(expected_reason_codes),
                    "output_confidence_basis": (
                        POLICY_FALLBACK_CONFIDENCE_BASIS
                    ),
                    "sources": [],
                    "canonical_group_id": canonical_group_id,
                    "baseline_material_id": old_material_id,
                    "baseline_tier": baseline_item_provenance["tier"],
                    **supports,
                },
            }
        )
        if high_confidence_provisional:
            expected_assignment["provenance"]["material_selection_basis"] = (
                QUALITY_REPAIR_PROVISIONAL_MATERIAL_BASIS
            )
        if source_identity_provenance is not None:
            expected_assignment["provenance"]["source_identity_anchor"] = (
                source_identity_provenance
            )
        if (
            dark_residual_support is not None
            and dark_diagnostic_summary is not None
            and dark_budget_fields is not None
        ):
            expected_assignment["provenance"]["dark_foreground_residual"] = {
                "lane": lane,
                "support": dark_residual_support,
                "diagnostic_sha256": dark_diagnostic_summary[
                    "diagnostic_sha256"
                ],
                **dark_budget_fields,
            }
        if validated_semantic_review_override is not None:
            expected_assignment["provenance"]["semantic_review_override"] = (
                validated_semantic_review_override
            )
        if repeated_dark_record is not None:
            expected_assignment["provenance"][
                "repeated_geometry_dark_residual"
            ] = repeated_dark_record["provenance"]
        if expected_parameters is not None and expected_mvinverse is not None:
            expected_assignment["parameters"] = expected_parameters
            expected_assignment["provenance"]["mvinverse"] = expected_mvinverse
            parameterized_part_ids.append(part_id)
        if output_assignment != expected_assignment:
            raise RuntimeError(
                f"Quality-repair assignment delta is unsafe for part {part_id}"
            )

    expected_provisional_group_ids = sorted(
        {
            str(item["canonical_group_id"])
            for item in changes
            if item.get("material_selection_basis")
            == QUALITY_REPAIR_PROVISIONAL_MATERIAL_BASIS
        }
    )
    if (
        audit.get("provisional_material_candidate_group_ids", [])
        != expected_provisional_group_ids
    ):
        raise RuntimeError(
            "Quality-repair provisional material candidate audit is inconsistent"
        )

    if dark_residual_part_ids:
        changed_dark_group_ids = {
            record["canonical_group_id"]
            for record in validated_dark_changes.values()
        }
        if set(dark_budget_by_group) != changed_dark_group_ids:
            raise RuntimeError(
                "Dark-foreground repair budget coverage is not deterministic"
            )
        for group_id in sorted(changed_dark_group_ids):
            budget = dark_budget_by_group[group_id]
            expected_budget_keys = {
                "canonical_group_id",
                "reference_view_id",
                "dark_residual_support",
                "budget_pixels",
                "budget_limit_pixels",
                "per_part_limit_pixels",
                "existing_contribution_pixels",
                "existing_parts",
                "candidates",
                "selected_part_ids",
                "selected_contribution_pixels",
                "total_contribution_pixels",
            }
            if set(budget) != expected_budget_keys:
                raise RuntimeError("Dark-foreground repair budget schema is invalid")
            group_changes = {
                part_id: record
                for part_id, record in validated_dark_changes.items()
                if record["canonical_group_id"] == group_id
            }
            support = next(iter(group_changes.values()))["support"]
            view_id = support["reference_view_id"]
            budget_pixels = _require_exact_int(
                budget.get("budget_pixels"),
                "Dark-foreground budget pixels",
                minimum=1,
            )
            budget_limit = _require_exact_int(
                budget.get("budget_limit_pixels"),
                "Dark-foreground budget limit",
                minimum=1,
            )
            per_part_limit = _require_exact_int(
                budget.get("per_part_limit_pixels"),
                "Dark-foreground per-part limit",
                minimum=1,
            )
            if (
                budget.get("canonical_group_id") != group_id
                or budget.get("reference_view_id") != view_id
                or budget.get("dark_residual_support") != support
                or budget_pixels != support["budget_pixels"]
                or budget_limit != support["budget_limit_pixels"]
                or per_part_limit
                != math.floor(
                    QUALITY_REPAIR_DARK_FOREGROUND_MAX_SINGLE_CONTRIBUTION_FACTOR
                    * budget_pixels
                )
            ):
                raise RuntimeError("Dark-foreground repair budget is inconsistent")

            expected_existing_parts: list[dict[str, Any]] = []
            for existing_part_id, existing_change in sorted(changes_by_part.items()):
                if (
                    existing_change.get("canonical_group_id") != group_id
                    or localization_lanes_by_part[existing_part_id]["lane"]
                    in {
                        QUALITY_REPAIR_DARK_FOREGROUND_RESIDUAL_LANE,
                        QUALITY_REPAIR_MULTIVIEW_DARK_IDENTITY_LANE,
                    }
                ):
                    continue
                existing_observation = _quality_dark_spatial_observation(
                    _quality_dark_spatial_part(spatial_report, existing_part_id),
                    view_id,
                )
                projected = existing_observation.get("projected_part_pixels")
                if (
                    not isinstance(projected, int)
                    or isinstance(projected, bool)
                    or projected <= 0
                ):
                    continue
                expected_existing_parts.append(
                    {
                        "part_id": existing_part_id,
                        "projected_part_pixels": projected,
                        "estimated_contribution_pixels": int(
                            math.ceil(
                                projected
                                / support["render_foreground_pixels"]
                                * support["normalized_reference_pixels"]
                            )
                        ),
                    }
                )
            existing_contribution = sum(
                item["estimated_contribution_pixels"]
                for item in expected_existing_parts
            )
            if (
                budget.get("existing_parts") != expected_existing_parts
                or budget.get("existing_contribution_pixels")
                != existing_contribution
            ):
                raise RuntimeError(
                    "Dark-foreground existing contribution is inconsistent"
                )

            raw_candidates = budget.get("candidates")
            if not isinstance(raw_candidates, list) or not raw_candidates:
                raise RuntimeError(
                    "Dark-foreground repair budget has no deterministic candidates"
                )
            expected_candidate_core: list[dict[str, Any]] = []
            for raw_candidate in raw_candidates:
                if not isinstance(raw_candidate, dict) or set(raw_candidate) != {
                    "part_id",
                    "evidence_strength",
                    "diagnostic_sha256",
                    "estimated_contribution_pixels",
                    "selected",
                    "reason_code",
                    "cumulative_contribution_pixels",
                }:
                    raise RuntimeError(
                        "Dark-foreground repair candidate schema is invalid"
                    )
                candidate_part_id = raw_candidate.get("part_id")
                if not isinstance(candidate_part_id, str):
                    raise RuntimeError(
                        "Dark-foreground repair candidate part is invalid"
                    )
                candidate_spatial_part = _quality_dark_spatial_part(
                    spatial_report, candidate_part_id
                )
                candidate_summary: dict[str, Any] | None = None
                contribution: int | None = None
                raw_consensus = candidate_spatial_part.get(
                    "multiview_dark_consensus"
                )
                if (
                    isinstance(raw_consensus, dict)
                    and raw_consensus.get("status") == "resolved"
                    and raw_consensus.get("canonical_group_id") == group_id
                ):
                    (
                        multiview_candidate_summary,
                        _,
                        _,
                    ) = multiview_dark_record(candidate_part_id, group_id)
                    if (
                        multiview_candidate_summary.get("diagnostic_sha256")
                        == raw_candidate.get("diagnostic_sha256")
                    ):
                        candidate_summary = multiview_candidate_summary
                        contribution = _require_exact_int(
                            candidate_summary.get(
                                "estimated_contribution_pixels"
                            ),
                            "Multiview dark candidate contribution",
                            minimum=1,
                        )
                if candidate_summary is None:
                    candidate_observation = _quality_dark_spatial_observation(
                        candidate_spatial_part, view_id
                    )
                    candidate_diagnostic = candidate_observation.get(
                        "dark_foreground_diagnostic"
                    )
                    if not isinstance(candidate_diagnostic, dict):
                        raise RuntimeError(
                            "Dark-foreground repair candidate diagnostic is missing"
                        )
                    candidate_summary = (
                        _validate_quality_dark_foreground_diagnostic(
                            diagnostic=candidate_diagnostic,
                            observation=candidate_observation,
                            alignment=_quality_dark_alignment(
                                spatial_report, view_id
                            ),
                            references=dark_references,
                            target_view_id=view_id,
                            canonical_group_id=group_id,
                        )
                    )
                    if _quality_dark_matched_semantic_conflict(
                        part_id=candidate_part_id,
                        canonical_group_id=group_id,
                        spatial_part=candidate_spatial_part,
                        spatial_gate_audit=spatial_gate_audit,
                        mapping_consensus=mapping_consensus,
                        minimum_semantic_confidence=minimum_semantic_confidence,
                    ):
                        raise RuntimeError(
                            "Dark-foreground budget contains a semantic conflict"
                        )
                    contribution = int(
                        math.ceil(
                            candidate_summary["projected_part_pixels"]
                            / support["render_foreground_pixels"]
                            * support["normalized_reference_pixels"]
                            * candidate_summary["dark_signal_share"]
                        )
                    )
                assert contribution is not None
                expected_candidate_core.append(
                    {
                        "part_id": candidate_part_id,
                        "evidence_strength": candidate_summary[
                            "evidence_strength"
                        ],
                        "diagnostic_sha256": candidate_summary[
                            "diagnostic_sha256"
                        ],
                        "estimated_contribution_pixels": contribution,
                    }
                )
            if expected_candidate_core != sorted(
                expected_candidate_core,
                key=lambda item: (-item["evidence_strength"], item["part_id"]),
            ) or len({item["part_id"] for item in expected_candidate_core}) != len(
                expected_candidate_core
            ):
                raise RuntimeError(
                    "Dark-foreground candidate ordering is not deterministic"
                )

            cumulative = existing_contribution
            selected_part_ids: list[str] = []
            selected_contribution = 0
            expected_candidates: list[dict[str, Any]] = []
            for core in expected_candidate_core:
                contribution = core["estimated_contribution_pixels"]
                selected = True
                reason_code: str | None = None
                if contribution <= 0:
                    selected = False
                    reason_code = "DARK_RESIDUAL_CONTRIBUTION_NOT_POSITIVE"
                elif contribution > per_part_limit:
                    selected = False
                    reason_code = "DARK_RESIDUAL_SINGLE_PART_BUDGET_EXCEEDED"
                elif cumulative + contribution > budget_limit:
                    selected = False
                    reason_code = "DARK_RESIDUAL_TOTAL_BUDGET_EXCEEDED"
                if selected:
                    cumulative += contribution
                    selected_contribution += contribution
                    selected_part_ids.append(core["part_id"])
                expected_candidates.append(
                    {
                        **core,
                        "selected": selected,
                        "reason_code": reason_code,
                        "cumulative_contribution_pixels": cumulative,
                    }
                )
            if (
                raw_candidates != expected_candidates
                or budget.get("selected_part_ids") != sorted(selected_part_ids)
                or set(selected_part_ids) != set(group_changes)
                or budget.get("selected_contribution_pixels")
                != selected_contribution
                or budget.get("total_contribution_pixels") != cumulative
                or cumulative > budget_limit
            ):
                raise RuntimeError(
                    "Dark-foreground contribution budget is inconsistent"
                )
            for selected_part_id in selected_part_ids:
                record = group_changes[selected_part_id]
                candidate = next(
                    item
                    for item in expected_candidates
                    if item["part_id"] == selected_part_id
                )
                expected_part_budget = {
                    "budget_pixels": budget_pixels,
                    "budget_limit_pixels": budget_limit,
                    "existing_contribution_pixels": existing_contribution,
                    "estimated_contribution_pixels": candidate[
                        "estimated_contribution_pixels"
                    ],
                    "selected_contribution_pixels": candidate[
                        "estimated_contribution_pixels"
                    ],
                    "cumulative_contribution_pixels": candidate[
                        "cumulative_contribution_pixels"
                    ],
                }
                if record["budget"] != expected_part_budget:
                    raise RuntimeError(
                        "Dark-foreground per-part budget audit is inconsistent"
                    )

    mvinverse_audit = audit.get("mvinverse")
    if (
        not isinstance(mvinverse_audit, dict)
        or mvinverse_audit.get("enabled") is not True
        or mvinverse_audit.get("parameterized_part_ids")
        != sorted(parameterized_part_ids)
        or not isinstance(mvinverse_audit.get("skipped"), list)
    ):
        raise RuntimeError("Quality-repair MVInverse audit is inconsistent")
    return changed_count

__all__ = [
    "_expected_repair_mvinverse_delta",
    "_validate_quality_repair_bundle",
    "_validate_quality_repair_dominant_assembly_cohorts",
]
