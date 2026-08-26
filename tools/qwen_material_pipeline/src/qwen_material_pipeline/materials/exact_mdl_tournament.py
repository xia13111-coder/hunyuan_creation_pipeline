"""Select an immutable NVIDIA MDL candidate from bounded render QA.

The tournament changes only the MDL identity.  Candidate renders must be
hash-linked to their candidate plans.  An all-view PASS always outranks a
complete, non-failing multi-view REVIEW; incomplete evidence and every hard
FAIL remain ineligible.  The winning material delta can then be replayed on a
newer repair plan without discarding unrelated, evidence-backed part repairs.
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


SCHEMA_VERSION = "qwen-exact-mdl-tournament/v1"
SEMANTIC_CONTRACT_SCHEMA_VERSION = "qwen-material-family-contract/v1"
SELECTION_OBJECTIVE_SEMANTIC = "semantic_compatible_visual"
SELECTION_OBJECTIVE_VISUAL = "visual_similarity"
SELECTION_OBJECTIVES = frozenset(
    {SELECTION_OBJECTIVE_SEMANTIC, SELECTION_OBJECTIVE_VISUAL}
)
QUALITY_TIER_INELIGIBLE = "INELIGIBLE"
QUALITY_TIER_COMPLETE_NONFAIL_REVIEW = "COMPLETE_NONFAIL_REVIEW"
QUALITY_TIER_ALL_VIEW_PASS = "ALL_VIEW_PASS"
QUALITY_TIER_RANK = {
    QUALITY_TIER_INELIGIBLE: 0,
    QUALITY_TIER_COMPLETE_NONFAIL_REVIEW: 1,
    QUALITY_TIER_ALL_VIEW_PASS: 2,
}


class ExactMdlTournamentError(ValueError):
    def __init__(
        self,
        message: str,
        *,
        audit: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.audit = dict(audit) if audit is not None else None


def _validate_selection_objective(value: str) -> str:
    if value not in SELECTION_OBJECTIVES:
        raise ExactMdlTournamentError(
            "selection_objective must be one of "
            + ", ".join(sorted(SELECTION_OBJECTIVES))
        )
    return value


def build_part_family_contract(
    *,
    plan: Mapping[str, Any],
    material_choice_audit: Mapping[str, Any],
    palette_fusion: Mapping[str, Any] | None = None,
) -> dict[str, set[str]]:
    """Build reliable per-material-entity catalog-family constraints.

    Material appearance is not enough to establish physical identity.  The
    staged Qwen/MVInverse analysis records whether a canonical material family
    is reliable and whether a metal part has a visible coating.  Only those
    reliable decisions become hard tournament constraints.  Painted metal may
    use either a metal MDL with a coating model or a paint MDL; it must never
    silently become plastic merely because one plastic preset has a closer
    color.  Assignment keys remain their historical Part IDs.  Face-subset
    keys use :func:`material_entity_contract_key`, so a plastic subset never
    inherits its containing metal assignment's family.
    """

    reliable_fusion_families: dict[str, str] = {}
    if palette_fusion is not None:
        canonical_palette = palette_fusion.get("canonical_palette")
        groups = (
            canonical_palette.get("groups")
            if isinstance(canonical_palette, Mapping)
            else None
        )
        if not isinstance(groups, Sequence) or isinstance(groups, (str, bytes)):
            raise ExactMdlTournamentError(
                "palette fusion lacks canonical material groups"
            )
        for raw_group in groups:
            if not isinstance(raw_group, Mapping):
                raise ExactMdlTournamentError(
                    "palette fusion contains an invalid material group"
                )
            group_id = raw_group.get("group_id")
            canonical_family = raw_group.get("family_hint")
            sources = raw_group.get("sources")
            if (
                not isinstance(group_id, str)
                or not group_id
                or not isinstance(canonical_family, str)
                or not canonical_family
                or not isinstance(sources, Sequence)
                or isinstance(sources, (str, bytes))
            ):
                continue
            votes: dict[str, tuple[set[str], float, float]] = {}
            malformed = False
            for source in sources:
                if not isinstance(source, Mapping):
                    malformed = True
                    break
                view_id = source.get("view_id")
                family = source.get("family_hint")
                confidence = source.get("confidence")
                if (
                    not isinstance(view_id, str)
                    or not view_id
                    or not isinstance(family, str)
                    or not family
                    or isinstance(confidence, bool)
                    or not isinstance(confidence, (int, float))
                ):
                    malformed = True
                    break
                view_ids, confidence_sum, maximum = votes.get(family, (set(), 0.0, 0.0))
                if view_id in view_ids:
                    malformed = True
                    break
                view_ids.add(view_id)
                votes[family] = (
                    view_ids,
                    confidence_sum + float(confidence),
                    max(maximum, float(confidence)),
                )
            if malformed or canonical_family not in votes:
                continue
            canonical_views, canonical_sum, canonical_maximum = votes[canonical_family]
            alternatives = [
                (len(view_ids), confidence_sum)
                for family, (view_ids, confidence_sum, _maximum) in votes.items()
                if family != canonical_family
            ]
            reliable = (
                len(canonical_views) >= 2
                and (
                    not alternatives
                    or (
                        len(canonical_views)
                        >= max(item[0] for item in alternatives) + 2
                        and canonical_sum >= max(item[1] for item in alternatives) + 0.6
                    )
                )
            ) or (
                len(canonical_views) == 1
                and not alternatives
                and canonical_maximum >= 0.85
            )
            if reliable:
                reliable_fusion_families[group_id] = canonical_family

    def group_families(group_id: Any) -> set[str] | None:
        if not isinstance(group_id, str) or not group_id:
            return None
        raw_group = material_choice_audit.get(group_id)
        if not isinstance(raw_group, Mapping):
            return None
        selection_group = raw_group.get("selection_group")
        retrieval = raw_group.get("retrieval_audit")
        if not isinstance(selection_group, Mapping) or not isinstance(
            retrieval, Mapping
        ):
            return None
        policy = retrieval.get("surface_interpretation_policy")
        family_reliable = (
            isinstance(policy, Mapping) and policy.get("family_reliable") is True
        )
        family = selection_group.get("family_hint")
        if group_id in reliable_fusion_families:
            family = reliable_fusion_families[group_id]
            family_reliable = True
        if not family_reliable or not isinstance(family, str) or not family:
            return None
        allowed = {family}
        finish = selection_group.get("finish_hint")
        coating_supported = (
            retrieval.get("applied_coating_confirmed") is True
            or retrieval.get("applied_coating_plausible") is True
            or (
                isinstance(policy, Mapping)
                and policy.get("semantic_surface_class") == "coating"
            )
        )
        if family == "metal" and finish == "painted" and coating_supported:
            allowed.add("paint")
        return allowed

    assignments = _assignments(plan, "semantic contract plan")
    contract: dict[str, set[str]] = {}
    for part_id, assignment in assignments.items():
        provenance = assignment.get("provenance")
        if not isinstance(provenance, Mapping):
            continue
        allowed = group_families(provenance.get("canonical_group_id"))
        if allowed:
            contract[material_entity_contract_key(part_id)] = allowed
        raw_subset_groups = provenance.get("face_subset_canonical_group_ids")
        if raw_subset_groups is None:
            continue
        if not isinstance(raw_subset_groups, Mapping):
            raise ExactMdlTournamentError(
                f"semantic contract plan/{part_id} face-subset group map "
                "must be an object"
            )
        subset_names = {
            subset.get("subset_name")
            for subset in assignment.get("face_subsets", []) or []
            if isinstance(subset, Mapping)
            and isinstance(subset.get("subset_name"), str)
        }
        for subset_name, group_id in raw_subset_groups.items():
            if (
                not isinstance(subset_name, str)
                or not subset_name
                or subset_name not in subset_names
            ):
                raise ExactMdlTournamentError(
                    f"semantic contract plan/{part_id} has an invalid "
                    "face-subset group map"
                )
            allowed = group_families(group_id)
            if allowed:
                contract[material_entity_contract_key(part_id, subset_name)] = allowed
    return contract


def material_entity_contract_key(
    part_id: str,
    subset_name: str | None = None,
) -> str:
    """Return the stable family-contract key for one material binding."""

    if not isinstance(part_id, str) or not part_id:
        raise ExactMdlTournamentError("material entity part_id must be non-empty")
    if subset_name is None:
        return part_id
    if not isinstance(subset_name, str) or not subset_name:
        raise ExactMdlTournamentError("material entity subset_name must be non-empty")
    return f"{part_id}#face_subset:{subset_name}"


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assignments(plan: Mapping[str, Any], label: str) -> dict[str, dict[str, Any]]:
    if plan.get("schema_version") != "1.0":
        raise ExactMdlTournamentError(f"{label} has an invalid schema")
    raw = plan.get("assignments")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ExactMdlTournamentError(f"{label}.assignments must be an array")
    output: dict[str, dict[str, Any]] = {}
    for item in raw:
        if not isinstance(item, Mapping):
            raise ExactMdlTournamentError(f"{label} has a non-object assignment")
        part_id = item.get("part_id")
        material_id = item.get("material_id")
        if (
            not isinstance(part_id, str)
            or not part_id
            or part_id in output
            or not isinstance(material_id, str)
            or not material_id.startswith("mdl:")
        ):
            raise ExactMdlTournamentError(f"{label} has an invalid assignment")
        parameters = item.get("parameters")
        if parameters is not None and (
            not isinstance(parameters, Mapping) or bool(parameters)
        ):
            raise ExactMdlTournamentError(
                f"{label}/{part_id} modifies selected MDL parameters"
            )
        raw_subsets = item.get("face_subsets", [])
        if raw_subsets is None:
            raw_subsets = []
        if not isinstance(raw_subsets, Sequence) or isinstance(
            raw_subsets, (str, bytes)
        ):
            raise ExactMdlTournamentError(
                f"{label}/{part_id}.face_subsets must be an array"
            )
        for index, subset in enumerate(raw_subsets):
            if not isinstance(subset, Mapping):
                raise ExactMdlTournamentError(
                    f"{label}/{part_id}.face_subsets[{index}] must be an object"
                )
            subset_parameters = subset.get("parameters")
            if subset_parameters is not None and (
                not isinstance(subset_parameters, Mapping)
                or bool(subset_parameters)
            ):
                raise ExactMdlTournamentError(
                    f"{label}/{part_id} modifies face-subset MDL parameters"
                )
        output[part_id] = copy.deepcopy(dict(item))
    return output


def _material_only_delta(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    label: str,
    allow_no_change: bool = False,
) -> list[dict[str, str]]:
    """Return assignment and face-subset MDL substitutions only.

    Face subsets are authoring entities in the material plan, but their schema
    intentionally has no provenance field.  A subset substitution is therefore
    represented in the audit by ``part_id`` plus ``subset_name`` and replayed
    through the owning assignment.  Every other subset field (including face
    indices and semantic text) must remain byte-for-byte equivalent as
    canonical JSON.
    """

    baseline_by_part = _assignments(baseline, "tournament baseline plan")
    candidate_by_part = _assignments(candidate, label)
    if set(baseline_by_part) != set(candidate_by_part):
        raise ExactMdlTournamentError(f"{label} does not exactly cover the baseline")
    changes: list[dict[str, str]] = []
    for part_id in sorted(baseline_by_part):
        before = baseline_by_part[part_id]
        after = candidate_by_part[part_id]
        before_material = str(before.pop("material_id"))
        after_material = str(after.pop("material_id"))
        before_subsets = _face_subsets_by_name(
            before.pop("face_subsets", []),
            f"tournament baseline plan/{part_id}",
        )
        after_subsets = _face_subsets_by_name(
            after.pop("face_subsets", []),
            f"{label}/{part_id}",
        )
        if list(before_subsets) != list(after_subsets):
            raise ExactMdlTournamentError(
                f"{label}/{part_id} does not preserve face-subset order"
            )
        if before != after:
            raise ExactMdlTournamentError(
                f"{label}/{part_id} changes fields other than material bindings"
            )
        if before_material != after_material:
            changes.append(
                {
                    "part_id": part_id,
                    "old_material_id": before_material,
                    "new_material_id": after_material,
                }
            )
        for subset_name in sorted(before_subsets):
            before_subset = before_subsets[subset_name]
            after_subset = after_subsets[subset_name]
            before_subset_material = str(before_subset.pop("material_id"))
            after_subset_material = str(after_subset.pop("material_id"))
            if before_subset != after_subset:
                raise ExactMdlTournamentError(
                    f"{label}/{part_id}/{subset_name} changes fields other "
                    "than material_id"
                )
            if before_subset_material != after_subset_material:
                changes.append(
                    {
                        "part_id": part_id,
                        "subset_name": subset_name,
                        "old_material_id": before_subset_material,
                        "new_material_id": after_subset_material,
                    }
                )
    if not changes and not allow_no_change:
        raise ExactMdlTournamentError(f"{label} has no exact MDL substitution")
    return changes


def _face_subsets_by_name(
    value: Any,
    label: str,
) -> dict[str, dict[str, Any]]:
    if value is None:
        return {}
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ExactMdlTournamentError(f"{label}.face_subsets must be an array")
    output: dict[str, dict[str, Any]] = {}
    for index, raw_subset in enumerate(value):
        if not isinstance(raw_subset, Mapping):
            raise ExactMdlTournamentError(
                f"{label}.face_subsets[{index}] must be an object"
            )
        subset = copy.deepcopy(dict(raw_subset))
        subset_name = subset.get("subset_name")
        material_id = subset.get("material_id")
        if (
            not isinstance(subset_name, str)
            or not subset_name
            or subset_name in output
            or not isinstance(material_id, str)
            or not material_id.startswith("mdl:")
        ):
            raise ExactMdlTournamentError(
                f"{label}.face_subsets[{index}] has an invalid material binding"
            )
        parameters = subset.get("parameters")
        if parameters is not None and (
            not isinstance(parameters, Mapping) or bool(parameters)
        ):
            raise ExactMdlTournamentError(
                f"{label}.face_subsets[{index}] modifies selected MDL parameters"
            )
        output[subset_name] = subset
    return output


def _quality_score(
    report: Mapping[str, Any],
    *,
    label: str,
    selection_objective: str,
) -> dict[str, Any]:
    if report.get("schema_version") != "qwen-reference-render-comparison/v1":
        raise ExactMdlTournamentError(f"{label} quality report schema is invalid")
    aggregate = report.get("aggregate")
    views = report.get("views")
    if not isinstance(aggregate, Mapping) or not isinstance(views, Sequence):
        raise ExactMdlTournamentError(f"{label} quality report is incomplete")
    view_ids: list[str] = []
    view_statuses: list[str] = []
    per_view_scores_complete = True
    for view in views:
        if not isinstance(view, Mapping):
            raise ExactMdlTournamentError(f"{label} has an invalid quality view")
        view_id = view.get("reference_view_id")
        if not isinstance(view_id, str) or not view_id or view_id in view_ids:
            raise ExactMdlTournamentError(f"{label} has duplicate quality views")
        view_ids.append(view_id)
        status = view.get("status")
        if status not in {"PASS", "REVIEW", "FAIL", "UNSCORABLE"}:
            raise ExactMdlTournamentError(
                f"{label} has an unsupported quality view status"
            )
        view_statuses.append(str(status))
        material_color = view.get("material_color")
        color_value = (
            material_color.get("score")
            if isinstance(material_color, Mapping)
            else None
        )
        if (
            isinstance(color_value, bool)
            or not isinstance(color_value, (int, float))
            or not 0.0 <= float(color_value) <= 1.0
        ):
            per_view_scores_complete = False
        if selection_objective == SELECTION_OBJECTIVE_VISUAL:
            material_texture = view.get("material_texture")
            texture_value = (
                material_texture.get("score")
                if isinstance(material_texture, Mapping)
                else None
            )
            appearance_value = view.get("material_appearance_score")
            if (
                isinstance(texture_value, bool)
                or not isinstance(texture_value, (int, float))
                or not 0.0 <= float(texture_value) <= 1.0
                or isinstance(appearance_value, bool)
                or not isinstance(appearance_value, (int, float))
                or not 0.0 <= float(appearance_value) <= 1.0
            ):
                per_view_scores_complete = False
    comparable = aggregate.get("comparable_view_count")
    color_score = aggregate.get("material_color_score")
    valid_color_score = (
        isinstance(color_score, (int, float))
        and not isinstance(color_score, bool)
        and 0.0 <= float(color_score) <= 1.0
    )
    status_counts = {
        status: view_statuses.count(status)
        for status in ("PASS", "REVIEW", "FAIL", "UNSCORABLE")
    }
    coverage_contract_satisfied = (
        isinstance(comparable, int)
        and not isinstance(comparable, bool)
        and comparable == len(view_ids)
        and len(view_ids) >= 2
        and aggregate.get("reference_view_count") == len(view_ids)
        and aggregate.get("passed_view_count") == status_counts["PASS"]
        and aggregate.get("review_view_count") == status_counts["REVIEW"]
        and aggregate.get("failed_view_count") == status_counts["FAIL"]
        and aggregate.get("unscorable_view_count") == status_counts["UNSCORABLE"]
        and aggregate.get("reference_view_coverage_status") == "PASS"
        and status_counts["UNSCORABLE"] == 0
        and per_view_scores_complete
        and valid_color_score
    )
    all_views_pass = (
        coverage_contract_satisfied
        and status_counts["PASS"] == len(view_ids)
        and status_counts["REVIEW"] == 0
        and status_counts["FAIL"] == 0
    )
    color_contract_satisfied = (
        aggregate.get("status") == "PASS"
        and aggregate.get("material_match_conclusion") == "PASS"
        and coverage_contract_satisfied
        and all_views_pass
    )
    texture_score = aggregate.get("material_texture_score")
    appearance_score = aggregate.get("material_appearance_score")
    texture_contract_satisfied = (
        isinstance(texture_score, (int, float))
        and not isinstance(texture_score, bool)
        and 0.0 <= float(texture_score) <= 1.0
        and isinstance(appearance_score, (int, float))
        and not isinstance(appearance_score, bool)
        and 0.0 <= float(appearance_score) <= 1.0
        and aggregate.get("texture_comparable_view_count") == len(view_ids)
        and aggregate.get("texture_unscorable_view_count") == 0
    )
    complete_nonfail_review = (
        selection_objective == SELECTION_OBJECTIVE_VISUAL
        and coverage_contract_satisfied
        and texture_contract_satisfied
        and len(view_ids) >= 3
        and status_counts["PASS"] + status_counts["REVIEW"] == len(view_ids)
        and status_counts["REVIEW"] >= 1
        and status_counts["FAIL"] == 0
        and aggregate.get("status") == "REVIEW"
        and aggregate.get("material_match_conclusion") == "NOT_CONCLUSIVE"
    )
    quality_tier = (
        QUALITY_TIER_ALL_VIEW_PASS
        if color_contract_satisfied
        and (
            texture_contract_satisfied
            if selection_objective == SELECTION_OBJECTIVE_VISUAL
            else True
        )
        else QUALITY_TIER_COMPLETE_NONFAIL_REVIEW
        if complete_nonfail_review
        else QUALITY_TIER_INELIGIBLE
    )
    reason_codes: list[str] = []
    if not color_contract_satisfied:
        reason_codes.append("ALL_VIEW_COLOR_PASS_CONTRACT_NOT_SATISFIED")
    if selection_objective == SELECTION_OBJECTIVE_VISUAL and not (
        texture_contract_satisfied
    ):
        reason_codes.append("COMPLETE_TEXTURE_APPEARANCE_EVIDENCE_REQUIRED")
    if quality_tier == QUALITY_TIER_COMPLETE_NONFAIL_REVIEW:
        reason_codes.append("COMPLETE_NONFAIL_MULTIVIEW_REVIEW_TIER")
    return {
        "eligible": quality_tier != QUALITY_TIER_INELIGIBLE,
        "quality_tier": quality_tier,
        "all_view_pass": quality_tier == QUALITY_TIER_ALL_VIEW_PASS,
        "complete_nonfail_review": (
            quality_tier == QUALITY_TIER_COMPLETE_NONFAIL_REVIEW
        ),
        "color_score": float(color_score) if valid_color_score else 0.0,
        "texture_score": (
            float(texture_score) if texture_contract_satisfied else None
        ),
        "appearance_score": (
            float(appearance_score) if texture_contract_satisfied else None
        ),
        "view_ids": sorted(view_ids),
        "reason_codes": reason_codes,
    }


def build_exact_mdl_candidate_plan(
    *,
    source_plan: Mapping[str, Any],
    source_material_id: str,
    candidate_material_id: str,
    candidate_id: str,
    allowed_material_ids: set[str],
    target_part_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Replace one selected MDL identity without authoring any parameters."""

    if (
        not source_material_id.startswith("mdl:")
        or not candidate_material_id.startswith("mdl:")
        or source_material_id == candidate_material_id
        or candidate_material_id not in allowed_material_ids
        or not candidate_id
    ):
        raise ExactMdlTournamentError("invalid exact MDL candidate request")
    output = copy.deepcopy(dict(source_plan))
    output_by_part = _assignments(output, "candidate source plan")
    changed_part_ids: list[str] = []
    for part_id, assignment in output_by_part.items():
        if target_part_ids is not None and part_id not in target_part_ids:
            continue
        if assignment["material_id"] != source_material_id:
            continue
        assignment["material_id"] = candidate_material_id
        changed_part_ids.append(part_id)
    if not changed_part_ids:
        raise ExactMdlTournamentError(
            "source material does not occur in the candidate source plan"
        )
    if target_part_ids is not None and set(changed_part_ids) != target_part_ids:
        missing = sorted(target_part_ids - set(changed_part_ids))
        raise ExactMdlTournamentError(
            "candidate target parts do not all use the source material: "
            + ", ".join(missing)
        )
    output["assignments"] = [
        output_by_part[str(assignment["part_id"])]
        for assignment in output["assignments"]
    ]
    provenance = output.setdefault("provenance", {})
    if not isinstance(provenance, dict):
        raise ExactMdlTournamentError("candidate plan provenance is invalid")
    provenance["immutable_mdl_after_selection"] = True
    provenance["exact_mdl_candidate"] = {
        "candidate_id": candidate_id,
        "source_plan_sha256": _canonical_sha256(source_plan),
        "source_material_id": source_material_id,
        "candidate_material_id": candidate_material_id,
        "changed_part_ids": sorted(changed_part_ids),
        "parameters_locked_to_library_defaults": True,
    }
    _assignments(output, "exact MDL candidate plan")
    return output


def build_bounded_exact_mdl_candidate_plans(
    *,
    source_plan: Mapping[str, Any],
    material_candidates_by_group: Mapping[str, Mapping[str, Any]],
    material_choice_audit: Mapping[str, Any],
    palette_fusion: Mapping[str, Any],
    allowed_material_ids: set[str],
    maximum_candidates: int = 4,
    selection_objective: str = SELECTION_OBJECTIVE_SEMANTIC,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Choose one dominant evidence group and build a bounded exact-MDL round.

    The round is intentionally one-dimensional: every candidate changes the
    same canonical material group and leaves every other part untouched.  This
    keeps render attribution auditable and avoids an exponential cross-product
    of material combinations.
    """

    if (
        isinstance(maximum_candidates, bool)
        or not isinstance(maximum_candidates, int)
        or maximum_candidates < 2
    ):
        raise ExactMdlTournamentError("maximum_candidates must be at least two")
    selection_objective = _validate_selection_objective(selection_objective)
    visual_similarity_first = selection_objective == SELECTION_OBJECTIVE_VISUAL
    assignments = _assignments(source_plan, "candidate planning source plan")
    family_contract = build_part_family_contract(
        plan=source_plan,
        material_choice_audit=material_choice_audit,
        palette_fusion=palette_fusion,
    )
    canonical_palette = palette_fusion.get("canonical_palette")
    groups = (
        canonical_palette.get("groups")
        if isinstance(canonical_palette, Mapping)
        else None
    )
    if not isinstance(groups, Sequence) or isinstance(groups, (str, bytes)):
        raise ExactMdlTournamentError("palette fusion lacks canonical groups")
    fusion_groups = {
        str(group["group_id"]): group
        for group in groups
        if isinstance(group, Mapping) and isinstance(group.get("group_id"), str)
    }

    parts_by_group: dict[str, list[str]] = {}
    for part_id, assignment in assignments.items():
        provenance = assignment.get("provenance")
        group_id = (
            provenance.get("canonical_group_id")
            if isinstance(provenance, Mapping)
            else None
        )
        if isinstance(group_id, str) and (
            visual_similarity_first or part_id in family_contract
        ):
            parts_by_group.setdefault(group_id, []).append(part_id)

    ranked_groups: list[tuple[float, int, str]] = []
    for group_id, part_ids in parts_by_group.items():
        candidate_document = material_candidates_by_group.get(group_id)
        fusion_group = fusion_groups.get(group_id)
        if not isinstance(candidate_document, Mapping) or not isinstance(
            fusion_group, Mapping
        ):
            continue
        raw_candidates = candidate_document.get(
            "tournament_candidates",
            candidate_document.get("candidates"),
        )
        if not isinstance(raw_candidates, Sequence) or isinstance(
            raw_candidates, (str, bytes)
        ):
            continue
        allowed_families = set().union(
            *(family_contract.get(part_id, set()) for part_id in part_ids)
        )
        compatible_candidate_count = sum(
            isinstance(candidate, Mapping)
            and isinstance(candidate.get("material_id"), str)
            and candidate.get("material_id") in allowed_material_ids
            and (visual_similarity_first or candidate.get("family") in allowed_families)
            for candidate in raw_candidates
        )
        if compatible_candidate_count < 1:
            continue
        footprint = 0.0
        sources = fusion_group.get("sources")
        if isinstance(sources, Sequence) and not isinstance(sources, (str, bytes)):
            for source in sources:
                if not isinstance(source, Mapping):
                    continue
                confidence = source.get("confidence")
                boxes = source.get("boxes")
                if (
                    isinstance(confidence, bool)
                    or not isinstance(confidence, (int, float))
                    or not isinstance(boxes, Sequence)
                    or isinstance(boxes, (str, bytes))
                ):
                    continue
                view_area = 0.0
                for box in boxes:
                    if (
                        isinstance(box, Sequence)
                        and not isinstance(box, (str, bytes))
                        and len(box) == 4
                        and all(
                            isinstance(value, (int, float))
                            and not isinstance(value, bool)
                            for value in box
                        )
                    ):
                        left, top, right, bottom = map(float, box)
                        view_area += (
                            max(0.0, right - left)
                            * max(0.0, bottom - top)
                            / 1_000_000.0
                        )
                footprint += min(1.0, view_area) * float(confidence)
        ranked_groups.append((footprint, len(part_ids), group_id))
    if not ranked_groups:
        raise ExactMdlTournamentError(
            "no reliable canonical group has bounded compatible MDL candidates"
        )
    _footprint, _part_count, selected_group_id = max(ranked_groups)
    selected_part_ids = sorted(parts_by_group[selected_group_id])
    source_material_ids = {
        str(assignments[part_id]["material_id"]) for part_id in selected_part_ids
    }
    if len(source_material_ids) != 1:
        raise ExactMdlTournamentError(
            "dominant canonical group does not have one current MDL identity"
        )
    source_material_id = next(iter(source_material_ids))
    allowed_families = set().union(
        *(family_contract.get(part_id, set()) for part_id in selected_part_ids)
    )
    candidate_document = material_candidates_by_group[selected_group_id]
    raw_candidates = candidate_document.get(
        "tournament_candidates",
        candidate_document["candidates"],
    )
    material_ids = [source_material_id]
    for candidate in raw_candidates:
        if not isinstance(candidate, Mapping):
            continue
        material_id = candidate.get("material_id")
        if (
            isinstance(material_id, str)
            and material_id in allowed_material_ids
            and (visual_similarity_first or candidate.get("family") in allowed_families)
            and material_id not in material_ids
        ):
            material_ids.append(material_id)
        if len(material_ids) >= maximum_candidates:
            break
    if len(material_ids) < 2:
        raise ExactMdlTournamentError(
            "dominant canonical group has fewer than two exact MDL identities"
        )

    records: list[dict[str, Any]] = []
    for index, material_id in enumerate(material_ids, start=1):
        candidate_id = (
            f"{selected_group_id.casefold()}_{index:02d}_"
            f"{hashlib.sha256(material_id.encode('utf-8')).hexdigest()[:10]}"
        )
        if material_id == source_material_id:
            plan = copy.deepcopy(dict(source_plan))
            provenance = plan.setdefault("provenance", {})
            if not isinstance(provenance, dict):
                raise ExactMdlTournamentError(
                    "candidate source plan provenance is invalid"
                )
            provenance["immutable_mdl_after_selection"] = True
            provenance["exact_mdl_candidate"] = {
                "candidate_id": candidate_id,
                "source_plan_sha256": _canonical_sha256(source_plan),
                "source_material_id": source_material_id,
                "candidate_material_id": material_id,
                "changed_part_ids": [],
                "parameters_locked_to_library_defaults": True,
            }
        else:
            plan = build_exact_mdl_candidate_plan(
                source_plan=source_plan,
                source_material_id=source_material_id,
                candidate_material_id=material_id,
                candidate_id=candidate_id,
                allowed_material_ids=allowed_material_ids,
                target_part_ids=set(selected_part_ids),
            )
        records.append(
            {
                "candidate_id": candidate_id,
                "group_id": selected_group_id,
                "material_id": material_id,
                "is_baseline": material_id == source_material_id,
                "plan": plan,
            }
        )
    audit = {
        "schema_version": "qwen-exact-mdl-candidate-planning/v1",
        "status": "PLANNED",
        "selected_group_id": selected_group_id,
        "selected_group_reference_footprint_score": _footprint,
        "selected_group_part_count": len(selected_part_ids),
        "selected_part_ids": selected_part_ids,
        "source_material_id": source_material_id,
        "allowed_families": sorted(allowed_families),
        "selection_objective": selection_objective,
        "semantic_family_gate_applied": not visual_similarity_first,
        "candidate_count": len(records),
        "candidate_material_ids": material_ids,
        "maximum_candidates": maximum_candidates,
        "one_group_at_a_time": True,
        "parameters_locked_to_library_defaults": True,
    }
    return records, audit


def select_and_replay_exact_mdl_candidate(
    *,
    baseline_plan: Mapping[str, Any],
    target_plan: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    allowed_material_ids: set[str],
    material_families_by_id: Mapping[str, str] | None = None,
    allowed_families_by_part: Mapping[str, set[str]] | None = None,
    selection_objective: str = SELECTION_OBJECTIVE_SEMANTIC,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Select the best all-view PASS and replay only its MDL substitutions."""

    if len(candidates) < 2:
        raise ExactMdlTournamentError("the exact MDL tournament needs >=2 candidates")
    selection_objective = _validate_selection_objective(selection_objective)
    semantic_family_gate_applied = selection_objective == SELECTION_OBJECTIVE_SEMANTIC
    family_contract_keys = (
        list(allowed_families_by_part) if allowed_families_by_part is not None else []
    )
    family_contract_subset_count = sum(
        "#face_subset:" in key for key in family_contract_keys
    )
    family_contract_part_count = (
        len(family_contract_keys) - family_contract_subset_count
    )
    baseline_by_part = _assignments(baseline_plan, "tournament baseline plan")
    target_by_part = _assignments(target_plan, "tournament target plan")
    if set(baseline_by_part) != set(target_by_part):
        raise ExactMdlTournamentError("target plan does not cover baseline parts")

    records: list[dict[str, Any]] = []
    expected_view_ids: list[str] | None = None
    for raw in candidates:
        if not isinstance(raw, Mapping):
            raise ExactMdlTournamentError("candidate must be an object")
        candidate_id = raw.get("candidate_id")
        plan = raw.get("plan")
        quality_report = raw.get("quality_report")
        apply_report = raw.get("apply_report")
        rendered_registry = raw.get("rendered_registry")
        rendered_registry_file_sha256 = raw.get("rendered_registry_file_sha256")
        if (
            not isinstance(candidate_id, str)
            or not candidate_id
            or not isinstance(plan, Mapping)
            or not isinstance(quality_report, Mapping)
            or not isinstance(apply_report, Mapping)
            or not isinstance(rendered_registry, Mapping)
            or not isinstance(rendered_registry_file_sha256, str)
        ):
            raise ExactMdlTournamentError("candidate bundle is incomplete")
        delta = _material_only_delta(
            baseline_plan,
            plan,
            label=f"candidate {candidate_id}",
            allow_no_change=True,
        )
        if any(
            change["new_material_id"] not in allowed_material_ids for change in delta
        ):
            raise ExactMdlTournamentError(
                f"candidate {candidate_id} contains a material outside the catalog"
            )
        semantic_eligible = True
        semantic_reason_codes: list[str] = []
        semantic_checks: list[dict[str, Any]] = []
        for change in delta:
            part_id = change["part_id"]
            subset_name = change.get("subset_name")
            contract_key = material_entity_contract_key(
                part_id,
                subset_name if isinstance(subset_name, str) else None,
            )
            allowed_families = (
                allowed_families_by_part.get(contract_key)
                if allowed_families_by_part is not None
                else None
            )
            if not allowed_families:
                continue
            material_family = (
                material_families_by_id.get(change["new_material_id"])
                if material_families_by_id is not None
                else None
            )
            check = {
                "part_id": part_id,
                "subset_name": subset_name,
                "material_entity_contract_key": contract_key,
                "material_id": change["new_material_id"],
                "material_family": material_family,
                "allowed_families": sorted(allowed_families),
                "compatible": material_family in allowed_families,
            }
            semantic_checks.append(check)
            if material_family is None and semantic_family_gate_applied:
                raise ExactMdlTournamentError(
                    f"candidate {candidate_id} material family is unavailable: "
                    f"{change['new_material_id']}"
                )
            if material_family not in allowed_families:
                semantic_eligible = False
                semantic_reason_codes.append(
                    "MATERIAL_FAMILY_CONFLICTS_WITH_MULTIVIEW_CONSENSUS"
                )
        plan_sha256 = _canonical_sha256(plan)
        if apply_report.get("plan_sha256") != plan_sha256:
            raise ExactMdlTournamentError(
                f"candidate {candidate_id} apply/plan hash mismatch"
            )
        output_sha256 = apply_report.get("output_sha256")
        if (
            not isinstance(output_sha256, str)
            or rendered_registry.get("asset_sha256") != output_sha256
        ):
            raise ExactMdlTournamentError(
                f"candidate {candidate_id} render/apply hash mismatch"
            )
        report_inputs = quality_report.get("inputs")
        if not isinstance(report_inputs, Mapping) or (
            report_inputs.get("rendered_registry_sha256")
            != rendered_registry_file_sha256
        ):
            raise ExactMdlTournamentError(
                f"candidate {candidate_id} QA/render registry hash mismatch"
            )
        quality = _quality_score(
            quality_report,
            label=f"candidate {candidate_id}",
            selection_objective=selection_objective,
        )
        visual_eligible = bool(quality["eligible"])
        color_score = float(quality["color_score"])
        texture_score = quality["texture_score"]
        appearance_score = quality["appearance_score"]
        view_ids = list(quality["view_ids"])
        visual_reason_codes = list(quality["reason_codes"])
        eligible = visual_eligible and (
            semantic_eligible or not semantic_family_gate_applied
        )
        if expected_view_ids is None:
            expected_view_ids = view_ids
        elif expected_view_ids != view_ids:
            raise ExactMdlTournamentError(
                "candidate QA reports do not cover the same reference views"
            )
        records.append(
            {
                "candidate_id": candidate_id,
                "eligible": eligible,
                "visual_eligible": visual_eligible,
                "quality_tier": quality["quality_tier"],
                "all_view_pass": quality["all_view_pass"],
                "complete_nonfail_review": quality["complete_nonfail_review"],
                "semantic_eligible": semantic_eligible,
                "semantic_reason_codes": sorted(set(semantic_reason_codes)),
                "visual_eligibility_reason_codes": visual_reason_codes,
                "semantic_checks": semantic_checks,
                "material_color_score": color_score,
                "material_texture_score": texture_score,
                "material_appearance_score": appearance_score,
                "selection_score": (
                    appearance_score
                    if selection_objective == SELECTION_OBJECTIVE_VISUAL
                    else color_score
                ),
                "selection_score_kind": (
                    "material_appearance_score"
                    if selection_objective == SELECTION_OBJECTIVE_VISUAL
                    else "material_color_score"
                ),
                "reference_view_ids": view_ids,
                "plan_sha256": plan_sha256,
                "quality_report_sha256": _canonical_sha256(quality_report),
                "rendered_registry_sha256": _canonical_sha256(rendered_registry),
                "rendered_registry_file_sha256": (rendered_registry_file_sha256),
                "material_changes": delta,
            }
        )
    eligible_records = [record for record in records if record["eligible"]]
    if not eligible_records:
        reason_code = (
            "NO_SEMANTICALLY_COMPATIBLE_ALL_VIEW_PASS_CANDIDATE"
            if semantic_family_gate_applied
            else "NO_COMPLETE_NONFAILING_VISUAL_CANDIDATE"
        )
        message = (
            "no semantically compatible exact MDL candidate passed every view"
            if semantic_family_gate_applied
            else "no exact MDL candidate has complete non-failing visual evidence"
        )
        raise ExactMdlTournamentError(
            message,
            audit={
                "schema_version": SCHEMA_VERSION,
                "status": "NO_ELIGIBLE_CANDIDATE",
                "selection_objective": selection_objective,
                "semantic_family_gate_applied": semantic_family_gate_applied,
                "baseline_plan_sha256": _canonical_sha256(baseline_plan),
                "target_plan_sha256": _canonical_sha256(target_plan),
                "candidate_count": len(records),
                "eligible_candidate_count": 0,
                "all_view_pass_candidate_count": 0,
                "complete_nonfail_review_candidate_count": 0,
                "quality_tier_order": [
                    QUALITY_TIER_ALL_VIEW_PASS,
                    QUALITY_TIER_COMPLETE_NONFAIL_REVIEW,
                    QUALITY_TIER_INELIGIBLE,
                ],
                "reference_view_ids": expected_view_ids,
                "candidates": sorted(
                    records, key=lambda record: record["candidate_id"]
                ),
                "selected_candidate_id": None,
                "selected_material_color_score": None,
                "selected_material_texture_score": None,
                "selected_material_appearance_score": None,
                "selected_selection_score": None,
                "selected_quality_tier": None,
                "selected_material_changes": [],
                "parameters_locked_to_library_defaults": True,
                "semantic_family_contract_part_count": (family_contract_part_count),
                "semantic_family_contract_entity_count": len(family_contract_keys),
                "semantic_family_contract_face_subset_entity_count": (
                    family_contract_subset_count
                ),
                "reason_codes": [reason_code],
            },
        )
    winner = sorted(
        eligible_records,
        key=lambda record: (
            -QUALITY_TIER_RANK[str(record["quality_tier"])],
            -float(record["selection_score"]),
            -float(record["material_texture_score"] or 0.0),
            -float(record["material_color_score"]),
            str(record["candidate_id"]),
        ),
    )[0]

    output = copy.deepcopy(dict(target_plan))
    output_by_part = {
        str(assignment["part_id"]): assignment for assignment in output["assignments"]
    }
    for change in winner["material_changes"]:
        part_id = change["part_id"]
        target_assignment = output_by_part[part_id]
        provenance = target_assignment.setdefault("provenance", {})
        if not isinstance(provenance, dict):
            raise ExactMdlTournamentError(
                f"target assignment {part_id} provenance is invalid"
            )
        subset_name = change.get("subset_name")
        if subset_name is None:
            if target_assignment["material_id"] != change["old_material_id"]:
                raise ExactMdlTournamentError(
                    f"target plan independently changed tournament part {part_id}"
                )
            target_assignment["material_id"] = change["new_material_id"]
            provenance["exact_mdl_tournament"] = {
                "candidate_id": winner["candidate_id"],
                "quality_report_sha256": winner["quality_report_sha256"],
                "old_material_id": change["old_material_id"],
                "new_material_id": change["new_material_id"],
                "parameters_locked_to_library_defaults": True,
            }
            continue
        if not isinstance(subset_name, str) or not subset_name:
            raise ExactMdlTournamentError(
                f"candidate {winner['candidate_id']} has an invalid subset change"
            )
        target_subsets = _face_subsets_by_name(
            target_assignment.get("face_subsets", []),
            f"tournament target plan/{part_id}",
        )
        target_subset = target_subsets.get(subset_name)
        if target_subset is None:
            raise ExactMdlTournamentError(
                f"target plan lost tournament face subset {part_id}:{subset_name}"
            )
        if target_subset["material_id"] != change["old_material_id"]:
            raise ExactMdlTournamentError(
                "target plan independently changed tournament face subset "
                f"{part_id}:{subset_name}"
            )
        raw_subsets = target_assignment.get("face_subsets")
        assert isinstance(raw_subsets, list)
        mutable_subset = next(
            subset
            for subset in raw_subsets
            if isinstance(subset, dict) and subset.get("subset_name") == subset_name
        )
        mutable_subset["material_id"] = change["new_material_id"]
        subset_audit = provenance.setdefault(
            "exact_mdl_face_subset_tournament",
            {},
        )
        if not isinstance(subset_audit, dict):
            raise ExactMdlTournamentError(
                f"target assignment {part_id} subset tournament audit is invalid"
            )
        subset_audit[subset_name] = {
            "candidate_id": winner["candidate_id"],
            "quality_report_sha256": winner["quality_report_sha256"],
            "old_material_id": change["old_material_id"],
            "new_material_id": change["new_material_id"],
            "parameters_locked_to_library_defaults": True,
        }
    output_provenance = output.setdefault("provenance", {})
    if not isinstance(output_provenance, dict):
        raise ExactMdlTournamentError("target plan provenance is invalid")
    baseline_provenance = baseline_plan.get("provenance")
    if isinstance(baseline_provenance, Mapping):
        source_asset_sha = baseline_provenance.get("asset_sha256")
        if isinstance(source_asset_sha, str):
            output_provenance["asset_sha256"] = source_asset_sha
    output_provenance["immutable_mdl_after_selection"] = True
    output_provenance["exact_mdl_tournament"] = {
        "schema_version": SCHEMA_VERSION,
        "selection_objective": selection_objective,
        "semantic_family_gate_applied": semantic_family_gate_applied,
        "selected_candidate_id": winner["candidate_id"],
        "quality_report_sha256": winner["quality_report_sha256"],
        "selection_score": winner["selection_score"],
        "selection_score_kind": winner["selection_score_kind"],
        "selected_quality_tier": winner["quality_tier"],
        "candidate_count": len(records),
        "eligible_candidate_count": len(eligible_records),
        "all_view_pass_candidate_count": sum(
            record["eligible"] and record["all_view_pass"] for record in records
        ),
        "complete_nonfail_review_candidate_count": sum(
            record["eligible"] and record["complete_nonfail_review"]
            for record in records
        ),
        "parameters_locked_to_library_defaults": True,
    }
    _assignments(output, "tournament output plan")
    audit = {
        "schema_version": SCHEMA_VERSION,
        "status": "SELECTED",
        "selection_objective": selection_objective,
        "semantic_family_gate_applied": semantic_family_gate_applied,
        "baseline_plan_sha256": _canonical_sha256(baseline_plan),
        "target_plan_sha256": _canonical_sha256(target_plan),
        "candidate_count": len(records),
        "eligible_candidate_count": len(eligible_records),
        "all_view_pass_candidate_count": sum(
            record["eligible"] and record["all_view_pass"] for record in records
        ),
        "complete_nonfail_review_candidate_count": sum(
            record["eligible"] and record["complete_nonfail_review"]
            for record in records
        ),
        "quality_tier_order": [
            QUALITY_TIER_ALL_VIEW_PASS,
            QUALITY_TIER_COMPLETE_NONFAIL_REVIEW,
            QUALITY_TIER_INELIGIBLE,
        ],
        "reference_view_ids": expected_view_ids,
        "candidates": sorted(records, key=lambda record: record["candidate_id"]),
        "selected_candidate_id": winner["candidate_id"],
        "selected_material_color_score": winner["material_color_score"],
        "selected_material_texture_score": winner["material_texture_score"],
        "selected_material_appearance_score": winner["material_appearance_score"],
        "selected_selection_score": winner["selection_score"],
        "selected_selection_score_kind": winner["selection_score_kind"],
        "selected_quality_tier": winner["quality_tier"],
        "selected_material_changes": winner["material_changes"],
        "parameters_locked_to_library_defaults": True,
        "semantic_family_contract_part_count": (family_contract_part_count),
        "semantic_family_contract_entity_count": len(family_contract_keys),
        "semantic_family_contract_face_subset_entity_count": (
            family_contract_subset_count
        ),
        "output_plan_sha256": _canonical_sha256(output),
    }
    return output, audit


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.expanduser().resolve(strict=True).read_text("utf-8"))
    if not isinstance(value, dict):
        raise ExactMdlTournamentError(f"{path} must contain a JSON object")
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-plan", type=Path, required=True)
    parser.add_argument("--target-plan", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--material-choice-audit", type=Path, required=True)
    parser.add_argument("--palette-fusion", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, action="append", required=True)
    parser.add_argument(
        "--quality-report-name",
        default="reference_render_comparison.json",
        help="Candidate-local quality report filename.",
    )
    parser.add_argument(
        "--selection-objective",
        choices=sorted(SELECTION_OBJECTIVES),
        default=SELECTION_OBJECTIVE_SEMANTIC,
    )
    parser.add_argument("--output-plan", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args(argv)

    candidates: list[dict[str, Any]] = []
    for raw_directory in args.candidate_dir:
        directory = raw_directory.expanduser().resolve(strict=True)
        rendered_registry_path = directory / "renders" / "part_registry.rendered.json"
        candidates.append(
            {
                "candidate_id": directory.name,
                "plan": _read_json(directory / "plan.json"),
                "apply_report": _read_json(directory / "apply_report.json"),
                "quality_report": _read_json(directory / args.quality_report_name),
                "rendered_registry": _read_json(rendered_registry_path),
                "rendered_registry_file_sha256": _file_sha256(rendered_registry_path),
            }
        )
    catalog = _read_json(args.catalog)
    catalog_materials = {
        str(item["material_id"]): str(item["family"])
        for item in catalog.get("materials", [])
        if (
            isinstance(item, Mapping)
            and isinstance(item.get("material_id"), str)
            and isinstance(item.get("family"), str)
        )
    }
    target_plan = _read_json(args.target_plan)
    family_contract = build_part_family_contract(
        plan=target_plan,
        material_choice_audit=_read_json(args.material_choice_audit),
        palette_fusion=_read_json(args.palette_fusion),
    )
    try:
        output, audit = select_and_replay_exact_mdl_candidate(
            baseline_plan=_read_json(args.baseline_plan),
            target_plan=target_plan,
            candidates=candidates,
            allowed_material_ids=set(catalog_materials),
            material_families_by_id=catalog_materials,
            allowed_families_by_part=family_contract,
            selection_objective=args.selection_objective,
        )
    except ExactMdlTournamentError as exc:
        if exc.audit is None:
            raise
        _write_json(args.audit, exc.audit)
        print(
            json.dumps(
                {
                    "status": exc.audit["status"],
                    "candidate_count": exc.audit["candidate_count"],
                    "eligible_candidate_count": 0,
                    "audit": str(args.audit.expanduser().resolve()),
                    "error": str(exc),
                },
                ensure_ascii=False,
            )
        )
        return 2
    _write_json(args.output_plan, output)
    _write_json(args.audit, audit)
    print(
        json.dumps(
            {
                "status": audit["status"],
                "selected_candidate_id": audit["selected_candidate_id"],
                "selected_material_color_score": audit["selected_material_color_score"],
                "selected_material_texture_score": audit[
                    "selected_material_texture_score"
                ],
                "selected_material_appearance_score": audit[
                    "selected_material_appearance_score"
                ],
                "selected_selection_score": audit["selected_selection_score"],
                "candidate_count": audit["candidate_count"],
                "eligible_candidate_count": audit["eligible_candidate_count"],
                "output_plan": str(args.output_plan.expanduser().resolve()),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ExactMdlTournamentError",
    "SCHEMA_VERSION",
    "SEMANTIC_CONTRACT_SCHEMA_VERSION",
    "SELECTION_OBJECTIVES",
    "SELECTION_OBJECTIVE_SEMANTIC",
    "SELECTION_OBJECTIVE_VISUAL",
    "build_exact_mdl_candidate_plan",
    "build_bounded_exact_mdl_candidate_plans",
    "build_part_family_contract",
    "material_entity_contract_key",
    "select_and_replay_exact_mdl_candidate",
]
