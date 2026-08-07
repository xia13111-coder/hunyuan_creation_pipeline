"""Bounded exact-MDL and dominant-membership tournament orchestration."""

from __future__ import annotations

import copy
import json
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from ..command import LogCallback, log_message
from ..progress import emit_progress
from .config import canonical_sha256, read_object, write_object
from .quality import (
    validated_exact_mdl_tournament_mapping as _validated_exact_mdl_tournament_mapping,
)
from .references import sha256_file
from .stages.common import require_file as _require_file
from .stages.runner import _run_stage
from qwen_material_pipeline.materials.membership_tournament import (
    MembershipTournamentError,
    build_membership_candidate_plans,
    discover_dominant_assembly_cohorts,
    select_membership_candidate,
)
from qwen_material_pipeline.materials.multigroup_exact_mdl_tournament import (
    BASELINE_GROUP_PRESENT_WITHOUT_LOCALIZATION,
    _baseline_group_presence_evidence,
)


CommandRunner = Callable[..., None]


def _log_exact_mdl_candidate_progress(
    log_cb: LogCallback,
    *,
    state: str,
    group_index: int,
    group_total: int,
    candidate_index: int,
    candidate_total: int,
    candidate_id: str,
    global_current: int,
    global_total: int,
    cache_status: str | None = None,
) -> None:
    """Report local identity while the progress bar spans the whole tournament."""

    cache_detail = f" cache={cache_status}" if cache_status else ""
    emit_progress(
        log_cb,
        scope="visual_materials.exact_mdl_tournament",
        stage="candidate",
        state=state,
        current=global_current,
        total=global_total,
        unit="candidate",
        detail=(
            f"group {group_index}/{group_total} "
            f"candidate {candidate_index}/{candidate_total} "
            f"id={candidate_id}{cache_detail}"
        ),
    )


def _log_exact_mdl_group_progress(
    log_cb: LogCallback,
    *,
    state: str,
    current: int,
    total: int,
    group_id: str,
) -> None:
    emit_progress(
        log_cb,
        scope="visual_materials.exact_mdl_tournament",
        stage="group",
        state=state,
        current=current,
        total=total,
        unit="group",
        detail=f"group_id={group_id}",
    )

def _multigroup_local_compare_command(
    *,
    qwen_python: Path,
    reference_manifest: Path,
    rendered_registry: Path,
    view_map: Path,
    palette_fusion: Path,
    group_id: str,
    target_part_ids: Sequence[str],
    target_entities: Sequence[Mapping[str, Any]] | None = None,
    reference_view_ids: Sequence[str],
    output: Path,
) -> list[str]:
    """Build the strict group-local comparison command for one candidate."""

    if not group_id:
        raise ValueError("group_id must be non-empty")
    part_ids = list(target_part_ids)
    if (
        not part_ids
        or part_ids != sorted(set(part_ids))
        or any(not isinstance(part_id, str) or not part_id for part_id in part_ids)
    ):
        raise ValueError("target_part_ids must be sorted, unique, and non-empty")
    view_ids = list(reference_view_ids)
    if (
        len(view_ids) < 2
        or view_ids != sorted(set(view_ids))
        or any(not isinstance(view_id, str) or not view_id for view_id in view_ids)
    ):
        raise ValueError(
            "reference_view_ids must be sorted, unique, and contain at least two IDs"
        )
    command = [
        str(qwen_python),
        "-m",
        "qwen_material_pipeline",
        "compare",
        "--reference-manifest",
        str(reference_manifest),
        "--rendered-registry",
        str(rendered_registry),
        "--view-map",
        str(view_map),
        "--minimum-comparable-views",
        str(len(view_ids)),
        "--target-group-id",
        group_id,
        "--palette-fusion",
        str(palette_fusion),
        "--output",
        str(output),
    ]
    for view_id in view_ids:
        command.extend(["--target-reference-view-id", view_id])
    for part_id in part_ids:
        command.extend(["--target-part-id", part_id])
    entities = (
        [
            {
                "entity_kind": "assignment",
                "part_id": part_id,
            }
            for part_id in part_ids
        ]
        if target_entities is None
        else list(target_entities)
    )
    if not entities or any(not isinstance(entity, Mapping) for entity in entities):
        raise ValueError("target_entities must contain material entity objects")
    for entity in entities:
        command.extend(
            [
                "--target-entity-json",
                json.dumps(
                    dict(entity),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            ]
        )
    return command


def _baseline_preserved_disagreement_exemptions(
    *,
    queue_audit: Mapping[str, Any],
    material_choice_audit: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Audit disagreements that are unused because baseline already delivers.

    This is intentionally narrower than a generic queue exclusion.  A group is
    exempt from render-confirming a forward/reverse *selection* only when the
    queue proved that no material entity was authored for it and every trusted
    source view reports exact baseline presence.  Groups with a real target,
    incomplete evidence, or any coverage blocker remain required.
    """

    raw_exclusions = queue_audit.get("excluded_groups")
    raw_groups = queue_audit.get("groups")
    raw_blockers = queue_audit.get("coverage_blockers")
    if (
        not isinstance(raw_exclusions, list)
        or not isinstance(raw_groups, list)
        or not isinstance(raw_blockers, list)
    ):
        return []
    queued_group_ids = {
        str(group["group_id"])
        for group in raw_groups
        if isinstance(group, Mapping)
        and isinstance(group.get("group_id"), str)
    }
    blocker_group_ids = {
        str(blocker["group_id"])
        for blocker in raw_blockers
        if isinstance(blocker, Mapping)
        and isinstance(blocker.get("group_id"), str)
    }
    exemptions: list[dict[str, Any]] = []
    seen_group_ids: set[str] = set()
    for exclusion in raw_exclusions:
        if not isinstance(exclusion, Mapping):
            continue
        group_id = exclusion.get("group_id")
        choice = (
            material_choice_audit.get(group_id)
            if isinstance(group_id, str)
            else None
        )
        exclusion_reason = exclusion.get("reason")
        unused_single_view = (
            exclusion_reason == "INSUFFICIENT_INDEPENDENT_REFERENCE_VIEWS"
            and exclusion.get("reference_view_count") == 1
        )
        if (
            not isinstance(group_id, str)
            or not group_id
            or group_id in seen_group_ids
            or (
                exclusion_reason != BASELINE_GROUP_PRESENT_WITHOUT_LOCALIZATION
                and not unused_single_view
            )
            or exclusion.get("baseline_preserved") is not True
            or exclusion.get("authored_target_entity_count") != 0
            or group_id in queued_group_ids
            or group_id in blocker_group_ids
            or not isinstance(choice, Mapping)
            or choice.get("confirmation_basis")
            != "forward_reverse_disagreement"
        ):
            continue
        disagreement_contract = choice.get("disagreement_tournament")
        presence = exclusion.get("baseline_presence_evidence")
        if (
            not isinstance(disagreement_contract, Mapping)
            or disagreement_contract.get("required") is not True
            or not isinstance(presence, Mapping)
            or presence.get("canonical_group_id") != group_id
            or presence.get("all_source_views_present") is not True
        ):
            continue
        reference_view_ids = presence.get("reference_view_ids")
        views = presence.get("views")
        if (
            not isinstance(reference_view_ids, list)
            or reference_view_ids != sorted(set(reference_view_ids))
            or len(reference_view_ids) < (1 if unused_single_view else 2)
            or not isinstance(views, list)
            or len(views) != len(reference_view_ids)
            or (
                unused_single_view
                and (
                    reference_view_ids != exclusion.get("reference_view_ids")
                    or presence.get("minimum_source_view_count") != 1
                )
            )
        ):
            continue
        evidence_by_view: dict[str, Mapping[str, Any]] = {}
        evidence_complete = True
        for view in views:
            if not isinstance(view, Mapping):
                evidence_complete = False
                break
            view_id = view.get("reference_view_id")
            recall = view.get("recall")
            if (
                not isinstance(view_id, str)
                or view_id in evidence_by_view
                or view_id not in reference_view_ids
                or view.get("trusted_reference_evidence") is not True
                or view.get("delivery_presence_status") != "PRESENT"
                or isinstance(recall, bool)
                or not isinstance(recall, (int, float))
                or float(recall) != 1.0
            ):
                evidence_complete = False
                break
            evidence_by_view[view_id] = view
        if not evidence_complete or sorted(evidence_by_view) != reference_view_ids:
            continue
        seen_group_ids.add(group_id)
        exemptions.append(
            {
                "group_id": group_id,
                "reason_code": (
                    "FORWARD_REVERSE_DISAGREEMENT_UNUSED_BASELINE_"
                    "PRESENT_WITHOUT_AUTHORED_TARGET"
                ),
                "queue_exclusion_reason": exclusion_reason,
                "baseline_preserved": True,
                "authored_target_entity_count": 0,
                "reference_view_ids": list(reference_view_ids),
                "baseline_presence_evidence_sha256": canonical_sha256(
                    dict(presence)
                ),
                "disagreement_tournament_contract_sha256": canonical_sha256(
                    dict(disagreement_contract)
                ),
                "render_confirmation_required": False,
                "material_selection_was_authored": False,
                "material_choice_resolved": False,
                "presence_only_not_material_identity_confirmation": True,
            }
        )
    return sorted(exemptions, key=lambda record: str(record["group_id"]))


def _final_baseline_preserved_disagreement_exemptions(
    *,
    queue_audit: Mapping[str, Any],
    material_choice_audit: Mapping[str, Any],
    final_plan: Mapping[str, Any],
    palette_fusion: Mapping[str, Any],
    final_quality_report_path: Path,
    final_rendered_registry_path: Path,
) -> list[dict[str, Any]]:
    """Revalidate unused-choice exemptions against the final rendered state.

    A single-view presence observation is never material-identity
    confirmation.  It may only prove that an unresolved choice is unused:
    the sealed plan must author no assignment or face-subset entity for the
    group, and the final whole-asset comparison must independently reproduce
    exact trusted presence from the registry it hash-identifies.  Any malformed
    plan, stale report, registry mismatch, or incomplete presence evidence
    returns no exemptions so the immutable lock gate remains fail-closed.
    """

    if (
        not final_quality_report_path.is_file()
        or not final_rendered_registry_path.is_file()
    ):
        return []
    try:
        final_quality_report = read_object(
            final_quality_report_path,
            "final disagreement-exemption visual quality report",
        )
        final_rendered_registry = read_object(
            final_rendered_registry_path,
            "final disagreement-exemption rendered registry",
        )
        final_quality_report_file_sha256 = sha256_file(final_quality_report_path)
        final_rendered_registry_file_sha256 = sha256_file(
            final_rendered_registry_path
        )
    except (OSError, RuntimeError, ValueError):
        return []
    if final_rendered_registry.get("schema_version") != "qwen-material-parts/v1":
        return []
    quality_inputs = final_quality_report.get("inputs")
    if not isinstance(quality_inputs, Mapping):
        return []
    reported_registry_path = quality_inputs.get("rendered_registry")
    reported_registry_sha256 = quality_inputs.get("rendered_registry_sha256")
    if (
        not isinstance(reported_registry_path, str)
        or not reported_registry_path
        or not isinstance(reported_registry_sha256, str)
        or reported_registry_sha256 != final_rendered_registry_file_sha256
    ):
        return []
    reported_registry = Path(reported_registry_path)
    if not reported_registry.is_absolute():
        reported_registry = final_quality_report_path.parent / reported_registry
    try:
        if reported_registry.resolve() != final_rendered_registry_path.resolve():
            return []
    except OSError:
        return []

    raw_assignments = final_plan.get("assignments")
    if not isinstance(raw_assignments, list):
        return []
    authored_entity_count_by_group: Counter[str] = Counter()
    for assignment in raw_assignments:
        if not isinstance(assignment, Mapping):
            return []
        provenance = assignment.get("provenance")
        if provenance is None:
            continue
        if not isinstance(provenance, Mapping):
            return []
        group_id = provenance.get("canonical_group_id")
        if isinstance(group_id, str) and group_id:
            authored_entity_count_by_group[group_id] += 1
        raw_subset_groups = provenance.get("face_subset_canonical_group_ids")
        if raw_subset_groups is None:
            continue
        if not isinstance(raw_subset_groups, Mapping):
            return []
        for subset_group_id in raw_subset_groups.values():
            if not isinstance(subset_group_id, str) or not subset_group_id:
                return []
            authored_entity_count_by_group[subset_group_id] += 1

    refreshed_queue_audit = copy.deepcopy(dict(queue_audit))
    raw_exclusions = refreshed_queue_audit.get("excluded_groups")
    if not isinstance(raw_exclusions, list):
        return []
    for exclusion in raw_exclusions:
        if not isinstance(exclusion, dict):
            return []
        reason = exclusion.get("reason")
        reference_view_ids = exclusion.get("reference_view_ids")
        reference_view_count = exclusion.get("reference_view_count")
        if reason == "INSUFFICIENT_INDEPENDENT_REFERENCE_VIEWS":
            minimum_source_view_count = 1
            if (
                not isinstance(reference_view_ids, list)
                or reference_view_ids != sorted(set(reference_view_ids))
                or reference_view_count != len(reference_view_ids)
                or len(reference_view_ids) != 1
            ):
                exclusion["baseline_presence_evidence"] = None
                continue
        elif reason == BASELINE_GROUP_PRESENT_WITHOUT_LOCALIZATION:
            minimum_source_view_count = 2
            if (
                not isinstance(reference_view_ids, list)
                or reference_view_ids != sorted(set(reference_view_ids))
                or reference_view_count != len(reference_view_ids)
                or len(reference_view_ids) < 2
            ):
                exclusion["baseline_presence_evidence"] = None
                continue
        else:
            continue
        group_id = exclusion.get("group_id")
        if not isinstance(group_id, str) or not group_id:
            exclusion["baseline_presence_evidence"] = None
            continue
        exclusion["baseline_presence_evidence"] = (
            _baseline_group_presence_evidence(
                group_id=group_id,
                source_view_ids=reference_view_ids,
                palette_fusion=palette_fusion,
                quality_report=final_quality_report,
                minimum_source_view_count=minimum_source_view_count,
            )
        )

    candidates = _baseline_preserved_disagreement_exemptions(
        queue_audit=refreshed_queue_audit,
        material_choice_audit=material_choice_audit,
    )
    # A below-significance group can legitimately have a staged Qwen
    # forward/reverse choice without that choice ever becoming part of the
    # authored Look.  Requiring an exact-MDL render tournament for such an
    # unused choice is both misleading and unbounded: the group has already
    # been excluded by the same reference-footprint policy that defines the
    # tournament queue, and there may be no material entity on which to apply
    # either candidate.  Audit this case only after the final plan proves that
    # the canonical group owns zero assignments/face subsets.  This is not
    # material-identity confirmation and never exempts queued groups, coverage
    # blockers, or any group that reappears in the final authored state.
    raw_groups = refreshed_queue_audit.get("groups")
    raw_blockers = refreshed_queue_audit.get("coverage_blockers")
    minimum_reference_footprint_score = refreshed_queue_audit.get(
        "minimum_reference_footprint_score"
    )
    if (
        isinstance(raw_groups, list)
        and isinstance(raw_blockers, list)
        and not isinstance(minimum_reference_footprint_score, bool)
        and isinstance(minimum_reference_footprint_score, (int, float))
        and float(minimum_reference_footprint_score) > 0.0
    ):
        queued_group_ids = {
            str(group["group_id"])
            for group in raw_groups
            if (
                isinstance(group, Mapping)
                and isinstance(group.get("group_id"), str)
            )
        }
        blocker_group_ids = {
            str(blocker["group_id"])
            for blocker in raw_blockers
            if (
                isinstance(blocker, Mapping)
                and isinstance(blocker.get("group_id"), str)
            )
        }
        candidate_group_ids = {
            str(candidate["group_id"])
            for candidate in candidates
            if (
                isinstance(candidate, Mapping)
                and isinstance(candidate.get("group_id"), str)
            )
        }
        for exclusion in raw_exclusions:
            if not isinstance(exclusion, Mapping):
                continue
            group_id = exclusion.get("group_id")
            footprint = exclusion.get("reference_footprint_score")
            choice = (
                material_choice_audit.get(group_id)
                if isinstance(group_id, str)
                else None
            )
            disagreement_contract = (
                choice.get("disagreement_tournament")
                if isinstance(choice, Mapping)
                else None
            )
            if (
                not isinstance(group_id, str)
                or not group_id
                or group_id in candidate_group_ids
                or group_id in queued_group_ids
                or group_id in blocker_group_ids
                or exclusion.get("reason")
                != "BELOW_REFERENCE_FOOTPRINT_THRESHOLD"
                or isinstance(footprint, bool)
                or not isinstance(footprint, (int, float))
                or float(footprint) < 0.0
                or float(footprint)
                >= float(minimum_reference_footprint_score)
                or authored_entity_count_by_group.get(group_id, 0) != 0
                or not isinstance(choice, Mapping)
                or choice.get("confirmation_basis")
                != "forward_reverse_disagreement"
                or not isinstance(disagreement_contract, Mapping)
                or disagreement_contract.get("required") is not True
            ):
                continue
            candidates.append(
                {
                    "group_id": group_id,
                    "reason_code": (
                        "FORWARD_REVERSE_DISAGREEMENT_UNUSED_BELOW_"
                        "SIGNIFICANCE_WITHOUT_AUTHORED_TARGET"
                    ),
                    "queue_exclusion_reason": exclusion.get("reason"),
                    "baseline_preserved": True,
                    "authored_target_entity_count": 0,
                    "reference_footprint_score": float(footprint),
                    "minimum_reference_footprint_score": float(
                        minimum_reference_footprint_score
                    ),
                    "queue_exclusion_sha256": canonical_sha256(dict(exclusion)),
                    "disagreement_tournament_contract_sha256": canonical_sha256(
                        dict(disagreement_contract)
                    ),
                    "render_confirmation_required": False,
                    "material_selection_was_authored": False,
                    "material_choice_resolved": False,
                    "presence_only_not_material_identity_confirmation": True,
                }
            )
            candidate_group_ids.add(group_id)
    final_plan_sha256 = canonical_sha256(dict(final_plan))
    final_assignment_state_sha256 = canonical_sha256(raw_assignments)
    exemptions: list[dict[str, Any]] = []
    for candidate in candidates:
        group_id = candidate.get("group_id")
        if (
            not isinstance(group_id, str)
            or not group_id
            or authored_entity_count_by_group.get(group_id, 0) != 0
            or candidate.get("material_choice_resolved") is not False
            or candidate.get("presence_only_not_material_identity_confirmation")
            is not True
        ):
            continue
        record = copy.deepcopy(candidate)
        record.update(
            {
                "final_state_revalidated": True,
                "final_authored_target_entity_count": 0,
                "final_plan_sha256": final_plan_sha256,
                "final_assignment_state_sha256": final_assignment_state_sha256,
                "final_quality_report_file_sha256": (
                    final_quality_report_file_sha256
                ),
                "final_rendered_registry_file_sha256": (
                    final_rendered_registry_file_sha256
                ),
                "evidence_phase": "final_whole_asset_post_tournament",
            }
        )
        exemptions.append(record)
    return sorted(exemptions, key=lambda record: str(record["group_id"]))


def _run_dominant_assembly_membership_tournaments(
    *,
    source_plan_path: Path,
    source: Path,
    apply_asset: Path,
    apply_subcommand: str,
    apply_asset_flag: str,
    effective_catalog: Path,
    material_root: Path,
    rendered_registry: Path,
    current_look_usd: Path,
    current_apply_report: Path,
    current_quality_report: Path,
    current_quality_rendered_registry: Path,
    reference_manifest: Path,
    palette_fusion_path: Path,
    tournament_dir: Path,
    tournament_view_map: Path,
    output_plan: Path,
    output_audit: Path,
    trusted_mapping: Mapping[str, str],
    mapped_render_resolution: int,
    render_rt_subframes: int,
    analysis_up_axis: str,
    analysis_front_axis: str,
    qwen_python: Path,
    isaac: Path,
    instance_root_count: int,
    applied_count: int,
    include_policy_fallback: bool,
    log_cb: LogCallback,
    command_runner: CommandRunner,
) -> dict[str, Any]:
    """Render M0/M1 and freeze cohort membership before exact-MDL rounds."""

    initial_plan = read_object(
        source_plan_path,
        "pre-membership material plan",
    )
    # Legacy/no-op plans need no palette or exact-MDL validation here.  Once a
    # cohort marker is present, the strict module below validates the complete
    # plan and fails closed on every malformed field.
    raw_assignments = initial_plan.get("assignments")
    has_membership_contract = bool(
        isinstance(raw_assignments, list)
        and any(
            isinstance(assignment, Mapping)
            and isinstance(assignment.get("provenance"), Mapping)
            and assignment["provenance"].get("dominant_assembly_cohort")
            is not None
            for assignment in raw_assignments
        )
    )
    if not has_membership_contract:
        return {
            "status": "NOT_REQUIRED",
            "cohort_count": 0,
            "selected_expanded_cohort_count": 0,
            "restored_m0_count": 0,
            "plan": source_plan_path,
            "look": current_look_usd,
            "apply": current_apply_report,
            "quality": current_quality_report,
            "rendered_registry": current_quality_rendered_registry,
            "applied_count": applied_count,
            "audit": None,
        }
    palette_fusion = read_object(palette_fusion_path, "membership palette fusion")
    try:
        cohorts = discover_dominant_assembly_cohorts(
            plan=initial_plan,
            palette_fusion=palette_fusion,
        )
    except MembershipTournamentError as exc:
        raise RuntimeError(
            "Dominant assembly cohort contract is invalid; physics was not "
            f"started: {exc}"
        ) from exc
    if not cohorts:
        raise RuntimeError(
            "Dominant assembly membership marker did not produce a complete "
            "cohort contract; physics was not started"
        )

    try:
        tournament_mapping = _validated_exact_mdl_tournament_mapping(
            quality_report=read_object(
                current_quality_report,
                "pre-membership quality report",
            ),
            reference_manifest=read_object(
                reference_manifest,
                "pre-membership reference manifest",
            ),
            trusted_mapping=dict(trusted_mapping),
            rendered_registry=read_object(
                current_quality_rendered_registry,
                "pre-membership rendered registry",
            ),
        )
    except (RuntimeError, ValueError) as exc:
        raise RuntimeError(
            "Dominant assembly membership tournament lacks a trusted render "
            f"mapping; physics was not started: {exc}"
        ) from exc
    mapped_render_views = list(tournament_mapping.values())
    if len(mapped_render_views) < 2:
        raise RuntimeError(
            "Dominant assembly membership tournament requires two trusted "
            "registered views; physics was not started"
        )
    write_object(
        tournament_view_map,
        {
            "schema_version": "qwen-reference-view-map/v1",
            "mapping": dict(sorted(tournament_mapping.items())),
            "source": "validated_pre_membership_quality_mapping",
        },
    )

    baseline_apply = read_object(
        current_apply_report,
        "pre-membership apply report",
    )
    expected_face_subset_count = baseline_apply.get("face_subset_count")
    if (
        isinstance(expected_face_subset_count, bool)
        or not isinstance(expected_face_subset_count, int)
        or expected_face_subset_count < 0
    ):
        raise RuntimeError(
            "Pre-membership apply report lacks a valid face_subset_count; "
            "physics was not started"
        )

    current_plan = initial_plan
    current_artifacts: dict[str, Any] = {
        "look": current_look_usd,
        "apply": current_apply_report,
        "quality": current_quality_report,
        "rendered_registry": current_quality_rendered_registry,
        "applied_count": applied_count,
    }
    round_audits: list[dict[str, Any]] = []
    selected_expanded_count = 0
    restored_m0_count = 0

    for cohort_index, discovered_cohort in enumerate(cohorts, start=1):
        cohort_id = str(discovered_cohort["cohort_id"])
        try:
            planned_candidates, round_contract = (
                build_membership_candidate_plans(
                    source_plan=current_plan,
                    palette_fusion=palette_fusion,
                    cohort_id=cohort_id,
                )
            )
        except MembershipTournamentError as exc:
            raise RuntimeError(
                f"Unable to plan dominant assembly membership cohort "
                f"{cohort_id}; physics was not started: {exc}"
            ) from exc
        group_id = str(round_contract["canonical_group_id"])
        target_part_ids = [
            str(part_id) for part_id in round_contract["cohort_part_ids"]
        ]
        target_entities = [
            {
                "entity_kind": "assignment",
                "part_id": part_id,
            }
            for part_id in target_part_ids
        ]
        reference_view_ids = [
            str(view_id) for view_id in round_contract["reference_view_ids"]
        ]
        missing_mapped_views = sorted(
            set(reference_view_ids) - set(tournament_mapping)
        )
        if missing_mapped_views:
            raise RuntimeError(
                f"Dominant assembly cohort {cohort_id} references unmapped "
                f"views {missing_mapped_views}; physics was not started"
            )
        log_message(
            log_cb,
            "Dominant assembly membership tournament "
            f"[{cohort_index}/{len(cohorts)}] cohort={cohort_id[:12]} "
            f"group={group_id} candidates=2",
        )

        rendered_bundles: list[dict[str, Any]] = []
        candidate_artifacts: dict[str, dict[str, Any]] = {}
        for candidate_index, planned_candidate in enumerate(
            planned_candidates,
            start=1,
        ):
            candidate_id = str(planned_candidate["candidate_id"])
            candidate_dir = tournament_dir / candidate_id
            candidate_plan_path = candidate_dir / "plan.json"
            candidate_apply_plan_path = candidate_plan_path
            candidate_look_usd = candidate_dir / "look.usda"
            candidate_apply_report = candidate_dir / "apply_report.json"
            candidate_registry = candidate_dir / "part_registry.json"
            candidate_render_dir = candidate_dir / "renders"
            candidate_rendered_registry = (
                candidate_render_dir / "part_registry.rendered.json"
            )
            candidate_quality_report = (
                candidate_dir / "reference_render_comparison.json"
            )
            candidate_plan = copy.deepcopy(dict(planned_candidate["plan"]))
            write_object(candidate_plan_path, candidate_plan)
            if instance_root_count:
                candidate_apply_plan_path = candidate_dir / "plan.apply.json"
                apply_plan = copy.deepcopy(candidate_plan)
                raw_provenance = apply_plan.get("provenance")
                sealed_provenance = (
                    dict(raw_provenance)
                    if isinstance(raw_provenance, Mapping)
                    else {}
                )
                sealed_provenance.update(
                    {
                        "asset_sha256": sha256_file(source),
                        "registry_sha256": canonical_sha256(
                            read_object(
                                rendered_registry,
                                "membership occurrence registry",
                            )
                        ),
                    }
                )
                apply_plan["provenance"] = sealed_provenance
                write_object(candidate_apply_plan_path, apply_plan)

            stage_prefix = (
                f"membership_{cohort_index:02d}_{candidate_index:02d}"
            )
            apply_command = [
                str(isaac),
                "-m",
                "qwen_material_pipeline",
                "usd",
                apply_subcommand,
                apply_asset_flag,
                str(apply_asset),
                "--catalog",
                str(effective_catalog),
                "--registry",
                str(rendered_registry),
                "--plan",
                str(candidate_apply_plan_path),
                "--output",
                str(candidate_look_usd),
                "--material-root",
                str(material_root),
                "--report",
                str(candidate_apply_report),
                # The manual-CAD workflow is unattended.  ``review`` records
                # are confidence annotations, not a request to leave the
                # source material on the Part-ID.
                "--include-review",
            ]
            if include_policy_fallback:
                apply_command.append("--include-policy-fallback")
            _run_stage(
                f"{stage_prefix}_apply",
                apply_command,
                log_cb,
                command_runner=command_runner,
                retry_native_crash=True,
            )
            _require_file(candidate_look_usd, f"{stage_prefix}_apply")
            _require_file(candidate_apply_report, f"{stage_prefix}_apply")
            candidate_apply = read_object(
                candidate_apply_report,
                f"membership candidate apply {candidate_id}",
            )
            candidate_applied_count = candidate_apply.get("applied_count")
            if candidate_applied_count != applied_count:
                raise RuntimeError(
                    "Dominant assembly membership candidate changed exact "
                    f"coverage: expected={applied_count}, "
                    f"actual={candidate_applied_count!r}"
                )
            if candidate_apply.get("face_subset_count") != (
                expected_face_subset_count
            ):
                raise RuntimeError(
                    "Dominant assembly membership candidate changed face-subset "
                    "coverage; physics was not started"
                )

            _run_stage(
                f"{stage_prefix}_registry",
                [
                    str(isaac),
                    "-m",
                    "qwen_material_pipeline",
                    "usd",
                    "registry",
                    "--usd",
                    str(candidate_look_usd),
                    "--output",
                    str(candidate_registry),
                ],
                log_cb,
                command_runner=command_runner,
                retry_native_crash=True,
            )
            _require_file(candidate_registry, f"{stage_prefix}_registry")
            _run_stage(
                f"{stage_prefix}_render",
                [
                    str(isaac),
                    "-m",
                    "qwen_material_pipeline",
                    "usd",
                    "render",
                    "--registry",
                    str(candidate_registry),
                    "--output-dir",
                    str(candidate_render_dir),
                    "--resolution",
                    str(mapped_render_resolution),
                    "--views",
                    ",".join(mapped_render_views),
                    "--rt-subframes",
                    str(render_rt_subframes),
                    "--lighting-profile",
                    "material-neutral",
                    "--analysis-up-axis",
                    analysis_up_axis,
                    f"--analysis-front-axis={analysis_front_axis}",
                ],
                log_cb,
                command_runner=command_runner,
                retry_native_crash=True,
            )
            _require_file(
                candidate_rendered_registry,
                f"{stage_prefix}_render",
            )
            _run_stage(
                f"{stage_prefix}_compare",
                _multigroup_local_compare_command(
                    qwen_python=qwen_python,
                    reference_manifest=reference_manifest,
                    rendered_registry=candidate_rendered_registry,
                    view_map=tournament_view_map,
                    palette_fusion=palette_fusion_path,
                    group_id=group_id,
                    target_part_ids=target_part_ids,
                    target_entities=target_entities,
                    reference_view_ids=reference_view_ids,
                    output=candidate_quality_report,
                ),
                log_cb,
                command_runner=command_runner,
            )
            _require_file(
                candidate_quality_report,
                f"{stage_prefix}_compare",
            )
            quality_document = read_object(
                candidate_quality_report,
                f"membership candidate quality {candidate_id}",
            )
            rendered_bundles.append(
                {
                    "candidate_id": candidate_id,
                    "plan": candidate_plan,
                    "quality_report": quality_document,
                }
            )
            candidate_artifacts[candidate_id] = {
                "look": candidate_look_usd,
                "apply": candidate_apply_report,
                "quality": candidate_quality_report,
                "rendered_registry": candidate_rendered_registry,
                "applied_count": candidate_applied_count,
            }
            log_message(
                log_cb,
                "Dominant assembly membership candidate "
                f"[{candidate_index}/2] complete id={candidate_id}",
            )

        try:
            selected_plan, round_audit = select_membership_candidate(
                contract=round_contract,
                candidates=rendered_bundles,
                palette_fusion=palette_fusion,
            )
        except MembershipTournamentError as exc:
            raise RuntimeError(
                f"Dominant assembly membership cohort {cohort_id} could not "
                f"be selected safely; physics was not started: {exc}"
            ) from exc
        selected_id = str(round_audit["selected_candidate_id"])
        selected_artifacts = candidate_artifacts.get(selected_id)
        if selected_artifacts is None:
            raise RuntimeError(
                f"Dominant assembly membership cohort {cohort_id} selected an "
                "unknown render candidate"
            )
        round_audit_path = (
            tournament_dir / f"{cohort_id[:12]}_round_audit.json"
        )
        write_object(round_audit_path, round_audit)
        round_audits.append(round_audit)
        current_plan = selected_plan
        current_artifacts = selected_artifacts
        if round_audit["status"] == "ACCEPTED_EXPANDED_COHORT":
            selected_expanded_count += 1
        else:
            restored_m0_count += 1
        log_message(
            log_cb,
            "Dominant assembly membership decision "
            f"cohort={cohort_id[:12]} status={round_audit['status']} "
            f"selected={selected_id}",
        )

    final_quality_report = (
        tournament_dir / "final_reference_render_comparison.json"
    )
    _run_stage(
        "membership_final_whole_asset_compare",
        [
            str(qwen_python),
            "-m",
            "qwen_material_pipeline",
            "compare",
            "--reference-manifest",
            str(reference_manifest),
            "--rendered-registry",
            str(current_artifacts["rendered_registry"]),
            "--view-map",
            str(tournament_view_map),
            "--minimum-comparable-views",
            str(len(tournament_mapping)),
            "--output",
            str(final_quality_report),
        ],
        log_cb,
        command_runner=command_runner,
    )
    _require_file(
        final_quality_report,
        "membership_final_whole_asset_compare",
    )
    # Reuse the selected candidate's all-view renders but restore a whole-asset
    # comparison contract for every downstream exact-MDL round and final gate.
    current_artifacts["quality"] = final_quality_report

    write_object(output_plan, current_plan)
    final_audit = {
        "schema_version": (
            "asset-pipeline-dominant-assembly-membership-tournament/v1"
        ),
        "status": "COMPLETED",
        "initial_plan_sha256": canonical_sha256(initial_plan),
        "final_plan_sha256": canonical_sha256(current_plan),
        "palette_fusion_sha256": canonical_sha256(palette_fusion),
        "cohort_count": len(cohorts),
        "selected_expanded_cohort_count": selected_expanded_count,
        "restored_m0_count": restored_m0_count,
        "rounds": round_audits,
        "all_memberships_frozen_before_exact_mdl_tournament": True,
        "parameters_locked_to_library_defaults": True,
    }
    write_object(output_audit, final_audit)
    return {
        "status": "COMPLETED",
        "cohort_count": len(cohorts),
        "selected_expanded_cohort_count": selected_expanded_count,
        "restored_m0_count": restored_m0_count,
        "plan": output_plan,
        **current_artifacts,
        "audit": output_audit,
    }

__all__ = [
    "_baseline_preserved_disagreement_exemptions",
    "_final_baseline_preserved_disagreement_exemptions",
    "_log_exact_mdl_candidate_progress",
    "_log_exact_mdl_group_progress",
    "_multigroup_local_compare_command",
    "_run_dominant_assembly_membership_tournaments",
]
