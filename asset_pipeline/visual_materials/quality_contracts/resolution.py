"""Final quality-resolution and repaired-Look acceptance contracts."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

from ..config import canonical_sha256
from ..policy_contract import CORROBORATED_SOURCE_MDL_TIER, SOURCE_VISUAL_PRESERVE_ACTION
from ..policy_exact_cover import _policy_geometry_signature, _require_exact_int
from .constants import (
    QUALITY_REPAIR_ANCHORED_SINGLE_VIEW_LANE,
    QUALITY_REPAIR_BOUNDED_SIGNATURE_SIBLING_COHORT_LANE,
    QUALITY_REPAIR_DARK_FOREGROUND_MAX_SINGLE_CONTRIBUTION_FACTOR,
    QUALITY_REPAIR_DARK_FOREGROUND_MAX_TOTAL_CONTRIBUTION_FACTOR,
    QUALITY_REPAIR_DARK_FOREGROUND_RESIDUAL_DEFICIT_SOURCE,
    QUALITY_REPAIR_DARK_FOREGROUND_RESIDUAL_LANE,
    QUALITY_REPAIR_DOMINANT_RESIDUAL_DEFICIT_SOURCE,
    QUALITY_REPAIR_DOMINANT_RESIDUAL_SINGLE_VIEW_LANE,
    QUALITY_REPAIR_LOCALIZATION_LANES,
    QUALITY_REPAIR_MULTIVIEW_DARK_IDENTITY_LANE,
    QUALITY_REPAIR_REPEATED_GEOMETRY_DARK_RESIDUAL_LANE,
    QUALITY_REPAIR_SEMANTIC_SINGLE_VIEW_LANE,
    QUALITY_REPAIR_SOURCE_IDENTITY_COHORT_CONSENSUS_LANE,
    QUALITY_REPAIR_SOURCE_IDENTITY_LANE,
    QUALITY_REPAIR_SPATIAL_ANCHOR_SINGLE_VIEW_LANE,
    QUALITY_RESOLUTION_FAIL_CLOSED,
    QUALITY_RESOLUTION_LIMITED_PASS,
    QUALITY_RESOLUTION_SCHEMA_VERSION,
    QUALITY_RESOLUTION_THRESHOLDS,
    QUALITY_STATUSES,
)
from .metrics import (
    _quality_evidence_contract,
    _quality_finite_number,
    _quality_group_recalls,
    _quality_sha256,
    _quality_unit,
    _validate_quality_dominant_mass,
)
from qwen_material_pipeline.materials.quality_resolution import (
    LIMITATION_CLASSIFICATION as QUALITY_RESOLUTION_CLASSIFICATION,
    LIMITATION_REASON as QUALITY_RESOLUTION_REASON,
    build_quality_resolution,
)


def _validate_quality_resolution_bundle(
    *,
    resolution: dict[str, Any],
    final_plan: dict[str, Any],
    policy_audit: dict[str, Any],
    quality_report: dict[str, Any],
    palette_fusion: dict[str, Any],
    spatial_report: dict[str, Any],
    geometry_risk: dict[str, Any],
    rendered_registry: dict[str, Any],
) -> str:
    """Validate the independent final material gate and return its status."""

    if resolution.get("schema_version") != QUALITY_RESOLUTION_SCHEMA_VERSION:
        raise RuntimeError("Visual-quality resolution has an unsupported schema")
    expected_hashes = {
        "final_plan_sha256": canonical_sha256(final_plan),
        "policy_audit_sha256": canonical_sha256(policy_audit),
        "quality_report_sha256": canonical_sha256(quality_report),
        "palette_fusion_sha256": canonical_sha256(palette_fusion),
        "spatial_report_sha256": canonical_sha256(spatial_report),
        "geometry_risk_sha256": canonical_sha256(geometry_risk),
        "rendered_registry_sha256": canonical_sha256(rendered_registry),
    }
    if resolution.get("input_hashes") != expected_hashes:
        raise RuntimeError("Visual-quality resolution input hashes do not match")
    if resolution.get("thresholds") != QUALITY_RESOLUTION_THRESHOLDS:
        raise RuntimeError("Visual-quality resolution changed a safety threshold")
    try:
        expected_resolution = build_quality_resolution(
            final_plan=final_plan,
            policy_audit=policy_audit,
            quality_report=quality_report,
            palette_fusion=palette_fusion,
            spatial_report=spatial_report,
            geometry_risk=geometry_risk,
            rendered_registry=rendered_registry,
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Visual-quality resolution inputs cannot be recomputed: {exc}"
        ) from exc
    if resolution != expected_resolution:
        raise RuntimeError(
            "Visual-quality resolution is not the deterministic result of its "
            "hash-bound inputs"
        )

    aggregate = quality_report.get("aggregate")
    quality_views = quality_report.get("views")
    if not isinstance(aggregate, dict) or not isinstance(quality_views, list):
        raise RuntimeError("Visual-quality resolution input report is invalid")
    raw_status = aggregate.get("status")
    if (
        raw_status not in QUALITY_STATUSES
        or resolution.get("raw_quality_status") != raw_status
    ):
        raise RuntimeError("Visual-quality resolution changed the raw QA status")

    limitations = resolution.get("limitations")
    candidates = resolution.get("limitation_candidates")
    reasons = resolution.get("reason_codes")
    summary = resolution.get("summary")
    if (
        not isinstance(limitations, list)
        or not isinstance(candidates, list)
        or not isinstance(reasons, list)
        or reasons != sorted(set(reasons))
        or not isinstance(summary, dict)
    ):
        raise RuntimeError("Visual-quality resolution has an invalid decision shape")

    passed_view_count = sum(
        isinstance(view, dict) and view.get("status") == "PASS"
        for view in quality_views
    )
    failed_candidate_count = sum(
        not isinstance(item, dict) or item.get("eligible") is not True
        for item in candidates
    )
    accepted_limitation_count = _require_exact_int(
        summary.get("accepted_limitation_count"),
        "Visual-quality accepted limitation count",
    )
    if (
        _require_exact_int(
            summary.get("quality_view_count"),
            "Visual-quality view count",
        )
        != len(quality_views)
        or _require_exact_int(
            summary.get("passed_view_count"),
            "Visual-quality passing view count",
        )
        != passed_view_count
        or _require_exact_int(
            summary.get("fail_closed_candidate_count"),
            "Visual-quality failed candidate count",
        )
        != failed_candidate_count
    ):
        raise RuntimeError("Visual-quality resolution summary does not recompute")

    status = resolution.get("resolution_status")
    accepted = resolution.get("material_stage_accepted")
    if status == "PASS":
        if (
            raw_status != "PASS"
            or accepted is not True
            or reasons != ["RAW_VISUAL_QA_PASS"]
            or limitations
            or candidates
            or accepted_limitation_count != 0
        ):
            raise RuntimeError("Visual-quality PASS resolution is inconsistent")
        return status
    if status == QUALITY_RESOLUTION_FAIL_CLOSED:
        if (
            accepted is not False
            or not reasons
            or limitations
            or accepted_limitation_count != 0
        ):
            raise RuntimeError(
                "Visual-quality fail-closed resolution is inconsistent"
            )
        return status
    if status != QUALITY_RESOLUTION_LIMITED_PASS:
        raise RuntimeError(f"Unsupported visual-quality resolution: {status!r}")
    if (
        raw_status == "PASS"
        or accepted is not True
        or reasons
        or not limitations
        or limitations != candidates
        or accepted_limitation_count != len(limitations)
        or failed_candidate_count
        or passed_view_count < 2
    ):
        raise RuntimeError(
            "Geometry-pose-limited visual-quality resolution is inconsistent"
        )
    # The exact deterministic recomputation above validates every lane-specific
    # source binding, repeated-geometry cohort, cross-view delivery record,
    # alignment bound, and evidence hash.  Do not maintain a second partial
    # implementation here: divergence between a producer and a hand-copied
    # validator previously made new fail-closed evidence lanes impossible to
    # evolve safely.
    return status

    quality_by_reference: dict[str, dict[str, Any]] = {}
    nonpass_comparable: set[str] = set()
    for raw_view in quality_views:
        if not isinstance(raw_view, dict):
            raise RuntimeError("Visual-quality report contains an invalid view")
        reference_id = raw_view.get("reference_view_id")
        if (
            not isinstance(reference_id, str)
            or not reference_id
            or reference_id in quality_by_reference
        ):
            raise RuntimeError("Visual-quality report repeats a reference view")
        quality_by_reference[reference_id] = raw_view
        if raw_view.get("status") not in {"PASS", "UNSCORABLE"}:
            nonpass_comparable.add(reference_id)

    risk_by_part: dict[str, bool] = {}
    for raw_part in geometry_risk.get("parts", []):
        if not isinstance(raw_part, dict) or not isinstance(
            raw_part.get("risk"), dict
        ):
            raise RuntimeError("Geometry-risk evidence is invalid")
        part_id = raw_part.get("part_id")
        risk = raw_part["risk"].get("multi_material_risk")
        if (
            not isinstance(part_id, str)
            or part_id in risk_by_part
            or not isinstance(risk, bool)
        ):
            raise RuntimeError("Geometry-risk evidence is invalid")
        risk_by_part[part_id] = risk

    assignments: dict[str, dict[str, Any]] = {}
    raw_assignments = final_plan.get("assignments", [])
    if not isinstance(raw_assignments, list):
        raise RuntimeError("Final material plan lacks assignments")
    for raw_assignment in raw_assignments:
        if not isinstance(raw_assignment, dict):
            raise RuntimeError("Final material plan contains an invalid assignment")
        part_id = raw_assignment.get("part_id")
        if not isinstance(part_id, str) or part_id in assignments:
            raise RuntimeError("Final material plan repeats a part")
        assignments[part_id] = raw_assignment

    raw_registry_parts = rendered_registry.get("parts")
    render_set = rendered_registry.get("render_set")
    raw_registry_views = (
        render_set.get("views") if isinstance(render_set, dict) else None
    )
    if (
        not isinstance(raw_registry_parts, list)
        or not isinstance(raw_registry_views, list)
    ):
        raise RuntimeError("Final rendered registry is incomplete")
    registry_parts_by_id = {
        str(item["part_id"]): item
        for item in raw_registry_parts
        if isinstance(item, dict) and isinstance(item.get("part_id"), str)
    }
    registry_part_ids = set(registry_parts_by_id)
    if (
        len(registry_part_ids) != len(raw_registry_parts)
        or registry_part_ids != set(assignments)
    ):
        raise RuntimeError(
            "Final resolution plan does not exactly cover the rendered registry"
        )
    visible_by_view: dict[str, dict[str, int]] = {}
    for raw_view in raw_registry_views:
        if not isinstance(raw_view, dict):
            raise RuntimeError("Final rendered registry contains an invalid view")
        view_id = raw_view.get("view_id")
        raw_visible_parts = raw_view.get("visible_parts")
        if (
            not isinstance(view_id, str)
            or not view_id
            or view_id in visible_by_view
            or not isinstance(raw_visible_parts, list)
        ):
            raise RuntimeError("Final rendered registry view contract is invalid")
        visibility: dict[str, int] = {}
        for raw_visible in raw_visible_parts:
            if not isinstance(raw_visible, dict):
                raise RuntimeError("Final rendered visibility record is invalid")
            part_id = raw_visible.get("part_id")
            pixels = raw_visible.get("pixels")
            if (
                not isinstance(part_id, str)
                or part_id not in registry_part_ids
                or part_id in visibility
                or isinstance(pixels, bool)
                or not isinstance(pixels, int)
                or pixels < 1
            ):
                raise RuntimeError("Final rendered visibility record is invalid")
            visibility[part_id] = pixels
        visible_by_view[view_id] = visibility
    if len(visible_by_view) < 2:
        raise RuntimeError("Final rendered registry has insufficient poses")

    source_corroboration = policy_audit.get("corroborated_source_visual")
    raw_source_groups = (
        source_corroboration.get("groups")
        if isinstance(source_corroboration, dict)
        else None
    )
    applied_source_ids = (
        source_corroboration.get("applied_part_ids")
        if isinstance(source_corroboration, dict)
        else None
    )
    if (
        not isinstance(raw_source_groups, list)
        or not isinstance(applied_source_ids, list)
        or applied_source_ids != sorted(set(applied_source_ids))
    ):
        raise RuntimeError(
            "Visual-quality limitation lacks source-corroboration evidence"
        )
    source_groups_by_id: dict[str, dict[str, Any]] = {}
    for raw_group in raw_source_groups:
        if not isinstance(raw_group, dict):
            raise RuntimeError("Source-corroboration group is invalid")
        group_id = raw_group.get("group_id")
        if (
            not isinstance(group_id, str)
            or not group_id
            or group_id in source_groups_by_id
        ):
            raise RuntimeError("Source-corroboration group identity is invalid")
        source_groups_by_id[group_id] = raw_group

    covered_views: set[str] = set()
    limitation_keys: set[tuple[str, str, str]] = set()
    limited_share_by_view: Counter[str] = Counter()
    for limitation in limitations:
        if not isinstance(limitation, dict):
            raise RuntimeError("Visual-quality limitation is invalid")
        reference_id = limitation.get("reference_view_id")
        local_group_id = limitation.get("local_group_id")
        canonical_group_id = limitation.get("canonical_group_id")
        key = (reference_id, local_group_id, canonical_group_id)
        if (
            not all(isinstance(value, str) and value for value in key)
            or key in limitation_keys
            or limitation.get("classification")
            != QUALITY_RESOLUTION_CLASSIFICATION
            or limitation.get("reason_code") != QUALITY_RESOLUTION_REASON
            or limitation.get("eligible") is not True
            or limitation.get("reason_codes") != []
        ):
            raise RuntimeError("Visual-quality limitation identity is invalid")
        limitation_keys.add(key)
        covered_views.add(reference_id)
        core = {
            field: value
            for field, value in limitation.items()
            if field not in {"eligible", "reason_codes", "evidence_sha256"}
        }
        if limitation.get("evidence_sha256") != canonical_sha256(core):
            raise RuntimeError("Visual-quality limitation evidence hash is invalid")

        quality_view = quality_by_reference.get(reference_id)
        if quality_view is None or quality_view.get("status") in {
            "PASS",
            "UNSCORABLE",
        }:
            raise RuntimeError(
                "Visual-quality limitation does not cover a non-passing view"
            )
        reference = quality_view.get("reference")
        if (
            not isinstance(reference, dict)
            or limitation.get("reference_sha256")
            != reference.get("image_sha256")
            or limitation.get("selected_render_view_id")
            != quality_view.get("render_view_id")
        ):
            raise RuntimeError(
                "Visual-quality limitation changed its reference or render pose"
            )

        alignment = limitation.get("alignment")
        evidence = limitation.get("reference_group_evidence")
        candidate = limitation.get("candidate_geometry")
        owner = limitation.get("foreign_owner")
        if not all(
            isinstance(value, dict)
            for value in (alignment, evidence, candidate, owner)
        ):
            raise RuntimeError("Visual-quality limitation lacks required evidence")
        if (
            _quality_unit(alignment.get("score"), "limitation alignment score")
            < QUALITY_RESOLUTION_THRESHOLDS["minimum_alignment_score"]
            or _quality_unit(
                alignment.get("projection_iou"), "limitation projection IoU"
            )
            < QUALITY_RESOLUTION_THRESHOLDS["minimum_projection_iou"]
            or _quality_unit(
                alignment.get("ecc_correlation"), "limitation ECC"
            )
            < QUALITY_RESOLUTION_THRESHOLDS["minimum_ecc_correlation"]
        ):
            raise RuntimeError("Visual-quality limitation alignment is too weak")
        evidence_pixels = _require_exact_int(
            evidence.get("evidence_pixels"),
            "Visual-quality limitation evidence pixels",
        )
        reference_share = _quality_unit(
            evidence.get("reference_group_share"),
            "Visual-quality limitation reference share",
        )
        if (
            evidence_pixels
            < QUALITY_RESOLUTION_THRESHOLDS[
                "minimum_reference_evidence_pixels"
            ]
            or reference_share
            > QUALITY_RESOLUTION_THRESHOLDS["maximum_reference_group_share"]
        ):
            raise RuntimeError("Visual-quality limitation exceeds its evidence bound")
        limited_share_by_view[reference_id] += reference_share

        safe_part_ids = candidate.get("safe_part_ids")
        target_pixels = candidate.get("target_visible_pixels_by_part")
        safe_cohorts = candidate.get("safe_geometry_cohorts")
        target_render_view_id = limitation.get("selected_render_view_id")
        if (
            not isinstance(safe_part_ids, list)
            or safe_part_ids != sorted(set(safe_part_ids))
            or len(safe_part_ids)
            < QUALITY_RESOLUTION_THRESHOLDS["minimum_repeated_candidate_count"]
            or not isinstance(target_pixels, dict)
            or set(target_pixels) != set(candidate.get("eligible_part_ids", []))
            or not isinstance(safe_cohorts, list)
            or not safe_cohorts
            or target_render_view_id not in visible_by_view
        ):
            raise RuntimeError("Visual-quality candidate geometry is invalid")
        for part_id in safe_part_ids:
            assignment = assignments.get(part_id)
            registry_part = registry_parts_by_id.get(part_id)
            source_path = (
                registry_part.get("existing_visual_material")
                if isinstance(registry_part, dict)
                else None
            )
            expected_binding_hash = canonical_sha256(
                {
                    "part_id": part_id,
                    "prim_path": (
                        registry_part.get("prim_path")
                        if isinstance(registry_part, dict)
                        else None
                    ),
                    "source_visual_material_prim_path": source_path,
                }
            )
            assignment_provenance = (
                assignment.get("provenance")
                if isinstance(assignment, dict)
                else None
            )
            assignment_corroboration = (
                assignment_provenance.get("source_visual_corroboration")
                if isinstance(assignment_provenance, dict)
                else None
            )
            source_visual_preserved = (
                isinstance(assignment, dict)
                and assignment.get("apply_action")
                == SOURCE_VISUAL_PRESERVE_ACTION
                and isinstance(source_path, str)
                and assignment.get("source_visual_material_prim_path")
                == source_path
                and assignment.get("source_visual_material_binding_sha256")
                == expected_binding_hash
            )
            source_visual_represented_by_mdl = (
                isinstance(assignment, dict)
                and assignment.get("apply_action") is None
                and isinstance(assignment_provenance, dict)
                and assignment_provenance.get("tier")
                == CORROBORATED_SOURCE_MDL_TIER
                and isinstance(assignment_corroboration, dict)
                and assignment_corroboration.get("confirmed_material_id")
                == assignment.get("material_id")
            )
            if (
                target_pixels.get(part_id) != 0
                or visible_by_view[target_render_view_id].get(part_id, 0) != 0
                or risk_by_part.get(part_id) is not False
                or not isinstance(assignment, dict)
                or assignment.get("apply_action")
                != SOURCE_VISUAL_PRESERVE_ACTION
                or part_id not in applied_source_ids
                or not isinstance(source_path, str)
                or assignment.get("source_visual_material_prim_path")
                != source_path
                or assignment.get("source_visual_material_binding_sha256")
                != expected_binding_hash
                or not isinstance(assignment_corroboration, dict)
                or assignment_corroboration.get("canonical_group_id")
                != canonical_group_id
                or not (
                    source_visual_preserved
                    or source_visual_represented_by_mdl
                )
            ):
                raise RuntimeError(
                    "Visual-quality candidate geometry is not safely preserved"
                )
        source_group = source_groups_by_id.get(canonical_group_id)
        source_group_cohorts = (
            source_group.get("geometry_cohorts")
            if isinstance(source_group, dict)
            else None
        )
        if not isinstance(source_group_cohorts, list):
            raise RuntimeError(
                "Visual-quality candidate lacks a source geometry cohort"
            )
        for cohort in safe_cohorts:
            if not isinstance(cohort, dict):
                raise RuntimeError("Visual-quality repeated cohort is invalid")
            cohort_parts = cohort.get("part_ids")
            visible_elsewhere = cohort.get(
                "other_visible_render_view_ids_by_part"
            )
            if (
                not isinstance(cohort_parts, list)
                or not set(cohort_parts) <= set(safe_part_ids)
                or cohort.get("repeat_count") != len(cohort_parts)
                or len(cohort_parts)
                < QUALITY_RESOLUTION_THRESHOLDS[
                    "minimum_repeated_candidate_count"
                ]
                or not isinstance(visible_elsewhere, dict)
                or any(
                    not isinstance(visible_elsewhere.get(part_id), list)
                    or visible_elsewhere[part_id]
                    != sorted(
                        view_id
                        for view_id, visibility in visible_by_view.items()
                        if view_id != target_render_view_id
                        and visibility.get(part_id, 0) > 0
                    )
                    or len(set(visible_elsewhere[part_id]))
                    < QUALITY_RESOLUTION_THRESHOLDS[
                        "minimum_other_visible_render_views"
                    ]
                    for part_id in cohort_parts
                )
            ):
                raise RuntimeError("Visual-quality repeated cohort is not observable")
            matching_source_cohorts = [
                raw_source_cohort
                for raw_source_cohort in source_group_cohorts
                if isinstance(raw_source_cohort, dict)
                and raw_source_cohort.get("geometry_signature_sha256")
                == cohort.get("geometry_signature_sha256")
                and raw_source_cohort.get("part_ids") == cohort_parts
                and raw_source_cohort.get("repeat_count") == len(cohort_parts)
            ]
            if len(matching_source_cohorts) != 1:
                raise RuntimeError(
                    "Visual-quality repeated cohort is not policy-bound"
                )
            source_cohort = matching_source_cohorts[0]
            expected_geometry = {
                "point_count": source_cohort.get("point_count"),
                "face_count": source_cohort.get("face_count"),
                "sorted_bbox_extents": source_cohort.get(
                    "sorted_bbox_extents"
                ),
            }
            if any(
                _policy_geometry_signature(
                    registry_parts_by_id[part_id],
                    label=f"final registry part {part_id}",
                )
                != expected_geometry
                for part_id in cohort_parts
            ):
                raise RuntimeError(
                    "Visual-quality repeated geometry is not reproducible"
                )

        owner_part_id = owner.get("part_id")
        owner_group_id = owner.get("canonical_group_id")
        owner_assignment = assignments.get(owner_part_id)
        owner_provenance = (
            owner_assignment.get("provenance")
            if isinstance(owner_assignment, dict)
            else None
        )
        if (
            owner.get("eligible") is not True
            or owner.get("reason_codes") != []
            or owner_part_id in safe_part_ids
            or owner_group_id == canonical_group_id
            or not isinstance(owner_provenance, dict)
            or owner_provenance.get("canonical_group_id") != owner_group_id
            or _require_exact_int(
                owner.get("projected_part_pixels"),
                "Visual-quality owner projected pixels",
            )
            < QUALITY_RESOLUTION_THRESHOLDS["minimum_owner_projected_pixels"]
            or _quality_unit(
                owner.get("direct_color_share"), "owner direct color share"
            )
            < QUALITY_RESOLUTION_THRESHOLDS["minimum_owner_color_share"]
            or _quality_unit(
                owner.get("bbox_color_share"), "owner bbox color share"
            )
            < QUALITY_RESOLUTION_THRESHOLDS["minimum_owner_color_share"]
            or _quality_unit(
                owner.get("direct_color_margin"), "owner direct color margin"
            )
            < QUALITY_RESOLUTION_THRESHOLDS["minimum_owner_color_margin"]
            or _quality_unit(
                owner.get("bbox_color_margin"), "owner bbox color margin"
            )
            < QUALITY_RESOLUTION_THRESHOLDS["minimum_owner_color_margin"]
            or _quality_unit(
                owner.get("target_color_share"), "owner target color share"
            )
            > QUALITY_RESOLUTION_THRESHOLDS["maximum_target_share_in_owner"]
        ):
            raise RuntimeError("Visual-quality foreign owner is not stable")
        perturbations = owner.get("perturbations")
        offsets = {
            tuple(item.get("offset_pixels", []))
            for item in perturbations
            if isinstance(item, dict)
        } if isinstance(perturbations, list) else set()
        if (
            not isinstance(perturbations, list)
            or len(perturbations) != 4
            or offsets != {(-2, 0), (2, 0), (0, -2), (0, 2)}
            or any(
                item.get("canonical_group_id") != owner_group_id
                or item.get("diagnostic_canonical_group_id") != owner_group_id
                for item in perturbations
            )
        ):
            raise RuntimeError("Visual-quality foreign owner is perturbation-unstable")
        overlap = owner.get("accepted_box_overlap")
        if (
            not isinstance(overlap, dict)
            or _require_exact_int(
                overlap.get("evidence_pixel_count"),
                "Visual-quality accepted-box evidence pixels",
            )
            != evidence_pixels
            or _quality_unit(
                overlap.get("projected_overlap_share"),
                "Visual-quality accepted-box owner overlap",
            )
            < QUALITY_RESOLUTION_THRESHOLDS[
                "minimum_accepted_box_owner_overlap"
            ]
        ):
            raise RuntimeError(
                "Visual-quality limitation lacks accepted-box owner overlap"
            )

    if covered_views != nonpass_comparable:
        raise RuntimeError(
            "Visual-quality limitations do not exactly cover non-passing views"
        )
    if any(
        share
        > QUALITY_RESOLUTION_THRESHOLDS["maximum_limited_share_per_view"]
        for share in limited_share_by_view.values()
    ):
        raise RuntimeError("Visual-quality limitations exceed a per-view share bound")
    return status



def _validate_quality_repair_outcome(
    *,
    baseline_quality: dict[str, Any],
    repaired_quality: dict[str, Any],
    repair_audit: dict[str, Any],
    allow_verified_pose_limitation: bool = False,
    allow_pending_immutable_tournament: bool = False,
) -> None:
    """Require causal recovery with no group/view regression."""

    baseline_aggregate = baseline_quality.get("aggregate")
    aggregate = repaired_quality.get("aggregate")
    if (
        not isinstance(baseline_aggregate, dict)
        or baseline_aggregate.get("status") != "FAIL"
        or not isinstance(aggregate, dict)
        or (
            aggregate.get("status") != "PASS"
            and not allow_verified_pose_limitation
            and not allow_pending_immutable_tournament
        )
        or isinstance(aggregate.get("comparable_view_count"), bool)
        or not isinstance(aggregate.get("comparable_view_count"), int)
        or aggregate["comparable_view_count"] < 2
    ):
        raise RuntimeError(
            "The bounded quality-repair round did not reach an accepted outcome"
        )
    if _quality_evidence_contract(baseline_quality) != _quality_evidence_contract(
        repaired_quality
    ):
        raise RuntimeError(
            "Quality-repair round changed references, mapping, or QA thresholds"
        )
    if allow_pending_immutable_tournament and aggregate.get("status") != "PASS":
        return
    baseline_dominant = _validate_quality_dominant_mass(baseline_quality)
    repaired_dominant = _validate_quality_dominant_mass(repaired_quality)
    if baseline_dominant["enabled"] != repaired_dominant["enabled"]:
        raise RuntimeError(
            "Quality-repair round changed the dominant-mass QA contract"
        )
    if baseline_dominant["enabled"]:
        baseline_dominant_views = baseline_dominant["views"]
        repaired_dominant_views = repaired_dominant["views"]
        if set(baseline_dominant_views) != set(repaired_dominant_views):
            raise RuntimeError(
                "Quality-repair round changed dominant-mass view coverage"
            )
        immutable_fields = (
            "family_key",
            "render_color_bins",
            "local_group_ids",
            "base_colors",
            "reference_share",
            "runner_up_reference_share",
            "reference_share_margin",
        )
        for view_id, baseline_view in baseline_dominant_views.items():
            repaired_view = repaired_dominant_views[view_id]
            baseline_families = baseline_view["families"]
            repaired_families = repaired_view["families"]
            if set(baseline_families) != set(repaired_families):
                raise RuntimeError(
                    "Quality-repair round changed dominant-family identity"
                )
            for family_key, baseline_family in baseline_families.items():
                repaired_family = repaired_families[family_key]
                if any(
                    baseline_family[field] != repaired_family[field]
                    for field in immutable_fields
                ):
                    raise RuntimeError(
                        "Quality-repair round changed dominant-family reference "
                        f"evidence: {view_id}/{family_key}"
                    )
        if repaired_dominant.get("failed_view_ids"):
            raise RuntimeError(
                "Quality repair left a dominant-family mass deficit"
            )
    baseline_threshold, baseline_recalls, baseline_statuses = (
        _quality_group_recalls(baseline_quality)
    )
    repaired_threshold, repaired_recalls, repaired_statuses = (
        _quality_group_recalls(repaired_quality)
    )
    if repaired_threshold != baseline_threshold:
        raise RuntimeError("Quality-repair round changed the QA threshold")
    for view_id, status in baseline_statuses.items():
        if status == "PASS" and repaired_statuses.get(view_id) != "PASS":
            raise RuntimeError(
                f"Quality repair regressed a previously passing view: {view_id}"
            )
    for view_id, groups in baseline_recalls.items():
        for group_id, recall in groups.items():
            if recall < baseline_threshold:
                continue
            repaired_recall = repaired_recalls.get(view_id, {}).get(group_id)
            if repaired_recall is None or repaired_recall < baseline_threshold:
                raise RuntimeError(
                    "Quality repair introduced a new trusted-group deficit: "
                    f"{view_id}/{group_id}"
                )

    def recovered_low_evidence_delivery(
        *,
        view_id: str,
        local_group_id: str,
    ) -> bool:
        """Recompute the bounded tiny-region delivery exception."""

        thresholds = repaired_quality.get("thresholds")
        raw_views = repaired_quality.get("views")
        if not isinstance(thresholds, dict) or not isinstance(raw_views, list):
            return False
        raw_minimum_pixels = thresholds.get(
            "minimum_reliable_group_evidence_pixels"
        )
        raw_tolerance = thresholds.get("low_evidence_recall_tolerance_ratio")
        raw_minimum_observed = thresholds.get(
            "minimum_low_evidence_observed_render_share"
        )
        if (
            isinstance(raw_minimum_pixels, bool)
            or not isinstance(raw_minimum_pixels, int)
            or raw_minimum_pixels < 1
            or isinstance(raw_tolerance, bool)
            or not isinstance(raw_tolerance, (int, float))
            or not 0.0 < float(raw_tolerance) <= 1.0
            or isinstance(raw_minimum_observed, bool)
            or not isinstance(raw_minimum_observed, (int, float))
            or not 0.0 <= float(raw_minimum_observed) <= 1.0
        ):
            return False
        matches = [
            raw_view
            for raw_view in raw_views
            if isinstance(raw_view, dict)
            and raw_view.get("reference_view_id") == view_id
        ]
        if len(matches) != 1:
            return False
        material_color = matches[0].get("material_color")
        group_recall = (
            material_color.get("trusted_evidence_group_recall")
            if isinstance(material_color, dict)
            else None
        )
        raw_groups = (
            group_recall.get("groups")
            if isinstance(group_recall, dict)
            else None
        )
        if not isinstance(raw_groups, list):
            return False
        groups = [
            raw_group
            for raw_group in raw_groups
            if isinstance(raw_group, dict)
            and raw_group.get("group_id") == local_group_id
        ]
        if len(groups) != 1:
            return False
        group = groups[0]
        raw_evidence_pixels = group.get("reference_evidence_weight")
        raw_recall = group.get("recall")
        raw_observed_share = group.get("observed_render_share")
        return (
            group.get("delivery_presence_status")
            == "LOW_EVIDENCE_NEAR_THRESHOLD_PRESENT"
            and not isinstance(raw_evidence_pixels, bool)
            and isinstance(raw_evidence_pixels, int)
            and 0 <= raw_evidence_pixels < raw_minimum_pixels
            and not isinstance(raw_recall, bool)
            and isinstance(raw_recall, (int, float))
            and repaired_threshold * float(raw_tolerance)
            <= float(raw_recall)
            < repaired_threshold
            and not isinstance(raw_observed_share, bool)
            and isinstance(raw_observed_share, (int, float))
            and float(raw_minimum_observed)
            <= float(raw_observed_share)
            <= 1.0
        )

    def dark_measurement(
        report: dict[str, Any],
        *,
        view_id: str,
    ) -> dict[str, Any]:
        raw_views = report.get("views")
        if not isinstance(raw_views, list):
            raise RuntimeError("Dark-foreground outcome lacks QA views")
        matches = [
            item
            for item in raw_views
            if isinstance(item, dict) and item.get("reference_view_id") == view_id
        ]
        if len(matches) != 1:
            raise RuntimeError("Dark-foreground outcome QA view is not unique")
        view = matches[0]
        reference = view.get("reference")
        render = view.get("render")
        material_color = view.get("material_color")
        alignment = view.get("alignment")
        if (
            not isinstance(reference, dict)
            or not isinstance(render, dict)
            or not isinstance(material_color, dict)
            or not isinstance(alignment, dict)
        ):
            raise RuntimeError("Dark-foreground outcome QA evidence is incomplete")
        reference_foreground = reference.get("foreground")
        render_foreground = render.get("foreground")
        reference_pixels = _require_exact_int(
            (
                reference_foreground.get("pixel_count")
                if isinstance(reference_foreground, dict)
                else None
            ),
            "Dark-foreground outcome reference pixels",
            minimum=1,
        )
        render_pixels = _require_exact_int(
            (
                render_foreground.get("pixel_count")
                if isinstance(render_foreground, dict)
                else None
            ),
            "Dark-foreground outcome render pixels",
            minimum=1,
        )
        reference_distribution = material_color.get("reference_distribution")
        render_distribution = material_color.get("render_distribution")
        if not isinstance(reference_distribution, dict) or not isinstance(
            render_distribution, dict
        ):
            raise RuntimeError(
                "Dark-foreground outcome color distributions are incomplete"
            )
        if (
            reference_distribution.get("sample_step") != 1
            or render_distribution.get("sample_step") != 1
            or reference_distribution.get("sampled_pixels") != reference_pixels
            or render_distribution.get("sampled_pixels") != render_pixels
        ):
            raise RuntimeError(
                "Dark-foreground outcome is not exact-pixel QA evidence"
            )
        reference_categories = reference_distribution.get(
            "category_distribution"
        )
        render_categories = render_distribution.get("category_distribution")
        if not isinstance(reference_categories, dict) or not isinstance(
            render_categories, dict
        ):
            raise RuntimeError(
                "Dark-foreground outcome black-family categories are missing"
            )
        try:
            reference_share = sum(
                _quality_unit(
                    reference_categories[label],
                    f"Dark-foreground outcome reference {label}",
                )
                for label in ("black", "achromatic_dark")
            )
            render_share = sum(
                _quality_unit(
                    render_categories[label],
                    f"Dark-foreground outcome render {label}",
                )
                for label in ("black", "achromatic_dark")
            )
        except KeyError as exc:
            raise RuntimeError(
                "Dark-foreground outcome black-family categories are missing"
            ) from exc
        deficit_share = max(0.0, reference_share - render_share)
        mass_recall = (
            min(1.0, render_share / reference_share)
            if reference_share > 0.0
            else 1.0
        )
        return {
            "view": view,
            "reference_distribution": reference_distribution,
            "reference_pixels": reference_pixels,
            "render_pixels": render_pixels,
            "reference_share": reference_share,
            "render_share": render_share,
            "deficit_share": deficit_share,
            "mass_recall": mass_recall,
            "alignment": alignment,
        }

    def require_recovered_support(support: dict[str, Any]) -> None:
        view_id = support.get("reference_view_id")
        local_group_id = support.get("local_group_id")
        if not isinstance(view_id, str) or not isinstance(local_group_id, str):
            raise RuntimeError("Quality-repair support record is invalid")
        raw_sources = support.get("deficit_sources")
        if raw_sources is None:
            sources = ["group_recall"]
        elif (
            not isinstance(raw_sources, list)
            or raw_sources != sorted(set(raw_sources))
            or not raw_sources
            or any(
                source
                not in {
                    "group_recall",
                    "dominant_mass",
                    QUALITY_REPAIR_DOMINANT_RESIDUAL_DEFICIT_SOURCE,
                    QUALITY_REPAIR_DARK_FOREGROUND_RESIDUAL_DEFICIT_SOURCE,
                }
                for source in raw_sources
            )
        ):
            raise RuntimeError("Quality-repair deficit sources are invalid")
        else:
            sources = raw_sources

        recovered = False
        if "group_recall" in sources:
            baseline_recall = baseline_recalls.get(view_id, {}).get(local_group_id)
            repaired_recall = repaired_recalls.get(view_id, {}).get(local_group_id)
            if baseline_recall is None or baseline_recall >= baseline_threshold:
                raise RuntimeError(
                    "Quality repair target was not a baseline evidence deficit: "
                    f"{view_id}/{local_group_id}"
                )
            if repaired_recall is None or (
                repaired_recall < repaired_threshold
                and not recovered_low_evidence_delivery(
                    view_id=view_id,
                    local_group_id=local_group_id,
                )
            ):
                raise RuntimeError(
                    "Quality repair did not recover its targeted evidence group: "
                    f"{view_id}/{local_group_id}"
                )
            recovered = True
        if "dominant_mass" in sources:
            if not baseline_dominant["enabled"]:
                raise RuntimeError(
                    "Quality-repair dominant support lacks a QA contract"
                )
            family_key = support.get("dominant_mass_family_key")
            if not isinstance(family_key, str) or not family_key:
                raise RuntimeError(
                    "Quality-repair dominant support lacks a family identity"
                )
            baseline_family = (
                baseline_dominant["views"]
                .get(view_id, {})
                .get("families", {})
                .get(family_key)
            )
            repaired_family = (
                repaired_dominant["views"]
                .get(view_id, {})
                .get("families", {})
                .get(family_key)
            )
            if (
                not isinstance(baseline_family, dict)
                or baseline_family.get("status") != "FAIL"
                or local_group_id
                not in baseline_family.get("local_group_ids", [])
            ):
                raise RuntimeError(
                    "Quality repair target was not a baseline dominant-mass "
                    f"deficit: {view_id}/{family_key}"
                )
            if (
                not isinstance(repaired_family, dict)
                or repaired_family.get("status") == "FAIL"
            ):
                raise RuntimeError(
                    "Quality repair did not recover its targeted dominant family: "
                    f"{view_id}/{family_key}"
                )
            recovered = True
        if QUALITY_REPAIR_DARK_FOREGROUND_RESIDUAL_DEFICIT_SOURCE in sources:
            if not local_group_id.startswith("__canonical_dark__:"):
                raise RuntimeError(
                    "Dark-foreground residual support has an invalid group identity"
                )
            baseline_dark = dark_measurement(
                baseline_quality,
                view_id=view_id,
            )
            repaired_dark = dark_measurement(
                repaired_quality,
                view_id=view_id,
            )
            if (
                repaired_dark["reference_distribution"]
                != baseline_dark["reference_distribution"]
                or repaired_dark["reference_pixels"]
                != baseline_dark["reference_pixels"]
            ):
                raise RuntimeError(
                    "Dark-foreground repair changed its reference color evidence"
                )
            support_alignment = support.get("alignment")
            expected_alignment = {
                field: _quality_unit(
                    baseline_dark["alignment"].get(field),
                    f"Dark-foreground baseline alignment {field}",
                )
                for field in (
                    "score",
                    "silhouette_iou",
                    "edge_f1_tolerance_3px",
                    "profile_similarity",
                    "bbox_aspect_similarity",
                )
            }
            expected_alignment = dict(sorted(expected_alignment.items()))
            if support_alignment != expected_alignment or any(
                not math.isclose(
                    _quality_unit(
                        repaired_dark["alignment"].get(field),
                        f"Dark-foreground repaired alignment {field}",
                    ),
                    expected_alignment[field],
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
                for field in expected_alignment
            ):
                raise RuntimeError(
                    "Dark-foreground repair changed its trusted QA alignment"
                )
            expected_baseline_fields = {
                "reference_share": baseline_dark["reference_share"],
                "observed_render_share": baseline_dark["render_share"],
                "deficit_share": baseline_dark["deficit_share"],
                "recall": baseline_dark["mass_recall"],
                "mass_recall": baseline_dark["mass_recall"],
            }
            if any(
                not math.isclose(
                    _quality_unit(
                        support.get(field),
                        f"Dark-foreground support {field}",
                    ),
                    value,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
                for field, value in expected_baseline_fields.items()
            ) or (
                support.get("normalized_reference_pixels")
                != baseline_dark["reference_pixels"]
                or support.get("render_foreground_pixels")
                != baseline_dark["render_pixels"]
                or support.get("budget_pixels")
                != math.ceil(
                    baseline_dark["deficit_share"]
                    * baseline_dark["reference_pixels"]
                )
                or support.get("budget_limit_pixels")
                != math.floor(
                    QUALITY_REPAIR_DARK_FOREGROUND_MAX_TOTAL_CONTRIBUTION_FACTOR
                    * int(support["budget_pixels"])
                )
            ):
                raise RuntimeError(
                    "Dark-foreground residual support does not match baseline evidence"
                )
            if (
                repaired_dark["render_share"] <= baseline_dark["render_share"]
                or repaired_dark["deficit_share"]
                >= baseline_dark["deficit_share"]
                or repaired_dark["mass_recall"] <= baseline_dark["mass_recall"]
            ):
                raise RuntimeError(
                    "Quality repair did not improve its dark-foreground residual: "
                    f"{view_id}"
                )
            if (
                repaired_dark["mass_recall"] < 0.80
                and repaired_dark["deficit_share"] >= 0.025
            ):
                raise RuntimeError(
                    "Quality repair did not recover its dark-foreground residual: "
                    f"{view_id}"
                )
            recovered = True
        if QUALITY_REPAIR_DOMINANT_RESIDUAL_DEFICIT_SOURCE in sources:
            if not baseline_dominant["enabled"]:
                raise RuntimeError(
                    "Quality-repair local dominant residual lacks a QA contract"
                )
            family_key = support.get("dominant_mass_family_key")
            if not isinstance(family_key, str) or not family_key:
                raise RuntimeError(
                    "Quality-repair local dominant residual lacks a family identity"
                )
            baseline_family = (
                baseline_dominant["views"]
                .get(view_id, {})
                .get("families", {})
                .get(family_key)
            )
            repaired_family = (
                repaired_dominant["views"]
                .get(view_id, {})
                .get("families", {})
                .get(family_key)
            )
            if (
                not isinstance(baseline_family, dict)
                or baseline_family.get("status") != "NOT_APPLICABLE"
                or baseline_family.get("reason_codes")
                != ["SILHOUETTE_IOU_BELOW_DOMINANT_FLOOR"]
                or local_group_id
                not in baseline_family.get("local_group_ids", [])
                or support.get("requires_strict_local_projection") is not True
            ):
                raise RuntimeError(
                    "Quality repair target was not a silhouette-limited local "
                    f"dominant residual: {view_id}/{family_key}"
                )
            if not isinstance(repaired_family, dict):
                raise RuntimeError(
                    "Quality repair lost its targeted local dominant family: "
                    f"{view_id}/{family_key}"
                )
            support_numeric_fields = {
                "reference_share": "reference_share",
                "observed_render_share": "observed_render_share",
                "deficit_share": "deficit_share",
                "mass_recall": "mass_recall",
            }
            for support_field, family_field in support_numeric_fields.items():
                support_value = _quality_unit(
                    support.get(support_field),
                    (
                        "Quality-repair local dominant residual "
                        f"{view_id}/{family_key}.{support_field}"
                    ),
                )
                if not math.isclose(
                    support_value,
                    baseline_family[family_field],
                    rel_tol=0.0,
                    abs_tol=1e-9,
                ):
                    raise RuntimeError(
                        "Quality-repair local dominant residual support does "
                        "not match baseline evidence: "
                        f"{view_id}/{family_key}/{support_field}"
                    )
            baseline_observed = baseline_family["observed_render_share"]
            baseline_deficit = baseline_family["deficit_share"]
            baseline_mass_recall = baseline_family["mass_recall"]
            repaired_observed = repaired_family["observed_render_share"]
            repaired_deficit = repaired_family["deficit_share"]
            repaired_mass_recall = repaired_family["mass_recall"]
            if (
                repaired_observed <= baseline_observed
                or repaired_deficit >= baseline_deficit
                or repaired_mass_recall <= baseline_mass_recall
            ):
                raise RuntimeError(
                    "Quality repair did not improve its targeted local dominant "
                    f"residual: {view_id}/{family_key}"
                )
            dominant_thresholds = baseline_dominant["thresholds"]
            if (
                repaired_mass_recall
                < dominant_thresholds["minimum_dominant_mass_recall"]
                and repaired_deficit
                >= dominant_thresholds["minimum_dominant_absolute_deficit"]
            ):
                raise RuntimeError(
                    "Quality repair did not recover its targeted local dominant "
                    f"residual: {view_id}/{family_key}"
                )
            recovered = True
        elif "requires_strict_local_projection" in support:
            raise RuntimeError(
                "Quality-repair strict local projection is not bound to a "
                "local dominant residual"
            )
        if not recovered:
            raise RuntimeError("Quality-repair support has no recoverable deficit")

    diagnostics = repair_audit.get("group_diagnostics")
    changes = repair_audit.get("changes")
    localization_lanes = repair_audit.get("localization_lanes")
    if (
        not isinstance(diagnostics, list)
        or not isinstance(changes, list)
        or not isinstance(localization_lanes, list)
    ):
        raise RuntimeError("Quality-repair audit lacks outcome diagnostics")
    diagnostics_by_group: dict[str, dict[str, Any]] = {}
    for diagnostic in diagnostics:
        if not isinstance(diagnostic, dict):
            raise RuntimeError("Quality-repair group diagnostic is invalid")
        group_id = diagnostic.get("canonical_group_id")
        if not isinstance(group_id, str) or group_id in diagnostics_by_group:
            raise RuntimeError("Quality-repair group diagnostic is invalid")
        diagnostics_by_group[group_id] = diagnostic

    lanes_by_part: dict[str, dict[str, Any]] = {}
    for lane in localization_lanes:
        if not isinstance(lane, dict):
            raise RuntimeError("Quality-repair localization lane is invalid")
        part_id = lane.get("part_id")
        if (
            not isinstance(part_id, str)
            or part_id in lanes_by_part
            or lane.get("lane") not in QUALITY_REPAIR_LOCALIZATION_LANES
            or not isinstance(lane.get("canonical_group_id"), str)
        ):
            raise RuntimeError("Quality-repair localization lane is invalid")
        lanes_by_part[part_id] = lane

    changed_part_ids: set[str] = set()
    for change in changes:
        if not isinstance(change, dict):
            raise RuntimeError("Quality-repair change record is invalid")
        part_id = change.get("part_id")
        group_id = change.get("canonical_group_id")
        support_view_ids = change.get("supporting_view_ids")
        lane = lanes_by_part.get(part_id) if isinstance(part_id, str) else None
        lane_name = lane.get("lane") if isinstance(lane, dict) else None
        minimum_support_count = (
            1
            if lane_name
            in {
                "exact_spatial_single_qa_view",
                QUALITY_REPAIR_SEMANTIC_SINGLE_VIEW_LANE,
                QUALITY_REPAIR_ANCHORED_SINGLE_VIEW_LANE,
                QUALITY_REPAIR_SPATIAL_ANCHOR_SINGLE_VIEW_LANE,
                QUALITY_REPAIR_DOMINANT_RESIDUAL_SINGLE_VIEW_LANE,
                QUALITY_REPAIR_DARK_FOREGROUND_RESIDUAL_LANE,
                QUALITY_REPAIR_REPEATED_GEOMETRY_DARK_RESIDUAL_LANE,
                QUALITY_REPAIR_SOURCE_IDENTITY_LANE,
                QUALITY_REPAIR_SOURCE_IDENTITY_COHORT_CONSENSUS_LANE,
                QUALITY_REPAIR_BOUNDED_SIGNATURE_SIBLING_COHORT_LANE,
            }
            else 2
        )
        exact_single_support = minimum_support_count == 1
        anchored_single_support = (
            lane_name == QUALITY_REPAIR_ANCHORED_SINGLE_VIEW_LANE
        )
        if (
            not isinstance(part_id, str)
            or part_id in changed_part_ids
            or not isinstance(group_id, str)
            or not isinstance(support_view_ids, list)
            or (
                len(support_view_ids) != 1
                if exact_single_support
                else len(support_view_ids) < minimum_support_count
            )
            or support_view_ids != sorted(set(support_view_ids))
            or any(not isinstance(view_id, str) for view_id in support_view_ids)
        ):
            raise RuntimeError("Quality-repair change support is invalid")
        changed_part_ids.add(part_id)
        if lane is None or lane.get("canonical_group_id") != group_id:
            raise RuntimeError(
                "Quality-repair change lacks a matching spatial localization lane"
            )
        diagnostic = diagnostics_by_group.get(group_id)
        dark_residual_lane = lane_name in {
            QUALITY_REPAIR_DARK_FOREGROUND_RESIDUAL_LANE,
            QUALITY_REPAIR_MULTIVIEW_DARK_IDENTITY_LANE,
        }
        diagnostic_authorized = (
            diagnostic is not None
            and (
                diagnostic.get("repairable") is True
                or (
                    dark_residual_lane
                    and diagnostic.get("dark_residual_repairable") is True
                )
                or (
                    lane.get("lane") == "exact_spatial_single_qa_view"
                    and diagnostic.get("single_view_spatial_repairable") is True
                )
            )
        )
        if not diagnostic_authorized:
            raise RuntimeError(
                "Quality-repair change lacks a repairable QA group deficit"
            )
        supports = diagnostic.get(
            (
                "dominant_residual_supporting_views"
                if lane_name
                == QUALITY_REPAIR_DOMINANT_RESIDUAL_SINGLE_VIEW_LANE
                else (
                    "dark_residual_supporting_views"
                    if dark_residual_lane
                    else "supporting_views"
                )
            )
        )
        if not isinstance(supports, list):
            raise RuntimeError("Quality-repair support diagnostics are invalid")
        support_by_view: dict[str, dict[str, Any]] = {}
        for support in supports:
            if not isinstance(support, dict):
                raise RuntimeError("Quality-repair support record is invalid")
            view_id = support.get("reference_view_id")
            local_group_id = support.get("local_group_id")
            if (
                not isinstance(view_id, str)
                or view_id in support_by_view
                or not isinstance(local_group_id, str)
            ):
                raise RuntimeError("Quality-repair support record is invalid")
            support_by_view[view_id] = support
        if (
            lane_name != QUALITY_REPAIR_MULTIVIEW_DARK_IDENTITY_LANE
            and not set(support_view_ids) <= set(support_by_view)
        ):
            raise RuntimeError(
                "Quality-repair part support is outside the QA deficit views"
            )
        if lane_name in {
            QUALITY_REPAIR_DARK_FOREGROUND_RESIDUAL_LANE,
            QUALITY_REPAIR_MULTIVIEW_DARK_IDENTITY_LANE,
        }:
            if len(support_by_view) != 1:
                raise RuntimeError(
                    "Dark-foreground outcome lacks one global QA residual"
                )
            dark_support = next(iter(support_by_view.values()))
            dark_diagnostic = change.get("dark_foreground_diagnostic")
            if (
                dark_support.get("local_group_id")
                != f"__canonical_dark__:{group_id}"
                or change.get("dark_residual_support") != dark_support
                or not isinstance(dark_diagnostic, dict)
                or dark_diagnostic.get(
                    "estimated_contribution_pixels"
                )
                != change.get("estimated_contribution_pixels")
                or (
                    lane_name == QUALITY_REPAIR_MULTIVIEW_DARK_IDENTITY_LANE
                    and (
                        dark_diagnostic.get("lane")
                        != QUALITY_REPAIR_MULTIVIEW_DARK_IDENTITY_LANE
                        or dark_diagnostic.get("supporting_view_ids")
                        != support_view_ids
                    )
                )
            ):
                raise RuntimeError(
                    "Dark-foreground outcome support audit is inconsistent"
                )
            for field in (
                "budget_pixels",
                "budget_limit_pixels",
                "existing_contribution_pixels",
                "estimated_contribution_pixels",
                "selected_contribution_pixels",
                "cumulative_contribution_pixels",
            ):
                _require_exact_int(
                    change.get(field),
                    f"Dark-foreground outcome {field}",
                )
            if (
                change["budget_pixels"] != dark_support.get("budget_pixels")
                or change["budget_limit_pixels"]
                != dark_support.get("budget_limit_pixels")
                or change["selected_contribution_pixels"]
                != change["estimated_contribution_pixels"]
                or change["estimated_contribution_pixels"] < 1
                or change["estimated_contribution_pixels"]
                > math.floor(
                    QUALITY_REPAIR_DARK_FOREGROUND_MAX_SINGLE_CONTRIBUTION_FACTOR
                    * change["budget_pixels"]
                )
                or change["cumulative_contribution_pixels"]
                > change["budget_limit_pixels"]
            ):
                raise RuntimeError(
                    "Dark-foreground outcome exceeds its contribution budget"
                )
        spatial_anchor_view_ids = change.get("spatial_anchor_view_ids")
        if lane_name == QUALITY_REPAIR_SPATIAL_ANCHOR_SINGLE_VIEW_LANE:
            if (
                not isinstance(spatial_anchor_view_ids, list)
                or not spatial_anchor_view_ids
                or spatial_anchor_view_ids
                != sorted(set(spatial_anchor_view_ids))
                or any(
                    not isinstance(view_id, str) or not view_id
                    for view_id in spatial_anchor_view_ids
                )
                or set(spatial_anchor_view_ids) & set(support_view_ids)
            ):
                raise RuntimeError(
                    "Quality-repair spatial anchor relationship is invalid"
                )
        elif "spatial_anchor_view_ids" in change:
            raise RuntimeError(
                "Quality-repair change has unauthorized spatial anchors"
            )
        if anchored_single_support:
            deficit_view_ids = sorted(support_by_view)
            anchor_part_ids = change.get("anchor_part_ids")
            anchor_supporting_view_ids = change.get(
                "anchor_supporting_view_ids"
            )
            expected_anchor_part_ids = sorted(
                str(anchor_change["part_id"])
                for anchor_change in changes
                if isinstance(anchor_change, dict)
                and isinstance(anchor_change.get("part_id"), str)
                and anchor_change.get("part_id") != part_id
                and anchor_change.get("canonical_group_id") == group_id
                and isinstance(
                    lanes_by_part.get(str(anchor_change["part_id"])), dict
                )
                and lanes_by_part[str(anchor_change["part_id"])].get("lane")
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
                    for anchor_change in changes
                    if isinstance(anchor_change, dict)
                    and anchor_change.get("part_id") in expected_anchor_part_ids
                    for view_id in anchor_change["supporting_view_ids"]
                }
            )
            if (
                len(deficit_view_ids) < 2
                or len(support_view_ids) != 1
                or not set(support_view_ids) < set(deficit_view_ids)
                or not isinstance(anchor_part_ids, list)
                or anchor_part_ids != expected_anchor_part_ids
                or not expected_anchor_part_ids
                or expected_anchor_view_ids != deficit_view_ids
                or anchor_part_ids != sorted(set(anchor_part_ids))
                or not isinstance(anchor_supporting_view_ids, list)
                or anchor_supporting_view_ids != deficit_view_ids
            ):
                raise RuntimeError(
                    "Quality-repair anchored single-view relationship is invalid"
                )
        elif (
            "anchor_part_ids" in change
            or "anchor_supporting_view_ids" in change
        ):
            raise RuntimeError(
                "Quality-repair change has unauthorized anchor fields"
            )
        for view_id in support_view_ids:
            require_recovered_support(support_by_view[view_id])

    if changed_part_ids != set(lanes_by_part):
        raise RuntimeError("Quality-repair localization lanes do not cover changes")
    if not changed_part_ids:
        raise RuntimeError("Quality-repair outcome has no changed parts")

    dark_change_ids = sorted(
        part_id
        for part_id, lane in lanes_by_part.items()
        if lane.get("lane")
        in {
            QUALITY_REPAIR_DARK_FOREGROUND_RESIDUAL_LANE,
            QUALITY_REPAIR_MULTIVIEW_DARK_IDENTITY_LANE,
        }
    )
    if dark_change_ids:
        raw_budgets = repair_audit.get("dark_residual_budgets")
        if not isinstance(raw_budgets, list):
            raise RuntimeError(
                "Dark-foreground outcome lacks its contribution budget"
            )
        budgets_by_group: dict[str, dict[str, Any]] = {}
        for raw_budget in raw_budgets:
            group_id = (
                raw_budget.get("canonical_group_id")
                if isinstance(raw_budget, dict)
                else None
            )
            if (
                not isinstance(group_id, str)
                or group_id in budgets_by_group
                or not isinstance(raw_budget, dict)
            ):
                raise RuntimeError(
                    "Dark-foreground outcome contribution budget is invalid"
                )
            budgets_by_group[group_id] = raw_budget
        dark_group_ids = {
            str(lanes_by_part[part_id]["canonical_group_id"])
            for part_id in dark_change_ids
        }
        if not dark_group_ids <= set(budgets_by_group):
            raise RuntimeError(
                "Dark-foreground outcome budget does not cover its changes"
            )
        changes_by_id = {
            str(change["part_id"]): change
            for change in changes
            if isinstance(change, dict)
            and isinstance(change.get("part_id"), str)
        }
        for group_id in sorted(dark_group_ids):
            budget = budgets_by_group[group_id]
            budget_pixels = _require_exact_int(
                budget.get("budget_pixels"),
                "Dark-foreground outcome budget pixels",
                minimum=1,
            )
            budget_limit = _require_exact_int(
                budget.get("budget_limit_pixels"),
                "Dark-foreground outcome budget limit",
                minimum=1,
            )
            per_part_limit = _require_exact_int(
                budget.get("per_part_limit_pixels"),
                "Dark-foreground outcome per-part limit",
                minimum=1,
            )
            existing_contribution = _require_exact_int(
                budget.get("existing_contribution_pixels"),
                "Dark-foreground outcome existing contribution",
            )
            raw_existing_parts = budget.get("existing_parts")
            if (
                budget_limit
                != math.floor(
                    QUALITY_REPAIR_DARK_FOREGROUND_MAX_TOTAL_CONTRIBUTION_FACTOR
                    * budget_pixels
                )
                or per_part_limit
                != math.floor(
                    QUALITY_REPAIR_DARK_FOREGROUND_MAX_SINGLE_CONTRIBUTION_FACTOR
                    * budget_pixels
                )
                or not isinstance(raw_existing_parts, list)
                or existing_contribution
                != sum(
                    _require_exact_int(
                        item.get("estimated_contribution_pixels"),
                        "Dark-foreground existing-part contribution",
                    )
                    for item in raw_existing_parts
                    if isinstance(item, dict)
                )
                or any(not isinstance(item, dict) for item in raw_existing_parts)
            ):
                raise RuntimeError(
                    "Dark-foreground outcome existing budget is inconsistent"
                )
            raw_candidates = budget.get("candidates")
            if not isinstance(raw_candidates, list):
                raise RuntimeError(
                    "Dark-foreground outcome candidates are invalid"
                )
            ordering: list[tuple[float, str]] = []
            cumulative = existing_contribution
            selected_ids: list[str] = []
            selected_contribution = 0
            candidate_by_part: dict[str, dict[str, Any]] = {}
            for candidate in raw_candidates:
                if not isinstance(candidate, dict):
                    raise RuntimeError(
                        "Dark-foreground outcome candidate is invalid"
                    )
                candidate_part_id = candidate.get("part_id")
                evidence_strength = _quality_finite_number(
                    candidate.get("evidence_strength"),
                    "Dark-foreground outcome evidence strength",
                    minimum=0.0,
                )
                contribution = _require_exact_int(
                    candidate.get("estimated_contribution_pixels"),
                    "Dark-foreground outcome candidate contribution",
                )
                if (
                    not isinstance(candidate_part_id, str)
                    or candidate_part_id in candidate_by_part
                    or not _quality_sha256(candidate.get("diagnostic_sha256"))
                ):
                    raise RuntimeError(
                        "Dark-foreground outcome candidate identity is invalid"
                    )
                ordering.append((-evidence_strength, candidate_part_id))
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
                    selected_ids.append(candidate_part_id)
                if (
                    candidate.get("selected") is not selected
                    or candidate.get("reason_code") != reason_code
                    or candidate.get("cumulative_contribution_pixels")
                    != cumulative
                ):
                    raise RuntimeError(
                        "Dark-foreground outcome candidate budget is inconsistent"
                    )
                candidate_by_part[candidate_part_id] = candidate
            expected_selected = sorted(
                part_id
                for part_id in dark_change_ids
                if lanes_by_part[part_id]["canonical_group_id"] == group_id
            )
            if (
                ordering != sorted(ordering)
                or budget.get("selected_part_ids") != sorted(selected_ids)
                or sorted(selected_ids) != expected_selected
                or budget.get("selected_contribution_pixels")
                != selected_contribution
                or budget.get("total_contribution_pixels") != cumulative
                or cumulative > budget_limit
            ):
                raise RuntimeError(
                    "Dark-foreground outcome total budget is inconsistent"
                )
            for part_id in expected_selected:
                change = changes_by_id[part_id]
                candidate = candidate_by_part[part_id]
                if (
                    change.get("budget_pixels") != budget_pixels
                    or change.get("budget_limit_pixels") != budget_limit
                    or change.get("existing_contribution_pixels")
                    != existing_contribution
                    or change.get("estimated_contribution_pixels")
                    != candidate.get("estimated_contribution_pixels")
                    or change.get("selected_contribution_pixels")
                    != candidate.get("estimated_contribution_pixels")
                    or change.get("cumulative_contribution_pixels")
                    != candidate.get("cumulative_contribution_pixels")
                    or change.get("dark_foreground_diagnostic", {}).get(
                        "diagnostic_sha256"
                    )
                    != candidate.get("diagnostic_sha256")
                ):
                    raise RuntimeError(
                        "Dark-foreground outcome per-part budget is inconsistent"
                    )

    changed_groups = {
        str(item["canonical_group_id"])
        for item in changes
        if isinstance(item, dict)
        and isinstance(item.get("canonical_group_id"), str)
    }
    for diagnostic in diagnostics:
        if diagnostic.get("canonical_group_id") not in changed_groups:
            continue
        supports = diagnostic.get("supporting_views")
        if not isinstance(supports, list):
            raise RuntimeError("Quality-repair support diagnostics are invalid")
        for support in supports:
            if not isinstance(support, dict):
                raise RuntimeError("Quality-repair support record is invalid")
            require_recovered_support(support)

__all__ = [
    "QUALITY_RESOLUTION_CLASSIFICATION",
    "QUALITY_RESOLUTION_REASON",
    "_validate_quality_repair_outcome",
    "_validate_quality_resolution_bundle",
]
