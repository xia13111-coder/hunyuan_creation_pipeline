"""Hash-bound cache contract for exact NVIDIA MDL render candidates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..paths import unique_path
from .config import canonical_sha256, read_object, write_object
from .policy_contract import POLICY_FALLBACK_STATUS, SOURCE_VISUAL_PRESERVE_ACTION
from .references import sha256_file
from .stages.final_acceptance import _final_visual_axis_token


class _ExactMdlCandidateCacheError(RuntimeError):
    """One candidate cache entry cannot be consumed as an atomic bundle."""


def _exact_mdl_material_application_contract(
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the ordered plan fields that can affect USD material application.

    Candidate renders may be reused across a deterministic replay whose
    diagnostics/provenance changed without changing any authored material.
    This deliberately remains conservative: assignment order, applicability,
    material identity, parameters, face subsets and source-preserve identity
    are all part of the contract.  Only evidence and diagnostic provenance
    that the USD apply implementations do not consume is excluded.
    """

    assignments = plan.get("assignments")
    if not isinstance(assignments, list):
        raise _ExactMdlCandidateCacheError(
            "candidate plan lacks an assignments application contract"
        )
    plan_provenance = plan.get("provenance")
    contracted_plan_provenance = {
        field: (
            plan_provenance.get(field) if isinstance(plan_provenance, Mapping) else None
        )
        for field in (
            # Policy-fallback authorization.
            "mode",
            "registry_asset_sha256",
            "registry_sha256",
            # Instance-aware application source authorization.  Its
            # registry_sha256 is intentionally the shared field above.
            "asset_sha256",
        )
    }
    contracted_assignments: list[dict[str, Any]] = []
    for index, raw_assignment in enumerate(assignments):
        if not isinstance(raw_assignment, Mapping):
            raise _ExactMdlCandidateCacheError(
                f"candidate plan assignment {index} is not an object"
            )
        action = raw_assignment.get("apply_action")
        contracted_assignment: dict[str, Any] = {
            "part_id": raw_assignment.get("part_id"),
            "material_id": raw_assignment.get("material_id"),
            "status": raw_assignment.get("status", "review"),
            "confidence": raw_assignment.get("confidence"),
            "parameters": raw_assignment.get("parameters"),
            "face_subsets": raw_assignment.get("face_subsets"),
            "preserve_parent_material_binding": raw_assignment.get(
                "preserve_parent_material_binding",
                False,
            ),
            "apply_action": action,
            "source_visual_material_prim_path": raw_assignment.get(
                "source_visual_material_prim_path"
            ),
            "source_visual_material_binding_sha256": raw_assignment.get(
                "source_visual_material_binding_sha256"
            ),
        }
        if action == SOURCE_VISUAL_PRESERVE_ACTION:
            provenance = raw_assignment.get("provenance")
            contracted_assignment["source_visual_provenance_tier"] = (
                provenance.get("tier") if isinstance(provenance, Mapping) else None
            )
        if raw_assignment.get("status") == POLICY_FALLBACK_STATUS:
            provenance = raw_assignment.get("provenance")
            raw_sources = (
                provenance.get("sources") if isinstance(provenance, Mapping) else None
            )
            if isinstance(raw_sources, list):
                contracted_sources: Any = [
                    {
                        field: (
                            source.get(field) if isinstance(source, Mapping) else None
                        )
                        for field in (
                            "part_id",
                            "source_status",
                            "source_confidence",
                            "source_evidence_views",
                        )
                    }
                    for source in raw_sources
                ]
            else:
                contracted_sources = raw_sources
            contracted_assignment["policy_fallback_evidence_views"] = (
                raw_assignment.get("evidence_views")
            )
            contracted_assignment["policy_fallback_provenance"] = {
                "tier": (
                    provenance.get("tier") if isinstance(provenance, Mapping) else None
                ),
                "reason_codes": (
                    provenance.get("reason_codes")
                    if isinstance(provenance, Mapping)
                    else None
                ),
                "output_confidence_basis": (
                    provenance.get("output_confidence_basis")
                    if isinstance(provenance, Mapping)
                    else None
                ),
                "sources": contracted_sources,
            }
        contracted_assignments.append(contracted_assignment)
    return {
        "schema_version": plan.get("schema_version"),
        "plan_authorization_provenance": contracted_plan_provenance,
        "assignments": contracted_assignments,
    }


def _exact_mdl_cache_file(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise _ExactMdlCandidateCacheError(
            f"{label} is missing or is not a regular file: {path}"
        )
    try:
        return path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise _ExactMdlCandidateCacheError(
            f"{label} cannot be resolved: {path}"
        ) from exc


def _exact_mdl_cache_reported_path(
    value: Any,
    *,
    expected: Path,
    label: str,
) -> None:
    if not isinstance(value, str) or not value:
        raise _ExactMdlCandidateCacheError(f"{label} path is invalid")
    try:
        reported = Path(value).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise _ExactMdlCandidateCacheError(
            f"{label} path cannot be resolved: {value}"
        ) from exc
    if reported != expected:
        raise _ExactMdlCandidateCacheError(
            f"{label} path mismatch: expected={expected} reported={reported}"
        )


def _validate_exact_mdl_whole_asset_quality_cache(
    *,
    candidate_dir: Path,
    quality_path: Path,
    rendered_registry_path: Path,
    expected_mapping: Mapping[str, str],
    reference_manifest: Path,
) -> dict[str, Any]:
    """Validate the whole-asset guard paired with a group-local candidate.

    The group-local report remains the causal material evidence.  This second
    report seals the same render against every independently verified
    reference mapping so a locally attractive MDL cannot regress an excluded
    source view.
    """

    if candidate_dir.is_symlink() or not candidate_dir.is_dir():
        raise _ExactMdlCandidateCacheError(
            f"candidate cache directory is invalid: {candidate_dir}"
        )
    try:
        resolved_candidate_dir = candidate_dir.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise _ExactMdlCandidateCacheError(
            f"candidate cache directory cannot be resolved: {candidate_dir}"
        ) from exc
    resolved_quality_path = _exact_mdl_cache_file(
        quality_path,
        "candidate whole-asset quality report",
    )
    resolved_rendered_registry = _exact_mdl_cache_file(
        rendered_registry_path,
        "candidate rendered registry",
    )
    if not resolved_quality_path.is_relative_to(
        resolved_candidate_dir
    ) or not resolved_rendered_registry.is_relative_to(resolved_candidate_dir):
        raise _ExactMdlCandidateCacheError(
            "candidate whole-asset guard references a required file outside "
            "its directory"
        )
    try:
        quality_report = read_object(
            resolved_quality_path,
            "cached candidate whole-asset quality report",
        )
        rendered_registry = read_object(
            resolved_rendered_registry,
            "cached candidate rendered registry",
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise _ExactMdlCandidateCacheError(
            f"candidate whole-asset guard is unreadable: {exc}"
        ) from exc
    if quality_report.get("schema_version") != ("qwen-reference-render-comparison/v1"):
        raise _ExactMdlCandidateCacheError(
            "candidate whole-asset quality report schema is invalid"
        )
    quality_inputs = quality_report.get("inputs")
    if not isinstance(quality_inputs, dict):
        raise _ExactMdlCandidateCacheError(
            "candidate whole-asset quality report lacks inputs"
        )
    _exact_mdl_cache_reported_path(
        quality_inputs.get("rendered_registry"),
        expected=resolved_rendered_registry,
        label="candidate whole-asset quality rendered registry",
    )
    rendered_registry_file_sha256 = sha256_file(resolved_rendered_registry)
    if quality_inputs.get("rendered_registry_sha256") != (
        rendered_registry_file_sha256
    ):
        raise _ExactMdlCandidateCacheError(
            "candidate whole-asset quality/rendered-registry hash mismatch"
        )
    resolved_reference_manifest = _exact_mdl_cache_file(
        reference_manifest,
        "candidate whole-asset reference manifest",
    )
    _exact_mdl_cache_reported_path(
        quality_inputs.get("reference_manifest"),
        expected=resolved_reference_manifest,
        label="candidate whole-asset quality reference manifest",
    )
    if quality_inputs.get("reference_manifest_sha256") != sha256_file(
        resolved_reference_manifest
    ):
        raise _ExactMdlCandidateCacheError(
            "candidate whole-asset quality/reference-manifest hash mismatch"
        )
    if not isinstance(expected_mapping, Mapping) or any(
        not isinstance(reference_id, str)
        or not reference_id
        or not isinstance(render_id, str)
        or not render_id
        for reference_id, render_id in expected_mapping.items()
    ):
        raise _ExactMdlCandidateCacheError(
            "candidate whole-asset expected mapping is invalid"
        )
    expected_mapping_document = dict(sorted(expected_mapping.items()))
    if len(expected_mapping_document) < 2 or (
        quality_inputs.get("selected_view_mapping") != expected_mapping_document
        or quality_inputs.get("seeded_view_mapping") != expected_mapping_document
    ):
        raise _ExactMdlCandidateCacheError(
            "candidate whole-asset quality mapping differs from the verified "
            "tournament mapping"
        )
    if quality_inputs.get("comparison_scope") != {"mode": "whole_asset"}:
        raise _ExactMdlCandidateCacheError(
            "candidate whole-asset quality report has a non-global scope"
        )

    render_set = rendered_registry.get("render_set")
    raw_render_views = (
        render_set.get("views") if isinstance(render_set, Mapping) else None
    )
    if not isinstance(raw_render_views, list):
        raise _ExactMdlCandidateCacheError(
            "candidate whole-asset rendered registry lacks render views"
        )
    render_views: dict[str, Mapping[str, Any]] = {}
    for raw_view in raw_render_views:
        if not isinstance(raw_view, Mapping):
            raise _ExactMdlCandidateCacheError(
                "candidate whole-asset rendered registry has an invalid view"
            )
        view_id = raw_view.get("view_id")
        if not isinstance(view_id, str) or not view_id or view_id in render_views:
            raise _ExactMdlCandidateCacheError(
                "candidate whole-asset rendered registry repeats a view"
            )
        render_views[view_id] = raw_view

    raw_quality_views = quality_report.get("views")
    if not isinstance(raw_quality_views, list):
        raise _ExactMdlCandidateCacheError(
            "candidate whole-asset quality report lacks per-view evidence"
        )
    seen_reference_ids: set[str] = set()
    for raw_view in raw_quality_views:
        if not isinstance(raw_view, Mapping):
            raise _ExactMdlCandidateCacheError(
                "candidate whole-asset quality report has an invalid view"
            )
        reference_id = raw_view.get("reference_view_id")
        render_id = raw_view.get("render_view_id")
        if (
            not isinstance(reference_id, str)
            or reference_id in seen_reference_ids
            or reference_id not in expected_mapping_document
            or render_id != expected_mapping_document[reference_id]
            or render_id not in render_views
        ):
            raise _ExactMdlCandidateCacheError(
                "candidate whole-asset quality view identity/mapping mismatch"
            )
        seen_reference_ids.add(reference_id)
        reference = raw_view.get("reference")
        render = raw_view.get("render")
        if not isinstance(reference, Mapping) or not isinstance(render, Mapping):
            raise _ExactMdlCandidateCacheError(
                "candidate whole-asset quality view lacks reference/render evidence"
            )
        reference_image = _exact_mdl_cache_file(
            Path(str(reference.get("image", ""))).expanduser(),
            f"candidate whole-asset reference image {reference_id}",
        )
        if reference.get("image_sha256") != sha256_file(reference_image):
            raise _ExactMdlCandidateCacheError(
                f"candidate whole-asset reference image hash mismatch: {reference_id}"
            )
        registry_view = render_views[str(render_id)]
        for report_key, registry_key, hash_key in (
            ("image", "rgb", "image_sha256"),
            ("part_ids", "part_ids", "part_ids_sha256"),
        ):
            render_file = _exact_mdl_cache_file(
                Path(str(render.get(report_key, ""))).expanduser(),
                f"candidate whole-asset render {reference_id}/{report_key}",
            )
            registry_file = _exact_mdl_cache_file(
                Path(str(registry_view.get(registry_key, ""))).expanduser(),
                f"candidate whole-asset registry render {render_id}/{registry_key}",
            )
            if (
                render_file != registry_file
                or not render_file.is_relative_to(resolved_candidate_dir)
                or render.get(hash_key) != sha256_file(render_file)
            ):
                raise _ExactMdlCandidateCacheError(
                    "candidate whole-asset quality render hash mismatch: "
                    f"{reference_id}/{report_key}"
                )
    if seen_reference_ids != set(expected_mapping_document):
        raise _ExactMdlCandidateCacheError(
            "candidate whole-asset quality report does not cover every verified "
            "reference view"
        )
    return quality_report


def _validate_exact_mdl_candidate_cache(
    *,
    candidate_dir: Path,
    candidate_id: str,
    expected_plan: dict[str, Any],
    apply_asset: Path,
    occurrence_registry: dict[str, Any],
    expected_applied_count: int,
    expected_face_subset_count: int,
    expected_mapping: Mapping[str, str],
    expected_render_view_ids: Sequence[str],
    expected_reference_view_ids: Sequence[str],
    expected_render_resolution: int,
    expected_analysis_up_axis: str,
    expected_analysis_front_axis: str,
    reference_manifest: Path,
    palette_fusion: Path | None,
    target_group_id: str | None,
    target_part_ids: Sequence[str] = (),
    target_entities: Sequence[Mapping[str, Any]] = (),
    whole_asset_quality_path: Path | None = None,
) -> dict[str, Any]:
    """Load a hash-bound candidate, rebasing only a render-equivalent plan."""

    if (
        candidate_dir.is_symlink()
        or not candidate_dir.is_dir()
        or candidate_dir.name != candidate_id
    ):
        raise _ExactMdlCandidateCacheError(
            f"candidate cache directory is invalid: {candidate_dir}"
        )
    try:
        resolved_candidate_dir = candidate_dir.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise _ExactMdlCandidateCacheError(
            f"candidate cache directory cannot be resolved: {candidate_dir}"
        ) from exc

    plan_path = _exact_mdl_cache_file(candidate_dir / "plan.json", "candidate plan")
    look_path = _exact_mdl_cache_file(candidate_dir / "look.usda", "candidate Look")
    apply_path = _exact_mdl_cache_file(
        candidate_dir / "apply_report.json",
        "candidate apply report",
    )
    registry_path = _exact_mdl_cache_file(
        candidate_dir / "part_registry.json",
        "candidate registry",
    )
    rendered_registry_path = _exact_mdl_cache_file(
        candidate_dir / "renders" / "part_registry.rendered.json",
        "candidate rendered registry",
    )
    quality_path = _exact_mdl_cache_file(
        candidate_dir / "reference_render_comparison.json",
        "candidate quality report",
    )
    if not all(
        path.is_relative_to(resolved_candidate_dir)
        for path in (
            plan_path,
            look_path,
            apply_path,
            registry_path,
            rendered_registry_path,
            quality_path,
        )
    ):
        raise _ExactMdlCandidateCacheError(
            "candidate cache contains a required file outside its directory"
        )

    try:
        plan = read_object(plan_path, f"cached candidate plan {candidate_id}")
        apply_report = read_object(
            apply_path,
            f"cached candidate apply report {candidate_id}",
        )
        registry = read_object(
            registry_path,
            f"cached candidate registry {candidate_id}",
        )
        rendered_registry = read_object(
            rendered_registry_path,
            f"cached candidate rendered registry {candidate_id}",
        )
        quality_report = read_object(
            quality_path,
            f"cached candidate quality report {candidate_id}",
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise _ExactMdlCandidateCacheError(
            f"candidate cache JSON is unreadable: {exc}"
        ) from exc

    cached_plan_sha256 = canonical_sha256(plan)
    expected_plan_sha256 = canonical_sha256(expected_plan)
    if apply_report.get("plan_sha256") != cached_plan_sha256:
        raise _ExactMdlCandidateCacheError(
            "candidate cached-plan/apply-plan hash mismatch"
        )
    cached_application_contract = _exact_mdl_material_application_contract(plan)
    expected_application_contract = _exact_mdl_material_application_contract(
        expected_plan
    )
    cached_application_contract_sha256 = canonical_sha256(cached_application_contract)
    expected_application_contract_sha256 = canonical_sha256(
        expected_application_contract
    )
    if cached_application_contract_sha256 != expected_application_contract_sha256:
        raise _ExactMdlCandidateCacheError(
            "candidate material-application contract mismatch"
        )

    resolved_apply_asset = _exact_mdl_cache_file(
        apply_asset,
        "candidate source asset",
    )
    _exact_mdl_cache_reported_path(
        apply_report.get("source_usd"),
        expected=resolved_apply_asset,
        label="candidate apply source",
    )
    if apply_report.get("source_sha256") != sha256_file(resolved_apply_asset):
        raise _ExactMdlCandidateCacheError("candidate apply source hash mismatch")
    if apply_report.get("registry_sha256") != canonical_sha256(occurrence_registry):
        raise _ExactMdlCandidateCacheError(
            "candidate apply occurrence-registry hash mismatch"
        )
    _exact_mdl_cache_reported_path(
        apply_report.get("output_usd"),
        expected=look_path,
        label="candidate apply output",
    )
    look_sha256 = sha256_file(look_path)
    if apply_report.get("output_sha256") != look_sha256:
        raise _ExactMdlCandidateCacheError("candidate apply output/Look hash mismatch")
    if (
        apply_report.get("applied_count") != expected_applied_count
        or apply_report.get("face_subset_count") != expected_face_subset_count
    ):
        raise _ExactMdlCandidateCacheError(
            "candidate apply coverage or face-subset count mismatch"
        )

    for label, document in (
        ("candidate registry", registry),
        ("candidate rendered registry", rendered_registry),
    ):
        if (
            document.get("schema_version") != "qwen-material-parts/v1"
            or document.get("asset_sha256") != look_sha256
            or document.get("part_count") != occurrence_registry.get("part_count")
            or document.get("instance_root_count")
            != occurrence_registry.get("instance_root_count")
        ):
            raise _ExactMdlCandidateCacheError(
                f"{label} does not describe the exact candidate Look"
            )
        _exact_mdl_cache_reported_path(
            document.get("asset_usd"),
            expected=look_path,
            label=f"{label} asset",
        )

    render_set = rendered_registry.get("render_set")
    if not isinstance(render_set, dict):
        raise _ExactMdlCandidateCacheError(
            "candidate rendered registry lacks render_set"
        )
    expected_render_ids = list(expected_render_view_ids)
    if (
        render_set.get("resolution")
        != [expected_render_resolution, expected_render_resolution]
        or render_set.get("lighting_profile") != "material-neutral"
        or render_set.get("requested_view_tokens") != expected_render_ids
        or _final_visual_axis_token(
            render_set.get("analysis_up_axis"),
            "candidate cache analysis_up_axis",
        )
        != expected_analysis_up_axis
        or _final_visual_axis_token(
            render_set.get("analysis_front_axis"),
            "candidate cache analysis_front_axis",
        )
        != expected_analysis_front_axis
    ):
        raise _ExactMdlCandidateCacheError(
            "candidate render settings differ from the current tournament"
        )
    raw_render_views = render_set.get("views")
    if not isinstance(raw_render_views, list):
        raise _ExactMdlCandidateCacheError(
            "candidate rendered registry lacks render views"
        )
    render_views: dict[str, dict[str, Any]] = {}
    for raw_view in raw_render_views:
        if not isinstance(raw_view, dict):
            raise _ExactMdlCandidateCacheError(
                "candidate rendered registry contains an invalid view"
            )
        view_id = raw_view.get("view_id")
        if not isinstance(view_id, str) or not view_id or view_id in render_views:
            raise _ExactMdlCandidateCacheError(
                "candidate rendered registry repeats a view"
            )
        render_views[view_id] = raw_view
    if set(render_views) != set(expected_render_ids):
        raise _ExactMdlCandidateCacheError(
            "candidate render views differ from the current tournament"
        )

    if quality_report.get("schema_version") != "qwen-reference-render-comparison/v1":
        raise _ExactMdlCandidateCacheError("candidate quality report schema is invalid")
    quality_inputs = quality_report.get("inputs")
    if not isinstance(quality_inputs, dict):
        raise _ExactMdlCandidateCacheError("candidate quality report lacks inputs")
    _exact_mdl_cache_reported_path(
        quality_inputs.get("rendered_registry"),
        expected=rendered_registry_path,
        label="candidate quality rendered registry",
    )
    rendered_registry_file_sha256 = sha256_file(rendered_registry_path)
    if quality_inputs.get("rendered_registry_sha256") != rendered_registry_file_sha256:
        raise _ExactMdlCandidateCacheError(
            "candidate quality/rendered-registry file hash mismatch"
        )
    resolved_reference_manifest = _exact_mdl_cache_file(
        reference_manifest,
        "candidate reference manifest",
    )
    _exact_mdl_cache_reported_path(
        quality_inputs.get("reference_manifest"),
        expected=resolved_reference_manifest,
        label="candidate quality reference manifest",
    )
    if quality_inputs.get("reference_manifest_sha256") != sha256_file(
        resolved_reference_manifest
    ):
        raise _ExactMdlCandidateCacheError(
            "candidate quality/reference-manifest hash mismatch"
        )
    if isinstance(expected_reference_view_ids, (str, bytes)):
        raise _ExactMdlCandidateCacheError(
            "candidate expected reference-view scope is invalid"
        )
    raw_expected_references = list(expected_reference_view_ids)
    if (
        not raw_expected_references
        or any(
            not isinstance(view_id, str) or not view_id
            for view_id in raw_expected_references
        )
        or len(set(raw_expected_references)) != len(raw_expected_references)
    ):
        raise _ExactMdlCandidateCacheError(
            "candidate expected reference-view scope is invalid"
        )
    if not isinstance(expected_mapping, Mapping) or any(
        not isinstance(reference_id, str)
        or not reference_id
        or not isinstance(render_id, str)
        or not render_id
        for reference_id, render_id in expected_mapping.items()
    ):
        raise _ExactMdlCandidateCacheError("current tournament view mapping is invalid")
    expected_references = sorted(raw_expected_references)
    missing_expected_mappings = sorted(set(expected_references) - set(expected_mapping))
    if missing_expected_mappings:
        raise _ExactMdlCandidateCacheError(
            "current tournament view mapping does not cover candidate scope: "
            f"{missing_expected_mappings}"
        )
    # A group-local comparison deliberately removes global reference views in
    # which the canonical group is absent.  Seal the cache against that exact
    # declared scope: unrelated global views may exist in the tournament map,
    # but the cached selected/seeded maps may contain neither fewer nor more
    # entries than the candidate's comparison scope.
    expected_mapping_document = {
        reference_id: expected_mapping[reference_id]
        for reference_id in expected_references
    }
    if (
        quality_inputs.get("selected_view_mapping") != expected_mapping_document
        or quality_inputs.get("seeded_view_mapping") != expected_mapping_document
    ):
        raise _ExactMdlCandidateCacheError(
            "candidate quality view mapping differs from the current tournament"
        )

    comparison_scope = quality_inputs.get("comparison_scope")
    expected_parts = sorted(str(part_id) for part_id in target_part_ids)
    expected_entities = [dict(entity) for entity in target_entities]
    if target_group_id is None:
        if comparison_scope is not None:
            raise _ExactMdlCandidateCacheError(
                "whole-asset candidate unexpectedly has a group-local scope"
            )
    else:
        if (
            not isinstance(comparison_scope, dict)
            or comparison_scope.get("mode") != "canonical_group_local"
            or comparison_scope.get("target_group_id") != target_group_id
            or comparison_scope.get("target_part_ids") != expected_parts
            or comparison_scope.get("target_entities") != expected_entities
            or comparison_scope.get("reference_view_ids") != expected_references
        ):
            raise _ExactMdlCandidateCacheError(
                "candidate group-local comparison scope mismatch"
            )
        if palette_fusion is None:
            raise _ExactMdlCandidateCacheError(
                "group-local candidate lacks a palette-fusion contract"
            )
        resolved_palette_fusion = _exact_mdl_cache_file(
            palette_fusion,
            "candidate palette fusion",
        )
        _exact_mdl_cache_reported_path(
            comparison_scope.get("palette_fusion"),
            expected=resolved_palette_fusion,
            label="candidate comparison palette fusion",
        )
        if comparison_scope.get("palette_fusion_sha256") != sha256_file(
            resolved_palette_fusion
        ):
            raise _ExactMdlCandidateCacheError(
                "candidate comparison/palette-fusion hash mismatch"
            )

    raw_quality_views = quality_report.get("views")
    if not isinstance(raw_quality_views, list):
        raise _ExactMdlCandidateCacheError(
            "candidate quality report lacks per-view evidence"
        )
    seen_reference_ids: set[str] = set()
    for raw_view in raw_quality_views:
        if not isinstance(raw_view, dict):
            raise _ExactMdlCandidateCacheError(
                "candidate quality report contains an invalid view"
            )
        reference_id = raw_view.get("reference_view_id")
        render_id = raw_view.get("render_view_id")
        if (
            not isinstance(reference_id, str)
            or reference_id in seen_reference_ids
            or reference_id not in expected_mapping_document
            or render_id != expected_mapping_document[reference_id]
            or render_id not in render_views
        ):
            raise _ExactMdlCandidateCacheError(
                "candidate quality view identity/mapping mismatch"
            )
        seen_reference_ids.add(reference_id)
        reference = raw_view.get("reference")
        render = raw_view.get("render")
        if not isinstance(reference, dict) or not isinstance(render, dict):
            raise _ExactMdlCandidateCacheError(
                "candidate quality view lacks reference/render evidence"
            )
        reference_image = _exact_mdl_cache_file(
            Path(str(reference.get("image", ""))).expanduser(),
            f"candidate reference image {reference_id}",
        )
        if reference.get("image_sha256") != sha256_file(reference_image):
            raise _ExactMdlCandidateCacheError(
                f"candidate reference image hash mismatch: {reference_id}"
            )
        registry_view = render_views[str(render_id)]
        for report_key, registry_key, hash_key in (
            ("image", "rgb", "image_sha256"),
            ("part_ids", "part_ids", "part_ids_sha256"),
        ):
            render_file = _exact_mdl_cache_file(
                Path(str(render.get(report_key, ""))).expanduser(),
                f"candidate render {reference_id}/{report_key}",
            )
            registry_file = _exact_mdl_cache_file(
                Path(str(registry_view.get(registry_key, ""))).expanduser(),
                f"candidate registry render {render_id}/{registry_key}",
            )
            if (
                render_file != registry_file
                or not render_file.is_relative_to(resolved_candidate_dir)
                or render.get(hash_key) != sha256_file(render_file)
            ):
                raise _ExactMdlCandidateCacheError(
                    f"candidate quality render hash mismatch: "
                    f"{reference_id}/{report_key}"
                )
    if seen_reference_ids != set(expected_references):
        raise _ExactMdlCandidateCacheError(
            "candidate quality report does not cover the expected reference views"
        )

    cache_rebase: dict[str, Any] | None = None
    effective_apply_report = apply_report
    effective_apply_report_path = apply_path
    effective_plan_path = plan_path
    if cached_plan_sha256 != expected_plan_sha256:
        effective_plan_path = candidate_dir / "plan.cache_rebased.json"
        if effective_plan_path.is_symlink():
            raise _ExactMdlCandidateCacheError(
                "candidate rebased plan path is a symlink"
            )
        if effective_plan_path.exists() and not effective_plan_path.is_file():
            raise _ExactMdlCandidateCacheError(
                "candidate rebased plan path is not a regular file"
            )
        write_object(effective_plan_path, expected_plan)
        cache_rebase = {
            "schema_version": ("asset-pipeline-exact-mdl-candidate-cache-rebase/v1"),
            "status": "RENDER_EQUIVALENT_PLAN_REBASE",
            "candidate_id": candidate_id,
            "cached_plan_sha256": cached_plan_sha256,
            "expected_plan_sha256": expected_plan_sha256,
            "cached_material_application_contract_sha256": (
                cached_application_contract_sha256
            ),
            "expected_material_application_contract_sha256": (
                expected_application_contract_sha256
            ),
            "cached_apply_report_sha256": canonical_sha256(apply_report),
            "cached_apply_report_file_sha256": sha256_file(apply_path),
            "cached_apply_report": str(apply_path),
            "expected_plan": str(effective_plan_path),
            "expected_plan_file_sha256": sha256_file(effective_plan_path),
            "expected_plan_canonical_sha256": canonical_sha256(
                read_object(
                    effective_plan_path,
                    f"rebased candidate plan {candidate_id}",
                )
            ),
        }
        effective_apply_report = dict(apply_report)
        effective_apply_report["plan_sha256"] = expected_plan_sha256
        effective_apply_report["candidate_cache_rebase"] = cache_rebase
        effective_apply_report_path = candidate_dir / "apply_report.cache_rebased.json"
        if effective_apply_report_path.is_symlink():
            raise _ExactMdlCandidateCacheError(
                "candidate rebased apply report path is a symlink"
            )
        if (
            effective_apply_report_path.exists()
            and not effective_apply_report_path.is_file()
        ):
            raise _ExactMdlCandidateCacheError(
                "candidate rebased apply report path is not a regular file"
            )
        write_object(effective_apply_report_path, effective_apply_report)

    whole_asset_quality_report = (
        _validate_exact_mdl_whole_asset_quality_cache(
            candidate_dir=candidate_dir,
            quality_path=whole_asset_quality_path,
            rendered_registry_path=rendered_registry_path,
            expected_mapping=expected_mapping,
            reference_manifest=reference_manifest,
        )
        if whole_asset_quality_path is not None
        else None
    )
    return {
        # The selector must compare against the plan from the current replay,
        # not the historical diagnostic/provenance envelope.
        "plan": expected_plan,
        "cached_plan": plan,
        "plan_path": effective_plan_path,
        "apply_report": effective_apply_report,
        "apply_report_path": effective_apply_report_path,
        "cache_rebase": cache_rebase,
        "registry": registry,
        "quality_report": quality_report,
        "whole_asset_quality_report": whole_asset_quality_report,
        "rendered_registry": rendered_registry,
        "rendered_registry_file_sha256": rendered_registry_file_sha256,
    }


def _archive_exact_mdl_candidate_cache_entry(
    *,
    destination: Path,
    candidate_path: Path,
    reason: str,
) -> Path:
    """Reversibly move one invalid/incomplete candidate before regenerating it."""

    if not candidate_path.exists() and not candidate_path.is_symlink():
        raise FileNotFoundError(
            f"Exact-MDL cache entry no longer exists: {candidate_path}"
        )
    try:
        relative_path = candidate_path.relative_to(destination)
    except ValueError as exc:
        raise RuntimeError(
            f"Exact-MDL cache entry is outside the visual output: {candidate_path}"
        ) from exc
    archive_root = destination / "analysis" / "recovery_archive"
    archive_root.mkdir(parents=True, exist_ok=True)
    archive_dir = unique_path(archive_root / "exact_mdl_candidate_cache_miss")
    archive_dir.mkdir(parents=False, exist_ok=False)
    archived_path = archive_dir / relative_path
    manifest_path = archive_dir / "archive_manifest.json"
    manifest: dict[str, Any] = {
        "schema_version": "asset-pipeline-exact-mdl-cache-archive/v1",
        "status": "IN_PROGRESS",
        "reason": reason,
        "original": str(relative_path),
        "archived": str(archived_path.relative_to(archive_dir)),
    }
    write_object(manifest_path, manifest)
    archived_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_path.rename(archived_path)
    manifest["status"] = "COMPLETED"
    write_object(manifest_path, manifest)
    return archive_dir

__all__ = [
    "_ExactMdlCandidateCacheError",
    "_archive_exact_mdl_candidate_cache_entry",
    "_exact_mdl_material_application_contract",
    "_validate_exact_mdl_candidate_cache",
]
