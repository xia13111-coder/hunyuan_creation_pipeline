"""Cross-runtime orchestration for automatic visual-material assignment.

The main pipeline must not import ``pxr`` or Transformers. This module only
starts explicit subprocess stages and exchanges validated JSON/USD artifacts.

Public call graph::

    asset_pipeline.manual_cad
      -> run_assign_visual_materials_job
         -> VisualMaterialPipelineContext.create (validated run state)
         -> stages.source_preparation            (registry/render/alignment)
         -> stages.material_inference            (Qwen/MVInverse/recovery)
         -> _run_policy_part_id_stage            (plan/Part-ID decisions)
         -> _run_look_application_stage          (candidate Look USD)
         -> _run_visual_qa_stage                 (fail-closed render QA)
         -> _run_material_selection_stage        (bounded tournaments/seal)
         -> _run_finalize_assignment_stage       (selection lock/final Look)
         -> stages.final_acceptance              (collected-USD gate)

Cross-stage policy remains here; path layout, command construction, execution
mechanics, and bounded stage implementations have independently testable owners.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..command import LogCallback, log_message, run_command
from ..paths import unique_path
from ..project_layout import ProjectLayout
from ..runtime import isaac_python, root_dir
from .bundled_projects import (
    BundledMaterialProject,
)
from .camera import (
    continuous_camera_view_specs as _continuous_camera_view_specs,
    render_view_arguments as _render_view_arguments,
    require_complete_live_camera_alignment as _require_complete_live_camera_alignment,
    validate_live_camera_registration_provenance as _validate_live_camera_registration_provenance,
)
from .config import (
    DEFAULT_CONFIG_PATH,
    VisualMaterialConfig,
    canonical_sha256,
    load_visual_material_config,
    read_object,
    write_object,
)
from .commands import (
    policy_exact_cover_command,
)
from .corresponding_color import (
    corresponding_material_part_ids,
    rebind_part_id_audit_for_corresponding_color,
    validate_corresponding_color_result,
)
from .context import VisualMaterialPipelineContext
from .exact_mdl_cache import (
    _ExactMdlCandidateCacheError,
    _archive_exact_mdl_candidate_cache_entry,
    _exact_mdl_material_application_contract,
    _validate_exact_mdl_candidate_cache,
    _validate_exact_mdl_whole_asset_quality_cache,
)
from .contracts import (
    ISOLATED_ENV_REMOVE,
    RESULT_SCHEMA_VERSION,
    USD_SUFFIXES,
)
from .immutable_optimum import (
    DECISION as IMMUTABLE_LIBRARY_OPTIMUM_DECISION,
    evaluate_immutable_library_optimum,
)
from .quality import (
    evaluate_part_id_quality_gate as _evaluate_part_id_quality_gate,
    part_id_quality_scope_from_camera_alignment as _part_id_quality_scope_from_camera_alignment,
    validated_exact_mdl_tournament_mapping as _validated_exact_mdl_tournament_mapping,
)
from .quality_validation import (
    APPEARANCE_OPTIMIZATION_STATUSES,
    GENERIC_STEEL_PAINTED,
    MATERIAL_SELECTION_OBJECTIVE_VISUAL,
    QUALITY_REPAIR_PLAN_MODE,
    QUALITY_REPAIR_PROVENANCE_FIELD,
    QUALITY_REPAIR_REASON_CODES,
    QUALITY_REPAIR_REPORT_SCHEMA_VERSION,
    QUALITY_RESOLUTION_FAIL_CLOSED,
    QUALITY_RESOLUTION_LIMITED_PASS,
    QUALITY_STATUSES,
    _appearance_baseline_safety_reason,
    _quality_can_measure_lighting_statistics,
    _quality_has_lighting_normalized_groups,
    _validate_quality_dominant_mass,
    _validate_quality_render_contract,
    _validate_quality_repair_bundle,
    _validate_quality_repair_dominant_assembly_cohorts,
    _validate_quality_repair_outcome,
    _validate_quality_resolution_bundle,
)
from .policy_contract import (
    APPLICABLE_ASSIGNMENT_STATUSES,
    CORROBORATED_SOURCE_MDL_TIER,
    POLICY_FALLBACK_CONFIDENCE_BASIS,
    POLICY_FALLBACK_STATUS,
    POLICY_INPUT_SCHEMA_VERSION,
    POLICY_PLAN_MODE,
    POLICY_REPORT_SCHEMA_VERSION,
)
from .policy_exact_cover import (
    _archive_stale_policy_exact_cover_checkpoint,
    _policy_checkpoint_matches_requested_overrides,
    _require_exact_int,
    _validate_corroborated_source_visual_assignments,
    _validate_policy_exact_cover_bundle,
)
from .references import parse_visual_references, sha256_file
from .sealed_dependencies import verify_sealed_dependency_lock
from .stages.common import require_file as _require_file
from .stages.final_acceptance import run_final_visual_acceptance_job
from .stages.material_inference import (
    _prepare_live_material_catalog,
    _run_qwen_mvinverse_with_recovery,
    run_material_inference,
)
from .stages.runner import _run_stage, _visual_control_cpu_stability_guard
from .stages.source_preparation import SourcePreparationResult, prepare_source_evidence
from .tournaments import (
    _baseline_preserved_disagreement_exemptions,
    _final_baseline_preserved_disagreement_exemptions,
    _log_exact_mdl_candidate_progress,
    _log_exact_mdl_group_progress,
    _multigroup_local_compare_command,
    _run_dominant_assembly_membership_tournaments,
)
from qwen_material_pipeline.materials.disagreement_tournament import (
    DisagreementTournamentContractError,
    disagreement_is_render_confirmed,
    validate_disagreement_tournament_contract,
)
from qwen_material_pipeline.materials.exact_mdl_tournament import (
    ExactMdlTournamentError,
    build_bounded_exact_mdl_candidate_plans,
    build_part_family_contract,
    select_and_replay_exact_mdl_candidate,
)
from qwen_material_pipeline.materials.part_id_parameter_tournament import (
    PartIdParameterTournamentError,
    build_h1_candidate_plan,
    pending_h1_part_ids,
    rebind_part_id_material_audit,
    score_part_id_render,
    select_parameter_tournament_winners,
)
from qwen_material_pipeline.materials.component_mdl_tournament import (
    ComponentMdlTournamentError,
    build_component_candidate_plan,
    component_candidate_material_ids,
    rebind_part_id_material_audit_for_component_mdl_tournament,
    score_component_render,
    select_component_mdl_winner,
)
from qwen_material_pipeline.materials.multigroup_exact_mdl_tournament import (
    MultigroupExactMdlTournamentError,
    build_exact_mdl_group_candidate_plans,
    build_multigroup_exact_mdl_queue,
    finalize_multigroup_exact_mdl_plan,
    select_exact_mdl_group_step,
)
from qwen_material_pipeline.materials.publish_quality_gate import (
    PublishQualityGateError,
    PublishQualityPolicy,
    build_publish_quality_gate,
    require_publish_quality_gate_passed,
)
from qwen_material_pipeline.materials.policy_exact_cover import (
    PolicyExactCoverError,
    build_policy_exact_cover,
)
from qwen_material_pipeline.materials.selection_lock import (
    build_material_selection_lock,
    validate_material_selection_lock,
)
from qwen_material_pipeline.workflows.corresponding_material_color_workflow import (
    CorrespondingMaterialColorWorkflowError,
    run_corresponding_material_color_workflow,
)
from qwen_material_pipeline.materials.visual_group_annotation import (
    VisualGroupAnnotationError,
    annotate_visual_groups,
)
from qwen_material_pipeline.evidence.appearance_component_material import (
    AppearanceComponentMaterialError,
    apply_fixed_component_mdl_choices,
    build_component_material_inputs,
    filter_components_for_material_evidence,
)
from qwen_material_pipeline.evidence.part_id_projection import (
    build_part_id_material_plan,
    build_part_id_reference_evidence,
    build_part_id_retrieval_request,
)
from qwen_material_pipeline.workflows.part_id_qwen import (
    MINIMUM_MATERIAL_SPECIES_CONFIDENCE,
    _catalog_material_species,
)
from qwen_material_pipeline.scripts.build_part_id_sam3_request import (
    build_request as build_part_id_sam3_request,
)


def _complete_coverage_assignment_statuses(
    *,
    material_assignment_unit: str,
    include_policy_fallback: bool,
) -> set[str]:
    """Return statuses that the immediately following USD apply will consume.

    Independent Part-ID choices in the review confidence band already have an
    exact retrieved NVIDIA MDL and are deliberately passed to ``usd apply
    --include-review`` as provisional render candidates.  Automatic render QA
    remains responsible for accepting the resulting Look.  Palette-group
    review assignments do not have this direct, exact-cover authorization.
    """

    statuses = set(APPLICABLE_ASSIGNMENT_STATUSES)
    if material_assignment_unit == "part_id":
        statuses.add("review")
    if include_policy_fallback:
        statuses.add(POLICY_FALLBACK_STATUS)
    return statuses


def _palette_group_disagreement_contract_applies(
    material_assignment_unit: str,
) -> bool:
    """Return whether the legacy palette-group tournament contract is relevant.

    A Part-ID plan is selected and verified per CAD Part-ID.  Palette groups
    remain useful upstream as image evidence, but never author a Part-ID MDL
    decision.  Consequently a stale or unresolved palette-group
    forward/reverse disagreement cannot block sealing an otherwise
    render-verified Part-ID plan.
    """

    return material_assignment_unit != "part_id"


CommandRunner = Callable[..., None]
IsaacPythonResolver = Callable[[], Path]
ConfigLoader = Callable[[str | Path | None], VisualMaterialConfig]
ReferenceParser = Callable[[Sequence[str]], tuple[tuple[str, Path], ...]]
SealedDependencyVerifier = Callable[..., dict[str, Any]]


def _partial_live_resume_terminal_paths(destination: Path) -> tuple[Path, ...]:
    """Return artifacts proving that a visual-material run reached its final lock."""

    analysis_dir = destination / "analysis"
    candidates = [
        analysis_dir / "material_selection_lock.json",
        destination / "apply_visual_materials_locked_report.json",
        destination / "_SUCCESS",
        destination / "SUCCESS",
    ]
    candidates.extend(destination.glob("*_look_locked.usda"))
    return tuple(
        sorted(
            (path for path in candidates if path.exists() or path.is_symlink()),
            key=lambda path: str(path),
        )
    )


def _verified_locked_precollection_resume_available(destination: Path) -> bool:
    """Return whether a final lock may be replayed after collection was blocked.

    A selected-MDL lock is normally terminal.  There is one safe exception:
    the immutable Look was completed and hash-verified, but no delivery or
    success marker was ever written.  That state can occur when a downstream
    visual gate improves between runs.  Replaying it is safe only after the
    lock, locked USD, apply report, source USD, and final comparison all prove
    an internally consistent pre-collection state.
    """

    if not destination.is_dir():
        return False
    analysis_dir = destination / "analysis"
    delivery_markers = (
        destination / "delivery_validation.json",
        destination / "_SUCCESS",
        destination / "SUCCESS",
    )
    if any(path.exists() or path.is_symlink() for path in delivery_markers):
        return False

    lock_path = analysis_dir / "material_selection_lock.json"
    apply_report_path = destination / "apply_visual_materials_locked_report.json"
    locked_looks = tuple(sorted(destination.glob("*_look_locked.usda")))
    final_quality_path = (
        destination
        / "visual_exact_mdl_tournament"
        / "final_reference_render_comparison.json"
    )
    if (
        not lock_path.is_file()
        or not apply_report_path.is_file()
        or len(locked_looks) != 1
        or not locked_looks[0].is_file()
        or not final_quality_path.is_file()
    ):
        return False
    try:
        lock = read_object(lock_path, "pre-collection selected-MDL lock")
        apply_report = read_object(
            apply_report_path,
            "pre-collection locked apply report",
        )
        final_quality = read_object(
            final_quality_path,
            "pre-collection final visual comparison",
        )
    except (OSError, RuntimeError, ValueError):
        return False

    integrity = lock.get("integrity")
    if (
        lock.get("schema_version") != "qwen-selected-mdl-lock/v1"
        or not isinstance(integrity, Mapping)
        or integrity.get("lock_sha256")
        != canonical_sha256(
            {key: value for key, value in lock.items() if key != "integrity"}
        )
    ):
        return False
    locked_look = locked_looks[0].resolve()
    try:
        reported_output = (
            Path(str(apply_report["output_usd"])).expanduser().resolve(strict=True)
        )
        reported_source = (
            Path(str(apply_report["source_usd"])).expanduser().resolve(strict=True)
        )
    except (KeyError, OSError, RuntimeError):
        return False
    validation = apply_report.get("validation")
    required_validation_flags = (
        "mesh_geometry_and_topology_values_unchanged",
        "xforms_unchanged",
        "physics_properties_unchanged",
        "physics_bindings_unchanged",
        "visual_bindings_resolve",
        "mdl_sources_and_parameters_verified",
        "selected_mdl_lock_verified",
        "face_subsets_verified",
    )
    if (
        reported_output != locked_look
        or apply_report.get("output_sha256") != sha256_file(locked_look)
        or not reported_source.is_file()
        or apply_report.get("source_sha256") != sha256_file(reported_source)
        or not isinstance(validation, Mapping)
        or any(validation.get(flag) is not True for flag in required_validation_flags)
    ):
        return False
    aggregate = final_quality.get("aggregate")
    if not isinstance(aggregate, Mapping) or aggregate.get("status") not in {
        "PASS",
        "REVIEW",
        "FAIL",
        "INSUFFICIENT_EVIDENCE",
    }:
        return False
    return True


def _partial_live_resume_provisional_paths(
    destination: Path,
) -> tuple[Path, ...]:
    """Return regenerable post-inference artifacts from an interrupted run."""

    analysis_dir = destination / "analysis"
    tournament_dir = destination / "visual_exact_mdl_tournament"
    membership_tournament_dir = destination / "visual_membership_tournament"
    candidates = [
        analysis_dir / "visual_quality_resolution.json",
        analysis_dir / "dominant_assembly_membership_plan.json",
        analysis_dir / "dominant_assembly_membership_tournament.json",
        analysis_dir / "publish_quality_gate.json",
        # Delivery validation is produced before the independent final visual
        # acceptance stage.  A failure in that later stage leaves this file
        # behind, so it is a regenerable downstream artifact rather than proof
        # that the whole run completed.
        destination / "delivery_validation.json",
        destination / "final_visual_acceptance",
    ]
    # Exact-MDL candidate directories are expensive, independently hash-bound
    # render products.  Keep directory entries in place so the tournament can
    # validate and reuse each complete candidate atomically.  Root-level
    # selection/view-map files are provisional and must still be archived
    # before deterministic replay.  Symlinks are never treated as cache
    # directories.
    if tournament_dir.is_dir() and not tournament_dir.is_symlink():
        candidates.extend(
            child
            for child in tournament_dir.iterdir()
            if child.is_symlink() or not child.is_dir()
        )
    elif tournament_dir.exists() or tournament_dir.is_symlink():
        candidates.append(tournament_dir)
    if membership_tournament_dir.exists() or membership_tournament_dir.is_symlink():
        candidates.append(membership_tournament_dir)
    candidates.extend(analysis_dir.glob("exact_mdl_tournament*.json"))
    candidates.extend(analysis_dir.glob("visual_group_annotat*.json"))
    candidates.extend(analysis_dir.glob("appearance_optimization_*.json"))
    # Per-Part-ID projection, SAM3 refinement, retrieval and Qwen reranking are
    # downstream of the reusable Qwen/MVInverse checkpoint.  They may be
    # complete when a run fails just before the provisional Look is applied;
    # archive them atomically before a deterministic late-stage replay instead
    # of mixing their output directories with a new attempt.
    candidates.extend(analysis_dir.glob("part_id_*"))
    candidates.extend(destination.glob("apply_visual_materials*_report.json"))
    candidates.extend(destination.glob("*_look*.usda"))
    candidates.extend(destination.glob("*instance*plan*.json"))
    candidates.extend(destination.glob("visual_quality*"))
    candidates.append(destination / "material_identity_color")

    terminal_paths = set(_partial_live_resume_terminal_paths(destination))
    existing = {
        path
        for path in candidates
        if (path.exists() or path.is_symlink()) and path not in terminal_paths
    }
    # When a whole directory is archived, do not also try to move descendants
    # gathered by another glob.
    selected: list[Path] = []
    for path in sorted(existing, key=lambda item: (len(item.parts), str(item))):
        if any(path.is_relative_to(parent) for parent in selected):
            continue
        selected.append(path)
    return tuple(selected)


def _archive_partial_live_resume_downstream_artifacts(
    destination: Path,
) -> Path | None:
    """Reversibly archive provisional post-inference outputs before replay.

    Heavy Qwen/MVInverse evidence, the run-local material catalog, policy
    exact-cover products and quality-repair plans deliberately remain in
    place. Their existing readers decide whether each checkpoint is reusable.
    """

    terminal_paths = _partial_live_resume_terminal_paths(destination)
    locked_precollection_resume = bool(
        terminal_paths
    ) and _verified_locked_precollection_resume_available(destination)
    if terminal_paths and not locked_precollection_resume:
        raise RuntimeError(
            "Late-stage visual-material resume rejected because final locked "
            "artifacts already exist: "
            + ", ".join(str(path) for path in terminal_paths)
        )

    provisional_paths = tuple(
        dict.fromkeys(
            (
                *_partial_live_resume_provisional_paths(destination),
                *(terminal_paths if locked_precollection_resume else ()),
            )
        )
    )
    if not provisional_paths:
        return None

    archive_root = destination / "analysis" / "recovery_archive"
    archive_root.mkdir(parents=True, exist_ok=True)
    archive_dir = unique_path(archive_root / "late_stage_resume")
    archive_dir.mkdir(parents=False, exist_ok=False)
    manifest_path = archive_dir / "archive_manifest.json"
    manifest: dict[str, Any] = {
        "schema_version": "asset-pipeline-visual-material-recovery-archive/v1",
        "status": "IN_PROGRESS",
        "reason": (
            "late_stage_resume_after_locked_precollection_failure"
            if locked_precollection_resume
            else "late_stage_resume_after_completed_inference"
        ),
        "archived": [],
        "planned": [str(path.relative_to(destination)) for path in provisional_paths],
    }
    write_object(manifest_path, manifest)

    archived: list[dict[str, str]] = []
    try:
        for source_path in provisional_paths:
            relative_path = source_path.relative_to(destination)
            archived_path = archive_dir / relative_path
            archived_path.parent.mkdir(parents=True, exist_ok=True)
            source_path.rename(archived_path)
            archived.append(
                {
                    "original": str(relative_path),
                    "archived": str(archived_path.relative_to(archive_dir)),
                }
            )
            manifest["archived"] = list(archived)
            write_object(manifest_path, manifest)
    except OSError:
        # The in-progress manifest is intentionally retained. It records every
        # completed rename so recovery remains inspectable and reversible.
        raise

    manifest["status"] = "COMPLETED"
    write_object(manifest_path, manifest)
    return archive_dir


def _verified_partial_live_resume_available(
    destination: Path,
    references: Sequence[tuple[str, Path]],
    config: VisualMaterialConfig | None = None,
    foreground_annotations: Path | None = None,
) -> bool:
    """Return whether a failed live run has safe heavy-stage checkpoints.

    Existing visual-material directories remain fail-closed by default.  The
    only resumable state is an unfinished live run whose MVInverse ledger and
    face-region manifest are both present and whose hash-bound reference
    manifest still describes the exact current inputs.  A completed inference
    checkpoint is also resumable when all of its deterministic artifacts are
    present. Provisional authored Looks and QA products are allowed only for
    that completed-inference case and are archived before deterministic replay.
    A final material lock, locked Look/report, or delivery marker remains
    terminal.
    """

    if not destination.is_dir():
        return False

    analysis_dir = destination / "analysis"
    stored_foreground_annotations = analysis_dir / "sam3_foreground_annotations.json"
    reference_manifest_path = analysis_dir / "mvinverse_reference_manifest.json"
    ledger_path = analysis_dir / "mvinverse" / "mvinverse_inference_ledger.json"
    face_manifest_path = analysis_dir / "face_regions" / "manifest.json"
    qwen_ledger_path = analysis_dir / "qwen_inference_ledger.json"
    sam3_foreground_manifest_path = analysis_dir / "sam3_foreground" / "manifest.json"
    sam3_manifest_path = analysis_dir / "sam3_regions" / "manifest.json"
    visual_retrieval_path = analysis_dir / "visual_retrieval" / "visual_retrieval.json"
    if _partial_live_resume_terminal_paths(destination):
        return False

    # Camera registration is the first expensive live-evidence stage.  It is
    # resumable before MVInverse/face-region ledgers exist, provided that its
    # source render and human-confirmed reference manifest are still exactly
    # the same.  This keeps a transient Isaac startup failure from forcing a
    # fresh CAD/pose search.
    camera_search_report = (
        destination
        / "camera_calibration"
        / "search_pass"
        / "camera_calibration_report.json"
    )
    camera_final_report = (
        destination / "camera_calibration" / "camera_calibration_report.json"
    )
    camera_search_specs = (
        destination / "camera_calibration" / "search_pass" / "final_view_specs.json"
    )
    source_registry = destination / "source_part_registry.json"
    rendered_registry = destination / "renders" / "part_registry.rendered.json"
    if (
        (camera_final_report.is_file() or camera_search_report.is_file())
        and camera_search_specs.is_file()
        and source_registry.is_file()
        and rendered_registry.is_file()
        and foreground_annotations is not None
    ):
        try:
            camera_report_path = (
                camera_final_report
                if camera_final_report.is_file()
                else camera_search_report
            )
            camera_report = read_object(
                camera_report_path, "partial-run camera registration report"
            )
            expected_registry = str(rendered_registry.resolve(strict=True))
            expected_annotations = str(foreground_annotations.resolve(strict=True))
            report_registry_hash = camera_report.get("source_registry_sha256")
            report_annotations_hash = camera_report.get("reference_manifest_sha256")
            hashes_match = (
                isinstance(report_registry_hash, str)
                and report_registry_hash == sha256_file(rendered_registry)
                and isinstance(report_annotations_hash, str)
                and report_annotations_hash == sha256_file(foreground_annotations)
            )
            legacy_path_match = (
                report_registry_hash is None
                and report_annotations_hash is None
                and camera_report.get("source_registry") == expected_registry
                and camera_report.get("reference_manifest") == expected_annotations
                # No downstream material evidence may be resumed without its
                # own hashes.  This narrow compatibility branch is only for
                # a pre-ledger camera-search interruption.
                and not analysis_dir.exists()
            )
            if (
                camera_report.get("schema_version")
                == "qwen-whole-asset-camera-calibration/v9"
                and camera_report.get("source_registry") == expected_registry
                and camera_report.get("reference_manifest") == expected_annotations
                and (hashes_match or legacy_path_match)
            ):
                return True
        except (OSError, RuntimeError, ValueError):
            pass
    if not (
        reference_manifest_path.is_file()
        and ledger_path.is_file()
        and face_manifest_path.is_file()
    ):
        return False

    if foreground_annotations is None:
        if stored_foreground_annotations.exists():
            return False
    elif not stored_foreground_annotations.is_file() or sha256_file(
        stored_foreground_annotations
    ) != sha256_file(foreground_annotations):
        return False

    try:
        reference_manifest = read_object(
            reference_manifest_path, "partial-run MVInverse reference manifest"
        )
        ledger = read_object(ledger_path, "partial-run MVInverse ledger")
        face_manifest = read_object(
            face_manifest_path, "partial-run face-region manifest"
        )
    except (OSError, RuntimeError, ValueError):
        return False

    if config is not None:
        if not qwen_ledger_path.is_file():
            return False
        try:
            qwen_ledger = read_object(
                qwen_ledger_path, "partial-run Qwen inference ledger"
            )
        except (OSError, RuntimeError, ValueError):
            return False
        model_identity = qwen_ledger.get("model_identity")
        qwen_ledger_schema = qwen_ledger.get("schema_version")
        if (
            qwen_ledger_schema
            not in {
                "qwen-local-inference-ledger/v1",
                "qwen-local-inference-ledger/v2",
            }
            or qwen_ledger.get("requested_model_family") != config.qwen_model_family
            or qwen_ledger.get("requested_model_revision") != config.qwen_model_revision
            or not isinstance(model_identity, dict)
            or not isinstance(qwen_ledger.get("integrity"), dict)
        ):
            return False
        if config.qwen_model_family == "openai_compatible":
            if (
                model_identity.get("model_type") != "openai_compatible"
                or model_identity.get("model") != config.openai_model
                or model_identity.get("base_url") != config.openai_base_url
                or model_identity.get("api_key_env") != config.openai_api_key_env
                or model_identity.get("generation", {}).get("reasoning_effort")
                != config.openai_reasoning_effort
            ):
                return False
        else:
            recorded_model_path = model_identity.get("model_path")
            try:
                resolved_model_path = (
                    Path(recorded_model_path).expanduser().resolve(strict=True)
                    if isinstance(recorded_model_path, str)
                    else None
                )
            except (OSError, RuntimeError):
                return False
            if (
                config.qwen_model_path is None
                or resolved_model_path != config.qwen_model_path.resolve()
            ):
                return False
        unsigned_qwen = {
            key: value for key, value in qwen_ledger.items() if key != "integrity"
        }
        if qwen_ledger["integrity"].get("ledger_sha256") != canonical_sha256(
            unsigned_qwen
        ):
            return False
        if qwen_ledger_schema == "qwen-local-inference-ledger/v2":
            expected_palette_policy = {
                "initial_max_new_tokens": config.qwen_max_new_tokens,
                "max_new_tokens_ceiling": config.qwen_max_new_tokens_ceiling,
                "truncation_growth_factor": 2,
                "retry_condition": "token_limit_reached_without_eos",
                "minimum_usable_views": (config.qwen_minimum_usable_palette_views),
                "minimum_usable_view_ratio": (
                    config.qwen_minimum_usable_palette_view_ratio
                ),
            }
            if qwen_ledger.get("palette_generation_policy") != expected_palette_policy:
                return False
    source_views = reference_manifest.get("source_views")
    if not isinstance(source_views, list) or len(source_views) != len(references):
        return False
    for view, (reference_id, reference_path) in zip(
        source_views, references, strict=True
    ):
        if not isinstance(view, dict) or view.get("id") != reference_id:
            return False
        image = view.get("image")
        original_image = view.get("original_image", image)
        if not isinstance(image, str) or not isinstance(original_image, str):
            return False
        try:
            recorded_path = Path(image).expanduser().resolve(strict=True)
            recorded_original_path = (
                Path(original_image).expanduser().resolve(strict=True)
            )
        except (OSError, RuntimeError):
            return False
        if (
            recorded_original_path != reference_path.resolve()
            or not recorded_path.is_file()
        ):
            return False

    if (
        ledger.get("schema_version") != "qwen-mvinverse-inference-ledger/v1"
        or ledger.get("fail_closed") is not True
        or ledger.get("status") not in {"SUCCESS", "REUSED"}
    ):
        return False
    inputs = ledger.get("inputs")
    if not isinstance(inputs, dict):
        return False
    ledger_manifest = inputs.get("reference_manifest")
    ledger_views = inputs.get("source_views")
    if (
        not isinstance(ledger_manifest, dict)
        or ledger_manifest.get("sha256") != sha256_file(reference_manifest_path)
        or not isinstance(ledger_views, list)
        or len(ledger_views) != len(references)
    ):
        return False
    for view, manifest_view in zip(ledger_views, source_views, strict=True):
        model_image = Path(str(manifest_view["image"])).expanduser().resolve()
        if (
            not isinstance(view, dict)
            or view.get("view_id") != manifest_view["id"]
            or view.get("sha256") != sha256_file(model_image)
        ):
            return False

    parts = face_manifest.get("parts")
    part_count = face_manifest.get("part_count")
    if (
        face_manifest.get("schema_version") != "qwen-face-region-evidence/v1"
        or not isinstance(part_count, int)
        or part_count <= 0
        or not isinstance(parts, list)
        or len(parts) != part_count
    ):
        return False
    if not all(
        isinstance(part, dict)
        and isinstance(part.get("evidence"), str)
        and (face_manifest_path.parent / part["evidence"]).is_file()
        for part in parts
    ):
        return False

    unattended_result_path = analysis_dir / "unattended_result.json"
    if unattended_result_path.exists():
        completed_inference_paths = (
            analysis_dir / "qwen_mvinverse_recovery.json",
            analysis_dir / "staged_result.json",
            analysis_dir / "confidence_gate.json",
            analysis_dir / "autonomous_material_plan.json",
            analysis_dir / "group_materials.json",
            analysis_dir / "mvinverse_pbr_evidence.json",
            analysis_dir / "part_mapping_multiview_audit.json",
            analysis_dir / "spatial_mapping_report.json",
            analysis_dir / "spatial_mapping_audit.json",
        )
        if config is not None:
            completed_inference_paths += (qwen_ledger_path,)
        if config is not None:
            completed_inference_paths += (sam3_foreground_manifest_path,)
            if config.material_assignment_unit == "palette_group":
                completed_inference_paths += (
                    sam3_manifest_path,
                    visual_retrieval_path,
                )
        if not all(path.is_file() for path in completed_inference_paths):
            return False
        try:
            unattended = read_object(
                unattended_result_path, "completed-inference unattended result"
            )
        except (OSError, RuntimeError, ValueError):
            return False
        if unattended.get("state") not in {
            "READY_TO_APPLY",
            "COMPLETED_SAFE_NOOP",
        }:
            return False
    elif _partial_live_resume_provisional_paths(destination):
        # A partial heavy-stage checkpoint cannot legitimately have apply/QA
        # outputs. Treat that mixture as stale instead of guessing provenance.
        return False
    return True


def _bundled_project_apply_command(
    *,
    isaac: Path,
    source: Path,
    catalog: Path,
    registry: Path,
    material_plan: Path,
    look_usd: Path,
    material_root: Path,
    apply_report: Path,
    instance_root_count: int,
) -> list[str]:
    """Build the USD apply command for the sealed source representation."""

    if instance_root_count:
        usd_command = "apply-instances"
        source_argument = "--source-usd"
    else:
        usd_command = "apply"
        source_argument = "--asset-usd"
    return [
        str(isaac),
        "-m",
        "qwen_material_pipeline",
        "usd",
        usd_command,
        source_argument,
        str(source),
        "--catalog",
        str(catalog),
        "--registry",
        str(registry),
        "--plan",
        str(material_plan),
        "--output",
        str(look_usd),
        "--material-root",
        str(material_root),
        "--report",
        str(apply_report),
    ]


def _bundled_project_inference_provenance(
    requested_inference_mode: str,
) -> dict[str, str]:
    if requested_inference_mode not in {"auto", "bundled"}:
        raise RuntimeError(
            "Bundled project assignment requires requested mode auto or bundled"
        )
    return {
        "requested_inference_mode": requested_inference_mode,
        "inference_mode": "bundled_project",
    }


def _run_bundled_project_assignment(
    *,
    project: BundledMaterialProject,
    source: Path,
    source_cad: Path,
    parsed_references: Sequence[tuple[str, Path]],
    destination: Path,
    source_registry: Path,
    registry: Path,
    rendered_registry: Path,
    editable_usd: Path | None,
    expand_report: Path | None,
    instance_root_count: int,
    requested_inference_mode: str,
    effective_config_path: str,
    config: VisualMaterialConfig,
    isaac: Path,
    log_cb: LogCallback,
    command_runner: CommandRunner,
) -> dict[str, Any]:
    """Replay one exact, audited project result after identity/topology checks."""

    inference_provenance = _bundled_project_inference_provenance(
        requested_inference_mode
    )
    analysis_dir = destination / "analysis" / f"project_{project.asset_id}"
    material_plan = analysis_dir / "complete_material_plan.json"
    project_audit = analysis_dir / "project_material_audit.json"
    unattended_result = analysis_dir / "unattended_result.json"
    sealed_evidence = analysis_dir / "sealed_qwen_mvinverse_evidence.json"
    dependency_verification_report = (
        analysis_dir / "sealed_dependency_verification.json"
    )
    look_usd = destination / f"{source.stem}_look.usda"
    apply_report = destination / "apply_visual_materials_report.json"
    preview_dir = destination / "preview_final"
    preview_registry = preview_dir / "part_registry.json"
    preview_rendered_registry = preview_dir / "part_registry.rendered.json"

    planner_command = [
        str(config.qwen_python),
        "-m",
        project.planner_module,
        "--project",
        str(project.project_file),
        "--source-cad",
        str(source_cad),
        "--source-usd",
        str(source),
        "--registry",
        str(rendered_registry),
    ]
    for reference_id, path in parsed_references:
        planner_command.extend(["--reference", f"{reference_id}={path}"])
    planner_command.extend(
        [
            "--output-plan",
            str(material_plan),
            "--audit",
            str(project_audit),
            "--unattended-result",
            str(unattended_result),
        ]
    )
    _run_stage(
        f"bundled_project_plan:{project.asset_id}",
        planner_command,
        log_cb,
        command_runner=command_runner,
    )
    for required in (material_plan, project_audit, unattended_result):
        _require_file(required, f"bundled_project_plan:{project.asset_id}")

    plan_document = read_object(material_plan, "bundled project material plan")
    audit_document = read_object(project_audit, "bundled project material audit")
    unattended_document = read_object(
        unattended_result, "bundled project unattended result"
    )
    registry_document = read_object(rendered_registry, "rendered occurrence registry")
    raw_parts = registry_document.get("parts")
    assignments = plan_document.get("assignments")
    plan_provenance = plan_document.get("provenance")
    if (
        audit_document.get("status") != "PASS"
        or unattended_document.get("state") != "READY_TO_APPLY"
        or not isinstance(raw_parts, list)
        or not isinstance(assignments, list)
        or not isinstance(plan_provenance, dict)
        or plan_provenance.get("template_sha256") != project.document["template_sha256"]
        or audit_document.get("plan_sha256") != canonical_sha256(plan_document)
        or unattended_document.get("audit_sha256") != canonical_sha256(audit_document)
    ):
        raise RuntimeError(
            f"Bundled project {project.asset_id!r} did not pass its replay audit"
        )
    registry_part_ids = {
        item.get("part_id") for item in raw_parts if isinstance(item, dict)
    }
    assignment_part_ids = {
        item.get("part_id") for item in assignments if isinstance(item, dict)
    }
    if (
        len(registry_part_ids) != len(raw_parts)
        or len(assignment_part_ids) != len(assignments)
        or assignment_part_ids != registry_part_ids
        or any(
            not isinstance(item, dict)
            or item.get("status") not in APPLICABLE_ASSIGNMENT_STATUSES
            for item in assignments
        )
    ):
        raise RuntimeError(
            f"Bundled project {project.asset_id!r} is not an exact-cover plan"
        )

    if (
        sha256_file(project.template) != project.document["template_sha256"]
        or sha256_file(project.catalog) != project.document["catalog_sha256"]
    ):
        raise RuntimeError(
            f"Bundled project {project.asset_id!r} template/catalog changed "
            "between matching and application"
        )
    dependency_verification = verify_sealed_dependency_lock(
        lock_path=project.dependency_lock,
        expected_lock_sha256=project.document["dependency_lock_sha256"],
        catalog_path=project.catalog,
        material_root=project.material_root,
        isaac_root=isaac.parent,
        expected_asset_id=project.asset_id,
    )
    if (
        dependency_verification["lock_sha256"]
        != project.dependency_lock_verification["lock_sha256"]
    ):
        raise RuntimeError(
            f"Bundled project {project.asset_id!r} dependency lock changed "
            "between matching and application"
        )
    write_object(dependency_verification_report, dependency_verification)
    dependency_verification_report_sha256 = sha256_file(dependency_verification_report)

    evidence_document = {
        "schema_version": "qwen-bundled-project-evidence/v1",
        "asset_id": project.asset_id,
        "method": project.document["evidence"]["method"],
        "historical_result_sha256": project.document["evidence"][
            "historical_result_sha256"
        ],
        "project": str(project.project_file),
        "project_sha256": sha256_file(project.project_file),
        "source_cad_sha256": sha256_file(source_cad),
        "reference_sha256": {
            reference_id: sha256_file(path) for reference_id, path in parsed_references
        },
        "template": str(project.template),
        "template_sha256": sha256_file(project.template),
        "catalog": str(project.catalog),
        "catalog_sha256": sha256_file(project.catalog),
        "dependency_lock_verified": True,
        "dependency_lock": str(project.dependency_lock),
        "dependency_lock_sha256": dependency_verification["lock_sha256"],
        "dependency_lock_verification": dependency_verification,
        "dependency_lock_verification_status": dependency_verification["status"],
        "dependency_lock_verification_report": str(
            dependency_verification_report.resolve(strict=True)
        ),
        "dependency_lock_verification_report_sha256": (
            dependency_verification_report_sha256
        ),
        "plan_sha256": canonical_sha256(plan_document),
        "audit_sha256": canonical_sha256(audit_document),
        "source_representation_id": project.source_representation_id,
        "source_registry_topology_role": (project.source_registry_topology_role),
        "live_inference_repeated": False,
        "historical_parameter_policy": dependency_verification[
            "historical_parameter_policy"
        ],
        "replay_policy": (
            "The accepted Qwen/MVInverse result is replayed only after exact "
            "CAD, photograph, occurrence-path, topology, and face-subset checks."
        ),
    }
    write_object(sealed_evidence, evidence_document)

    _run_stage(
        f"bundled_project_apply:{project.asset_id}",
        _bundled_project_apply_command(
            isaac=isaac,
            source=source,
            catalog=project.catalog,
            registry=rendered_registry,
            material_plan=material_plan,
            look_usd=look_usd,
            material_root=project.material_root,
            apply_report=apply_report,
            instance_root_count=instance_root_count,
        ),
        log_cb,
        command_runner=command_runner,
        retry_native_crash=True,
    )
    _require_file(look_usd, f"bundled_project_apply:{project.asset_id}")
    _require_file(apply_report, f"bundled_project_apply:{project.asset_id}")
    applied_document = read_object(apply_report, "bundled project apply report")
    expected_count = len(raw_parts)
    expected_assembly = project.document["expected_assembly"]
    if (
        applied_document.get("applied_count") != expected_count
        or applied_document.get("mesh_occurrence_count")
        != expected_assembly["mesh_occurrences"]
        or applied_document.get("point_occurrence_count")
        != expected_assembly["point_occurrence_count"]
        or applied_document.get("face_occurrence_count")
        != expected_assembly["face_occurrence_count"]
        or applied_document.get("face_subset_count")
        != expected_assembly["subset_count"]
        or applied_document.get("covered_face_occurrence_count")
        != expected_assembly["face_occurrence_count"]
    ):
        raise RuntimeError(
            f"Bundled project {project.asset_id!r} material application is incomplete"
        )
    _run_stage(
        f"bundled_project_preview_registry:{project.asset_id}",
        [
            str(isaac),
            "-m",
            "qwen_material_pipeline",
            "usd",
            "registry",
            "--usd",
            str(look_usd),
            "--output",
            str(preview_registry),
        ],
        log_cb,
        command_runner=command_runner,
        retry_native_crash=True,
    )
    _require_file(preview_registry, "bundled project preview registry")
    render_contract = project.acceptance["render"]
    _run_stage(
        f"bundled_project_preview_render:{project.asset_id}",
        [
            str(isaac),
            "-m",
            "qwen_material_pipeline",
            "usd",
            "render",
            "--registry",
            str(preview_registry),
            "--output-dir",
            str(preview_dir),
            "--resolution",
            str(render_contract["resolution"]),
            "--views",
            str(render_contract["views"]),
            "--rt-subframes",
            str(render_contract["rt_subframes"]),
            "--lighting-profile",
            str(render_contract["lighting_profile"]),
            "--analysis-up-axis",
            str(render_contract["analysis_up_axis"]),
            f"--analysis-front-axis={render_contract['analysis_front_axis']}",
            "--rgb-only",
        ],
        log_cb,
        command_runner=command_runner,
        retry_native_crash=True,
    )
    _require_file(preview_rendered_registry, "bundled project preview render")

    log_message(
        log_cb,
        f"Restored bundled material project {project.asset_id!r}: "
        f"{expected_count} Mesh assignments applied and verified.",
    )
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "state": "APPLIED",
        "source_usd": str(source),
        "source_usd_sha256": sha256_file(source),
        "source_cad": str(source_cad),
        "source_cad_sha256": sha256_file(source_cad),
        "effective_usd": str(look_usd.resolve(strict=True)),
        "effective_usd_sha256": sha256_file(look_usd),
        "output_dir": str(destination),
        "references": [
            {
                "id": reference_id,
                "image": str(path),
                "sha256": sha256_file(path),
            }
            for reference_id, path in parsed_references
        ],
        "config": str(Path(effective_config_path).expanduser().resolve()),
        **inference_provenance,
        "material_root": str(project.material_root),
        "material_catalog": str(project.catalog),
        "material_allowlist": None,
        "material_catalog_count": None,
        "instance_root_count": instance_root_count,
        "instance_aware": bool(instance_root_count),
        "material_project": project.asset_id,
        "material_project_manifest": str(project.project_file),
        "material_project_manifest_sha256": sha256_file(project.project_file),
        "material_project_acceptance": copy.deepcopy(project.acceptance),
        "material_project_acceptance_sha256": canonical_sha256(project.acceptance),
        "material_project_acceptance_evidence": copy.deepcopy(
            project.acceptance_evidence
        ),
        "material_project_acceptance_evidence_sha256": canonical_sha256(
            project.acceptance_evidence
        ),
        "dependency_lock_verified": True,
        "dependency_lock": str(project.dependency_lock),
        "dependency_lock_sha256": dependency_verification["lock_sha256"],
        "dependency_lock_verification_status": dependency_verification["status"],
        "dependency_lock_verification_report": str(
            dependency_verification_report.resolve(strict=True)
        ),
        "dependency_lock_verification_report_sha256": (
            dependency_verification_report_sha256
        ),
        "source_representation_id": project.source_representation_id,
        "source_registry_topology_role": (project.source_registry_topology_role),
        "complete_coverage_required": True,
        "source_registry": str(source_registry.resolve(strict=True)),
        "registry": str(registry.resolve(strict=True)),
        "rendered_registry": str(rendered_registry.resolve(strict=True)),
        "unattended_result": str(unattended_result.resolve(strict=True)),
        "staged_state": "READY_TO_APPLY",
        "staged_material_plan": str(material_plan.resolve(strict=True)),
        "material_plan": str(material_plan.resolve(strict=True)),
        "instance_material_plan": str(material_plan.resolve(strict=True)),
        "project_material_audit": str(project_audit.resolve(strict=True)),
        "sealed_qwen_mvinverse_evidence": str(sealed_evidence.resolve(strict=True)),
        "catalog": str(project.catalog),
        "editable_usd": (
            str(editable_usd.resolve(strict=True)) if editable_usd is not None else None
        ),
        "expand_report": (
            str(expand_report.resolve(strict=True))
            if expand_report is not None
            else None
        ),
        "mvinverse_ledger": None,
        "apply_report": str(apply_report.resolve(strict=True)),
        "assignment_count": expected_count,
        "applied_count": applied_document["applied_count"],
        "face_subset_count": applied_document["face_subset_count"],
        "visual_quality_status": "RESTORED_HISTORICAL_BASELINE",
        "visual_quality_rendered_registry": str(
            preview_rendered_registry.resolve(strict=True)
        ),
        "preview_dir": str(preview_dir.resolve(strict=True)),
        "preview_rendered_registry": str(
            preview_rendered_registry.resolve(strict=True)
        ),
    }


@dataclass(frozen=True)
class PolicyPartIdStageResult:
    effective_catalog: Path
    effective_whitelist: Path
    live_material_count: int | None
    state: Any
    use_policy_fallback: bool
    effective_material_plan: Path
    policy_fallback_count: int
    policy_plan_document: dict[str, Any] | None
    policy_audit_document: dict[str, Any] | None
    plan: dict[str, Any]
    assignments: list[dict[str, Any]]
    selection_lock_document: dict[str, Any] | None


def _validate_catalog_family_first_result(
    *,
    qwen_document: Mapping[str, Any],
    choices: Mapping[str, Any],
    confidences: Mapping[str, float],
    catalog_document: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    raw_predictions = qwen_document.get("material_predictions")
    if not isinstance(raw_predictions, list):
        raise RuntimeError(
            "Part-ID Qwen result contains no material-identity predictions"
        )
    predictions: dict[str, dict[str, Any]] = {}
    for raw_prediction in raw_predictions:
        part_id = (
            raw_prediction.get("part_id")
            if isinstance(raw_prediction, Mapping)
            else None
        )
        if not isinstance(part_id, str) or part_id in predictions:
            raise RuntimeError(
                "Part-ID Qwen result has invalid material-identity predictions"
            )
        predictions[part_id] = copy.deepcopy(dict(raw_prediction))
    if set(predictions) != set(choices) or set(confidences) != set(choices):
        raise RuntimeError(
            "Part-ID Qwen material predictions, choices and confidences "
            "do not have the same exact cover"
        )
    raw_catalog_materials = catalog_document.get("materials")
    if not isinstance(raw_catalog_materials, list):
        raise RuntimeError("NVIDIA MDL catalog has no materials")
    catalog_by_id = {
        str(row["material_id"]): row
        for row in raw_catalog_materials
        if isinstance(row, Mapping) and isinstance(row.get("material_id"), str)
    }

    def identity_semantics(
        material_id: str,
    ) -> tuple[set[str], str, str, str]:
        record = catalog_by_id[material_id]
        semantics = record.get("surface_semantics")
        substrates = (
            semantics.get("compatible_substrates")
            if isinstance(semantics, Mapping)
            else None
        )
        treatment = (
            semantics.get("surface_treatment")
            if isinstance(semantics, Mapping)
            else None
        )
        optical = (
            semantics.get("optical_behavior")
            if isinstance(semantics, Mapping)
            else None
        )
        semantic_confidence = (
            semantics.get("confidence") if isinstance(semantics, Mapping) else None
        )
        if (
            not isinstance(substrates, list)
            or not substrates
            or not all(isinstance(value, str) for value in substrates)
            or not isinstance(treatment, str)
            or not isinstance(optical, str)
            or not isinstance(semantic_confidence, str)
        ):
            raise RuntimeError(
                f"NVIDIA MDL {material_id} has invalid physical identity semantics"
            )
        return (
            {value.casefold() for value in substrates},
            treatment.casefold(),
            optical.casefold(),
            semantic_confidence.casefold(),
        )

    for part_id, material_id in choices.items():
        if not isinstance(material_id, str):
            raise RuntimeError(f"Part-ID {part_id} has an invalid MDL choice")
        prediction = predictions[part_id]
        selected_catalog_row = catalog_by_id.get(material_id)
        if not isinstance(selected_catalog_row, Mapping):
            raise RuntimeError(
                f"Part-ID {part_id} selected a material outside the catalog"
            )
        if prediction.get("status") == "APPLYABLE":
            substrate = prediction.get("physical_substrate")
            species = prediction.get("material_species")
            species_confidence = prediction.get("species_confidence")
            treatment = prediction.get("surface_treatment")
            optical = prediction.get("optical_behavior")
            identity_resolution = prediction.get("identity_resolution")
            if (
                not isinstance(substrate, str)
                or not isinstance(species, str)
                or isinstance(species_confidence, bool)
                or not isinstance(species_confidence, (int, float))
                or not isinstance(treatment, str)
                or not isinstance(optical, str)
                or identity_resolution
                not in {"exact_material", "corresponding_material"}
            ):
                raise RuntimeError(
                    f"Part-ID {part_id} has an incomplete physical identity prediction"
                )
            substrate = substrate.casefold()
            species = species.casefold()
            treatment = treatment.casefold()
            optical = optical.casefold()
            (
                selected_substrates,
                selected_treatment,
                selected_optical,
                selected_confidence,
            ) = identity_semantics(material_id)
            selected_species = _catalog_material_species(
                material_id,
                selected_catalog_row,
            )
            exact_treatment_exists = any(
                substrate in candidate_substrates
                and candidate_treatment == treatment
                and candidate_optical == optical
                and candidate_confidence != "low"
                for candidate_material_id in catalog_by_id
                for (
                    candidate_substrates,
                    candidate_treatment,
                    candidate_optical,
                    candidate_confidence,
                ) in [identity_semantics(candidate_material_id)]
            )
            if (
                substrate not in selected_substrates
                or selected_optical != optical
                or selected_confidence == "low"
                or (
                    species != "unknown"
                    and float(species_confidence) >= MINIMUM_MATERIAL_SPECIES_CONFIDENCE
                    and selected_species != species
                )
                or (
                    identity_resolution == "exact_material"
                    and exact_treatment_exists
                    and selected_treatment != treatment
                )
            ):
                raise RuntimeError(
                    f"Part-ID {part_id} selected an MDL outside its predicted "
                    "physical material identity"
                )
        elif prediction.get("status") == "INSUFFICIENT_EVIDENCE":
            if confidences[part_id] != 0.0:
                raise RuntimeError(
                    f"Part-ID {part_id} has insufficient material identity "
                    "evidence but a nonzero assignment confidence"
                )
        else:
            raise RuntimeError(
                f"Part-ID {part_id} has an invalid material-prediction status"
            )
    component_groups: dict[str, list[str]] = {}
    for part_id, prediction in predictions.items():
        component_id = prediction.get("component_id")
        if component_id is None:
            continue
        if not isinstance(component_id, str) or not component_id:
            raise RuntimeError("Part-ID prediction has an invalid component identity")
        component_groups.setdefault(component_id, []).append(part_id)
    for component_id, member_ids in component_groups.items():
        expected_members = sorted(member_ids)
        contracts = {
            (
                predictions[part_id].get("physical_substrate"),
                predictions[part_id].get("material_species"),
                predictions[part_id].get("surface_treatment"),
                predictions[part_id].get("optical_behavior"),
                predictions[part_id].get("surface_finish"),
            )
            for part_id in member_ids
        }
        declared_members = {
            tuple(predictions[part_id].get("component_member_part_ids", []))
            for part_id in member_ids
        }
        if (
            len(contracts) != 1
            or declared_members != {tuple(expected_members)}
            or len({choices[part_id] for part_id in member_ids}) != 1
        ):
            raise RuntimeError(
                f"appearance component {component_id} does not share one exact "
                "physical material identity and MDL"
            )
    refinement = qwen_document.get("component_membership_refinement")
    refined_groups: dict[str, list[str]] = {}
    if isinstance(refinement, Mapping) and refinement.get("status") == "COMPLETED":
        raw_components = refinement.get("components")
        if not isinstance(raw_components, list):
            raise RuntimeError(
                "Part-ID Qwen component refinement has no component records"
            )
        for raw_component in raw_components:
            component_id = (
                raw_component.get("component_id")
                if isinstance(raw_component, Mapping)
                else None
            )
            member_ids = (
                raw_component.get("refined_member_part_ids")
                if isinstance(raw_component, Mapping)
                else None
            )
            if (
                not isinstance(component_id, str)
                or component_id in refined_groups
                or not isinstance(member_ids, list)
                or len(member_ids) < 2
                or not all(isinstance(part_id, str) for part_id in member_ids)
                or len(set(member_ids)) != len(member_ids)
            ):
                raise RuntimeError(
                    "Part-ID Qwen component refinement has an invalid identity scope"
                )
            refined_groups[component_id] = sorted(member_ids)
    repeated = qwen_document.get("repeated_assembly_role_consistency")
    if isinstance(repeated, Mapping) and repeated.get("status") == "COMPLETED":
        raw_scopes = repeated.get("final_component_scopes")
        raw_structures = repeated.get("structures")
        if not isinstance(raw_scopes, list) or not isinstance(raw_structures, list):
            raise RuntimeError("Part-ID Qwen repeated-assembly audit is incomplete")
        final_scopes: dict[str, tuple[str, list[str]]] = {}
        for raw_scope in raw_scopes:
            component_id = (
                raw_scope.get("component_id")
                if isinstance(raw_scope, Mapping)
                else None
            )
            prediction_scope = (
                raw_scope.get("prediction_scope")
                if isinstance(raw_scope, Mapping)
                else None
            )
            member_ids = (
                raw_scope.get("member_part_ids")
                if isinstance(raw_scope, Mapping)
                else None
            )
            if (
                not isinstance(component_id, str)
                or component_id in final_scopes
                or prediction_scope
                not in {"appearance_component", "repeated_assembly_role"}
                or not isinstance(member_ids, list)
                or len(member_ids) < 2
                or not all(isinstance(part_id, str) for part_id in member_ids)
                or len(set(member_ids)) != len(member_ids)
            ):
                raise RuntimeError(
                    "Part-ID Qwen repeated-assembly audit has an invalid final scope"
                )
            final_scopes[component_id] = (
                str(prediction_scope),
                sorted(member_ids),
            )
        if {
            component_id: members
            for component_id, (_scope, members) in final_scopes.items()
        } != {
            component_id: sorted(member_ids)
            for component_id, member_ids in component_groups.items()
        }:
            raise RuntimeError(
                "Part-ID Qwen repeated-assembly and prediction scopes differ"
            )
        for component_id, (prediction_scope, member_ids) in final_scopes.items():
            if any(
                predictions[part_id].get("prediction_scope") != prediction_scope
                for part_id in member_ids
            ):
                raise RuntimeError(
                    "Part-ID Qwen repeated-assembly prediction kind is inconsistent"
                )
        role_additions: dict[str, set[str]] = {}
        created_role_members: dict[str, set[str]] = {}
        allowed_role_statuses = {
            "EXTENDED_EXISTING_COMPONENT",
            "CREATED_ROLE_COMPONENT",
            "ALREADY_CONSTRAINED",
            "PHYSICAL_SURFACE_CONFLICT",
            "GEOMETRY_IDENTITY_CONFLICT",
            "AMBIGUOUS_EXISTING_COMPONENTS",
        }
        for raw_structure in raw_structures:
            raw_roles = (
                raw_structure.get("roles")
                if isinstance(raw_structure, Mapping)
                else None
            )
            if not isinstance(raw_roles, list):
                raise RuntimeError(
                    "Part-ID Qwen repeated-assembly structure has no roles"
                )
            for raw_role in raw_roles:
                status = (
                    raw_role.get("status") if isinstance(raw_role, Mapping) else None
                )
                member_ids = (
                    raw_role.get("observed_member_part_ids")
                    if isinstance(raw_role, Mapping)
                    else None
                )
                if (
                    status not in allowed_role_statuses
                    or not isinstance(member_ids, list)
                    or len(member_ids) < 2
                    or not all(isinstance(part_id, str) for part_id in member_ids)
                    or len(set(member_ids)) != len(member_ids)
                ):
                    raise RuntimeError(
                        "Part-ID Qwen repeated-assembly role record is invalid"
                    )
                if status not in {
                    "EXTENDED_EXISTING_COMPONENT",
                    "CREATED_ROLE_COMPONENT",
                    "ALREADY_CONSTRAINED",
                }:
                    continue
                component_id = raw_role.get("component_id")
                added_ids = raw_role.get("added_member_part_ids")
                if (
                    not isinstance(component_id, str)
                    or component_id not in final_scopes
                    or not isinstance(added_ids, list)
                    or not all(isinstance(part_id, str) for part_id in added_ids)
                    or len(set(added_ids)) != len(added_ids)
                    or not set(member_ids) <= set(final_scopes[component_id][1])
                    or not set(added_ids) <= set(member_ids)
                ):
                    raise RuntimeError(
                        "Part-ID Qwen repeated-assembly role binding is invalid"
                    )
                role_additions.setdefault(component_id, set()).update(added_ids)
                if status == "CREATED_ROLE_COMPONENT":
                    created_role_members[component_id] = set(member_ids)
        expected_final_groups = {
            component_id: set(member_ids)
            for component_id, member_ids in refined_groups.items()
        }
        for component_id, additions in role_additions.items():
            expected_final_groups.setdefault(component_id, set()).update(additions)
        for component_id, members in created_role_members.items():
            if component_id in refined_groups:
                raise RuntimeError(
                    "Part-ID Qwen repeated role collides with an appearance component"
                )
            expected_final_groups[component_id] = set(members)
        if {
            component_id: sorted(members)
            for component_id, members in expected_final_groups.items()
        } != {
            component_id: members
            for component_id, (_scope, members) in final_scopes.items()
        }:
            raise RuntimeError(
                "Part-ID Qwen repeated-role additions do not reproduce final scopes"
            )
    elif refined_groups != {
        component_id: sorted(member_ids)
        for component_id, member_ids in component_groups.items()
    }:
        raise RuntimeError(
            "Part-ID Qwen component refinement and prediction scopes differ"
        )
    return predictions


def _run_policy_part_id_stage(
    context: VisualMaterialPipelineContext,
    *,
    prepared_source: SourcePreparationResult,
    allow_policy_material_fallback: bool,
    log_cb: LogCallback,
    command_runner: CommandRunner,
) -> PolicyPartIdStageResult:
    """Produce the exact, independently audited material plan to author."""

    config = context.config
    destination = context.destination
    workspace = context.workspace
    rendered_registry = prepared_source.rendered_registry
    source_registry = prepared_source.source_registry
    inference_paths = workspace.inference
    part_id_paths = workspace.part_id
    appearance_paths = workspace.appearance
    legacy_paths = workspace.legacy
    analysis_dir = inference_paths.root
    staged_result = inference_paths.staged_result
    confidence_gate = inference_paths.confidence_gate
    staged_material_plan = inference_paths.staged_material_plan
    group_materials = inference_paths.group_materials
    mvinverse_pbr_evidence = inference_paths.mvinverse_pbr_evidence
    policy_input = inference_paths.policy_input
    policy_plan = inference_paths.policy_plan
    policy_audit = inference_paths.policy_audit
    mvinverse_ledger = inference_paths.mvinverse_ledger
    completed_inference_resume = (
        context.partial_live_resume and inference_paths.unattended_result.is_file()
    )
    camera_calibration_dir = workspace.source.camera_dir
    quality_repair_plan = legacy_paths.quality_repair_plan
    quality_repair_audit = legacy_paths.quality_repair_audit
    part_id_evidence_dir = part_id_paths.evidence_dir
    part_id_evidence_path = part_id_paths.evidence
    part_id_coarse_evidence_dir = part_id_paths.coarse_evidence_dir
    part_id_coarse_evidence_path = part_id_paths.coarse_evidence
    part_id_sam3_request = part_id_paths.sam3_request
    part_id_sam3_dir = part_id_paths.sam3_dir
    part_id_sam3_manifest = part_id_paths.sam3_manifest
    part_id_retrieval_request = part_id_paths.retrieval_request
    part_id_retrieval_dir = part_id_paths.retrieval_dir
    part_id_retrieval_result = part_id_paths.retrieval_result
    part_id_qwen_dir = part_id_paths.qwen_dir
    part_id_qwen_result = part_id_paths.qwen_result
    part_id_material_plan = part_id_paths.material_plan
    part_id_material_audit = part_id_paths.material_audit
    appearance_components_report = appearance_paths.components
    appearance_component_input_dir = appearance_paths.input_dir
    appearance_component_evidence = appearance_paths.evidence
    appearance_component_material_memberships = appearance_paths.memberships
    appearance_component_retrieval_request = appearance_paths.retrieval_request
    appearance_component_retrieval_dir = appearance_paths.retrieval_dir
    appearance_component_retrieval_result = appearance_paths.retrieval_result
    appearance_component_qwen_dir = appearance_paths.qwen_dir
    appearance_component_qwen_result = appearance_paths.qwen_result
    appearance_component_mdl_selection_audit = appearance_paths.mdl_selection_audit
    _command_runner = command_runner
    policy_plan_document: dict[str, Any] | None = None
    policy_audit_document: dict[str, Any] | None = None

    inference = run_material_inference(
        context,
        rendered_registry=rendered_registry,
        log_cb=log_cb,
        command_runner=_command_runner,
    )
    effective_catalog = inference.catalog
    effective_whitelist = inference.whitelist
    live_material_count = inference.material_count
    unattended = inference.unattended
    state = unattended.get("state")
    use_policy_fallback = bool(allow_policy_material_fallback)
    if use_policy_fallback and state not in {
        "READY_TO_APPLY",
        "COMPLETED_SAFE_NOOP",
    }:
        raise RuntimeError(
            "Automatic visual material inference did not produce a usable staged "
            f"result for policy exact-cover (state={state!r}); physics was not started"
        )
    if not use_policy_fallback and state != "READY_TO_APPLY":
        raise RuntimeError(
            "Automatic visual material inference produced no safely applicable "
            f"assignments (state={state!r}); physics was not started"
        )

    effective_material_plan = staged_material_plan
    policy_fallback_count = 0
    if use_policy_fallback:
        for required_path in (
            staged_result,
            confidence_gate,
            group_materials,
            mvinverse_pbr_evidence,
            analysis_dir / "palette_fusion.json",
        ):
            _require_file(required_path, "qwen_mvinverse")
        # STEP exports frequently carry vivid assembly/display colours whose
        # purpose is part separation, not physical appearance.  A workflow
        # with real reference photos therefore treats those bindings as
        # unverified and starts unresolved parts from a neutral material.
        # Strong Qwen/spatial evidence and the later QA-repair pass remain
        # free to replace that neutral baseline automatically.
        policy_input_document: dict[str, Any] = {
            "schema_version": POLICY_INPUT_SCHEMA_VERSION,
            "source_visual_strategy": "neutralize_unverified",
        }
        if config.material_selection_objective == MATERIAL_SELECTION_OBJECTIVE_VISUAL:
            # Part names such as BOLT, NUT or PLASTIC are physical-category
            # hints, not reference-photo appearance evidence.  In a
            # visual-similarity workflow they must not turn hundreds of
            # unresolved CAD parts dark before render QA has localized them.
            # Keep unresolved entities on the neutral stainless baseline;
            # reference-bound groups and the exact-MDL tournament remain free
            # to replace that baseline.
            policy_input_document["semantic_rules"] = []
        write_object(policy_input, policy_input_document)
        policy_command = policy_exact_cover_command(
            python=config.qwen_python,
            registry=rendered_registry,
            staged_result=staged_result,
            confidence_gate=confidence_gate,
            whitelist=effective_whitelist,
            base_plan=staged_material_plan,
            group_materials=group_materials,
            mvinverse_pbr_evidence=mvinverse_pbr_evidence,
            palette_fusion=analysis_dir / "palette_fusion.json",
            policy=policy_input,
            output_plan=policy_plan,
            audit=policy_audit,
            immutable_mdl_after_selection=(config.immutable_mdl_after_selection),
        )
        policy_checkpoint_resume = (
            completed_inference_resume
            and policy_plan.is_file()
            and policy_audit.is_file()
        )
        stale_policy_reason: str | None = None
        if policy_checkpoint_resume:
            try:
                _validate_policy_exact_cover_bundle(
                    plan=read_object(
                        policy_plan,
                        "existing policy exact-cover plan",
                    ),
                    audit=read_object(
                        policy_audit,
                        "existing policy exact-cover audit",
                    ),
                    registry=read_object(rendered_registry, "rendered registry"),
                    staged_result=read_object(staged_result, "staged result"),
                    confidence_gate=read_object(confidence_gate, "confidence gate"),
                    base_plan=read_object(
                        staged_material_plan,
                        "autonomous material plan",
                    ),
                    group_materials=read_object(
                        group_materials,
                        "group materials",
                    ),
                    mvinverse_pbr_evidence=read_object(
                        mvinverse_pbr_evidence,
                        "MVInverse PBR evidence",
                    ),
                    whitelist=read_object(
                        effective_whitelist,
                        "material whitelist",
                    ),
                    palette_fusion=read_object(
                        analysis_dir / "palette_fusion.json",
                        "palette fusion",
                    ),
                    expected_source_visual_strategy="neutralize_unverified",
                    expected_policy_overrides=policy_input_document,
                    expected_immutable_mdl_after_selection=(
                        config.immutable_mdl_after_selection
                    ),
                )
            except (OSError, RuntimeError, ValueError) as exc:
                stale_policy_reason = str(exc)
        if policy_checkpoint_resume and stale_policy_reason is not None:
            stale_archive = _archive_stale_policy_exact_cover_checkpoint(
                destination=destination,
                paths=(
                    policy_plan,
                    policy_audit,
                    quality_repair_plan,
                    quality_repair_audit,
                ),
                reason=(
                    "policy_checkpoint_full_provenance_changed: " + stale_policy_reason
                ),
            )
            log_message(
                log_cb,
                "Archived a stale policy exact-cover checkpoint after full "
                "input-provenance validation and before rebuilding the visual "
                f"fallback policy: {stale_archive}",
            )
            policy_checkpoint_resume = False
        if policy_checkpoint_resume:
            log_message(
                log_cb,
                "Reusing the existing policy exact-cover checkpoint; its full "
                "input hashes and output plan will be revalidated.",
            )
        else:
            if policy_plan.exists() or policy_audit.exists():
                raise RuntimeError(
                    "Partial policy exact-cover checkpoint is incomplete; refusing "
                    "to overwrite or mix deterministic policy artifacts"
                )
            _run_stage(
                "policy_exact_cover",
                policy_command,
                log_cb,
                command_runner=_command_runner,
            )
        _require_file(policy_plan, "policy_exact_cover")
        _require_file(policy_audit, "policy_exact_cover")
        effective_material_plan = policy_plan
        policy_plan_document = read_object(policy_plan, "policy exact-cover plan")
        policy_audit_document = read_object(policy_audit, "policy exact-cover audit")
        policy_fallback_count = _validate_policy_exact_cover_bundle(
            plan=policy_plan_document,
            audit=policy_audit_document,
            registry=read_object(rendered_registry, "rendered registry"),
            staged_result=read_object(staged_result, "staged result"),
            confidence_gate=read_object(confidence_gate, "confidence gate"),
            base_plan=read_object(staged_material_plan, "autonomous material plan"),
            group_materials=read_object(group_materials, "group materials"),
            mvinverse_pbr_evidence=read_object(
                mvinverse_pbr_evidence, "MVInverse PBR evidence"
            ),
            whitelist=read_object(effective_whitelist, "material whitelist"),
            palette_fusion=read_object(
                analysis_dir / "palette_fusion.json", "palette fusion"
            ),
            expected_source_visual_strategy="neutralize_unverified",
            expected_policy_overrides=policy_input_document,
            expected_immutable_mdl_after_selection=(
                config.immutable_mdl_after_selection
            ),
        )

    if completed_inference_resume:
        late_resume_archive = _archive_partial_live_resume_downstream_artifacts(
            destination
        )
        if late_resume_archive is not None:
            log_message(
                log_cb,
                "Archived provisional post-inference Part-ID, Look, QA and "
                "selection artifacts before deterministic late-stage replay: "
                f"{late_resume_archive}",
            )

    if config.material_assignment_unit == "part_id":
        if not use_policy_fallback:
            raise RuntimeError(
                "Part-ID material assignment requires exact-cover fallback for "
                "parts that are hidden in every reference photo"
            )
        if effective_material_plan != policy_plan:
            raise RuntimeError(
                "Part-ID material assignment requires the validated policy "
                "exact-cover plan as its hidden-part baseline"
            )
        log_message(
            log_cb,
            "Material assignment unit: CAD part_id. Human-confirmed SAM3 masks "
            "are used only as whole-workpiece foreground; palette G01/G02 "
            "groups cannot assign or share a material.",
        )
        for required_path in (
            analysis_dir / "reference_manifest.json",
            analysis_dir / "spatial_mapping_report.json",
            rendered_registry,
            mvinverse_ledger,
            policy_plan,
        ):
            _require_file(required_path, "part_id_material_assignment")
        # Layer 1 establishes one camera and one coarse CAD box per Part-ID.
        # Layer 2 prompts local SAM3 inside those boxes.  Any image-plane
        # residual is estimated once from the whole workpiece and shared by
        # every Part-ID in that view; no individual mesh may move to follow a
        # segmentation candidate. CAD/USD geometry is never moved.
        for stale_dir in (part_id_coarse_evidence_dir, part_id_evidence_dir):
            if stale_dir.exists() or stale_dir.is_symlink():
                archived = unique_path(
                    analysis_dir / "recovery_archive" / f"stale_{stale_dir.name}"
                )
                archived.parent.mkdir(parents=True, exist_ok=True)
                stale_dir.rename(archived)
        coarse_evidence_document = build_part_id_reference_evidence(
            reference_manifest=analysis_dir / "reference_manifest.json",
            rendered_registry=rendered_registry,
            spatial_mapping_report=(analysis_dir / "spatial_mapping_report.json"),
            camera_alignment_acceptance=(
                camera_calibration_dir / "camera_alignment_acceptance.json"
                if (
                    camera_calibration_dir / "camera_alignment_acceptance.json"
                ).is_file()
                else None
            ),
            mvinverse_ledger=mvinverse_ledger,
            output_dir=part_id_coarse_evidence_dir,
        )
        write_object(part_id_coarse_evidence_path, coarse_evidence_document)
        write_object(
            part_id_sam3_request,
            build_part_id_sam3_request(part_id_coarse_evidence_path),
        )
        if part_id_sam3_dir.exists() or part_id_sam3_dir.is_symlink():
            archived = unique_path(
                analysis_dir / "recovery_archive" / "stale_part_id_sam3_regions"
            )
            archived.parent.mkdir(parents=True, exist_ok=True)
            part_id_sam3_dir.rename(archived)
        _run_stage(
            "part_id_sam3_local_refinement",
            [
                str(config.sam3_python),
                str(
                    ProjectLayout.from_root(root_dir()).material_pipeline
                    / "segmentation"
                    / "sam3_regions.py"
                ),
                "--request",
                str(part_id_sam3_request),
                "--repository",
                str(config.sam3_repository),
                "--checkpoint",
                str(config.sam3_checkpoint),
                "--output-dir",
                str(part_id_sam3_dir),
                "--device",
                config.sam3_device,
                "--minimum-model-score",
                str(config.sam3_minimum_model_score),
                "--minimum-prompt-overlap",
                str(config.sam3_minimum_prompt_overlap),
                "--maximum-image-fraction",
                str(config.sam3_maximum_image_fraction),
                "--minimum-mask-pixels",
                str(config.sam3_minimum_mask_pixels),
            ],
            log_cb,
            command_runner=_command_runner,
            required_files=(part_id_sam3_manifest,),
        )
        evidence_document = build_part_id_reference_evidence(
            reference_manifest=analysis_dir / "reference_manifest.json",
            rendered_registry=rendered_registry,
            spatial_mapping_report=(analysis_dir / "spatial_mapping_report.json"),
            camera_alignment_acceptance=(
                camera_calibration_dir / "camera_alignment_acceptance.json"
                if (
                    camera_calibration_dir / "camera_alignment_acceptance.json"
                ).is_file()
                else None
            ),
            mvinverse_ledger=mvinverse_ledger,
            part_id_sam3_manifest=part_id_sam3_manifest,
            output_dir=part_id_evidence_dir,
        )
        write_object(part_id_evidence_path, evidence_document)
        log_message(
            log_cb,
            "Two-layer one-to-one Part-ID mapping completed: every coarse box "
            "inherits the rigid whole-asset camera and the single residual shared "
            "by that view, then automatic local SAM3 refines its photo evidence. "
            "No CAD/USD transform was changed. "
            f"{evidence_document['summary'].get('sam3_refined_observation_count', 0)} "
            "observations passed local refinement; "
            f"{evidence_document['summary'].get('global_projection_fallback_observation_count', 0)} "
            "used the audited coarse fallback. "
            f"{evidence_document['summary'].get('chromatic_isolated_observation_count', 0)} "
            "single-view chromatic components were isolated, including "
            f"{evidence_document['summary'].get('tiny_chromatic_rescue_observation_count', 0)} "
            "small-part rescues. No Part-ID-local translation, rotation, scale, "
            "or CAD/USD geometry change is allowed. "
            "Selected-view coverage: "
            f"{evidence_document['summary'].get('selected_reference_view_coverage', {})}.",
        )
        # The initial exact-cover plan is built before local Part-ID visibility
        # is known. Rebuild it deterministically from the same trusted inputs
        # after final evidence exists so a hidden CAD part can never inherit a
        # palette group or repeated-identity material chosen for visible
        # siblings. This contract applies to every Part-ID workflow, not to a
        # particular asset, view, material group, or selection mode.
        registry_document = read_object(
            rendered_registry,
            "rendered registry for final Part-ID policy convergence",
        )
        staged_result_document = read_object(
            staged_result,
            "staged result for final Part-ID policy convergence",
        )
        confidence_gate_document = read_object(
            confidence_gate,
            "confidence gate for final Part-ID policy convergence",
        )
        base_plan_document = read_object(
            staged_material_plan,
            "autonomous plan for final Part-ID policy convergence",
        )
        group_materials_document = read_object(
            group_materials,
            "group materials for final Part-ID policy convergence",
        )
        mvinverse_pbr_document = read_object(
            mvinverse_pbr_evidence,
            "MVInverse evidence for final Part-ID policy convergence",
        )
        palette_fusion_document = read_object(
            analysis_dir / "palette_fusion.json",
            "palette fusion for final Part-ID policy convergence",
        )
        whitelist_document = read_object(
            effective_whitelist,
            "material whitelist for final Part-ID policy convergence",
        )
        try:
            policy_plan_document, policy_audit_document = build_policy_exact_cover(
                registry=registry_document,
                staged_result=staged_result_document,
                confidence_gate=confidence_gate_document,
                whitelist=whitelist_document,
                policy=policy_input_document,
                base_plan=base_plan_document,
                group_materials=group_materials_document,
                mvinverse_pbr_evidence=mvinverse_pbr_document,
                palette_fusion=palette_fusion_document,
                part_id_evidence=evidence_document,
                acknowledge_policy_fallback=True,
                immutable_mdl_after_selection=(config.immutable_mdl_after_selection),
            )
        except PolicyExactCoverError as exc:
            raise RuntimeError(
                "Part-ID workflow could not converge hidden-part policy "
                f"fallbacks to final visibility: {exc}"
            ) from exc
        policy_fallback_count = _validate_policy_exact_cover_bundle(
            plan=policy_plan_document,
            audit=policy_audit_document,
            registry=registry_document,
            staged_result=staged_result_document,
            confidence_gate=confidence_gate_document,
            base_plan=base_plan_document,
            group_materials=group_materials_document,
            mvinverse_pbr_evidence=mvinverse_pbr_document,
            whitelist=whitelist_document,
            palette_fusion=palette_fusion_document,
            part_id_evidence=evidence_document,
            expected_source_visual_strategy="neutralize_unverified",
            expected_policy_overrides=policy_input_document,
            expected_immutable_mdl_after_selection=(
                config.immutable_mdl_after_selection
            ),
        )
        write_object(policy_plan, policy_plan_document)
        write_object(policy_audit, policy_audit_document)
        convergence_summary = policy_audit_document["summary"]
        log_message(
            log_cb,
            "Policy baseline converged to final Part-ID visibility: "
            f"{convergence_summary['part_id_evidence_unobserved_count']} "
            "unobserved parts use independent policy fallbacks; palette groups "
            "and identity propagation remain restricted to observed parts.",
        )
        retrieval_request_document = build_part_id_retrieval_request(
            evidence=evidence_document,
            catalog=effective_catalog,
            material_root=config.material_root,
        )
        write_object(part_id_retrieval_request, retrieval_request_document)
        if part_id_retrieval_dir.exists():
            # A direct Part-ID request is hash-bound to its projected masks.
            # Never mix it with a prior palette-group retrieval directory.
            archived = unique_path(
                analysis_dir / "recovery_archive" / "stale_part_id_retrieval"
            )
            archived.parent.mkdir(parents=True, exist_ok=True)
            part_id_retrieval_dir.rename(archived)
            log_message(
                log_cb,
                "Archived an existing Part-ID retrieval directory before "
                f"recomputing the current mask-bound request: {archived}",
            )
        retrieval_command = [
            str(config.retrieval_python),
            str(ProjectLayout.from_root(root_dir()).material_retrieval_script),
            "--request",
            str(part_id_retrieval_request),
            "--siglip2-model",
            str(config.siglip2_model_path),
            "--dinov2-model",
            str(config.dinov2_model_path),
            "--cache-dir",
            str(config.retrieval_cache_dir),
            "--output-dir",
            str(part_id_retrieval_dir),
            "--device",
            config.retrieval_device,
            "--siglip-top-k",
            str(config.siglip_top_k),
            "--final-top-k",
            str(config.retrieval_final_top_k),
            "--batch-size",
            str(config.retrieval_batch_size),
        ]
        if config.retrieval_observation_bank_dir is not None:
            retrieval_command.extend(
                [
                    "--observation-bank",
                    str(config.retrieval_observation_bank_dir),
                ]
            )
        _run_stage(
            "part_id_visual_retrieval",
            retrieval_command,
            log_cb,
            command_runner=_command_runner,
            required_files=(part_id_retrieval_result,),
        )
        _require_file(part_id_retrieval_result, "part_id_visual_retrieval")
        if config.qwen_model_path is None:
            raise RuntimeError(
                "Part-ID assignment requires a local Qwen model path for the "
                "independent candidate rerank"
            )
        part_id_qwen_command = [
            str(config.qwen_python),
            "-m",
            "qwen_material_pipeline",
            "part-id-qwen",
            "--evidence",
            str(part_id_evidence_path),
            "--retrieval",
            str(part_id_retrieval_result),
            "--catalog",
            str(effective_catalog),
            "--model-path",
            str(config.qwen_model_path),
            "--output-dir",
            str(part_id_qwen_dir),
            "--output",
            str(part_id_qwen_result),
            "--batch-size",
            "4",
            "--candidate-count",
            "8",
            "--max-new-tokens",
            str(config.qwen_max_new_tokens),
        ]
        if (
            not config.immutable_mdl_after_selection
            and config.material_parameter_candidate_mode == "evidence_gated_h0_h1"
        ):
            part_id_qwen_command.append("--allow-mdl-color-tuning")
        if config.material_prediction_mode == "catalog_family_first":
            _require_file(
                appearance_components_report,
                "material-identity-first appearance-component evidence",
            )
            try:
                identity_component_document = filter_components_for_material_evidence(
                    appearance_components=read_object(
                        appearance_components_report,
                        "material-identity-first appearance components",
                    ),
                    part_id_evidence=evidence_document,
                )
            except AppearanceComponentMaterialError as exc:
                raise RuntimeError(
                    "Unable to bind material-identity components to final Part-ID "
                    f"evidence: {exc}"
                ) from exc
            write_object(
                appearance_component_material_memberships,
                identity_component_document,
            )
            part_id_qwen_command.extend(
                [
                    "--require-material-family-prediction",
                    "--appearance-components",
                    str(appearance_component_material_memberships),
                ]
            )
            if config.material_identity_local_context:
                part_id_qwen_command.append("--material-identity-local-context")
        _run_stage(
            "part_id_qwen_rerank",
            part_id_qwen_command,
            log_cb,
            command_runner=_command_runner,
            required_files=(part_id_qwen_result,),
        )
        qwen_part_document = read_object(
            part_id_qwen_result,
            "Part-ID Qwen choices",
        )
        if config.material_prediction_mode != "disabled":
            qwen_part_integrity = qwen_part_document.get("integrity")
            qwen_part_unsigned = copy.deepcopy(qwen_part_document)
            qwen_part_unsigned.pop("integrity", None)
            if not isinstance(qwen_part_integrity, Mapping) or qwen_part_integrity.get(
                "document_sha256"
            ) != canonical_sha256(qwen_part_unsigned):
                raise RuntimeError("Part-ID Qwen result failed its integrity seal")
            if qwen_part_document.get("material_prediction_mode") != (
                config.material_prediction_mode
            ):
                raise RuntimeError(
                    "Part-ID Qwen material-prediction mode does not match the config"
                )
            expected_identity_evidence_mode = (
                "isolated_target_with_local_context_for_independent_parts"
                if config.material_identity_local_context
                else "isolated_target_only"
            )
            if qwen_part_document.get("material_identity_evidence_mode") != (
                expected_identity_evidence_mode
            ):
                raise RuntimeError(
                    "Part-ID Qwen material-identity evidence mode does not match "
                    "the config"
                )
        qwen_part_choices = qwen_part_document.get("choices")
        if not isinstance(qwen_part_choices, dict):
            raise RuntimeError("Part-ID Qwen result contains no choices")
        qwen_part_confidences = {
            str(row["part_id"]): float(row["confidence"])
            for row in qwen_part_document.get("selections", [])
            if isinstance(row, dict)
            and isinstance(row.get("part_id"), str)
            and isinstance(row.get("confidence"), (int, float))
            and not isinstance(row.get("confidence"), bool)
        }
        qwen_material_predictions: dict[str, dict[str, Any]] | None = None
        if config.material_prediction_mode == "catalog_family_first":
            qwen_material_predictions = _validate_catalog_family_first_result(
                qwen_document=qwen_part_document,
                choices=qwen_part_choices,
                confidences=qwen_part_confidences,
                catalog_document=read_object(
                    effective_catalog,
                    "material-identity-first NVIDIA MDL catalog",
                ),
            )
        appearance_component_document: dict[str, Any] | None = None
        appearance_component_retrieval_document: dict[str, Any] | None = None
        appearance_component_qwen_document: dict[str, Any] | None = None
        if (
            config.material_prediction_mode == "disabled"
            and appearance_components_report.is_file()
        ):
            candidate_component_document = read_object(
                appearance_components_report,
                "photo-supported appearance components",
            )
            try:
                candidate_component_document = filter_components_for_material_evidence(
                    appearance_components=candidate_component_document,
                    part_id_evidence=evidence_document,
                )
            except AppearanceComponentMaterialError as exc:
                raise RuntimeError(
                    "Unable to filter photo-supported appearance components against "
                    f"selected Part-ID evidence; material inference was not started: {exc}"
                ) from exc
            write_object(
                appearance_component_material_memberships,
                candidate_component_document,
            )
            raw_components = candidate_component_document.get("components")
            if isinstance(raw_components, list) and raw_components:
                try:
                    (
                        component_evidence_document,
                        component_retrieval_request_document,
                    ) = build_component_material_inputs(
                        appearance_components=candidate_component_document,
                        part_id_evidence=evidence_document,
                        catalog=effective_catalog,
                        material_root=config.material_root,
                        output_dir=appearance_component_input_dir,
                    )
                except AppearanceComponentMaterialError as exc:
                    raise RuntimeError(
                        "Unable to construct aggregate photo-appearance component "
                        f"material inputs; material inference was not started: {exc}"
                    ) from exc
                write_object(appearance_component_evidence, component_evidence_document)
                write_object(
                    appearance_component_retrieval_request,
                    component_retrieval_request_document,
                )
                if appearance_component_retrieval_dir.exists():
                    archived = unique_path(
                        analysis_dir
                        / "recovery_archive"
                        / "stale_appearance_component_retrieval"
                    )
                    archived.parent.mkdir(parents=True, exist_ok=True)
                    appearance_component_retrieval_dir.rename(archived)
                    log_message(
                        log_cb,
                        "Archived a stale appearance-component retrieval before "
                        f"recomputing aggregate evidence: {archived}",
                    )
                component_retrieval_command = [
                    str(config.retrieval_python),
                    str(ProjectLayout.from_root(root_dir()).material_retrieval_script),
                    "--request",
                    str(appearance_component_retrieval_request),
                    "--siglip2-model",
                    str(config.siglip2_model_path),
                    "--dinov2-model",
                    str(config.dinov2_model_path),
                    "--cache-dir",
                    str(config.retrieval_cache_dir),
                    "--output-dir",
                    str(appearance_component_retrieval_dir),
                    "--device",
                    config.retrieval_device,
                    "--siglip-top-k",
                    str(config.siglip_top_k),
                    "--final-top-k",
                    str(config.retrieval_final_top_k),
                    "--batch-size",
                    str(config.retrieval_batch_size),
                ]
                if config.retrieval_observation_bank_dir is not None:
                    component_retrieval_command.extend(
                        [
                            "--observation-bank",
                            str(config.retrieval_observation_bank_dir),
                        ]
                    )
                _run_stage(
                    "appearance_component_visual_retrieval",
                    component_retrieval_command,
                    log_cb,
                    command_runner=_command_runner,
                    required_files=(appearance_component_retrieval_result,),
                )
                _require_file(
                    appearance_component_retrieval_result,
                    "appearance_component_visual_retrieval",
                )
                component_qwen_command = [
                    str(config.qwen_python),
                    "-m",
                    "qwen_material_pipeline",
                    "appearance-component-qwen",
                    "--evidence",
                    str(appearance_component_evidence),
                    "--retrieval",
                    str(appearance_component_retrieval_result),
                    "--catalog",
                    str(effective_catalog),
                    "--model-path",
                    str(config.qwen_model_path),
                    "--output-dir",
                    str(appearance_component_qwen_dir),
                    "--output",
                    str(appearance_component_qwen_result),
                    "--batch-size",
                    "4",
                    "--candidate-count",
                    "8",
                    "--max-new-tokens",
                    str(config.qwen_max_new_tokens),
                ]
                _run_stage(
                    "appearance_component_qwen_rerank",
                    component_qwen_command,
                    log_cb,
                    command_runner=_command_runner,
                    required_files=(appearance_component_qwen_result,),
                )
                appearance_component_document = candidate_component_document
                appearance_component_retrieval_document = read_object(
                    appearance_component_retrieval_result,
                    "appearance-component visual retrieval result",
                )
                appearance_component_qwen_document = read_object(
                    appearance_component_qwen_result,
                    "appearance-component Qwen choices",
                )
        direct_plan_document, direct_audit_document = build_part_id_material_plan(
            base_plan=read_object(policy_plan, "Part-ID exact-cover baseline"),
            evidence=evidence_document,
            retrieval_result=read_object(
                part_id_retrieval_result,
                "Part-ID visual retrieval result",
            ),
            qwen_choices=qwen_part_choices,
            qwen_confidences=qwen_part_confidences,
            qwen_material_predictions=qwen_material_predictions,
            allow_color_parameters=(
                not config.immutable_mdl_after_selection
                and config.material_parameter_candidate_mode == "evidence_gated_h0_h1"
            ),
            part_registry=read_object(
                source_registry,
                "Part-ID source registry for coating consistency",
            ),
            enforce_coating_consistency=(
                appearance_component_document is None
                and config.material_prediction_mode == "disabled"
            ),
        )
        if (
            appearance_component_document is not None
            and appearance_component_retrieval_document is not None
            and appearance_component_qwen_document is not None
        ):
            try:
                (
                    direct_plan_document,
                    direct_audit_document,
                ) = apply_fixed_component_mdl_choices(
                    base_plan=direct_plan_document,
                    base_audit=direct_audit_document,
                    appearance_components=appearance_component_document,
                    part_id_evidence=evidence_document,
                    component_evidence=component_evidence_document,
                    component_retrieval=appearance_component_retrieval_document,
                    component_qwen_choices=appearance_component_qwen_document,
                )
            except AppearanceComponentMaterialError as exc:
                raise RuntimeError(
                    "Photo-supported appearance-component fixed-MDL selection "
                    f"failed closed; material inference was not started: {exc}"
                ) from exc
            write_object(
                appearance_component_mdl_selection_audit,
                direct_audit_document["appearance_component_mdl_selection"],
            )
        write_object(part_id_material_plan, direct_plan_document)
        write_object(part_id_material_audit, direct_audit_document)
        effective_material_plan = part_id_material_plan
        log_message(
            log_cb,
            "Independent Part-ID material decisions completed: "
            f"{direct_audit_document['summary']['independently_selected_count']} "
            "photo-observed parts selected independently; "
            f"{direct_audit_document['summary']['unobserved_preserved_count']} "
            "fully hidden parts retained the exact-cover fallback.",
        )
        coating_summary = direct_audit_document["coating_consistency_gate"]["summary"]
        if appearance_component_document is not None:
            component_summary = direct_audit_document["summary"]
            log_message(
                log_cb,
                "Photo-supported appearance-component fixed-MDL selection completed: "
                f"{component_summary['appearance_component_count']} components "
                f"constrained {component_summary['appearance_component_constrained_part_count']} "
                "Part IDs. Each selected NVIDIA Base MDL remains immutable.",
            )
        elif config.material_prediction_mode == "disabled":
            log_message(
                log_cb,
                "Automatic same-coating consistency passed: "
                f"{coating_summary['component_count']} safe components constrained "
                f"{coating_summary['constrained_part_count']} Part IDs; "
                f"{coating_summary['material_changed_part_count']} independent "
                "material choices were unified without palette G-groups.",
            )
        else:
            prediction_summary = qwen_part_document.get("summary", {})
            log_message(
                log_cb,
                "Material-identity-first Part-ID assignment completed: "
                f"{prediction_summary.get('material_prediction_applyable_count', 0)} "
                "predictions were applyable; "
                f"{prediction_summary.get('material_prediction_insufficient_evidence_count', 0)} "
                "parts retained their fallback; cross-family fallback count=0. "
                "No colour parameters or appearance-component identity sharing "
                "were allowed in this stage.",
            )

    plan = read_object(effective_material_plan, "material plan")
    assignments = plan.get("assignments")
    if not isinstance(assignments, list) or not assignments:
        raise RuntimeError(
            "READY_TO_APPLY result contains no material assignments; "
            "physics was not started"
        )
    selection_lock_document: dict[str, Any] | None = None
    if config.immutable_mdl_after_selection:
        log_message(
            log_cb,
            "The exact-cover Look is a provisional render candidate. Final NVIDIA "
            "MDL selection will be sealed only after automatic render QA and the "
            "single evidence-bounded selection refinement pass.",
        )

    return PolicyPartIdStageResult(
        effective_catalog=effective_catalog,
        effective_whitelist=effective_whitelist,
        live_material_count=live_material_count,
        state=state,
        use_policy_fallback=use_policy_fallback,
        effective_material_plan=effective_material_plan,
        policy_fallback_count=policy_fallback_count,
        policy_plan_document=policy_plan_document,
        policy_audit_document=policy_audit_document,
        plan=plan,
        assignments=assignments,
        selection_lock_document=selection_lock_document,
    )


@dataclass(frozen=True)
class LookApplicationStageResult:
    apply_subcommand: str
    apply_asset_flag: str
    apply_asset: Path
    instance_plan: Path | None
    rendered_registry_document: dict[str, Any] | None
    applied_count: int


def _run_look_application_stage(
    context: VisualMaterialPipelineContext,
    *,
    prepared_source: SourcePreparationResult,
    planning: PolicyPartIdStageResult,
    require_complete_coverage: bool,
    log_cb: LogCallback,
    command_runner: CommandRunner,
) -> LookApplicationStageResult:
    """Apply one provisional non-destructive Look with exact coverage checks."""

    source = context.source
    config = context.config
    isaac = context.isaac_python
    destination = context.destination
    rendered_registry = prepared_source.rendered_registry
    instance_root_count = prepared_source.instance_root_count
    effective_material_plan = planning.effective_material_plan
    effective_catalog = planning.effective_catalog
    use_policy_fallback = planning.use_policy_fallback
    plan = planning.plan
    assignments = planning.assignments
    selection_lock_document = planning.selection_lock_document
    look_usd = context.workspace.look.initial_usd
    apply_report = context.workspace.look.initial_apply_report
    material_selection_lock = context.workspace.inference.material_selection_lock
    _command_runner = command_runner

    apply_plan = effective_material_plan
    apply_subcommand = "apply"
    apply_asset_flag = "--asset-usd"
    apply_asset = source
    instance_plan: Path | None = None
    rendered_registry_document: dict[str, Any] | None = None
    if instance_root_count or require_complete_coverage:
        rendered_registry_document = read_object(
            rendered_registry, "rendered part registry"
        )
        raw_parts = rendered_registry_document.get("parts")
        if not isinstance(raw_parts, list) or not raw_parts:
            raise RuntimeError(
                "CAD registry contains no material-addressable parts; "
                "physics was not started"
            )
        registry_part_ids = [
            item.get("part_id") if isinstance(item, dict) else None
            for item in raw_parts
        ]
        assignment_part_ids = [
            item.get("part_id") if isinstance(item, dict) else None
            for item in assignments
        ]
        if any(not isinstance(part_id, str) for part_id in registry_part_ids) or len(
            set(registry_part_ids)
        ) != len(registry_part_ids):
            raise RuntimeError("CAD registry has invalid or duplicate part IDs")
        if (
            any(not isinstance(part_id, str) for part_id in assignment_part_ids)
            or len(set(assignment_part_ids)) != len(assignment_part_ids)
            or set(assignment_part_ids) != set(registry_part_ids)
        ):
            missing = sorted(set(registry_part_ids) - set(assignment_part_ids))
            unexpected = sorted(set(assignment_part_ids) - set(registry_part_ids))
            raise RuntimeError(
                "A STEP/STP workpiece requires one safe material assignment for every "
                "material-addressable Mesh part; "
                f"missing={missing[:20]} total_missing={len(missing)} "
                f"unexpected={unexpected[:20]} "
                f"total_unexpected={len(unexpected)}. "
                "Physics was not started."
            )
        complete_coverage_statuses = _complete_coverage_assignment_statuses(
            material_assignment_unit=config.material_assignment_unit,
            include_policy_fallback=use_policy_fallback,
        )
        invalid_statuses = sorted(
            {
                str(item.get("status", "review"))
                for item in assignments
                if not isinstance(item, dict)
                or item.get("status", "review") not in complete_coverage_statuses
            }
        )
        if invalid_statuses:
            raise RuntimeError(
                "STEP/STP exact-cover plan contains non-applicable assignment "
                f"statuses: {invalid_statuses}; physics was not started"
            )

    if instance_root_count:
        if rendered_registry_document is None:
            raise AssertionError("instance registry was not loaded")
        instance_plan = destination / "autonomous_instance_material_plan.json"
        instance_document = dict(plan)
        raw_provenance = instance_document.get("provenance")
        sealed_provenance = (
            dict(raw_provenance) if isinstance(raw_provenance, dict) else {}
        )
        sealed_provenance.update(
            {
                "asset_sha256": sha256_file(source),
                "registry_sha256": canonical_sha256(rendered_registry_document),
            }
        )
        instance_document["provenance"] = sealed_provenance
        write_object(instance_plan, instance_document)
        apply_plan = instance_plan
        apply_subcommand = "apply-instances"
        apply_asset_flag = "--source-usd"

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
        str(apply_plan),
        "--output",
        str(look_usd),
        "--material-root",
        str(config.material_root),
        "--report",
        str(apply_report),
        # A selective-regression result must be written for every observed
        # Part-ID even when its confidence remains in the review band.
        "--include-review",
    ]
    if use_policy_fallback:
        apply_command.append("--include-policy-fallback")
    if selection_lock_document is not None:
        apply_command.extend(["--selection-lock", str(material_selection_lock)])
    _run_stage(
        "apply_usd",
        apply_command,
        log_cb,
        command_runner=_command_runner,
        retry_native_crash=True,
        required_files=(look_usd, apply_report),
    )
    _require_file(look_usd, "apply_usd")
    _require_file(apply_report, "apply_usd")
    applied = read_object(apply_report, "apply report")
    applied_count = applied.get("applied_count")
    if (
        isinstance(applied_count, bool)
        or not isinstance(applied_count, int)
        or applied_count <= 0
    ):
        raise RuntimeError(
            f"Visual-material apply report has invalid applied_count: {applied_count!r}"
        )
    if require_complete_coverage and rendered_registry_document is not None:
        registry_part_count = len(rendered_registry_document["parts"])
        if applied_count != registry_part_count:
            raise RuntimeError(
                "STEP/STP material apply did not cover every Mesh part: "
                f"applied={applied_count} expected={registry_part_count}; "
                "physics was not started"
            )

    if isinstance(applied_count, bool) or not isinstance(applied_count, int):
        raise RuntimeError("USD apply report has an invalid applied_count")
    return LookApplicationStageResult(
        apply_subcommand=apply_subcommand,
        apply_asset_flag=apply_asset_flag,
        apply_asset=apply_asset,
        instance_plan=instance_plan,
        rendered_registry_document=rendered_registry_document,
        applied_count=applied_count,
    )


@dataclass(frozen=True)
class VisualQaStageResult:
    quality_render_view_arguments: Sequence[str]
    spatial_report_path: Path
    parameter_tournament_document: dict[str, Any] | None
    component_actual_mdl_tournament_document: dict[str, Any] | None
    corresponding_color_calibration_document: dict[str, Any] | None
    trusted_mapping: dict[str, str]
    raw_quality_status: Any
    quality_status: str
    part_id_quality_gate_document: dict[str, Any] | None
    effective_look_usd: Path
    effective_apply_report: Path
    effective_quality_report: Path
    effective_quality_rendered_registry: Path
    quality_repair_used: bool
    quality_repair_changed_count: int
    quality_round_count: int
    quality_repair_round_count: int
    quality_gate_status: str
    quality_resolution_document: dict[str, Any] | None
    quality_limitation_count: int
    final_instance_plan: Path | None
    effective_material_plan: Path
    assignments: list[dict[str, Any]]
    applied_count: int
    quality_decision: str
    membership_tournament_status: str
    membership_tournament_cohort_count: int
    membership_selected_expanded_count: int
    membership_restored_m0_count: int


def _run_visual_qa_stage(
    context: VisualMaterialPipelineContext,
    *,
    prepared_source: SourcePreparationResult,
    planning: PolicyPartIdStageResult,
    look: LookApplicationStageResult,
    require_complete_coverage: bool,
    log_cb: LogCallback,
    command_runner: CommandRunner,
) -> VisualQaStageResult:
    """Render, compare and run the single bounded pre-selection QA repair."""

    source = context.source
    config = context.config
    isaac = context.isaac_python
    destination = context.destination
    workspace = context.workspace
    rendered_registry = prepared_source.rendered_registry
    source_registry = prepared_source.source_registry
    instance_root_count = prepared_source.instance_root_count
    rendered_registry_document = look.rendered_registry_document
    apply_subcommand = look.apply_subcommand
    apply_asset_flag = look.apply_asset_flag
    apply_asset = look.apply_asset
    instance_plan = look.instance_plan
    applied_count = look.applied_count
    effective_catalog = planning.effective_catalog
    effective_whitelist = planning.effective_whitelist
    effective_material_plan = planning.effective_material_plan
    use_policy_fallback = planning.use_policy_fallback
    policy_plan_document = planning.policy_plan_document
    policy_audit_document = planning.policy_audit_document
    plan = planning.plan
    assignments = planning.assignments
    completed_inference_resume = (
        context.partial_live_resume and workspace.inference.unattended_result.is_file()
    )
    analysis_dir = workspace.inference.root
    mvinverse_pbr_evidence = workspace.inference.mvinverse_pbr_evidence
    policy_plan = workspace.inference.policy_plan
    policy_audit = workspace.inference.policy_audit
    group_materials = workspace.inference.group_materials
    look_usd = workspace.look.initial_usd
    apply_report = workspace.look.initial_apply_report
    repaired_look_usd = workspace.look.repaired_usd
    repaired_apply_report = workspace.look.repaired_apply_report
    repaired_instance_plan = workspace.look.repaired_instance_plan
    quality_paths = workspace.quality
    quality_registry = quality_paths.registry
    quality_render_dir = quality_paths.render_dir
    quality_rendered_registry = quality_paths.rendered_registry
    quality_view_map = quality_paths.view_map
    quality_report = quality_paths.report
    quality_camera_view_specs = workspace.quality_camera_view_specs
    repaired_quality_paths = workspace.repaired_quality
    repaired_quality_registry = repaired_quality_paths.registry
    repaired_quality_render_dir = repaired_quality_paths.render_dir
    repaired_quality_rendered_registry = repaired_quality_paths.rendered_registry
    repaired_quality_view_map = repaired_quality_paths.view_map
    repaired_quality_report = repaired_quality_paths.report
    part_id_paths = workspace.part_id
    part_id_evidence_path = part_id_paths.evidence
    part_id_material_audit = part_id_paths.material_audit
    part_id_quality_gate = part_id_paths.quality_gate
    part_id_parameter_tournament_dir = part_id_paths.parameter_tournament_dir
    part_id_parameter_tournament_plan = part_id_paths.parameter_tournament_plan
    part_id_parameter_tournament_audit = part_id_paths.parameter_tournament_audit
    appearance_paths = workspace.appearance
    appearance_component_mdl_selection_audit = appearance_paths.mdl_selection_audit
    appearance_component_retrieval_result = appearance_paths.retrieval_result
    appearance_component_qwen_result = appearance_paths.qwen_result
    appearance_component_actual_mdl_tournament_dir = (
        appearance_paths.actual_mdl_tournament_dir
    )
    appearance_component_actual_mdl_tournament_plan = (
        appearance_paths.actual_mdl_tournament_plan
    )
    appearance_component_actual_mdl_tournament_audit = (
        appearance_paths.actual_mdl_tournament_audit
    )
    legacy_paths = workspace.legacy
    quality_repair_plan = legacy_paths.quality_repair_plan
    quality_repair_audit = legacy_paths.quality_repair_audit
    quality_resolution = legacy_paths.quality_resolution
    membership_tournament_dir = legacy_paths.membership_dir
    membership_tournament_view_map = legacy_paths.membership_view_map
    membership_tournament_plan = legacy_paths.membership_plan
    membership_tournament_instance_plan = legacy_paths.membership_instance_plan
    membership_tournament_audit = legacy_paths.membership_audit
    _command_runner = command_runner
    corresponding_color_calibration_document: dict[str, Any] | None = None

    _run_stage(
        "quality_registry",
        [
            str(isaac),
            "-m",
            "qwen_material_pipeline",
            "usd",
            "registry",
            "--usd",
            str(look_usd),
            "--output",
            str(quality_registry),
        ],
        log_cb,
        command_runner=_command_runner,
        retry_native_crash=True,
    )
    _require_file(quality_registry, "quality_registry")
    quality_render_view_arguments = _render_view_arguments(
        baseline_registry=(
            rendered_registry_document
            if rendered_registry_document is not None
            else read_object(rendered_registry, "rendered part registry")
        ),
        view_specs_output=quality_camera_view_specs,
        fallback_views=config.render_views,
    )
    _run_stage(
        "quality_render",
        [
            str(isaac),
            "-m",
            "qwen_material_pipeline",
            "usd",
            "render",
            "--registry",
            str(quality_registry),
            "--output-dir",
            str(quality_render_dir),
            "--resolution",
            str(config.render_resolution),
            *quality_render_view_arguments,
            "--rt-subframes",
            str(config.render_rt_subframes),
            "--lighting-profile",
            config.quality_lighting_profile,
            "--analysis-up-axis",
            config.analysis_up_axis,
            f"--analysis-front-axis={config.analysis_front_axis}",
        ],
        log_cb,
        command_runner=_command_runner,
        retry_native_crash=True,
    )
    _require_file(quality_rendered_registry, "quality_render")

    corresponding_part_ids: tuple[str, ...] = ()
    if config.corresponding_color_calibration_mode == "adaptive_actual_cad":
        corresponding_part_ids = corresponding_material_part_ids(
            read_object(
                part_id_paths.qwen_result,
                "material-identity Qwen choices for colour calibration",
            )
        )
        if not corresponding_part_ids:
            log_message(
                log_cb,
                "All photo-observed material selections are exact library "
                "presets; corresponding-material colour calibration is not "
                "required.",
            )
        else:
            log_message(
                log_cb,
                "Corresponding-material colour eligibility: "
                f"{len(corresponding_part_ids)} Part IDs. Exact presets and "
                "unobserved policy fallbacks remain unchanged.",
            )
    if (
        config.corresponding_color_calibration_mode == "adaptive_actual_cad"
        and corresponding_part_ids
    ):
        if instance_root_count:
            raise RuntimeError(
                "Adaptive corresponding-material colour calibration does not "
                "yet support occurrence-instance authoring; refusing to change "
                "the identity-fixed plan"
            )
        color_paths = workspace.corresponding_color
        if color_paths.root.exists() or color_paths.root.is_symlink():
            raise RuntimeError(
                "Corresponding-material colour calibration requires a fresh "
                f"stage directory: {color_paths.root}"
            )
        color_command_index = 0

        def run_color_command(
            command: Sequence[str],
            _log_path: Path,
            _environment: Mapping[str, str],
            _timeout_seconds: int,
        ) -> None:
            nonlocal color_command_index
            color_command_index += 1
            _run_stage(
                f"corresponding_material_color_{color_command_index:02d}",
                [str(value) for value in command],
                log_cb,
                command_runner=_command_runner,
                retry_native_crash=True,
            )

        log_message(
            log_cb,
            "Material identities are fixed. Starting automatic per-scope colour "
            "calibration only for corresponding-material assignments; exact "
            "library presets remain untouched.",
        )
        try:
            run_corresponding_material_color_workflow(
                source_plan_path=effective_material_plan,
                qwen_choices_path=part_id_paths.qwen_result,
                part_id_evidence_path=part_id_evidence_path,
                spatial_mapping_report_path=(
                    analysis_dir / "spatial_mapping_report.json"
                ),
                asset_usd_path=Path(apply_asset),
                catalog_path=effective_catalog,
                registry_path=rendered_registry,
                material_root_path=config.material_root,
                view_specs_path=quality_camera_view_specs,
                reference_manifest_path=(analysis_dir / "reference_manifest.json"),
                isaac_python_path=isaac,
                output_dir=color_paths.root,
                max_adaptive_iterations=(config.corresponding_color_max_iterations),
                resolution=config.render_resolution,
                rt_subframes=config.render_rt_subframes,
                command_runner=run_color_command,
            )
        except CorrespondingMaterialColorWorkflowError as exc:
            raise RuntimeError(
                "Identity-preserving corresponding-material colour calibration "
                f"failed closed: {exc}"
            ) from exc
        color_result = validate_corresponding_color_result(
            manifest_path=color_paths.manifest,
            source_plan_path=effective_material_plan,
            qwen_choices_path=part_id_paths.qwen_result,
            part_id_evidence_path=part_id_evidence_path,
            spatial_mapping_report_path=(analysis_dir / "spatial_mapping_report.json"),
            asset_usd_path=Path(apply_asset),
            catalog_path=effective_catalog,
            registry_path=rendered_registry,
            material_root_path=config.material_root,
            view_specs_path=quality_camera_view_specs,
            reference_manifest_path=(analysis_dir / "reference_manifest.json"),
            isaac_python_path=isaac,
            selected_plan_path=color_paths.selected_plan,
            selection_audit_path=color_paths.selection_audit,
            look_usd_path=color_paths.look_usd,
            apply_report_path=color_paths.apply_report,
            registry_output_path=color_paths.registry,
            rendered_registry_path=color_paths.rendered_registry,
            quality_report_path=color_paths.quality_report,
        )
        corresponding_color_calibration_document = color_result.manifest
        rebound_part_id_audit = rebind_part_id_audit_for_corresponding_color(
            source_audit=read_object(
                part_id_material_audit,
                "identity-fixed Part-ID material audit",
            ),
            source_plan=plan,
            final_plan=color_result.selected_plan,
            selection_audit=color_result.selection_audit,
        )
        write_object(part_id_material_audit, rebound_part_id_audit)
        effective_material_plan = color_paths.selected_plan
        plan = color_result.selected_plan
        assignments = plan["assignments"]
        look_usd = color_paths.look_usd
        apply_report = color_paths.apply_report
        quality_rendered_registry = color_paths.rendered_registry
        applied_count = color_result.applied_count
        log_message(
            log_cb,
            "Automatic corresponding-material colour calibration completed: "
            f"iterations={len(color_result.manifest.get('candidates', []))}; "
            "MDL identity changes=0. The main all-view QA will now independently "
            "re-evaluate the selected actual-CAD render.",
        )

    spatial_report_path = analysis_dir / "spatial_mapping_report.json"
    parameter_tournament_document: dict[str, Any] | None = None
    if (
        config.material_assignment_unit == "part_id"
        and config.material_parameter_candidate_mode == "evidence_gated_h0_h1"
    ):
        pending_parameter_part_ids = pending_h1_part_ids(plan)
        if pending_parameter_part_ids:
            log_message(
                log_cb,
                "Part-ID parameter tournament: "
                f"{len(pending_parameter_part_ids)} evidence-gated H1 color "
                "candidates will be compared against untouched H0 on actual "
                "CAD renders.",
            )
            evidence_for_parameter_tournament = read_object(
                part_id_evidence_path,
                "Part-ID parameter tournament evidence",
            )
            spatial_for_parameter_tournament = read_object(
                spatial_report_path,
                "Part-ID parameter tournament spatial registration",
            )
            baseline_rendered_for_parameter_tournament = read_object(
                quality_rendered_registry,
                "Part-ID parameter tournament H0 render registry",
            )
            baseline_parameter_scores: dict[str, dict[str, Any]] = {}
            h1_parameter_scores: dict[str, dict[str, Any]] = {}

            for index, part_id in enumerate(
                pending_parameter_part_ids,
                start=1,
            ):
                log_message(
                    log_cb,
                    "Part-ID parameter tournament "
                    f"{index}/{len(pending_parameter_part_ids)}: {part_id}",
                )
                baseline_parameter_scores[part_id] = score_part_id_render(
                    part_id=part_id,
                    evidence=evidence_for_parameter_tournament,
                    spatial_mapping_report=spatial_for_parameter_tournament,
                    rendered_registry=(baseline_rendered_for_parameter_tournament),
                )
                candidate_dir = part_id_parameter_tournament_dir / part_id / "H1"
                if candidate_dir.exists() or candidate_dir.is_symlink():
                    archived = unique_path(
                        analysis_dir
                        / "recovery_archive"
                        / f"stale_parameter_tournament_{part_id}_H1"
                    )
                    archived.parent.mkdir(parents=True, exist_ok=True)
                    candidate_dir.rename(archived)
                candidate_dir.mkdir(parents=True, exist_ok=True)
                candidate_plan_path = candidate_dir / "plan.json"
                candidate_apply_plan_path = candidate_plan_path
                candidate_look = candidate_dir / "look.usda"
                candidate_apply_report = candidate_dir / "apply_report.json"
                candidate_registry = candidate_dir / "part_registry.json"
                candidate_render_dir = candidate_dir / "renders"
                candidate_rendered_registry = (
                    candidate_render_dir / "part_registry.rendered.json"
                )
                try:
                    candidate_plan_document = build_h1_candidate_plan(
                        source_plan=plan,
                        part_id=part_id,
                    )
                except PartIdParameterTournamentError as exc:
                    raise RuntimeError(
                        f"Unable to build Part-ID H1 candidate for {part_id}: {exc}"
                    ) from exc
                if instance_root_count:
                    candidate_provenance = candidate_plan_document.get("provenance")
                    sealed_candidate_provenance = (
                        dict(candidate_provenance)
                        if isinstance(candidate_provenance, dict)
                        else {}
                    )
                    sealed_candidate_provenance.update(
                        {
                            "asset_sha256": sha256_file(source),
                            "registry_sha256": canonical_sha256(
                                rendered_registry_document
                            ),
                        }
                    )
                    candidate_plan_document["provenance"] = sealed_candidate_provenance
                write_object(candidate_plan_path, candidate_plan_document)
                candidate_apply_command = [
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
                    str(candidate_look),
                    "--material-root",
                    str(config.material_root),
                    "--report",
                    str(candidate_apply_report),
                    "--include-review",
                ]
                if use_policy_fallback:
                    candidate_apply_command.append("--include-policy-fallback")
                _run_stage(
                    f"part_id_parameter_{part_id}_h1_apply",
                    candidate_apply_command,
                    log_cb,
                    command_runner=_command_runner,
                    retry_native_crash=True,
                    required_files=(candidate_look, candidate_apply_report),
                )
                _run_stage(
                    f"part_id_parameter_{part_id}_h1_registry",
                    [
                        str(isaac),
                        "-m",
                        "qwen_material_pipeline",
                        "usd",
                        "registry",
                        "--usd",
                        str(candidate_look),
                        "--output",
                        str(candidate_registry),
                    ],
                    log_cb,
                    command_runner=_command_runner,
                    retry_native_crash=True,
                    required_files=(candidate_registry,),
                )
                _run_stage(
                    f"part_id_parameter_{part_id}_h1_render",
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
                        str(config.render_resolution),
                        *quality_render_view_arguments,
                        "--rt-subframes",
                        str(config.render_rt_subframes),
                        "--lighting-profile",
                        config.quality_lighting_profile,
                        "--analysis-up-axis",
                        config.analysis_up_axis,
                        f"--analysis-front-axis={config.analysis_front_axis}",
                    ],
                    log_cb,
                    command_runner=_command_runner,
                    retry_native_crash=True,
                    required_files=(candidate_rendered_registry,),
                )
                h1_parameter_scores[part_id] = score_part_id_render(
                    part_id=part_id,
                    evidence=evidence_for_parameter_tournament,
                    spatial_mapping_report=spatial_for_parameter_tournament,
                    rendered_registry=read_object(
                        candidate_rendered_registry,
                        f"Part-ID {part_id} H1 rendered registry",
                    ),
                )

            try:
                (
                    parameter_tournament_plan_document,
                    parameter_tournament_document,
                ) = select_parameter_tournament_winners(
                    source_plan=plan,
                    baseline_scores=baseline_parameter_scores,
                    h1_scores=h1_parameter_scores,
                    minimum_score_improvement=(
                        config.exact_mdl_tournament_minimum_score_improvement
                    ),
                )
                rebound_part_id_audit = rebind_part_id_material_audit(
                    source_audit=read_object(
                        part_id_material_audit,
                        "pre-tournament Part-ID material audit",
                    ),
                    final_plan=parameter_tournament_plan_document,
                    tournament_audit=parameter_tournament_document,
                )
            except PartIdParameterTournamentError as exc:
                raise RuntimeError(
                    f"Part-ID H0/H1 render tournament failed: {exc}"
                ) from exc
            write_object(
                part_id_parameter_tournament_plan,
                parameter_tournament_plan_document,
            )
            write_object(
                part_id_parameter_tournament_audit,
                parameter_tournament_document,
            )
            write_object(part_id_material_audit, rebound_part_id_audit)
            effective_material_plan = part_id_parameter_tournament_plan
            plan = parameter_tournament_plan_document
            assignments = plan["assignments"]

            final_parameter_apply_plan = effective_material_plan
            if instance_root_count:
                final_parameter_apply_plan = (
                    destination / "part_id_parameter_tournament_instance_plan.json"
                )
                final_parameter_instance_document = copy.deepcopy(plan)
                final_parameter_provenance = final_parameter_instance_document.get(
                    "provenance"
                )
                sealed_final_parameter_provenance = (
                    dict(final_parameter_provenance)
                    if isinstance(final_parameter_provenance, dict)
                    else {}
                )
                sealed_final_parameter_provenance.update(
                    {
                        "asset_sha256": sha256_file(source),
                        "registry_sha256": canonical_sha256(rendered_registry_document),
                    }
                )
                final_parameter_instance_document[
                    "provenance"
                ] = sealed_final_parameter_provenance
                write_object(
                    final_parameter_apply_plan,
                    final_parameter_instance_document,
                )
                instance_plan = final_parameter_apply_plan

            final_parameter_apply_command = [
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
                str(final_parameter_apply_plan),
                "--output",
                str(look_usd),
                "--material-root",
                str(config.material_root),
                "--report",
                str(apply_report),
                "--include-review",
            ]
            if use_policy_fallback:
                final_parameter_apply_command.append("--include-policy-fallback")
            _run_stage(
                "part_id_parameter_tournament_final_apply",
                final_parameter_apply_command,
                log_cb,
                command_runner=_command_runner,
                retry_native_crash=True,
                required_files=(look_usd, apply_report),
            )
            _run_stage(
                "part_id_parameter_tournament_final_registry",
                [
                    str(isaac),
                    "-m",
                    "qwen_material_pipeline",
                    "usd",
                    "registry",
                    "--usd",
                    str(look_usd),
                    "--output",
                    str(quality_registry),
                ],
                log_cb,
                command_runner=_command_runner,
                retry_native_crash=True,
                required_files=(quality_registry,),
            )
            _run_stage(
                "part_id_parameter_tournament_final_render",
                [
                    str(isaac),
                    "-m",
                    "qwen_material_pipeline",
                    "usd",
                    "render",
                    "--registry",
                    str(quality_registry),
                    "--output-dir",
                    str(quality_render_dir),
                    "--resolution",
                    str(config.render_resolution),
                    *quality_render_view_arguments,
                    "--rt-subframes",
                    str(config.render_rt_subframes),
                    "--lighting-profile",
                    config.quality_lighting_profile,
                    "--analysis-up-axis",
                    config.analysis_up_axis,
                    f"--analysis-front-axis={config.analysis_front_axis}",
                ],
                log_cb,
                command_runner=_command_runner,
                retry_native_crash=True,
                required_files=(quality_rendered_registry,),
            )
            applied = read_object(
                apply_report,
                "post-parameter-tournament apply report",
            )
            applied_count = applied.get("applied_count")
            log_message(
                log_cb,
                "Part-ID parameter tournament completed: "
                f"H1 winners={parameter_tournament_document['h1_winner_count']}/"
                f"{parameter_tournament_document['candidate_part_count']}; "
                "every winner is locked by actual CAD render evidence.",
            )

    # Retrieval-bank previews cannot faithfully predict an MDL on the actual
    # CAD geometry: transmission, thickness and reflection can make a
    # thumbnail's apparent colour entirely wrong.  For the few photo-supported
    # appearance components whose authored H0 is visibly weak, make one small
    # fixed-MDL tournament on the registered CAD.  This remains a Part-ID
    # process (no palette groups), preserves all assignments outside the
    # component, and never writes colour/PBR parameters.
    component_actual_mdl_tournament_document: dict[str, Any] | None = None
    component_actual_tournament_inputs = (
        appearance_component_mdl_selection_audit,
        appearance_component_retrieval_result,
        appearance_component_qwen_result,
        part_id_evidence_path,
        spatial_report_path,
        quality_rendered_registry,
    )
    if (
        config.material_assignment_unit == "part_id"
        and config.material_prediction_mode == "disabled"
        and config.immutable_mdl_after_selection
        and all(path.is_file() for path in component_actual_tournament_inputs)
    ):
        component_selection_document = read_object(
            appearance_component_mdl_selection_audit,
            "appearance-component immutable MDL selections",
        )
        component_retrieval_document = read_object(
            appearance_component_retrieval_result,
            "appearance-component visual retrieval",
        )
        component_qwen_document = read_object(
            appearance_component_qwen_result,
            "appearance-component Qwen choices",
        )
        raw_selections = component_selection_document.get("selections")
        raw_retrieval_groups = component_retrieval_document.get("groups")
        raw_compatibility_parts = (
            component_qwen_document.get("visual_compatibility_gate", {}).get(
                "parts", []
            )
            if isinstance(
                component_qwen_document.get("visual_compatibility_gate"), Mapping
            )
            else []
        )
        if not isinstance(raw_selections, list) or not isinstance(
            raw_retrieval_groups, list
        ):
            raise RuntimeError(
                "Appearance-component immutable MDL tournament inputs are invalid"
            )
        retrieval_by_component = {
            row.get("group_id"): row
            for row in raw_retrieval_groups
            if isinstance(row, Mapping) and isinstance(row.get("group_id"), str)
        }
        compatibility_by_component = {
            row.get("part_id"): row
            for row in raw_compatibility_parts
            if isinstance(row, Mapping) and isinstance(row.get("part_id"), str)
        }
        evidence_for_component_tournament = read_object(
            part_id_evidence_path,
            "appearance-component actual-MDL evidence",
        )
        spatial_for_component_tournament = read_object(
            spatial_report_path,
            "appearance-component actual-MDL spatial registration",
        )
        baseline_component_rendered_registry = read_object(
            quality_rendered_registry,
            "appearance-component actual-MDL baseline render registry",
        )
        component_results: list[dict[str, Any]] = []
        final_component_plan_document = copy.deepcopy(plan)
        winner_count = 0
        candidate_count = 0
        # A component below this score materially limits the visual result;
        # tiny/decent components are already better served by the stable H0.
        minimum_component_pixels = 64
        minimum_component_appearance_score = 0.60
        for selection in sorted(
            (row for row in raw_selections if isinstance(row, Mapping)),
            key=lambda row: str(row.get("component_id", "")),
        ):
            component_id = selection.get("component_id")
            members = selection.get("member_part_ids")
            baseline_material_id = selection.get("material_id")
            if (
                not isinstance(component_id, str)
                or not component_id
                or not isinstance(members, list)
                or not isinstance(baseline_material_id, str)
                or not baseline_material_id.startswith("mdl:")
            ):
                raise RuntimeError(
                    "Appearance-component immutable MDL selection is malformed"
                )
            try:
                baseline_score = score_component_render(
                    component_id=component_id,
                    member_part_ids=members,
                    evidence=evidence_for_component_tournament,
                    spatial_mapping_report=spatial_for_component_tournament,
                    rendered_registry=baseline_component_rendered_registry,
                )
            except ComponentMdlTournamentError as exc:
                raise RuntimeError(
                    f"Unable to score appearance component {component_id}: {exc}"
                ) from exc
            baseline_pixels = int(baseline_score["comparison_pixel_count"])
            baseline_appearance = float(baseline_score["appearance_score"])
            result: dict[str, Any] = {
                "component_id": component_id,
                "member_part_ids": sorted(members),
                "baseline_material_id": baseline_material_id,
                "baseline_score": baseline_score,
                "candidate_scores": {baseline_material_id: baseline_score},
                "candidate_artifacts": [],
                "mdl_parameter_mutation_allowed": False,
            }
            if (
                baseline_pixels < minimum_component_pixels
                or baseline_appearance >= minimum_component_appearance_score
            ):
                result.update(
                    {
                        "selection_status": "BASELINE_NOT_WEAK_ENOUGH_FOR_TOURNAMENT",
                        "selected_material_id": baseline_material_id,
                        "reason": (
                            "INSUFFICIENT_REGISTERED_PIXELS"
                            if baseline_pixels < minimum_component_pixels
                            else "BASELINE_APPEARANCE_ABOVE_TOURNAMENT_FLOOR"
                        ),
                    }
                )
                component_results.append(result)
                continue
            retrieval_group = retrieval_by_component.get(component_id)
            if not isinstance(retrieval_group, Mapping):
                result.update(
                    {
                        "selection_status": "BASELINE_RETAINED_INPUT_UNAVAILABLE",
                        "selected_material_id": baseline_material_id,
                        "reason": "RETRIEVAL_GROUP_UNAVAILABLE",
                    }
                )
                component_results.append(result)
                continue
            try:
                material_ids = component_candidate_material_ids(
                    baseline_material_id=baseline_material_id,
                    retrieval_group=retrieval_group,
                    visual_compatibility=compatibility_by_component.get(component_id),
                    maximum_candidates=min(
                        4, config.exact_mdl_tournament_max_candidates
                    ),
                )
            except ComponentMdlTournamentError as exc:
                result.update(
                    {
                        "selection_status": "BASELINE_RETAINED_INPUT_UNAVAILABLE",
                        "selected_material_id": baseline_material_id,
                        "reason": f"CANDIDATE_SHORTLIST_INVALID: {exc}",
                    }
                )
                component_results.append(result)
                continue
            log_message(
                log_cb,
                "Appearance-component actual-CAD MDL tournament: "
                f"{component_id} candidates={len(material_ids)}",
            )
            for candidate_index, material_id in enumerate(material_ids[1:], start=2):
                candidate_count += 1
                candidate_suffix = canonical_sha256(
                    {"component_id": component_id, "material_id": material_id}
                )[:12]
                candidate_dir = (
                    appearance_component_actual_mdl_tournament_dir
                    / component_id
                    / f"C{candidate_index:02d}_{candidate_suffix}"
                )
                candidate_dir.mkdir(parents=True, exist_ok=True)
                candidate_plan_path = candidate_dir / "plan.json"
                candidate_apply_plan_path = candidate_plan_path
                candidate_look = candidate_dir / "look.usda"
                candidate_apply_report = candidate_dir / "apply_report.json"
                candidate_registry = candidate_dir / "part_registry.json"
                candidate_render_dir = candidate_dir / "renders"
                candidate_rendered_registry = (
                    candidate_render_dir / "part_registry.rendered.json"
                )
                try:
                    candidate_plan_document = build_component_candidate_plan(
                        source_plan=plan,
                        component_id=component_id,
                        member_part_ids=members,
                        material_id=material_id,
                    )
                except ComponentMdlTournamentError as exc:
                    raise RuntimeError(
                        f"Unable to build appearance-component candidate "
                        f"{component_id}/{material_id}: {exc}"
                    ) from exc
                if instance_root_count:
                    candidate_instance_plan_path = candidate_dir / "instance_plan.json"
                    candidate_provenance = candidate_plan_document.get("provenance")
                    sealed_candidate_provenance = (
                        dict(candidate_provenance)
                        if isinstance(candidate_provenance, Mapping)
                        else {}
                    )
                    sealed_candidate_provenance.update(
                        {
                            "asset_sha256": sha256_file(source),
                            "registry_sha256": canonical_sha256(
                                rendered_registry_document
                            ),
                        }
                    )
                    candidate_plan_document["provenance"] = sealed_candidate_provenance
                    write_object(candidate_instance_plan_path, candidate_plan_document)
                    candidate_apply_plan_path = candidate_instance_plan_path
                else:
                    write_object(candidate_plan_path, candidate_plan_document)
                candidate_apply_command = [
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
                    str(candidate_look),
                    "--material-root",
                    str(config.material_root),
                    "--report",
                    str(candidate_apply_report),
                    "--include-review",
                ]
                if use_policy_fallback:
                    candidate_apply_command.append("--include-policy-fallback")
                _run_stage(
                    f"appearance_component_{component_id}_mdl_{candidate_index}_apply",
                    candidate_apply_command,
                    log_cb,
                    command_runner=_command_runner,
                    retry_native_crash=True,
                    required_files=(candidate_look, candidate_apply_report),
                )
                _run_stage(
                    f"appearance_component_{component_id}_mdl_{candidate_index}_registry",
                    [
                        str(isaac),
                        "-m",
                        "qwen_material_pipeline",
                        "usd",
                        "registry",
                        "--usd",
                        str(candidate_look),
                        "--output",
                        str(candidate_registry),
                    ],
                    log_cb,
                    command_runner=_command_runner,
                    retry_native_crash=True,
                    required_files=(candidate_registry,),
                )
                _run_stage(
                    f"appearance_component_{component_id}_mdl_{candidate_index}_render",
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
                        str(config.render_resolution),
                        *quality_render_view_arguments,
                        "--rt-subframes",
                        str(config.render_rt_subframes),
                        "--lighting-profile",
                        config.quality_lighting_profile,
                        "--analysis-up-axis",
                        config.analysis_up_axis,
                        f"--analysis-front-axis={config.analysis_front_axis}",
                    ],
                    log_cb,
                    command_runner=_command_runner,
                    retry_native_crash=True,
                    required_files=(candidate_rendered_registry,),
                )
                try:
                    candidate_score = score_component_render(
                        component_id=component_id,
                        member_part_ids=members,
                        evidence=evidence_for_component_tournament,
                        spatial_mapping_report=spatial_for_component_tournament,
                        rendered_registry=read_object(
                            candidate_rendered_registry,
                            f"appearance-component {component_id} candidate render",
                        ),
                    )
                except ComponentMdlTournamentError as exc:
                    raise RuntimeError(
                        f"Unable to score appearance-component candidate "
                        f"{component_id}/{material_id}: {exc}"
                    ) from exc
                result["candidate_scores"][material_id] = candidate_score
                result["candidate_artifacts"].append(
                    {
                        "material_id": material_id,
                        "plan": str(candidate_plan_path),
                        "look_usd": str(candidate_look),
                        "rendered_registry": str(candidate_rendered_registry),
                    }
                )
            try:
                winner = select_component_mdl_winner(
                    component_id=component_id,
                    baseline_material_id=baseline_material_id,
                    candidate_scores=result["candidate_scores"],
                    minimum_score_improvement=(
                        config.exact_mdl_tournament_minimum_score_improvement
                    ),
                )
            except ComponentMdlTournamentError as exc:
                raise RuntimeError(
                    f"Appearance-component actual-MDL tournament failed for "
                    f"{component_id}: {exc}"
                ) from exc
            result.update(winner)
            selected_material_id = winner["selected_material_id"]
            if selected_material_id != baseline_material_id:
                final_component_plan_document = build_component_candidate_plan(
                    source_plan=final_component_plan_document,
                    component_id=component_id,
                    member_part_ids=members,
                    material_id=selected_material_id,
                )
                winner_count += 1
            component_results.append(result)

        component_actual_mdl_tournament_document = {
            "schema_version": "qwen-appearance-component-actual-mdl-tournament/v1",
            "assignment_unit": "part_id",
            "selection_authority": "registered_actual_cad_render",
            "mdl_parameter_mutation_allowed": False,
            "minimum_component_pixels": minimum_component_pixels,
            "minimum_component_appearance_score": (minimum_component_appearance_score),
            "maximum_candidates_per_component": min(
                4, config.exact_mdl_tournament_max_candidates
            ),
            "minimum_score_improvement": (
                config.exact_mdl_tournament_minimum_score_improvement
            ),
            "candidate_count": candidate_count,
            "winner_count": winner_count,
            "components": component_results,
        }
        write_object(
            appearance_component_actual_mdl_tournament_audit,
            component_actual_mdl_tournament_document,
        )
        if winner_count:
            write_object(
                appearance_component_actual_mdl_tournament_plan,
                final_component_plan_document,
            )
            try:
                rebound_component_part_id_audit = (
                    rebind_part_id_material_audit_for_component_mdl_tournament(
                        source_audit=read_object(
                            part_id_material_audit,
                            "pre-component-tournament Part-ID material audit",
                        ),
                        final_plan=final_component_plan_document,
                        tournament_audit=component_actual_mdl_tournament_document,
                    )
                )
            except ComponentMdlTournamentError as exc:
                raise RuntimeError(
                    "Unable to bind the final immutable component-MDL plan to "
                    f"its Part-ID audit: {exc}"
                ) from exc
            write_object(part_id_material_audit, rebound_component_part_id_audit)
            effective_material_plan = appearance_component_actual_mdl_tournament_plan
            plan = final_component_plan_document
            assignments = plan["assignments"]
            final_component_apply_plan = effective_material_plan
            if instance_root_count:
                final_component_apply_plan = (
                    destination
                    / "appearance_component_actual_mdl_tournament_instance_plan.json"
                )
                final_component_instance_document = copy.deepcopy(plan)
                final_component_provenance = final_component_instance_document.get(
                    "provenance"
                )
                sealed_final_component_provenance = (
                    dict(final_component_provenance)
                    if isinstance(final_component_provenance, Mapping)
                    else {}
                )
                sealed_final_component_provenance.update(
                    {
                        "asset_sha256": sha256_file(source),
                        "registry_sha256": canonical_sha256(rendered_registry_document),
                    }
                )
                final_component_instance_document[
                    "provenance"
                ] = sealed_final_component_provenance
                write_object(
                    final_component_apply_plan,
                    final_component_instance_document,
                )
                instance_plan = final_component_apply_plan
            final_component_apply_command = [
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
                str(final_component_apply_plan),
                "--output",
                str(look_usd),
                "--material-root",
                str(config.material_root),
                "--report",
                str(apply_report),
                "--include-review",
            ]
            if use_policy_fallback:
                final_component_apply_command.append("--include-policy-fallback")
            _run_stage(
                "appearance_component_actual_mdl_tournament_final_apply",
                final_component_apply_command,
                log_cb,
                command_runner=_command_runner,
                retry_native_crash=True,
                required_files=(look_usd, apply_report),
            )
            _run_stage(
                "appearance_component_actual_mdl_tournament_final_registry",
                [
                    str(isaac),
                    "-m",
                    "qwen_material_pipeline",
                    "usd",
                    "registry",
                    "--usd",
                    str(look_usd),
                    "--output",
                    str(quality_registry),
                ],
                log_cb,
                command_runner=_command_runner,
                retry_native_crash=True,
                required_files=(quality_registry,),
            )
            _run_stage(
                "appearance_component_actual_mdl_tournament_final_render",
                [
                    str(isaac),
                    "-m",
                    "qwen_material_pipeline",
                    "usd",
                    "render",
                    "--registry",
                    str(quality_registry),
                    "--output-dir",
                    str(quality_render_dir),
                    "--resolution",
                    str(config.render_resolution),
                    *quality_render_view_arguments,
                    "--rt-subframes",
                    str(config.render_rt_subframes),
                    "--lighting-profile",
                    config.quality_lighting_profile,
                    "--analysis-up-axis",
                    config.analysis_up_axis,
                    f"--analysis-front-axis={config.analysis_front_axis}",
                ],
                log_cb,
                command_runner=_command_runner,
                retry_native_crash=True,
                required_files=(quality_rendered_registry,),
            )
            final_component_apply_document = read_object(
                apply_report,
                "post-appearance-component actual-MDL tournament apply report",
            )
            final_component_applied_count = final_component_apply_document.get(
                "applied_count"
            )
            if final_component_applied_count != applied_count:
                raise RuntimeError(
                    "Appearance-component actual-MDL tournament changed exact "
                    "Part-ID coverage"
                )
            log_message(
                log_cb,
                "Appearance-component actual-CAD MDL tournament completed: "
                f"winners={winner_count}; every winner keeps the library-default "
                "MDL parameters unchanged.",
            )

    trusted_mapping: dict[str, str] = {}
    if spatial_report_path.is_file():
        spatial_report = read_object(spatial_report_path, "spatial mapping report")
        raw_alignments = spatial_report.get("view_alignments", [])
        if isinstance(raw_alignments, list):
            for alignment in raw_alignments:
                if (
                    not isinstance(alignment, dict)
                    or alignment.get("trusted") is not True
                ):
                    continue
                reference_id = alignment.get("reference_view_id")
                render_id = alignment.get("selected_render_view_id")
                if (
                    isinstance(reference_id, str)
                    and reference_id
                    and isinstance(render_id, str)
                    and render_id
                    and reference_id not in trusted_mapping
                    and render_id not in trusted_mapping.values()
                ):
                    trusted_mapping[reference_id] = render_id
    compare_command = [
        str(config.qwen_python),
        "-m",
        "qwen_material_pipeline",
        "compare",
        "--reference-manifest",
        str(analysis_dir / "reference_manifest.json"),
        "--rendered-registry",
        str(quality_rendered_registry),
        "--output",
        str(quality_report),
        "--minimum-comparable-views",
        "2",
    ]
    if len(trusted_mapping) >= 2:
        write_object(
            quality_view_map,
            {
                "schema_version": "qwen-reference-view-map/v1",
                "mapping": dict(sorted(trusted_mapping.items())),
                "source": "trusted_spatial_registration",
            },
        )
        compare_command.extend(["--view-map", str(quality_view_map)])
    _run_stage(
        "quality_compare",
        compare_command,
        log_cb,
        command_runner=_command_runner,
    )
    _require_file(quality_report, "quality_compare")
    quality = read_object(quality_report, "visual quality report")
    if (
        config.material_assignment_unit == "part_id"
        and workspace.source.camera_acceptance.is_file()
    ):
        try:
            quality[
                "part_id_quality_scope"
            ] = _part_id_quality_scope_from_camera_alignment(
                read_object(
                    workspace.source.camera_acceptance,
                    "camera alignment acceptance",
                )
            )
        except ValueError as exc:
            raise RuntimeError(
                "Camera alignment acceptance cannot define the Part-ID final-QA "
                f"view scope: {exc}"
            ) from exc
        # The scope is part of the evidence contract and must be covered by
        # the quality-report hash stored in part_id_quality_gate.json.
        write_object(quality_report, quality)
    aggregate = quality.get("aggregate")
    if not isinstance(aggregate, dict):
        raise RuntimeError("Visual quality report is missing its aggregate decision")
    raw_quality_status = aggregate.get("status")
    quality_status = raw_quality_status
    comparable_view_count = aggregate.get("comparable_view_count")
    if quality_status not in QUALITY_STATUSES:
        raise RuntimeError(
            f"Visual quality report has an unsupported status: {quality_status!r}"
        )
    if (
        isinstance(comparable_view_count, bool)
        or not isinstance(comparable_view_count, int)
        or comparable_view_count < 2
    ):
        raise RuntimeError(
            "Automatic visual QA rejected the authored Look "
            f"(status={quality_status!r}, "
            f"comparable_views={comparable_view_count!r}); physics was not started"
        )

    part_id_quality_gate_document: dict[str, Any] | None = None
    if config.material_assignment_unit == "part_id":
        part_id_quality_gate_document = _evaluate_part_id_quality_gate(
            quality,
            minimum_aggregate_appearance_score=(
                config.final_visual_gate_minimum_final_appearance_score
            ),
            minimum_view_appearance_score=(
                config.final_visual_gate_minimum_final_view_appearance_score
            ),
            minimum_comparable_views=2,
            coating_consistency_audit=read_object(
                part_id_material_audit,
                "Part-ID coating consistency audit",
            ),
        )
        part_id_quality_gate_document["bindings"] = {
            "quality_report": str(quality_report),
            "quality_report_sha256": sha256_file(quality_report),
            "material_plan": str(effective_material_plan),
            "material_plan_sha256": sha256_file(effective_material_plan),
            "apply_report": str(apply_report),
            "apply_report_sha256": sha256_file(apply_report),
            "applied_count": applied_count,
        }
        write_object(part_id_quality_gate, part_id_quality_gate_document)
        if part_id_quality_gate_document["acceptance_allowed"] is True:
            quality_status = "PASS"
            log_message(
                log_cb,
                "Independent Part-ID visual QA passed: whole-asset and every "
                "comparable-view appearance floor passed; palette-group "
                "completeness reasons were retained in the raw report but are "
                "not applicable to Part-ID assignment.",
            )

    effective_look_usd = look_usd
    effective_apply_report = apply_report
    effective_quality_report = quality_report
    effective_quality_rendered_registry = quality_rendered_registry
    quality_repair_used = False
    quality_repair_changed_count = 0
    quality_round_count = 1
    quality_gate_status = quality_status
    quality_resolution_document: dict[str, Any] | None = None
    quality_limitation_count = 0
    final_instance_plan = instance_plan
    if quality_status == "FAIL":
        if config.material_assignment_unit == "part_id":
            raise RuntimeError(
                "Independent Part-ID visual QA failed its appearance/evidence "
                "floors. Palette-group repair is disabled by contract; inspect "
                "part_id_quality_gate.json and part_id_material_audit.json "
                "before retrying."
            )
        if not use_policy_fallback:
            raise RuntimeError(
                "Automatic visual QA failed and no exact-cover policy baseline "
                "is available for bounded repair; physics was not started"
            )
        repair_inputs = {
            "palette_fusion": analysis_dir / "palette_fusion.json",
            "spatial_report": analysis_dir / "spatial_mapping_report.json",
            "spatial_gate_audit": analysis_dir / "spatial_mapping_audit.json",
            "mapping_consensus": (analysis_dir / "part_mapping_multiview_audit.json"),
            "geometry_risk": (analysis_dir / "geometry_uniform_material_risk.json"),
            "mvinverse_pbr_evidence": mvinverse_pbr_evidence,
        }
        for label, path in repair_inputs.items():
            _require_file(path, f"quality_repair_{label}")
        repair_command = [
            str(config.qwen_python),
            "-m",
            "qwen_material_pipeline",
            "quality-repair-plan",
            "--baseline-plan",
            str(policy_plan),
            "--baseline-policy-audit",
            str(policy_audit),
            "--quality-report",
            str(quality_report),
            "--palette-fusion",
            str(repair_inputs["palette_fusion"]),
            "--spatial-report",
            str(repair_inputs["spatial_report"]),
            "--spatial-gate-audit",
            str(repair_inputs["spatial_gate_audit"]),
            "--mapping-consensus",
            str(repair_inputs["mapping_consensus"]),
            "--geometry-risk",
            str(repair_inputs["geometry_risk"]),
            "--group-materials",
            str(group_materials),
            "--mvinverse-pbr-evidence",
            str(repair_inputs["mvinverse_pbr_evidence"]),
            "--registry",
            str(rendered_registry),
            "--whitelist",
            str(effective_whitelist),
            "--output-plan",
            str(quality_repair_plan),
            "--audit",
            str(quality_repair_audit),
            "--material-selection-objective",
            config.material_selection_objective,
        ]
        if config.immutable_mdl_after_selection:
            repair_command.append("--immutable-mdl-after-selection")
        repair_checkpoint_reused = False
        if completed_inference_resume and (
            quality_repair_plan.exists() or quality_repair_audit.exists()
        ):
            repair_checkpoint_valid = False
            if quality_repair_plan.is_file() and quality_repair_audit.is_file():
                try:
                    checkpoint_plan = read_object(
                        quality_repair_plan,
                        "quality-repair resume plan",
                    )
                    checkpoint_audit = read_object(
                        quality_repair_audit,
                        "quality-repair resume audit",
                    )
                    expected_checkpoint_hashes = {
                        "baseline_plan_sha256": canonical_sha256(policy_plan_document),
                        "baseline_policy_audit_sha256": canonical_sha256(
                            policy_audit_document
                        ),
                        "quality_report_sha256": canonical_sha256(quality),
                        "palette_fusion_sha256": canonical_sha256(
                            read_object(
                                repair_inputs["palette_fusion"],
                                "palette fusion",
                            )
                        ),
                        "spatial_report_sha256": canonical_sha256(
                            read_object(
                                repair_inputs["spatial_report"],
                                "spatial mapping report",
                            )
                        ),
                        "spatial_gate_audit_sha256": canonical_sha256(
                            read_object(
                                repair_inputs["spatial_gate_audit"],
                                "spatial gate audit",
                            )
                        ),
                        "mapping_consensus_sha256": canonical_sha256(
                            read_object(
                                repair_inputs["mapping_consensus"],
                                "mapping consensus",
                            )
                        ),
                        "geometry_risk_sha256": canonical_sha256(
                            read_object(
                                repair_inputs["geometry_risk"],
                                "geometry risk",
                            )
                        ),
                        "group_materials_sha256": canonical_sha256(
                            read_object(group_materials, "group materials")
                        ),
                        "mvinverse_pbr_evidence_sha256": canonical_sha256(
                            read_object(
                                repair_inputs["mvinverse_pbr_evidence"],
                                "MVInverse PBR evidence",
                            )
                        ),
                        "registry_sha256": canonical_sha256(
                            read_object(rendered_registry, "rendered registry")
                        ),
                        "whitelist_sha256": canonical_sha256(
                            read_object(
                                effective_whitelist,
                                "material whitelist",
                            )
                        ),
                    }
                    if (
                        config.material_selection_objective
                        == MATERIAL_SELECTION_OBJECTIVE_VISUAL
                    ):
                        expected_checkpoint_hashes[
                            "material_selection_objective"
                        ] = config.material_selection_objective
                    repair_checkpoint_valid = bool(
                        checkpoint_audit.get("schema_version")
                        == QUALITY_REPAIR_REPORT_SCHEMA_VERSION
                        and checkpoint_audit.get("input_hashes")
                        == expected_checkpoint_hashes
                        and checkpoint_audit.get("output_plan_sha256")
                        == canonical_sha256(checkpoint_plan)
                    )
                except (OSError, RuntimeError, ValueError):
                    repair_checkpoint_valid = False
            if repair_checkpoint_valid:
                repair_checkpoint_reused = True
                log_message(
                    log_cb,
                    "Reusing the hash-verified quality-repair checkpoint.",
                )
            else:
                repair_archive_root = analysis_dir / "recovery_archive"
                repair_archive_root.mkdir(parents=True, exist_ok=True)
                repair_archive = unique_path(
                    repair_archive_root / "stale_quality_repair"
                )
                repair_archive.mkdir(parents=False, exist_ok=False)
                for stale_path in (
                    quality_repair_plan,
                    quality_repair_audit,
                ):
                    if stale_path.exists():
                        stale_path.rename(repair_archive / stale_path.name)
                log_message(
                    log_cb,
                    "Archived a stale quality-repair checkpoint before "
                    f"recompilation: {repair_archive}",
                )
        if not repair_checkpoint_reused:
            _run_stage(
                "quality_repair_compile",
                repair_command,
                log_cb,
                command_runner=_command_runner,
                # This compiler is deterministic, but its isolated runtime
                # still imports native image/ML libraries.  A process-level
                # SIGSEGV/SIGABRT is transient infrastructure evidence, not a
                # material-plan rejection, so give it the same bounded clean
                # process recovery used by the Isaac subprocesses.  Python
                # tracebacks and ordinary non-zero exits remain fail-closed.
                retry_native_crash=True,
            )
        _require_file(quality_repair_plan, "quality_repair_compile")
        _require_file(quality_repair_audit, "quality_repair_compile")
        repair_plan_document = read_object(quality_repair_plan, "quality-repair plan")
        repair_audit_document = read_object(
            quality_repair_audit, "quality-repair audit"
        )
        repair_registry_document = read_object(rendered_registry, "rendered registry")
        quality_repair_changed_count = _validate_quality_repair_bundle(
            plan=repair_plan_document,
            audit=repair_audit_document,
            baseline_plan=policy_plan_document,
            baseline_policy_audit=policy_audit_document,
            quality_report=quality,
            palette_fusion=read_object(
                repair_inputs["palette_fusion"], "palette fusion"
            ),
            spatial_report=read_object(
                repair_inputs["spatial_report"], "spatial mapping report"
            ),
            spatial_gate_audit=read_object(
                repair_inputs["spatial_gate_audit"], "spatial gate audit"
            ),
            mapping_consensus=read_object(
                repair_inputs["mapping_consensus"], "mapping consensus"
            ),
            geometry_risk=read_object(repair_inputs["geometry_risk"], "geometry risk"),
            group_materials=read_object(group_materials, "group materials"),
            mvinverse_pbr_evidence=read_object(
                repair_inputs["mvinverse_pbr_evidence"],
                "MVInverse PBR evidence",
            ),
            registry=repair_registry_document,
            whitelist=read_object(effective_whitelist, "material whitelist"),
            material_selection_objective=(config.material_selection_objective),
        )
        quality_repair_was_noop = quality_repair_changed_count == 0
        if quality_repair_was_noop:
            # SAFE_NOOP is an expected, sealed outcome when QA identifies a
            # deficit but no part can be changed without inventing evidence.
            # Keep the already-rendered baseline and let the immutable exact
            # MDL tournament resolve visual choice; do not waste another Isaac
            # apply/render round on a byte-identical plan.
            repaired_look_usd = look_usd
            repaired_apply_report = apply_report
            repaired_quality_rendered_registry = quality_rendered_registry
            repaired_quality_report = quality_report
            log_message(
                log_cb,
                "Quality repair returned hash-verified SAFE_NOOP; reusing the "
                "baseline Look and render for the immutable MDL tournament.",
            )

        repair_apply_plan = quality_repair_plan
        if instance_root_count:
            repaired_instance_document = dict(repair_plan_document)
            repaired_provenance = repaired_instance_document.get("provenance")
            sealed_repaired_provenance = (
                dict(repaired_provenance)
                if isinstance(repaired_provenance, dict)
                else {}
            )
            sealed_repaired_provenance.update(
                {
                    "asset_sha256": sha256_file(source),
                    "registry_sha256": canonical_sha256(repair_registry_document),
                }
            )
            repaired_instance_document["provenance"] = sealed_repaired_provenance
            write_object(repaired_instance_plan, repaired_instance_document)
            repair_apply_plan = repaired_instance_plan
            final_instance_plan = repaired_instance_plan

        repair_apply_command = [
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
            str(repair_apply_plan),
            "--output",
            str(repaired_look_usd),
            "--material-root",
            str(config.material_root),
            "--report",
            str(repaired_apply_report),
            "--include-review",
            "--include-policy-fallback",
        ]
        if not quality_repair_was_noop:
            _run_stage(
                "quality_repair_apply_usd",
                repair_apply_command,
                log_cb,
                command_runner=_command_runner,
                retry_native_crash=True,
            )
        _require_file(repaired_look_usd, "quality_repair_apply_usd")
        _require_file(repaired_apply_report, "quality_repair_apply_usd")
        repaired_applied = read_object(
            repaired_apply_report, "quality-repair apply report"
        )
        repaired_applied_count = repaired_applied.get("applied_count")
        if (
            isinstance(repaired_applied_count, bool)
            or not isinstance(repaired_applied_count, int)
            or repaired_applied_count != applied_count
        ):
            raise RuntimeError(
                "Quality repair changed the applied part count: "
                f"baseline={applied_count} repaired={repaired_applied_count!r}"
            )

        if not quality_repair_was_noop:
            _run_stage(
                "quality_repair_registry",
                [
                    str(isaac),
                    "-m",
                    "qwen_material_pipeline",
                    "usd",
                    "registry",
                    "--usd",
                    str(repaired_look_usd),
                    "--output",
                    str(repaired_quality_registry),
                ],
                log_cb,
                command_runner=_command_runner,
                retry_native_crash=True,
            )
            _require_file(repaired_quality_registry, "quality_repair_registry")
            _run_stage(
                "quality_repair_render",
                [
                    str(isaac),
                    "-m",
                    "qwen_material_pipeline",
                    "usd",
                    "render",
                    "--registry",
                    str(repaired_quality_registry),
                    "--output-dir",
                    str(repaired_quality_render_dir),
                    "--resolution",
                    str(config.render_resolution),
                    *quality_render_view_arguments,
                    "--rt-subframes",
                    str(config.render_rt_subframes),
                    "--lighting-profile",
                    config.quality_lighting_profile,
                    "--analysis-up-axis",
                    config.analysis_up_axis,
                    f"--analysis-front-axis={config.analysis_front_axis}",
                ],
                log_cb,
                command_runner=_command_runner,
                retry_native_crash=True,
            )
        _require_file(repaired_quality_rendered_registry, "quality_repair_render")
        repaired_compare_command = [
            str(config.qwen_python),
            "-m",
            "qwen_material_pipeline",
            "compare",
            "--reference-manifest",
            str(analysis_dir / "reference_manifest.json"),
            "--rendered-registry",
            str(repaired_quality_rendered_registry),
            "--output",
            str(repaired_quality_report),
            "--minimum-comparable-views",
            "2",
        ]
        if len(trusted_mapping) >= 2:
            write_object(
                repaired_quality_view_map,
                {
                    "schema_version": "qwen-reference-view-map/v1",
                    "mapping": dict(sorted(trusted_mapping.items())),
                    "source": "trusted_spatial_registration",
                },
            )
            repaired_compare_command.extend(
                ["--view-map", str(repaired_quality_view_map)]
            )
        if not quality_repair_was_noop:
            _run_stage(
                "quality_repair_compare",
                repaired_compare_command,
                log_cb,
                command_runner=_command_runner,
            )
        _require_file(repaired_quality_report, "quality_repair_compare")
        repaired_quality = read_object(
            repaired_quality_report, "repaired visual quality report"
        )
        repaired_quality_registry_document = read_object(
            repaired_quality_rendered_registry,
            "repaired rendered registry",
        )
        palette_fusion_document = read_object(
            repair_inputs["palette_fusion"], "palette fusion"
        )
        spatial_report_document = read_object(
            repair_inputs["spatial_report"], "spatial mapping report"
        )
        geometry_risk_document = read_object(
            repair_inputs["geometry_risk"], "geometry risk"
        )
        _validate_quality_render_contract(
            quality_report=repaired_quality,
            rendered_registry=repaired_quality_registry_document,
            rendered_registry_path=repaired_quality_rendered_registry,
            spatial_report=spatial_report_document,
        )
        _run_stage(
            "quality_resolution",
            [
                str(config.qwen_python),
                "-m",
                "qwen_material_pipeline",
                "quality-resolution",
                "--final-plan",
                str(quality_repair_plan),
                "--policy-audit",
                str(policy_audit),
                "--quality-report",
                str(repaired_quality_report),
                "--palette-fusion",
                str(repair_inputs["palette_fusion"]),
                "--spatial-report",
                str(repair_inputs["spatial_report"]),
                "--geometry-risk",
                str(repair_inputs["geometry_risk"]),
                "--rendered-registry",
                str(repaired_quality_rendered_registry),
                "--output",
                str(quality_resolution),
            ],
            log_cb,
            command_runner=_command_runner,
        )
        _require_file(quality_resolution, "quality_resolution")
        quality_resolution_document = read_object(
            quality_resolution, "visual-quality resolution"
        )
        resolution_status = _validate_quality_resolution_bundle(
            resolution=quality_resolution_document,
            final_plan=repair_plan_document,
            policy_audit=policy_audit_document,
            quality_report=repaired_quality,
            palette_fusion=palette_fusion_document,
            spatial_report=spatial_report_document,
            geometry_risk=geometry_risk_document,
            rendered_registry=repaired_quality_registry_document,
        )
        if (
            resolution_status == QUALITY_RESOLUTION_FAIL_CLOSED
            and not config.immutable_mdl_after_selection
        ):
            raise RuntimeError(
                "Automatic material QA remained fail-closed after bounded repair: "
                f"{quality_resolution_document.get('reason_codes', [])}; "
                "physics was not started"
            )
        if not quality_repair_was_noop:
            _validate_quality_repair_outcome(
                baseline_quality=quality,
                repaired_quality=repaired_quality,
                repair_audit=repair_audit_document,
                allow_verified_pose_limitation=(
                    resolution_status == QUALITY_RESOLUTION_LIMITED_PASS
                ),
                allow_pending_immutable_tournament=(
                    config.immutable_mdl_after_selection
                    and resolution_status == QUALITY_RESOLUTION_FAIL_CLOSED
                ),
            )
        quality_repair_used = not quality_repair_was_noop
        quality_round_count = 1 if quality_repair_was_noop else 2
        repaired_aggregate = repaired_quality.get("aggregate")
        if not isinstance(repaired_aggregate, dict):
            raise RuntimeError("Repaired visual quality report lacks its aggregate")
        quality_status = repaired_aggregate.get("status")
        quality_gate_status = resolution_status
        quality_limitation_count = len(
            quality_resolution_document.get("limitations", [])
        )
        if resolution_status == QUALITY_RESOLUTION_FAIL_CLOSED:
            quality_decision = "PENDING_IMMUTABLE_EXACT_MDL_RENDER_TOURNAMENT"
            log_message(
                log_cb,
                "Bounded repair improved the Look but remained REVIEW; "
                "continuing to the immutable exact NVIDIA MDL tournament.",
            )
        else:
            quality_decision = (
                "ACCEPTED_AFTER_BOUNDED_QA_REPAIR_WITH_GEOMETRY_POSE_LIMITATION"
                if resolution_status == QUALITY_RESOLUTION_LIMITED_PASS
                else "ACCEPTED_AFTER_BOUNDED_QA_REPAIR"
            )
        effective_material_plan = (
            policy_plan if quality_repair_was_noop else quality_repair_plan
        )
        effective_look_usd = repaired_look_usd
        effective_apply_report = repaired_apply_report
        effective_quality_report = repaired_quality_report
        effective_quality_rendered_registry = repaired_quality_rendered_registry
        applied_count = repaired_applied_count
    elif quality_status == "INSUFFICIENT_EVIDENCE":
        raise RuntimeError(
            "Automatic visual QA has insufficient evidence; bounded repair "
            "cannot infer a safe target and physics was not started"
        )
    elif quality_status == "REVIEW":
        raise RuntimeError(
            "Automatic visual QA is inconclusive (REVIEW); unattended mode "
            "accepts only PASS and physics was not started"
        )
    elif quality_status == "PASS":
        quality_decision = "ACCEPTED"
    else:  # QUALITY_STATUSES is validated above; keep this branch fail-closed.
        raise RuntimeError(f"Unsupported automatic visual QA status: {quality_status}")

    if config.material_assignment_unit == "part_id":
        membership_tournament_result = {
            "status": "NOT_REQUIRED_PART_ID_ASSIGNMENT",
            "cohort_count": 0,
            "selected_expanded_cohort_count": 0,
            "restored_m0_count": 0,
        }
    else:
        membership_tournament_result = _run_dominant_assembly_membership_tournaments(
            source_plan_path=effective_material_plan,
            source=source,
            apply_asset=Path(apply_asset),
            apply_subcommand=apply_subcommand,
            apply_asset_flag=apply_asset_flag,
            effective_catalog=effective_catalog,
            material_root=config.material_root,
            rendered_registry=rendered_registry,
            current_look_usd=effective_look_usd,
            current_apply_report=effective_apply_report,
            current_quality_report=effective_quality_report,
            current_quality_rendered_registry=(effective_quality_rendered_registry),
            reference_manifest=analysis_dir / "reference_manifest.json",
            palette_fusion_path=analysis_dir / "palette_fusion.json",
            tournament_dir=membership_tournament_dir,
            tournament_view_map=membership_tournament_view_map,
            output_plan=membership_tournament_plan,
            output_audit=membership_tournament_audit,
            trusted_mapping=trusted_mapping,
            mapped_render_resolution=config.render_resolution,
            render_rt_subframes=config.render_rt_subframes,
            analysis_up_axis=config.analysis_up_axis,
            analysis_front_axis=config.analysis_front_axis,
            qwen_python=config.qwen_python,
            isaac=isaac,
            instance_root_count=instance_root_count,
            applied_count=applied_count,
            include_policy_fallback=use_policy_fallback,
            log_cb=log_cb,
            command_runner=_command_runner,
        )
    membership_tournament_status = str(membership_tournament_result["status"])
    membership_tournament_cohort_count = int(
        membership_tournament_result["cohort_count"]
    )
    membership_selected_expanded_count = int(
        membership_tournament_result["selected_expanded_cohort_count"]
    )
    membership_restored_m0_count = int(
        membership_tournament_result["restored_m0_count"]
    )
    if membership_tournament_status == "COMPLETED":
        effective_material_plan = Path(membership_tournament_result["plan"])
        effective_look_usd = Path(membership_tournament_result["look"])
        effective_apply_report = Path(membership_tournament_result["apply"])
        effective_quality_report = Path(membership_tournament_result["quality"])
        effective_quality_rendered_registry = Path(
            membership_tournament_result["rendered_registry"]
        )
        applied_count = int(membership_tournament_result["applied_count"])
        if instance_root_count:
            membership_instance_document = read_object(
                effective_material_plan,
                "selected dominant assembly membership plan",
            )
            raw_membership_provenance = membership_instance_document.get("provenance")
            sealed_membership_provenance = (
                dict(raw_membership_provenance)
                if isinstance(raw_membership_provenance, Mapping)
                else {}
            )
            sealed_membership_provenance.update(
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
            membership_instance_document["provenance"] = sealed_membership_provenance
            write_object(
                membership_tournament_instance_plan,
                membership_instance_document,
            )
            final_instance_plan = membership_tournament_instance_plan

    quality_repair_round_count = quality_round_count

    if not isinstance(assignments, list):
        raise RuntimeError("Visual QA returned invalid material assignments")
    return VisualQaStageResult(
        quality_render_view_arguments=quality_render_view_arguments,
        spatial_report_path=spatial_report_path,
        parameter_tournament_document=parameter_tournament_document,
        component_actual_mdl_tournament_document=(
            component_actual_mdl_tournament_document
        ),
        corresponding_color_calibration_document=(
            corresponding_color_calibration_document
        ),
        trusted_mapping=trusted_mapping,
        raw_quality_status=raw_quality_status,
        quality_status=quality_status,
        part_id_quality_gate_document=part_id_quality_gate_document,
        effective_look_usd=effective_look_usd,
        effective_apply_report=effective_apply_report,
        effective_quality_report=effective_quality_report,
        effective_quality_rendered_registry=(effective_quality_rendered_registry),
        quality_repair_used=quality_repair_used,
        quality_repair_changed_count=quality_repair_changed_count,
        quality_round_count=quality_round_count,
        quality_repair_round_count=quality_repair_round_count,
        quality_gate_status=quality_gate_status,
        quality_resolution_document=quality_resolution_document,
        quality_limitation_count=quality_limitation_count,
        final_instance_plan=final_instance_plan,
        effective_material_plan=effective_material_plan,
        assignments=assignments,
        applied_count=applied_count,
        quality_decision=quality_decision,
        membership_tournament_status=membership_tournament_status,
        membership_tournament_cohort_count=membership_tournament_cohort_count,
        membership_selected_expanded_count=membership_selected_expanded_count,
        membership_restored_m0_count=membership_restored_m0_count,
    )


@dataclass(frozen=True)
class MaterialSelectionStageResult:
    exact_mdl_tournament_status: str
    exact_mdl_tournament_candidate_count: int
    exact_mdl_tournament_selected_candidate_id: str | None
    exact_mdl_tournament_selected_candidate_ids: list[str]
    exact_mdl_tournament_group_count: int
    render_confirmed_disagreement_group_ids: set[str]
    baseline_preserved_disagreement_exemptions: list[dict[str, Any]]
    visual_group_annotation_status: str
    quality_round_count: int
    final_instance_plan: Path | None
    effective_material_plan: Path
    effective_look_usd: Path
    effective_apply_report: Path
    effective_quality_report: Path
    effective_quality_rendered_registry: Path
    applied_count: int
    quality_status: str
    quality_gate_status: str
    quality_resolution_document: dict[str, Any] | None
    quality_limitation_count: int
    quality_decision: str
    appearance_status: str
    appearance_adjustment_count: int
    appearance_changed_count: int
    appearance_candidate_round_count: int
    appearance_error: str | None
    appearance_reason_codes: list[str]
    appearance_measurement_used: bool
    baseline_lighting_profile: str | None


def _run_material_selection_stage(
    context: VisualMaterialPipelineContext,
    *,
    prepared_source: SourcePreparationResult,
    planning: PolicyPartIdStageResult,
    look: LookApplicationStageResult,
    visual_qa: VisualQaStageResult,
    log_cb: LogCallback,
    command_runner: CommandRunner,
) -> MaterialSelectionStageResult:
    """Resolve exact MDLs and the optional bounded appearance candidate."""

    source = context.source
    config = context.config
    isaac = context.isaac_python
    destination = context.destination
    workspace = context.workspace
    rendered_registry = prepared_source.rendered_registry
    instance_root_count = prepared_source.instance_root_count
    rendered_registry_document = look.rendered_registry_document
    apply_subcommand = look.apply_subcommand
    apply_asset_flag = look.apply_asset_flag
    apply_asset = look.apply_asset
    effective_catalog = planning.effective_catalog
    use_policy_fallback = planning.use_policy_fallback
    analysis_dir = workspace.inference.root
    mvinverse_pbr_evidence = workspace.inference.mvinverse_pbr_evidence
    effective_material_plan = visual_qa.effective_material_plan
    effective_look_usd = visual_qa.effective_look_usd
    effective_apply_report = visual_qa.effective_apply_report
    effective_quality_report = visual_qa.effective_quality_report
    effective_quality_rendered_registry = visual_qa.effective_quality_rendered_registry
    quality_render_view_arguments = visual_qa.quality_render_view_arguments
    spatial_report_path = visual_qa.spatial_report_path
    trusted_mapping = visual_qa.trusted_mapping
    applied_count = visual_qa.applied_count
    quality_status = visual_qa.quality_status
    quality_gate_status = visual_qa.quality_gate_status
    quality_resolution_document = visual_qa.quality_resolution_document
    quality_limitation_count = visual_qa.quality_limitation_count
    quality_decision = visual_qa.quality_decision
    quality_round_count = visual_qa.quality_round_count
    final_instance_plan = visual_qa.final_instance_plan
    legacy_paths = workspace.legacy
    visual_group_annotated_plan = legacy_paths.visual_group_plan
    visual_group_annotation_audit = legacy_paths.visual_group_audit
    exact_mdl_tournament_dir = legacy_paths.exact_mdl_dir
    exact_mdl_tournament_planning = legacy_paths.exact_mdl_planning
    exact_mdl_tournament_audit = legacy_paths.exact_mdl_audit
    exact_mdl_tournament_plan = legacy_paths.exact_mdl_plan
    exact_mdl_tournament_instance_plan = legacy_paths.exact_mdl_instance_plan
    exact_mdl_tournament_view_map = legacy_paths.exact_mdl_view_map
    exact_mdl_tournament_final_quality = legacy_paths.exact_mdl_final_quality
    appearance_baseline_quality = legacy_paths.appearance_baseline_quality
    appearance_baseline_measurement = legacy_paths.appearance_baseline_measurement
    appearance_contract = legacy_paths.appearance_contract
    appearance_candidate_plan = legacy_paths.appearance_candidate_plan
    appearance_candidate_plan_apply = legacy_paths.appearance_candidate_plan_apply
    appearance_candidate_look_usd = legacy_paths.appearance_candidate_usd
    appearance_candidate_usd_apply = legacy_paths.appearance_candidate_apply_report
    appearance_candidate_instance_plan = legacy_paths.appearance_candidate_instance_plan
    appearance_candidate_quality_registry = (
        legacy_paths.appearance_candidate_quality.registry
    )
    appearance_candidate_render_dir = (
        legacy_paths.appearance_candidate_quality.render_dir
    )
    appearance_candidate_rendered_registry = (
        legacy_paths.appearance_candidate_quality.rendered_registry
    )
    appearance_candidate_view_map = legacy_paths.appearance_candidate_quality.view_map
    appearance_candidate_raw_quality = legacy_paths.appearance_candidate_raw_quality
    appearance_candidate_quality = legacy_paths.appearance_candidate_measured_quality
    appearance_candidate_measurement = legacy_paths.appearance_candidate_measurement
    appearance_validation = legacy_paths.appearance_validation
    _command_runner = command_runner

    exact_mdl_tournament_status = "NOT_REQUIRED"
    exact_mdl_tournament_candidate_count = 0
    exact_mdl_tournament_selected_candidate_id: str | None = None
    exact_mdl_tournament_selected_candidate_ids: list[str] = []
    exact_mdl_tournament_group_count = 0
    render_confirmed_disagreement_group_ids: set[str] = set()
    baseline_preserved_disagreement_exemptions: list[dict[str, Any]] = []
    visual_group_annotation_status = "NOT_REQUIRED"
    run_exact_mdl_tournament = (
        config.immutable_mdl_after_selection
        and (
            quality_status != "PASS"
            or config.material_selection_objective == "visual_similarity"
        )
        and config.material_assignment_unit != "part_id"
    )

    # The default immutable path evaluates every annotated, candidate-bearing
    # canonical visual group.  Each round renders the current accepted plan,
    # changes exactly one group, uses that group's Part-ID/reference masks, and
    # either accepts a clear improvement or keeps the baseline.  The next
    # group is therefore conditioned on every earlier accepted decision.
    if run_exact_mdl_tournament and config.exact_mdl_tournament_all_groups:
        exact_mdl_tournament_status = "RUNNING"
        material_candidate_dir = analysis_dir / "material_candidates"
        if not material_candidate_dir.is_dir():
            raise RuntimeError(
                "Multi-group immutable MDL tournament requires staged material "
                "candidates; physics was not started"
            )
        material_candidates_by_group: dict[str, dict[str, Any]] = {}
        for candidate_path in sorted(material_candidate_dir.glob("*.json")):
            candidate_document = read_object(
                candidate_path,
                f"material candidates {candidate_path.stem}",
            )
            candidate_group = candidate_document.get("group")
            group_id = (
                candidate_group.get("group_id")
                if isinstance(candidate_group, dict)
                else candidate_path.stem
            )
            if isinstance(group_id, str) and group_id:
                material_candidates_by_group[group_id] = candidate_document

        pre_annotation_plan_document = read_object(
            effective_material_plan,
            "pre-tournament immutable material plan",
        )
        material_choice_document = read_object(
            analysis_dir / "material_choice_audit.json",
            "material choice audit",
        )
        palette_fusion_path = analysis_dir / "palette_fusion.json"
        palette_fusion_document = read_object(
            palette_fusion_path,
            "palette fusion",
        )
        try:
            current_plan_document, annotation_audit_document = annotate_visual_groups(
                material_plan=pre_annotation_plan_document,
                palette_fusion=palette_fusion_document,
                spatial_mapping_report=read_object(
                    analysis_dir / "spatial_mapping_report.json",
                    "pre-tournament spatial mapping report",
                ),
                part_registry=read_object(
                    rendered_registry,
                    "source-appearance rendered part registry",
                ),
            )
        except VisualGroupAnnotationError as exc:
            raise RuntimeError(
                "Unable to annotate the complete material plan with canonical "
                f"visual groups; physics was not started: {exc}"
            ) from exc
        write_object(visual_group_annotated_plan, current_plan_document)
        write_object(visual_group_annotation_audit, annotation_audit_document)
        visual_group_annotation_status = str(
            annotation_audit_document.get("status", "UNKNOWN")
        )

        catalog_document = read_object(
            effective_catalog,
            "effective NVIDIA MDL catalog",
        )
        catalog_materials = catalog_document.get("materials")
        if not isinstance(catalog_materials, list):
            raise RuntimeError("Effective NVIDIA MDL catalog lacks materials")
        material_families_by_id = {
            str(material["material_id"]): str(material["family"])
            for material in catalog_materials
            if (
                isinstance(material, dict)
                and isinstance(material.get("material_id"), str)
                and isinstance(material.get("family"), str)
            )
        }
        allowed_material_ids = set(material_families_by_id)
        try:
            group_queue, queue_audit_document = build_multigroup_exact_mdl_queue(
                source_plan=current_plan_document,
                material_candidates_by_group=material_candidates_by_group,
                material_choice_audit=material_choice_document,
                palette_fusion=palette_fusion_document,
                allowed_material_ids=allowed_material_ids,
                maximum_candidates=(config.exact_mdl_tournament_max_candidates),
                selection_objective=config.material_selection_objective,
                minimum_reference_footprint_score=(
                    config.final_visual_gate_minimum_significant_reference_share
                ),
                quality_report=read_object(
                    effective_quality_report,
                    "pre-tournament effective visual quality report",
                ),
                visual_group_annotation_audit=annotation_audit_document,
            )
        except MultigroupExactMdlTournamentError as exc:
            raise RuntimeError(
                "Unable to build the complete multi-group immutable MDL queue; "
                f"physics was not started: {exc}"
            ) from exc
        baseline_preserved_disagreement_exemptions = (
            _baseline_preserved_disagreement_exemptions(
                queue_audit=queue_audit_document,
                material_choice_audit=material_choice_document,
            )
        )
        if queue_audit_document.get("coverage_blocker_count") != 0:
            raise RuntimeError(
                "Multi-group immutable MDL queue does not cover every "
                "discovered significant group; physics was not started: "
                f"{queue_audit_document.get('coverage_blockers')!r}"
            )
        if not group_queue:
            raise RuntimeError(
                "Multi-group immutable MDL queue has no candidate-bearing "
                "visual groups; physics was not started"
            )
        exact_mdl_tournament_group_count = len(group_queue)
        tournament_candidate_total = 0
        for queued_group in group_queue:
            queued_candidate_count = queued_group.get("candidate_count")
            if (
                isinstance(queued_candidate_count, bool)
                or not isinstance(queued_candidate_count, int)
                or queued_candidate_count < 2
            ):
                raise RuntimeError(
                    "Multi-group immutable MDL queue has an invalid candidate "
                    f"count: {queued_candidate_count!r}"
                )
            tournament_candidate_total += queued_candidate_count
        tournament_candidate_completed = 0
        tournament_cache_hits = 0
        tournament_cache_misses = 0

        tournament_mapping = _validated_exact_mdl_tournament_mapping(
            quality_report=read_object(
                effective_quality_report,
                "pre-tournament visual quality report",
            ),
            reference_manifest=read_object(
                analysis_dir / "reference_manifest.json",
                "pre-tournament reference manifest",
            ),
            trusted_mapping=trusted_mapping,
            rendered_registry=read_object(
                effective_quality_rendered_registry,
                "pre-tournament rendered registry",
            ),
        )
        mapped_tournament_views = list(tournament_mapping.values())
        if len(mapped_tournament_views) < 2:
            raise RuntimeError(
                "Multi-group immutable MDL tournament lacks two trusted "
                "registered views; physics was not started"
            )
        write_object(
            exact_mdl_tournament_view_map,
            {
                "schema_version": "qwen-reference-view-map/v1",
                "mapping": dict(sorted(tournament_mapping.items())),
                "source": "validated_effective_quality_selected_mapping",
            },
        )
        family_contract = build_part_family_contract(
            plan=current_plan_document,
            material_choice_audit=material_choice_document,
            palette_fusion=palette_fusion_document,
        )
        planning_document: dict[str, Any] = {
            "schema_version": ("asset-pipeline-multigroup-exact-mdl-planning/v1"),
            "status": "RUNNING",
            "mode": "all_significant_groups_coordinate_descent",
            "selection_objective": config.material_selection_objective,
            "maximum_candidates_per_group": (
                config.exact_mdl_tournament_max_candidates
            ),
            "minimum_score_improvement": (
                config.exact_mdl_tournament_minimum_score_improvement
            ),
            "minimum_winner_margin": (
                config.exact_mdl_tournament_minimum_winner_margin
            ),
            "visual_group_annotation": str(visual_group_annotation_audit),
            "queue": queue_audit_document,
            "baseline_preserved_forward_reverse_disagreement_exemptions": (
                copy.deepcopy(baseline_preserved_disagreement_exemptions)
            ),
            "rounds": [],
            "cache_resume": {
                "schema_version": "asset-pipeline-exact-mdl-cache-resume/v1",
                "candidate_total": tournament_candidate_total,
                "cache_hit_count": 0,
                "cache_miss_count": 0,
                "entries": [],
            },
        }
        write_object(exact_mdl_tournament_planning, planning_document)

        current_visual_artifacts: dict[str, Any] = {
            "look": effective_look_usd,
            "apply": effective_apply_report,
            "quality": effective_quality_report,
            "rendered_registry": effective_quality_rendered_registry,
            "applied_count": applied_count,
        }
        baseline_apply_document = read_object(
            effective_apply_report,
            "pre-tournament immutable apply report",
        )
        expected_face_subset_count = baseline_apply_document.get("face_subset_count")
        if (
            isinstance(expected_face_subset_count, bool)
            or not isinstance(expected_face_subset_count, int)
            or expected_face_subset_count < 0
        ):
            raise RuntimeError(
                "Multi-group immutable MDL tournament baseline apply report "
                "lacks a valid face_subset_count; physics was not started"
            )
        round_audits: list[dict[str, Any]] = []
        accepted_candidate_ids: list[str] = []
        tournament_occurrence_registry = read_object(
            rendered_registry,
            "tournament occurrence registry",
        )

        for group_index, group_spec in enumerate(group_queue, start=1):
            group_id = str(group_spec["group_id"])
            _log_exact_mdl_group_progress(
                log_cb,
                state="start",
                current=group_index - 1,
                total=len(group_queue),
                group_id=group_id,
            )
            target_part_ids = [
                str(part_id) for part_id in group_spec["target_part_ids"]
            ]
            target_entities = group_spec.get("target_entities")
            if not isinstance(target_entities, list) or not target_entities:
                raise RuntimeError(
                    f"Visual group {group_id} lacks target material entities; "
                    "physics was not started"
                )
            reference_view_ids = [
                str(view_id) for view_id in group_spec.get("reference_view_ids", [])
            ]
            if len(reference_view_ids) < 2:
                raise RuntimeError(
                    f"Visual group {group_id} lacks two independent reference "
                    "views; physics was not started"
                )
            missing_mapped_views = sorted(
                set(reference_view_ids) - set(tournament_mapping)
            )
            if missing_mapped_views:
                raise RuntimeError(
                    f"Visual group {group_id} references unmapped views "
                    f"{missing_mapped_views}; physics was not started"
                )
            candidate_document = material_candidates_by_group.get(group_id)
            if not isinstance(candidate_document, dict):
                raise RuntimeError(
                    f"Visual group {group_id} lost its candidate document"
                )
            disagreement_contract: dict[str, Any] | None = None
            raw_choice = material_choice_document.get(group_id)
            if (
                isinstance(raw_choice, Mapping)
                and raw_choice.get("confirmation_basis")
                == "forward_reverse_disagreement"
            ):
                raw_contract = raw_choice.get("disagreement_tournament")
                candidate_contract = candidate_document.get("disagreement_tournament")
                if (
                    not isinstance(raw_contract, Mapping)
                    or candidate_contract != raw_contract
                ):
                    raise RuntimeError(
                        f"Visual group {group_id} has unresolved forward/reverse "
                        "MDL choices but lacks one hash-consistent exact-MDL "
                        "tournament contract; physics was not started"
                    )
                forward = raw_choice.get("forward")
                reverse = raw_choice.get("reverse")
                queued_material_ids = group_spec.get("candidate_material_ids")
                if (
                    not isinstance(forward, Mapping)
                    or not isinstance(reverse, Mapping)
                    or not isinstance(queued_material_ids, list)
                ):
                    raise RuntimeError(
                        f"Visual group {group_id} has an invalid disagreement "
                        "tournament input; physics was not started"
                    )
                try:
                    disagreement_contract = validate_disagreement_tournament_contract(
                        raw_contract,
                        forward_material_id=str(forward.get("material_id") or ""),
                        reverse_material_id=str(reverse.get("material_id") or ""),
                        tournament_candidate_material_ids=queued_material_ids,
                    )
                except DisagreementTournamentContractError as exc:
                    raise RuntimeError(
                        f"Visual group {group_id} disagreement tournament is "
                        f"incomplete; physics was not started: {exc}"
                    ) from exc
            try:
                (
                    planned_candidates,
                    round_planning,
                ) = build_exact_mdl_group_candidate_plans(
                    source_plan=current_plan_document,
                    group_id=group_id,
                    target_part_ids=target_part_ids,
                    target_entities=target_entities,
                    candidate_document=candidate_document,
                    allowed_material_ids=allowed_material_ids,
                    maximum_candidates=(config.exact_mdl_tournament_max_candidates),
                    selection_objective=(config.material_selection_objective),
                    allowed_families=set(group_spec.get("allowed_families", [])),
                )
            except MultigroupExactMdlTournamentError as exc:
                raise RuntimeError(
                    f"Unable to plan immutable MDL round {group_id}; "
                    f"physics was not started: {exc}"
                ) from exc
            if len(planned_candidates) != group_spec.get("candidate_count"):
                raise RuntimeError(
                    f"Immutable MDL round {group_id} candidate count changed "
                    "after the global progress contract was established"
                )
            exact_mdl_tournament_candidate_count += len(planned_candidates)
            planning_document["rounds"].append(round_planning)
            write_object(exact_mdl_tournament_planning, planning_document)

            rendered_candidate_bundles: list[dict[str, Any]] = []
            candidate_artifacts: dict[str, dict[str, Any]] = {}
            baseline_candidate_id: str | None = None
            candidate_total = len(planned_candidates)
            for candidate_index, planned_candidate in enumerate(
                planned_candidates,
                start=1,
            ):
                candidate_id = str(planned_candidate["candidate_id"])
                _log_exact_mdl_candidate_progress(
                    log_cb,
                    state="start",
                    group_index=group_index,
                    group_total=len(group_queue),
                    candidate_index=candidate_index,
                    candidate_total=candidate_total,
                    candidate_id=candidate_id,
                    global_current=tournament_candidate_completed,
                    global_total=tournament_candidate_total,
                )
                candidate_dir = exact_mdl_tournament_dir / candidate_id
                candidate_plan_path = candidate_dir / "plan.json"
                candidate_effective_plan_path = candidate_plan_path
                candidate_look_usd = candidate_dir / "look.usda"
                candidate_apply_report = candidate_dir / "apply_report.json"
                candidate_effective_apply_report = candidate_apply_report
                candidate_registry = candidate_dir / "part_registry.json"
                candidate_render_dir = candidate_dir / "renders"
                candidate_rendered_registry = (
                    candidate_render_dir / "part_registry.rendered.json"
                )
                candidate_quality_report = (
                    candidate_dir / "reference_render_comparison.json"
                )
                candidate_whole_asset_quality_report = (
                    candidate_dir / "whole_asset_reference_render_comparison.json"
                )
                candidate_plan_document = dict(planned_candidate["plan"])
                if instance_root_count:
                    candidate_provenance = candidate_plan_document.get("provenance")
                    sealed_candidate_provenance = (
                        dict(candidate_provenance)
                        if isinstance(candidate_provenance, dict)
                        else {}
                    )
                    sealed_candidate_provenance.update(
                        {
                            "asset_sha256": sha256_file(source),
                            "registry_sha256": canonical_sha256(
                                read_object(
                                    rendered_registry,
                                    "tournament occurrence registry",
                                )
                            ),
                        }
                    )
                    candidate_plan_document["provenance"] = sealed_candidate_provenance
                if planned_candidate.get("is_baseline") is True:
                    baseline_candidate_id = candidate_id

                cached_candidate: dict[str, Any] | None = None
                cache_miss_reason = "CACHE_ENTRY_NOT_FOUND"
                cache_archive: Path | None = None
                if candidate_dir.exists() or candidate_dir.is_symlink():
                    try:
                        cached_candidate = _validate_exact_mdl_candidate_cache(
                            candidate_dir=candidate_dir,
                            candidate_id=candidate_id,
                            expected_plan=candidate_plan_document,
                            apply_asset=Path(apply_asset),
                            occurrence_registry=tournament_occurrence_registry,
                            expected_applied_count=applied_count,
                            expected_face_subset_count=expected_face_subset_count,
                            expected_mapping=tournament_mapping,
                            expected_render_view_ids=mapped_tournament_views,
                            expected_reference_view_ids=reference_view_ids,
                            expected_render_resolution=config.render_resolution,
                            expected_analysis_up_axis=config.analysis_up_axis,
                            expected_analysis_front_axis=config.analysis_front_axis,
                            reference_manifest=(
                                analysis_dir / "reference_manifest.json"
                            ),
                            palette_fusion=palette_fusion_path,
                            target_group_id=group_id,
                            target_part_ids=target_part_ids,
                            target_entities=target_entities,
                            whole_asset_quality_path=(
                                candidate_whole_asset_quality_report
                                if candidate_whole_asset_quality_report.is_file()
                                else None
                            ),
                        )
                    except (OSError, RuntimeError, ValueError) as exc:
                        cache_miss_reason = f"{type(exc).__name__}: {str(exc).strip()}"
                        cache_archive = _archive_exact_mdl_candidate_cache_entry(
                            destination=destination,
                            candidate_path=candidate_dir,
                            reason=cache_miss_reason,
                        )
                        log_message(
                            log_cb,
                            "Exact MDL candidate cache CACHE_MISS "
                            f"id={candidate_id} reason={cache_miss_reason} "
                            f"archived={cache_archive}",
                        )
                if cached_candidate is not None:
                    tournament_cache_hits += 1
                    candidate_cache_status = "CACHE_HIT"
                    candidate_apply_document = cached_candidate["apply_report"]
                    candidate_effective_plan_path = Path(cached_candidate["plan_path"])
                    candidate_effective_apply_report = Path(
                        cached_candidate["apply_report_path"]
                    )
                    candidate_applied_count = int(
                        candidate_apply_document["applied_count"]
                    )
                    candidate_quality_document = cached_candidate["quality_report"]
                    candidate_whole_asset_quality_document = cached_candidate[
                        "whole_asset_quality_report"
                    ]
                    candidate_rendered_registry_document = cached_candidate[
                        "rendered_registry"
                    ]
                    candidate_rendered_registry_file_sha256 = cached_candidate[
                        "rendered_registry_file_sha256"
                    ]
                    log_message(
                        log_cb,
                        f"Exact MDL candidate cache CACHE_HIT id={candidate_id}",
                    )
                else:
                    tournament_cache_misses += 1
                    candidate_cache_status = "CACHE_MISS_RENDERED"
                    if cache_miss_reason == "CACHE_ENTRY_NOT_FOUND":
                        log_message(
                            log_cb,
                            "Exact MDL candidate cache CACHE_MISS "
                            f"id={candidate_id} reason={cache_miss_reason}",
                        )
                    write_object(candidate_plan_path, candidate_plan_document)

                candidate_apply_command = [
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
                    str(candidate_plan_path),
                    "--output",
                    str(candidate_look_usd),
                    "--material-root",
                    str(config.material_root),
                    "--report",
                    str(candidate_apply_report),
                    "--include-review",
                ]
                if use_policy_fallback:
                    candidate_apply_command.append("--include-policy-fallback")
                stage_prefix = f"exact_mdl_group_{group_index:02d}_{candidate_id}"
                if cached_candidate is None:
                    _run_stage(
                        f"{stage_prefix}_apply",
                        candidate_apply_command,
                        log_cb,
                        command_runner=_command_runner,
                        retry_native_crash=True,
                    )
                    _require_file(candidate_look_usd, f"{stage_prefix}_apply")
                    _require_file(candidate_apply_report, f"{stage_prefix}_apply")
                    candidate_apply_document = read_object(
                        candidate_apply_report,
                        f"tournament apply report {candidate_id}",
                    )
                    candidate_applied_count = candidate_apply_document.get(
                        "applied_count"
                    )
                    if (
                        isinstance(candidate_applied_count, bool)
                        or not isinstance(candidate_applied_count, int)
                        or candidate_applied_count != applied_count
                    ):
                        raise RuntimeError(
                            "Multi-group immutable MDL tournament changed exact "
                            f"coverage: baseline={applied_count} "
                            f"candidate={candidate_applied_count!r}"
                        )
                    candidate_face_subset_count = candidate_apply_document.get(
                        "face_subset_count"
                    )
                    if candidate_face_subset_count != expected_face_subset_count:
                        raise RuntimeError(
                            "Multi-group immutable MDL tournament changed exact "
                            "face-subset coverage: "
                            f"baseline={expected_face_subset_count} "
                            f"candidate={candidate_face_subset_count!r}"
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
                        command_runner=_command_runner,
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
                            str(config.render_resolution),
                            "--views",
                            ",".join(mapped_tournament_views),
                            "--rt-subframes",
                            str(config.render_rt_subframes),
                            "--lighting-profile",
                            "material-neutral",
                            "--analysis-up-axis",
                            config.analysis_up_axis,
                            f"--analysis-front-axis={config.analysis_front_axis}",
                        ],
                        log_cb,
                        command_runner=_command_runner,
                        retry_native_crash=True,
                    )
                    _require_file(
                        candidate_rendered_registry,
                        f"{stage_prefix}_render",
                    )
                    candidate_compare_command = _multigroup_local_compare_command(
                        qwen_python=config.qwen_python,
                        reference_manifest=(analysis_dir / "reference_manifest.json"),
                        rendered_registry=candidate_rendered_registry,
                        view_map=exact_mdl_tournament_view_map,
                        palette_fusion=palette_fusion_path,
                        group_id=group_id,
                        target_part_ids=target_part_ids,
                        target_entities=target_entities,
                        reference_view_ids=reference_view_ids,
                        output=candidate_quality_report,
                    )
                    _run_stage(
                        f"{stage_prefix}_compare_local",
                        candidate_compare_command,
                        log_cb,
                        command_runner=_command_runner,
                    )
                    _require_file(
                        candidate_quality_report,
                        f"{stage_prefix}_compare_local",
                    )
                    candidate_quality_document = read_object(
                        candidate_quality_report,
                        f"tournament quality report {candidate_id}",
                    )
                    candidate_rendered_registry_document = read_object(
                        candidate_rendered_registry,
                        f"tournament rendered registry {candidate_id}",
                    )
                    candidate_rendered_registry_file_sha256 = sha256_file(
                        candidate_rendered_registry
                    )
                    candidate_whole_asset_quality_document = None
                whole_asset_cache_status = "CACHE_HIT"
                if candidate_whole_asset_quality_document is None:
                    whole_asset_cache_status = "CACHE_MISS_COMPUTED"
                    candidate_whole_asset_compare_command = [
                        str(config.qwen_python),
                        "-m",
                        "qwen_material_pipeline",
                        "compare",
                        "--reference-manifest",
                        str(analysis_dir / "reference_manifest.json"),
                        "--rendered-registry",
                        str(candidate_rendered_registry),
                        "--view-map",
                        str(exact_mdl_tournament_view_map),
                        "--minimum-comparable-views",
                        str(len(tournament_mapping)),
                        "--output",
                        str(candidate_whole_asset_quality_report),
                    ]
                    _run_stage(
                        f"{stage_prefix}_compare_whole_asset",
                        candidate_whole_asset_compare_command,
                        log_cb,
                        command_runner=_command_runner,
                    )
                    _require_file(
                        candidate_whole_asset_quality_report,
                        f"{stage_prefix}_compare_whole_asset",
                    )
                    candidate_whole_asset_quality_document = (
                        _validate_exact_mdl_whole_asset_quality_cache(
                            candidate_dir=candidate_dir,
                            quality_path=candidate_whole_asset_quality_report,
                            rendered_registry_path=candidate_rendered_registry,
                            expected_mapping=tournament_mapping,
                            reference_manifest=(
                                analysis_dir / "reference_manifest.json"
                            ),
                        )
                    )
                rendered_candidate_bundles.append(
                    {
                        "candidate_id": candidate_id,
                        "is_baseline": (planned_candidate.get("is_baseline") is True),
                        "plan": candidate_plan_document,
                        "apply_report": candidate_apply_document,
                        "quality_report": candidate_quality_document,
                        "global_quality_report": (
                            candidate_whole_asset_quality_document
                        ),
                        "rendered_registry": (candidate_rendered_registry_document),
                        "rendered_registry_file_sha256": (
                            candidate_rendered_registry_file_sha256
                        ),
                    }
                )
                candidate_artifacts[candidate_id] = {
                    "plan": candidate_effective_plan_path,
                    "look": candidate_look_usd,
                    "apply": candidate_effective_apply_report,
                    "quality": candidate_whole_asset_quality_report,
                    "local_quality": candidate_quality_report,
                    "global_quality": candidate_whole_asset_quality_report,
                    "rendered_registry": candidate_rendered_registry,
                    "applied_count": candidate_applied_count,
                }
                cache_resume = planning_document["cache_resume"]
                assert isinstance(cache_resume, dict)
                cache_entries = cache_resume["entries"]
                assert isinstance(cache_entries, list)
                cache_entries.append(
                    {
                        "candidate_id": candidate_id,
                        "group_id": group_id,
                        "status": candidate_cache_status,
                        "plan_sha256": canonical_sha256(candidate_plan_document),
                        "rendered_registry_file_sha256": (
                            candidate_rendered_registry_file_sha256
                        ),
                        "quality_report_sha256": canonical_sha256(
                            candidate_quality_document
                        ),
                        "global_quality_report_sha256": canonical_sha256(
                            candidate_whole_asset_quality_document
                        ),
                        "global_quality_cache_status": whole_asset_cache_status,
                        "archive": (
                            str(cache_archive) if cache_archive is not None else None
                        ),
                        "cache_miss_reason": (
                            cache_miss_reason if cached_candidate is None else None
                        ),
                        "candidate_cache_rebase": (
                            cached_candidate.get("cache_rebase")
                            if cached_candidate is not None
                            else None
                        ),
                    }
                )
                cache_resume["cache_hit_count"] = tournament_cache_hits
                cache_resume["cache_miss_count"] = tournament_cache_misses
                write_object(exact_mdl_tournament_planning, planning_document)
                tournament_candidate_completed += 1
                _log_exact_mdl_candidate_progress(
                    log_cb,
                    state="complete",
                    group_index=group_index,
                    group_total=len(group_queue),
                    candidate_index=candidate_index,
                    candidate_total=candidate_total,
                    candidate_id=candidate_id,
                    global_current=tournament_candidate_completed,
                    global_total=tournament_candidate_total,
                    cache_status=candidate_cache_status,
                )
            if baseline_candidate_id is None:
                raise AssertionError(f"Immutable MDL round {group_id} has no baseline")
            try:
                current_plan_document, round_audit = select_exact_mdl_group_step(
                    current_plan=current_plan_document,
                    group_id=group_id,
                    target_part_ids=target_part_ids,
                    target_entities=target_entities,
                    candidates=rendered_candidate_bundles,
                    allowed_material_ids=allowed_material_ids,
                    material_families_by_id=material_families_by_id,
                    allowed_families_by_part=family_contract,
                    selection_objective=(config.material_selection_objective),
                    minimum_score_improvement=(
                        config.exact_mdl_tournament_minimum_score_improvement
                    ),
                    minimum_winner_margin=(
                        config.exact_mdl_tournament_minimum_winner_margin
                    ),
                )
            except MultigroupExactMdlTournamentError as exc:
                raise RuntimeError(
                    f"Unable to select immutable MDL round {group_id}; "
                    f"physics was not started: {exc}"
                ) from exc
            round_audit_path = (
                exact_mdl_tournament_dir / f"{group_id.casefold()}_round_audit.json"
            )
            if disagreement_contract is not None:
                try:
                    disagreement_resolved = disagreement_is_render_confirmed(
                        disagreement_contract,
                        round_audit=round_audit,
                    )
                except DisagreementTournamentContractError as exc:
                    raise RuntimeError(
                        f"Visual group {group_id} disagreement resolution audit "
                        f"is invalid; physics was not started: {exc}"
                    ) from exc
                round_audit["forward_reverse_disagreement_resolution"] = {
                    "required": True,
                    "resolved": disagreement_resolved,
                    "resolution_policy": disagreement_contract["resolution_policy"],
                    "provisional_seed_material_id": disagreement_contract[
                        "provisional_seed_material_id"
                    ],
                    "required_candidate_material_ids": disagreement_contract[
                        "required_candidate_material_ids"
                    ],
                }
                if not disagreement_resolved:
                    write_object(round_audit_path, round_audit)
                    raise RuntimeError(
                        f"Visual group {group_id} retained an unconfirmed "
                        "forward/reverse disagreement seed after exact-MDL "
                        "rendering; final material lock was refused"
                    )
                render_confirmed_disagreement_group_ids.add(group_id)
            round_audits.append(round_audit)
            write_object(round_audit_path, round_audit)
            round_planning["round_audit"] = str(round_audit_path)
            round_planning["decision_status"] = round_audit["status"]
            accepted_candidate_id = round_audit.get("accepted_candidate_id")
            if isinstance(accepted_candidate_id, str):
                accepted_candidate_ids.append(accepted_candidate_id)
                current_artifact_id = accepted_candidate_id
            else:
                current_artifact_id = baseline_candidate_id
            current_visual_artifacts = candidate_artifacts[current_artifact_id]
            quality_round_count += 1
            log_message(
                log_cb,
                f"Immutable MDL visual group {group_id} completed with "
                f"decision {round_audit['status']}.",
            )
            _log_exact_mdl_group_progress(
                log_cb,
                state="complete",
                current=group_index,
                total=len(group_queue),
                group_id=group_id,
            )

        if tournament_candidate_completed != tournament_candidate_total:
            raise RuntimeError(
                "Exact-MDL global candidate progress ended with an incomplete "
                f"count: {tournament_candidate_completed}/"
                f"{tournament_candidate_total}"
            )
        try:
            (
                selected_plan_document,
                tournament_audit_document,
            ) = finalize_multigroup_exact_mdl_plan(
                initial_plan=read_object(
                    visual_group_annotated_plan,
                    "annotated tournament baseline plan",
                ),
                current_plan=current_plan_document,
                significant_group_ids=[str(group["group_id"]) for group in group_queue],
                round_audits=round_audits,
                selection_objective=(config.material_selection_objective),
            )
        except MultigroupExactMdlTournamentError as exc:
            raise RuntimeError(
                "Unable to finalize multi-group immutable MDL plan; "
                f"physics was not started: {exc}"
            ) from exc
        write_object(exact_mdl_tournament_plan, selected_plan_document)

        final_compare_command = [
            str(config.qwen_python),
            "-m",
            "qwen_material_pipeline",
            "compare",
            "--reference-manifest",
            str(analysis_dir / "reference_manifest.json"),
            "--rendered-registry",
            str(current_visual_artifacts["rendered_registry"]),
            "--view-map",
            str(exact_mdl_tournament_view_map),
            "--minimum-comparable-views",
            str(len(tournament_mapping)),
            "--output",
            str(exact_mdl_tournament_final_quality),
        ]
        _run_stage(
            "exact_mdl_multigroup_compare_final_whole_asset",
            final_compare_command,
            log_cb,
            command_runner=_command_runner,
        )
        _require_file(
            exact_mdl_tournament_final_quality,
            "exact_mdl_multigroup_compare_final_whole_asset",
        )
        final_quality_document = read_object(
            exact_mdl_tournament_final_quality,
            "final multi-group whole-asset quality report",
        )
        final_quality_aggregate = final_quality_document.get("aggregate")
        if not isinstance(final_quality_aggregate, dict):
            raise RuntimeError(
                "Final multi-group whole-asset quality report lacks aggregate"
            )
        final_quality_status = final_quality_aggregate.get("status")
        if final_quality_status not in QUALITY_STATUSES:
            raise RuntimeError(
                "Final multi-group whole-asset quality report has an "
                f"unsupported status: {final_quality_status!r}"
            )
        baseline_preserved_disagreement_exemptions = (
            _final_baseline_preserved_disagreement_exemptions(
                queue_audit=queue_audit_document,
                material_choice_audit=material_choice_document,
                final_plan=selected_plan_document,
                palette_fusion=palette_fusion_document,
                final_quality_report_path=exact_mdl_tournament_final_quality,
                final_rendered_registry_path=Path(
                    current_visual_artifacts["rendered_registry"]
                ),
            )
        )

        tournament_audit_document["visual_group_annotation"] = {
            "status": visual_group_annotation_status,
            "plan": str(visual_group_annotated_plan),
            "audit": str(visual_group_annotation_audit),
        }
        tournament_audit_document["queue"] = queue_audit_document
        tournament_audit_document[
            "baseline_preserved_forward_reverse_disagreement_exemptions"
        ] = copy.deepcopy(baseline_preserved_disagreement_exemptions)
        planning_document[
            "baseline_preserved_forward_reverse_disagreement_exemptions"
        ] = copy.deepcopy(baseline_preserved_disagreement_exemptions)
        tournament_audit_document["accepted_candidate_ids"] = list(
            accepted_candidate_ids
        )
        tournament_audit_document["cache_resume"] = planning_document["cache_resume"]
        tournament_audit_document["final_whole_asset_quality"] = {
            "report": str(exact_mdl_tournament_final_quality),
            "report_sha256": sha256_file(exact_mdl_tournament_final_quality),
            "status": final_quality_status,
            "material_color_score": final_quality_aggregate.get("material_color_score"),
            "material_texture_score": final_quality_aggregate.get(
                "material_texture_score"
            ),
            "material_appearance_score": final_quality_aggregate.get(
                "material_appearance_score"
            ),
        }
        write_object(exact_mdl_tournament_audit, tournament_audit_document)
        planning_document.update(
            {
                "status": "COMPLETED",
                "group_count": len(group_queue),
                "candidate_count": exact_mdl_tournament_candidate_count,
                "accepted_candidate_ids": list(accepted_candidate_ids),
                "final_plan": str(exact_mdl_tournament_plan),
                "audit": str(exact_mdl_tournament_audit),
            }
        )
        write_object(exact_mdl_tournament_planning, planning_document)

        if instance_root_count:
            selected_instance_document = dict(selected_plan_document)
            selected_instance_provenance = selected_instance_document.get("provenance")
            sealed_selected_provenance = (
                dict(selected_instance_provenance)
                if isinstance(selected_instance_provenance, dict)
                else {}
            )
            sealed_selected_provenance.update(
                {
                    "asset_sha256": sha256_file(source),
                    "registry_sha256": canonical_sha256(
                        read_object(
                            rendered_registry,
                            "selected tournament occurrence registry",
                        )
                    ),
                }
            )
            selected_instance_document["provenance"] = sealed_selected_provenance
            write_object(
                exact_mdl_tournament_instance_plan,
                selected_instance_document,
            )
            final_instance_plan = exact_mdl_tournament_instance_plan

        effective_material_plan = exact_mdl_tournament_plan
        effective_look_usd = current_visual_artifacts["look"]
        effective_apply_report = current_visual_artifacts["apply"]
        effective_quality_report = exact_mdl_tournament_final_quality
        effective_quality_rendered_registry = current_visual_artifacts[
            "rendered_registry"
        ]
        applied_count = int(current_visual_artifacts["applied_count"])
        quality_status = str(final_quality_status)
        quality_gate_status = str(final_quality_status)
        quality_resolution_document = None
        quality_limitation_count = 0
        quality_decision = (
            "ACCEPTED_AFTER_MULTIGROUP_EXACT_MDL_RENDER_TOURNAMENT"
            if final_quality_status == "PASS"
            else "COMPLETED_MULTIGROUP_EXACT_MDL_PENDING_FINAL_VISUAL_GATE"
        )
        exact_mdl_tournament_selected_candidate_ids = list(accepted_candidate_ids)
        exact_mdl_tournament_selected_candidate_id = (
            accepted_candidate_ids[-1] if accepted_candidate_ids else None
        )
        exact_mdl_tournament_status = (
            "SELECTED" if accepted_candidate_ids else "BASELINE_PRESERVED"
        )
        log_message(
            log_cb,
            "All significant canonical visual groups completed the immutable "
            "exact NVIDIA MDL coordinate-descent tournament; the final "
            f"whole-asset QA status is {final_quality_status}.",
        )

    # Immutable selection cannot turn a visually inconclusive Look into an
    # accepted deliverable merely by sealing it.  When bounded repair leaves a
    # REVIEW/limited result, render a small set of exact NVIDIA exports for the
    # dominant, reliably classified material group.  Every candidate uses
    # library defaults, changes one canonical group only, and must satisfy both
    # the material-identity contract and every registered reference view.
    elif run_exact_mdl_tournament:
        exact_mdl_tournament_status = "RUNNING"
        material_candidate_dir = analysis_dir / "material_candidates"
        if not material_candidate_dir.is_dir():
            raise RuntimeError(
                "Immutable MDL tournament requires staged material candidates; "
                "physics was not started"
            )
        material_candidates_by_group: dict[str, dict[str, Any]] = {}
        for candidate_path in sorted(material_candidate_dir.glob("*.json")):
            candidate_document = read_object(
                candidate_path,
                f"material candidates {candidate_path.stem}",
            )
            candidate_group = candidate_document.get("group")
            group_id = (
                candidate_group.get("group_id")
                if isinstance(candidate_group, dict)
                else candidate_path.stem
            )
            if isinstance(group_id, str) and group_id:
                material_candidates_by_group[group_id] = candidate_document
        current_plan_document = read_object(
            effective_material_plan,
            "pre-tournament immutable material plan",
        )
        material_choice_document = read_object(
            analysis_dir / "material_choice_audit.json",
            "material choice audit",
        )
        palette_fusion_document = read_object(
            analysis_dir / "palette_fusion.json",
            "palette fusion",
        )
        catalog_document = read_object(
            effective_catalog,
            "effective NVIDIA MDL catalog",
        )
        catalog_materials = catalog_document.get("materials")
        if not isinstance(catalog_materials, list):
            raise RuntimeError("Effective NVIDIA MDL catalog lacks materials")
        material_families_by_id = {
            str(material["material_id"]): str(material["family"])
            for material in catalog_materials
            if (
                isinstance(material, dict)
                and isinstance(material.get("material_id"), str)
                and isinstance(material.get("family"), str)
            )
        }
        try:
            (
                planned_candidates,
                planning_document,
            ) = build_bounded_exact_mdl_candidate_plans(
                source_plan=current_plan_document,
                material_candidates_by_group=material_candidates_by_group,
                material_choice_audit=material_choice_document,
                palette_fusion=palette_fusion_document,
                allowed_material_ids=set(material_families_by_id),
                maximum_candidates=(config.exact_mdl_tournament_max_candidates),
                selection_objective=config.material_selection_objective,
            )
        except ExactMdlTournamentError as exc:
            raise RuntimeError(
                "Unable to plan a bounded immutable MDL tournament; "
                f"physics was not started: {exc}"
            ) from exc
        selected_single_group_id = planning_document.get("selected_group_id")
        single_disagreement_contract: dict[str, Any] | None = None
        selected_single_choice = (
            material_choice_document.get(selected_single_group_id)
            if isinstance(selected_single_group_id, str)
            else None
        )
        if (
            isinstance(selected_single_choice, Mapping)
            and selected_single_choice.get("confirmation_basis")
            == "forward_reverse_disagreement"
        ):
            selected_candidate_document = material_candidates_by_group.get(
                selected_single_group_id
            )
            raw_contract = selected_single_choice.get("disagreement_tournament")
            candidate_contract = (
                selected_candidate_document.get("disagreement_tournament")
                if isinstance(selected_candidate_document, Mapping)
                else None
            )
            forward = selected_single_choice.get("forward")
            reverse = selected_single_choice.get("reverse")
            candidate_material_ids = planning_document.get("candidate_material_ids")
            if (
                not isinstance(raw_contract, Mapping)
                or candidate_contract != raw_contract
                or not isinstance(forward, Mapping)
                or not isinstance(reverse, Mapping)
                or not isinstance(candidate_material_ids, list)
            ):
                raise RuntimeError(
                    f"Visual group {selected_single_group_id} has an incomplete "
                    "forward/reverse disagreement tournament contract; physics "
                    "was not started"
                )
            try:
                single_disagreement_contract = (
                    validate_disagreement_tournament_contract(
                        raw_contract,
                        forward_material_id=str(forward.get("material_id") or ""),
                        reverse_material_id=str(reverse.get("material_id") or ""),
                        tournament_candidate_material_ids=candidate_material_ids,
                    )
                )
            except DisagreementTournamentContractError as exc:
                raise RuntimeError(
                    f"Visual group {selected_single_group_id} disagreement "
                    f"tournament is incomplete; physics was not started: {exc}"
                ) from exc
        planning_document["cache_resume"] = {
            "schema_version": "asset-pipeline-exact-mdl-cache-resume/v1",
            "candidate_total": len(planned_candidates),
            "cache_hit_count": 0,
            "cache_miss_count": 0,
            "entries": [],
        }
        write_object(exact_mdl_tournament_planning, planning_document)
        exact_mdl_tournament_candidate_count = len(planned_candidates)
        tournament_candidate_completed = 0
        tournament_cache_hits = 0
        tournament_cache_misses = 0
        tournament_mapping = _validated_exact_mdl_tournament_mapping(
            quality_report=read_object(
                effective_quality_report,
                "pre-tournament visual quality report",
            ),
            reference_manifest=read_object(
                analysis_dir / "reference_manifest.json",
                "pre-tournament reference manifest",
            ),
            trusted_mapping=trusted_mapping,
            rendered_registry=read_object(
                effective_quality_rendered_registry,
                "pre-tournament rendered registry",
            ),
        )
        mapped_tournament_views = list(tournament_mapping.values())
        if len(mapped_tournament_views) < 2:
            raise RuntimeError(
                "Immutable MDL tournament lacks two trusted registered views; "
                "physics was not started"
            )
        write_object(
            exact_mdl_tournament_view_map,
            {
                "schema_version": "qwen-reference-view-map/v1",
                "mapping": dict(sorted(tournament_mapping.items())),
                "source": "validated_effective_quality_selected_mapping",
            },
        )
        rendered_candidate_bundles: list[dict[str, Any]] = []
        candidate_artifacts: dict[str, dict[str, Path]] = {}
        baseline_candidate_plan: dict[str, Any] | None = None
        candidate_total = len(planned_candidates)
        baseline_apply_document = read_object(
            effective_apply_report,
            "pre-tournament immutable apply report",
        )
        expected_face_subset_count = baseline_apply_document.get("face_subset_count")
        if (
            isinstance(expected_face_subset_count, bool)
            or not isinstance(expected_face_subset_count, int)
            or expected_face_subset_count < 0
        ):
            raise RuntimeError(
                "Immutable MDL tournament baseline apply report lacks a valid "
                "face_subset_count; physics was not started"
            )
        tournament_occurrence_registry = read_object(
            rendered_registry,
            "tournament occurrence registry",
        )
        _log_exact_mdl_group_progress(
            log_cb,
            state="start",
            current=0,
            total=1,
            group_id=str(planning_document.get("selected_group_id", "dominant")),
        )
        for candidate_index, planned_candidate in enumerate(
            planned_candidates,
            start=1,
        ):
            candidate_id = str(planned_candidate["candidate_id"])
            _log_exact_mdl_candidate_progress(
                log_cb,
                state="start",
                group_index=1,
                group_total=1,
                candidate_index=candidate_index,
                candidate_total=candidate_total,
                candidate_id=candidate_id,
                global_current=tournament_candidate_completed,
                global_total=candidate_total,
            )
            candidate_dir = exact_mdl_tournament_dir / candidate_id
            candidate_plan_path = candidate_dir / "plan.json"
            candidate_effective_plan_path = candidate_plan_path
            candidate_look_usd = candidate_dir / "look.usda"
            candidate_apply_report = candidate_dir / "apply_report.json"
            candidate_effective_apply_report = candidate_apply_report
            candidate_registry = candidate_dir / "part_registry.json"
            candidate_render_dir = candidate_dir / "renders"
            candidate_rendered_registry = (
                candidate_render_dir / "part_registry.rendered.json"
            )
            candidate_quality_report = (
                candidate_dir / "reference_render_comparison.json"
            )
            candidate_plan_document = dict(planned_candidate["plan"])
            if instance_root_count:
                candidate_provenance = candidate_plan_document.get("provenance")
                sealed_candidate_provenance = (
                    dict(candidate_provenance)
                    if isinstance(candidate_provenance, dict)
                    else {}
                )
                sealed_candidate_provenance.update(
                    {
                        "asset_sha256": sha256_file(source),
                        "registry_sha256": canonical_sha256(
                            read_object(
                                rendered_registry,
                                "tournament occurrence registry",
                            )
                        ),
                    }
                )
                candidate_plan_document["provenance"] = sealed_candidate_provenance
            if baseline_candidate_plan is None:
                baseline_candidate_plan = candidate_plan_document
            cached_candidate: dict[str, Any] | None = None
            cache_miss_reason = "CACHE_ENTRY_NOT_FOUND"
            cache_archive: Path | None = None
            if candidate_dir.exists() or candidate_dir.is_symlink():
                try:
                    cached_candidate = _validate_exact_mdl_candidate_cache(
                        candidate_dir=candidate_dir,
                        candidate_id=candidate_id,
                        expected_plan=candidate_plan_document,
                        apply_asset=Path(apply_asset),
                        occurrence_registry=tournament_occurrence_registry,
                        expected_applied_count=applied_count,
                        expected_face_subset_count=expected_face_subset_count,
                        expected_mapping=tournament_mapping,
                        expected_render_view_ids=mapped_tournament_views,
                        expected_reference_view_ids=sorted(tournament_mapping),
                        expected_render_resolution=config.render_resolution,
                        expected_analysis_up_axis=config.analysis_up_axis,
                        expected_analysis_front_axis=config.analysis_front_axis,
                        reference_manifest=(analysis_dir / "reference_manifest.json"),
                        palette_fusion=None,
                        target_group_id=None,
                    )
                except (OSError, RuntimeError, ValueError) as exc:
                    cache_miss_reason = f"{type(exc).__name__}: {str(exc).strip()}"
                    cache_archive = _archive_exact_mdl_candidate_cache_entry(
                        destination=destination,
                        candidate_path=candidate_dir,
                        reason=cache_miss_reason,
                    )
                    log_message(
                        log_cb,
                        "Exact MDL candidate cache CACHE_MISS "
                        f"id={candidate_id} reason={cache_miss_reason} "
                        f"archived={cache_archive}",
                    )
            if cached_candidate is not None:
                tournament_cache_hits += 1
                candidate_cache_status = "CACHE_HIT"
                candidate_apply_document = cached_candidate["apply_report"]
                candidate_effective_plan_path = Path(cached_candidate["plan_path"])
                candidate_effective_apply_report = Path(
                    cached_candidate["apply_report_path"]
                )
                candidate_applied_count = int(candidate_apply_document["applied_count"])
                candidate_quality_document = cached_candidate["quality_report"]
                candidate_rendered_registry_document = cached_candidate[
                    "rendered_registry"
                ]
                candidate_rendered_registry_file_sha256 = cached_candidate[
                    "rendered_registry_file_sha256"
                ]
                log_message(
                    log_cb,
                    f"Exact MDL candidate cache CACHE_HIT id={candidate_id}",
                )
            else:
                tournament_cache_misses += 1
                candidate_cache_status = "CACHE_MISS_RENDERED"
                if cache_miss_reason == "CACHE_ENTRY_NOT_FOUND":
                    log_message(
                        log_cb,
                        "Exact MDL candidate cache CACHE_MISS "
                        f"id={candidate_id} reason={cache_miss_reason}",
                    )
                write_object(candidate_plan_path, candidate_plan_document)
            candidate_apply_command = [
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
                str(candidate_plan_path),
                "--output",
                str(candidate_look_usd),
                "--material-root",
                str(config.material_root),
                "--report",
                str(candidate_apply_report),
                "--include-review",
            ]
            if use_policy_fallback:
                candidate_apply_command.append("--include-policy-fallback")
            if cached_candidate is None:
                _run_stage(
                    f"exact_mdl_tournament_apply_{candidate_id}",
                    candidate_apply_command,
                    log_cb,
                    command_runner=_command_runner,
                    retry_native_crash=True,
                )
                _require_file(candidate_look_usd, "exact_mdl_tournament_apply")
                _require_file(candidate_apply_report, "exact_mdl_tournament_apply")
                candidate_apply_document = read_object(
                    candidate_apply_report,
                    f"tournament apply report {candidate_id}",
                )
                candidate_applied_count = candidate_apply_document.get("applied_count")
                if (
                    isinstance(candidate_applied_count, bool)
                    or not isinstance(candidate_applied_count, int)
                    or candidate_applied_count != applied_count
                ):
                    raise RuntimeError(
                        "Immutable MDL tournament changed exact coverage: "
                        f"baseline={applied_count} "
                        f"candidate={candidate_applied_count!r}"
                    )
                if (
                    candidate_apply_document.get("face_subset_count")
                    != expected_face_subset_count
                ):
                    raise RuntimeError(
                        "Immutable MDL tournament changed exact face-subset coverage"
                    )
                _run_stage(
                    f"exact_mdl_tournament_registry_{candidate_id}",
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
                    command_runner=_command_runner,
                    retry_native_crash=True,
                )
                _require_file(candidate_registry, "exact_mdl_tournament_registry")
                _run_stage(
                    f"exact_mdl_tournament_render_{candidate_id}",
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
                        str(config.render_resolution),
                        "--views",
                        ",".join(mapped_tournament_views),
                        "--rt-subframes",
                        str(config.render_rt_subframes),
                        "--lighting-profile",
                        "material-neutral",
                        "--analysis-up-axis",
                        config.analysis_up_axis,
                        f"--analysis-front-axis={config.analysis_front_axis}",
                    ],
                    log_cb,
                    command_runner=_command_runner,
                    retry_native_crash=True,
                )
                _require_file(
                    candidate_rendered_registry,
                    "exact_mdl_tournament_render",
                )
                _run_stage(
                    f"exact_mdl_tournament_compare_{candidate_id}",
                    [
                        str(config.qwen_python),
                        "-m",
                        "qwen_material_pipeline",
                        "compare",
                        "--reference-manifest",
                        str(analysis_dir / "reference_manifest.json"),
                        "--rendered-registry",
                        str(candidate_rendered_registry),
                        "--view-map",
                        str(exact_mdl_tournament_view_map),
                        "--minimum-comparable-views",
                        str(len(tournament_mapping)),
                        "--output",
                        str(candidate_quality_report),
                    ],
                    log_cb,
                    command_runner=_command_runner,
                )
                _require_file(
                    candidate_quality_report,
                    "exact_mdl_tournament_compare",
                )
                candidate_quality_document = read_object(
                    candidate_quality_report,
                    f"tournament quality report {candidate_id}",
                )
                candidate_rendered_registry_document = read_object(
                    candidate_rendered_registry,
                    f"tournament rendered registry {candidate_id}",
                )
                candidate_rendered_registry_file_sha256 = sha256_file(
                    candidate_rendered_registry
                )
            rendered_candidate_bundles.append(
                {
                    "candidate_id": candidate_id,
                    "plan": candidate_plan_document,
                    "apply_report": candidate_apply_document,
                    "quality_report": candidate_quality_document,
                    "rendered_registry": candidate_rendered_registry_document,
                    "rendered_registry_file_sha256": (
                        candidate_rendered_registry_file_sha256
                    ),
                }
            )
            candidate_artifacts[candidate_id] = {
                "plan": candidate_effective_plan_path,
                "look": candidate_look_usd,
                "apply": candidate_effective_apply_report,
                "quality": candidate_quality_report,
                "rendered_registry": candidate_rendered_registry,
            }
            cache_resume = planning_document["cache_resume"]
            assert isinstance(cache_resume, dict)
            cache_entries = cache_resume["entries"]
            assert isinstance(cache_entries, list)
            cache_entries.append(
                {
                    "candidate_id": candidate_id,
                    "status": candidate_cache_status,
                    "plan_sha256": canonical_sha256(candidate_plan_document),
                    "rendered_registry_file_sha256": (
                        candidate_rendered_registry_file_sha256
                    ),
                    "quality_report_sha256": canonical_sha256(
                        candidate_quality_document
                    ),
                    "archive": (
                        str(cache_archive) if cache_archive is not None else None
                    ),
                    "cache_miss_reason": (
                        cache_miss_reason if cached_candidate is None else None
                    ),
                    "candidate_cache_rebase": (
                        cached_candidate.get("cache_rebase")
                        if cached_candidate is not None
                        else None
                    ),
                }
            )
            cache_resume["cache_hit_count"] = tournament_cache_hits
            cache_resume["cache_miss_count"] = tournament_cache_misses
            write_object(exact_mdl_tournament_planning, planning_document)
            tournament_candidate_completed += 1
            _log_exact_mdl_candidate_progress(
                log_cb,
                state="complete",
                group_index=1,
                group_total=1,
                candidate_index=candidate_index,
                candidate_total=candidate_total,
                candidate_id=candidate_id,
                global_current=tournament_candidate_completed,
                global_total=candidate_total,
                cache_status=candidate_cache_status,
            )
        if baseline_candidate_plan is None:
            raise AssertionError("exact MDL tournament planned no candidates")
        family_contract = build_part_family_contract(
            plan=current_plan_document,
            material_choice_audit=material_choice_document,
            palette_fusion=palette_fusion_document,
        )
        try:
            (
                selected_plan_document,
                tournament_audit_document,
            ) = select_and_replay_exact_mdl_candidate(
                baseline_plan=baseline_candidate_plan,
                target_plan=current_plan_document,
                candidates=rendered_candidate_bundles,
                allowed_material_ids=set(material_families_by_id),
                material_families_by_id=material_families_by_id,
                allowed_families_by_part=family_contract,
                selection_objective=config.material_selection_objective,
            )
        except ExactMdlTournamentError as exc:
            exact_mdl_tournament_status = "NO_ELIGIBLE_CANDIDATE"
            if exc.audit is not None:
                write_object(exact_mdl_tournament_audit, exc.audit)
            raise RuntimeError(
                "No exact NVIDIA MDL satisfying the configured selection "
                "objective passed every "
                f"registered reference view; physics was not started: {exc}"
            ) from exc
        tournament_audit_document["cache_resume"] = planning_document["cache_resume"]
        if single_disagreement_contract is not None and isinstance(
            selected_single_group_id, str
        ):
            # The bounded selector returns only an all-view-PASS exact MDL;
            # otherwise it raises before reaching this point.
            tournament_audit_document["forward_reverse_disagreement_resolution"] = {
                "required": True,
                "resolved": True,
                "resolution_policy": single_disagreement_contract["resolution_policy"],
                "required_candidate_material_ids": (
                    single_disagreement_contract["required_candidate_material_ids"]
                ),
            }
            render_confirmed_disagreement_group_ids.add(selected_single_group_id)
        write_object(exact_mdl_tournament_plan, selected_plan_document)
        write_object(exact_mdl_tournament_audit, tournament_audit_document)
        planning_document["status"] = "COMPLETED"
        write_object(exact_mdl_tournament_planning, planning_document)
        _log_exact_mdl_group_progress(
            log_cb,
            state="complete",
            current=1,
            total=1,
            group_id=str(planning_document.get("selected_group_id", "dominant")),
        )
        selected_candidate_id = tournament_audit_document.get("selected_candidate_id")
        if (
            not isinstance(selected_candidate_id, str)
            or selected_candidate_id not in candidate_artifacts
        ):
            raise RuntimeError("Immutable MDL tournament selected an unknown candidate")
        selected_artifacts = candidate_artifacts[selected_candidate_id]
        selected_apply_document = read_object(
            selected_artifacts["apply"],
            "selected tournament apply report",
        )
        selected_applied_count = selected_apply_document.get("applied_count")
        if (
            isinstance(selected_applied_count, bool)
            or not isinstance(selected_applied_count, int)
            or selected_applied_count != applied_count
        ):
            raise RuntimeError(
                "Selected immutable MDL candidate changed exact coverage"
            )
        if instance_root_count:
            selected_instance_document = dict(selected_plan_document)
            selected_instance_provenance = selected_instance_document.get("provenance")
            sealed_selected_provenance = (
                dict(selected_instance_provenance)
                if isinstance(selected_instance_provenance, dict)
                else {}
            )
            sealed_selected_provenance.update(
                {
                    "asset_sha256": sha256_file(source),
                    "registry_sha256": canonical_sha256(
                        read_object(
                            rendered_registry,
                            "selected tournament occurrence registry",
                        )
                    ),
                }
            )
            selected_instance_document["provenance"] = sealed_selected_provenance
            write_object(
                exact_mdl_tournament_instance_plan,
                selected_instance_document,
            )
            final_instance_plan = exact_mdl_tournament_instance_plan
        effective_material_plan = exact_mdl_tournament_plan
        effective_look_usd = selected_artifacts["look"]
        effective_apply_report = selected_artifacts["apply"]
        effective_quality_report = selected_artifacts["quality"]
        effective_quality_rendered_registry = selected_artifacts["rendered_registry"]
        applied_count = selected_applied_count
        quality_status = "PASS"
        quality_gate_status = "PASS"
        quality_resolution_document = None
        quality_limitation_count = 0
        quality_decision = "ACCEPTED_AFTER_EXACT_MDL_RENDER_TOURNAMENT"
        quality_round_count += 1
        exact_mdl_tournament_status = "SELECTED"
        exact_mdl_tournament_selected_candidate_id = selected_candidate_id
        exact_mdl_tournament_selected_candidate_ids = [selected_candidate_id]
        exact_mdl_tournament_group_count = 1
        log_message(
            log_cb,
            "An NVIDIA MDL passed every registered reference view and won "
            "the immutable exact-material tournament under objective "
            f"{config.material_selection_objective}.",
        )

    # A material-neutral QA render can support one additional, bounded shared
    # parameter step.  The measurement phase first divides target luminance by
    # a neutral anchor independently in each reference/render pair.  No
    # candidate is authored unless that phase emitted the strict statistics
    # contract, and no candidate replaces the already accepted Look unless a
    # second neutral render proves objective improvement.
    appearance_status = "SKIPPED_INPUTS_UNAVAILABLE"
    appearance_adjustment_count = 0
    appearance_changed_count = 0
    appearance_candidate_round_count = 0
    appearance_error: str | None = None
    appearance_reason_codes: list[str] = []
    appearance_measurement_used = False
    appearance_contract_document: dict[str, Any] | None = None
    appearance_validation_document: dict[str, Any] | None = None
    appearance_inputs = (
        mvinverse_pbr_evidence,
        analysis_dir / "palette_fusion.json",
        spatial_report_path,
        effective_material_plan,
        effective_quality_report,
        effective_quality_rendered_registry,
    )
    baseline_quality_document = read_object(
        effective_quality_report, "effective visual quality report"
    )
    appearance_inputs_available = all(path.is_file() for path in appearance_inputs)
    appearance_quality_measurable = _quality_can_measure_lighting_statistics(
        baseline_quality_document
    )
    baseline_lighting_profile: str | None = None
    if effective_quality_rendered_registry.is_file():
        baseline_rendered_registry_document = read_object(
            effective_quality_rendered_registry,
            "effective quality rendered registry",
        )
        baseline_render_set = baseline_rendered_registry_document.get("render_set")
        if isinstance(baseline_render_set, dict) and isinstance(
            baseline_render_set.get("lighting_profile"), str
        ):
            baseline_lighting_profile = baseline_render_set["lighting_profile"]
    appearance_baseline_safety_reason = _appearance_baseline_safety_reason(
        quality_gate_status=quality_gate_status,
        lighting_profile=baseline_lighting_profile,
    )
    if config.material_assignment_unit == "part_id":
        appearance_status = "NOT_APPLICABLE"
        appearance_reason_codes.append(
            "PALETTE_GROUP_APPEARANCE_OPTIMIZATION_NOT_APPLICABLE_TO_PART_ID"
        )
    elif config.immutable_mdl_after_selection:
        appearance_status = "SKIPPED_SELECTED_MDL_IMMUTABLE"
        appearance_reason_codes.append(
            "SELECTED_MDL_IDENTITY_AND_PARAMETERS_ARE_LOCKED"
        )
    elif not appearance_inputs_available:
        appearance_reason_codes.append("APPEARANCE_INPUT_ARTIFACTS_UNAVAILABLE")
    elif not appearance_quality_measurable:
        appearance_reason_codes.append("QUALITY_IMAGE_MEASUREMENT_CONTRACT_UNAVAILABLE")
    elif appearance_baseline_safety_reason is not None:
        # A LIMITED_PASS resolution is hash-bound to its exact plan and QA
        # report.  Replacing that plan without rerunning the entire pose
        # resolution would leave stale evidence.  Likewise, baseline and
        # candidate must share the orientation-stable material-neutral light;
        # mixing profiles would turn illumination into false improvement.
        appearance_status = "SKIPPED_UNSAFE_BASELINE"
        appearance_reason_codes.append(appearance_baseline_safety_reason)
    else:
        try:
            _run_stage(
                "appearance_optimization_measure_baseline",
                [
                    str(config.qwen_python),
                    "-m",
                    ("qwen_material_pipeline.materials.appearance_optimization"),
                    "measure",
                    "--final-plan",
                    str(effective_material_plan),
                    "--quality-report",
                    str(effective_quality_report),
                    "--mvinverse-evidence",
                    str(mvinverse_pbr_evidence),
                    "--palette-fusion",
                    str(analysis_dir / "palette_fusion.json"),
                    "--spatial-report",
                    str(spatial_report_path),
                    "--rendered-registry",
                    str(effective_quality_rendered_registry),
                    "--output-quality-report",
                    str(appearance_baseline_quality),
                    "--output-report",
                    str(appearance_baseline_measurement),
                ],
                log_cb,
                command_runner=_command_runner,
            )
            _require_file(
                appearance_baseline_quality,
                "appearance_optimization_measure_baseline",
            )
            _require_file(
                appearance_baseline_measurement,
                "appearance_optimization_measure_baseline",
            )
            appearance_measurement_used = True
            measured_baseline_quality = read_object(
                appearance_baseline_quality,
                "lighting-normalized baseline quality report",
            )
            if not _quality_has_lighting_normalized_groups(measured_baseline_quality):
                appearance_status = "SKIPPED_NO_LIGHTING_NORMALIZED_GROUPS"
                appearance_reason_codes.append("NO_LIGHTING_NORMALIZED_GROUPS_MEASURED")
            else:
                _run_stage(
                    "appearance_optimization_build",
                    [
                        str(config.qwen_python),
                        "-m",
                        ("qwen_material_pipeline.materials.appearance_optimization"),
                        "build",
                        "--final-plan",
                        str(effective_material_plan),
                        "--quality-report",
                        str(appearance_baseline_quality),
                        "--mvinverse-evidence",
                        str(mvinverse_pbr_evidence),
                        "--rendered-registry",
                        str(effective_quality_rendered_registry),
                        "--palette-fusion",
                        str(analysis_dir / "palette_fusion.json"),
                        "--output",
                        str(appearance_contract),
                    ],
                    log_cb,
                    command_runner=_command_runner,
                )
                _require_file(appearance_contract, "appearance_optimization_build")
                appearance_contract_document = read_object(
                    appearance_contract, "appearance optimization contract"
                )
                contract_summary = appearance_contract_document.get("summary")
                if not isinstance(contract_summary, dict):
                    raise RuntimeError(
                        "Appearance optimization contract lacks its summary"
                    )
                appearance_adjustment_count = _require_exact_int(
                    contract_summary.get("adjustment_count"),
                    "Appearance optimization adjustment_count",
                )
                if appearance_adjustment_count == 0:
                    # The contract itself is the audit: it distinguishes
                    # preserve, inconsistent-lighting and insufficient lanes.
                    appearance_status = "NOT_APPLICABLE"
                    appearance_reason_codes.append(
                        "OPTIMIZATION_CONTRACT_HAS_NO_ADJUSTMENT"
                    )
                else:
                    appearance_candidate_round_count = 1
                    quality_round_count += 1
                    _run_stage(
                        "appearance_optimization_apply_plan",
                        [
                            str(config.qwen_python),
                            "-m",
                            (
                                "qwen_material_pipeline.materials."
                                "appearance_optimization"
                            ),
                            "apply",
                            "--final-plan",
                            str(effective_material_plan),
                            "--contract",
                            str(appearance_contract),
                            "--output-plan",
                            str(appearance_candidate_plan),
                            "--output-report",
                            str(appearance_candidate_plan_apply),
                        ],
                        log_cb,
                        command_runner=_command_runner,
                        retry_native_crash=True,
                    )
                    _require_file(
                        appearance_candidate_plan,
                        "appearance_optimization_apply_plan",
                    )
                    _require_file(
                        appearance_candidate_plan_apply,
                        "appearance_optimization_apply_plan",
                    )
                    candidate_plan_document = read_object(
                        appearance_candidate_plan,
                        "appearance optimization candidate plan",
                    )
                    candidate_plan_apply_document = read_object(
                        appearance_candidate_plan_apply,
                        "appearance optimization plan-apply report",
                    )
                    appearance_changed_count = _require_exact_int(
                        candidate_plan_apply_document.get("changed_part_count"),
                        "Appearance optimization changed_part_count",
                        minimum=1,
                    )

                    appearance_apply_plan = appearance_candidate_plan
                    candidate_instance_plan_written = False
                    if instance_root_count:
                        if rendered_registry_document is None:
                            raise AssertionError("instance registry was not loaded")
                        candidate_instance_document = dict(candidate_plan_document)
                        candidate_provenance = candidate_instance_document.get(
                            "provenance"
                        )
                        sealed_candidate_provenance = (
                            dict(candidate_provenance)
                            if isinstance(candidate_provenance, dict)
                            else {}
                        )
                        sealed_candidate_provenance.update(
                            {
                                "asset_sha256": sha256_file(source),
                                "registry_sha256": canonical_sha256(
                                    rendered_registry_document
                                ),
                            }
                        )
                        candidate_instance_document[
                            "provenance"
                        ] = sealed_candidate_provenance
                        write_object(
                            appearance_candidate_instance_plan,
                            candidate_instance_document,
                        )
                        appearance_apply_plan = appearance_candidate_instance_plan
                        candidate_instance_plan_written = True

                    appearance_usd_apply_command = [
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
                        str(appearance_apply_plan),
                        "--output",
                        str(appearance_candidate_look_usd),
                        "--material-root",
                        str(config.material_root),
                        "--report",
                        str(appearance_candidate_usd_apply),
                        "--include-review",
                    ]
                    if use_policy_fallback:
                        appearance_usd_apply_command.append("--include-policy-fallback")
                    _run_stage(
                        "appearance_optimization_apply_usd",
                        appearance_usd_apply_command,
                        log_cb,
                        command_runner=_command_runner,
                        retry_native_crash=True,
                    )
                    _require_file(
                        appearance_candidate_look_usd,
                        "appearance_optimization_apply_usd",
                    )
                    _require_file(
                        appearance_candidate_usd_apply,
                        "appearance_optimization_apply_usd",
                    )
                    appearance_usd_apply_document = read_object(
                        appearance_candidate_usd_apply,
                        "appearance optimization USD apply report",
                    )
                    candidate_applied_count = _require_exact_int(
                        appearance_usd_apply_document.get("applied_count"),
                        "Appearance optimization applied_count",
                        minimum=1,
                    )
                    if candidate_applied_count != applied_count:
                        raise RuntimeError(
                            "Appearance optimization changed the applied part "
                            f"count: baseline={applied_count} "
                            f"candidate={candidate_applied_count}"
                        )

                    _run_stage(
                        "appearance_optimization_registry",
                        [
                            str(isaac),
                            "-m",
                            "qwen_material_pipeline",
                            "usd",
                            "registry",
                            "--usd",
                            str(appearance_candidate_look_usd),
                            "--output",
                            str(appearance_candidate_quality_registry),
                        ],
                        log_cb,
                        command_runner=_command_runner,
                        retry_native_crash=True,
                    )
                    _require_file(
                        appearance_candidate_quality_registry,
                        "appearance_optimization_registry",
                    )
                    _run_stage(
                        "appearance_optimization_render",
                        [
                            str(isaac),
                            "-m",
                            "qwen_material_pipeline",
                            "usd",
                            "render",
                            "--registry",
                            str(appearance_candidate_quality_registry),
                            "--output-dir",
                            str(appearance_candidate_render_dir),
                            "--resolution",
                            str(config.render_resolution),
                            *quality_render_view_arguments,
                            "--rt-subframes",
                            str(config.render_rt_subframes),
                            "--lighting-profile",
                            "material-neutral",
                            "--analysis-up-axis",
                            config.analysis_up_axis,
                            f"--analysis-front-axis={config.analysis_front_axis}",
                        ],
                        log_cb,
                        command_runner=_command_runner,
                        retry_native_crash=True,
                    )
                    _require_file(
                        appearance_candidate_rendered_registry,
                        "appearance_optimization_render",
                    )
                    candidate_rendered_registry_document = read_object(
                        appearance_candidate_rendered_registry,
                        "appearance candidate rendered registry",
                    )
                    candidate_render_set = candidate_rendered_registry_document.get(
                        "render_set"
                    )
                    if (
                        not isinstance(candidate_render_set, dict)
                        or candidate_render_set.get("lighting_profile")
                        != "material-neutral"
                    ):
                        raise RuntimeError(
                            "Appearance candidate render did not preserve the "
                            "material-neutral lighting contract"
                        )
                    appearance_compare_command = [
                        str(config.qwen_python),
                        "-m",
                        "qwen_material_pipeline",
                        "compare",
                        "--reference-manifest",
                        str(analysis_dir / "reference_manifest.json"),
                        "--rendered-registry",
                        str(appearance_candidate_rendered_registry),
                        "--output",
                        str(appearance_candidate_raw_quality),
                        "--minimum-comparable-views",
                        "2",
                    ]
                    if len(trusted_mapping) >= 2:
                        write_object(
                            appearance_candidate_view_map,
                            {
                                "schema_version": ("qwen-reference-view-map/v1"),
                                "mapping": dict(sorted(trusted_mapping.items())),
                                "source": "trusted_spatial_registration",
                            },
                        )
                        appearance_compare_command.extend(
                            [
                                "--view-map",
                                str(appearance_candidate_view_map),
                            ]
                        )
                    _run_stage(
                        "appearance_optimization_compare",
                        appearance_compare_command,
                        log_cb,
                        command_runner=_command_runner,
                    )
                    _require_file(
                        appearance_candidate_raw_quality,
                        "appearance_optimization_compare",
                    )
                    _run_stage(
                        "appearance_optimization_measure_candidate",
                        [
                            str(config.qwen_python),
                            "-m",
                            (
                                "qwen_material_pipeline.materials."
                                "appearance_optimization"
                            ),
                            "measure",
                            "--final-plan",
                            str(appearance_candidate_plan),
                            "--quality-report",
                            str(appearance_candidate_raw_quality),
                            "--mvinverse-evidence",
                            str(mvinverse_pbr_evidence),
                            "--palette-fusion",
                            str(analysis_dir / "palette_fusion.json"),
                            "--spatial-report",
                            str(spatial_report_path),
                            "--rendered-registry",
                            str(appearance_candidate_rendered_registry),
                            "--output-quality-report",
                            str(appearance_candidate_quality),
                            "--output-report",
                            str(appearance_candidate_measurement),
                        ],
                        log_cb,
                        command_runner=_command_runner,
                    )
                    _require_file(
                        appearance_candidate_quality,
                        "appearance_optimization_measure_candidate",
                    )
                    _require_file(
                        appearance_candidate_measurement,
                        "appearance_optimization_measure_candidate",
                    )
                    _run_stage(
                        "appearance_optimization_validate",
                        [
                            str(config.qwen_python),
                            "-m",
                            (
                                "qwen_material_pipeline.materials."
                                "appearance_optimization"
                            ),
                            "validate",
                            "--source-plan",
                            str(effective_material_plan),
                            "--contract",
                            str(appearance_contract),
                            "--candidate-plan",
                            str(appearance_candidate_plan),
                            "--candidate-quality-report",
                            str(appearance_candidate_quality),
                            "--output",
                            str(appearance_validation),
                        ],
                        log_cb,
                        command_runner=_command_runner,
                    )
                    _require_file(
                        appearance_validation,
                        "appearance_optimization_validate",
                    )
                    appearance_validation_document = read_object(
                        appearance_validation,
                        "appearance optimization validation",
                    )
                    validation_status = appearance_validation_document.get("status")
                    if validation_status not in {"PASS", "FAIL_CLOSED"}:
                        raise RuntimeError(
                            "Appearance optimization validation has an "
                            f"unsupported status: {validation_status!r}"
                        )
                    if validation_status == "PASS":
                        candidate_quality_document = read_object(
                            appearance_candidate_quality,
                            "appearance candidate quality report",
                        )
                        candidate_aggregate = candidate_quality_document.get(
                            "aggregate"
                        )
                        if not isinstance(candidate_aggregate, dict):
                            raise RuntimeError(
                                "Appearance candidate quality report lacks "
                                "its aggregate"
                            )
                        candidate_quality_status = candidate_aggregate.get("status")
                        if candidate_quality_status not in QUALITY_STATUSES:
                            raise RuntimeError(
                                "Appearance candidate quality report has an "
                                "unsupported aggregate status"
                            )
                        if candidate_quality_status != "PASS":
                            appearance_status = "REJECTED_FAIL_CLOSED"
                            appearance_reason_codes.append(
                                "CANDIDATE_QUALITY_GATE_IS_NOT_PASS"
                            )
                            log_message(
                                log_cb,
                                "Appearance optimization validation passed its "
                                "parameter objective, but aggregate visual QA "
                                "did not PASS; the previous Look remains effective.",
                            )
                        else:
                            appearance_status = "ACCEPTED"
                            effective_material_plan = appearance_candidate_plan
                            effective_look_usd = appearance_candidate_look_usd
                            effective_apply_report = appearance_candidate_usd_apply
                            effective_quality_report = appearance_candidate_quality
                            effective_quality_rendered_registry = (
                                appearance_candidate_rendered_registry
                            )
                            applied_count = candidate_applied_count
                            quality_status = candidate_quality_status
                            quality_gate_status = "PASS"
                            # A PASS candidate has its own independently
                            # validated QA report; never expose a resolution
                            # artifact hash-bound to the previous plan/report.
                            quality_resolution_document = None
                            quality_limitation_count = 0
                            if candidate_instance_plan_written:
                                final_instance_plan = appearance_candidate_instance_plan
                            quality_decision = (
                                "ACCEPTED_AFTER_SHARED_MATERIAL_APPEARANCE_OPTIMIZATION"
                            )
                    else:
                        appearance_status = "REJECTED_FAIL_CLOSED"
                        appearance_reason_codes.append("VALIDATION_REJECTED_CANDIDATE")
                        log_message(
                            log_cb,
                            "Appearance optimization candidate was rejected; "
                            "the previously accepted Look remains effective.",
                        )
        except (RuntimeError, ValueError) as exc:
            # This round is an optional, isolated refinement.  Its candidate
            # never becomes an input to the accepted Look until validation
            # passes, so retaining the baseline is the correct fail-closed
            # behavior for renderer/measurement/validation failures.
            appearance_status = "REJECTED_FAIL_CLOSED"
            appearance_error = str(exc)
            appearance_reason_codes.append(
                (
                    "CANDIDATE_STAGE_FAILED"
                    if appearance_candidate_round_count
                    else "BASELINE_APPEARANCE_STAGE_FAILED"
                )
            )
            if not appearance_validation.is_file():
                write_object(
                    appearance_validation,
                    {
                        "schema_version": (
                            "asset-pipeline-appearance-candidate-discard/v1"
                        ),
                        "status": "FAIL_CLOSED",
                        "reason_codes": list(appearance_reason_codes),
                        "error": appearance_error,
                    },
                )
                appearance_validation_document = read_object(
                    appearance_validation,
                    "appearance candidate discard audit",
                )
            log_message(
                log_cb,
                "Appearance optimization was discarded safely; the previously "
                f"accepted Look remains effective ({appearance_error}).",
            )

    if appearance_status not in APPEARANCE_OPTIMIZATION_STATUSES:
        raise AssertionError(
            f"Unsupported appearance optimization status: {appearance_status}"
        )

    return MaterialSelectionStageResult(
        exact_mdl_tournament_status=exact_mdl_tournament_status,
        exact_mdl_tournament_candidate_count=exact_mdl_tournament_candidate_count,
        exact_mdl_tournament_selected_candidate_id=(
            exact_mdl_tournament_selected_candidate_id
        ),
        exact_mdl_tournament_selected_candidate_ids=(
            exact_mdl_tournament_selected_candidate_ids
        ),
        exact_mdl_tournament_group_count=exact_mdl_tournament_group_count,
        render_confirmed_disagreement_group_ids=(
            render_confirmed_disagreement_group_ids
        ),
        baseline_preserved_disagreement_exemptions=(
            baseline_preserved_disagreement_exemptions
        ),
        visual_group_annotation_status=visual_group_annotation_status,
        quality_round_count=quality_round_count,
        final_instance_plan=final_instance_plan,
        effective_material_plan=effective_material_plan,
        effective_look_usd=effective_look_usd,
        effective_apply_report=effective_apply_report,
        effective_quality_report=effective_quality_report,
        effective_quality_rendered_registry=(effective_quality_rendered_registry),
        applied_count=applied_count,
        quality_status=quality_status,
        quality_gate_status=quality_gate_status,
        quality_resolution_document=quality_resolution_document,
        quality_limitation_count=quality_limitation_count,
        quality_decision=quality_decision,
        appearance_status=appearance_status,
        appearance_adjustment_count=appearance_adjustment_count,
        appearance_changed_count=appearance_changed_count,
        appearance_candidate_round_count=appearance_candidate_round_count,
        appearance_error=appearance_error,
        appearance_reason_codes=appearance_reason_codes,
        appearance_measurement_used=appearance_measurement_used,
        baseline_lighting_profile=baseline_lighting_profile,
    )


def _run_finalize_assignment_stage(
    context: VisualMaterialPipelineContext,
    *,
    prepared_source: SourcePreparationResult,
    planning: PolicyPartIdStageResult,
    look: LookApplicationStageResult,
    visual_qa: VisualQaStageResult,
    selection: MaterialSelectionStageResult,
    require_complete_coverage: bool,
    log_cb: LogCallback,
    command_runner: CommandRunner,
) -> dict[str, Any]:
    """Seal the selected Look, publish its gates and build the public result."""

    source = context.source
    resolved_source_cad = context.source_cad
    parsed_references = context.references
    resolved_foreground_annotations = context.foreground_annotations
    effective_config_path = context.config_path
    config = context.config
    isaac = context.isaac_python
    destination = context.destination
    workspace = context.workspace
    source_registry = prepared_source.source_registry
    registry = prepared_source.registry
    rendered_registry = prepared_source.rendered_registry
    editable_usd = prepared_source.editable_usd
    expand_report = prepared_source.expand_report
    instance_root_count = prepared_source.instance_root_count
    effective_catalog = planning.effective_catalog
    live_material_count = planning.live_material_count
    state = planning.state
    use_policy_fallback = planning.use_policy_fallback
    policy_fallback_count = planning.policy_fallback_count
    policy_audit_document = planning.policy_audit_document
    selection_lock_document = planning.selection_lock_document
    assignments = visual_qa.assignments
    parameter_tournament_document = visual_qa.parameter_tournament_document
    component_actual_mdl_tournament_document = (
        visual_qa.component_actual_mdl_tournament_document
    )
    corresponding_color_calibration_document = (
        visual_qa.corresponding_color_calibration_document
    )
    raw_quality_status = visual_qa.raw_quality_status
    part_id_quality_gate_document = visual_qa.part_id_quality_gate_document
    quality_repair_used = visual_qa.quality_repair_used
    quality_repair_changed_count = visual_qa.quality_repair_changed_count
    quality_repair_round_count = visual_qa.quality_repair_round_count
    membership_tournament_status = visual_qa.membership_tournament_status
    membership_tournament_cohort_count = visual_qa.membership_tournament_cohort_count
    membership_selected_expanded_count = visual_qa.membership_selected_expanded_count
    membership_restored_m0_count = visual_qa.membership_restored_m0_count
    apply_subcommand = look.apply_subcommand
    apply_asset_flag = look.apply_asset_flag
    apply_asset = look.apply_asset
    effective_material_plan = selection.effective_material_plan
    effective_look_usd = selection.effective_look_usd
    effective_apply_report = selection.effective_apply_report
    effective_quality_report = selection.effective_quality_report
    effective_quality_rendered_registry = selection.effective_quality_rendered_registry
    applied_count = selection.applied_count
    quality_status = selection.quality_status
    quality_gate_status = selection.quality_gate_status
    quality_resolution_document = selection.quality_resolution_document
    quality_limitation_count = selection.quality_limitation_count
    quality_decision = selection.quality_decision
    quality_round_count = selection.quality_round_count
    final_instance_plan = selection.final_instance_plan
    exact_mdl_tournament_status = selection.exact_mdl_tournament_status
    exact_mdl_tournament_candidate_count = (
        selection.exact_mdl_tournament_candidate_count
    )
    exact_mdl_tournament_selected_candidate_id = (
        selection.exact_mdl_tournament_selected_candidate_id
    )
    exact_mdl_tournament_selected_candidate_ids = (
        selection.exact_mdl_tournament_selected_candidate_ids
    )
    exact_mdl_tournament_group_count = selection.exact_mdl_tournament_group_count
    render_confirmed_disagreement_group_ids = (
        selection.render_confirmed_disagreement_group_ids
    )
    baseline_preserved_disagreement_exemptions = (
        selection.baseline_preserved_disagreement_exemptions
    )
    visual_group_annotation_status = selection.visual_group_annotation_status
    appearance_status = selection.appearance_status
    appearance_adjustment_count = selection.appearance_adjustment_count
    appearance_changed_count = selection.appearance_changed_count
    appearance_candidate_round_count = selection.appearance_candidate_round_count
    appearance_error = selection.appearance_error
    appearance_reason_codes = selection.appearance_reason_codes
    appearance_measurement_used = selection.appearance_measurement_used
    baseline_lighting_profile = selection.baseline_lighting_profile
    inference_paths = workspace.inference
    analysis_dir = inference_paths.root
    unattended_result = inference_paths.unattended_result
    confidence_gate = inference_paths.confidence_gate
    staged_material_plan = inference_paths.staged_material_plan
    policy_input = inference_paths.policy_input
    policy_plan = inference_paths.policy_plan
    policy_audit = inference_paths.policy_audit
    publish_quality_gate_report = inference_paths.publish_quality_gate
    immutable_library_optimum_gate = inference_paths.immutable_library_optimum_gate
    material_selection_lock = inference_paths.material_selection_lock
    mvinverse_ledger = inference_paths.mvinverse_ledger
    inference_recovery = inference_paths.inference_recovery
    finalized_look_usd = workspace.look.locked_usd
    finalized_apply_report = workspace.look.locked_apply_report
    part_id_paths = workspace.part_id
    part_id_evidence_path = part_id_paths.evidence
    part_id_retrieval_result = part_id_paths.retrieval_result
    part_id_qwen_result = part_id_paths.qwen_result
    part_id_material_audit = part_id_paths.material_audit
    part_id_quality_gate = part_id_paths.quality_gate
    part_id_parameter_tournament_audit = part_id_paths.parameter_tournament_audit
    appearance_paths = workspace.appearance
    appearance_components_report = appearance_paths.components
    appearance_component_evidence = appearance_paths.evidence
    appearance_component_retrieval_result = appearance_paths.retrieval_result
    appearance_component_qwen_result = appearance_paths.qwen_result
    appearance_component_mdl_selection_audit = appearance_paths.mdl_selection_audit
    appearance_component_actual_mdl_tournament_audit = (
        appearance_paths.actual_mdl_tournament_audit
    )
    legacy_paths = workspace.legacy
    visual_group_annotated_plan = legacy_paths.visual_group_plan
    visual_group_annotation_audit = legacy_paths.visual_group_audit
    exact_mdl_tournament_planning = legacy_paths.exact_mdl_planning
    exact_mdl_tournament_audit = legacy_paths.exact_mdl_audit
    membership_tournament_plan = legacy_paths.membership_plan
    membership_tournament_audit = legacy_paths.membership_audit
    quality_repair_plan = legacy_paths.quality_repair_plan
    quality_repair_audit = legacy_paths.quality_repair_audit
    quality_resolution = legacy_paths.quality_resolution
    appearance_baseline_quality = legacy_paths.appearance_baseline_quality
    appearance_baseline_measurement = legacy_paths.appearance_baseline_measurement
    appearance_contract = legacy_paths.appearance_contract
    appearance_candidate_plan = legacy_paths.appearance_candidate_plan
    appearance_candidate_plan_apply = legacy_paths.appearance_candidate_plan_apply
    appearance_candidate_usd_apply = legacy_paths.appearance_candidate_apply_report
    appearance_candidate_raw_quality = legacy_paths.appearance_candidate_raw_quality
    appearance_candidate_quality = legacy_paths.appearance_candidate_measured_quality
    appearance_candidate_measurement = legacy_paths.appearance_candidate_measurement
    appearance_validation = legacy_paths.appearance_validation
    quality_report = workspace.quality.report
    _command_runner = command_runner

    publish_quality_gate_document: dict[str, Any] | None = None
    publish_gate_required = bool(
        config.immutable_mdl_after_selection
        or require_complete_coverage
        or use_policy_fallback
    )
    if publish_gate_required:
        publish_annotation_document = (
            read_object(
                visual_group_annotation_audit,
                "pre-publish visual group annotation audit",
            )
            if visual_group_annotation_audit.is_file()
            else None
        )
        publish_queue_document: Mapping[str, Any] | None = None
        if exact_mdl_tournament_planning.is_file():
            publish_planning_document = read_object(
                exact_mdl_tournament_planning,
                "pre-publish exact MDL tournament planning",
            )
            raw_publish_queue = publish_planning_document.get("queue")
            if not isinstance(raw_publish_queue, Mapping):
                raise RuntimeError(
                    "Exact-MDL planning lacks its pre-publish queue audit; "
                    "physics was not started"
                )
            publish_queue_document = raw_publish_queue
        try:
            publish_quality_gate_document = build_publish_quality_gate(
                confidence_gate=read_object(
                    confidence_gate,
                    "pre-publish material confidence gate",
                ),
                final_plan=read_object(
                    effective_material_plan,
                    "pre-publish final effective material plan",
                ),
                annotation_audit=publish_annotation_document,
                policy_audit=(
                    read_object(policy_audit, "pre-publish policy exact-cover audit")
                    if use_policy_fallback
                    else None
                ),
                policy_plan=(
                    read_object(policy_plan, "pre-publish policy exact-cover plan")
                    if use_policy_fallback
                    and config.material_assignment_unit == "part_id"
                    else None
                ),
                part_id_material_audit=(
                    read_object(
                        part_id_material_audit,
                        "pre-publish Part-ID material audit",
                    )
                    if config.material_assignment_unit == "part_id"
                    else None
                ),
                queue_audit=publish_queue_document,
                tournament_audit=(
                    read_object(
                        exact_mdl_tournament_audit,
                        "pre-publish exact MDL tournament audit",
                    )
                    if exact_mdl_tournament_audit.is_file()
                    else None
                ),
                rendered_registry=read_object(
                    rendered_registry,
                    "pre-publish rendered part registry",
                ),
                spatial_mapping_report=read_object(
                    analysis_dir / "spatial_mapping_report.json",
                    "pre-publish spatial mapping report",
                ),
                policy=PublishQualityPolicy(
                    maximum_policy_fallback_fraction=(
                        config.final_visual_gate_maximum_policy_fallback_fraction
                    ),
                    maximum_neutral_fallback_fraction=(
                        config.final_visual_gate_maximum_neutral_fallback_fraction
                    ),
                    maximum_unresolved_entity_fraction=(
                        config.final_visual_gate_maximum_unresolved_entity_fraction
                    ),
                    maximum_unresolved_face_subset_fraction=(
                        config.final_visual_gate_maximum_unresolved_face_subset_fraction
                    ),
                    minimum_owner_local_resolved_fraction=(
                        config.final_visual_gate_minimum_owner_local_resolved_fraction
                    ),
                    maximum_visible_fallback_fraction=(
                        config.final_visual_gate_maximum_visible_fallback_fraction
                    ),
                ),
            )
        except PublishQualityGateError as exc:
            raise RuntimeError(
                "Unable to validate automatic material publication coverage; "
                f"physics was not started: {exc}"
            ) from exc
        write_object(publish_quality_gate_report, publish_quality_gate_document)
        try:
            require_publish_quality_gate_passed(publish_quality_gate_document)
        except PublishQualityGateError as exc:
            raise RuntimeError(
                "Automatic material selection failed the fail-closed publication "
                "coverage gate; no MDL lock was created and physics was not "
                f"started: {exc}"
            ) from exc
        log_message(
            log_cb,
            "Automatic material publication coverage gate passed; confidence, "
            "fallback, unresolved-entity and owner-local evidence are within policy.",
        )

    if config.immutable_mdl_after_selection:
        final_material_choice_audit = (
            read_object(
                analysis_dir / "material_choice_audit.json",
                "final material choice audit",
            )
            if _palette_group_disagreement_contract_applies(
                config.material_assignment_unit
            )
            else {}
        )
        disagreement_choices = {
            str(group_id): raw_choice
            for group_id, raw_choice in final_material_choice_audit.items()
            if (
                isinstance(group_id, str)
                and isinstance(raw_choice, Mapping)
                and raw_choice.get("confirmation_basis")
                == "forward_reverse_disagreement"
            )
        }
        baseline_preserved_disagreement_group_ids = {
            str(record["group_id"])
            for record in baseline_preserved_disagreement_exemptions
            if (
                isinstance(record, Mapping)
                and isinstance(record.get("group_id"), str)
                and record.get("render_confirmation_required") is False
                and record.get("material_selection_was_authored") is False
                and record.get("material_choice_resolved") is False
                and record.get("presence_only_not_material_identity_confirmation")
                is True
                and record.get("final_state_revalidated") is True
                and record.get("final_authored_target_entity_count") == 0
            )
        }
        required_disagreement_choices = {
            group_id: choice
            for group_id, choice in disagreement_choices.items()
            if group_id not in baseline_preserved_disagreement_group_ids
        }
        missing_disagreement_contracts = sorted(
            group_id
            for group_id, raw_choice in required_disagreement_choices.items()
            if (
                not isinstance(
                    raw_choice.get("disagreement_tournament"),
                    Mapping,
                )
                or raw_choice["disagreement_tournament"].get("required") is not True
            )
        )
        if missing_disagreement_contracts:
            raise RuntimeError(
                "Refusing to lock forward/reverse NVIDIA MDL disagreements "
                "without an immutable visual-tournament contract: "
                f"{missing_disagreement_contracts}"
            )
        required_disagreement_group_ids = set(required_disagreement_choices)
        unresolved_disagreement_group_ids = sorted(
            required_disagreement_group_ids - render_confirmed_disagreement_group_ids
        )
        if unresolved_disagreement_group_ids:
            raise RuntimeError(
                "Refusing to lock unresolved forward/reverse NVIDIA MDL "
                "choices without exact-MDL render confirmation: "
                f"{unresolved_disagreement_group_ids}"
            )
        final_plan_document = read_object(
            effective_material_plan,
            "final render-validated material selection",
        )
        try:
            selection_lock_document = build_material_selection_lock(
                plan=final_plan_document,
                catalog_path=effective_catalog,
                material_root=config.material_root,
                allow_reviewed_color_parameters=(
                    config.corresponding_color_calibration_mode == "adaptive_actual_cad"
                ),
            )
        except ValueError as exc:
            raise RuntimeError(
                "Unable to seal the final render-validated NVIDIA MDL selection; "
                f"physics was not started: {exc}"
            ) from exc
        write_object(material_selection_lock, selection_lock_document)
        finalized_apply_plan = (
            final_instance_plan
            if final_instance_plan is not None
            else effective_material_plan
        )
        finalized_apply_command = [
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
            str(finalized_apply_plan),
            "--output",
            str(finalized_look_usd),
            "--material-root",
            str(config.material_root),
            "--report",
            str(finalized_apply_report),
            "--include-review",
            "--selection-lock",
            str(material_selection_lock),
        ]
        if use_policy_fallback:
            finalized_apply_command.append("--include-policy-fallback")
        _run_stage(
            "finalize_immutable_mdl_selection",
            finalized_apply_command,
            log_cb,
            command_runner=_command_runner,
            retry_native_crash=True,
        )
        _require_file(finalized_look_usd, "finalize_immutable_mdl_selection")
        _require_file(finalized_apply_report, "finalize_immutable_mdl_selection")
        finalized_apply_document = read_object(
            finalized_apply_report,
            "final immutable MDL apply report",
        )
        finalized_apply_plan_document = read_object(
            finalized_apply_plan,
            "final immutable MDL apply plan",
        )
        if finalized_apply_document.get("plan_sha256") != canonical_sha256(
            finalized_apply_plan_document
        ):
            raise RuntimeError(
                "Final selected-MDL apply report is not hash-bound to the "
                "exact finalized apply plan"
            )
        finalized_applied_count = finalized_apply_document.get("applied_count")
        if (
            isinstance(finalized_applied_count, bool)
            or not isinstance(finalized_applied_count, int)
            or finalized_applied_count != applied_count
        ):
            raise RuntimeError(
                "Final selected-MDL lock changed exact coverage: "
                f"candidate={applied_count} finalized={finalized_applied_count!r}"
            )
        effective_look_usd = finalized_look_usd
        effective_apply_report = finalized_apply_report
        applied_count = finalized_applied_count
        log_message(
            log_cb,
            "Render-guided MDL selection is complete and immutable. Library "
            "exports, selected MDL identities, audited parameter values, face "
            "subsets, catalog content, and source MDL modules are sealed; no "
            "later material mutation is permitted.",
        )
    selection_lock_verified = False
    if selection_lock_document is not None:
        try:
            validate_material_selection_lock(
                lock=read_object(
                    material_selection_lock,
                    "selected NVIDIA MDL lock",
                ),
                plan=read_object(
                    effective_material_plan,
                    "effective immutable material plan",
                ),
                catalog_path=effective_catalog,
                material_root=config.material_root,
            )
        except ValueError as exc:
            raise RuntimeError(
                "Selected NVIDIA MDL lock changed after selection; physics was "
                f"not started: {exc}"
            ) from exc
        effective_apply_document = read_object(
            effective_apply_report,
            "effective immutable material apply report",
        )
        validation = effective_apply_document.get("validation")
        if (
            not isinstance(validation, dict)
            or validation.get("selected_mdl_lock_verified") is not True
        ):
            raise RuntimeError(
                "Isaac material application did not verify the selected-MDL lock"
            )
        selection_lock_verified = True

    immutable_library_optimum_document: dict[str, Any] | None = None
    if (
        quality_gate_status == "REVIEW"
        and config.immutable_mdl_after_selection
        and config.material_selection_objective == MATERIAL_SELECTION_OBJECTIVE_VISUAL
        and config.exact_mdl_tournament_all_groups
        and exact_mdl_tournament_status in {"SELECTED", "BASELINE_PRESERVED"}
        and exact_mdl_tournament_group_count > 0
        and exact_mdl_tournament_planning.is_file()
        and exact_mdl_tournament_audit.is_file()
        and publish_quality_gate_document is not None
        and publish_quality_gate_document.get("status") == "PASS"
        and selection_lock_verified
    ):
        optimum_quality = read_object(
            effective_quality_report,
            "immutable-library optimum quality report",
        )
        optimum_audit = evaluate_immutable_library_optimum(
            optimum_quality,
            minimum_aggregate_appearance_score=(
                config.final_visual_gate_minimum_final_appearance_score
            ),
            minimum_view_appearance_score=(
                config.final_visual_gate_minimum_final_view_appearance_score
            ),
        )
        optimum_planning = read_object(
            exact_mdl_tournament_planning,
            "immutable-library optimum tournament planning",
        )
        optimum_tournament = read_object(
            exact_mdl_tournament_audit,
            "immutable-library optimum tournament audit",
        )
        cache_resume = optimum_planning.get("cache_resume")
        candidate_total = (
            cache_resume.get("candidate_total")
            if isinstance(cache_resume, Mapping)
            else None
        )
        cache_entries = (
            cache_resume.get("entries") if isinstance(cache_resume, Mapping) else None
        )
        planning_complete = (
            optimum_planning.get("status") == "COMPLETED"
            and optimum_planning.get("group_count") == exact_mdl_tournament_group_count
            and optimum_planning.get("candidate_count")
            == exact_mdl_tournament_candidate_count
            and candidate_total == exact_mdl_tournament_candidate_count
            and isinstance(cache_entries, list)
            and len(cache_entries) == exact_mdl_tournament_candidate_count
        )
        final_quality = optimum_tournament.get("final_whole_asset_quality")
        tournament_quality_bound = (
            isinstance(final_quality, Mapping)
            and final_quality.get("report") == str(effective_quality_report)
            and final_quality.get("report_sha256")
            == sha256_file(effective_quality_report)
            and final_quality.get("status") == "REVIEW"
        )
        optimum_reasons = list(optimum_audit.get("reason_codes", []))
        if not planning_complete:
            optimum_reasons.append("TOURNAMENT_SEARCH_NOT_COMPLETE")
        if not tournament_quality_bound:
            optimum_reasons.append("TOURNAMENT_FINAL_QUALITY_BINDING_INVALID")
        immutable_library_optimum_document = {
            **optimum_audit,
            "status": (
                "PASS"
                if optimum_audit.get("acceptance_allowed") is True
                and planning_complete
                and tournament_quality_bound
                else "FAIL_CLOSED"
            ),
            "acceptance_allowed": bool(
                optimum_audit.get("acceptance_allowed") is True
                and planning_complete
                and tournament_quality_bound
            ),
            "decision": (
                IMMUTABLE_LIBRARY_OPTIMUM_DECISION
                if optimum_audit.get("acceptance_allowed") is True
                and planning_complete
                and tournament_quality_bound
                else None
            ),
            "reason_codes": sorted(set(optimum_reasons)),
            "bindings": {
                "quality_report": str(effective_quality_report),
                "quality_report_sha256": sha256_file(effective_quality_report),
                "tournament_planning": str(exact_mdl_tournament_planning),
                "tournament_planning_sha256": sha256_file(
                    exact_mdl_tournament_planning
                ),
                "tournament_audit": str(exact_mdl_tournament_audit),
                "tournament_audit_sha256": sha256_file(exact_mdl_tournament_audit),
                "publish_quality_gate": str(publish_quality_gate_report),
                "publish_quality_gate_sha256": sha256_file(publish_quality_gate_report),
                "material_selection_lock": str(material_selection_lock),
                "material_selection_lock_sha256": sha256_file(material_selection_lock),
            },
            "constraints": {
                "material_root": str(config.material_root),
                "selected_mdl_parameters_immutable": True,
                "material_selection_lock_verified": True,
                "all_significant_groups_tournament_enabled": True,
                "group_count": exact_mdl_tournament_group_count,
                "candidate_count": exact_mdl_tournament_candidate_count,
                "candidate_search_complete": planning_complete,
                "publish_quality_gate_status": "PASS",
            },
        }
        write_object(
            immutable_library_optimum_gate,
            immutable_library_optimum_document,
        )
        if immutable_library_optimum_document["acceptance_allowed"] is True:
            quality_gate_status = "PASS"
            quality_decision = IMMUTABLE_LIBRARY_OPTIMUM_DECISION
            log_message(
                log_cb,
                "The immutable NVIDIA Base-library search is exhausted and "
                "accepted as a constrained visual optimum: every view is "
                "PASS/REVIEW, absolute color/texture/appearance floors pass, "
                "and the only remaining REVIEW evidence is photometric value "
                "mismatch. Raw visual status remains REVIEW in the audit.",
            )

    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "state": "APPLIED",
        "source_usd": str(source),
        "source_usd_sha256": sha256_file(source),
        "source_cad": (
            str(resolved_source_cad) if resolved_source_cad is not None else None
        ),
        "source_cad_sha256": (
            sha256_file(resolved_source_cad)
            if resolved_source_cad is not None
            else None
        ),
        "effective_usd": str(effective_look_usd.resolve(strict=True)),
        "effective_usd_sha256": sha256_file(effective_look_usd),
        "output_dir": str(destination),
        "references": [
            {"id": reference_id, "image": str(path)}
            for reference_id, path in parsed_references
        ],
        "foreground_annotation_mode": (
            "human_confirmed_sam3_points"
            if resolved_foreground_annotations is not None
            else "automatic_sam3_foreground"
        ),
        "foreground_annotations": (
            str(
                (analysis_dir / "sam3_foreground_annotations.json").resolve(strict=True)
            )
            if resolved_foreground_annotations is not None
            else None
        ),
        "foreground_annotations_sha256": (
            sha256_file(analysis_dir / "sam3_foreground_annotations.json")
            if resolved_foreground_annotations is not None
            else None
        ),
        "foreground_annotation_source": (
            str(
                (analysis_dir / "sam3_foreground_annotation_source.json").resolve(
                    strict=True
                )
            )
            if resolved_foreground_annotations is not None
            else None
        ),
        "foreground_annotation_source_sha256": (
            sha256_file(analysis_dir / "sam3_foreground_annotation_source.json")
            if resolved_foreground_annotations is not None
            else None
        ),
        "foreground_annotations_input": (
            str(resolved_foreground_annotations)
            if resolved_foreground_annotations is not None
            else None
        ),
        "config": str(Path(effective_config_path).expanduser().resolve()),
        "instance_root_count": instance_root_count,
        "instance_aware": bool(instance_root_count),
        "inference_mode": "qwen_mvinverse",
        "complete_coverage_required": bool(
            instance_root_count or require_complete_coverage
        ),
        "source_registry": str(source_registry.resolve(strict=True)),
        "registry": str(registry.resolve(strict=True)),
        "rendered_registry": str(rendered_registry.resolve(strict=True)),
        "unattended_result": str(unattended_result.resolve(strict=True)),
        "staged_state": state,
        "staged_material_plan": str(staged_material_plan.resolve(strict=True)),
        "material_plan": str(effective_material_plan.resolve(strict=True)),
        "material_assignment_unit": config.material_assignment_unit,
        "part_id_reference_evidence": (
            str(part_id_evidence_path.resolve(strict=True))
            if config.material_assignment_unit == "part_id"
            else None
        ),
        "appearance_components": (
            str(appearance_components_report.resolve(strict=True))
            if appearance_components_report.is_file()
            else None
        ),
        "appearance_component_evidence": (
            str(appearance_component_evidence.resolve(strict=True))
            if appearance_component_evidence.is_file()
            else None
        ),
        "appearance_component_visual_retrieval": (
            str(appearance_component_retrieval_result.resolve(strict=True))
            if appearance_component_retrieval_result.is_file()
            else None
        ),
        "appearance_component_qwen_choices": (
            str(appearance_component_qwen_result.resolve(strict=True))
            if appearance_component_qwen_result.is_file()
            else None
        ),
        "appearance_component_mdl_selection": (
            str(appearance_component_mdl_selection_audit.resolve(strict=True))
            if appearance_component_mdl_selection_audit.is_file()
            else None
        ),
        "part_id_visual_retrieval": (
            str(part_id_retrieval_result.resolve(strict=True))
            if config.material_assignment_unit == "part_id"
            else None
        ),
        "part_id_qwen_choices": (
            str(part_id_qwen_result.resolve(strict=True))
            if config.material_assignment_unit == "part_id"
            else None
        ),
        "part_id_material_audit": (
            str(part_id_material_audit.resolve(strict=True))
            if config.material_assignment_unit == "part_id"
            else None
        ),
        "part_id_quality_gate": (
            str(part_id_quality_gate.resolve(strict=True))
            if part_id_quality_gate_document is not None
            else None
        ),
        "palette_groups_used_for_material_assignment": (
            config.material_assignment_unit != "part_id"
        ),
        "immutable_mdl_after_selection": config.immutable_mdl_after_selection,
        "material_parameter_candidate_mode": (config.material_parameter_candidate_mode),
        "material_prediction_mode": config.material_prediction_mode,
        "material_identity_local_context": config.material_identity_local_context,
        "corresponding_color_calibration_mode": (
            config.corresponding_color_calibration_mode
        ),
        "corresponding_color_max_iterations": (
            config.corresponding_color_max_iterations
        ),
        "part_id_parameter_tournament": (
            str(part_id_parameter_tournament_audit.resolve(strict=True))
            if part_id_parameter_tournament_audit.is_file()
            else None
        ),
        "part_id_parameter_tournament_candidate_count": (
            parameter_tournament_document["candidate_part_count"]
            if parameter_tournament_document is not None
            else 0
        ),
        "part_id_parameter_tournament_h1_winner_count": (
            parameter_tournament_document["h1_winner_count"]
            if parameter_tournament_document is not None
            else 0
        ),
        "appearance_component_actual_mdl_tournament": (
            str(appearance_component_actual_mdl_tournament_audit.resolve(strict=True))
            if appearance_component_actual_mdl_tournament_audit.is_file()
            else None
        ),
        "appearance_component_actual_mdl_tournament_candidate_count": (
            component_actual_mdl_tournament_document["candidate_count"]
            if component_actual_mdl_tournament_document is not None
            else 0
        ),
        "appearance_component_actual_mdl_tournament_winner_count": (
            component_actual_mdl_tournament_document["winner_count"]
            if component_actual_mdl_tournament_document is not None
            else 0
        ),
        "corresponding_material_color_calibration": (
            str(workspace.corresponding_color.manifest.resolve(strict=True))
            if corresponding_color_calibration_document is not None
            else None
        ),
        "corresponding_material_color_calibration_status": (
            corresponding_color_calibration_document.get("workflow_state")
            if corresponding_color_calibration_document is not None
            else "NOT_REQUIRED"
        ),
        "corresponding_material_color_calibration_candidate_count": (
            len(corresponding_color_calibration_document.get("candidates", []))
            if corresponding_color_calibration_document is not None
            else 0
        ),
        "material_selection_objective": config.material_selection_objective,
        "dominant_assembly_membership_tournament_status": (
            membership_tournament_status
        ),
        "dominant_assembly_membership_cohort_count": (
            membership_tournament_cohort_count
        ),
        "dominant_assembly_membership_selected_expanded_count": (
            membership_selected_expanded_count
        ),
        "dominant_assembly_membership_restored_m0_count": (
            membership_restored_m0_count
        ),
        "dominant_assembly_membership_plan": (
            str(membership_tournament_plan.resolve(strict=True))
            if membership_tournament_plan.is_file()
            else None
        ),
        "dominant_assembly_membership_tournament_audit": (
            str(membership_tournament_audit.resolve(strict=True))
            if membership_tournament_audit.is_file()
            else None
        ),
        "exact_mdl_tournament_status": exact_mdl_tournament_status,
        "exact_mdl_tournament_candidate_count": (exact_mdl_tournament_candidate_count),
        "exact_mdl_tournament_selected_candidate_id": (
            exact_mdl_tournament_selected_candidate_id
        ),
        "exact_mdl_tournament_selected_candidate_ids": (
            exact_mdl_tournament_selected_candidate_ids
        ),
        "exact_mdl_tournament_group_count": (exact_mdl_tournament_group_count),
        "baseline_preserved_disagreement_exemption_count": len(
            baseline_preserved_disagreement_exemptions
        ),
        "baseline_preserved_disagreement_exempt_group_ids": [
            str(record["group_id"])
            for record in baseline_preserved_disagreement_exemptions
        ],
        "exact_mdl_tournament_all_groups": (config.exact_mdl_tournament_all_groups),
        "exact_mdl_tournament_minimum_score_improvement": (
            config.exact_mdl_tournament_minimum_score_improvement
        ),
        "exact_mdl_tournament_minimum_winner_margin": (
            config.exact_mdl_tournament_minimum_winner_margin
        ),
        "exact_mdl_tournament_planning": (
            str(exact_mdl_tournament_planning.resolve(strict=True))
            if exact_mdl_tournament_planning.is_file()
            else None
        ),
        "exact_mdl_tournament_audit": (
            str(exact_mdl_tournament_audit.resolve(strict=True))
            if exact_mdl_tournament_audit.is_file()
            else None
        ),
        "visual_group_annotation_status": visual_group_annotation_status,
        "visual_group_annotated_plan": (
            str(visual_group_annotated_plan.resolve(strict=True))
            if visual_group_annotated_plan.is_file()
            else None
        ),
        "visual_group_annotation_audit": (
            str(visual_group_annotation_audit.resolve(strict=True))
            if visual_group_annotation_audit.is_file()
            else None
        ),
        "publish_quality_gate": (
            str(publish_quality_gate_report.resolve(strict=True))
            if publish_quality_gate_document is not None
            else None
        ),
        "publish_quality_gate_status": (
            publish_quality_gate_document["status"]
            if publish_quality_gate_document is not None
            else "NOT_REQUIRED"
        ),
        "publish_quality_gate_reason_codes": (
            list(publish_quality_gate_document["reason_codes"])
            if publish_quality_gate_document is not None
            else []
        ),
        "material_selection_lock": (
            str(material_selection_lock.resolve(strict=True))
            if selection_lock_document is not None
            else None
        ),
        "material_selection_lock_verified": selection_lock_verified,
        "policy_exact_cover_used": use_policy_fallback,
        "policy_exact_cover_plan": (
            str(policy_plan.resolve(strict=True)) if use_policy_fallback else None
        ),
        "policy_exact_cover_input": (
            str(policy_input.resolve(strict=True)) if use_policy_fallback else None
        ),
        "policy_exact_cover_audit": (
            str(policy_audit.resolve(strict=True)) if use_policy_fallback else None
        ),
        "policy_fallback_count": policy_fallback_count,
        "policy_fallback_explicitly_authorized": use_policy_fallback,
        "source_visual_strategy": (
            policy_audit_document.get("source_visual_strategy")
            if use_policy_fallback
            else None
        ),
        "instance_material_plan": (
            str(final_instance_plan.resolve(strict=True))
            if final_instance_plan is not None
            else None
        ),
        "editable_usd": (
            str(editable_usd.resolve(strict=True)) if editable_usd is not None else None
        ),
        "expand_report": (
            str(expand_report.resolve(strict=True))
            if expand_report is not None
            else None
        ),
        "mvinverse_ledger": str(mvinverse_ledger.resolve(strict=True)),
        "qwen_mvinverse_recovery": str(inference_recovery.resolve(strict=True)),
        "apply_report": str(effective_apply_report.resolve(strict=True)),
        "quality_repair_used": quality_repair_used,
        "quality_repair_round_count": quality_repair_round_count,
        "visual_quality_round_count": quality_round_count,
        "quality_repair_changed_count": quality_repair_changed_count,
        "quality_repair_plan": (
            str(quality_repair_plan.resolve(strict=True))
            if quality_repair_used
            else None
        ),
        "quality_repair_audit": (
            str(quality_repair_audit.resolve(strict=True))
            if quality_repair_used
            else None
        ),
        "baseline_visual_quality_report": str(quality_report.resolve(strict=True)),
        "visual_quality_decision": quality_decision,
        "visual_quality_raw_status": raw_quality_status,
        "visual_quality_status": quality_status,
        "visual_quality_gate_status": quality_gate_status,
        "immutable_library_optimum_accepted": (
            immutable_library_optimum_document is not None
            and immutable_library_optimum_document.get("acceptance_allowed") is True
        ),
        "immutable_library_optimum_gate": (
            str(immutable_library_optimum_gate.resolve(strict=True))
            if immutable_library_optimum_document is not None
            else None
        ),
        "visual_quality_resolution": (
            str(quality_resolution.resolve(strict=True))
            if quality_resolution_document is not None
            else None
        ),
        "visual_quality_limitation_count": quality_limitation_count,
        "visual_quality_report": str(effective_quality_report.resolve(strict=True)),
        "visual_quality_rendered_registry": str(
            effective_quality_rendered_registry.resolve(strict=True)
        ),
        "appearance_optimization_status": appearance_status,
        "appearance_optimization_reason_codes": sorted(set(appearance_reason_codes)),
        "appearance_optimization_baseline_lighting_profile": (
            baseline_lighting_profile
        ),
        "appearance_optimization_measurement_used": (appearance_measurement_used),
        "appearance_optimization_adjustment_count": (appearance_adjustment_count),
        "appearance_optimization_changed_part_count": (appearance_changed_count),
        "appearance_optimization_candidate_round_count": (
            appearance_candidate_round_count
        ),
        "appearance_optimization_error": appearance_error,
        "appearance_optimization_baseline_quality_report": (
            str(appearance_baseline_quality.resolve(strict=True))
            if appearance_baseline_quality.is_file()
            else None
        ),
        "appearance_optimization_baseline_measurement": (
            str(appearance_baseline_measurement.resolve(strict=True))
            if appearance_baseline_measurement.is_file()
            else None
        ),
        "appearance_optimization_contract": (
            str(appearance_contract.resolve(strict=True))
            if appearance_contract.is_file()
            else None
        ),
        "appearance_optimization_candidate_plan": (
            str(appearance_candidate_plan.resolve(strict=True))
            if appearance_candidate_plan.is_file()
            else None
        ),
        "appearance_optimization_candidate_plan_apply_report": (
            str(appearance_candidate_plan_apply.resolve(strict=True))
            if appearance_candidate_plan_apply.is_file()
            else None
        ),
        "appearance_optimization_candidate_usd_apply_report": (
            str(appearance_candidate_usd_apply.resolve(strict=True))
            if appearance_candidate_usd_apply.is_file()
            else None
        ),
        "appearance_optimization_candidate_raw_quality_report": (
            str(appearance_candidate_raw_quality.resolve(strict=True))
            if appearance_candidate_raw_quality.is_file()
            else None
        ),
        "appearance_optimization_candidate_quality_report": (
            str(appearance_candidate_quality.resolve(strict=True))
            if appearance_candidate_quality.is_file()
            else None
        ),
        "appearance_optimization_candidate_measurement": (
            str(appearance_candidate_measurement.resolve(strict=True))
            if appearance_candidate_measurement.is_file()
            else None
        ),
        "appearance_optimization_validation": (
            str(appearance_validation.resolve(strict=True))
            if appearance_validation.is_file()
            else None
        ),
        "assignment_count": len(assignments),
        "applied_count": applied_count,
    }


@_visual_control_cpu_stability_guard
def run_assign_visual_materials_job(
    *,
    source_usd: str,
    source_cad: str | None = None,
    references: Sequence[str],
    foreground_annotations: str | None = None,
    output_dir: str | None = None,
    config_path: str | None = None,
    inference_mode: str = "live",
    acknowledge_mvinverse_noncommercial: bool = False,
    require_complete_coverage: bool = False,
    allow_policy_material_fallback: bool = False,
    log_cb: LogCallback = None,
    _command_runner: CommandRunner = run_command,
    _isaac_python_resolver: IsaacPythonResolver = isaac_python,
    _config_loader: ConfigLoader = load_visual_material_config,
    _reference_parser: ReferenceParser = parse_visual_references,
    _default_config_path: Path = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    """Create and validate one non-destructive visual-material Look USD.

    Underscore-prefixed dependency parameters keep the compatibility module's
    established monkeypatch points working without coupling this owner module
    back to ``asset_pipeline.jobs.material``.
    """

    if not acknowledge_mvinverse_noncommercial:
        raise ValueError(
            "STEP/STP reference-image material assignment requires explicit "
            "--acknowledge-mvinverse-noncommercial"
        )
    context = VisualMaterialPipelineContext.create(
        source_usd=source_usd,
        source_cad=source_cad,
        references=references,
        foreground_annotations=foreground_annotations,
        output_dir=output_dir,
        config_path=config_path,
        inference_mode=inference_mode,
        default_config_path=_default_config_path,
        config_loader=_config_loader,
        isaac_python_resolver=_isaac_python_resolver,
        reference_parser=_reference_parser,
        resume_validator=_verified_partial_live_resume_available,
    )
    source = context.source
    resolved_source_cad = context.source_cad
    parsed_references = context.references
    resolved_foreground_annotations = context.foreground_annotations
    effective_config_path = context.config_path
    config = context.config
    isaac = context.isaac_python
    destination = context.destination
    partial_live_resume = context.partial_live_resume

    if partial_live_resume:
        log_message(
            log_cb,
            "Resuming unfinished live visual-material run from hash-verified "
            f"evidence/inference checkpoints: {destination}",
        )
    log_message(log_cb, f"Visual material input USD: {source}")
    log_message(log_cb, f"Visual material output: {destination}")
    log_message(log_cb, f"Visual material config: {effective_config_path}")
    log_message(log_cb, f"Visual material inference mode: {inference_mode}")
    log_message(
        log_cb,
        "Visual material references: "
        + ", ".join(
            f"{reference_id}={path}" for reference_id, path in parsed_references
        ),
    )
    if resolved_foreground_annotations is not None:
        log_message(
            log_cb,
            "Human-confirmed SAM3 foreground annotations: "
            f"{resolved_foreground_annotations}",
        )

    prepared_source = prepare_source_evidence(
        context,
        log_cb=log_cb,
        command_runner=_command_runner,
    )
    source_registry = prepared_source.source_registry
    registry = prepared_source.registry
    rendered_registry = prepared_source.rendered_registry
    editable_usd = prepared_source.editable_usd
    expand_report = prepared_source.expand_report
    instance_root_count = prepared_source.instance_root_count
    bundled_project = prepared_source.bundled_project
    if bundled_project is not None:
        if resolved_source_cad is None:
            raise AssertionError("Bundled project matched without a source CAD")
        return _run_bundled_project_assignment(
            project=bundled_project,
            source=source,
            source_cad=resolved_source_cad,
            parsed_references=parsed_references,
            destination=destination,
            source_registry=source_registry,
            registry=registry,
            rendered_registry=rendered_registry,
            editable_usd=editable_usd,
            expand_report=expand_report,
            instance_root_count=instance_root_count,
            requested_inference_mode=inference_mode,
            effective_config_path=effective_config_path,
            config=config,
            isaac=isaac,
            log_cb=log_cb,
            command_runner=_command_runner,
        )

    planning = _run_policy_part_id_stage(
        context,
        prepared_source=prepared_source,
        allow_policy_material_fallback=allow_policy_material_fallback,
        log_cb=log_cb,
        command_runner=_command_runner,
    )
    look = _run_look_application_stage(
        context,
        prepared_source=prepared_source,
        planning=planning,
        require_complete_coverage=require_complete_coverage,
        log_cb=log_cb,
        command_runner=_command_runner,
    )
    # Close the unattended loop with a fresh render of the authored Look.  The
    # spatial stage already established reference-to-canonical-view
    # registration; reuse only its trusted one-to-one associations so the
    # quality comparison cannot silently compare the wrong sides.
    visual_qa = _run_visual_qa_stage(
        context,
        prepared_source=prepared_source,
        planning=planning,
        look=look,
        require_complete_coverage=require_complete_coverage,
        log_cb=log_cb,
        command_runner=_command_runner,
    )
    selection = _run_material_selection_stage(
        context,
        prepared_source=prepared_source,
        planning=planning,
        look=look,
        visual_qa=visual_qa,
        log_cb=log_cb,
        command_runner=_command_runner,
    )
    # A whole-image render PASS is not publication evidence when almost all
    # parts came from a generic exact-cover fallback or when canonical-group
    # ownership remains unresolved.  Strict publication is required for an
    # immutable lock and for either exact-cover mode.  The legacy partial-look
    # mode remains compatible, while the sealed bundled-project path returns
    # above and is intentionally unaffected.
    return _run_finalize_assignment_stage(
        context,
        prepared_source=prepared_source,
        planning=planning,
        look=look,
        visual_qa=visual_qa,
        selection=selection,
        require_complete_coverage=require_complete_coverage,
        log_cb=log_cb,
        command_runner=_command_runner,
    )


__all__ = [
    "APPLICABLE_ASSIGNMENT_STATUSES",
    "ISOLATED_ENV_REMOVE",
    "RESULT_SCHEMA_VERSION",
    "USD_SUFFIXES",
    "run_assign_visual_materials_job",
    "run_final_visual_acceptance_job",
]
