"""Build photo evidence for each CAD Part-ID.

This stage owns the current production segmentation route:

1. render the target Part-ID as an isolated CAD-model template;
2. run first-pass SAM3 and EntitySeg candidates;
3. infer the target location from non-target assembly neighbours;
4. rerun both segmenters in the relation-guided region; and
5. iteratively fuse model shape, photo structure, and safe neural candidates.

The stage only transforms image-space masks.  It never edits a CAD/USD
transform or moves an individual mesh.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ...command import LogCallback, log_message
from ...paths import unique_path
from ...project_layout import ProjectLayout
from ...runtime import root_dir
from ..commands import cad_mesh_template_command
from ..config import VisualMaterialConfig, write_object
from ..context import VisualMaterialPipelineContext
from .runner import CommandRunner, _run_stage
from qwen_material_pipeline.evidence.part_id_projection import (
    build_part_id_reference_evidence,
)
from qwen_material_pipeline.segmentation.part_id_request import (
    build_request as build_part_id_sam3_request,
)
from qwen_material_pipeline.segmentation.part_relation_guidance import (
    build_relation_guided_request,
)


def _archive_stage_directory(directory: Path, *, analysis_dir: Path) -> None:
    """Move a stale, regenerable stage directory into the recovery archive."""

    if not (directory.exists() or directory.is_symlink()):
        return
    archived = unique_path(
        analysis_dir / "recovery_archive" / f"stale_{directory.name}"
    )
    archived.parent.mkdir(parents=True, exist_ok=True)
    directory.rename(archived)


def _sam3_region_command(
    config: VisualMaterialConfig,
    *,
    request: Path,
    output_dir: Path,
) -> list[str]:
    script = (
        ProjectLayout.from_root(root_dir()).material_pipeline
        / "segmentation"
        / "sam3_regions.py"
    )
    return [
        str(config.sam3_python),
        str(script),
        "--request",
        str(request),
        "--repository",
        str(config.sam3_repository),
        "--checkpoint",
        str(config.sam3_checkpoint),
        "--output-dir",
        str(output_dir),
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
        "--seed",
        "0",
    ]


def _entityseg_region_command(
    config: VisualMaterialConfig,
    *,
    request: Path,
    output_dir: Path,
) -> list[str]:
    _require_entityseg_runtime(config)
    return [
        str(config.entityseg_python),
        "-m",
        "qwen_material_pipeline.segmentation.entityseg_regions",
        "--request",
        str(request),
        "--cropformer-root",
        str(config.entityseg_cropformer_root),
        "--config",
        str(config.entityseg_config),
        "--checkpoint",
        str(config.entityseg_checkpoint),
        "--output-dir",
        str(output_dir),
        "--minimum-model-score",
        str(config.entityseg_minimum_model_score),
        "--seed",
        "0",
    ]


def _require_entityseg_runtime(config: VisualMaterialConfig) -> None:
    """Fail before replacing reusable evidence when EntitySeg is incomplete."""

    if any(
        value is None
        for value in (
            config.entityseg_python,
            config.entityseg_cropformer_root,
            config.entityseg_config,
            config.entityseg_checkpoint,
        )
    ):
        raise RuntimeError("EntitySeg fusion is enabled but its runtime is incomplete")


def _hybrid_mask_command(
    config: VisualMaterialConfig,
    *,
    sam_manifest: Path,
    entityseg_manifest: Path,
    amodal_manifest: Path,
    output_dir: Path,
    prior_hybrid_manifest: Path | None = None,
) -> list[str]:
    command = [
        str(config.sam3_python),
        "-m",
        "qwen_material_pipeline.segmentation.hybrid_part_masks",
        "--sam-manifest",
        str(sam_manifest),
        "--entity-manifest",
        str(entityseg_manifest),
        "--amodal-manifest",
        str(amodal_manifest),
    ]
    if prior_hybrid_manifest is not None:
        command.extend(["--prior-hybrid-manifest", str(prior_hybrid_manifest)])
    command.extend(["--output-dir", str(output_dir)])
    return command


def _require_complete_reference_views(
    *,
    evidence: Mapping[str, Any],
    expected_view_ids: set[str],
    label: str,
) -> None:
    """Require every registered photo to contribute selected Part-ID evidence."""

    expected = {
        view_id.strip()
        for view_id in expected_view_ids
        if isinstance(view_id, str) and view_id.strip()
    }
    if not expected or len(expected) != len(expected_view_ids):
        raise RuntimeError(f"{label} has an invalid expected reference-view set")
    summary = evidence.get("summary")
    if not isinstance(summary, Mapping):
        raise RuntimeError(f"{label} has no summary")
    coverage = summary.get("selected_reference_view_coverage")
    if not isinstance(coverage, Mapping) or any(
        not isinstance(view_id, str) or not isinstance(row, Mapping)
        for view_id, row in coverage.items()
    ):
        raise RuntimeError(f"{label} has invalid selected reference-view coverage")
    if set(coverage) != expected:
        raise RuntimeError(
            f"{label} does not cover every registered reference view: "
            f"expected={sorted(expected)}, actual={sorted(coverage)}"
        )
    trusted_count = summary.get("trusted_reference_view_count")
    if (
        isinstance(trusted_count, bool)
        or not isinstance(trusted_count, int)
        or trusted_count != len(expected)
    ):
        raise RuntimeError(
            f"{label} trusted reference-view count does not match the "
            "registered reference set"
        )

    parts = evidence.get("parts")
    if not isinstance(parts, list):
        raise RuntimeError(f"{label} has no Part-ID observations")
    visible_parts_by_view = {view_id: 0 for view_id in expected}
    selected_parts_by_view = {view_id: 0 for view_id in expected}
    observed_view_ids: set[str] = set()
    selected_view_ids: set[str] = set()
    for part_index, part in enumerate(parts):
        if not isinstance(part, Mapping):
            raise RuntimeError(f"{label} Part-ID row {part_index} is invalid")
        observations = part.get("observations")
        if not isinstance(observations, list):
            raise RuntimeError(
                f"{label} Part-ID row {part_index} has invalid observations"
            )
        part_visible_views: set[str] = set()
        part_selected_views: set[str] = set()
        for observation_index, observation in enumerate(observations):
            if not isinstance(observation, Mapping):
                raise RuntimeError(
                    f"{label} observation {part_index}:{observation_index} is invalid"
                )
            view_id = observation.get("view_id")
            if not isinstance(view_id, str) or not view_id:
                raise RuntimeError(
                    f"{label} observation {part_index}:{observation_index} has "
                    "no view_id"
                )
            observed_view_ids.add(view_id)
            part_visible_views.add(view_id)
            if observation.get("selected_for_material_inference") is True:
                selected_view_ids.add(view_id)
                part_selected_views.add(view_id)
        for view_id in part_visible_views & expected:
            visible_parts_by_view[view_id] += 1
        for view_id in part_selected_views & expected:
            selected_parts_by_view[view_id] += 1

    if observed_view_ids != expected or selected_view_ids != expected:
        raise RuntimeError(
            f"{label} underlying observations do not use every registered "
            "reference view"
        )
    for view_id in sorted(expected):
        row = coverage[view_id]
        visible = row.get("visible_part_count")
        selected = row.get("selected_part_count")
        if (
            isinstance(visible, bool)
            or not isinstance(visible, int)
            or visible < 1
            or visible != visible_parts_by_view[view_id]
            or isinstance(selected, bool)
            or not isinstance(selected, int)
            or selected < 1
            or selected != selected_parts_by_view[view_id]
        ):
            raise RuntimeError(
                f"{label} coverage for {view_id!r} does not match its "
                "underlying selected observations"
            )


def run_part_id_evidence_stage(
    context: VisualMaterialPipelineContext,
    *,
    rendered_registry: Path,
    mvinverse_ledger: Path,
    log_cb: LogCallback,
    command_runner: CommandRunner,
) -> dict[str, Any]:
    """Run the production Part-ID localization and mask-refinement route."""

    config = context.config
    workspace = context.workspace
    paths = workspace.part_id
    analysis_dir = workspace.inference.root
    camera_acceptance = workspace.source.camera_acceptance

    for stale_dir in (
        paths.coarse_evidence_dir,
        paths.evidence_dir,
        paths.amodal_template_dir,
    ):
        _archive_stage_directory(stale_dir, analysis_dir=analysis_dir)

    evidence_kwargs = {
        "reference_manifest": analysis_dir / "reference_manifest.json",
        "rendered_registry": rendered_registry,
        "spatial_mapping_report": analysis_dir / "spatial_mapping_report.json",
        "camera_alignment_acceptance": (
            camera_acceptance if camera_acceptance.is_file() else None
        ),
        "mvinverse_ledger": mvinverse_ledger,
    }
    coarse_evidence = build_part_id_reference_evidence(
        **evidence_kwargs,
        output_dir=paths.coarse_evidence_dir,
    )
    write_object(paths.coarse_evidence, coarse_evidence)
    expected_views = {view_id for view_id, _path in context.references}
    if config.material_prediction_mode == "catalog_family_first":
        _require_complete_reference_views(
            evidence=coarse_evidence,
            expected_view_ids=expected_views,
            label="coarse Part-ID evidence",
        )

    _run_stage(
        "part_id_cad_amodal_templates",
        cad_mesh_template_command(
            isaac_python=context.isaac_python,
            registry=rendered_registry,
            spatial_report=analysis_dir / "spatial_mapping_report.json",
            evidence=paths.coarse_evidence,
            output_dir=paths.amodal_template_dir,
        ),
        log_cb,
        command_runner=command_runner,
        required_files=(paths.amodal_template_manifest,),
    )
    write_object(
        paths.sam3_request,
        build_part_id_sam3_request(
            paths.coarse_evidence,
            amodal_templates_path=paths.amodal_template_manifest,
        ),
    )
    _archive_stage_directory(paths.sam3_dir, analysis_dir=analysis_dir)
    _run_stage(
        "part_id_sam3_local_refinement",
        _sam3_region_command(
            config,
            request=paths.sam3_request,
            output_dir=paths.sam3_dir,
        ),
        log_cb,
        command_runner=command_runner,
        required_files=(paths.sam3_manifest,),
    )

    effective_manifest = paths.sam3_manifest
    if config.entityseg_enabled:
        _require_entityseg_runtime(config)
        for stale_dir in (
            paths.entityseg_dir,
            paths.initial_hybrid_mask_dir,
            paths.relation_guidance_dir,
            paths.relation_sam3_dir,
            paths.relation_entityseg_dir,
            paths.hybrid_mask_dir,
        ):
            _archive_stage_directory(stale_dir, analysis_dir=analysis_dir)
        _run_stage(
            "part_id_entityseg_boundary_candidates",
            _entityseg_region_command(
                config,
                request=paths.sam3_request,
                output_dir=paths.entityseg_dir,
            ),
            log_cb,
            command_runner=command_runner,
            required_files=(paths.entityseg_manifest,),
        )
        _run_stage(
            "part_id_initial_sam3_entityseg_fusion",
            _hybrid_mask_command(
                config,
                sam_manifest=paths.sam3_manifest,
                entityseg_manifest=paths.entityseg_manifest,
                amodal_manifest=paths.amodal_template_manifest,
                output_dir=paths.initial_hybrid_mask_dir,
            ),
            log_cb,
            command_runner=command_runner,
            required_files=(paths.initial_hybrid_mask_manifest,),
        )
        write_object(
            paths.relation_guided_request,
            build_relation_guided_request(
                initial_request_path=paths.sam3_request,
                sam_manifest_path=paths.sam3_manifest,
                entity_manifest_path=paths.entityseg_manifest,
                amodal_manifest_path=paths.amodal_template_manifest,
                output_dir=paths.relation_guidance_dir,
            ),
        )
        _run_stage(
            "part_id_relation_guided_sam3_refinement",
            _sam3_region_command(
                config,
                request=paths.relation_guided_request,
                output_dir=paths.relation_sam3_dir,
            ),
            log_cb,
            command_runner=command_runner,
            required_files=(paths.relation_sam3_manifest,),
        )
        _run_stage(
            "part_id_relation_guided_entityseg_boundaries",
            _entityseg_region_command(
                config,
                request=paths.relation_guided_request,
                output_dir=paths.relation_entityseg_dir,
            ),
            log_cb,
            command_runner=command_runner,
            required_files=(paths.relation_entityseg_manifest,),
        )
        _run_stage(
            "part_id_relation_guided_iterative_fusion",
            _hybrid_mask_command(
                config,
                sam_manifest=paths.relation_sam3_manifest,
                entityseg_manifest=paths.relation_entityseg_manifest,
                amodal_manifest=paths.amodal_template_manifest,
                prior_hybrid_manifest=paths.initial_hybrid_mask_manifest,
                output_dir=paths.hybrid_mask_dir,
            ),
            log_cb,
            command_runner=command_runner,
            required_files=(paths.hybrid_mask_manifest,),
        )
        effective_manifest = paths.hybrid_mask_manifest

    evidence = build_part_id_reference_evidence(
        **evidence_kwargs,
        part_id_sam3_manifest=(
            None if config.entityseg_enabled else effective_manifest
        ),
        part_id_hybrid_manifest=(
            effective_manifest if config.entityseg_enabled else None
        ),
        output_dir=paths.evidence_dir,
    )
    write_object(paths.evidence, evidence)
    if config.material_prediction_mode == "catalog_family_first":
        _require_complete_reference_views(
            evidence=evidence,
            expected_view_ids=expected_views,
            label="refined Part-ID evidence",
        )

    summary = evidence["summary"]
    log_message(
        log_cb,
        "Two-layer one-to-one Part-ID mapping completed: every coarse box "
        "inherits the rigid whole-asset camera and the single residual shared "
        "by that view, then the isolated mesh, current-view CAD visibility, "
        "SAM3/EntitySeg priors, and photo edges iteratively refine its boundary. "
        "No CAD/USD transform was changed. "
        f"{summary.get('sam3_refined_observation_count', 0)} observations passed "
        "local refinement ("
        f"EntitySeg={summary.get('entityseg_refined_observation_count', 0)}, "
        f"SAM3={summary.get('sam3_selected_observation_count', 0)}); "
        f"iterative={summary.get('shape_guided_iterative_refined_observation_count', 0)}; "
        f"{summary.get('global_projection_fallback_observation_count', 0)} used "
        "the audited coarse fallback. "
        f"{summary.get('chromatic_isolated_observation_count', 0)} single-view "
        "chromatic components were isolated, including "
        f"{summary.get('tiny_chromatic_rescue_observation_count', 0)} small-part "
        "rescues. No Part-ID-local translation, rotation, scale, or CAD/USD "
        "geometry change is allowed. Selected-view coverage: "
        f"{summary.get('selected_reference_view_coverage', {})}.",
    )
    return evidence


__all__ = ["run_part_id_evidence_stage"]
