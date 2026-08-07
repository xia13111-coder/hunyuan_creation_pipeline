"""USD registry, instance expansion, rendering and camera-registration stage."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ...command import LogCallback, log_message
from ..bundled_projects import BundledMaterialProject, match_bundled_project
from ..camera import (
    require_complete_live_camera_alignment as _require_complete_live_camera_alignment,
    validate_live_camera_registration_provenance as _validate_live_camera_registration_provenance,
)
from ..commands import (
    camera_registration_command,
    usd_expand_instances_command,
    usd_registry_command,
    usd_render_command,
)
from ..config import read_object, write_object
from ..context import VisualMaterialPipelineContext
from ..references import sha256_file
from .common import require_file as _require_file
from .runner import _run_stage
from qwen_material_pipeline.evidence.appearance_components import (
    AppearanceComponentError,
    build_appearance_components,
)


CommandRunner = Callable[..., None]


@dataclass(frozen=True)
class SourcePreparationResult:
    source_registry: Path
    registry: Path
    rendered_registry: Path
    editable_usd: Path | None
    expand_report: Path | None
    instance_root_count: int
    bundled_project: BundledMaterialProject | None


def prepare_source_evidence(
    context: VisualMaterialPipelineContext,
    *,
    log_cb: LogCallback,
    command_runner: CommandRunner,
) -> SourcePreparationResult:
    """Prepare the exact occurrence representation consumed by inference."""

    source = context.source
    resolved_source_cad = context.source_cad
    parsed_references = context.references
    resolved_foreground_annotations = context.foreground_annotations
    config = context.config
    isaac = context.isaac_python
    inference_mode = context.inference_mode
    partial_live_resume = context.partial_live_resume
    source_paths = context.workspace.source
    source_registry = source_paths.source_registry
    registry = source_registry
    editable_usd: Path | None = None
    expand_report: Path | None = None
    render_dir = source_paths.render_dir
    rendered_registry = source_paths.rendered_registry
    camera_calibration_dir = source_paths.camera_dir
    camera_calibration_search_dir = source_paths.camera_search_dir
    camera_calibration_search_specs = source_paths.camera_search_specs
    camera_calibration_search_report = source_paths.camera_search_report
    camera_calibrated_registry = source_paths.camera_calibrated_registry
    camera_calibration_report = source_paths.camera_report
    face_region_manifest = context.workspace.inference.face_region_manifest
    resume_has_material_checkpoint = (
        partial_live_resume and face_region_manifest.is_file()
    )
    appearance_components_report = context.workspace.appearance.components

    if not partial_live_resume:
        _run_stage(
            "source_registry",
            usd_registry_command(
                isaac_python=isaac,
                usd=source,
                output=source_registry,
            ),
            log_cb,
            command_runner=command_runner,
            retry_native_crash=True,
        )
    _require_file(
        source_registry,
        "partial_resume_source_registry" if partial_live_resume else "source_registry",
    )
    source_registry_document = read_object(source_registry, "source registry")
    if partial_live_resume and source_registry_document.get(
        "asset_sha256"
    ) != sha256_file(source):
        raise RuntimeError(
            "Partial visual-material resume rejected: converted source USD changed"
        )
    instance_root_count = source_registry_document.get("instance_root_count", 0)
    if (
        isinstance(instance_root_count, bool)
        or not isinstance(instance_root_count, int)
        or instance_root_count < 0
    ):
        raise RuntimeError(
            f"Source registry has invalid instance_root_count: {instance_root_count!r}"
        )

    if instance_root_count:
        editable_usd = source_paths.editable_usd
        expand_report = source_paths.expand_report
        if not partial_live_resume:
            _run_stage(
                "expand_instances",
                usd_expand_instances_command(
                    isaac_python=isaac,
                    source_usd=source,
                    output_usd=editable_usd,
                    report=expand_report,
                ),
                log_cb,
                command_runner=command_runner,
                retry_native_crash=True,
            )
        _require_file(editable_usd, "expand_instances")
        _require_file(expand_report, "expand_instances")
        registry = source_paths.expanded_registry
        if not partial_live_resume:
            _run_stage(
                "editable_registry",
                usd_registry_command(
                    isaac_python=isaac,
                    usd=editable_usd,
                    output=registry,
                ),
                log_cb,
                command_runner=command_runner,
                retry_native_crash=True,
            )
        _require_file(registry, "editable_registry")

    occurrence_registry_document = read_object(registry, "occurrence registry")
    bundled_project = None
    if inference_mode != "live":
        bundled_project = match_bundled_project(
            source_cad=resolved_source_cad,
            references=parsed_references,
            source_registry=source_registry_document,
            occurrence_registry=occurrence_registry_document,
            configured_material_root=config.material_root,
            isaac_root=isaac.parent,
        )
        if inference_mode == "bundled" and bundled_project is None:
            raise RuntimeError(
                "visual inference mode 'bundled' requires an exact sealed-project "
                "match for the CAD, references, occurrence paths and topology"
            )
    render_resolution = config.render_resolution
    render_views = config.render_views
    render_rt_subframes = config.render_rt_subframes
    analysis_up_axis = config.analysis_up_axis
    analysis_front_axis = config.analysis_front_axis
    if bundled_project is not None:
        render_resolution = int(bundled_project.render["resolution"])
        render_views = str(bundled_project.render["views"])
        render_rt_subframes = int(bundled_project.render["rt_subframes"])
        analysis_up_axis = str(bundled_project.render["analysis_up_axis"])
        analysis_front_axis = str(bundled_project.render["analysis_front_axis"])
        log_message(
            log_cb,
            "Exact STP/reference/topology match selected restored material project "
            f"{bundled_project.asset_id!r}.",
        )

    source_render_command = usd_render_command(
        isaac_python=isaac,
        registry=registry,
        output_dir=render_dir,
        resolution=render_resolution,
        views=render_views,
        rt_subframes=render_rt_subframes,
        analysis_up_axis=analysis_up_axis,
        analysis_front_axis=analysis_front_axis,
        # Exact project matching already binds every occurrence path,
        # topology and face subset; isolated part crops add no replay evidence.
        rgb_only=bundled_project is not None,
    )
    if not partial_live_resume:
        _run_stage(
            "render",
            source_render_command,
            log_cb,
            command_runner=command_runner,
            retry_native_crash=True,
        )
    _require_file(rendered_registry, "render")
    if (
        config.material_assignment_unit == "part_id"
        and resolved_foreground_annotations is not None
    ):
        # A resume created before continuous camera registration was added
        # legitimately has the source render but no calibrated registry.
        # Fill that missing deterministic stage instead of requiring users to
        # discard the otherwise reusable run.
        calibration_is_reusable = (
            partial_live_resume
            and camera_calibrated_registry.is_file()
            and camera_calibration_report.is_file()
        )
        camera_search_is_reusable = (
            partial_live_resume
            and camera_calibration_search_specs.is_file()
            and camera_calibration_search_report.is_file()
        )
        if camera_search_is_reusable:
            try:
                _validate_live_camera_registration_provenance(
                    camera_calibration_search_report,
                    source_registry=rendered_registry,
                    initial_view_specs=None,
                )
            except (OSError, RuntimeError, ValueError):
                # A partial run may have been created by an older contract or
                # interrupted while writing its report.  It is safer to redo
                # the low-resolution seed pass than to inherit unverifiable
                # camera evidence.
                camera_search_is_reusable = False
        if not calibration_is_reusable:
            if not camera_search_is_reusable:
                _run_stage(
                    "continuous_camera_registration_search",
                    camera_registration_command(
                        python=config.sam3_python,
                        registry=rendered_registry,
                        reference_manifest=resolved_foreground_annotations,
                        isaac_python=isaac,
                        output_dir=camera_calibration_search_dir,
                        search_resolution=256,
                        final_resolution=render_resolution,
                        rt_subframes=render_rt_subframes,
                        analysis_up_axis=analysis_up_axis,
                        analysis_front_axis=analysis_front_axis,
                    ),
                    log_cb,
                    command_runner=command_runner,
                    required_files=(
                        camera_calibration_search_specs,
                        camera_calibration_search_report,
                    ),
                )
            else:
                log_message(
                    log_cb,
                    "Reusing verified low-resolution camera-search checkpoint; "
                    "resuming only the incomplete high-resolution registration.",
                )
            # Thin pipes, fasteners and pedal edges can occupy only a few
            # pixels at the broad 256 px search resolution.  Re-open a bounded
            # pose/perspective neighborhood around that winner at delivery
            # resolution so raster quantization cannot lock the production
            # Part-ID projection to the wrong sub-pixel camera.
            _run_stage(
                "continuous_camera_registration_highres",
                camera_registration_command(
                    python=config.sam3_python,
                    registry=rendered_registry,
                    reference_manifest=resolved_foreground_annotations,
                    isaac_python=isaac,
                    output_dir=camera_calibration_dir,
                    search_resolution=render_resolution,
                    final_resolution=render_resolution,
                    rt_subframes=render_rt_subframes,
                    analysis_up_axis=analysis_up_axis,
                    analysis_front_axis=analysis_front_axis,
                    initial_view_specs=camera_calibration_search_specs,
                    search_phases=(
                        "orthographic",
                        "component_pose_recheck",
                        "settle",
                        "micro",
                        "target",
                        "lens_micro",
                        "nano",
                        "target_micro",
                        "pico",
                        "target_pico",
                    ),
                ),
                log_cb,
                command_runner=command_runner,
                required_files=(
                    camera_calibrated_registry,
                    camera_calibration_report,
                ),
            )
        _require_file(
            camera_calibrated_registry,
            "continuous_camera_registration",
        )
        _require_file(
            camera_calibration_report,
            "continuous_camera_registration",
        )
        _validate_live_camera_registration_provenance(
            camera_calibration_search_report,
            source_registry=rendered_registry,
            initial_view_specs=None,
        )
        final_camera_report = _validate_live_camera_registration_provenance(
            camera_calibration_report,
            source_registry=rendered_registry,
            initial_view_specs=camera_calibration_search_specs,
        )
        camera_alignment_acceptance = _require_complete_live_camera_alignment(
            final_camera_report,
            expected_reference_ids={
                reference_id for reference_id, _ in parsed_references
            },
        )
        write_object(
            source_paths.camera_acceptance,
            camera_alignment_acceptance,
        )
        rendered_registry = camera_calibrated_registry
        log_message(
            log_cb,
            "Part-ID evidence now uses independently calibrated continuous "
            "whole-asset cameras for every reference image; no Part-ID "
            "transform was edited. Alignment evidence tiers: "
            + ", ".join(
                f"{view_id}={row['tier']}"
                for view_id, row in sorted(camera_alignment_acceptance["views"].items())
            )
            + ".",
        )
        # This is a pre-retrieval visual-coherence constraint only.  Unlike
        # the historical source-appearance cohorts it learns membership from
        # the aligned reference pixels themselves.  It intentionally cannot
        # alter a Material ID, MDL parameter, CAD transform, or Part-ID.
        try:
            appearance_components_document = build_appearance_components(
                rendered_registry=rendered_registry,
                reference_manifest=resolved_foreground_annotations,
                camera_report=camera_calibration_report,
            )
        except AppearanceComponentError as exc:
            raise RuntimeError(
                "Unable to derive photo-supported Part-ID appearance components; "
                f"material inference was not started: {exc}"
            ) from exc
        write_object(appearance_components_report, appearance_components_document)
        appearance_summary = appearance_components_document["summary"]
        log_message(
            log_cb,
            "Photo-supported appearance components completed without material "
            "mutation: "
            f"{appearance_summary['component_count']} components cover "
            f"{appearance_summary['component_member_count']} observed CAD Part IDs; "
            f"{appearance_summary['independent_observed_part_count']} observed "
            "Part IDs remain independent.",
        )
    if resume_has_material_checkpoint:
        face_checkpoint = read_object(
            face_region_manifest, "partial-resume face-region manifest"
        )
        occurrence_asset = editable_usd or source
        if face_checkpoint.get("asset_sha256") != sha256_file(
            occurrence_asset
        ) or face_checkpoint.get("registry_sha256") != sha256_file(rendered_registry):
            raise RuntimeError(
                "Partial visual-material resume rejected: occurrence USD or "
                "rendered evidence changed"
            )
        log_message(
            log_cb,
            "Reusing the original hash-bound occurrence registry and rendered "
            "evidence; renderer output is intentionally not regenerated.",
        )


    return SourcePreparationResult(
        source_registry=source_registry,
        registry=registry,
        rendered_registry=rendered_registry,
        editable_usd=editable_usd,
        expand_report=expand_report,
        instance_root_count=instance_root_count,
        bundled_project=bundled_project,
    )


__all__ = ["SourcePreparationResult", "prepare_source_evidence"]
