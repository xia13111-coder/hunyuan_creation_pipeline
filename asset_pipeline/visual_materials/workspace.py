"""Typed artifact layout for one visual-material pipeline run.

The orchestrator used to construct every output path inline.  That mixed file
layout, control flow, and subprocess commands in one very large function.  The
types in this module make artifact ownership explicit without creating files or
changing any on-disk names.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SourceArtifacts:
    """USD registry, source renders, and whole-asset camera registration."""

    source_registry: Path
    editable_usd: Path
    expand_report: Path
    expanded_registry: Path
    render_dir: Path
    rendered_registry: Path
    camera_dir: Path
    camera_search_dir: Path
    camera_search_specs: Path
    camera_search_report: Path
    camera_calibrated_registry: Path
    camera_report: Path
    camera_acceptance: Path

    @classmethod
    def create(cls, destination: Path, source: Path) -> "SourceArtifacts":
        render_dir = destination / "renders"
        camera_dir = destination / "camera_calibration"
        camera_search_dir = camera_dir / "search_pass"
        return cls(
            source_registry=destination / "source_part_registry.json",
            editable_usd=destination / f"{source.stem}_editable.usda",
            expand_report=destination / "expand_instances_report.json",
            expanded_registry=destination / "part_registry.json",
            render_dir=render_dir,
            rendered_registry=render_dir / "part_registry.rendered.json",
            camera_dir=camera_dir,
            camera_search_dir=camera_search_dir,
            camera_search_specs=camera_search_dir / "final_view_specs.json",
            camera_search_report=camera_search_dir / "camera_calibration_report.json",
            camera_calibrated_registry=(
                camera_dir / "part_registry.camera_calibrated.json"
            ),
            camera_report=camera_dir / "camera_calibration_report.json",
            camera_acceptance=camera_dir / "camera_alignment_acceptance.json",
        )


@dataclass(frozen=True)
class InferenceArtifacts:
    """Qwen/MVInverse outputs, exact-cover policy, and publication locks."""

    root: Path
    unattended_result: Path
    staged_result: Path
    confidence_gate: Path
    staged_material_plan: Path
    group_materials: Path
    mvinverse_pbr_evidence: Path
    policy_input: Path
    policy_plan: Path
    policy_audit: Path
    publish_quality_gate: Path
    immutable_library_optimum_gate: Path
    material_selection_lock: Path
    mvinverse_ledger: Path
    inference_recovery: Path
    face_region_manifest: Path

    @classmethod
    def create(cls, destination: Path) -> "InferenceArtifacts":
        root = destination / "analysis"
        return cls(
            root=root,
            unattended_result=root / "unattended_result.json",
            staged_result=root / "staged_result.json",
            confidence_gate=root / "confidence_gate.json",
            staged_material_plan=root / "autonomous_material_plan.json",
            group_materials=root / "group_materials.json",
            mvinverse_pbr_evidence=root / "mvinverse_pbr_evidence.json",
            policy_input=root / "policy_exact_cover_input.json",
            policy_plan=root / "policy_exact_cover_plan.json",
            policy_audit=root / "policy_exact_cover_audit.json",
            publish_quality_gate=root / "publish_quality_gate.json",
            immutable_library_optimum_gate=(
                root / "immutable_library_optimum_gate.json"
            ),
            material_selection_lock=root / "material_selection_lock.json",
            mvinverse_ledger=(root / "mvinverse" / "mvinverse_inference_ledger.json"),
            inference_recovery=root / "qwen_mvinverse_recovery.json",
            face_region_manifest=root / "face_regions" / "manifest.json",
        )


@dataclass(frozen=True)
class QualityRoundArtifacts:
    """One rendered quality-comparison round."""

    root: Path
    registry: Path
    render_dir: Path
    rendered_registry: Path
    view_map: Path
    report: Path

    @classmethod
    def create(cls, root: Path) -> "QualityRoundArtifacts":
        render_dir = root / "renders"
        return cls(
            root=root,
            registry=root / "part_registry.json",
            render_dir=render_dir,
            rendered_registry=render_dir / "part_registry.rendered.json",
            view_map=root / "reference_view_map.json",
            report=root / "reference_render_comparison.json",
        )


@dataclass(frozen=True)
class LookArtifacts:
    """Authored Look layers and their apply reports."""

    initial_usd: Path
    initial_apply_report: Path
    locked_usd: Path
    locked_apply_report: Path
    repaired_usd: Path
    repaired_apply_report: Path
    repaired_instance_plan: Path

    @classmethod
    def create(cls, destination: Path, source: Path) -> "LookArtifacts":
        return cls(
            initial_usd=destination / f"{source.stem}_look.usda",
            initial_apply_report=destination / "apply_visual_materials_report.json",
            locked_usd=destination / f"{source.stem}_look_locked.usda",
            locked_apply_report=(
                destination / "apply_visual_materials_locked_report.json"
            ),
            repaired_usd=destination / f"{source.stem}_look_repaired.usda",
            repaired_apply_report=(
                destination / "apply_visual_materials_repair_report.json"
            ),
            repaired_instance_plan=destination / "quality_repair_instance_plan.json",
        )


@dataclass(frozen=True)
class PartIdArtifacts:
    """Independent Part-ID evidence, retrieval, choices, plans, and QA."""

    evidence_dir: Path
    evidence: Path
    coarse_evidence_dir: Path
    coarse_evidence: Path
    sam3_request: Path
    sam3_dir: Path
    sam3_manifest: Path
    retrieval_request: Path
    retrieval_dir: Path
    retrieval_result: Path
    qwen_dir: Path
    qwen_result: Path
    material_plan: Path
    material_audit: Path
    quality_gate: Path
    parameter_tournament_dir: Path
    parameter_tournament_plan: Path
    parameter_tournament_audit: Path

    @classmethod
    def create(cls, destination: Path, analysis: Path) -> "PartIdArtifacts":
        retrieval_dir = analysis / "part_id_visual_retrieval"
        return cls(
            evidence_dir=analysis / "part_id_reference_evidence",
            evidence=analysis / "part_id_reference_evidence.json",
            coarse_evidence_dir=analysis / "part_id_reference_evidence_coarse",
            coarse_evidence=analysis / "part_id_reference_evidence_coarse.json",
            sam3_request=analysis / "part_id_sam3_request.json",
            sam3_dir=analysis / "part_id_sam3_regions",
            sam3_manifest=analysis / "part_id_sam3_regions" / "manifest.json",
            retrieval_request=analysis / "part_id_retrieval_request.json",
            retrieval_dir=retrieval_dir,
            retrieval_result=retrieval_dir / "visual_retrieval.json",
            qwen_dir=analysis / "part_id_qwen_rerank",
            qwen_result=analysis / "part_id_qwen_choices.json",
            material_plan=analysis / "part_id_material_plan.json",
            material_audit=analysis / "part_id_material_audit.json",
            quality_gate=analysis / "part_id_quality_gate.json",
            parameter_tournament_dir=(
                destination / "visual_part_id_parameter_tournament"
            ),
            parameter_tournament_plan=(
                analysis / "part_id_parameter_tournament_plan.json"
            ),
            parameter_tournament_audit=(analysis / "part_id_parameter_tournament.json"),
        )


@dataclass(frozen=True)
class AppearanceArtifacts:
    """Photo-supported components and immutable-MDL component selection."""

    components: Path
    input_dir: Path
    evidence: Path
    memberships: Path
    retrieval_request: Path
    retrieval_dir: Path
    retrieval_result: Path
    qwen_dir: Path
    qwen_result: Path
    mdl_selection_audit: Path
    actual_mdl_tournament_dir: Path
    actual_mdl_tournament_plan: Path
    actual_mdl_tournament_audit: Path

    @classmethod
    def create(cls, destination: Path, analysis: Path) -> "AppearanceArtifacts":
        retrieval_dir = analysis / "appearance_component_visual_retrieval"
        return cls(
            components=analysis / "appearance_components.json",
            input_dir=analysis / "appearance_component_inputs",
            evidence=analysis / "appearance_component_evidence.json",
            memberships=analysis / "appearance_component_material_memberships.json",
            retrieval_request=(
                analysis / "appearance_component_retrieval_request.json"
            ),
            retrieval_dir=retrieval_dir,
            retrieval_result=retrieval_dir / "visual_retrieval.json",
            qwen_dir=analysis / "appearance_component_qwen_rerank",
            qwen_result=analysis / "appearance_component_qwen_choices.json",
            mdl_selection_audit=analysis / "appearance_component_mdl_selection.json",
            actual_mdl_tournament_dir=(
                destination / "visual_appearance_component_mdl_tournament"
            ),
            actual_mdl_tournament_plan=(
                analysis / "appearance_component_actual_mdl_tournament_plan.json"
            ),
            actual_mdl_tournament_audit=(
                analysis / "appearance_component_actual_mdl_tournament.json"
            ),
        )


@dataclass(frozen=True)
class LegacyOptimizationArtifacts:
    """Compatibility paths for older palette and appearance repair branches."""

    quality_repair_plan: Path
    quality_repair_audit: Path
    quality_resolution: Path
    membership_dir: Path
    membership_view_map: Path
    membership_plan: Path
    membership_instance_plan: Path
    membership_audit: Path
    appearance_baseline_quality: Path
    appearance_baseline_measurement: Path
    appearance_contract: Path
    appearance_candidate_plan: Path
    appearance_candidate_plan_apply: Path
    appearance_candidate_usd: Path
    appearance_candidate_apply_report: Path
    appearance_candidate_instance_plan: Path
    appearance_candidate_quality: QualityRoundArtifacts
    appearance_candidate_raw_quality: Path
    appearance_candidate_measured_quality: Path
    appearance_candidate_measurement: Path
    appearance_validation: Path
    exact_mdl_dir: Path
    exact_mdl_planning: Path
    exact_mdl_audit: Path
    exact_mdl_plan: Path
    exact_mdl_instance_plan: Path
    exact_mdl_view_map: Path
    exact_mdl_final_quality: Path
    visual_group_plan: Path
    visual_group_audit: Path

    @classmethod
    def create(
        cls,
        destination: Path,
        source: Path,
        analysis: Path,
    ) -> "LegacyOptimizationArtifacts":
        membership_dir = destination / "visual_membership_tournament"
        appearance_quality_root = destination / "visual_quality_appearance_candidate"
        appearance_quality = QualityRoundArtifacts.create(appearance_quality_root)
        exact_mdl_dir = destination / "visual_exact_mdl_tournament"
        return cls(
            quality_repair_plan=analysis / "quality_repair_plan.json",
            quality_repair_audit=analysis / "quality_repair_audit.json",
            quality_resolution=analysis / "visual_quality_resolution.json",
            membership_dir=membership_dir,
            membership_view_map=membership_dir / "reference_view_map.json",
            membership_plan=analysis / "dominant_assembly_membership_plan.json",
            membership_instance_plan=(
                destination / "dominant_assembly_membership_instance_plan.json"
            ),
            membership_audit=(
                analysis / "dominant_assembly_membership_tournament.json"
            ),
            appearance_baseline_quality=(
                analysis / "appearance_optimization_baseline_quality.json"
            ),
            appearance_baseline_measurement=(
                analysis / "appearance_optimization_baseline_measurement.json"
            ),
            appearance_contract=analysis / "appearance_optimization_contract.json",
            appearance_candidate_plan=(
                analysis / "appearance_optimization_candidate_plan.json"
            ),
            appearance_candidate_plan_apply=(
                analysis / "appearance_optimization_candidate_plan_apply.json"
            ),
            appearance_candidate_usd=(
                destination / f"{source.stem}_look_appearance_candidate.usda"
            ),
            appearance_candidate_apply_report=(
                destination / "apply_visual_materials_appearance_candidate_report.json"
            ),
            appearance_candidate_instance_plan=(
                destination / "appearance_candidate_instance_plan.json"
            ),
            appearance_candidate_quality=appearance_quality,
            appearance_candidate_raw_quality=(
                appearance_quality_root / "reference_render_comparison.json"
            ),
            appearance_candidate_measured_quality=(
                appearance_quality_root / "reference_render_comparison.measured.json"
            ),
            appearance_candidate_measurement=(
                analysis / "appearance_optimization_candidate_measurement.json"
            ),
            appearance_validation=analysis / "appearance_optimization_validation.json",
            exact_mdl_dir=exact_mdl_dir,
            exact_mdl_planning=analysis / "exact_mdl_tournament_planning.json",
            exact_mdl_audit=analysis / "exact_mdl_tournament.json",
            exact_mdl_plan=analysis / "exact_mdl_tournament_plan.json",
            exact_mdl_instance_plan=destination
            / "exact_mdl_tournament_instance_plan.json",
            exact_mdl_view_map=exact_mdl_dir / "reference_view_map.json",
            exact_mdl_final_quality=(
                exact_mdl_dir / "final_reference_render_comparison.json"
            ),
            visual_group_plan=analysis / "visual_group_annotated_plan.json",
            visual_group_audit=analysis / "visual_group_annotation.json",
        )


@dataclass(frozen=True)
class VisualMaterialWorkspace:
    """Complete, side-effect-free artifact map for one pipeline invocation."""

    destination: Path
    source: SourceArtifacts
    inference: InferenceArtifacts
    look: LookArtifacts
    quality: QualityRoundArtifacts
    repaired_quality: QualityRoundArtifacts
    part_id: PartIdArtifacts
    appearance: AppearanceArtifacts
    legacy: LegacyOptimizationArtifacts
    quality_camera_view_specs: Path

    @classmethod
    def create(
        cls,
        *,
        destination: Path,
        source: Path,
    ) -> "VisualMaterialWorkspace":
        inference = InferenceArtifacts.create(destination)
        return cls(
            destination=destination,
            source=SourceArtifacts.create(destination, source),
            inference=inference,
            look=LookArtifacts.create(destination, source),
            quality=QualityRoundArtifacts.create(destination / "visual_quality"),
            repaired_quality=QualityRoundArtifacts.create(
                destination / "visual_quality_repair"
            ),
            part_id=PartIdArtifacts.create(destination, inference.root),
            appearance=AppearanceArtifacts.create(destination, inference.root),
            legacy=LegacyOptimizationArtifacts.create(
                destination,
                source,
                inference.root,
            ),
            quality_camera_view_specs=(
                destination / "visual_quality" / "camera_view_specs.json"
            ),
        )


__all__ = [
    "AppearanceArtifacts",
    "InferenceArtifacts",
    "LegacyOptimizationArtifacts",
    "LookArtifacts",
    "PartIdArtifacts",
    "QualityRoundArtifacts",
    "SourceArtifacts",
    "VisualMaterialWorkspace",
]
