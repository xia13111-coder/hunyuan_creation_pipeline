"""Stable constants for visual-quality evidence contracts."""

from __future__ import annotations


MATERIAL_SELECTION_OBJECTIVE_SEMANTIC = "semantic_compatible_visual"
MATERIAL_SELECTION_OBJECTIVE_VISUAL = "visual_similarity"
QUALITY_REPAIR_REPORT_SCHEMA_VERSION = "qwen-quality-repair-report/v1"
QUALITY_REPAIR_PLAN_MODE = "quality_missing_canonical_group_repair/v1"
QUALITY_REPAIR_PROVENANCE_FIELD = "quality_repair"
QUALITY_REPAIR_REASON_CODES = (
    "QA_MISSING_CANONICAL_GROUP_MULTI_VIEW",
    "QA_TRUSTED_PART_GROUP_LOCALIZATION",
    "QA_CONFIRMED_WHITELIST_MATERIAL",
)
QUALITY_REPAIR_PROVISIONAL_REASON_CODES = (
    "QA_MISSING_CANONICAL_GROUP_MULTI_VIEW",
    "QA_TRUSTED_PART_GROUP_LOCALIZATION",
    "QA_HIGH_CONFIDENCE_WHITELIST_MATERIAL_CANDIDATE",
    "QA_POST_RENDER_VALIDATION_REQUIRED",
)
QUALITY_REPAIR_PROVISIONAL_MATERIAL_BASIS = (
    "high_confidence_whitelist_candidate_pending_render_qa"
)
QUALITY_REPAIR_MIN_PROVISIONAL_MATERIAL_CONFIDENCE = 0.85
QUALITY_REPAIR_LOCALIZATION_LANES = {
    "stable_spatial_multiview",
    "bounded_spatial_multiview",
    "qa_confirmed_three_view_semantic_review",
    "exact_spatial_single_qa_view",
    "exact_spatial_single_qa_view_with_semantic_anchor",
    "exact_spatial_single_view_with_multiview_anchor",
    "exact_spatial_single_qa_view_with_spatial_anchor",
    "dominant_chromatic_residual_exact_single_view",
    "dark_foreground_achromatic_residual_exact_projection",
    "multiview_dark_part_identity_consensus",
    "repeated_geometry_dark_residual_exact_projection",
    "source_identity_anchored_single_view_diagnostic",
    "source_identity_cohort_multiview_consensus",
    "dominant_assembly_cohort_expansion",
    "single_strict_anchor_bounded_signature_sibling",
}
QUALITY_REPAIR_MULTIVIEW_SEMANTIC_REVIEW_LANE = (
    "qa_confirmed_three_view_semantic_review"
)
QUALITY_REPAIR_MULTIVIEW_SEMANTIC_REVIEW_MIN_SUPPORTS = 3
QUALITY_REPAIR_SEMANTIC_SINGLE_VIEW_LANE = (
    "exact_spatial_single_qa_view_with_semantic_anchor"
)
QUALITY_REPAIR_ANCHORED_SINGLE_VIEW_LANE = (
    "exact_spatial_single_view_with_multiview_anchor"
)
QUALITY_REPAIR_SPATIAL_ANCHOR_SINGLE_VIEW_LANE = (
    "exact_spatial_single_qa_view_with_spatial_anchor"
)
QUALITY_REPAIR_DOMINANT_RESIDUAL_SINGLE_VIEW_LANE = (
    "dominant_chromatic_residual_exact_single_view"
)
QUALITY_REPAIR_DOMINANT_RESIDUAL_DEFICIT_SOURCE = "dominant_mass_local_projection"
QUALITY_REPAIR_DARK_FOREGROUND_RESIDUAL_LANE = (
    "dark_foreground_achromatic_residual_exact_projection"
)
QUALITY_REPAIR_MULTIVIEW_DARK_IDENTITY_LANE = "multiview_dark_part_identity_consensus"
QUALITY_REPAIR_DARK_FOREGROUND_RESIDUAL_DEFICIT_SOURCE = (
    "dark_foreground_achromatic_residual"
)
QUALITY_REPAIR_REPEATED_GEOMETRY_DARK_RESIDUAL_LANE = (
    "repeated_geometry_dark_residual_exact_projection"
)
QUALITY_REPAIR_SOURCE_IDENTITY_LANE = "source_identity_anchored_single_view_diagnostic"
QUALITY_REPAIR_SOURCE_IDENTITY_COHORT_CONSENSUS_LANE = (
    "source_identity_cohort_multiview_consensus"
)
QUALITY_REPAIR_DOMINANT_ASSEMBLY_COHORT_LANE = "dominant_assembly_cohort_expansion"
QUALITY_REPAIR_BOUNDED_SIGNATURE_SIBLING_COHORT_LANE = (
    "single_strict_anchor_bounded_signature_sibling"
)
QUALITY_REPAIR_SOURCE_IDENTITY_MIN_SIGNATURE_COUNT = 2
QUALITY_REPAIR_SOURCE_IDENTITY_MAX_REGISTRY_FRACTION = 0.05
QUALITY_REPAIR_SOURCE_IDENTITY_MAX_ASSEMBLY_COHORT_SIZE = 4
QUALITY_REPAIR_DARK_FOREGROUND_REASON_CODES = (
    "QA_DARK_FOREGROUND_ACHROMATIC_RESIDUAL",
    "QA_TRUSTED_PART_GROUP_LOCALIZATION",
    "QA_CONFIRMED_WHITELIST_MATERIAL",
)
QUALITY_REPAIR_REPEATED_GEOMETRY_DARK_REASON_CODES = (
    "QA_MISSING_CANONICAL_GROUP_SINGLE_VIEW",
    "QA_REPEATED_GEOMETRY_COHORT_EXACT_PROJECTION",
    "QA_CONFIRMED_WHITELIST_MATERIAL",
)
QUALITY_REPAIR_DOMINANT_ASSEMBLY_REASON_CODES = (
    "QA_DOMINANT_CHROMATIC_GROUP_DEFICIT",
    "QA_STRICT_MULTIVIEW_SPATIAL_ASSEMBLY_ANCHORS",
    "QA_HASH_BOUND_ASSEMBLY_COHORT_EXPANSION",
    "QA_POST_RENDER_MEMBERSHIP_VALIDATION_REQUIRED",
)
QUALITY_REPAIR_BOUNDED_SIGNATURE_SIBLING_REASON_CODES = (
    "QA_MISSING_CANONICAL_GROUP_MULTI_VIEW",
    "QA_EXACT_DIAGNOSTIC_SOURCE_SIGNATURE_ANCHOR",
    "QA_BOUNDED_SOURCE_SIGNATURE_SIBLING",
    "QA_POST_RENDER_MEMBERSHIP_VALIDATION_REQUIRED",
)
QUALITY_REPAIR_DARK_FOREGROUND_THRESHOLDS: dict[str, float | int] = {
    "normalized_long_edge_pixels": 512,
    "minimum_normalized_projected_pixels": 96,
    "near_black_max_channel_exclusive": 97,
    "near_black_max_channel_spread": 32,
    "minimum_near_black_share": 0.60,
    "minimum_non_background_pixels": 24,
    "minimum_dark_signal_share": 0.20,
    "minimum_dark_signal_purity": 0.45,
    "core_distance_pixels": 2.2,
    "minimum_core_pixels": 16,
    "minimum_core_dark_signal_share": 0.25,
    "minimum_adaptive_edge_density": 0.25,
    "minimum_null_offset_pixels": 7,
    "minimum_null_valid_area_ratio": 0.80,
    "minimum_valid_null_shifts": 4,
    "minimum_null_q75_margin": 0.10,
    "minimum_alignment_score": 0.85,
    "minimum_projection_score": 0.85,
    "minimum_projection_iou": 0.85,
    "minimum_ecc_correlation": 0.90,
}




QUALITY_REPAIR_DARK_FOREGROUND_MAX_SINGLE_CONTRIBUTION_FACTOR = 1.25
QUALITY_REPAIR_DARK_FOREGROUND_MAX_TOTAL_CONTRIBUTION_FACTOR = 1.35
QUALITY_REPAIR_REPEATED_GEOMETRY_DARK_THRESHOLDS: dict[str, float | int] = {
    "minimum_cohort_size": 2,
    "maximum_cohort_size": 4,
    "maximum_registry_fraction": 0.05,
    "minimum_alignment_score": 0.80,
    "minimum_projection_score": 0.85,
    "minimum_projection_iou": 0.85,
    "minimum_ecc_correlation": 0.90,
    "minimum_projected_pixels": 256,
    "minimum_direct_target_share": 0.60,
    "minimum_direct_target_margin": 0.25,
    "minimum_bbox_target_share": 0.90,
    "minimum_bbox_target_margin": 0.80,
    "minimum_perturbation_target_share": 0.60,
    "minimum_perturbation_target_margin": 0.20,
    "minimum_mapping_confidence": 0.60,
    "minimum_budget_contribution_factor": 0.75,
    "maximum_budget_contribution_factor": 1.35,
    "maximum_semantic_rejected_group_share": 0.05,
    "maximum_semantic_rejected_group_outlier_share": 0.15,
    "minimum_clean_semantic_disproof_samples": 5,
    "maximum_review_rejected_group_share": 0.15,
    "maximum_review_rejected_group_outlier_share": 0.25,
}
QUALITY_REPAIR_SEMANTIC_REVIEW_OVERRIDE_THRESHOLDS: dict[str, float | int] = {
    "minimum_alignment_score": 0.75,
    "minimum_projection_iou": 0.80,
    "minimum_ecc_correlation": 0.85,
    "minimum_projected_pixels": 1024,
    "minimum_direct_target_share": 0.70,
    "minimum_direct_target_margin": 0.60,
    "minimum_bbox_target_share": 0.65,
    "minimum_bbox_target_margin": 0.55,
    "minimum_perturbation_target_share": 0.70,
    "minimum_perturbation_target_margin": 0.60,
    "maximum_rejected_group_share": 0.05,
    "maximum_rejected_group_outlier_share": 0.15,
    "minimum_clean_rejected_group_samples": 5,
    "minimum_anchor_effective_confidence": 0.80,
}
QUALITY_RESOLUTION_SCHEMA_VERSION = "qwen-visual-quality-resolution/v1"
QUALITY_RESOLUTION_LIMITED_PASS = "MATERIAL_ACCEPTED_WITH_GEOMETRY_POSE_LIMITATION"
QUALITY_RESOLUTION_FAIL_CLOSED = "FAIL_CLOSED"
QUALITY_RESOLUTION_THRESHOLDS = {
    "minimum_alignment_score": 0.75,
    "minimum_projection_iou": 0.80,
    "minimum_ecc_correlation": 0.85,
    "minimum_coverage_projection_iou": 0.70,
    "minimum_coverage_ecc_correlation": 0.75,
    "minimum_mapping_preview_score": 0.60,
    "minimum_mapping_preview_silhouette_iou": 0.50,
    "minimum_reference_evidence_pixels": 128,
    "maximum_reference_group_share": 0.05,
    "maximum_limited_share_per_view": 0.075,
    "minimum_limited_aggregate_color_score": 0.60,
    "minimum_repeated_candidate_count": 4,
    "minimum_other_visible_render_views": 2,
    "minimum_source_coverage_recall": 0.35,
    "minimum_cross_view_coverage_recall": 0.10,
    "minimum_cross_view_delivered_views": 2,
    "minimum_owner_projected_pixels": 256,
    "minimum_owner_color_share": 0.70,
    "minimum_owner_color_margin": 0.60,
    "maximum_target_share_in_owner": 0.05,
    "minimum_accepted_box_owner_overlap": 0.50,
}
QUALITY_DOMINANT_THRESHOLD_FIELDS = (
    "minimum_dominant_reference_share",
    "minimum_dominant_share_margin",
    "minimum_dominant_mass_recall",
    "minimum_dominant_absolute_deficit",
    "minimum_dominant_silhouette_iou",
)
# Public compatibility constants retained for policy/report consumers.  New
# parameterization preserves the exact selected material instead of
# substituting this generic export.
GENERIC_STEEL_PAINTED = "mdl:Miscellaneous/Paint_Matte.mdl#Paint_Matte"
STEEL_PAINTED_MODULE_PREFIX = "mdl:Miscellaneous/Paint_Matte.mdl#"
QUALITY_STATUSES = {"PASS", "REVIEW", "FAIL", "INSUFFICIENT_EVIDENCE"}
PART_ID_QUALITY_GATE_SCHEMA_VERSION = "asset-pipeline-part-id-quality-gate/v1"
PART_ID_INAPPLICABLE_PALETTE_REASONS = frozenset(
    {
        "trusted_palette_group_missing_from_render",
    }
)
LIGHTING_STATISTICS_SCHEMA_VERSION = "qwen-lighting-normalized-group-statistics/v1"
APPEARANCE_OPTIMIZATION_STATUSES = {
    "SKIPPED_INPUTS_UNAVAILABLE",
    "SKIPPED_UNSAFE_BASELINE",
    "SKIPPED_NO_LIGHTING_NORMALIZED_GROUPS",
    "NOT_APPLICABLE",
    "ACCEPTED",
    "REJECTED_FAIL_CLOSED",
    "SKIPPED_SELECTED_MDL_IMMUTABLE",
}

__all__ = [
    "APPEARANCE_OPTIMIZATION_STATUSES",
    "GENERIC_STEEL_PAINTED",
    "LIGHTING_STATISTICS_SCHEMA_VERSION",
    "MATERIAL_SELECTION_OBJECTIVE_SEMANTIC",
    "MATERIAL_SELECTION_OBJECTIVE_VISUAL",
    "PART_ID_INAPPLICABLE_PALETTE_REASONS",
    "PART_ID_QUALITY_GATE_SCHEMA_VERSION",
    "QUALITY_DOMINANT_THRESHOLD_FIELDS",
    "QUALITY_REPAIR_ANCHORED_SINGLE_VIEW_LANE",
    "QUALITY_REPAIR_BOUNDED_SIGNATURE_SIBLING_COHORT_LANE",
    "QUALITY_REPAIR_BOUNDED_SIGNATURE_SIBLING_REASON_CODES",
    "QUALITY_REPAIR_DARK_FOREGROUND_MAX_SINGLE_CONTRIBUTION_FACTOR",
    "QUALITY_REPAIR_DARK_FOREGROUND_MAX_TOTAL_CONTRIBUTION_FACTOR",
    "QUALITY_REPAIR_DARK_FOREGROUND_REASON_CODES",
    "QUALITY_REPAIR_DARK_FOREGROUND_RESIDUAL_DEFICIT_SOURCE",
    "QUALITY_REPAIR_DARK_FOREGROUND_RESIDUAL_LANE",
    "QUALITY_REPAIR_DARK_FOREGROUND_THRESHOLDS",
    "QUALITY_REPAIR_DOMINANT_ASSEMBLY_COHORT_LANE",
    "QUALITY_REPAIR_DOMINANT_ASSEMBLY_REASON_CODES",
    "QUALITY_REPAIR_DOMINANT_RESIDUAL_DEFICIT_SOURCE",
    "QUALITY_REPAIR_DOMINANT_RESIDUAL_SINGLE_VIEW_LANE",
    "QUALITY_REPAIR_LOCALIZATION_LANES",
    "QUALITY_REPAIR_MIN_PROVISIONAL_MATERIAL_CONFIDENCE",
    "QUALITY_REPAIR_MULTIVIEW_DARK_IDENTITY_LANE",
    "QUALITY_REPAIR_MULTIVIEW_SEMANTIC_REVIEW_LANE",
    "QUALITY_REPAIR_MULTIVIEW_SEMANTIC_REVIEW_MIN_SUPPORTS",
    "QUALITY_REPAIR_PLAN_MODE",
    "QUALITY_REPAIR_PROVENANCE_FIELD",
    "QUALITY_REPAIR_PROVISIONAL_MATERIAL_BASIS",
    "QUALITY_REPAIR_PROVISIONAL_REASON_CODES",
    "QUALITY_REPAIR_REASON_CODES",
    "QUALITY_REPAIR_REPEATED_GEOMETRY_DARK_REASON_CODES",
    "QUALITY_REPAIR_REPEATED_GEOMETRY_DARK_RESIDUAL_LANE",
    "QUALITY_REPAIR_REPEATED_GEOMETRY_DARK_THRESHOLDS",
    "QUALITY_REPAIR_REPORT_SCHEMA_VERSION",
    "QUALITY_REPAIR_SEMANTIC_REVIEW_OVERRIDE_THRESHOLDS",
    "QUALITY_REPAIR_SEMANTIC_SINGLE_VIEW_LANE",
    "QUALITY_REPAIR_SOURCE_IDENTITY_COHORT_CONSENSUS_LANE",
    "QUALITY_REPAIR_SOURCE_IDENTITY_LANE",
    "QUALITY_REPAIR_SOURCE_IDENTITY_MAX_ASSEMBLY_COHORT_SIZE",
    "QUALITY_REPAIR_SOURCE_IDENTITY_MAX_REGISTRY_FRACTION",
    "QUALITY_REPAIR_SOURCE_IDENTITY_MIN_SIGNATURE_COUNT",
    "QUALITY_REPAIR_SPATIAL_ANCHOR_SINGLE_VIEW_LANE",
    "QUALITY_RESOLUTION_FAIL_CLOSED",
    "QUALITY_RESOLUTION_LIMITED_PASS",
    "QUALITY_RESOLUTION_SCHEMA_VERSION",
    "QUALITY_RESOLUTION_THRESHOLDS",
    "QUALITY_STATUSES",
    "STEEL_PAINTED_MODULE_PREFIX",
]
