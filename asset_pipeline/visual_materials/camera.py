"""Camera-registration evidence contracts for the visual-material pipeline."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .config import read_object, write_object


CAMERA_ALIGNMENT_USABLE_MINIMUM_IOU = 0.92
CAMERA_ALIGNMENT_USABLE_MAXIMUM_BOUNDARY_P95_PX = 10.0
CAMERA_ALIGNMENT_DOWNWEIGHTED_MINIMUM_IOU = 0.88
CAMERA_ALIGNMENT_DOWNWEIGHTED_MAXIMUM_BOUNDARY_P95_PX = 15.0
CAMERA_ALIGNMENT_LOCAL_BOX_MINIMUM_IOU = 0.80
CAMERA_ALIGNMENT_LOCAL_BOX_MAXIMUM_BOUNDARY_P95_PX = 28.0
CAMERA_ALIGNMENT_LOCAL_BOX_MINIMUM_RECALL = 0.90
CAMERA_ALIGNMENT_LOCAL_BOX_MINIMUM_STRUCTURE_SCORE = 0.60


def continuous_camera_view_specs(
    registry: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Reconstruct reusable custom camera specs from a rendered registry."""

    render_set = registry.get("render_set")
    if not isinstance(render_set, Mapping):
        return None
    requested = render_set.get("requested_view_tokens")
    raw_views = render_set.get("views")
    if (
        not isinstance(requested, list)
        or not requested
        or any(not isinstance(item, str) or not item for item in requested)
        or not isinstance(raw_views, list)
    ):
        return None
    views_by_id = {
        item.get("view_id"): item
        for item in raw_views
        if isinstance(item, Mapping)
        and isinstance(item.get("view_id"), str)
        and item.get("view_id")
    }
    has_continuous_calibration = any(
        isinstance(item.get("camera_calibration"), Mapping)
        for item in views_by_id.values()
    )
    if not has_continuous_calibration:
        return None

    specs: list[dict[str, Any]] = []
    for view_id in requested:
        view = views_by_id.get(view_id)
        if not isinstance(view, Mapping):
            raise RuntimeError(
                f"Continuous camera registry lacks requested view {view_id!r}"
            )
        direction = view.get("analysis_direction")
        up_axis = view.get("analysis_camera_up_axis")
        focal_length = view.get("focal_length_mm")
        distance_multiplier = view.get("camera_distance_multiplier")
        calibration = view.get("camera_calibration")
        target_offset_u = view.get(
            "camera_target_offset_u",
            calibration.get("target_offset_u", 0.0)
            if isinstance(calibration, Mapping)
            else 0.0,
        )
        target_offset_v = view.get(
            "camera_target_offset_v",
            calibration.get("target_offset_v", 0.0)
            if isinstance(calibration, Mapping)
            else 0.0,
        )
        roll_degrees = view.get(
            "camera_roll_degrees",
            calibration.get("roll_degrees", 0.0)
            if isinstance(calibration, Mapping)
            else 0.0,
        )
        principal_point_u = view.get(
            "camera_principal_point_u",
            calibration.get("principal_point_u", 0.0)
            if isinstance(calibration, Mapping)
            else 0.0,
        )
        principal_point_v = view.get(
            "camera_principal_point_v",
            calibration.get("principal_point_v", 0.0)
            if isinstance(calibration, Mapping)
            else 0.0,
        )
        radial_distortion_k1 = view.get(
            "camera_radial_distortion_k1",
            calibration.get("radial_distortion_k1", 0.0)
            if isinstance(calibration, Mapping)
            else 0.0,
        )
        radial_distortion_k2 = view.get(
            "camera_radial_distortion_k2",
            calibration.get("radial_distortion_k2", 0.0)
            if isinstance(calibration, Mapping)
            else 0.0,
        )
        projection_mode = view.get(
            "camera_projection_mode",
            calibration.get("projection_mode", "perspective")
            if isinstance(calibration, Mapping)
            else "perspective",
        )
        orthographic_span_multiplier = view.get(
            "camera_orthographic_span_multiplier",
            calibration.get("orthographic_span_multiplier", 2.0)
            if isinstance(calibration, Mapping)
            else 2.0,
        )
        if (
            not isinstance(direction, list)
            or len(direction) != 3
            or not isinstance(up_axis, list)
            or len(up_axis) != 3
            or any(
                isinstance(component, bool) or not isinstance(component, (int, float))
                for component in [*direction, *up_axis]
            )
            or isinstance(focal_length, bool)
            or not isinstance(focal_length, (int, float))
            or isinstance(distance_multiplier, bool)
            or not isinstance(distance_multiplier, (int, float))
            or not isinstance(calibration, Mapping)
            or isinstance(target_offset_u, bool)
            or not isinstance(target_offset_u, (int, float))
            or isinstance(target_offset_v, bool)
            or not isinstance(target_offset_v, (int, float))
            or projection_mode not in {"perspective", "orthographic"}
            or any(
                isinstance(value, bool) or not isinstance(value, (int, float))
                for value in (
                    roll_degrees,
                    principal_point_u,
                    principal_point_v,
                    radial_distortion_k1,
                    radial_distortion_k2,
                )
            )
            or isinstance(orthographic_span_multiplier, bool)
            or not isinstance(orthographic_span_multiplier, (int, float))
        ):
            raise RuntimeError(
                "Continuous camera registry has an incomplete camera contract "
                f"for view {view_id!r}"
            )
        specs.append(
            {
                "view_id": view_id,
                "analysis_direction": [float(value) for value in direction],
                "analysis_up_axis": [float(value) for value in up_axis],
                "focal_length_mm": float(focal_length),
                "distance_multiplier": float(distance_multiplier),
                "target_offset_u": float(target_offset_u),
                "target_offset_v": float(target_offset_v),
                "roll_degrees": float(roll_degrees),
                "principal_point_u": float(principal_point_u),
                "principal_point_v": float(principal_point_v),
                "radial_distortion_k1": float(radial_distortion_k1),
                "radial_distortion_k2": float(radial_distortion_k2),
                "projection_mode": str(projection_mode),
                "orthographic_span_multiplier": float(orthographic_span_multiplier),
                "calibration": dict(calibration),
            }
        )
    return {
        "schema_version": "qwen-camera-view-specs/v1",
        "views": specs,
    }


def validate_live_camera_registration_provenance(
    report_path: Path,
    *,
    source_registry: Path,
    initial_view_specs: Path | None,
) -> dict[str, Any]:
    """Reject stale, legacy, spatial-map or per-Part camera reuse in live mode."""

    report = read_object(report_path, "live camera calibration report")
    if report.get("schema_version") != "qwen-whole-asset-camera-calibration/v9":
        raise RuntimeError(
            "Live camera calibration was not produced by the current from-zero "
            "whole-asset registration contract"
        )
    expected_registry = str(source_registry.expanduser().resolve(strict=True))
    if report.get("source_registry") != expected_registry:
        raise RuntimeError(
            "Live camera calibration source registry differs from this run"
        )
    if report.get("source_spatial_mapping") is not None:
        raise RuntimeError(
            "Live camera calibration must not reuse a previous spatial mapping"
        )
    seed_search = report.get("seed_search")
    if not isinstance(seed_search, Mapping):
        raise RuntimeError("Live camera calibration lacks seed provenance")
    if initial_view_specs is None:
        expected_specs: str | None = None
        expected_mode = "source_render_continuous_seed"
    else:
        expected_specs = str(initial_view_specs.expanduser().resolve(strict=True))
        expected_mode = "existing_continuous_camera_specs"
    if (
        report.get("source_initial_view_specs") != expected_specs
        or seed_search.get("mode") != expected_mode
    ):
        raise RuntimeError(
            "Live camera calibration used camera seeds outside the current two-pass run"
        )
    if (
        report.get("whole_asset_only") is not True
        or report.get("per_part_geometric_warp_applied") is not False
    ):
        raise RuntimeError(
            "Live camera calibration violated the rigid whole-asset contract"
        )
    intrinsics = report.get("camera_intrinsics_optimized")
    extrinsics = report.get("camera_extrinsics_optimized")
    if (
        not isinstance(intrinsics, list)
        or "projection_mode" not in intrinsics
        or "focal_length_mm" not in intrinsics
        or "principal_point_u" not in intrinsics
        or "principal_point_v" not in intrinsics
        or "radial_distortion_k1" not in intrinsics
        or "radial_distortion_k2" not in intrinsics
        or not isinstance(extrinsics, list)
        or "orbit_azimuth" not in extrinsics
        or "orbit_elevation" not in extrinsics
        or "camera_roll" not in extrinsics
        or "optical_axis_target_u" not in extrinsics
        or "optical_axis_target_v" not in extrinsics
    ):
        raise RuntimeError(
            "Live camera calibration did not optimize the complete generic camera model"
        )
    return report


def require_complete_live_camera_alignment(
    report: Mapping[str, Any],
    *,
    expected_reference_ids: set[str],
) -> dict[str, Any]:
    """Classify usable whole-asset registration views by evidence weight."""

    raw_views = report.get("views")
    if not isinstance(raw_views, list) or not raw_views:
        raise RuntimeError("Live camera calibration report has no reference views")
    views: dict[str, Mapping[str, Any]] = {}
    for raw in raw_views:
        if not isinstance(raw, Mapping):
            raise RuntimeError("Live camera calibration view is malformed")
        reference_id = raw.get("reference_view_id")
        if (
            not isinstance(reference_id, str)
            or not reference_id
            or reference_id in views
        ):
            raise RuntimeError("Live camera calibration reference IDs are invalid")
        views[reference_id] = raw
    if set(views) != expected_reference_ids:
        raise RuntimeError(
            "Live camera calibration does not exactly cover this run's reference views"
        )
    rejected: list[str] = []
    acceptance: dict[str, dict[str, Any]] = {}
    anchor_view_ids: list[str] = []
    for reference_id in sorted(expected_reference_ids):
        view = views[reference_id]
        final = view.get("final")
        target = view.get("complete_alignment_target")
        if not isinstance(final, Mapping) or not isinstance(target, Mapping):
            raise RuntimeError(
                f"Live camera calibration lacks final metrics for {reference_id}"
            )
        iou = final.get("projection_iou")
        boundary = final.get("boundary_p95_px")
        if (
            isinstance(iou, bool)
            or not isinstance(iou, (int, float))
            or isinstance(boundary, bool)
            or not isinstance(boundary, (int, float))
        ):
            raise RuntimeError(
                f"Live camera calibration metrics are invalid for {reference_id}"
            )
        numeric_iou = float(iou)
        numeric_boundary = float(boundary)
        raw_recall = final.get("target_recall")
        raw_structure = final.get("structure_score")
        numeric_recall = (
            float(raw_recall)
            if isinstance(raw_recall, (int, float))
            and not isinstance(raw_recall, bool)
            else None
        )
        numeric_structure = (
            float(raw_structure)
            if isinstance(raw_structure, (int, float))
            and not isinstance(raw_structure, bool)
            else None
        )
        if view.get("complete_alignment_passed") is True:
            tier = "strict"
            weight = 1.0
            anchor_view_ids.append(reference_id)
        elif (
            numeric_iou >= CAMERA_ALIGNMENT_USABLE_MINIMUM_IOU
            and numeric_boundary <= CAMERA_ALIGNMENT_USABLE_MAXIMUM_BOUNDARY_P95_PX
        ):
            tier = "usable_box_correspondence"
            weight = 0.80
            anchor_view_ids.append(reference_id)
        elif (
            numeric_iou >= CAMERA_ALIGNMENT_DOWNWEIGHTED_MINIMUM_IOU
            and numeric_boundary
            <= CAMERA_ALIGNMENT_DOWNWEIGHTED_MAXIMUM_BOUNDARY_P95_PX
        ):
            tier = "downweighted_box_correspondence"
            weight = 0.55
            anchor_view_ids.append(reference_id)
        elif (
            numeric_iou >= CAMERA_ALIGNMENT_LOCAL_BOX_MINIMUM_IOU
            and numeric_boundary <= CAMERA_ALIGNMENT_LOCAL_BOX_MAXIMUM_BOUNDARY_P95_PX
            and numeric_recall is not None
            and numeric_recall >= CAMERA_ALIGNMENT_LOCAL_BOX_MINIMUM_RECALL
            and numeric_structure is not None
            and numeric_structure
            >= CAMERA_ALIGNMENT_LOCAL_BOX_MINIMUM_STRUCTURE_SCORE
        ):
            # The main assembly is registered, but detached accessories or
            # thin hoses make the whole silhouette unsuitable as a global
            # anchor. Keep this view only for bounded per-Part box refinement.
            tier = "local_box_refinement_only"
            weight = 0.35
        else:
            rejected.append(
                f"{reference_id}(IoU={numeric_iou:.4f}, "
                f"boundary_p95={numeric_boundary:.2f}px)"
            )
            acceptance[reference_id] = {
                "tier": "rejected_for_part_id_evidence",
                "observation_eligible": False,
                "evidence_weight": 0.0,
                "projection_iou": numeric_iou,
                "boundary_p95_px": numeric_boundary,
                "target_recall": numeric_recall,
                "structure_score": numeric_structure,
            }
            continue
        acceptance[reference_id] = {
            "tier": tier,
            "observation_eligible": True,
            "evidence_weight": weight,
            "projection_iou": numeric_iou,
            "boundary_p95_px": numeric_boundary,
            "target_recall": numeric_recall,
            "structure_score": numeric_structure,
        }
    minimum_anchor_views = min(2, len(expected_reference_ids))
    if len(anchor_view_ids) < minimum_anchor_views:
        raise RuntimeError(
            "Whole-asset camera registration has too few reliable main-assembly "
            f"anchors ({len(anchor_view_ids)}/{minimum_anchor_views}); refusing "
            "Part-ID material inference. Rejected or local-only views: "
            + ", ".join(rejected or sorted(expected_reference_ids))
        )
    return {
        "policy": "two_layer_box_first_part_id_alignment/v2",
        "strict_target": {
            "minimum_iou": 0.97,
            "maximum_boundary_p95_px": 3.0,
        },
        "usable_target": {
            "minimum_iou": CAMERA_ALIGNMENT_USABLE_MINIMUM_IOU,
            "maximum_boundary_p95_px": (
                CAMERA_ALIGNMENT_USABLE_MAXIMUM_BOUNDARY_P95_PX
            ),
        },
        "downweighted_target": {
            "minimum_iou": CAMERA_ALIGNMENT_DOWNWEIGHTED_MINIMUM_IOU,
            "maximum_boundary_p95_px": (
                CAMERA_ALIGNMENT_DOWNWEIGHTED_MAXIMUM_BOUNDARY_P95_PX
            ),
        },
        "local_box_refinement_target": {
            "minimum_iou": CAMERA_ALIGNMENT_LOCAL_BOX_MINIMUM_IOU,
            "maximum_boundary_p95_px": (
                CAMERA_ALIGNMENT_LOCAL_BOX_MAXIMUM_BOUNDARY_P95_PX
            ),
            "minimum_target_recall": CAMERA_ALIGNMENT_LOCAL_BOX_MINIMUM_RECALL,
            "minimum_structure_score": (
                CAMERA_ALIGNMENT_LOCAL_BOX_MINIMUM_STRUCTURE_SCORE
            ),
        },
        "minimum_anchor_views": minimum_anchor_views,
        "anchor_view_ids": anchor_view_ids,
        "rejected_view_ids": [
            view_id
            for view_id, row in acceptance.items()
            if row["observation_eligible"] is not True
        ],
        "views": acceptance,
    }


def render_view_arguments(
    *,
    baseline_registry: Mapping[str, Any],
    view_specs_output: Path,
    fallback_views: str,
) -> list[str]:
    """Choose custom continuous-camera specs or the configured pose bank."""

    view_specs = continuous_camera_view_specs(baseline_registry)
    if view_specs is None:
        return ["--views", fallback_views]
    write_object(view_specs_output, view_specs)
    return ["--view-specs", str(view_specs_output)]


__all__ = [
    "continuous_camera_view_specs",
    "render_view_arguments",
    "require_complete_live_camera_alignment",
    "validate_live_camera_registration_provenance",
]
