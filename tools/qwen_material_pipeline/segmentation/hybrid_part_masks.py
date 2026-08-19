#!/usr/bin/env python3
"""Iteratively refine photo masks with CAD shape and visibility constraints."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

from .entityseg_regions import EntitySegRegionError, _boundary_metrics


SCHEMA_VERSION = "qwen-cad-sam3-entityseg-hybrid/v2"
MAXIMUM_ENTITY_TO_CAD_AREA_RATIO = 1.85
MINIMUM_ENTITY_CAD_DIRECT_IOU = 0.50
MINIMUM_ENTITY_CAD_SHAPE_IOU = 0.55
MAXIMUM_ENTITY_CAD_CENTROID_DISTANCE = 0.15
MINIMUM_ENTITY_EDGE_SUPPORT = 0.70
MINIMUM_ENTITY_EDGE_IMPROVEMENT = 0.03
MINIMUM_SAM_ENTITY_OVERLAP_SMALLER = 0.50
MINIMUM_DIRECT_IOU_WHEN_SAM_DISAGREES = 0.60
MINIMUM_CONNECTED_COMPONENT_PIXELS = 16
MAXIMUM_FINAL_TO_CAD_AREA_RATIO = 1.25
MAXIMUM_CAD_SUPPORT_RADIUS_FRACTION = 0.04
MINIMUM_CAD_SUPPORT_RADIUS_PIXELS = 2
MINIMUM_AMODAL_CANDIDATE_PRECISION = 0.88
MINIMUM_AMODAL_COMPLETION_SHAPE_IOU = 0.62
SHAPE_GUIDED_OPTIMIZATION_ITERATIONS = 5
MAXIMUM_VISIBLE_SUPPORT_RADIUS_FRACTION = 0.025
MAXIMUM_VISIBLE_SUPPORT_RADIUS_PIXELS = 12
MAXIMUM_VISIBLE_CORE_RADIUS_PIXELS = 3
MINIMUM_REFINED_TO_VISIBLE_AREA_RATIO = 0.35
MINIMUM_REFINED_AMODAL_PRECISION = 0.85


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _read_manifest(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.expanduser().resolve(strict=True).read_text("utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EntitySegRegionError(f"unable to read {label}: {path}") from exc
    if not isinstance(value, dict) or not isinstance(value.get("records"), list):
        raise EntitySegRegionError(f"{label} is not a region manifest")
    return value


def _records(
    document: Mapping[str, Any], label: str
) -> dict[tuple[str, str], Mapping[str, Any]]:
    output: dict[tuple[str, str], Mapping[str, Any]] = {}
    for index, row in enumerate(document["records"]):
        if not isinstance(row, Mapping):
            raise EntitySegRegionError(f"{label} record {index} is malformed")
        key = (str(row.get("view_id")), str(row.get("group_id")))
        if key in output:
            raise EntitySegRegionError(f"duplicate {label} region: {key}")
        output[key] = row
    return output


def _resolved_mask_path(root: Path, row: Mapping[str, Any]) -> Path:
    mask = row.get("mask")
    if not isinstance(mask, Mapping) or not isinstance(mask.get("path"), str):
        raise EntitySegRegionError("accepted region has no mask path")
    path = Path(mask["path"])
    if not path.is_absolute():
        path = root / path
    return path.expanduser().resolve(strict=True)


def _sam_selected_shape_candidate(row: Mapping[str, Any]) -> dict[str, Any]:
    for box_audit in row.get("box_audits", []):
        if not isinstance(box_audit, Mapping) or box_audit.get("accepted") is not True:
            continue
        selected_index = box_audit.get("selected_candidate_index")
        for candidate in box_audit.get("candidates", []):
            if (
                isinstance(candidate, Mapping)
                and candidate.get("candidate_index") == selected_index
                and candidate.get("accepted") is True
            ):
                return dict(candidate)
    raise EntitySegRegionError(
        "accepted SAM3 region has no selected CAD-shape candidate"
    )


def _load_mask(path: Path, expected_shape: tuple[int, int]) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.shape != expected_shape:
        raise EntitySegRegionError(f"invalid mask: {path}")
    return mask >= 128


def _sam_aligned_cad_seed(
    seed: np.ndarray,
    sam_row: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Replay the bounded shared-camera residual used to prompt SAM3."""

    for box_audit in sam_row.get("box_audits", []):
        if not isinstance(box_audit, Mapping):
            continue
        refinement = box_audit.get("shape_point_refinement")
        if (
            not isinstance(refinement, Mapping)
            or refinement.get("accepted") is not True
        ):
            continue
        prompt = refinement.get("prompt_audit")
        translation = (
            prompt.get("translation_xy_pixels") if isinstance(prompt, Mapping) else None
        )
        if (
            not isinstance(translation, list)
            or len(translation) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, (int, float))
                for value in translation
            )
        ):
            raise EntitySegRegionError("SAM3 aligned CAD translation is malformed")
        matrix = np.asarray(
            [
                [1.0, 0.0, float(translation[0])],
                [0.0, 1.0, float(translation[1])],
            ],
            dtype=np.float32,
        )
        aligned = (
            cv2.warpAffine(
                seed.astype(np.uint8),
                matrix,
                (seed.shape[1], seed.shape[0]),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            > 0
        )
        return aligned, {
            "source": "sam3_same_view_cad_template_prompt",
            "translation_xy_pixels": [float(translation[0]), float(translation[1])],
            "per_mesh_pose_change_allowed": False,
        }
    return seed, {
        "source": "registered_shared_camera_projection",
        "translation_xy_pixels": [0.0, 0.0],
        "per_mesh_pose_change_allowed": False,
    }


def _entity_aligned_cad_seed(
    seed: np.ndarray,
    entity_row: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Replay the selected EntitySeg candidate's bounded camera residual."""

    selected = entity_row.get("selected_candidate")
    alignment = (
        selected.get("cad_template_alignment")
        if isinstance(selected, Mapping)
        else None
    )
    translation = (
        alignment.get("translation_xy_pixels")
        if isinstance(alignment, Mapping)
        else None
    )
    if translation is None:
        return seed, {
            "source": "registered_shared_camera_projection",
            "translation_xy_pixels": [0.0, 0.0],
            "per_mesh_pose_change_allowed": False,
        }
    if (
        not isinstance(translation, list)
        or len(translation) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in translation
        )
        or alignment.get("per_mesh_pose_change_allowed") is not False
    ):
        raise EntitySegRegionError("EntitySeg aligned CAD translation is malformed")
    matrix = np.asarray(
        [
            [1.0, 0.0, float(translation[0])],
            [0.0, 1.0, float(translation[1])],
        ],
        dtype=np.float32,
    )
    aligned = (
        cv2.warpAffine(
            seed.astype(np.uint8),
            matrix,
            (seed.shape[1], seed.shape[0]),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        > 0
    )
    return aligned, {
        "source": "entityseg_selected_candidate_bounded_camera_residual",
        "translation_xy_pixels": [float(translation[0]), float(translation[1])],
        "per_mesh_pose_change_allowed": False,
    }


def _align_with_audit(mask: np.ndarray, audit: Mapping[str, Any]) -> np.ndarray:
    translation = audit.get("translation_xy_pixels")
    if not isinstance(translation, list) or len(translation) != 2:
        raise EntitySegRegionError("aligned CAD audit has no translation")
    matrix = np.asarray(
        [
            [1.0, 0.0, float(translation[0])],
            [0.0, 1.0, float(translation[1])],
        ],
        dtype=np.float32,
    )
    return (
        cv2.warpAffine(
            mask.astype(np.uint8),
            matrix,
            (mask.shape[1], mask.shape[0]),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        > 0
    )


def _connected_component_count(mask: np.ndarray) -> int:
    count, _labels, statistics, _centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    return int(
        sum(
            int(statistics[index, cv2.CC_STAT_AREA])
            >= MINIMUM_CONNECTED_COMPONENT_PIXELS
            for index in range(1, count)
        )
    )


def _trim_entity_to_cad_support(
    entity_mask: np.ndarray,
    cad_seed: np.ndarray,
    *,
    maximum_final_to_cad_area_ratio: float = MAXIMUM_FINAL_TO_CAD_AREA_RATIO,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Keep EntitySeg detail only inside a bounded CAD Part-ID support band.

    Entity segmentation is class agnostic and may merge touching CAD parts into
    one visual entity.  Intersecting with the exact CAD projection would throw
    away useful photo-boundary detail, so grow the projection by a small,
    resolution-independent radius.  Select the largest radius whose result is
    still within the caller's bounded CAD-area ratio.  EntitySeg keeps the
    1.25x default; SAM3 uses a stricter 1.15x bound after template prompting.
    """

    entity = np.asarray(entity_mask, dtype=bool)
    seed = np.asarray(cad_seed, dtype=bool)
    if entity.shape != seed.shape or not np.any(entity) or not np.any(seed):
        raise EntitySegRegionError("CAD support trim masks are empty or incompatible")
    seed_y, seed_x = np.where(seed)
    diagonal = float(
        np.hypot(
            int(seed_x.max() - seed_x.min() + 1),
            int(seed_y.max() - seed_y.min() + 1),
        )
    )
    maximum_radius = max(
        MINIMUM_CAD_SUPPORT_RADIUS_PIXELS,
        int(round(MAXIMUM_CAD_SUPPORT_RADIUS_FRACTION * diagonal)),
    )
    seed_pixels = int(np.count_nonzero(seed))
    maximum_pixels = int(np.floor(maximum_final_to_cad_area_ratio * seed_pixels))
    selected_radius = 0
    selected = entity & seed
    for radius in range(maximum_radius + 1):
        if radius == 0:
            support = seed
        else:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
            )
            support = cv2.dilate(seed.astype(np.uint8), kernel) > 0
        candidate = entity & support
        if int(np.count_nonzero(candidate)) > maximum_pixels:
            break
        selected_radius = radius
        selected = candidate
    selected_pixels = int(np.count_nonzero(selected))
    if selected_pixels == 0:
        raise EntitySegRegionError(
            "CAD support trim removed the complete EntitySeg mask"
        )
    entity_pixels = int(np.count_nonzero(entity))
    return selected, {
        "maximum_support_radius_pixels": maximum_radius,
        "selected_support_radius_pixels": selected_radius,
        "maximum_final_to_cad_area_ratio": maximum_final_to_cad_area_ratio,
        "untrimmed_entity_pixels": entity_pixels,
        "trimmed_entity_pixels": selected_pixels,
        "retained_entity_fraction": selected_pixels / entity_pixels,
        "final_to_cad_area_ratio": selected_pixels / seed_pixels,
    }


def _ellipse_morphology(mask: np.ndarray, *, radius: int, dilate: bool) -> np.ndarray:
    source = np.asarray(mask, dtype=bool)
    if radius <= 0:
        return source.copy()
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
    )
    operation = cv2.dilate if dilate else cv2.erode
    return operation(source.astype(np.uint8), kernel) > 0


def _mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    union = int(np.count_nonzero(left | right))
    if union == 0:
        return 0.0
    return int(np.count_nonzero(left & right)) / union


def _automatic_refinement_radii(visible_seed: np.ndarray) -> tuple[int, int, int]:
    ys, xs = np.where(visible_seed)
    if not len(xs):
        raise EntitySegRegionError("visible CAD seed is empty")
    diagonal = float(
        np.hypot(
            int(xs.max() - xs.min() + 1),
            int(ys.max() - ys.min() + 1),
        )
    )
    support_radius = max(
        MINIMUM_CAD_SUPPORT_RADIUS_PIXELS,
        min(
            MAXIMUM_VISIBLE_SUPPORT_RADIUS_PIXELS,
            int(round(MAXIMUM_VISIBLE_SUPPORT_RADIUS_FRACTION * diagonal)),
        ),
    )
    core_radius = max(
        1,
        min(
            MAXIMUM_VISIBLE_CORE_RADIUS_PIXELS,
            int(round(0.008 * diagonal)),
        ),
    )
    occlusion_margin = max(1, min(2, int(round(0.006 * diagonal))))
    return support_radius, core_radius, occlusion_margin


def _refinement_metrics(
    *,
    image: np.ndarray,
    mask: np.ndarray,
    visible_seed: np.ndarray,
    amodal_seed: np.ndarray,
    candidate_masks: list[np.ndarray],
) -> dict[str, float | int]:
    mask_pixels = int(np.count_nonzero(mask))
    visible_pixels = int(np.count_nonzero(visible_seed))
    amodal_pixels = int(np.count_nonzero(amodal_seed))
    visible_intersection = int(np.count_nonzero(mask & visible_seed))
    amodal_intersection = int(np.count_nonzero(mask & amodal_seed))
    candidate_agreements = [_mask_iou(mask, candidate) for candidate in candidate_masks]
    edge = _boundary_metrics(image, mask)
    visible_iou = _mask_iou(mask, visible_seed)
    mean_candidate_iou = float(np.mean(candidate_agreements))
    edge_support = float(edge["image_edge_support_fraction_025"])
    # An unweighted geometric mean prevents any one authority (CAD, image
    # edge, or prior model result) from dominating the optimization.
    objective = float(
        np.cbrt(
            max(edge_support, 1e-9)
            * max(visible_iou, 1e-9)
            * max(mean_candidate_iou, 1e-9)
        )
    )
    return {
        "mask_pixels": mask_pixels,
        "visible_seed_pixels": visible_pixels,
        "visible_seed_recall": visible_intersection / max(visible_pixels, 1),
        "visible_seed_precision": visible_intersection / max(mask_pixels, 1),
        "visible_seed_iou": visible_iou,
        "final_to_visible_area_ratio": mask_pixels / max(visible_pixels, 1),
        "amodal_candidate_precision": amodal_intersection / max(mask_pixels, 1),
        "final_to_amodal_area_ratio": mask_pixels / max(amodal_pixels, 1),
        "mean_prior_candidate_iou": mean_candidate_iou,
        "minimum_prior_candidate_iou": min(candidate_agreements),
        "image_edge_support": edge_support,
        "normalized_image_edge_mean": float(edge["normalized_image_edge_mean"]),
        "objective_geometric_mean": objective,
    }


def _iterative_shape_guided_refinement(
    *,
    image: np.ndarray,
    visible_seed: np.ndarray,
    amodal_seed: np.ndarray | None,
    candidate_masks: list[tuple[str, np.ndarray]],
    primary_candidate_source: str,
) -> tuple[np.ndarray, dict[str, Any], dict[str, float | int]]:
    """Optimize one visible instance without choosing one model mask verbatim.

    The isolated mesh is the complete-shape authority, while the assembled CAD
    projection is the current-view visibility/occlusion authority.  SAM3 and
    EntitySeg are treated as previous estimates.  GrabCut then updates the
    boundary inside a scale-derived narrow band for several iterations and the
    best safe iterate is selected by the unweighted agreement of all three
    authorities.
    """

    visible = np.asarray(visible_seed, dtype=bool)
    amodal = (
        np.asarray(amodal_seed, dtype=bool)
        if amodal_seed is not None
        else visible.copy()
    )
    if image.shape[:2] != visible.shape or amodal.shape != visible.shape:
        raise EntitySegRegionError("shape-guided refinement inputs are incompatible")
    if not candidate_masks or not np.any(visible) or not np.any(amodal):
        raise EntitySegRegionError("shape-guided refinement has no usable authority")
    candidate_by_source = {
        source: np.asarray(mask, dtype=bool) for source, mask in candidate_masks
    }
    if len(candidate_by_source) != len(candidate_masks):
        raise EntitySegRegionError("duplicate shape-guided candidate source")
    if primary_candidate_source not in candidate_by_source:
        raise EntitySegRegionError("primary candidate is absent from refinement inputs")
    if any(
        mask.shape != visible.shape or not np.any(mask)
        for mask in candidate_by_source.values()
    ):
        raise EntitySegRegionError(
            "shape-guided candidate mask is empty or incompatible"
        )

    support_radius, core_radius, occlusion_margin = _automatic_refinement_radii(visible)
    visible_support = _ellipse_morphology(visible, radius=support_radius, dilate=True)
    complete_support = _ellipse_morphology(amodal, radius=support_radius, dilate=True)
    optimization_support = visible_support & complete_support
    visible_with_margin = _ellipse_morphology(
        visible, radius=occlusion_margin, dilate=True
    )
    known_occluded = amodal & ~visible_with_margin
    prior_union = np.logical_or.reduce(list(candidate_by_source.values()))
    primary_unbounded = candidate_by_source[primary_candidate_source]
    initial, _initial_visible_bound = _trim_entity_to_cad_support(
        primary_unbounded,
        visible,
        maximum_final_to_cad_area_ratio=MAXIMUM_FINAL_TO_CAD_AREA_RATIO,
    )
    initial &= optimization_support
    initial &= ~known_occluded
    if not np.any(initial):
        raise EntitySegRegionError("visibility constraints removed the primary mask")

    visible_core = _ellipse_morphology(visible, radius=core_radius, dilate=False)
    if not np.any(visible_core):
        distance = cv2.distanceTransform(visible.astype(np.uint8), cv2.DIST_L2, 3)
        maximum = float(distance.max())
        visible_core = visible & (distance >= max(0.5, 0.5 * maximum))
    if not np.any(visible_core):
        raise EntitySegRegionError("visible CAD seed has no stable interior core")

    candidate_values = list(candidate_by_source.values())
    initial_metrics = _refinement_metrics(
        image=image,
        mask=initial,
        visible_seed=visible,
        amodal_seed=amodal,
        candidate_masks=candidate_values,
    )
    best_mask = initial
    best_metrics = initial_metrics
    selected_iteration = 0
    iteration_audits: list[dict[str, Any]] = []
    previous_iterate = initial

    ys, xs = np.where(optimization_support)
    pad = 2
    top = max(0, int(ys.min()) - pad)
    bottom = min(visible.shape[0], int(ys.max()) + pad + 1)
    left = max(0, int(xs.min()) - pad)
    right = min(visible.shape[1], int(xs.max()) + pad + 1)
    crop = np.s_[top:bottom, left:right]
    labels = np.full(visible[crop].shape, cv2.GC_PR_BGD, dtype=np.uint8)
    local_support = optimization_support[crop]
    local_occluded = known_occluded[crop]
    labels[~local_support] = cv2.GC_BGD
    labels[
        ((prior_union | visible) & optimization_support & ~known_occluded)[crop]
    ] = cv2.GC_PR_FGD
    labels[visible_core[crop]] = cv2.GC_FGD
    labels[local_occluded] = cv2.GC_BGD
    if not np.any(labels == cv2.GC_FGD) or not np.any(labels == cv2.GC_BGD):
        raise EntitySegRegionError("shape-guided optimization lacks hard seeds")

    background_model = np.zeros((1, 65), dtype=np.float64)
    foreground_model = np.zeros((1, 65), dtype=np.float64)
    cv2.setRNGSeed(0)
    minimum_visible_recall = max(
        0.50, float(initial_metrics["visible_seed_recall"]) - 0.02
    )
    for iteration in range(1, SHAPE_GUIDED_OPTIMIZATION_ITERATIONS + 1):
        try:
            cv2.grabCut(
                image[crop],
                labels,
                None,
                background_model,
                foreground_model,
                1,
                cv2.GC_INIT_WITH_MASK if iteration == 1 else cv2.GC_EVAL,
            )
        except cv2.error as exc:
            iteration_audits.append(
                {
                    "iteration": iteration,
                    "accepted": False,
                    "reason_codes": ["opencv_grabcut_failed"],
                    "error": str(exc),
                }
            )
            break
        local_mask = (labels == cv2.GC_FGD) | (labels == cv2.GC_PR_FGD)
        refined = np.zeros_like(visible)
        refined[crop] = local_mask
        refined &= optimization_support
        refined &= ~known_occluded
        metrics = _refinement_metrics(
            image=image,
            mask=refined,
            visible_seed=visible,
            amodal_seed=amodal,
            candidate_masks=candidate_values,
        )
        reasons: list[str] = []
        if not np.any(refined):
            reasons.append("refinement_is_empty")
        if not np.all(refined[visible_core]):
            reasons.append("visible_cad_core_was_not_preserved")
        if float(metrics["visible_seed_recall"]) < minimum_visible_recall:
            reasons.append("visible_cad_recall_regressed")
        area_ratio = float(metrics["final_to_visible_area_ratio"])
        if (
            not MINIMUM_REFINED_TO_VISIBLE_AREA_RATIO
            <= area_ratio
            <= MAXIMUM_FINAL_TO_CAD_AREA_RATIO
        ):
            reasons.append("refined_area_outside_visible_cad_bound")
        if (
            float(metrics["amodal_candidate_precision"])
            < MINIMUM_REFINED_AMODAL_PRECISION
        ):
            reasons.append("refinement_extends_outside_complete_mesh")
        accepted = not reasons
        iteration_audits.append(
            {
                "iteration": iteration,
                "accepted": accepted,
                "reason_codes": reasons,
                "metrics": metrics,
                "changed_pixels_from_previous_iteration": int(
                    np.count_nonzero(refined ^ previous_iterate)
                ),
            }
        )
        if accepted and float(metrics["objective_geometric_mean"]) > float(
            best_metrics["objective_geometric_mean"]
        ):
            best_mask = refined
            best_metrics = metrics
            selected_iteration = iteration
        previous_iterate = refined

    initial_pixels = int(np.count_nonzero(initial))
    final_pixels = int(np.count_nonzero(best_mask))
    unbounded_pixels = int(np.count_nonzero(primary_unbounded))
    support_audit: dict[str, float | int] = {
        "maximum_support_radius_pixels": support_radius,
        "selected_support_radius_pixels": support_radius,
        "visible_core_radius_pixels": core_radius,
        "occlusion_margin_pixels": occlusion_margin,
        "maximum_final_to_cad_area_ratio": MAXIMUM_FINAL_TO_CAD_AREA_RATIO,
        "untrimmed_entity_pixels": unbounded_pixels,
        "trimmed_entity_pixels": final_pixels,
        "retained_entity_fraction": final_pixels / max(unbounded_pixels, 1),
        "final_to_cad_area_ratio": final_pixels
        / max(int(np.count_nonzero(visible)), 1),
    }
    audit = {
        "method": "iterative_visible_mesh_edge_optimization",
        "candidate_sources": sorted(candidate_by_source),
        "primary_candidate_source": primary_candidate_source,
        "iteration_budget": SHAPE_GUIDED_OPTIMIZATION_ITERATIONS,
        "selected_iteration": selected_iteration,
        "executed_iteration_count": len(iteration_audits),
        "optimization_converged": bool(
            iteration_audits
            and iteration_audits[-1].get(
                "changed_pixels_from_previous_iteration"
            )
            == 0
        ),
        "automatic_radii": {
            "visible_support_radius_pixels": support_radius,
            "visible_core_radius_pixels": core_radius,
            "occlusion_margin_pixels": occlusion_margin,
        },
        "complete_shape_authority": "isolated_mesh_amodal_projection",
        "current_view_visibility_authority": "whole_assembly_part_id_projection",
        "image_boundary_authority": "current_reference_view_edges",
        "prior_candidate_role": "probable_foreground_initialization_only",
        "known_occluded_pixels": int(np.count_nonzero(known_occluded)),
        "known_occluded_primary_candidate_pixels_removed": int(
            np.count_nonzero(primary_unbounded & known_occluded)
        ),
        "initial_metrics": initial_metrics,
        "final_metrics": best_metrics,
        "final_changed_pixels_from_initial": int(np.count_nonzero(best_mask ^ initial)),
        "iterations": iteration_audits,
    }
    return best_mask, audit, support_audit


def _entity_rejection_reasons(
    metrics: Mapping[str, float | int], *, sam_accepted: bool
) -> list[str]:
    reasons: list[str] = []
    if int(metrics["connected_component_count"]) != 1:
        reasons.append("entity_mask_is_not_one_connected_component")
    if float(metrics["entity_to_cad_area_ratio"]) > MAXIMUM_ENTITY_TO_CAD_AREA_RATIO:
        reasons.append("entity_mask_area_exceeds_cad_bound")
    if float(metrics["entity_cad_direct_iou"]) < MINIMUM_ENTITY_CAD_DIRECT_IOU:
        reasons.append("entity_direct_cad_iou_below_threshold")
    if float(metrics["entity_cad_shape_iou"]) < MINIMUM_ENTITY_CAD_SHAPE_IOU:
        reasons.append("entity_cad_shape_iou_below_threshold")
    if "entity_amodal_candidate_precision" in metrics and (
        float(metrics["entity_amodal_candidate_precision"])
        < MINIMUM_AMODAL_CANDIDATE_PRECISION
    ):
        reasons.append("entity_extends_outside_complete_mesh_shape")
    if "entity_amodal_shape_iou" in metrics and (
        float(metrics["entity_amodal_shape_iou"]) < MINIMUM_AMODAL_COMPLETION_SHAPE_IOU
    ):
        reasons.append("entity_occlusion_aware_amodal_shape_mismatch")
    if (
        float(metrics["entity_cad_centroid_distance"])
        > MAXIMUM_ENTITY_CAD_CENTROID_DISTANCE
    ):
        reasons.append("entity_centroid_too_far_from_cad_part")
    if float(metrics["entity_edge_support"]) < MINIMUM_ENTITY_EDGE_SUPPORT:
        reasons.append("entity_boundary_has_insufficient_image_edge_support")
    if sam_accepted:
        if float(metrics["entity_edge_improvement"]) < MINIMUM_ENTITY_EDGE_IMPROVEMENT:
            reasons.append("entity_boundary_does_not_improve_over_sam3")
        if (
            float(metrics["sam_entity_overlap_smaller"])
            < MINIMUM_SAM_ENTITY_OVERLAP_SMALLER
            and float(metrics["entity_cad_direct_iou"])
            < MINIMUM_DIRECT_IOU_WHEN_SAM_DISAGREES
        ):
            reasons.append("entity_disagrees_with_both_sam3_and_cad_location")
    return reasons


def _entity_metrics(
    *,
    image: np.ndarray,
    seed: np.ndarray,
    entity_mask: np.ndarray,
    entity_row: Mapping[str, Any],
    sam_mask: np.ndarray | None,
) -> dict[str, float | int]:
    selected = entity_row.get("selected_candidate")
    if not isinstance(selected, Mapping):
        raise EntitySegRegionError(
            "accepted EntitySeg record has no selected candidate"
        )
    seed_pixels = int(np.count_nonzero(seed))
    entity_boundary = _boundary_metrics(image, entity_mask)
    output: dict[str, float | int] = {
        "connected_component_count": _connected_component_count(entity_mask),
        "entity_mask_pixels": int(np.count_nonzero(entity_mask)),
        "cad_seed_pixels": seed_pixels,
        "entity_to_cad_area_ratio": int(np.count_nonzero(entity_mask))
        / max(seed_pixels, 1),
        "entity_cad_direct_iou": float(selected.get("cad_direct_iou", -1.0)),
        "entity_cad_shape_iou": float(selected.get("cad_shape_iou", -1.0)),
        "entity_cad_centroid_distance": float(
            selected.get("cad_centroid_distance_normalized", float("inf"))
        ),
        "entity_edge_support": float(
            entity_boundary["image_edge_support_fraction_025"]
        ),
    }
    if "cad_amodal_candidate_precision" in selected:
        output.update(
            {
                "entity_amodal_candidate_precision": float(
                    selected["cad_amodal_candidate_precision"]
                ),
                "entity_amodal_completion_iou": float(
                    selected["cad_amodal_completion_iou"]
                ),
                "entity_amodal_shape_iou": float(selected["cad_amodal_shape_iou"]),
                "entity_to_amodal_area_ratio": float(
                    selected["candidate_to_cad_amodal_area_ratio"]
                ),
            }
        )
    if sam_mask is not None:
        intersection = int(np.count_nonzero(sam_mask & entity_mask))
        sam_pixels = int(np.count_nonzero(sam_mask))
        entity_pixels = int(np.count_nonzero(entity_mask))
        sam_boundary = _boundary_metrics(image, sam_mask)
        output.update(
            {
                "sam_mask_pixels": sam_pixels,
                "sam_entity_iou": intersection
                / max(int(np.count_nonzero(sam_mask | entity_mask)), 1),
                "sam_entity_overlap_smaller": intersection
                / max(min(sam_pixels, entity_pixels), 1),
                "sam_edge_support": float(
                    sam_boundary["image_edge_support_fraction_025"]
                ),
                "entity_edge_improvement": float(
                    entity_boundary["image_edge_support_fraction_025"]
                )
                - float(sam_boundary["image_edge_support_fraction_025"]),
            }
        )
    return output


def build_hybrid_masks(
    *,
    sam_manifest_path: Path,
    entity_manifest_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    sam_manifest_path = sam_manifest_path.expanduser().resolve(strict=True)
    entity_manifest_path = entity_manifest_path.expanduser().resolve(strict=True)
    sam_document = _read_manifest(sam_manifest_path, "SAM3 manifest")
    entity_document = _read_manifest(entity_manifest_path, "EntitySeg manifest")
    if sam_document.get("request") != entity_document.get("request"):
        raise EntitySegRegionError(
            "SAM3 and EntitySeg manifests bind different requests"
        )
    sam_records = _records(sam_document, "SAM3")
    entity_records = _records(entity_document, "EntitySeg")
    if set(sam_records) != set(entity_records):
        raise EntitySegRegionError("SAM3 and EntitySeg region sets differ")
    sam_root = sam_manifest_path.parent
    entity_root = entity_manifest_path.parent
    output_dir = output_dir.expanduser().resolve()
    masks_dir = output_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    decision_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    for key in sorted(sam_records):
        sam_row = sam_records[key]
        entity_row = entity_records[key]
        sam_shared = sam_row.get("view_shared_alignment")
        entity_shared = entity_row.get("view_shared_alignment")
        if (
            not isinstance(sam_shared, Mapping)
            or not isinstance(entity_shared, Mapping)
            or dict(sam_shared) != dict(entity_shared)
            or sam_shared.get("part_specific_translation_allowed") is not False
        ):
            raise EntitySegRegionError(
                f"SAM3 and EntitySeg do not share one whole-workpiece alignment: {key}"
            )
        authority = entity_row if entity_row.get("source_image") else sam_row
        source_path = (
            Path(str(authority["source_image"])).expanduser().resolve(strict=True)
        )
        source_sha256 = _sha256_file(source_path)
        if (
            sam_row.get("source_image_sha256") != source_sha256
            or entity_row.get("source_image_sha256") != source_sha256
        ):
            raise EntitySegRegionError(f"source image hash mismatch: {key}")
        image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        if image is None:
            raise EntitySegRegionError(f"unable to decode source image: {source_path}")
        seed_doc = entity_row.get("cad_projection_seed") or sam_row.get(
            "cad_projection_seed"
        )
        if not isinstance(seed_doc, Mapping):
            raise EntitySegRegionError(f"missing CAD seed: {key}")
        seed_path = Path(str(seed_doc["path"])).expanduser().resolve(strict=True)
        if seed_doc.get("sha256") != _sha256_file(seed_path):
            raise EntitySegRegionError(f"CAD seed hash mismatch: {key}")
        seed = _load_mask(seed_path, image.shape[:2])
        sam_aligned_seed, sam_alignment_audit = _sam_aligned_cad_seed(seed, sam_row)
        entity_aligned_seed, entity_alignment_audit = _entity_aligned_cad_seed(
            seed, entity_row
        )
        expected_translation = [
            float(value) for value in sam_shared["translation_xy_pixels"]
        ]
        if (
            sam_alignment_audit.get("translation_xy_pixels") != expected_translation
            or entity_alignment_audit.get("translation_xy_pixels")
            != expected_translation
        ):
            raise EntitySegRegionError(
                f"candidate alignment differs from the view-shared alignment: {key}"
            )
        sam_amodal_doc = sam_row.get("cad_amodal_template")
        entity_amodal_doc = entity_row.get("cad_amodal_template")
        amodal_doc = (
            sam_amodal_doc if isinstance(sam_amodal_doc, Mapping) else entity_amodal_doc
        )
        amodal: np.ndarray | None = None
        if isinstance(amodal_doc, Mapping):
            amodal_path = (
                Path(str(amodal_doc["path"])).expanduser().resolve(strict=True)
            )
            expected_amodal_hash = amodal_doc.get("sha256")
            if (
                not isinstance(expected_amodal_hash, str)
                or _sha256_file(amodal_path) != expected_amodal_hash
            ):
                raise EntitySegRegionError(f"CAD amodal hash mismatch: {key}")
            if isinstance(entity_amodal_doc, Mapping):
                entity_path = (
                    Path(str(entity_amodal_doc["path"]))
                    .expanduser()
                    .resolve(strict=True)
                )
                if (
                    entity_path != amodal_path
                    or entity_amodal_doc.get("sha256") != expected_amodal_hash
                ):
                    raise EntitySegRegionError(
                        f"SAM3 and EntitySeg bind different CAD amodal templates: {key}"
                    )
            contract = amodal_doc.get("projection_contract")
            if (
                not isinstance(contract, Mapping)
                or contract.get("whole_asset_camera_unchanged") is not True
                or contract.get("whole_asset_transform_unchanged") is not True
                or contract.get("per_mesh_pose_change_allowed") is not False
            ):
                raise EntitySegRegionError(f"CAD amodal contract mismatch: {key}")
            amodal = _load_mask(amodal_path, image.shape[:2])
        sam_aligned_amodal = (
            _align_with_audit(amodal, sam_alignment_audit)
            if amodal is not None
            else None
        )
        entity_aligned_amodal = (
            _align_with_audit(amodal, entity_alignment_audit)
            if amodal is not None
            else None
        )
        sam_accepted = sam_row.get("accepted") is True
        entity_accepted = entity_row.get("accepted") is True
        sam_mask = (
            _load_mask(_resolved_mask_path(sam_root, sam_row), image.shape[:2])
            if sam_accepted
            else None
        )
        entity_mask = (
            _load_mask(_resolved_mask_path(entity_root, entity_row), image.shape[:2])
            if entity_accepted
            else None
        )
        metrics: dict[str, float | int] | None = None
        entity_reasons: list[str] = []
        if entity_mask is not None:
            metrics = _entity_metrics(
                image=image,
                seed=entity_aligned_seed,
                entity_mask=entity_mask,
                entity_row=entity_row,
                sam_mask=sam_mask,
            )
            entity_reasons = _entity_rejection_reasons(
                metrics, sam_accepted=sam_accepted
            )

        final_mask: np.ndarray | None = None
        cad_support_trim: dict[str, float | int] | None = None
        iterative_refinement: dict[str, Any] | None = None
        candidate_masks: list[tuple[str, np.ndarray]] = []
        primary_candidate_source: str | None = None
        if entity_mask is not None and not entity_reasons:
            candidate_masks.append(("entityseg", entity_mask))
            primary_candidate_source = "entityseg"
        if sam_mask is not None:
            candidate_masks.append(("sam3", sam_mask))
            if primary_candidate_source is None:
                primary_candidate_source = "sam3"
        aligned_seed_audit = (
            entity_alignment_audit
            if primary_candidate_source == "entityseg"
            else sam_alignment_audit
        )
        if primary_candidate_source is not None:
            aligned_visible = (
                entity_aligned_seed
                if primary_candidate_source == "entityseg"
                else sam_aligned_seed
            )
            aligned_amodal = (
                entity_aligned_amodal
                if primary_candidate_source == "entityseg"
                else sam_aligned_amodal
            )
            try:
                (
                    final_mask,
                    iterative_refinement,
                    cad_support_trim,
                ) = _iterative_shape_guided_refinement(
                    image=image,
                    visible_seed=aligned_visible,
                    amodal_seed=aligned_amodal,
                    candidate_masks=candidate_masks,
                    primary_candidate_source=primary_candidate_source,
                )
            except EntitySegRegionError as exc:
                final_mask = None
                selected_source = "none"
                decision = "iterative_refinement_rejected"
                iterative_refinement = {
                    "method": "iterative_visible_mesh_edge_optimization",
                    "accepted": False,
                    "reason_codes": ["shape_guided_refinement_failed"],
                    "error": str(exc),
                }
            else:
                selected_source = "shape_guided_iterative"
                candidate_sources = {source for source, _mask in candidate_masks}
                if candidate_sources == {"sam3", "entityseg"}:
                    decision = "iterative_refinement_from_sam3_entityseg"
                elif candidate_sources == {"entityseg"}:
                    decision = "iterative_refinement_from_entityseg"
                else:
                    decision = "iterative_refinement_from_sam3"
        else:
            selected_source = "none"
            decision = "no_safe_candidate"

        mask_document: dict[str, Any] | None = None
        shape_candidate: dict[str, Any] | None = None
        if primary_candidate_source == "entityseg":
            selected_entity = entity_row.get("selected_candidate")
            if not isinstance(selected_entity, Mapping):
                raise EntitySegRegionError(
                    f"accepted EntitySeg region has no shape candidate: {key}"
                )
            shape_candidate = dict(selected_entity)
        elif primary_candidate_source == "sam3":
            shape_candidate = _sam_selected_shape_candidate(sam_row)
        if final_mask is not None:
            mask_path = masks_dir / f"{key[0]}__{key[1]}.png"
            if not cv2.imwrite(str(mask_path), final_mask.astype(np.uint8) * 255):
                raise EntitySegRegionError(f"unable to write hybrid mask: {mask_path}")
            mask_document = {
                "path": str(mask_path.relative_to(output_dir)),
                "sha256": _sha256_file(mask_path),
                "mask_pixels": int(np.count_nonzero(final_mask)),
            }
        decision_counts[decision] += 1
        source_counts[selected_source] += 1
        records.append(
            {
                "view_id": key[0],
                "group_id": key[1],
                "source_image": str(source_path),
                "source_image_sha256": source_sha256,
                "view_shared_alignment": dict(sam_shared),
                "cad_projection_seed": dict(seed_doc),
                "accepted": final_mask is not None,
                "selected_source": selected_source,
                "primary_candidate_source": primary_candidate_source,
                "candidate_sources": [source for source, _mask in candidate_masks],
                "decision": decision,
                "entityseg_candidate_accepted": entity_accepted,
                "entityseg_fusion_rejection_reasons": entity_reasons,
                "fusion_metrics": metrics,
                "cad_support_trim": cad_support_trim,
                "iterative_refinement": iterative_refinement,
                "aligned_cad_template": aligned_seed_audit,
                "shape_candidate": shape_candidate,
                "fusion_audit": {
                    "selected_source": selected_source,
                    "decision": decision,
                    "fusion_metrics": metrics,
                    "cad_support_trim": cad_support_trim,
                    "iterative_refinement": iterative_refinement,
                },
                "cad_amodal_template": (
                    {
                        **dict(amodal_doc),
                        "aligned_with_visible_template_translation": True,
                    }
                    if isinstance(amodal_doc, Mapping)
                    else None
                ),
                "mask": mask_document,
            }
        )

    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "inputs": {
            "sam3_manifest": {
                "path": str(sam_manifest_path),
                "sha256": _sha256_file(sam_manifest_path),
                "document_sha256": _canonical_sha256(sam_document),
            },
            "entityseg_manifest": {
                "path": str(entity_manifest_path),
                "sha256": _sha256_file(entity_manifest_path),
                "document_sha256": _canonical_sha256(entity_document),
            },
        },
        "request": dict(sam_document.get("request", {})),
        "policy": {
            "identity_authority": "registered_cad_part_id_plus_sam3_instance",
            "sam3_role": "probable_foreground_initialization_only",
            "entityseg_role": "probable_foreground_initialization_only",
            "maximum_entity_to_cad_area_ratio": MAXIMUM_ENTITY_TO_CAD_AREA_RATIO,
            "minimum_entity_cad_direct_iou": MINIMUM_ENTITY_CAD_DIRECT_IOU,
            "minimum_entity_cad_shape_iou": MINIMUM_ENTITY_CAD_SHAPE_IOU,
            "maximum_entity_cad_centroid_distance": MAXIMUM_ENTITY_CAD_CENTROID_DISTANCE,
            "minimum_entity_edge_support": MINIMUM_ENTITY_EDGE_SUPPORT,
            "minimum_entity_edge_improvement": MINIMUM_ENTITY_EDGE_IMPROVEMENT,
            "minimum_sam_entity_overlap_smaller": MINIMUM_SAM_ENTITY_OVERLAP_SMALLER,
            "minimum_direct_iou_when_sam_disagrees": MINIMUM_DIRECT_IOU_WHEN_SAM_DISAGREES,
            "maximum_final_to_cad_area_ratio": MAXIMUM_FINAL_TO_CAD_AREA_RATIO,
            "maximum_cad_support_radius_fraction": MAXIMUM_CAD_SUPPORT_RADIUS_FRACTION,
            "minimum_cad_support_radius_pixels": MINIMUM_CAD_SUPPORT_RADIUS_PIXELS,
            "minimum_amodal_candidate_precision": MINIMUM_AMODAL_CANDIDATE_PRECISION,
            "minimum_amodal_completion_shape_iou": (
                MINIMUM_AMODAL_COMPLETION_SHAPE_IOU
            ),
            "shape_authority": "isolated_mesh_amodal_projection",
            "visibility_authority": "whole_assembly_part_id_projection",
            "final_boundary_method": "iterative_visible_mesh_edge_optimization",
            "shape_guided_optimization_iterations": SHAPE_GUIDED_OPTIMIZATION_ITERATIONS,
            "candidate_selection_policy": (
                "joint_iterative_optimization_not_single_model_arbitration"
            ),
            "optimization_objective": (
                "unweighted_geometric_mean_of_current_view_edges_visible_cad_"
                "agreement_and_prior_candidate_agreement"
            ),
            "known_occlusion_policy": (
                "amodal_minus_current_view_visible_projection_is_background"
            ),
            "alignment_model": "one_whole_workpiece_translation_per_view",
            "per_mesh_pose_change_allowed": False,
            "part_specific_translation_allowed": False,
        },
        "records": records,
        "summary": {
            "region_count": len(records),
            "accepted_region_count": sum(row["accepted"] for row in records),
            "selected_source_counts": dict(sorted(source_counts.items())),
            "decision_counts": dict(sorted(decision_counts.items())),
            "selected_unique_part_count": len(
                {row["group_id"] for row in records if row["accepted"]}
            ),
        },
    }
    result["integrity"] = {"result_sha256": _canonical_sha256(result)}
    (output_dir / "manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sam-manifest", required=True, type=Path)
    parser.add_argument("--entity-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    result = build_hybrid_masks(
        sam_manifest_path=args.sam_manifest,
        entity_manifest_path=args.entity_manifest,
        output_dir=args.output_dir,
    )
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
