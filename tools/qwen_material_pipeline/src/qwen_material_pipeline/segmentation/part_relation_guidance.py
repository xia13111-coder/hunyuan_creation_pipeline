#!/usr/bin/env python3
"""Build a second-pass Part-ID request located by neighbouring CAD parts.

The first SAM3/EntitySeg pass is used only to discover robust assembly anchors.
For every target, its own first-pass mask is excluded.  A deterministic robust
similarity model and nearby-anchor residual votes infer the target location;
the target CAD silhouette is then projected with that neighbour-derived view
transform.  No USD, camera, or per-mesh transform is changed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

from .part_id_request import _box
from .hybrid_part_masks import (
    _best_model_domain_shape_agreement,
    _model_domain_shape_references,
)


SCHEMA_VERSION = "qwen-part-relation-guidance/v1"
REQUEST_SCHEMA_VERSION = "qwen-sam3-region-request/v1"
MINIMUM_RELATION_ANCHORS = 3
MINIMUM_GUIDANCE_MASK_PIXELS = 6
NEIGHBOR_EXCLUSION_RADIUS_FRACTION = 0.04


class PartRelationGuidanceError(ValueError):
    """Raised when relation guidance inputs violate their sealed contract."""


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
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    path = path.expanduser().resolve(strict=True)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PartRelationGuidanceError(f"unable to read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise PartRelationGuidanceError(f"{label} must be a JSON object")
    return value


def _resolve_document_path(
    document: Mapping[str, Any], *, owner: Path, label: str
) -> Path:
    value = document.get("path")
    expected = document.get("sha256")
    if not isinstance(value, str) or not isinstance(expected, str):
        raise PartRelationGuidanceError(f"{label} binding is malformed")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = owner.parent / path
    path = path.resolve(strict=True)
    if _sha256_file(path) != expected:
        raise PartRelationGuidanceError(f"{label} hash mismatch")
    return path


def _load_mask(document: Mapping[str, Any], *, owner: Path, label: str) -> np.ndarray:
    path = _resolve_document_path(document, owner=owner, label=label)
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.ndim != 2 or not np.any(mask >= 128):
        raise PartRelationGuidanceError(f"{label} is not a non-empty mask")
    return mask >= 128


def _centroid(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask)
    if not len(xs):
        raise PartRelationGuidanceError("cannot locate an empty relation mask")
    return np.asarray([float(xs.mean()), float(ys.mean())], dtype=np.float64)


def _bbox_diagonal(mask: np.ndarray) -> float:
    ys, xs = np.where(mask)
    if not len(xs):
        return 0.0
    return float(
        np.hypot(
            int(xs.max() - xs.min() + 1),
            int(ys.max() - ys.min() + 1),
        )
    )


def _overlap_smaller(first: np.ndarray, second: np.ndarray) -> float:
    intersection = int(np.count_nonzero(first & second))
    return intersection / max(
        min(int(np.count_nonzero(first)), int(np.count_nonzero(second))), 1
    )


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values, kind="stable")
    ordered_values = values[order]
    ordered_weights = weights[order]
    threshold = 0.5 * float(ordered_weights.sum())
    index = int(np.searchsorted(np.cumsum(ordered_weights), threshold, side="left"))
    return float(ordered_values[min(index, len(ordered_values) - 1)])


def _similarity_from_pair(
    source_a: np.ndarray,
    source_b: np.ndarray,
    photo_a: np.ndarray,
    photo_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    source_delta = source_b - source_a
    photo_delta = photo_b - photo_a
    denominator = float(np.dot(source_delta, source_delta))
    if denominator <= 1e-9:
        return None
    real = float(np.dot(photo_delta, source_delta) / denominator)
    imaginary = float(
        (photo_delta[1] * source_delta[0] - photo_delta[0] * source_delta[1])
        / denominator
    )
    scale = float(np.hypot(real, imaginary))
    if not math.isfinite(scale) or not 0.25 <= scale <= 4.0:
        return None
    linear = np.asarray([[real, -imaginary], [imaginary, real]], dtype=np.float64)
    translation = 0.5 * ((photo_a - linear @ source_a) + (photo_b - linear @ source_b))
    return linear, translation


def _refine_similarity(
    source: np.ndarray,
    photo: np.ndarray,
    weights: np.ndarray,
    inliers: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rows: list[list[float]] = []
    values: list[float] = []
    for source_point, photo_point, weight, selected in zip(
        source, photo, weights, inliers
    ):
        if not selected:
            continue
        root_weight = math.sqrt(max(float(weight), 1e-9))
        x_value, y_value = map(float, source_point)
        rows.extend(
            [
                [root_weight * x_value, -root_weight * y_value, root_weight, 0.0],
                [root_weight * y_value, root_weight * x_value, 0.0, root_weight],
            ]
        )
        values.extend(
            [root_weight * float(photo_point[0]), root_weight * float(photo_point[1])]
        )
    real, imaginary, translate_x, translate_y = np.linalg.lstsq(
        np.asarray(rows, dtype=np.float64),
        np.asarray(values, dtype=np.float64),
        rcond=None,
    )[0]
    return (
        np.asarray([[real, -imaginary], [imaginary, real]], dtype=np.float64),
        np.asarray([translate_x, translate_y], dtype=np.float64),
    )


def _fit_robust_similarity(
    observations: list[dict[str, Any]], *, target_part_id: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, list[dict[str, Any]]]:
    available = [
        observation
        for observation in observations
        if observation["part_id"] != target_part_id
    ]
    if len(available) < MINIMUM_RELATION_ANCHORS:
        raise PartRelationGuidanceError("insufficient non-target assembly anchors")

    areas = np.asarray([observation["mask_pixels"] for observation in available])
    qualities = np.asarray([observation["quality"] for observation in available])
    area_floor = (
        float(np.quantile(areas, 0.25)) if len(areas) >= 4 else float(areas.min())
    )
    quality_median = float(np.median(qualities))
    quality_mad = float(np.median(np.abs(qualities - quality_median)))
    quality_floor = max(0.0, quality_median - 1.4826 * quality_mad)
    filtered = [
        observation
        for observation in available
        if observation["mask_pixels"] >= area_floor
        and observation["quality"] >= quality_floor
    ]
    if len(filtered) < MINIMUM_RELATION_ANCHORS:
        filtered = sorted(
            available,
            key=lambda observation: (
                observation["quality"],
                observation["mask_pixels"],
                observation["part_id"],
            ),
            reverse=True,
        )[: max(MINIMUM_RELATION_ANCHORS, int(math.ceil(math.sqrt(len(available)))))]

    source = np.stack([observation["model_centroid_xy"] for observation in filtered])
    photo = np.stack([observation["photo_centroid_xy"] for observation in filtered])
    weights = np.asarray(
        [max(float(observation["quality"]), 1e-3) for observation in filtered],
        dtype=np.float64,
    )
    diagonals = np.asarray(
        [observation["photo_bbox_diagonal"] for observation in filtered],
        dtype=np.float64,
    )
    residual_threshold = max(2.0, 0.15 * float(np.median(diagonals)))
    span = float(np.hypot(*(np.ptp(source, axis=0))))
    minimum_pair_separation = max(1.0, 0.05 * span)

    best: tuple[
        tuple[float, int, float, float], np.ndarray, np.ndarray, np.ndarray
    ] | None = None
    for first in range(len(filtered)):
        for second in range(first + 1, len(filtered)):
            if (
                float(np.linalg.norm(source[second] - source[first]))
                < minimum_pair_separation
            ):
                continue
            model = _similarity_from_pair(
                source[first], source[second], photo[first], photo[second]
            )
            if model is None:
                continue
            linear, translation = model
            residuals = np.linalg.norm(source @ linear.T + translation - photo, axis=1)
            inliers = residuals <= residual_threshold
            if int(np.count_nonzero(inliers)) < MINIMUM_RELATION_ANCHORS:
                continue
            key = (
                float(weights[inliers].sum()),
                int(np.count_nonzero(inliers)),
                -float(np.median(residuals[inliers])),
                -float(np.median(residuals)),
            )
            if best is None or key > best[0]:
                best = (key, linear, translation, inliers)
    if best is None:
        raise PartRelationGuidanceError("assembly anchors have no robust consensus")

    _key, linear, translation, inliers = best
    for _iteration in range(3):
        linear, translation = _refine_similarity(source, photo, weights, inliers)
        residuals = np.linalg.norm(source @ linear.T + translation - photo, axis=1)
        updated = residuals <= residual_threshold
        if np.array_equal(updated, inliers):
            break
        inliers = updated
    if int(np.count_nonzero(inliers)) < MINIMUM_RELATION_ANCHORS:
        raise PartRelationGuidanceError("refined assembly consensus lost its anchors")
    minimum_inliers = max(MINIMUM_RELATION_ANCHORS, int(math.ceil(0.5 * len(filtered))))
    if int(np.count_nonzero(inliers)) < minimum_inliers:
        raise PartRelationGuidanceError("assembly anchor consensus is not a majority")
    return linear, translation, inliers, residual_threshold, filtered


def _infer_target_affine(
    *,
    target_part_id: str,
    target_shape: np.ndarray,
    observations: list[dict[str, Any]],
) -> tuple[np.ndarray, dict[str, Any]]:
    linear, translation, inliers, residual_threshold, filtered = _fit_robust_similarity(
        observations, target_part_id=target_part_id
    )
    source = np.stack([observation["model_centroid_xy"] for observation in filtered])
    photo = np.stack([observation["photo_centroid_xy"] for observation in filtered])
    quality = np.asarray(
        [max(float(observation["quality"]), 1e-3) for observation in filtered]
    )
    target_centroid = _centroid(target_shape)
    global_target = linear @ target_centroid + translation
    residuals = photo - (source @ linear.T + translation)
    source_distances = np.linalg.norm(source - target_centroid, axis=1)
    inlier_indices = np.where(inliers)[0]
    local_count = min(
        len(inlier_indices),
        max(MINIMUM_RELATION_ANCHORS, int(math.ceil(math.sqrt(len(inlier_indices))))),
    )
    nearest = sorted(
        inlier_indices,
        key=lambda index: (
            float(source_distances[index]),
            str(filtered[index]["part_id"]),
        ),
    )[:local_count]
    source_scale = max(float(np.hypot(*np.ptp(source[inliers], axis=0))), 1.0)
    vote_weights = np.asarray(
        [
            quality[index]
            / max(float(source_distances[index]), 0.02 * source_scale, 1.0)
            for index in nearest
        ],
        dtype=np.float64,
    )
    votes = np.stack([global_target + residuals[index] for index in nearest])
    relation_target = np.asarray(
        [
            _weighted_median(votes[:, 0], vote_weights),
            _weighted_median(votes[:, 1], vote_weights),
        ],
        dtype=np.float64,
    )
    vote_errors = np.linalg.norm(votes - relation_target, axis=1)
    uncertainty = _weighted_median(vote_errors, vote_weights)
    if uncertainty > 1.5 * residual_threshold:
        raise PartRelationGuidanceError(
            "nearby anchor votes are spatially inconsistent"
        )
    relation_translation = relation_target - linear @ target_centroid
    affine = np.column_stack((linear, relation_translation)).astype(np.float64)
    scale = float(np.hypot(linear[0, 0], linear[1, 0]))
    rotation = float(np.degrees(np.arctan2(linear[1, 0], linear[0, 0])))
    anchor_audits = []
    for index in nearest:
        anchor_audits.append(
            {
                "part_id": str(filtered[index]["part_id"]),
                "model_delta_xy": [
                    float(value) for value in source[index] - target_centroid
                ],
                "photo_centroid_xy": [float(value) for value in photo[index]],
                "target_vote_xy": [
                    float(value) for value in global_target + residuals[index]
                ],
                "global_model_residual_pixels": float(np.linalg.norm(residuals[index])),
                "quality": float(quality[index]),
            }
        )
    inlier_ids = [
        str(observation["part_id"])
        for observation, selected in zip(filtered, inliers)
        if selected
    ]
    return affine, {
        "method": "leave_one_target_out_multi_anchor_relation_voting",
        "target_part_id": target_part_id,
        "target_first_pass_mask_used": False,
        "target_direct_cad_projection_used_for_location": False,
        "candidate_anchor_count": len(filtered),
        "inlier_anchor_count": len(inlier_ids),
        "inlier_anchor_part_ids": inlier_ids,
        "local_anchor_count": len(nearest),
        "local_anchor_votes": anchor_audits,
        "global_similarity_scale": scale,
        "global_similarity_rotation_degrees": rotation,
        "global_target_centroid_xy": [float(value) for value in global_target],
        "relation_target_centroid_xy": [float(value) for value in relation_target],
        "relation_shift_from_global_xy": [
            float(value) for value in relation_target - global_target
        ],
        "anchor_residual_threshold_pixels": residual_threshold,
        "local_vote_uncertainty_pixels": uncertainty,
        "model_to_photo_affine_2x3": [
            [float(value) for value in row] for row in affine
        ],
        "whole_asset_transform_changed": False,
        "assembly_camera_changed": False,
        "per_mesh_pose_change_allowed": False,
    }


def _record_map(
    document: Mapping[str, Any], label: str
) -> dict[tuple[str, str], Mapping[str, Any]]:
    records: dict[tuple[str, str], Mapping[str, Any]] = {}
    for index, row in enumerate(document.get("records", [])):
        if not isinstance(row, Mapping):
            raise PartRelationGuidanceError(f"{label} record {index} is malformed")
        key = (str(row.get("view_id")), str(row.get("group_id")))
        if key in records:
            raise PartRelationGuidanceError(f"duplicate {label} record: {key}")
        records[key] = row
    return records


def _anchor_observations(
    *,
    sam_document: Mapping[str, Any],
    sam_path: Path,
    entity_document: Mapping[str, Any],
    entity_path: Path,
    model_references: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    sam_records = _record_map(sam_document, "SAM3")
    entity_records = _record_map(entity_document, "EntitySeg")
    observations: dict[str, list[dict[str, Any]]] = {}
    for key, sam_row in sam_records.items():
        if sam_row.get("accepted") is not True or key not in model_references:
            continue
        mask_document = sam_row.get("mask")
        if not isinstance(mask_document, Mapping):
            continue
        sam_mask = _load_mask(
            mask_document,
            owner=sam_path,
            label=f"SAM3 anchor {key}",
        )
        model_shapes = [
            np.asarray(shape, dtype=bool)
            for shape in model_references[key]["visible_shape"]
        ]
        relation_shape = np.asarray(
            model_references[key]["relation_visible_shape"], dtype=bool
        )
        shape_agreement = _best_model_domain_shape_agreement(sam_mask, model_shapes)
        entity_agreement = 1.0
        entity_row = entity_records.get(key)
        if isinstance(entity_row, Mapping) and entity_row.get("accepted") is True:
            entity_document_mask = entity_row.get("mask")
            if isinstance(entity_document_mask, Mapping):
                entity_mask = _load_mask(
                    entity_document_mask,
                    owner=entity_path,
                    label=f"EntitySeg anchor {key}",
                )
                entity_agreement = _overlap_smaller(sam_mask, entity_mask)
        model_score = float(shape_agreement["model_shape_score"])
        quality = math.sqrt(max(model_score, 1e-9) * max(entity_agreement, 1e-9))
        observations.setdefault(key[0], []).append(
            {
                "part_id": key[1],
                # The centroid comes from the complete target Part-ID shape in
                # the model render, never from a component selected through the
                # old reference-photo projection.
                "model_centroid_xy": _centroid(relation_shape),
                "photo_centroid_xy": _centroid(sam_mask),
                "photo_bbox_diagonal": _bbox_diagonal(sam_mask),
                "mask_pixels": int(np.count_nonzero(sam_mask)),
                "model_shape_score": model_score,
                "sam_entity_overlap_smaller": entity_agreement,
                "quality": quality,
            }
        )
    return observations


def build_relation_guided_request(
    *,
    initial_request_path: Path,
    sam_manifest_path: Path,
    entity_manifest_path: Path,
    amodal_manifest_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    initial_request_path = initial_request_path.expanduser().resolve(strict=True)
    sam_manifest_path = sam_manifest_path.expanduser().resolve(strict=True)
    entity_manifest_path = entity_manifest_path.expanduser().resolve(strict=True)
    amodal_manifest_path = amodal_manifest_path.expanduser().resolve(strict=True)
    initial = _read_json(initial_request_path, "initial Part-ID request")
    sam_document = _read_json(sam_manifest_path, "initial SAM3 manifest")
    entity_document = _read_json(entity_manifest_path, "initial EntitySeg manifest")
    if initial.get("schema_version") != REQUEST_SCHEMA_VERSION:
        raise PartRelationGuidanceError("initial request schema is unsupported")
    if sam_document.get("request") != entity_document.get("request"):
        raise PartRelationGuidanceError("initial SAM3 and EntitySeg requests differ")
    request_binding = sam_document.get("request")
    if (
        not isinstance(request_binding, Mapping)
        or request_binding.get("path") != str(initial_request_path)
        or request_binding.get("sha256") != _sha256_file(initial_request_path)
    ):
        raise PartRelationGuidanceError(
            "initial segmentation does not bind the request"
        )
    keys = {
        (str(region.get("view_id")), str(region.get("group_id")))
        for region in initial.get("regions", [])
        if isinstance(region, Mapping)
    }
    model_references, _amodal_document = _model_domain_shape_references(
        manifest_path=amodal_manifest_path,
        expected_keys=keys,
    )
    observations = _anchor_observations(
        sam_document=sam_document,
        sam_path=sam_manifest_path,
        entity_document=entity_document,
        entity_path=entity_manifest_path,
        model_references=model_references,
    )

    output_dir = output_dir.expanduser().resolve()
    visible_dir = output_dir / "visible_masks"
    amodal_dir = output_dir / "amodal_masks"
    visible_dir.mkdir(parents=True, exist_ok=True)
    amodal_dir.mkdir(parents=True, exist_ok=True)
    relation_records: list[dict[str, Any]] = []
    guided_regions: list[dict[str, Any]] = []
    for raw_region in initial.get("regions", []):
        if not isinstance(raw_region, Mapping):
            raise PartRelationGuidanceError("initial region is malformed")
        region = dict(raw_region)
        key = (str(region.get("view_id")), str(region.get("group_id")))
        reference = model_references.get(key)
        original_seed_document = region.get("cad_projection_seed")
        if not isinstance(reference, Mapping) or not isinstance(
            original_seed_document, Mapping
        ):
            raise PartRelationGuidanceError(f"region has no CAD authority: {key}")
        original_seed = _load_mask(
            original_seed_document,
            owner=initial_request_path,
            label=f"initial CAD seed {key}",
        )
        audit: dict[str, Any]
        try:
            target_shape = np.asarray(reference["relation_visible_shape"], dtype=bool)
            affine, audit = _infer_target_affine(
                target_part_id=key[1],
                target_shape=target_shape,
                observations=observations.get(key[0], []),
            )
            guided_visible = (
                cv2.warpAffine(
                    target_shape.astype(np.uint8),
                    affine.astype(np.float32),
                    (original_seed.shape[1], original_seed.shape[0]),
                    flags=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )
                > 0
            )
            complete_shape = np.asarray(reference["complete_shape"], dtype=bool)
            guided_amodal = (
                cv2.warpAffine(
                    complete_shape.astype(np.uint8),
                    affine.astype(np.float32),
                    (original_seed.shape[1], original_seed.shape[0]),
                    flags=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )
                > 0
            )
            assembly_shape = np.asarray(
                reference["relation_assembly_shape"], dtype=bool
            )
            guided_assembly = (
                cv2.warpAffine(
                    assembly_shape.astype(np.uint8),
                    affine.astype(np.float32),
                    (original_seed.shape[1], original_seed.shape[0]),
                    flags=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )
                > 0
            )
            if (
                int(np.count_nonzero(guided_visible)) < MINIMUM_GUIDANCE_MASK_PIXELS
                or int(np.count_nonzero(guided_amodal)) < MINIMUM_GUIDANCE_MASK_PIXELS
            ):
                raise PartRelationGuidanceError("relation warp removed the target mask")
            ys, xs = np.where(guided_visible)
            short_extent = min(
                int(xs.max() - xs.min() + 1),
                int(ys.max() - ys.min() + 1),
            )
            exclusion_radius = max(
                1,
                int(round(NEIGHBOR_EXCLUSION_RADIUS_FRACTION * short_extent)),
            )
            target_exclusion = (
                cv2.dilate(
                    guided_visible.astype(np.uint8),
                    cv2.getStructuringElement(
                        cv2.MORPH_ELLIPSE,
                        (2 * exclusion_radius + 1, 2 * exclusion_radius + 1),
                    ),
                )
                > 0
            )
            guided_neighbors = guided_assembly & ~target_exclusion
        except PartRelationGuidanceError as exc:
            audit = {
                "method": "leave_one_target_out_multi_anchor_relation_voting",
                "accepted": False,
                "fallback": "initial_sealed_cad_request",
                "reason": str(exc),
                "target_part_id": key[1],
                "target_first_pass_mask_used": False,
                "target_direct_cad_projection_used_for_location": True,
                "whole_asset_transform_changed": False,
                "assembly_camera_changed": False,
                "per_mesh_pose_change_allowed": False,
            }
            relation_records.append({"view_id": key[0], "part_id": key[1], **audit})
            guided_regions.append(region)
            continue

        safe_name = f"{key[0]}__{key[1]}.png"
        visible_path = visible_dir / safe_name
        amodal_path = amodal_dir / safe_name
        neighbor_path = output_dir / "neighbor_masks" / safe_name
        neighbor_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(visible_path), guided_visible.astype(np.uint8) * 255):
            raise PartRelationGuidanceError(f"unable to write relation seed: {key}")
        if not cv2.imwrite(str(amodal_path), guided_amodal.astype(np.uint8) * 255):
            raise PartRelationGuidanceError(f"unable to write relation amodal: {key}")
        if not cv2.imwrite(str(neighbor_path), guided_neighbors.astype(np.uint8) * 255):
            raise PartRelationGuidanceError(
                f"unable to write relation neighbor context: {key}"
            )
        normalized_box, box_audit = _box(visible_path)
        original_amodal = region.get("cad_amodal_template")
        if not isinstance(original_amodal, Mapping):
            raise PartRelationGuidanceError(f"region has no amodal template: {key}")
        audit.update(
            {
                "accepted": True,
                "fallback": None,
                "guided_visible_pixels": int(np.count_nonzero(guided_visible)),
                "guided_amodal_pixels": int(np.count_nonzero(guided_amodal)),
                "visible_mask_sha256": _sha256_file(visible_path),
                "amodal_mask_sha256": _sha256_file(amodal_path),
                "neighbor_mask_sha256": _sha256_file(neighbor_path),
                "neighbor_mask_pixels": int(np.count_nonzero(guided_neighbors)),
                "neighbor_exclusion_radius_pixels": exclusion_radius,
            }
        )
        relation_records.append({"view_id": key[0], "part_id": key[1], **audit})
        region.update(
            {
                "prompt": (
                    "one individual rigid component at the location inferred from "
                    "multiple neighboring CAD parts; segment the target matching "
                    "the isolated CAD shape and exclude adjacent components"
                ),
                "boxes": [normalized_box],
                "cad_projection_seed": {
                    "path": str(visible_path),
                    "sha256": _sha256_file(visible_path),
                    **box_audit,
                    "location_authority": (
                        "leave_one_target_out_multi_anchor_relation_voting"
                    ),
                    "target_direct_projection_used": False,
                },
                "cad_amodal_template": {
                    **dict(original_amodal),
                    "path": str(amodal_path),
                    "sha256": _sha256_file(amodal_path),
                    "mask_size": [guided_amodal.shape[1], guided_amodal.shape[0]],
                    "amodal_mask_pixels": int(np.count_nonzero(guided_amodal)),
                    "amodal_bbox_pixels": _box(amodal_path)[1]["projected_bbox_pixels"],
                    "selection_role": ("relation_located_complete_mesh_shape_prior"),
                },
                "cad_assembly_neighbor_context": {
                    "path": str(neighbor_path),
                    "sha256": _sha256_file(neighbor_path),
                    "mask_pixels": int(np.count_nonzero(guided_neighbors)),
                    "location_authority": (
                        "leave_one_target_out_multi_anchor_relation_voting"
                    ),
                },
                "relation_guidance": audit,
            }
        )
        guided_regions.append(region)

    accepted_count = sum(record.get("accepted") is True for record in relation_records)
    unsigned = {
        **initial,
        "regions": sorted(
            guided_regions,
            key=lambda region: (str(region["view_id"]), str(region["group_id"])),
        ),
        "prompt_authority": (
            "leave_one_target_out_neighbor_relation_location_plus_isolated_mesh_shape"
        ),
        "relation_guidance": {
            "schema_version": SCHEMA_VERSION,
            "policy": {
                "anchor_source": "initial_sam3_with_optional_entityseg_consensus",
                "target_first_pass_mask_used_for_own_location": False,
                "target_direct_cad_projection_used_when_relation_accepted": False,
                "view_model": "deterministic_robust_similarity_plus_local_residual_votes",
                "anchor_selection": "automatic_area_quality_and_majority_consensus",
                "local_anchor_count": "ceil_sqrt_inlier_count_minimum_three",
                "failure_policy": "sealed_initial_request_fallback_continue_batch",
                "whole_asset_transform_changed": False,
                "assembly_camera_changed": False,
                "per_mesh_pose_change_allowed": False,
            },
            "inputs": {
                "initial_request": {
                    "path": str(initial_request_path),
                    "sha256": _sha256_file(initial_request_path),
                },
                "initial_sam3_manifest": {
                    "path": str(sam_manifest_path),
                    "sha256": _sha256_file(sam_manifest_path),
                },
                "initial_entityseg_manifest": {
                    "path": str(entity_manifest_path),
                    "sha256": _sha256_file(entity_manifest_path),
                },
                "cad_amodal_templates": {
                    "path": str(amodal_manifest_path),
                    "sha256": _sha256_file(amodal_manifest_path),
                },
            },
            "records": relation_records,
            "summary": {
                "region_count": len(relation_records),
                "relation_guided_count": accepted_count,
                "fallback_count": len(relation_records) - accepted_count,
                "view_anchor_counts": {
                    view_id: len(view_observations)
                    for view_id, view_observations in sorted(observations.items())
                },
            },
        },
    }
    unsigned["relation_guidance"]["integrity"] = {
        "document_sha256": _canonical_sha256(unsigned["relation_guidance"])
    }
    return unsigned


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-request", type=Path, required=True)
    parser.add_argument("--sam-manifest", type=Path, required=True)
    parser.add_argument("--entity-manifest", type=Path, required=True)
    parser.add_argument("--amodal-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    document = build_relation_guided_request(
        initial_request_path=args.initial_request,
        sam_manifest_path=args.sam_manifest,
        entity_manifest_path=args.entity_manifest,
        amodal_manifest_path=args.amodal_manifest,
        output_dir=args.output_dir,
    )
    output_path = args.output_dir.expanduser().resolve() / "request.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output_path),
                **document["relation_guidance"]["summary"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
