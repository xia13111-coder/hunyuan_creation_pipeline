#!/usr/bin/env python3
"""Generate CAD Part-ID-localized CropFormer/EntitySeg mask candidates.

CropFormer is an automatic, class-agnostic entity segmenter.  It does not know
which predicted entity corresponds to a renderer-authored CAD Part ID.  This
runner therefore treats EntitySeg only as a boundary candidate generator:

1. the registered CAD Part-ID mask supplies a local crop and shape prior;
2. CropFormer proposes entities inside that crop (and once on the full view);
3. a location-independent CAD-shape contract selects at most one entity;
4. unmatched regions fail closed instead of becoming material evidence.

The external CropFormer repository and checkpoint remain optional runtime
inputs so importing the qwen material pipeline does not require Detectron2.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from .sam3_regions import (
    DEFAULT_MINIMUM_CAD_SHAPE_AREA_AGREEMENT,
    DEFAULT_MINIMUM_CAD_SHAPE_IOU,
    DEFAULT_MINIMUM_CAD_SHAPE_SEED_PIXELS,
    _box_pixels,
    _bounded_shared_camera_alignment,
    _normalized_shape_agreement,
)


SCHEMA_VERSION = "qwen-entityseg-region-result/v1"
DEFAULT_MINIMUM_MODEL_SCORE = 0.30
DEFAULT_MINIMUM_MASK_PIXELS = 16
DEFAULT_MAXIMUM_CANDIDATES_PER_SOURCE = 12
DEFAULT_MAXIMUM_CAD_CENTROID_DISTANCE = 0.15


class EntitySegRegionError(ValueError):
    """Raised when EntitySeg inference or its CAD binding is invalid."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_object(path: Path, label: str) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EntitySegRegionError(f"unable to read {label}: {resolved}") from exc
    if not isinstance(value, dict):
        raise EntitySegRegionError(f"{label} must be a JSON object")
    return value


def _resolve_file(value: Any, *, owner: Path, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise EntitySegRegionError(f"{label} must be a non-empty path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = owner / path
    try:
        return path.resolve(strict=True)
    except OSError as exc:
        raise EntitySegRegionError(f"{label} does not exist: {path}") from exc


def _expanded_crop(
    box: Sequence[int],
    *,
    width: int,
    height: int,
    context_fraction: float,
    minimum_extent: int = 64,
) -> tuple[int, int, int, int]:
    left, top, right, bottom = _box_pixels(box, width=width, height=height)
    box_width = right - left
    box_height = bottom - top
    context = max(4, int(round(context_fraction * max(box_width, box_height))))
    target_width = max(minimum_extent, box_width + 2 * context)
    target_height = max(minimum_extent, box_height + 2 * context)
    center_x = 0.5 * (left + right)
    center_y = 0.5 * (top + bottom)
    crop_left = max(0, int(math.floor(center_x - 0.5 * target_width)))
    crop_top = max(0, int(math.floor(center_y - 0.5 * target_height)))
    crop_right = min(width, crop_left + target_width)
    crop_bottom = min(height, crop_top + target_height)
    crop_left = max(0, crop_right - target_width)
    crop_top = max(0, crop_bottom - target_height)
    return crop_left, crop_top, crop_right, crop_bottom


def _boundary_metrics(image_bgr: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    binary = np.asarray(mask, dtype=bool)
    if binary.shape != image_bgr.shape[:2] or not np.any(binary):
        raise EntitySegRegionError("boundary metric mask is empty or has wrong shape")
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gradient_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gradient_x, gradient_y)
    normalizer = float(np.percentile(gradient, 95.0))
    normalized = np.clip(gradient / max(normalizer, 1e-6), 0.0, 1.0)
    boundary = cv2.morphologyEx(
        binary.astype(np.uint8),
        cv2.MORPH_GRADIENT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    ) > 0
    values = normalized[boundary]
    contours, _hierarchy = cv2.findContours(
        binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    perimeter = float(sum(cv2.arcLength(contour, True) for contour in contours))
    area = int(np.count_nonzero(binary))
    return {
        "boundary_pixel_count": int(values.size),
        "normalized_image_edge_mean": float(values.mean()) if values.size else 0.0,
        "normalized_image_edge_median": (
            float(np.median(values)) if values.size else 0.0
        ),
        "image_edge_support_fraction_025": (
            float(np.mean(values >= 0.25)) if values.size else 0.0
        ),
        "perimeter_pixels": perimeter,
        "mask_pixels": area,
        "perimeter_to_area": perimeter / max(1, area),
    }


def _cad_location_agreement(mask: np.ndarray, seed: np.ndarray) -> dict[str, Any]:
    """Measure candidate displacement relative to the registered CAD seed.

    Shape agreement is intentionally translation invariant, which is useful for
    small camera residuals but ambiguous when several identical components sit
    next to each other.  Normalize centroid displacement by the CAD seed bbox
    diagonal so this guard works at any image resolution and object scale.
    """

    candidate = np.asarray(mask, dtype=bool)
    cad_seed = np.asarray(seed, dtype=bool)
    if (
        candidate.shape != cad_seed.shape
        or not np.any(candidate)
        or not np.any(cad_seed)
    ):
        raise EntitySegRegionError("location agreement masks are empty or incompatible")
    seed_y, seed_x = np.where(cad_seed)
    mask_y, mask_x = np.where(candidate)
    seed_width = int(seed_x.max() - seed_x.min() + 1)
    seed_height = int(seed_y.max() - seed_y.min() + 1)
    seed_diagonal = math.hypot(seed_width, seed_height)
    displacement = math.hypot(
        float(mask_x.mean() - seed_x.mean()),
        float(mask_y.mean() - seed_y.mean()),
    )
    intersection = int(np.count_nonzero(candidate & cad_seed))
    union = int(np.count_nonzero(candidate | cad_seed))
    return {
        "cad_centroid_distance_pixels": displacement,
        "cad_centroid_distance_normalized": displacement / max(seed_diagonal, 1.0),
        "cad_direct_intersection_pixels": intersection,
        "cad_direct_iou": intersection / max(union, 1),
    }


def _setup_predictor(
    *,
    cropformer_root: Path,
    config_path: Path,
    checkpoint_path: Path,
) -> Any:
    root = cropformer_root.expanduser().resolve(strict=True)
    demo_root = root / "demo_cropformer"
    for path in (str(demo_root), str(root)):
        if path not in sys.path:
            sys.path.insert(0, path)
    try:
        from detectron2.config import get_cfg
        from detectron2.projects.deeplab import add_deeplab_config
        from mask2former import add_maskformer2_config
        from predictor import VisualizationDemo
    except Exception as exc:  # pragma: no cover - depends on external runtime
        raise EntitySegRegionError(
            "unable to import the external CropFormer/Detectron2 runtime"
        ) from exc

    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(str(config_path.expanduser().resolve(strict=True)))
    cfg.MODEL.WEIGHTS = str(checkpoint_path.expanduser().resolve(strict=True))
    cfg.freeze()
    return VisualizationDemo(cfg)


def _prediction_candidates(
    predictor: Any,
    image_bgr: np.ndarray,
    *,
    source: str,
    full_shape: tuple[int, int],
    origin_xy: tuple[int, int] = (0, 0),
    minimum_score: float,
) -> list[dict[str, Any]]:
    predictions = predictor.run_on_image(image_bgr)
    instances = predictions.get("instances") if isinstance(predictions, Mapping) else None
    if instances is None or not hasattr(instances, "pred_masks"):
        raise EntitySegRegionError("CropFormer output has no instance masks")
    masks = instances.pred_masks.detach().to("cpu").numpy()
    scores = instances.scores.detach().float().to("cpu").numpy().reshape(-1)
    if masks.ndim != 3 or len(masks) != len(scores):
        raise EntitySegRegionError("CropFormer mask and score arrays are incompatible")
    full_height, full_width = full_shape
    origin_x, origin_y = origin_xy
    output: list[dict[str, Any]] = []
    for index, (raw_mask, raw_score) in enumerate(zip(masks, scores)):
        score = float(raw_score)
        if not math.isfinite(score) or score < minimum_score:
            continue
        local = np.asarray(raw_mask, dtype=bool)
        if int(np.count_nonzero(local)) < DEFAULT_MINIMUM_MASK_PIXELS:
            continue
        full = np.zeros((full_height, full_width), dtype=bool)
        bottom = min(full_height, origin_y + local.shape[0])
        right = min(full_width, origin_x + local.shape[1])
        full[origin_y:bottom, origin_x:right] = local[
            : bottom - origin_y, : right - origin_x
        ]
        output.append(
            {
                "source": source,
                "prediction_index": index,
                "model_score": score,
                "mask": full,
            }
        )
    return output


def _select_candidate(
    candidates: Sequence[Mapping[str, Any]],
    *,
    seed: np.ndarray,
    source_image: np.ndarray,
    minimum_shape_iou: float,
    minimum_area_agreement: float,
    maximum_centroid_distance: float,
    box: Sequence[int] | None = None,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    audited: list[tuple[tuple[float, ...], dict[str, Any], np.ndarray]] = []
    for candidate in candidates:
        mask = np.asarray(candidate["mask"], dtype=bool)
        registered_location = _cad_location_agreement(mask, seed)
        if (
            box is None
            or int(np.count_nonzero(seed))
            < DEFAULT_MINIMUM_CAD_SHAPE_SEED_PIXELS
        ):
            aligned_seed = seed
            alignment = {
                "translation_xy_pixels": [0.0, 0.0],
                "alignment_model": "registered_cad_seed_without_residual_alignment",
                "per_mesh_pose_change_allowed": False,
            }
        else:
            aligned_seed, alignment = _bounded_shared_camera_alignment(
                seed,
                mask,
                box=box,
                width=source_image.shape[1],
                height=source_image.shape[0],
            )
        metrics = _normalized_shape_agreement(mask, aligned_seed)
        location = _cad_location_agreement(mask, aligned_seed)
        reasons: list[str] = []
        if int(metrics["cad_shape_seed_pixels"]) < DEFAULT_MINIMUM_CAD_SHAPE_SEED_PIXELS:
            reasons.append("cad_shape_seed_too_small")
        if float(metrics["cad_shape_iou"]) < minimum_shape_iou:
            reasons.append("cad_shape_iou_below_threshold")
        if float(metrics["cad_shape_area_agreement"]) < minimum_area_agreement:
            reasons.append("cad_shape_area_mismatch")
        if (
            float(registered_location["cad_centroid_distance_normalized"])
            > maximum_centroid_distance
        ):
            reasons.append("cad_centroid_too_far_from_registered_part")
        seed_intersection = int(np.count_nonzero(mask & aligned_seed))
        row = {
            "source": candidate["source"],
            "prediction_index": int(candidate["prediction_index"]),
            "model_score": float(candidate["model_score"]),
            "mask_pixels": int(np.count_nonzero(mask)),
            "cad_seed_intersection_pixels": seed_intersection,
            **metrics,
            **location,
            "registered_cad_centroid_distance_pixels": registered_location[
                "cad_centroid_distance_pixels"
            ],
            "registered_cad_centroid_distance_normalized": registered_location[
                "cad_centroid_distance_normalized"
            ],
            "cad_template_alignment": alignment,
            "accepted": not reasons,
            "reason_codes": reasons,
            "boundary": _boundary_metrics(source_image, mask),
        }
        rank = (
            -float(registered_location["cad_centroid_distance_normalized"]),
            float(metrics["cad_shape_iou"]),
            float(metrics["cad_shape_minimum_precision_recall"]),
            float(metrics["cad_shape_area_agreement"]),
            float(row["boundary"]["normalized_image_edge_mean"]),
            float(candidate["model_score"]),
        )
        audited.append((rank, row, mask))
    audited.sort(key=lambda item: item[0], reverse=True)
    accepted = next((item for item in audited if item[1]["accepted"]), None)
    compact_items = list(audited[:DEFAULT_MAXIMUM_CANDIDATES_PER_SOURCE])
    if accepted is not None and not any(item is accepted for item in compact_items):
        compact_items[-1:] = [accepted]
    compact = [item[1] for item in compact_items]
    if accepted is None:
        return None, compact
    selected = dict(accepted[1])
    selected["mask"] = accepted[2]
    return selected, compact


def run(
    *,
    request_path: Path,
    cropformer_root: Path,
    config_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
    minimum_score: float = DEFAULT_MINIMUM_MODEL_SCORE,
    minimum_shape_iou: float = DEFAULT_MINIMUM_CAD_SHAPE_IOU,
    minimum_area_agreement: float = DEFAULT_MINIMUM_CAD_SHAPE_AREA_AGREEMENT,
    maximum_centroid_distance: float = DEFAULT_MAXIMUM_CAD_CENTROID_DISTANCE,
    local_context_fraction: float = 0.12,
) -> dict[str, Any]:
    request_path = request_path.expanduser().resolve(strict=True)
    request = _read_object(request_path, "EntitySeg request")
    owner = request_path.parent
    output_dir = output_dir.expanduser().resolve()
    masks_dir = output_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    source_by_view: dict[str, tuple[Path, np.ndarray]] = {}
    for index, row in enumerate(request.get("source_views", [])):
        if not isinstance(row, Mapping) or not isinstance(row.get("id"), str):
            raise EntitySegRegionError(f"source_views[{index}] is malformed")
        path = _resolve_file(row.get("image"), owner=owner, label="source image")
        if row.get("image_sha256") != _sha256_file(path):
            raise EntitySegRegionError(f"source image hash mismatch: {path}")
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise EntitySegRegionError(f"unable to decode source image: {path}")
        source_by_view[str(row["id"])] = (path, image)

    predictor = _setup_predictor(
        cropformer_root=cropformer_root,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
    )
    full_candidates: dict[str, list[dict[str, Any]]] = {}
    for view_id, (_path, image) in source_by_view.items():
        full_candidates[view_id] = _prediction_candidates(
            predictor,
            image,
            source="full_image",
            full_shape=image.shape[:2],
            minimum_score=minimum_score,
        )

    records: list[dict[str, Any]] = []
    for index, raw in enumerate(request.get("regions", []), start=1):
        if not isinstance(raw, Mapping):
            raise EntitySegRegionError(f"regions[{index - 1}] is malformed")
        view_id = str(raw.get("view_id"))
        part_id = str(raw.get("group_id"))
        if view_id not in source_by_view:
            raise EntitySegRegionError(f"unknown region view: {view_id}")
        boxes = raw.get("boxes")
        if not isinstance(boxes, list) or len(boxes) != 1:
            raise EntitySegRegionError(f"{view_id}/{part_id} must contain one box")
        seed_doc = raw.get("cad_projection_seed")
        if not isinstance(seed_doc, Mapping):
            raise EntitySegRegionError(f"{view_id}/{part_id} has no CAD seed")
        seed_path = _resolve_file(
            seed_doc.get("path"), owner=owner, label="CAD projection seed"
        )
        if seed_doc.get("sha256") != _sha256_file(seed_path):
            raise EntitySegRegionError(f"CAD seed hash mismatch: {view_id}/{part_id}")
        seed = cv2.imread(str(seed_path), cv2.IMREAD_GRAYSCALE)
        if seed is None:
            raise EntitySegRegionError(f"unable to decode CAD seed: {seed_path}")
        seed = seed >= 128
        image_path, image = source_by_view[view_id]
        if seed.shape != image.shape[:2]:
            raise EntitySegRegionError(f"CAD seed shape mismatch: {view_id}/{part_id}")
        left, top, right, bottom = _expanded_crop(
            boxes[0],
            width=image.shape[1],
            height=image.shape[0],
            context_fraction=local_context_fraction,
        )
        crop = image[top:bottom, left:right]
        local_candidates = _prediction_candidates(
            predictor,
            crop,
            source="cad_local_crop",
            full_shape=image.shape[:2],
            origin_xy=(left, top),
            minimum_score=minimum_score,
        )
        selected, audits = _select_candidate(
            [*local_candidates, *full_candidates[view_id]],
            seed=seed,
            source_image=image,
            minimum_shape_iou=minimum_shape_iou,
            minimum_area_agreement=minimum_area_agreement,
            maximum_centroid_distance=maximum_centroid_distance,
            box=boxes[0],
        )
        mask_doc: dict[str, Any] | None = None
        if selected is not None:
            mask = np.asarray(selected.pop("mask"), dtype=np.uint8) * 255
            mask_path = masks_dir / f"{view_id}__{part_id}.png"
            if not cv2.imwrite(str(mask_path), mask):
                raise EntitySegRegionError(f"unable to write EntitySeg mask: {mask_path}")
            mask_doc = {
                "path": str(mask_path.relative_to(output_dir)),
                "sha256": _sha256_file(mask_path),
            }
        records.append(
            {
                "view_id": view_id,
                "group_id": part_id,
                "source_image": str(image_path),
                "cad_projection_seed": {
                    "path": str(seed_path),
                    "sha256": _sha256_file(seed_path),
                    "mask_pixels": int(np.count_nonzero(seed)),
                },
                "local_crop_xyxy": [left, top, right, bottom],
                "full_candidate_count": len(full_candidates[view_id]),
                "local_candidate_count": len(local_candidates),
                "accepted": selected is not None,
                "selected_candidate": selected,
                "candidate_audits": audits,
                "mask": mask_doc,
            }
        )
        print(
            f"[ENTITYSEG] {index}/{len(request.get('regions', []))} "
            f"{view_id}/{part_id}: {'accepted' if selected else 'rejected'}",
            flush=True,
        )

    accepted = [row for row in records if row["accepted"]]
    source_counts = Counter(
        str(row["selected_candidate"]["source"])
        for row in accepted
        if isinstance(row.get("selected_candidate"), Mapping)
    )
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "request": {
            "path": str(request_path),
            "sha256": _sha256_file(request_path),
            "document_sha256": _canonical_sha256(request),
        },
        "backend": {
            "name": "cropformer_entityseg",
            "repository": str(cropformer_root.expanduser().resolve(strict=True)),
            "config": str(config_path.expanduser().resolve(strict=True)),
            "checkpoint": str(checkpoint_path.expanduser().resolve(strict=True)),
            "checkpoint_sha256": _sha256_file(
                checkpoint_path.expanduser().resolve(strict=True)
            ),
        },
        "policy": {
            "minimum_model_score": minimum_score,
            "minimum_cad_shape_iou": minimum_shape_iou,
            "minimum_cad_shape_area_agreement": minimum_area_agreement,
            "maximum_cad_centroid_distance_normalized": maximum_centroid_distance,
            "local_context_fraction": local_context_fraction,
            "role": "boundary_candidate_only_cad_part_id_remains_identity_authority",
        },
        "records": records,
        "summary": {
            "region_count": len(records),
            "accepted_region_count": len(accepted),
            "rejected_region_count": len(records) - len(accepted),
            "accepted_source_counts": dict(sorted(source_counts.items())),
            "accepted_view_counts": dict(
                sorted(Counter(row["view_id"] for row in accepted).items())
            ),
            "accepted_unique_part_count": len(
                {row["group_id"] for row in accepted}
            ),
        },
    }
    result["integrity"] = {"result_sha256": _canonical_sha256(result)}
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--cropformer-root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--minimum-model-score", type=float, default=DEFAULT_MINIMUM_MODEL_SCORE)
    parser.add_argument("--minimum-shape-iou", type=float, default=DEFAULT_MINIMUM_CAD_SHAPE_IOU)
    parser.add_argument(
        "--minimum-area-agreement",
        type=float,
        default=DEFAULT_MINIMUM_CAD_SHAPE_AREA_AGREEMENT,
    )
    parser.add_argument("--local-context-fraction", type=float, default=0.12)
    parser.add_argument(
        "--maximum-centroid-distance",
        type=float,
        default=DEFAULT_MAXIMUM_CAD_CENTROID_DISTANCE,
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    run(
        request_path=args.request,
        cropformer_root=args.cropformer_root,
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        minimum_score=args.minimum_model_score,
        minimum_shape_iou=args.minimum_shape_iou,
        minimum_area_agreement=args.minimum_area_agreement,
        maximum_centroid_distance=args.maximum_centroid_distance,
        local_context_fraction=args.local_context_fraction,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
