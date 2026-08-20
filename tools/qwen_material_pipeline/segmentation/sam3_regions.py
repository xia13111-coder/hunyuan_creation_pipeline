#!/usr/bin/env python3
"""Run local SAM3 on Qwen-authored material regions with fail-closed gates.

The command is intentionally standalone.  The staged material process can run
Qwen in a dedicated environment, release its GPU allocation, and invoke this
file with the main ``hunyuan_sam3d`` Python without importing either runtime
into the other.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
from pathlib import Path
import random
import subprocess
import sys
from typing import Any, Mapping, Sequence

# This file is deliberately launched by path in a sanitized subprocess so the
# Qwen runtime cannot leak its PYTHONPATH into the SAM3 runtime.  Direct script
# execution puts only this ``segmentation`` directory on sys.path, whereas the
# shared human-foreground replay code lives in the sibling package.  Add the
# repository's ``tools`` directory explicitly without restoring any inherited
# environment paths.
if __package__ in {None, ""}:
    tools_dir = str(Path(__file__).resolve().parents[2])
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)

from qwen_material_pipeline.segmentation.human_foreground import (
    ANNOTATION_SCHEMA_VERSION as HUMAN_ANNOTATION_SCHEMA_VERSION,
    CONFIRMED_MASK_BOUNDED_MINIMUM_PRECISION,
    CONFIRMED_MASK_BOUNDED_MINIMUM_RECALL,
    CONFIRMED_MASK_STRICT_MINIMUM_IOU,
    CONFIRMED_MASK_SYMMETRIC_MINIMUM_IOU,
    grid_to_pixel,
    replay_ordered_click_set,
    select_point_candidate,
    sha256_file as _human_sha256_file,
    validate_click_sets,
    validate_ordered_click_sets,
)


SCHEMA_VERSION = "qwen-sam3-region-request/v1"
POINT_SCHEMA_VERSION = "qwen-sam3-region-request/v2"
ORDERED_POINT_SCHEMA_VERSION = "qwen-sam3-region-request/v3"
RESULT_SCHEMA_VERSION = "qwen-sam3-region-result/v1"
DEFAULT_MINIMUM_MODEL_SCORE = 0.45
DEFAULT_MINIMUM_PROMPT_OVERLAP = 0.25
DEFAULT_MAXIMUM_IMAGE_FRACTION = 0.80
DEFAULT_MINIMUM_MASK_PIXELS = 32
DEFAULT_MINIMUM_CAD_SHAPE_SEED_PIXELS = 16
DEFAULT_MINIMUM_CAD_SHAPE_IOU = 0.60
DEFAULT_MINIMUM_CAD_SHAPE_AREA_AGREEMENT = 0.50
DEFAULT_MINIMUM_ALIGNED_CAD_OVERLAP_SMALLER = 0.55
DEFAULT_MAXIMUM_ALIGNED_CAD_AREA_RATIO = 1.60
DEFAULT_MINIMUM_AMODAL_CANDIDATE_PRECISION = 0.88
DEFAULT_MINIMUM_AMODAL_COMPLETION_SHAPE_IOU = 0.62
DEFAULT_MAXIMUM_CANDIDATE_TO_AMODAL_AREA_RATIO = 1.25
DEFAULT_MINIMUM_VISIBLE_CAD_SEED_RECALL = 0.68
DEFAULT_MAXIMUM_FINAL_SAM_TO_CAD_AREA_RATIO = 1.15
DEFAULT_MAXIMUM_CAD_SUPPORT_RADIUS_FRACTION = 0.04
DEFAULT_MAXIMUM_NEIGHBOR_NEGATIVE_TARGET_PIXELS = 4096
DEFAULT_DENSE_CAD_MASK_LOGIT_MAGNITUDE = 4.0
DEFAULT_MAXIMUM_VIEW_SHARED_TRANSLATION_PIXELS = 12
DEFAULT_MAXIMUM_OTHER_PART_OVERLAP_FRACTION = 0.12
CAD_SHAPE_NORMALIZATION_SIZE = 96
CROSS_GROUP_NEAR_DUPLICATE_IOU = 0.90
CROSS_GROUP_ARBITRATION_SCHEMA_VERSION = (
    "sam3-cross-group-near-duplicate-arbitration/v1"
)
PROGRESS_PREFIX = "@@ASSET_PROGRESS "
PROGRESS_SCHEMA_VERSION = "asset-pipeline-progress/v1"
DEFAULT_INFERENCE_SEED = 0


class Sam3RegionError(ValueError):
    """Raised when the SAM3 request or output violates the frozen contract."""


def _seed_inference(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise Sam3RegionError("inference seed must be a non-negative integer")
    random.seed(seed)
    import numpy as np
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _emit_progress(
    *,
    stage: str,
    state: str,
    detail: str,
    current: int | None = None,
    total: int | None = None,
    unit: str | None = None,
) -> None:
    event = {
        "schema_version": PROGRESS_SCHEMA_VERSION,
        "scope": "sam3_regions",
        "stage": stage,
        "state": state,
        "current": current,
        "total": total,
        "unit": unit,
        "detail": detail,
    }
    print(
        PROGRESS_PREFIX
        + json.dumps(
            event,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ),
        file=sys.stderr,
        flush=True,
    )


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
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Sam3RegionError(f"unable to read {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise Sam3RegionError(f"{label} must be a JSON object")
    return value


def _resolve_file(value: Any, *, base: Path, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise Sam3RegionError(f"{label} must be a non-empty path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise Sam3RegionError(f"{label} does not exist: {path}") from exc
    if not resolved.is_file():
        raise Sam3RegionError(f"{label} is not a file: {resolved}")
    return resolved


def _unit_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Sam3RegionError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise Sam3RegionError(f"{label} must be between zero and one")
    return result


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise Sam3RegionError(f"{label} must be a positive integer")
    return value


def _validated_box(value: Any, label: str) -> list[int]:
    if (
        not isinstance(value, list)
        or len(value) != 4
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise Sam3RegionError(f"{label} must be [x0,y0,x1,y1] integer coordinates")
    x0, y0, x1, y1 = value
    if not (0 <= x0 < x1 <= 1000 and 0 <= y0 < y1 <= 1000):
        raise Sam3RegionError(f"{label} must be ordered within the 0..1000 grid")
    return [x0, y0, x1, y1]


def _load_request(path: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    request = _read_json(path, "SAM3 region request")
    request_schema = request.get("schema_version")
    if request_schema not in {
        SCHEMA_VERSION,
        POINT_SCHEMA_VERSION,
        ORDERED_POINT_SCHEMA_VERSION,
    }:
        raise Sam3RegionError(f"unsupported SAM3 request schema: {request_schema!r}")
    raw_views = request.get("source_views")
    raw_regions = request.get("regions")
    if not isinstance(raw_views, list) or not raw_views:
        raise Sam3RegionError("source_views must be a non-empty array")
    if not isinstance(raw_regions, list) or not raw_regions:
        raise Sam3RegionError("regions must be a non-empty array")
    source_paths: dict[str, Path] = {}
    for index, raw in enumerate(raw_views):
        if not isinstance(raw, Mapping):
            raise Sam3RegionError(f"source_views[{index}] must be an object")
        view_id = raw.get("id")
        if not isinstance(view_id, str) or not view_id or view_id in source_paths:
            raise Sam3RegionError("source view IDs must be unique non-empty strings")
        source_paths[view_id] = _resolve_file(
            raw.get("image"),
            base=path.parent,
            label=f"source_views[{index}].image",
        )
        foreground = raw.get("whole_workpiece_foreground")
        if foreground is not None:
            foreground_path = _resolve_file(
                foreground,
                base=path.parent,
                label=f"source_views[{index}].whole_workpiece_foreground",
            )
            expected_foreground_digest = raw.get(
                "whole_workpiece_foreground_sha256"
            )
            if (
                not isinstance(expected_foreground_digest, str)
                or expected_foreground_digest != _sha256_file(foreground_path)
            ):
                raise Sam3RegionError(
                    f"source_views[{index}] foreground hash mismatch"
                )
            raw["whole_workpiece_foreground"] = str(foreground_path)
    identities: set[tuple[str, str]] = set()
    for index, raw in enumerate(raw_regions):
        if not isinstance(raw, dict):
            raise Sam3RegionError(f"regions[{index}] must be an object")
        view_id = raw.get("view_id")
        group_id = raw.get("group_id")
        if view_id not in source_paths:
            raise Sam3RegionError(f"regions[{index}] references unknown view_id")
        if not isinstance(group_id, str) or not group_id:
            raise Sam3RegionError(f"regions[{index}].group_id is invalid")
        identity = (view_id, group_id)
        if identity in identities:
            raise Sam3RegionError(f"duplicate region identity: {identity}")
        identities.add(identity)
        boxes = raw.get("boxes")
        click_sets = raw.get("click_sets")
        if click_sets is not None:
            if request_schema not in {
                POINT_SCHEMA_VERSION,
                ORDERED_POINT_SCHEMA_VERSION,
            }:
                raise Sam3RegionError(
                    f"regions[{index}].click_sets require the v2 or v3 request schema"
                )
            if boxes not in (None, []):
                raise Sam3RegionError(
                    f"regions[{index}] cannot mix boxes and interactive click_sets"
                )
            if group_id != "__foreground__":
                raise Sam3RegionError(
                    "interactive click_sets are restricted to whole-workpiece foreground"
                )
            try:
                validator = (
                    validate_ordered_click_sets
                    if request_schema == ORDERED_POINT_SCHEMA_VERSION
                    else validate_click_sets
                )
                raw["click_sets"] = validator(
                    click_sets, f"regions[{index}].click_sets"
                )
            except ValueError as exc:
                raise Sam3RegionError(str(exc)) from exc
            confirmed_mask = raw.get("confirmed_mask")
            if not isinstance(confirmed_mask, Mapping):
                raise Sam3RegionError(
                    f"regions[{index}].confirmed_mask must be an object"
                )
            mask_path = _resolve_file(
                confirmed_mask.get("path"),
                base=path.parent,
                label=f"regions[{index}].confirmed_mask.path",
            )
            expected_digest = confirmed_mask.get("sha256")
            if not isinstance(
                expected_digest, str
            ) or expected_digest != _human_sha256_file(mask_path):
                raise Sam3RegionError(f"regions[{index}].confirmed_mask hash mismatch")
            raw["confirmed_mask"] = {
                **dict(confirmed_mask),
                "path": str(mask_path),
            }
            raw["boxes"] = []
            raw["prompt"] = "visual"
        else:
            if not isinstance(boxes, list) or not boxes:
                raise Sam3RegionError(f"regions[{index}].boxes must be non-empty")
            raw["boxes"] = [
                _validated_box(box, f"regions[{index}].boxes[{box_index}]")
                for box_index, box in enumerate(boxes)
            ]
            prompt = raw.get("prompt", "visual")
            if not isinstance(prompt, str) or not prompt.strip():
                raise Sam3RegionError(f"regions[{index}].prompt must be non-empty")
            raw["prompt"] = prompt.strip()
            cad_seed = raw.get("cad_projection_seed")
            if cad_seed is not None:
                if not isinstance(cad_seed, Mapping):
                    raise Sam3RegionError(
                        f"regions[{index}].cad_projection_seed must be an object"
                    )
                seed_path = _resolve_file(
                    cad_seed.get("path"),
                    base=path.parent,
                    label=f"regions[{index}].cad_projection_seed.path",
                )
                expected_digest = cad_seed.get("sha256")
                if not isinstance(
                    expected_digest, str
                ) or expected_digest != _sha256_file(seed_path):
                    raise Sam3RegionError(
                        f"regions[{index}].cad_projection_seed hash mismatch"
                    )
                raw["cad_projection_seed"] = {
                    **dict(cad_seed),
                    "path": str(seed_path),
                }
            amodal_template = raw.get("cad_amodal_template")
            if amodal_template is not None:
                if not isinstance(amodal_template, Mapping):
                    raise Sam3RegionError(
                        f"regions[{index}].cad_amodal_template must be an object"
                    )
                if cad_seed is None:
                    raise Sam3RegionError(
                        f"regions[{index}] cannot use an amodal template without a visible CAD seed"
                    )
                amodal_path = _resolve_file(
                    amodal_template.get("path"),
                    base=path.parent,
                    label=f"regions[{index}].cad_amodal_template.path",
                )
                expected_digest = amodal_template.get("sha256")
                if not isinstance(
                    expected_digest, str
                ) or expected_digest != _sha256_file(amodal_path):
                    raise Sam3RegionError(
                        f"regions[{index}].cad_amodal_template hash mismatch"
                    )
                contract = amodal_template.get("projection_contract")
                if (
                    not isinstance(contract, Mapping)
                    or contract.get("whole_asset_camera_unchanged") is not True
                    or contract.get("whole_asset_transform_unchanged") is not True
                    or contract.get("per_mesh_pose_change_allowed") is not False
                    or contract.get("other_mesh_occlusion_disabled_for_shape_only")
                    is not True
                ):
                    raise Sam3RegionError(
                        f"regions[{index}].cad_amodal_template projection contract is invalid"
                    )
                raw["cad_amodal_template"] = {
                    **dict(amodal_template),
                    "path": str(amodal_path),
                }
    if request_schema == ORDERED_POINT_SCHEMA_VERSION:
        if request.get("prompt_authority") != (
            "human_confirmed_sam3_interactive_points"
        ):
            raise Sam3RegionError(
                "ordered SAM3 request has an invalid prompt authority"
            )
        human_annotation = request.get("human_annotation")
        expected_annotation_fields = {
            "schema_version",
            "document_sha256",
            "all_views_confirmed",
            "human_mask_is_authoritative",
            "formal_rerun_minimum_iou",
        }
        annotation_digest = (
            human_annotation.get("document_sha256")
            if isinstance(human_annotation, Mapping)
            else None
        )
        if (
            not isinstance(human_annotation, Mapping)
            or set(human_annotation) != expected_annotation_fields
            or human_annotation.get("schema_version") != HUMAN_ANNOTATION_SCHEMA_VERSION
            or not isinstance(annotation_digest, str)
            or len(annotation_digest) != 64
            or any(
                character not in "0123456789abcdef" for character in annotation_digest
            )
            or human_annotation.get("all_views_confirmed") is not True
            or human_annotation.get("human_mask_is_authoritative") is not True
            or human_annotation.get("formal_rerun_minimum_iou") != 0.995
        ):
            raise Sam3RegionError(
                "ordered SAM3 request has an invalid human annotation record"
            )
        if len(raw_regions) != len(source_paths) or {
            str(region.get("view_id")) for region in raw_regions
        } != set(source_paths):
            raise Sam3RegionError(
                "ordered SAM3 request must contain exactly one region per source view"
            )
        if any(
            region.get("group_id") != "__foreground__" or not region.get("click_sets")
            for region in raw_regions
        ):
            raise Sam3RegionError(
                "ordered SAM3 request regions must all use foreground click sets"
            )
    return request, source_paths


def _git_revision(repository: Path) -> str | None:
    try:
        process = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    revision = process.stdout.strip()
    return revision if len(revision) == 40 else None


def _box_pixels(
    box: Sequence[int], *, width: int, height: int
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    left = max(0, min(width - 1, int(math.floor(x0 * width / 1000))))
    top = max(0, min(height - 1, int(math.floor(y0 * height / 1000))))
    right = max(left + 1, min(width, int(math.ceil(x1 * width / 1000))))
    bottom = max(top + 1, min(height, int(math.ceil(y1 * height / 1000))))
    return left, top, right, bottom


def _normalized_cxcywh(box: Sequence[int], *, width: int, height: int) -> list[float]:
    left, top, right, bottom = _box_pixels(box, width=width, height=height)
    return [
        ((left + right) * 0.5) / width,
        ((top + bottom) * 0.5) / height,
        (right - left) / width,
        (bottom - top) / height,
    ]


def _normalized_shape_agreement(
    candidate_mask: Any,
    cad_seed_mask: Any,
    *,
    normalization_size: int = CAD_SHAPE_NORMALIZATION_SIZE,
    maximum_rotation_degrees: float = 35.0,
) -> dict[str, Any]:
    """Compare a SAM mask with a CAD silhouette independently of position.

    The projected CAD mask is only a location prior.  For candidate selection,
    both masks are cropped to their own support, isotropically normalized on a
    shared square canvas, and compared over a bounded rotation search.  This
    lets a correctly segmented photographed component win even when the coarse
    camera projection is translated or slightly rotated, without deforming the
    CAD asset or treating the projection as photo ground truth.
    """

    import cv2
    import numpy as np

    if (
        isinstance(normalization_size, bool)
        or not isinstance(normalization_size, int)
        or normalization_size < 32
    ):
        raise Sam3RegionError("shape normalization size must be at least 32")

    def normalize(value: Any, label: str) -> tuple[Any, int, float]:
        mask = np.asarray(value, dtype=bool)
        pixels = int(np.count_nonzero(mask))
        if mask.ndim != 2 or pixels == 0:
            raise Sam3RegionError(f"{label} must be a non-empty 2-D mask")
        ys, xs = np.where(mask)
        crop = (mask[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]).astype(np.uint8)
        height, width = crop.shape
        margin = max(4, normalization_size // 12)
        available = normalization_size - 2 * margin
        scale = available / max(height, width)
        resized_width = max(1, int(round(width * scale)))
        resized_height = max(1, int(round(height * scale)))
        resized = cv2.resize(
            crop,
            (resized_width, resized_height),
            interpolation=cv2.INTER_NEAREST,
        )
        canvas = np.zeros((normalization_size, normalization_size), dtype=np.uint8)
        left = (normalization_size - resized_width) // 2
        top = (normalization_size - resized_height) // 2
        canvas[top : top + resized_height, left : left + resized_width] = resized
        return canvas > 0, pixels, width / max(1, height)

    candidate, candidate_pixels, candidate_aspect = normalize(
        candidate_mask, "SAM3 candidate"
    )
    seed, seed_pixels, seed_aspect = normalize(cad_seed_mask, "CAD shape seed")
    center = ((normalization_size - 1) * 0.5, (normalization_size - 1) * 0.5)
    best: tuple[float, float, float, float, float] | None = None
    if (
        isinstance(maximum_rotation_degrees, bool)
        or not isinstance(maximum_rotation_degrees, (int, float))
        or not math.isfinite(float(maximum_rotation_degrees))
        or not 0.0 <= float(maximum_rotation_degrees) <= 35.0
    ):
        raise Sam3RegionError("shape rotation bound must be between zero and 35 degrees")
    rotations = (
        np.asarray([0.0], dtype=np.float64)
        if float(maximum_rotation_degrees) == 0.0
        else np.linspace(
            -float(maximum_rotation_degrees),
            float(maximum_rotation_degrees),
            15,
        )
    )
    for rotation in rotations:
        matrix = cv2.getRotationMatrix2D(center, float(rotation), 1.0)
        rotated_seed = (
            cv2.warpAffine(
                seed.astype(np.uint8),
                matrix,
                (normalization_size, normalization_size),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            > 0
        )
        intersection = int(np.count_nonzero(candidate & rotated_seed))
        candidate_normalized_pixels = int(np.count_nonzero(candidate))
        seed_normalized_pixels = int(np.count_nonzero(rotated_seed))
        union = candidate_normalized_pixels + seed_normalized_pixels - intersection
        iou = intersection / max(1, union)
        precision = intersection / max(1, candidate_normalized_pixels)
        recall = intersection / max(1, seed_normalized_pixels)
        overlap_smaller = intersection / max(
            1, min(candidate_normalized_pixels, seed_normalized_pixels)
        )
        score = iou + 0.04 * min(precision, recall)
        result = (score, iou, min(precision, recall), overlap_smaller, float(rotation))
        if best is None or result > best:
            best = result
    if best is None:  # pragma: no cover - non-empty masks always enter the loop
        raise Sam3RegionError("unable to compare CAD and SAM3 shapes")
    _score, iou, minimum_precision_recall, overlap_smaller, rotation = best
    area_ratio = candidate_pixels / max(1, seed_pixels)
    area_agreement = min(area_ratio, 1.0 / max(area_ratio, 1e-12))
    return {
        "cad_shape_seed_pixels": seed_pixels,
        "cad_shape_candidate_pixels": candidate_pixels,
        "cad_shape_iou": iou,
        "cad_shape_minimum_precision_recall": minimum_precision_recall,
        "cad_shape_overlap_smaller": overlap_smaller,
        "cad_shape_rotation_degrees": rotation,
        "cad_shape_seed_aspect_ratio": seed_aspect,
        "cad_shape_candidate_aspect_ratio": candidate_aspect,
        "cad_shape_candidate_to_seed_area_ratio": area_ratio,
        "cad_shape_area_agreement": area_agreement,
        "cad_shape_normalization_size": normalization_size,
        "cad_shape_location_invariant": True,
        "cad_shape_rotation_search_degrees": float(maximum_rotation_degrees),
    }


def _occlusion_aware_amodal_agreement(
    candidate_mask: Any,
    visible_seed_mask: Any,
    amodal_mask: Any,
) -> dict[str, Any]:
    """Compare a photo mask with complete CAD shape without penalizing occlusion.

    ``visible_seed_mask`` is the full-assembly modal Part-ID projection.  Pixels
    present in the isolated mesh but absent from that mask are known CAD
    occlusion.  Completing the photo candidate only at those known pixels lets
    the full mesh constrain shape while keeping the modal mask authoritative
    for photo visibility and location.
    """

    import cv2
    import numpy as np

    candidate = np.asarray(candidate_mask, dtype=bool)
    visible = np.asarray(visible_seed_mask, dtype=bool)
    amodal = np.asarray(amodal_mask, dtype=bool)
    if candidate.ndim != 2 or visible.shape != candidate.shape or amodal.shape != candidate.shape:
        raise Sam3RegionError("candidate, visible CAD seed, and amodal template must align")
    candidate_pixels = int(np.count_nonzero(candidate))
    visible_pixels = int(np.count_nonzero(visible))
    amodal_pixels = int(np.count_nonzero(amodal))
    if min(candidate_pixels, visible_pixels, amodal_pixels) <= 0:
        raise Sam3RegionError("occlusion-aware CAD masks must be non-empty")
    rows, columns = np.where(amodal)
    diagonal = math.hypot(
        int(columns.max() - columns.min() + 1),
        int(rows.max() - rows.min() + 1),
    )
    support_radius = max(1, int(round(0.02 * diagonal)))
    support = cv2.dilate(
        amodal.astype(np.uint8),
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * support_radius + 1, 2 * support_radius + 1),
        ),
    ) > 0
    supported_pixels = int(np.count_nonzero(candidate & support))
    visible_inside_amodal = visible & support
    known_occlusion = amodal & ~visible_inside_amodal
    completed = candidate | known_occlusion
    completed_intersection = int(np.count_nonzero(completed & amodal))
    completed_union = int(np.count_nonzero(completed | amodal))
    normalized = _normalized_shape_agreement(
        completed,
        amodal,
        maximum_rotation_degrees=0.0,
    )
    return {
        "cad_amodal_pixels": amodal_pixels,
        "cad_visible_seed_pixels": visible_pixels,
        "cad_known_occluded_pixels": int(np.count_nonzero(known_occlusion)),
        "cad_visible_fraction_of_amodal": int(np.count_nonzero(visible & amodal))
        / max(1, amodal_pixels),
        "cad_amodal_support_radius_pixels": support_radius,
        "cad_amodal_candidate_precision": supported_pixels / max(1, candidate_pixels),
        "cad_amodal_completion_iou": completed_intersection
        / max(1, completed_union),
        "candidate_to_cad_amodal_area_ratio": candidate_pixels
        / max(1, amodal_pixels),
        "cad_amodal_shape_iou": float(normalized["cad_shape_iou"]),
        "cad_amodal_shape_minimum_precision_recall": float(
            normalized["cad_shape_minimum_precision_recall"]
        ),
        "cad_amodal_shape_overlap_smaller": float(
            normalized["cad_shape_overlap_smaller"]
        ),
        "cad_amodal_shape_rotation_degrees": float(
            normalized["cad_shape_rotation_degrees"]
        ),
        "cad_amodal_shape_rotation_search_degrees": 0.0,
        "cad_amodal_occlusion_aware": True,
    }


def _candidate_metrics(
    mask: Any,
    *,
    box: Sequence[int],
    width: int,
    height: int,
    cad_seed: Any | None = None,
    cad_amodal: Any | None = None,
) -> dict[str, Any]:
    import numpy as np

    candidate = np.asarray(mask, dtype=bool)
    if candidate.shape != (height, width):
        raise Sam3RegionError(
            f"SAM3 returned mask shape {candidate.shape}, expected {(height, width)}"
        )
    left, top, right, bottom = _box_pixels(box, width=width, height=height)
    prompt = np.zeros((height, width), dtype=bool)
    prompt[top:bottom, left:right] = True
    intersection = int(np.count_nonzero(candidate & prompt))
    mask_pixels = int(np.count_nonzero(candidate))
    prompt_pixels = int(np.count_nonzero(prompt))
    overlap_smaller = intersection / max(1, min(mask_pixels, prompt_pixels))
    prompt_coverage = intersection / max(1, prompt_pixels)
    mask_precision = intersection / max(1, mask_pixels)
    center_x = min(width - 1, max(0, (left + right - 1) // 2))
    center_y = min(height - 1, max(0, (top + bottom - 1) // 2))
    result = {
        "mask_pixels": mask_pixels,
        "image_fraction": mask_pixels / max(1, width * height),
        "intersection_pixels": intersection,
        "overlap_smaller": overlap_smaller,
        "prompt_coverage": prompt_coverage,
        "mask_precision": mask_precision,
        "prompt_center_inside": bool(candidate[center_y, center_x]),
    }
    if cad_seed is not None:
        seed = np.asarray(cad_seed, dtype=bool)
        if seed.shape != candidate.shape:
            raise Sam3RegionError(
                f"CAD seed shape {seed.shape}, expected {candidate.shape}"
            )
        seed_pixels = int(np.count_nonzero(seed))
        seed_intersection = int(np.count_nonzero(candidate & seed))
        seed_union = mask_pixels + seed_pixels - seed_intersection
        result.update(
            {
                "cad_seed_pixels": seed_pixels,
                "cad_seed_intersection_pixels": seed_intersection,
                "cad_seed_iou": seed_intersection / max(1, seed_union),
                "cad_seed_precision": seed_intersection / max(1, mask_pixels),
                "cad_seed_recall": seed_intersection / max(1, seed_pixels),
                "cad_seed_overlap_smaller": seed_intersection
                / max(1, min(mask_pixels, seed_pixels)),
                "candidate_to_cad_seed_area_ratio": mask_pixels / max(1, seed_pixels),
                **_normalized_shape_agreement(candidate, seed),
            }
        )
        if cad_amodal is not None:
            result.update(
                _occlusion_aware_amodal_agreement(candidate, seed, cad_amodal)
            )
    elif cad_amodal is not None:
        raise Sam3RegionError("amodal CAD template requires a visible CAD seed")
    return result


def _segment_box(
    *,
    processor: Any,
    image: Any,
    image_state: Mapping[str, Any] | None = None,
    prompt: str,
    box: Sequence[int],
    minimum_model_score: float,
    minimum_prompt_overlap: float,
    maximum_image_fraction: float,
    minimum_mask_pixels: int,
    cad_seed: Any | None = None,
    cad_amodal: Any | None = None,
) -> tuple[Any | None, dict[str, Any]]:
    import numpy as np

    width, height = image.size
    if image_state is None:
        state = processor.set_image(image)
    else:
        # SAM3's image backbone is the expensive part.  Its prompt methods only
        # append language/geometric entries, so give every independent prompt a
        # fresh mapping while sharing the immutable image tensors.
        state = {
            "original_height": image_state["original_height"],
            "original_width": image_state["original_width"],
            "backbone_out": dict(image_state["backbone_out"]),
        }
    processor.set_text_prompt(prompt=prompt, state=state)
    output = processor.add_geometric_prompt(
        box=_normalized_cxcywh(box, width=width, height=height),
        label=True,
        state=state,
    )
    masks_raw = output.get("masks")
    scores_raw = output.get("scores")
    if masks_raw is None or scores_raw is None:
        raise Sam3RegionError("SAM3 output is missing masks or scores")
    masks = masks_raw.detach().to("cpu").numpy()
    # Instance-interactive SAM3 may expose detector confidences as bfloat16;
    # NumPy has no portable bfloat16 scalar type.  The ranking contract stores
    # ordinary finite floats, so normalize on-device before crossing runtimes.
    scores = scores_raw.detach().float().to("cpu").numpy().reshape(-1)
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim != 3 or len(masks) != len(scores):
        raise Sam3RegionError("SAM3 masks/scores have incompatible shapes")
    candidates: list[tuple[tuple[Any, ...], Any, dict[str, Any]]] = []
    audits: list[dict[str, Any]] = []
    for index, (mask, raw_score) in enumerate(zip(masks, scores)):
        score = float(raw_score)
        metrics = _candidate_metrics(
            mask,
            box=box,
            width=width,
            height=height,
            cad_seed=cad_seed,
            cad_amodal=cad_amodal,
        )
        reasons: list[str] = []
        if score < minimum_model_score:
            reasons.append("model_score_below_threshold")
        if metrics["mask_pixels"] < minimum_mask_pixels:
            reasons.append("mask_too_small")
        if metrics["image_fraction"] > maximum_image_fraction:
            reasons.append("mask_too_large")
        if metrics["overlap_smaller"] < minimum_prompt_overlap:
            reasons.append("insufficient_prompt_overlap")
        if (
            cad_seed is None
            and not metrics["prompt_center_inside"]
            and metrics["prompt_coverage"] < 0.10
        ):
            reasons.append("prompt_not_localized")
        if cad_seed is not None:
            if int(metrics["cad_shape_seed_pixels"]) < DEFAULT_MINIMUM_CAD_SHAPE_SEED_PIXELS:
                reasons.append("cad_shape_seed_too_small")
            elif cad_amodal is not None:
                if (
                    float(metrics["cad_seed_recall"])
                    < DEFAULT_MINIMUM_VISIBLE_CAD_SEED_RECALL
                ):
                    reasons.append("insufficient_visible_cad_seed_recall")
                if (
                    float(metrics["cad_amodal_candidate_precision"])
                    < DEFAULT_MINIMUM_AMODAL_CANDIDATE_PRECISION
                ):
                    reasons.append("candidate_extends_outside_complete_mesh_shape")
                if (
                    float(metrics["cad_amodal_shape_iou"])
                    < DEFAULT_MINIMUM_AMODAL_COMPLETION_SHAPE_IOU
                ):
                    reasons.append("occlusion_aware_amodal_shape_mismatch")
                if (
                    float(metrics["candidate_to_cad_amodal_area_ratio"])
                    > DEFAULT_MAXIMUM_CANDIDATE_TO_AMODAL_AREA_RATIO
                ):
                    reasons.append("candidate_area_exceeds_complete_mesh_shape")
            else:
                if float(metrics["cad_shape_iou"]) < DEFAULT_MINIMUM_CAD_SHAPE_IOU:
                    reasons.append("cad_shape_iou_below_threshold")
                if (
                    float(metrics["cad_shape_area_agreement"])
                    < DEFAULT_MINIMUM_CAD_SHAPE_AREA_AGREEMENT
                ):
                    reasons.append("cad_shape_area_mismatch")
        audit = {
            "candidate_index": index,
            "model_score": round(score, 8),
            **{
                key: round(float(value), 8) if isinstance(value, float) else value
                for key, value in metrics.items()
            },
            "accepted": not reasons,
            "reason_codes": reasons,
        }
        audits.append(audit)
        if not reasons:
            if cad_seed is not None:
                rank = (
                    float(
                        metrics.get("cad_amodal_shape_iou", metrics["cad_shape_iou"])
                    ),
                    float(metrics.get("cad_amodal_candidate_precision", 1.0)),
                    float(metrics["cad_shape_minimum_precision_recall"]),
                    float(metrics["cad_shape_overlap_smaller"]),
                    float(metrics["cad_shape_area_agreement"]),
                    score,
                    -index,
                )
            else:
                rank = (
                    score,
                    float(metrics["overlap_smaller"]),
                    float(metrics["prompt_coverage"]),
                    -index,
                )
            candidates.append((rank, np.asarray(mask, dtype=bool), audit))
    if not candidates:
        return None, {"accepted": False, "candidates": audits}
    _rank, selected, selected_audit = max(candidates, key=lambda item: item[0])
    return selected, {
        "accepted": True,
        "selected_candidate_index": selected_audit["candidate_index"],
        "candidates": audits,
    }


def _segment_points(
    *,
    model: Any,
    image_state: Mapping[str, Any],
    image: Any,
    click_set: Mapping[str, Any],
    minimum_model_score: float,
    minimum_prompt_overlap: float,
    maximum_image_fraction: float,
    minimum_mask_pixels: int,
) -> tuple[Any | None, dict[str, Any]]:
    """Run the official SAM3 instance-interactivity path for one click set."""

    import numpy as np

    positive_points = click_set["positive_points"]
    negative_points = click_set["negative_points"]
    labelled = [(point, 1) for point in positive_points] + [
        (point, 0) for point in negative_points
    ]
    point_coords = np.asarray(
        [
            grid_to_pixel(point, width=image.width, height=image.height)
            for point, _label in labelled
        ],
        dtype=np.float32,
    )
    point_labels = np.asarray([label for _point, label in labelled], dtype=np.int32)
    masks, scores, _low_resolution_logits = model.predict_inst(
        image_state,
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=True,
    )
    try:
        return select_point_candidate(
            masks,
            scores,
            positive_points=positive_points,
            negative_points=negative_points,
            width=image.width,
            height=image.height,
            minimum_model_score=minimum_model_score,
            minimum_prompt_agreement=minimum_prompt_overlap,
            maximum_image_fraction=maximum_image_fraction,
            minimum_mask_pixels=minimum_mask_pixels,
        )
    except ValueError as exc:
        raise Sam3RegionError(str(exc)) from exc


def _predict_inst_accepts_mask_input(model: Any) -> bool:
    """Return whether the installed instance predictor exposes dense prompts."""

    try:
        parameters = inspect.signature(model.predict_inst).parameters.values()
    except (TypeError, ValueError, AttributeError):
        return False
    return any(
        parameter.name == "mask_input"
        or parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    )


def _dense_cad_mask_logits(
    mask: Any,
    *,
    output_size: tuple[int, int] = (256, 256),
    magnitude: float = DEFAULT_DENSE_CAD_MASK_LOGIT_MAGNITUDE,
) -> Any:
    """Encode a registered CAD mask as SAM-compatible signed low-res logits.

    A signed distance field is used instead of a hard binary tensor so the
    image decoder can move the boundary.  This is a prior, never a final mask.
    ``output_size`` follows OpenCV's ``(width, height)`` convention.
    """

    import cv2
    import numpy as np

    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2 or not np.any(binary):
        raise Sam3RegionError("dense CAD mask prompt is empty or malformed")
    width, height = output_size
    if width <= 0 or height <= 0:
        raise Sam3RegionError("dense CAD mask prompt size must be positive")
    # INTER_AREA retains subpixel occupancy for thin projected components.
    occupancy = cv2.resize(
        binary.astype(np.float32),
        (int(width), int(height)),
        interpolation=cv2.INTER_AREA,
    )
    low_res = occupancy > 0.05
    if not np.any(low_res):
        maximum = np.unravel_index(int(np.argmax(occupancy)), occupancy.shape)
        low_res[maximum] = True
    inside = cv2.distanceTransform(low_res.astype(np.uint8), cv2.DIST_L2, 5)
    outside = cv2.distanceTransform((~low_res).astype(np.uint8), cv2.DIST_L2, 5)
    signed = inside - outside
    # Normalize from foreground thickness, not the much larger background.
    # Otherwise a tiny industrial part receives a near-zero positive logit
    # merely because most of the image lies far outside the part.
    foreground_distances = inside[low_res]
    scale = max(1.0, float(np.percentile(foreground_distances, 75)))
    logits = np.clip(signed / scale, -1.0, 1.0) * float(magnitude)
    return logits.astype(np.float32)[None, :, :]


def _segment_dense_mask_points(
    *,
    model: Any,
    image_state: Mapping[str, Any],
    image: Any,
    click_set: Mapping[str, Any],
    aligned_cad_seed: Any,
    minimum_model_score: float,
    minimum_prompt_overlap: float,
    maximum_image_fraction: float,
    minimum_mask_pixels: int,
) -> tuple[Any | None, dict[str, Any]]:
    """Run SAM3 with automatic CAD mask, positive, and negative prompts."""

    import numpy as np

    if not _predict_inst_accepts_mask_input(model):
        return None, {
            "accepted": False,
            "available": False,
            "reason_codes": ["installed_predictor_has_no_dense_mask_input"],
        }
    positive_points = click_set["positive_points"]
    negative_points = click_set["negative_points"]
    labelled = [(point, 1) for point in positive_points] + [
        (point, 0) for point in negative_points
    ]
    point_coords = np.asarray(
        [
            grid_to_pixel(point, width=image.width, height=image.height)
            for point, _label in labelled
        ],
        dtype=np.float32,
    )
    point_labels = np.asarray([label for _point, label in labelled], dtype=np.int32)
    mask_input_size = (256, 256)
    predictor = getattr(model, "inst_interactive_predictor", None)
    prompt_encoder = getattr(getattr(predictor, "model", None), "sam_prompt_encoder", None)
    declared_size = getattr(prompt_encoder, "mask_input_size", None)
    if (
        isinstance(declared_size, (tuple, list))
        and len(declared_size) == 2
        and all(isinstance(value, int) and value > 0 for value in declared_size)
    ):
        # The model declares (height, width), OpenCV consumes (width, height).
        mask_input_size = (int(declared_size[1]), int(declared_size[0]))
    mask_logits = _dense_cad_mask_logits(
        aligned_cad_seed,
        output_size=mask_input_size,
    )
    masks, scores, low_resolution_logits = model.predict_inst(
        image_state,
        point_coords=point_coords,
        point_labels=point_labels,
        mask_input=mask_logits,
        multimask_output=False,
    )
    try:
        selected, audit = select_point_candidate(
            masks,
            scores,
            positive_points=positive_points,
            negative_points=negative_points,
            width=image.width,
            height=image.height,
            minimum_model_score=minimum_model_score,
            minimum_prompt_agreement=minimum_prompt_overlap,
            maximum_image_fraction=maximum_image_fraction,
            minimum_mask_pixels=minimum_mask_pixels,
            allow_rejected_preview=True,
        )
    except ValueError as exc:
        raise Sam3RegionError(str(exc)) from exc
    # Neighbor/ring negatives are synthesized from a rendered alignment and
    # can land on the true photo boundary by a few pixels.  For the dense CAD
    # branch only, treat those negatives as advisory and let the stricter
    # visible/amodal shape gate decide.  Score, size, and positive-point
    # failures remain hard vetoes.
    if selected is not None and audit.get("accepted") is not True:
        preview_index = audit.get("preview_candidate_index")
        preview = next(
            (
                row
                for row in audit.get("candidates", [])
                if isinstance(row, Mapping)
                and row.get("candidate_index") == preview_index
            ),
            None,
        )
        allowed = {"negative_point_inside_mask", "insufficient_point_agreement"}
        reasons = set(preview.get("reason_codes", [])) if preview else set()
        if not reasons or not reasons <= allowed:
            selected = None
        else:
            audit = {
                **audit,
                "automatic_negative_points_advisory": True,
                "requires_complete_shape_gate": True,
            }
    return selected, {
        **audit,
        "available": True,
        "mask_prompt": {
            "source": "aligned_whole_assembly_visible_part_id",
            "representation": "signed_distance_low_resolution_logits",
            "input_size": [int(mask_logits.shape[2]), int(mask_logits.shape[1])],
            "minimum_logit": float(mask_logits.min()),
            "maximum_logit": float(mask_logits.max()),
            "per_mesh_pose_change_allowed": False,
        },
        "low_resolution_logit_shape": list(np.asarray(low_resolution_logits).shape),
    }


def _translate_binary_mask(mask: Any, translation_xy: Sequence[float]) -> Any:
    """Translate one renderer mask by a view-shared image-plane residual."""

    import cv2
    import numpy as np

    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2:
        raise Sam3RegionError("view-shared alignment mask must be two-dimensional")
    if len(translation_xy) != 2:
        raise Sam3RegionError("view-shared translation must contain x and y")
    dx, dy = (float(translation_xy[0]), float(translation_xy[1]))
    return cv2.warpAffine(
        binary.astype(np.uint8),
        np.asarray([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32),
        (binary.shape[1], binary.shape[0]),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ) > 0


def _estimate_view_shared_translation(
    cad_seeds: Mapping[str, Any],
    whole_workpiece_foreground: Any | None,
) -> dict[str, Any]:
    """Estimate one rigid image residual for every Part-ID in a view.

    The union of renderer-authored visible Part IDs is aligned to the photo's
    whole-workpiece foreground.  No individual Part-ID is allowed to vote for
    or receive a different translation.
    """

    import numpy as np

    if not cad_seeds:
        # Whole-workpiece foreground replay and generic box-prompt requests do
        # not have per-Part-ID CAD projections.  View-shared alignment is a
        # Part-ID refinement contract, so it is explicitly inapplicable here
        # rather than an error or an inferred per-mesh motion.
        return {
            "translation_xy_pixels": [0.0, 0.0],
            "maximum_translation_xy_pixels": [0, 0],
            "estimation_mode": "not_applicable_no_cad_seeds",
            "part_specific_translation_allowed": False,
            "cad_union_pixels": 0,
        }
    shapes = {np.asarray(mask, dtype=bool).shape for mask in cad_seeds.values()}
    if len(shapes) != 1:
        raise Sam3RegionError("view CAD seeds have inconsistent dimensions")
    height, width = next(iter(shapes))
    union = np.zeros((height, width), dtype=bool)
    for mask in cad_seeds.values():
        union |= np.asarray(mask, dtype=bool)
    union_pixels = int(np.count_nonzero(union))
    if whole_workpiece_foreground is None:
        return {
            "translation_xy_pixels": [0.0, 0.0],
            "maximum_translation_xy_pixels": [0, 0],
            "estimation_mode": "no_foreground_fail_closed_zero_translation",
            "part_specific_translation_allowed": False,
            "cad_union_pixels": union_pixels,
        }
    foreground = np.asarray(whole_workpiece_foreground, dtype=bool)
    if foreground.shape != union.shape or not np.any(foreground):
        raise Sam3RegionError(
            "whole-workpiece foreground is empty or differs from CAD dimensions"
        )
    maximum_dx = min(
        DEFAULT_MAXIMUM_VIEW_SHARED_TRANSLATION_PIXELS,
        max(2, int(round(0.03 * width))),
    )
    maximum_dy = min(
        DEFAULT_MAXIMUM_VIEW_SHARED_TRANSLATION_PIXELS,
        max(2, int(round(0.03 * height))),
    )
    foreground_pixels = int(np.count_nonzero(foreground))
    candidates: list[tuple[tuple[float, ...], int, int, int, float]] = []
    for dy in range(-maximum_dy, maximum_dy + 1):
        for dx in range(-maximum_dx, maximum_dx + 1):
            shifted = _translate_binary_mask(union, (dx, dy))
            shifted_pixels = int(np.count_nonzero(shifted))
            intersection = int(np.count_nonzero(shifted & foreground))
            union_count = shifted_pixels + foreground_pixels - intersection
            cad_recall = intersection / max(1, shifted_pixels)
            iou = intersection / max(1, union_count)
            # Prefer foreground containment, then IoU, and finally the
            # smallest rigid residual.  This makes a broad foreground plateau
            # deterministically choose the calibrated zero pose.
            rank = (
                cad_recall,
                iou,
                -math.hypot(dx, dy),
                -abs(dx),
                -abs(dy),
                -dx,
                -dy,
            )
            candidates.append((rank, dx, dy, intersection, iou))
    _rank, dx, dy, intersection, iou = max(candidates, key=lambda row: row[0])
    return {
        "translation_xy_pixels": [float(dx), float(dy)],
        "maximum_translation_xy_pixels": [maximum_dx, maximum_dy],
        "estimation_mode": (
            "whole_workpiece_foreground_to_visible_cad_union_integer_translation"
        ),
        "part_specific_translation_allowed": False,
        "cad_union_pixels": union_pixels,
        "foreground_pixels": foreground_pixels,
        "intersection_pixels": intersection,
        "cad_union_foreground_recall": round(
            intersection / max(1, union_pixels), 8
        ),
        "union_iou": round(iou, 8),
    }


def _bounded_shared_camera_alignment(
    cad_seed: Any,
    coarse_candidate: Any,
    *,
    box: Sequence[int],
    width: int,
    height: int,
    view_shared_alignment: Mapping[str, Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Return a view-aligned mask without applying a Part-ID-specific motion."""

    import numpy as np

    seed = np.asarray(cad_seed, dtype=bool)
    coarse = np.asarray(coarse_candidate, dtype=bool)
    if seed.shape != (height, width) or coarse.shape != (height, width):
        raise Sam3RegionError("CAD shape point masks differ from the source image")
    if int(np.count_nonzero(seed)) < DEFAULT_MINIMUM_CAD_SHAPE_SEED_PIXELS:
        raise Sam3RegionError("CAD shape seed is too small for automatic points")

    def centroid(mask: Any) -> tuple[float, float]:
        ys, xs = np.where(mask)
        if not len(xs):
            raise Sam3RegionError("automatic shape point mask is empty")
        return float(xs.mean()), float(ys.mean())

    seed_center_x, seed_center_y = centroid(seed)
    coarse_x, coarse_y = centroid(coarse)
    requested_dx = coarse_x - seed_center_x
    requested_dy = coarse_y - seed_center_y
    shared = dict(view_shared_alignment or {})
    translation = shared.get("translation_xy_pixels", [0.0, 0.0])
    return _translate_binary_mask(seed, translation), {
        **shared,
        "translation_xy_pixels": [float(translation[0]), float(translation[1])],
        "candidate_centroid_residual_xy_pixels": [
            round(requested_dx, 8),
            round(requested_dy, 8),
        ],
        "part_local_translation_xy_pixels": [0.0, 0.0],
        "alignment_model": "whole_asset_view_shared_translation_only",
        "per_mesh_pose_change_allowed": False,
        "part_specific_translation_allowed": False,
    }


def _shape_seed_click_set(
    cad_seed: Any,
    coarse_candidate: Any,
    *,
    box: Sequence[int],
    width: int,
    height: int,
    other_part_seeds: Any | None = None,
    view_shared_alignment: Mapping[str, Any] | None = None,
) -> tuple[dict[str, list[list[int]]], dict[str, Any]]:
    """Turn a same-view CAD silhouette into automatic SAM3 prompts.

    Positive points come from the selected Part-ID template.  Nearby rendered
    Part-ID templates contribute negative points, which prevents repeated
    bolts, knobs, and touching panels from claiming one another.
    """

    import cv2
    import numpy as np

    shifted, alignment_audit = _bounded_shared_camera_alignment(
        cad_seed,
        coarse_candidate,
        box=box,
        width=width,
        height=height,
        view_shared_alignment=view_shared_alignment,
    )
    distance = cv2.distanceTransform(shifted.astype(np.uint8), cv2.DIST_L2, 5)
    positive: list[tuple[int, int]] = []
    ranked = np.dstack(
        np.unravel_index(np.argsort(distance.ravel())[::-1], distance.shape)
    )[0]
    shifted_y, shifted_x = np.where(shifted > 0)
    diagonal = math.hypot(
        int(shifted_x.max() - shifted_x.min() + 1),
        int(shifted_y.max() - shifted_y.min() + 1),
    )
    spacing = max(3.0, 0.20 * diagonal)
    for y, x in ranked:
        if distance[y, x] <= 0:
            break
        point = (int(x), int(y))
        if all(
            math.hypot(point[0] - px, point[1] - py) >= spacing for px, py in positive
        ):
            positive.append(point)
        if len(positive) >= 4:
            break
    if not positive:
        raise Sam3RegionError("CAD shape seed produced no positive point")

    ring_radius = max(4, int(round(0.10 * diagonal)))
    shifted_u8 = shifted.astype(np.uint8)
    inner = cv2.dilate(
        shifted_u8,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )
    outer = cv2.dilate(
        shifted_u8,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * ring_radius + 1, 2 * ring_radius + 1),
        ),
        iterations=1,
    )
    ring_y, ring_x = np.where((outer > 0) & (inner == 0))
    negative: list[tuple[int, int]] = []
    center_x = float(shifted_x.mean())
    center_y = float(shifted_y.mean())
    ring_point_limit = 4 if other_part_seeds is not None else 8
    for angle in np.linspace(
        0.0, 2.0 * math.pi, ring_point_limit, endpoint=False
    ):
        if not len(ring_x):
            break
        target_x = center_x + math.cos(float(angle)) * (0.5 * diagonal + ring_radius)
        target_y = center_y + math.sin(float(angle)) * (0.5 * diagonal + ring_radius)
        index = int(np.argmin((ring_x - target_x) ** 2 + (ring_y - target_y) ** 2))
        point = (int(ring_x[index]), int(ring_y[index]))
        if point not in negative:
            negative.append(point)

    neighboring_negative_count = 0
    if other_part_seeds is not None:
        others = np.asarray(other_part_seeds, dtype=bool)
        if others.shape != shifted.shape:
            raise Sam3RegionError("neighbor CAD seed dimensions are incompatible")
        exclusion = others & ~(
            cv2.dilate(
                shifted.astype(np.uint8),
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                iterations=1,
            )
            > 0
        )
        left, top, right, bottom = _box_pixels(box, width=width, height=height)
        local = np.zeros_like(exclusion)
        local[top:bottom, left:right] = True
        exclusion &= local
        count, _labels, statistics, centroids = cv2.connectedComponentsWithStats(
            exclusion.astype(np.uint8), connectivity=8
        )
        components: list[tuple[float, tuple[int, int]]] = []
        for index in range(1, count):
            if int(statistics[index, cv2.CC_STAT_AREA]) < 2:
                continue
            point = (
                int(round(float(centroids[index][0]))),
                int(round(float(centroids[index][1]))),
            )
            distance_to_target = math.hypot(
                point[0] - center_x,
                point[1] - center_y,
            )
            components.append((distance_to_target, point))
        for _distance, point in sorted(components)[:4]:
            if point not in negative:
                negative.append(point)
                neighboring_negative_count += 1

    def grid(point: tuple[int, int]) -> list[int]:
        x, y = point
        return [
            int(round(x * 1000 / max(1, width - 1))),
            int(round(y * 1000 / max(1, height - 1))),
        ]

    return {
        "positive_points": [grid(point) for point in positive],
        "negative_points": [grid(point) for point in negative],
    }, {
        **alignment_audit,
        "translation_policy": (
            "whole_workpiece_foreground_to_visible_cad_union_shared_per_view"
        ),
        "positive_point_count": len(positive),
        "negative_point_count": len(negative),
        "neighboring_part_negative_point_count": neighboring_negative_count,
        "ring_radius_pixels": ring_radius,
    }


def _evaluate_shape_guided_candidate(
    candidate: Any,
    *,
    box: Sequence[int],
    width: int,
    height: int,
    aligned_seed: Any,
    aligned_amodal: Any | None,
    minimum_mask_pixels: int,
    other_part_seeds: Any | None = None,
) -> tuple[Any | None, dict[str, Any]]:
    """Apply the same CAD/occlusion gate to one sparse or dense proposal."""

    import cv2
    import numpy as np

    selected = np.asarray(candidate, dtype=bool)
    selected_array = np.asarray(selected, dtype=bool)
    direct_intersection = int(np.count_nonzero(selected_array & aligned_seed))
    direct_overlap_smaller = direct_intersection / max(
        1,
        min(
            int(np.count_nonzero(selected_array)),
            int(np.count_nonzero(aligned_seed)),
        ),
    )
    support_audit: dict[str, Any] = {
        "unbounded_mask_pixels": int(np.count_nonzero(selected_array)),
        "pre_bound_direct_overlap_smaller": round(direct_overlap_smaller, 8),
        "applied": False,
    }
    if direct_overlap_smaller >= 0.25:
        support_seed = aligned_amodal if aligned_amodal is not None else aligned_seed
        rows, columns = np.where(support_seed)
        diagonal = math.hypot(
            int(columns.max() - columns.min() + 1),
            int(rows.max() - rows.min() + 1),
        )
        maximum_radius = max(
            2,
            int(round(DEFAULT_MAXIMUM_CAD_SUPPORT_RADIUS_FRACTION * diagonal)),
        )
        maximum_pixels = int(
            math.floor(
                DEFAULT_MAXIMUM_FINAL_SAM_TO_CAD_AREA_RATIO
                * int(np.count_nonzero(support_seed))
            )
        )
        bounded = selected_array & support_seed
        selected_radius = 0
        for radius in range(maximum_radius + 1):
            support = (
                support_seed
                if radius == 0
                else cv2.dilate(
                    support_seed.astype(np.uint8),
                    cv2.getStructuringElement(
                        cv2.MORPH_ELLIPSE,
                        (2 * radius + 1, 2 * radius + 1),
                    ),
                )
                > 0
            )
            candidate = selected_array & support
            if int(np.count_nonzero(candidate)) > maximum_pixels:
                break
            bounded = candidate
            selected_radius = radius
        if int(np.count_nonzero(bounded)) >= minimum_mask_pixels:
            selected = bounded
            support_audit.update(
                {
                    "applied": True,
                    "selected_support_radius_pixels": selected_radius,
                    "bounded_mask_pixels": int(np.count_nonzero(bounded)),
                    "maximum_final_to_cad_area_ratio": (
                        DEFAULT_MAXIMUM_FINAL_SAM_TO_CAD_AREA_RATIO
                    ),
                }
            )
    metrics = _candidate_metrics(
        selected,
        box=box,
        width=width,
        height=height,
        cad_seed=aligned_seed,
        cad_amodal=aligned_amodal,
    )
    reasons: list[str] = []
    if int(metrics["cad_shape_seed_pixels"]) < DEFAULT_MINIMUM_CAD_SHAPE_SEED_PIXELS:
        reasons.append("cad_shape_seed_too_small")
    if aligned_amodal is not None:
        if (
            float(metrics["cad_seed_recall"])
            < DEFAULT_MINIMUM_VISIBLE_CAD_SEED_RECALL
        ):
            reasons.append("insufficient_visible_cad_seed_recall")
        if (
            float(metrics["cad_amodal_candidate_precision"])
            < DEFAULT_MINIMUM_AMODAL_CANDIDATE_PRECISION
        ):
            reasons.append("candidate_extends_outside_complete_mesh_shape")
        if (
            float(metrics["cad_amodal_shape_iou"])
            < DEFAULT_MINIMUM_AMODAL_COMPLETION_SHAPE_IOU
        ):
            reasons.append("occlusion_aware_amodal_shape_mismatch")
        if (
            float(metrics["candidate_to_cad_amodal_area_ratio"])
            > DEFAULT_MAXIMUM_CANDIDATE_TO_AMODAL_AREA_RATIO
        ):
            reasons.append("candidate_area_exceeds_complete_mesh_shape")
    else:
        if float(metrics["cad_shape_iou"]) < DEFAULT_MINIMUM_CAD_SHAPE_IOU:
            reasons.append("cad_shape_iou_below_threshold")
        if (
            float(metrics["cad_shape_area_agreement"])
            < DEFAULT_MINIMUM_CAD_SHAPE_AREA_AGREEMENT
        ):
            reasons.append("cad_shape_area_mismatch")
    if (
        float(metrics["cad_seed_overlap_smaller"])
        < DEFAULT_MINIMUM_ALIGNED_CAD_OVERLAP_SMALLER
    ):
        reasons.append("aligned_cad_direct_overlap_below_threshold")
    if other_part_seeds is not None:
        others = np.asarray(other_part_seeds, dtype=bool)
        if others.shape != selected.shape:
            raise Sam3RegionError("neighbor CAD masks differ from candidate size")
        other_intersection = int(np.count_nonzero(selected & others))
        other_fraction = other_intersection / max(
            1, int(np.count_nonzero(selected))
        )
        metrics["other_part_intersection_pixels"] = other_intersection
        metrics["other_part_overlap_fraction"] = other_fraction
        if other_fraction > DEFAULT_MAXIMUM_OTHER_PART_OVERLAP_FRACTION:
            reasons.append("candidate_claims_neighboring_cad_parts")
    # The visible renderer mask may be a small fraction of an occluded mesh.
    # Its area is a valid bound only when no amodal shape is available.  With
    # an amodal template, candidate/amodal area and visible recall above are
    # the authoritative, non-conflicting constraints.
    if aligned_amodal is None:
        area_ratio = float(metrics["candidate_to_cad_seed_area_ratio"])
        if not (
            1.0 / DEFAULT_MAXIMUM_ALIGNED_CAD_AREA_RATIO
            <= area_ratio
            <= DEFAULT_MAXIMUM_ALIGNED_CAD_AREA_RATIO
        ):
            reasons.append("aligned_cad_direct_area_mismatch")
    result = {
        "accepted": not reasons,
        "reason_codes": reasons,
        "shape_metrics": {
            key: round(float(value), 8) if isinstance(value, float) else value
            for key, value in metrics.items()
        },
        "aligned_cad_support_bound": support_audit,
        "shape_authority": (
            "isolated_mesh_amodal_plus_whole_assembly_visibility"
            if aligned_amodal is not None
            else "whole_assembly_visible_part_id_only"
        ),
    }
    return (selected if not reasons else None), result


def _selected_point_model_score(audit: Mapping[str, Any]) -> float:
    selected_index = audit.get("selected_candidate_index")
    if not isinstance(selected_index, int):
        selected_index = audit.get("preview_candidate_index")
    candidates = audit.get("candidates")
    if not isinstance(selected_index, int) or not isinstance(candidates, list):
        return 0.0
    selected = next(
        (
            row
            for row in candidates
            if isinstance(row, Mapping)
            and row.get("candidate_index") == selected_index
        ),
        None,
    )
    value = selected.get("model_score") if isinstance(selected, Mapping) else None
    return float(value) if isinstance(value, (int, float)) else 0.0


def _segment_shape_guided_points(
    *,
    model: Any,
    image_state: Mapping[str, Any],
    image: Any,
    cad_seed: Any,
    coarse_candidate: Any,
    box: Sequence[int],
    minimum_model_score: float,
    minimum_prompt_overlap: float,
    maximum_image_fraction: float,
    minimum_mask_pixels: int,
    cad_amodal: Any | None = None,
    other_part_seeds: Any | None = None,
    view_shared_alignment: Mapping[str, Any] | None = None,
) -> tuple[Any | None, dict[str, Any]]:
    """A/B sparse points against a dense CAD mask prompt, without human input."""

    import cv2
    import numpy as np

    translation = (view_shared_alignment or {}).get(
        "translation_xy_pixels", [0.0, 0.0]
    )
    aligned_other_part_seeds = (
        _translate_binary_mask(other_part_seeds, translation)
        if other_part_seeds is not None
        else None
    )
    click_set, prompt_audit = _shape_seed_click_set(
        cad_seed,
        coarse_candidate,
        box=box,
        width=image.width,
        height=image.height,
        other_part_seeds=aligned_other_part_seeds,
        view_shared_alignment=view_shared_alignment,
    )
    matrix = np.asarray(
        [
            [1.0, 0.0, prompt_audit["translation_xy_pixels"][0]],
            [0.0, 1.0, prompt_audit["translation_xy_pixels"][1]],
        ],
        dtype=np.float32,
    )
    aligned_seed = cv2.warpAffine(
        np.asarray(cad_seed, dtype=np.uint8),
        matrix,
        (image.width, image.height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ) > 0
    aligned_amodal = (
        cv2.warpAffine(
            np.asarray(cad_amodal, dtype=np.uint8),
            matrix,
            (image.width, image.height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        > 0
        if cad_amodal is not None
        else None
    )

    sparse_selected, sparse_point_audit = _segment_points(
        model=model,
        image_state=image_state,
        image=image,
        click_set=click_set,
        minimum_model_score=minimum_model_score,
        minimum_prompt_overlap=minimum_prompt_overlap,
        maximum_image_fraction=maximum_image_fraction,
        minimum_mask_pixels=minimum_mask_pixels,
    )
    sparse_final = None
    sparse_shape_audit: dict[str, Any] | None = None
    if sparse_selected is not None:
        sparse_final, sparse_shape_audit = _evaluate_shape_guided_candidate(
            sparse_selected,
            box=box,
            width=image.width,
            height=image.height,
            aligned_seed=aligned_seed,
            aligned_amodal=aligned_amodal,
            minimum_mask_pixels=minimum_mask_pixels,
            other_part_seeds=aligned_other_part_seeds,
        )

    dense_selected, dense_point_audit = _segment_dense_mask_points(
        model=model,
        image_state=image_state,
        image=image,
        click_set=click_set,
        aligned_cad_seed=aligned_seed,
        minimum_model_score=minimum_model_score,
        minimum_prompt_overlap=minimum_prompt_overlap,
        maximum_image_fraction=maximum_image_fraction,
        minimum_mask_pixels=minimum_mask_pixels,
    )
    dense_final = None
    dense_shape_audit: dict[str, Any] | None = None
    if dense_selected is not None:
        dense_final, dense_shape_audit = _evaluate_shape_guided_candidate(
            dense_selected,
            box=box,
            width=image.width,
            height=image.height,
            aligned_seed=aligned_seed,
            aligned_amodal=aligned_amodal,
            minimum_mask_pixels=minimum_mask_pixels,
            other_part_seeds=aligned_other_part_seeds,
        )

    # Preserve a previously safe sparse result.  Dense guidance is allowed to
    # rescue only a sparse failure, so this experiment cannot regress accepted
    # baseline masks by silently changing their source.
    if sparse_final is not None:
        selected = sparse_final
        selected_prompt_mode = "sparse_positive_negative_points"
        selected_shape_audit = sparse_shape_audit
        selected_model_score = _selected_point_model_score(sparse_point_audit)
    elif dense_final is not None:
        selected = dense_final
        selected_prompt_mode = "dense_cad_mask_plus_positive_negative_points"
        selected_shape_audit = dense_shape_audit
        selected_model_score = _selected_point_model_score(dense_point_audit)
    else:
        selected = None
        selected_prompt_mode = "none"
        selected_shape_audit = None
        selected_model_score = 0.0

    result: dict[str, Any] = {
        "prompt_mode": "automatic_cad_sparse_and_dense_ab",
        "selected_prompt_mode": selected_prompt_mode,
        "selected_model_score": selected_model_score,
        "click_set": click_set,
        "prompt_audit": prompt_audit,
        # Keep the legacy key for downstream readers while adding a complete
        # branch audit for the new dense experiment.
        "sam3_point_audit": sparse_point_audit,
        "sparse_branch": {
            "point_audit": sparse_point_audit,
            "shape_gate": sparse_shape_audit,
            "accepted": sparse_final is not None,
        },
        "dense_branch": {
            "point_audit": dense_point_audit,
            "shape_gate": dense_shape_audit,
            "accepted": dense_final is not None,
        },
        "accepted": selected is not None,
        "reason_codes": [],
    }
    if selected is None:
        if sparse_selected is None:
            result["reason_codes"].append("no_sparse_point_prediction_accepted")
        elif sparse_shape_audit is not None:
            result["reason_codes"].extend(
                f"sparse:{reason}" for reason in sparse_shape_audit["reason_codes"]
            )
        if dense_selected is None:
            result["reason_codes"].append("no_dense_mask_prediction_accepted")
        elif dense_shape_audit is not None:
            result["reason_codes"].extend(
                f"dense:{reason}" for reason in dense_shape_audit["reason_codes"]
            )
        return None, result
    assert selected_shape_audit is not None
    result.update(selected_shape_audit)
    result["accepted"] = True
    return selected, result


def _segment_ordered_points(
    *,
    model: Any,
    image_state: Mapping[str, Any],
    image: Any,
    click_set: Mapping[str, Any],
    minimum_prompt_overlap: float,
    maximum_image_fraction: float,
    minimum_mask_pixels: int,
) -> tuple[Any | None, dict[str, Any]]:
    """Replay v3 events using the same logits refinement path as the UI."""

    try:
        mask, _logits, audit = replay_ordered_click_set(
            model=model,
            image_state=image_state,
            image=image,
            click_set=click_set,
            minimum_prompt_agreement=minimum_prompt_overlap,
            maximum_image_fraction=maximum_image_fraction,
            minimum_mask_pixels=minimum_mask_pixels,
        )
    except ValueError as exc:
        raise Sam3RegionError(str(exc)) from exc
    return mask if audit.get("accepted") is True else None, audit


def _median(values: Sequence[float]) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return 0.5 * (ordered[midpoint - 1] + ordered[midpoint])


def _mask_quality(record: Mapping[str, Any]) -> dict[str, float]:
    """Summarize only selected, accepted SAM candidates for arbitration."""

    box_audits = record.get("box_audits")
    selected: list[Mapping[str, Any]] = []
    if isinstance(box_audits, list):
        for box_audit in box_audits:
            if (
                not isinstance(box_audit, Mapping)
                or box_audit.get("accepted") is not True
            ):
                continue
            selected_index = box_audit.get("selected_candidate_index")
            candidates = box_audit.get("candidates")
            if (
                not isinstance(selected_index, int)
                or isinstance(selected_index, bool)
                or not isinstance(candidates, list)
            ):
                continue
            match = next(
                (
                    candidate
                    for candidate in candidates
                    if isinstance(candidate, Mapping)
                    and candidate.get("candidate_index") == selected_index
                    and candidate.get("accepted") is True
                ),
                None,
            )
            if match is not None:
                selected.append(match)
    box_count = record.get("box_count")
    accepted_box_count = record.get("accepted_box_count")
    ratio = (
        float(accepted_box_count) / float(box_count)
        if isinstance(box_count, int)
        and not isinstance(box_count, bool)
        and box_count > 0
        and isinstance(accepted_box_count, int)
        and not isinstance(accepted_box_count, bool)
        else 0.0
    )
    return {
        "accepted_box_ratio": ratio,
        "median_model_score": _median(
            [
                float(item["model_score"])
                for item in selected
                if isinstance(item.get("model_score"), (int, float))
                and not isinstance(item.get("model_score"), bool)
            ]
        ),
        "median_prompt_overlap": _median(
            [
                float(item["overlap_smaller"])
                for item in selected
                if isinstance(item.get("overlap_smaller"), (int, float))
                and not isinstance(item.get("overlap_smaller"), bool)
            ]
        ),
        "median_mask_precision": _median(
            [
                float(item["mask_precision"])
                for item in selected
                if isinstance(item.get("mask_precision"), (int, float))
                and not isinstance(item.get("mask_precision"), bool)
            ]
        ),
        "median_cad_shape_iou": _median(
            [
                float(item["cad_shape_iou"])
                for item in selected
                if isinstance(item.get("cad_shape_iou"), (int, float))
                and not isinstance(item.get("cad_shape_iou"), bool)
            ]
        ),
    }


def _arbitrate_view_group_masks(
    records: list[dict[str, Any]],
    *,
    minimum_intersection_pixels: int,
    near_duplicate_iou: float = CROSS_GROUP_NEAR_DUPLICATE_IOU,
) -> list[dict[str, Any]]:
    """Reject lower-quality near-duplicate material masks per source view.

    Containment alone is intentionally not a conflict: a small bolt can be
    valid inside the bounding region of a larger plate.  Only masks with very
    high symmetric IoU enter connected-component arbitration.  The winner is
    independent of request order and selected from existing SAM evidence.
    ``__foreground__`` is a whole-object registration mask, not a material
    group, and is excluded.
    """

    import numpy as np

    eligible = [
        index
        for index, record in enumerate(records)
        if record.get("accepted") is True
        and record.get("group_id") != "__foreground__"
        and isinstance(record.get("_mask"), np.ndarray)
    ]
    adjacency = {index: set() for index in eligible}
    pair_audits: dict[int, list[dict[str, Any]]] = {index: [] for index in eligible}
    for offset, left_index in enumerate(eligible):
        left_mask = np.asarray(records[left_index]["_mask"], dtype=bool)
        left_pixels = int(np.count_nonzero(left_mask))
        for right_index in eligible[offset + 1 :]:
            right_mask = np.asarray(records[right_index]["_mask"], dtype=bool)
            right_pixels = int(np.count_nonzero(right_mask))
            intersection = int(np.count_nonzero(left_mask & right_mask))
            union = left_pixels + right_pixels - intersection
            iou = intersection / max(1, union)
            near_duplicate = (
                intersection >= minimum_intersection_pixels
                and iou >= near_duplicate_iou
            )
            pair = {
                "other_group_id": records[right_index]["group_id"],
                "intersection_pixels": intersection,
                "iou": round(iou, 8),
                "near_duplicate": near_duplicate,
            }
            pair_audits[left_index].append(pair)
            pair_audits[right_index].append(
                {
                    **pair,
                    "other_group_id": records[left_index]["group_id"],
                }
            )
            if near_duplicate:
                adjacency[left_index].add(right_index)
                adjacency[right_index].add(left_index)

    quality = {index: _mask_quality(records[index]) for index in eligible}
    components: list[list[int]] = []
    unseen = set(eligible)
    while unseen:
        seed = min(unseen, key=lambda index: str(records[index]["group_id"]))
        stack = [seed]
        component: list[int] = []
        unseen.remove(seed)
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbour in sorted(
                adjacency[current],
                key=lambda index: str(records[index]["group_id"]),
                reverse=True,
            ):
                if neighbour in unseen:
                    unseen.remove(neighbour)
                    stack.append(neighbour)
        components.append(component)

    for component in components:
        winner = sorted(
            component,
            key=lambda index: (
                -quality[index]["accepted_box_ratio"],
                -quality[index]["median_cad_shape_iou"],
                -quality[index]["median_model_score"],
                -quality[index]["median_prompt_overlap"],
                -quality[index]["median_mask_precision"],
                str(records[index]["group_id"]),
            ),
        )[0]
        component_groups = sorted(
            str(records[index]["group_id"]) for index in component
        )
        winner_group = str(records[winner]["group_id"])
        for index in component:
            record = records[index]
            record["cross_group_arbitration"] = {
                "schema_version": CROSS_GROUP_ARBITRATION_SCHEMA_VERSION,
                "eligible": True,
                "near_duplicate_iou_threshold": near_duplicate_iou,
                "minimum_intersection_pixels": minimum_intersection_pixels,
                "component_group_ids": component_groups,
                "winner_group_id": winner_group,
                "quality": {
                    key: round(value, 8) for key, value in quality[index].items()
                },
                "pairwise": sorted(
                    pair_audits[index],
                    key=lambda item: str(item["other_group_id"]),
                ),
            }
            if len(component) > 1 and index != winner:
                record["pre_arbitration_mask_pixels"] = record["mask_pixels"]
                record["pre_arbitration_image_fraction"] = record["image_fraction"]
                record["accepted"] = False
                record["reason_codes"].append("cross_group_near_duplicate_loser")
                record["mask_pixels"] = 0
                record["image_fraction"] = 0.0
                record["_mask"] = None

    for index, record in enumerate(records):
        if index not in pair_audits:
            record["cross_group_arbitration"] = {
                "schema_version": CROSS_GROUP_ARBITRATION_SCHEMA_VERSION,
                "eligible": False,
                "reason": (
                    "whole_asset_foreground_excluded"
                    if record.get("group_id") == "__foreground__"
                    else "self_gate_rejected"
                ),
            }
    return records


def result_policy(
    *,
    minimum_model_score: float,
    minimum_prompt_overlap: float,
    maximum_image_fraction: float,
    minimum_mask_pixels: int,
    human_interactive_requested: bool,
    automatic_shape_interactive_requested: bool,
    ordered_interaction_requested: bool,
) -> dict[str, Any]:
    """Build the single producer/validator SAM3 result policy contract."""

    policy: dict[str, Any] = {
        "minimum_model_score": minimum_model_score,
        "minimum_prompt_overlap": minimum_prompt_overlap,
        "maximum_image_fraction": maximum_image_fraction,
        "minimum_mask_pixels": minimum_mask_pixels,
        "minimum_cad_shape_seed_pixels": DEFAULT_MINIMUM_CAD_SHAPE_SEED_PIXELS,
        "minimum_cad_shape_iou": DEFAULT_MINIMUM_CAD_SHAPE_IOU,
        "minimum_cad_shape_area_agreement": (
            DEFAULT_MINIMUM_CAD_SHAPE_AREA_AGREEMENT
        ),
        "minimum_amodal_candidate_precision": (
            DEFAULT_MINIMUM_AMODAL_CANDIDATE_PRECISION
        ),
        "minimum_amodal_completion_shape_iou": (
            DEFAULT_MINIMUM_AMODAL_COMPLETION_SHAPE_IOU
        ),
        "maximum_candidate_to_amodal_area_ratio": (
            DEFAULT_MAXIMUM_CANDIDATE_TO_AMODAL_AREA_RATIO
        ),
        "minimum_visible_cad_seed_recall": DEFAULT_MINIMUM_VISIBLE_CAD_SEED_RECALL,
        "dual_template_roles": {
            "whole_assembly_part_id": "location_visibility_and_occlusion_prior",
            "isolated_mesh_projection": "complete_shape_prior",
        },
        "cad_shape_candidate_policy": (
            "same_view_rendered_part_template_view_shared_translation_direct_support"
        ),
        "per_mesh_pose_change_allowed": False,
        "maximum_view_shared_translation_pixels": (
            DEFAULT_MAXIMUM_VIEW_SHARED_TRANSLATION_PIXELS
        ),
        "maximum_other_part_overlap_fraction": (
            DEFAULT_MAXIMUM_OTHER_PART_OVERLAP_FRACTION
        ),
        "maximum_final_sam_to_cad_area_ratio": (
            DEFAULT_MAXIMUM_FINAL_SAM_TO_CAD_AREA_RATIO
        ),
        "maximum_neighbor_negative_target_pixels": (
            DEFAULT_MAXIMUM_NEIGHBOR_NEGATIVE_TARGET_PIXELS
        ),
        "rejected_mask_policy": "fail_closed_no_mask_evidence",
        "cross_group_arbitration_schema": CROSS_GROUP_ARBITRATION_SCHEMA_VERSION,
        "cross_group_near_duplicate_iou": CROSS_GROUP_NEAR_DUPLICATE_IOU,
        "cross_group_minimum_intersection_pixels": minimum_mask_pixels,
        "cross_group_foreground_policy": "exclude_whole_asset_foreground",
    }
    if human_interactive_requested:
        policy["human_point_model_score_policy"] = (
            "advisory_when_human_confirmed"
        )
    if automatic_shape_interactive_requested:
        policy["automatic_shape_point_refinement"] = (
            "always_run_same_view_cad_shape_positive_negative_points"
        )
    if ordered_interaction_requested:
        policy["human_point_replay_policy"] = (
            "first_multimask_then_previous_logits_single_mask"
        )
    return policy


def _automatic_cad_shape_contract_accepted(
    box_audits: Sequence[Mapping[str, Any]],
    view_shared_alignment: Mapping[str, Any],
) -> bool:
    """Return whether one automatic box satisfies the downstream CAD contract.

    A detector box is only a coarse localization hint.  For an automatic
    CAD-guided request, it must not make the whole record successful unless
    SAM3 also produced one accepted shape refinement bound to the shared
    whole-assembly camera alignment.  Human click requests and ordinary text
    box requests do not use this contract.
    """

    translation = view_shared_alignment.get("translation_xy_pixels")
    if (
        not isinstance(translation, Sequence)
        or isinstance(translation, (str, bytes))
        or len(translation) != 2
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in translation
        )
        or view_shared_alignment.get("part_specific_translation_allowed") is not False
        or len(box_audits) != 1
    ):
        return False

    box_audit = box_audits[0]
    selected_index = box_audit.get("selected_candidate_index")
    candidates = box_audit.get("candidates")
    refinement = box_audit.get("shape_point_refinement")
    prompt_audit = (
        refinement.get("prompt_audit") if isinstance(refinement, Mapping) else None
    )
    selected = (
        next(
            (
                item
                for item in candidates
                if isinstance(item, Mapping)
                and item.get("candidate_index") == selected_index
                and item.get("accepted") is True
            ),
            None,
        )
        if isinstance(candidates, list)
        and isinstance(selected_index, int)
        and not isinstance(selected_index, bool)
        else None
    )
    return bool(
        box_audit.get("accepted") is True
        and selected is not None
        and isinstance(refinement, Mapping)
        and refinement.get("accepted") is True
        and isinstance(prompt_audit, Mapping)
        and prompt_audit.get("per_mesh_pose_change_allowed") is False
        and prompt_audit.get("part_specific_translation_allowed") is False
        and prompt_audit.get("part_local_translation_xy_pixels") == [0.0, 0.0]
        and prompt_audit.get("translation_xy_pixels")
        == [float(translation[0]), float(translation[1])]
    )


def run(
    *,
    request_path: Path,
    repository: Path,
    checkpoint: Path,
    output_dir: Path,
    device: str,
    minimum_model_score: float,
    minimum_prompt_overlap: float,
    maximum_image_fraction: float,
    minimum_mask_pixels: int,
    seed: int = DEFAULT_INFERENCE_SEED,
) -> dict[str, Any]:
    import numpy as np
    from PIL import Image, ImageOps

    raw_request_document = _read_json(request_path, "SAM3 region request")
    request, source_paths = _load_request(request_path)
    ordered_interaction_requested = (
        request.get("schema_version") == ORDERED_POINT_SCHEMA_VERSION
    )
    repository = repository.expanduser().resolve(strict=True)
    checkpoint = checkpoint.expanduser().resolve(strict=True)
    if not repository.is_dir():
        raise Sam3RegionError(f"SAM3 repository is not a directory: {repository}")
    if not checkpoint.is_file():
        raise Sam3RegionError(f"SAM3 checkpoint is not a file: {checkpoint}")
    if device not in {"cuda", "cpu"}:
        raise Sam3RegionError("device must be 'cuda' or 'cpu'")
    _seed_inference(seed)

    sys.path.insert(0, str(repository))
    try:
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model
    except ImportError as exc:
        raise Sam3RegionError(
            f"unable to import SAM3 from repository {repository}: {exc}"
        ) from exc

    human_interactive_requested = any(
        bool(region.get("click_sets")) for region in request["regions"]
    )
    automatic_shape_interactive_requested = any(
        isinstance(region.get("cad_projection_seed"), Mapping)
        or isinstance(region.get("cad_amodal_template"), Mapping)
        for region in request["regions"]
    )
    interactive_requested = (
        human_interactive_requested or automatic_shape_interactive_requested
    )
    model = build_sam3_image_model(
        checkpoint_path=str(checkpoint),
        load_from_HF=False,
        device=device,
        eval_mode=True,
        enable_inst_interactivity=interactive_requested,
    )
    processor = Sam3Processor(
        model,
        device=device,
        confidence_threshold=minimum_model_score,
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    masks_dir = output_dir / "masks"
    masks_dir.mkdir(parents=True)

    regions_by_view: dict[str, list[dict[str, Any]]] = {}
    for region in request["regions"]:
        regions_by_view.setdefault(str(region["view_id"]), []).append(region)
    records: list[dict[str, Any]] = []
    total = len(request["regions"])
    current = 0
    for view_id in sorted(regions_by_view):
        with Image.open(source_paths[view_id]) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
        image_state = processor.set_image(image)
        source_record = next(
            row for row in request["source_views"] if row["id"] == view_id
        )
        foreground_array = None
        foreground_path = source_record.get("whole_workpiece_foreground")
        if isinstance(foreground_path, str):
            with Image.open(Path(foreground_path)) as foreground_opened:
                foreground_array = (
                    np.asarray(foreground_opened.convert("L"), dtype=np.uint8) >= 128
                )
        view_cad_seeds: dict[str, Any] = {}
        view_cad_amodal: dict[str, Any] = {}
        for seed_region in regions_by_view[view_id]:
            seed_record = seed_region.get("cad_projection_seed")
            if not isinstance(seed_record, Mapping):
                continue
            with Image.open(Path(str(seed_record["path"]))) as seed_opened:
                seed_array = (
                    np.asarray(seed_opened.convert("L"), dtype=np.uint8) >= 128
                )
            if seed_array.shape != (image.height, image.width):
                raise Sam3RegionError(
                    f"CAD seed dimensions for {view_id}/"
                    f"{seed_region['group_id']} differ from the source image"
                )
            view_cad_seeds[str(seed_region["group_id"])] = seed_array
            amodal_record = seed_region.get("cad_amodal_template")
            if isinstance(amodal_record, Mapping):
                with Image.open(Path(str(amodal_record["path"]))) as amodal_opened:
                    amodal_array = (
                        np.asarray(amodal_opened.convert("L"), dtype=np.uint8) >= 128
                    )
                if amodal_array.shape != (image.height, image.width):
                    raise Sam3RegionError(
                        f"CAD amodal template dimensions for {view_id}/"
                        f"{seed_region['group_id']} differ from the source image"
                    )
                view_cad_amodal[str(seed_region["group_id"])] = amodal_array
        view_shared_alignment = _estimate_view_shared_translation(
            view_cad_seeds,
            foreground_array,
        )
        shared_translation_xy = view_shared_alignment["translation_xy_pixels"]
        view_aligned_cad_seeds = {
            group_id: _translate_binary_mask(mask, shared_translation_xy)
            for group_id, mask in view_cad_seeds.items()
        }
        view_aligned_cad_amodal = {
            group_id: _translate_binary_mask(mask, shared_translation_xy)
            for group_id, mask in view_cad_amodal.items()
        }
        view_records: list[dict[str, Any]] = []
        for region in sorted(
            regions_by_view[view_id], key=lambda item: str(item["group_id"])
        ):
            current += 1
            _emit_progress(
                stage="segment",
                state="update",
                current=current,
                total=total,
                unit="regions",
                detail=f"SAM3 {view_id}/{region['group_id']}",
            )
            union = np.zeros((image.height, image.width), dtype=bool)
            box_audits: list[dict[str, Any]] = []
            point_set_audits: list[dict[str, Any]] = []
            click_sets = region.get("click_sets")
            cad_seed_array = None
            cad_seed_record = region.get("cad_projection_seed")
            cad_amodal_record = region.get("cad_amodal_template")
            cad_amodal_array = None
            aligned_cad_seed_array = None
            aligned_cad_amodal_array = None
            if isinstance(cad_seed_record, Mapping):
                cad_seed_array = view_cad_seeds[str(region["group_id"])]
                aligned_cad_seed_array = view_aligned_cad_seeds[
                    str(region["group_id"])
                ]
            if isinstance(cad_amodal_record, Mapping):
                cad_amodal_array = view_cad_amodal[str(region["group_id"])]
                aligned_cad_amodal_array = view_aligned_cad_amodal[
                    str(region["group_id"])
                ]
            if click_sets:
                if model.inst_interactive_predictor is None:
                    raise Sam3RegionError(
                        "SAM3 point request was run without instance interactivity"
                    )
                for set_index, click_set in enumerate(click_sets):
                    if ordered_interaction_requested:
                        selected, audit = _segment_ordered_points(
                            model=model,
                            image_state=image_state,
                            image=image,
                            click_set=click_set,
                            minimum_prompt_overlap=minimum_prompt_overlap,
                            maximum_image_fraction=maximum_image_fraction,
                            minimum_mask_pixels=minimum_mask_pixels,
                        )
                    else:
                        selected, audit = _segment_points(
                            model=model,
                            image_state=image_state,
                            image=image,
                            click_set=click_set,
                            # Human approval and exact confirmed-mask replay replace
                            # the detector-score veto for interactive point sets.
                            minimum_model_score=0.0,
                            minimum_prompt_overlap=minimum_prompt_overlap,
                            maximum_image_fraction=maximum_image_fraction,
                            minimum_mask_pixels=minimum_mask_pixels,
                        )
                    point_set_audits.append(
                        {
                            "click_set_index": set_index,
                            "positive_points": click_set["positive_points"],
                            "negative_points": click_set["negative_points"],
                            **(
                                {
                                    "events": click_set["events"],
                                    "initial_candidate_index": click_set[
                                        "initial_candidate_index"
                                    ],
                                }
                                if ordered_interaction_requested
                                else {}
                            ),
                            **audit,
                        }
                    )
                    if selected is not None:
                        union |= selected
            else:
                for box_index, box in enumerate(region["boxes"]):
                    initial_selected, initial_shape_audit = _segment_box(
                        processor=processor,
                        image=image,
                        image_state=image_state,
                        prompt=str(region["prompt"]),
                        box=box,
                        minimum_model_score=minimum_model_score,
                        minimum_prompt_overlap=minimum_prompt_overlap,
                        maximum_image_fraction=maximum_image_fraction,
                        minimum_mask_pixels=minimum_mask_pixels,
                        cad_seed=aligned_cad_seed_array,
                        cad_amodal=aligned_cad_amodal_array,
                    )
                    selected = initial_selected
                    audit = initial_shape_audit
                    if (
                        cad_seed_array is not None
                        and int(np.count_nonzero(cad_seed_array))
                        >= DEFAULT_MINIMUM_CAD_SHAPE_SEED_PIXELS
                    ):
                        if initial_selected is not None:
                            coarse_selected = initial_selected
                            coarse_audit = {
                                "accepted": True,
                                "source": "initial_cad_shape_box_candidate",
                                "reused_initial_candidate": True,
                            }
                        else:
                            coarse_selected, coarse_audit = _segment_box(
                                processor=processor,
                                image=image,
                                image_state=image_state,
                                prompt=str(region["prompt"]),
                                box=box,
                                minimum_model_score=minimum_model_score,
                                minimum_prompt_overlap=minimum_prompt_overlap,
                                maximum_image_fraction=maximum_image_fraction,
                                minimum_mask_pixels=minimum_mask_pixels,
                                cad_seed=None,
                            )
                        if coarse_selected is None and cad_amodal_array is not None:
                            # A tiny/thin part may have no generic detector
                            # proposal at all.  Start the dense experiment from
                            # the registered modal mask with zero residual; the
                            # decoded photo mask must still pass every strict
                            # visible/amodal gate below.
                            coarse_selected = aligned_cad_seed_array
                            coarse_audit = {
                                **coarse_audit,
                                "accepted": False,
                                "dense_mask_bootstrap": True,
                                "source": "registered_visible_cad_seed",
                                "per_mesh_pose_change_allowed": False,
                            }
                        shape_point_audit: dict[str, Any] | None = None
                        if coarse_selected is not None:
                            other_part_seeds = np.zeros_like(cad_seed_array)
                            for other_group_id, other_seed in view_cad_seeds.items():
                                if other_group_id != str(region["group_id"]):
                                    other_part_seeds |= other_seed
                            selected, shape_point_audit = _segment_shape_guided_points(
                                model=model,
                                image_state=image_state,
                                image=image,
                                cad_seed=cad_seed_array,
                                cad_amodal=cad_amodal_array,
                                coarse_candidate=coarse_selected,
                                box=box,
                                minimum_model_score=minimum_model_score,
                                minimum_prompt_overlap=minimum_prompt_overlap,
                                maximum_image_fraction=maximum_image_fraction,
                                minimum_mask_pixels=minimum_mask_pixels,
                                other_part_seeds=other_part_seeds,
                                view_shared_alignment=view_shared_alignment,
                            )
                        if selected is not None and shape_point_audit is not None:
                            candidate_index = len(initial_shape_audit["candidates"])
                            rescued_candidate = {
                                "candidate_index": candidate_index,
                                "candidate_source": (
                                    "automatic_cad_sparse_or_dense_refinement"
                                ),
                                "prompt_mode": shape_point_audit[
                                    "selected_prompt_mode"
                                ],
                                "model_score": float(
                                    shape_point_audit.get("selected_model_score", 0.0)
                                ),
                                **shape_point_audit["shape_metrics"],
                                "accepted": True,
                                "reason_codes": [],
                            }
                            audit = {
                                "accepted": True,
                                "selected_candidate_index": candidate_index,
                                "candidates": [
                                    *initial_shape_audit["candidates"],
                                    rescued_candidate,
                                ],
                                "initial_box_shape_audit": initial_shape_audit,
                                "coarse_box_audit": coarse_audit,
                                "shape_point_refinement": shape_point_audit,
                                "selection_policy": (
                                    "modal_visibility_plus_amodal_mesh_shape_guides_sam3"
                                    if cad_amodal_array is not None
                                    else "same_view_cad_template_always_guides_sam3"
                                ),
                            }
                        else:
                            # The final hybrid stage will still bind any fallback
                            # to the aligned CAD support band.  Reuse only a box
                            # candidate that already passed the CAD-shape gate;
                            # an unconstrained coarse candidate remains rejected.
                            selected = initial_selected
                            audit = {
                                **initial_shape_audit,
                                "coarse_box_audit": coarse_audit,
                                "shape_point_refinement": shape_point_audit,
                                "selection_policy": (
                                    "aligned_cad_bounded_shape_candidate_fallback"
                                    if initial_selected is not None
                                    else "same_view_cad_template_required_fail_closed"
                                ),
                            }
                    box_audits.append({"box_index": box_index, "box": box, **audit})
                    if selected is not None:
                        union |= selected
            accepted_boxes = sum(bool(item["accepted"]) for item in box_audits)
            accepted_point_sets = sum(
                bool(item["accepted"]) for item in point_set_audits
            )
            confirmed_mask_audit: dict[str, Any] | None = None
            if click_sets:
                confirmed_mask = region["confirmed_mask"]
                confirmed_path = Path(str(confirmed_mask["path"]))
                with Image.open(confirmed_path) as confirmed_opened:
                    confirmed_array = (
                        np.asarray(confirmed_opened.convert("L"), dtype=np.uint8) > 0
                    )
                if confirmed_array.shape != union.shape:
                    raise Sam3RegionError(
                        "human-confirmed SAM3 mask dimensions differ from source image"
                    )
                intersection = int(np.count_nonzero(union & confirmed_array))
                union_pixels_before_confirmation = int(np.count_nonzero(union))
                confirmed_pixels = int(np.count_nonzero(confirmed_array))
                reproduction_iou = intersection / max(
                    1,
                    union_pixels_before_confirmation + confirmed_pixels - intersection,
                )
                reproduction_precision = intersection / max(
                    1, union_pixels_before_confirmation
                )
                reproduction_recall = intersection / max(1, confirmed_pixels)
                strict_reproduction = (
                    reproduction_iou >= CONFIRMED_MASK_STRICT_MINIMUM_IOU
                )
                symmetric_reproduction = (
                    reproduction_iou >= CONFIRMED_MASK_SYMMETRIC_MINIMUM_IOU
                )
                bounded_reproduction = (
                    reproduction_precision >= CONFIRMED_MASK_BOUNDED_MINIMUM_PRECISION
                    and reproduction_recall >= CONFIRMED_MASK_BOUNDED_MINIMUM_RECALL
                )
                reproduction_accepted = (
                    strict_reproduction
                    or symmetric_reproduction
                    or bounded_reproduction
                )
                confirmed_mask_audit = {
                    "path": str(confirmed_path),
                    "sha256": _sha256_file(confirmed_path),
                    "confirmed_mask_pixels": confirmed_pixels,
                    "rerun_mask_pixels": union_pixels_before_confirmation,
                    "intersection_pixels": intersection,
                    "reproduction_iou": round(reproduction_iou, 8),
                    "minimum_reproduction_iou": (CONFIRMED_MASK_STRICT_MINIMUM_IOU),
                    "symmetric_minimum_reproduction_iou": (
                        CONFIRMED_MASK_SYMMETRIC_MINIMUM_IOU
                    ),
                    "reproduction_precision": round(reproduction_precision, 8),
                    "reproduction_recall": round(reproduction_recall, 8),
                    "bounded_minimum_precision": (
                        CONFIRMED_MASK_BOUNDED_MINIMUM_PRECISION
                    ),
                    "bounded_minimum_recall": (CONFIRMED_MASK_BOUNDED_MINIMUM_RECALL),
                    "acceptance_mode": (
                        "strict_iou"
                        if strict_reproduction
                        else "symmetric_boundary_drift"
                        if symmetric_reproduction
                        else "bounded_human_confirmed"
                        if bounded_reproduction
                        else "rejected"
                    ),
                    "accepted": reproduction_accepted,
                    "authoritative_output": "human_confirmed_mask",
                }
                # The user-approved pixels are authoritative.  Re-running the
                # exact clicks first verifies that the configured checkpoint
                # still reproduces them before the approved mask is imported.
                if confirmed_mask_audit["accepted"]:
                    union = confirmed_array
            union_pixels = int(np.count_nonzero(union))
            union_fraction = union_pixels / max(1, image.width * image.height)
            reasons: list[str] = []
            automatic_cad_shape_requested = (
                not click_sets and isinstance(cad_seed_record, Mapping)
            )
            automatic_cad_shape_contract_accepted = (
                _automatic_cad_shape_contract_accepted(
                    box_audits,
                    view_shared_alignment,
                )
                if automatic_cad_shape_requested
                else None
            )
            if click_sets and accepted_point_sets != len(click_sets):
                reasons.append("not_all_human_click_sets_reproduced")
            if click_sets and (
                confirmed_mask_audit is None
                or confirmed_mask_audit["accepted"] is not True
            ):
                reasons.append("confirmed_mask_reproduction_mismatch")
            if not click_sets and accepted_boxes == 0:
                reasons.append("no_box_prediction_accepted")
            if (
                automatic_cad_shape_requested
                and automatic_cad_shape_contract_accepted is not True
            ):
                reasons.append("no_shape_guided_refinement_accepted")
            if union_pixels < minimum_mask_pixels:
                reasons.append("union_mask_too_small")
            if union_fraction > maximum_image_fraction:
                reasons.append("union_mask_too_large")
            accepted = not reasons
            view_records.append(
                {
                    "view_id": view_id,
                    "group_id": region["group_id"],
                    "local_group_id": region.get("local_group_id"),
                    "prompt": region["prompt"],
                    "boxes": region["boxes"],
                    "prompt_mode": (
                        "human_ordered_incremental_points"
                        if ordered_interaction_requested and click_sets
                        else "human_interactive_points"
                        if click_sets
                        else "cad_shape_guided_box_then_automatic_points"
                        if automatic_cad_shape_contract_accepted is True
                        else "text_and_positive_boxes"
                    ),
                    "interaction_replay_mode": (
                        "ordered_events_previous_logits"
                        if ordered_interaction_requested and click_sets
                        else "unordered_single_call"
                        if click_sets
                        else None
                    ),
                    "event_count": (
                        sum(len(click_set["events"]) for click_set in click_sets)
                        if ordered_interaction_requested and click_sets
                        else 0
                    ),
                    "click_sets": click_sets or [],
                    "source_image": str(source_paths[view_id]),
                    "source_image_sha256": _sha256_file(source_paths[view_id]),
                    "view_shared_alignment": view_shared_alignment,
                    "automatic_cad_shape_contract_accepted": (
                        automatic_cad_shape_contract_accepted
                    ),
                    "accepted": accepted,
                    "reason_codes": reasons,
                    "accepted_box_count": accepted_boxes,
                    "box_count": len(region["boxes"]),
                    "accepted_point_set_count": accepted_point_sets,
                    "point_set_count": len(click_sets or []),
                    "mask_pixels": union_pixels,
                    "image_fraction": round(union_fraction, 8),
                    "mask": None,
                    "box_audits": box_audits,
                    "point_set_audits": point_set_audits,
                    "confirmed_mask_audit": confirmed_mask_audit,
                    "cad_projection_seed": (
                        {
                            **dict(cad_seed_record),
                            "selection_role": (
                                "rank_sam3_candidates_by_cad_seed_agreement"
                            ),
                        }
                        if isinstance(cad_seed_record, Mapping)
                        else None
                    ),
                    "cad_amodal_template": (
                        {
                            **dict(cad_amodal_record),
                            "selection_role": (
                                "complete_mesh_shape_prior_occlusion_aware_gate"
                            ),
                        }
                        if isinstance(cad_amodal_record, Mapping)
                        else None
                    ),
                    "_mask": union if accepted else None,
                }
            )
        _arbitrate_view_group_masks(
            view_records,
            minimum_intersection_pixels=minimum_mask_pixels,
        )
        safe_view = (
            "".join(
                character if character.isalnum() or character in "._-" else "_"
                for character in view_id
            ).strip("._")
            or "view"
        )
        for record in view_records:
            mask_array = record.pop("_mask")
            if record["accepted"]:
                if mask_array is None:
                    raise Sam3RegionError(
                        "accepted SAM3 record lost its mask during arbitration"
                    )
                safe_group = (
                    "".join(
                        character if character.isalnum() or character in "._-" else "_"
                        for character in str(record["group_id"])
                    ).strip("._")
                    or "group"
                )
                mask_path = masks_dir / f"{safe_view}__{safe_group}.png"
                Image.fromarray(np.asarray(mask_array, dtype=np.uint8) * 255).save(
                    mask_path
                )
                record["mask"] = {
                    "path": mask_path.relative_to(output_dir).as_posix(),
                    "sha256": _sha256_file(mask_path),
                }
            records.append(record)
        image.close()

    policy = result_policy(
        minimum_model_score=minimum_model_score,
        minimum_prompt_overlap=minimum_prompt_overlap,
        maximum_image_fraction=maximum_image_fraction,
        minimum_mask_pixels=minimum_mask_pixels,
        human_interactive_requested=human_interactive_requested,
        automatic_shape_interactive_requested=(
            automatic_shape_interactive_requested
        ),
        ordered_interaction_requested=ordered_interaction_requested,
    )
    unsigned: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "request": {
            "path": str(request_path.resolve()),
            "sha256": _sha256_file(request_path),
            "document_sha256": _canonical_sha256(raw_request_document),
        },
        "backend": {
            "name": "facebook_sam3",
            "repository": str(repository),
            "repository_revision": _git_revision(repository),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256_file(checkpoint),
            "device": device,
            "instance_interactivity_enabled": interactive_requested,
        },
        "policy": policy,
        "records": records,
        "summary": {
            "region_count": len(records),
            "accepted_region_count": sum(bool(item["accepted"]) for item in records),
            "rejected_region_count": sum(
                not bool(item["accepted"]) for item in records
            ),
        },
    }
    unsigned["policy"]["inference_seed"] = seed
    unsigned["policy"]["deterministic_algorithms"] = True
    result = {
        **unsigned,
        "integrity": {"result_sha256": _canonical_sha256(unsigned)},
    }
    result_path = output_dir / "manifest.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--minimum-model-score", type=float, default=DEFAULT_MINIMUM_MODEL_SCORE
    )
    parser.add_argument(
        "--minimum-prompt-overlap",
        type=float,
        default=DEFAULT_MINIMUM_PROMPT_OVERLAP,
    )
    parser.add_argument(
        "--maximum-image-fraction",
        type=float,
        default=DEFAULT_MAXIMUM_IMAGE_FRACTION,
    )
    parser.add_argument(
        "--minimum-mask-pixels", type=int, default=DEFAULT_MINIMUM_MASK_PIXELS
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_INFERENCE_SEED)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    request = args.request.expanduser().resolve(strict=True)
    minimum_model_score = _unit_float(args.minimum_model_score, "--minimum-model-score")
    minimum_prompt_overlap = _unit_float(
        args.minimum_prompt_overlap, "--minimum-prompt-overlap"
    )
    maximum_image_fraction = _unit_float(
        args.maximum_image_fraction, "--maximum-image-fraction"
    )
    minimum_mask_pixels = _positive_int(
        args.minimum_mask_pixels, "--minimum-mask-pixels"
    )
    result = run(
        request_path=request,
        repository=args.repository,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir.expanduser().resolve(),
        device=args.device,
        minimum_model_score=minimum_model_score,
        minimum_prompt_overlap=minimum_prompt_overlap,
        maximum_image_fraction=maximum_image_fraction,
        minimum_mask_pixels=minimum_mask_pixels,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "output": str(
                    (args.output_dir.expanduser().resolve() / "manifest.json")
                ),
                **result["summary"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
