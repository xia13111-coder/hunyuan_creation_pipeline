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
import json
import math
from pathlib import Path
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
CROSS_GROUP_NEAR_DUPLICATE_IOU = 0.90
CROSS_GROUP_ARBITRATION_SCHEMA_VERSION = (
    "sam3-cross-group-near-duplicate-arbitration/v1"
)
PROGRESS_PREFIX = "@@ASSET_PROGRESS "
PROGRESS_SCHEMA_VERSION = "asset-pipeline-progress/v1"


class Sam3RegionError(ValueError):
    """Raised when the SAM3 request or output violates the frozen contract."""


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
                if (
                    not isinstance(expected_digest, str)
                    or expected_digest != _sha256_file(seed_path)
                ):
                    raise Sam3RegionError(
                        f"regions[{index}].cad_projection_seed hash mismatch"
                    )
                raw["cad_projection_seed"] = {
                    **dict(cad_seed),
                    "path": str(seed_path),
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


def _candidate_metrics(
    mask: Any,
    *,
    box: Sequence[int],
    width: int,
    height: int,
    cad_seed: Any | None = None,
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
                "candidate_to_cad_seed_area_ratio": mask_pixels
                / max(1, seed_pixels),
            }
        )
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
    scores = scores_raw.detach().to("cpu").numpy().reshape(-1)
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim != 3 or len(masks) != len(scores):
        raise Sam3RegionError("SAM3 masks/scores have incompatible shapes")
    candidates: list[tuple[tuple[float, float, float, int], Any, dict[str, Any]]] = []
    audits: list[dict[str, Any]] = []
    for index, (mask, raw_score) in enumerate(zip(masks, scores)):
        score = float(raw_score)
        metrics = _candidate_metrics(
            mask,
            box=box,
            width=width,
            height=height,
            cad_seed=cad_seed,
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
        if not metrics["prompt_center_inside"] and metrics["prompt_coverage"] < 0.10:
            reasons.append("prompt_not_localized")
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
                    float(metrics["cad_seed_iou"]),
                    min(
                        float(metrics["cad_seed_precision"]),
                        float(metrics["cad_seed_recall"]),
                    ),
                    float(metrics["cad_seed_overlap_smaller"]),
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

    sys.path.insert(0, str(repository))
    try:
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model
    except ImportError as exc:
        raise Sam3RegionError(
            f"unable to import SAM3 from repository {repository}: {exc}"
        ) from exc

    interactive_requested = any(
        bool(region.get("click_sets")) for region in request["regions"]
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
            if isinstance(cad_seed_record, Mapping):
                seed_path = Path(str(cad_seed_record["path"]))
                with Image.open(seed_path) as seed_opened:
                    cad_seed_array = (
                        np.asarray(seed_opened.convert("L"), dtype=np.uint8)
                        >= 128
                    )
                if cad_seed_array.shape != (image.height, image.width):
                    raise Sam3RegionError(
                        f"CAD seed dimensions for {view_id}/"
                        f"{region['group_id']} differ from the source image"
                    )
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
                    selected, audit = _segment_box(
                        processor=processor,
                        image=image,
                        image_state=image_state,
                        prompt=str(region["prompt"]),
                        box=box,
                        minimum_model_score=minimum_model_score,
                        minimum_prompt_overlap=minimum_prompt_overlap,
                        maximum_image_fraction=maximum_image_fraction,
                        minimum_mask_pixels=minimum_mask_pixels,
                        cad_seed=cad_seed_array,
                    )
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
                bounded_reproduction = (
                    reproduction_precision
                    >= CONFIRMED_MASK_BOUNDED_MINIMUM_PRECISION
                    and reproduction_recall
                    >= CONFIRMED_MASK_BOUNDED_MINIMUM_RECALL
                )
                reproduction_accepted = (
                    strict_reproduction or bounded_reproduction
                )
                confirmed_mask_audit = {
                    "path": str(confirmed_path),
                    "sha256": _sha256_file(confirmed_path),
                    "confirmed_mask_pixels": confirmed_pixels,
                    "rerun_mask_pixels": union_pixels_before_confirmation,
                    "intersection_pixels": intersection,
                    "reproduction_iou": round(reproduction_iou, 8),
                    "minimum_reproduction_iou": (
                        CONFIRMED_MASK_STRICT_MINIMUM_IOU
                    ),
                    "reproduction_precision": round(reproduction_precision, 8),
                    "reproduction_recall": round(reproduction_recall, 8),
                    "bounded_minimum_precision": (
                        CONFIRMED_MASK_BOUNDED_MINIMUM_PRECISION
                    ),
                    "bounded_minimum_recall": (
                        CONFIRMED_MASK_BOUNDED_MINIMUM_RECALL
                    ),
                    "acceptance_mode": (
                        "strict_iou"
                        if strict_reproduction
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
            if click_sets and accepted_point_sets != len(click_sets):
                reasons.append("not_all_human_click_sets_reproduced")
            if click_sets and (
                confirmed_mask_audit is None
                or confirmed_mask_audit["accepted"] is not True
            ):
                reasons.append("confirmed_mask_reproduction_mismatch")
            if not click_sets and accepted_boxes == 0:
                reasons.append("no_box_prediction_accepted")
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

    policy: dict[str, Any] = {
        "minimum_model_score": minimum_model_score,
        "minimum_prompt_overlap": minimum_prompt_overlap,
        "maximum_image_fraction": maximum_image_fraction,
        "minimum_mask_pixels": minimum_mask_pixels,
        "rejected_mask_policy": "fail_closed_no_mask_evidence",
        "cross_group_arbitration_schema": (CROSS_GROUP_ARBITRATION_SCHEMA_VERSION),
        "cross_group_near_duplicate_iou": CROSS_GROUP_NEAR_DUPLICATE_IOU,
        "cross_group_minimum_intersection_pixels": minimum_mask_pixels,
        "cross_group_foreground_policy": "exclude_whole_asset_foreground",
    }
    if interactive_requested:
        policy["human_point_model_score_policy"] = "advisory_when_human_confirmed"
    if ordered_interaction_requested:
        policy[
            "human_point_replay_policy"
        ] = "first_multimask_then_previous_logits_single_mask"
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
