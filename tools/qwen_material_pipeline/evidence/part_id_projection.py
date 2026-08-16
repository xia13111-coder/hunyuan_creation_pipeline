"""Build reference-photo material evidence for each CAD ``part_id``.

The human-confirmed whole-workpiece SAM3 mask remains a foreground/background
authority only.  Each renderer-authored Part-ID projection supplies a coarse
box and a shape prior for a second, local SAM3 instance segmentation.  When
that photographed mask passes the location-independent CAD-shape contract it
becomes the pixel authority; otherwise the projection is retained only as an
explicitly audited fallback.

This module deliberately has no palette or canonical-group assignment input.
Its primary keys are the stable CAD Part IDs (``P0001``, ``P0002``, ...).
For a small projected Part-ID, deterministic pixel-colour coherence may isolate
one dominant chromatic component inside that same Part-ID.  This is a
single-view contamination guard, never a material-group or cross-part vote.
"""

from __future__ import annotations

import copy
import colorsys
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
from PIL import Image, ImageOps

from .color_semantics import pixel_color_label
from ..segmentation.sam3_regions import (
    DEFAULT_MAXIMUM_CANDIDATE_TO_AMODAL_AREA_RATIO,
    DEFAULT_MINIMUM_AMODAL_CANDIDATE_PRECISION,
    DEFAULT_MINIMUM_AMODAL_COMPLETION_SHAPE_IOU,
    DEFAULT_MINIMUM_CAD_SHAPE_AREA_AGREEMENT,
    DEFAULT_MINIMUM_CAD_SHAPE_IOU,
    DEFAULT_MINIMUM_CAD_SHAPE_SEED_PIXELS,
    DEFAULT_MINIMUM_VISIBLE_CAD_SEED_RECALL,
)
from ..materials.tuning import (
    color_parameters_for_target_srgb,
    tuning_profile_for_material,
)

SCHEMA_VERSION = "qwen-part-id-reference-evidence/v1"
RETRIEVAL_REQUEST_SCHEMA_VERSION = "qwen-visual-material-retrieval-request/v1"
ASSIGNMENT_SCHEMA_VERSION = "1.0"
DEFAULT_MINIMUM_PROJECTED_PIXELS = 32
DEFAULT_MINIMUM_CHROMATIC_RESCUE_PIXELS = 6
DEFAULT_MINIMUM_CHROMATIC_RESCUE_SHARE = 0.30
# A thin hose/connector can legitimately be surrounded by the same coloured
# parent hose, so the floor is intentionally low.  Fully occluded projections
# still score zero because their entire exterior ring matches the occluder.
DEFAULT_MINIMUM_CHROMATIC_LOCAL_CONTRAST = 0.15
DEFAULT_MAXIMUM_CHROMATIC_RESCUE_PART_PIXELS = 512
DEFAULT_MINIMUM_FOREGROUND_OVERLAP = 0.50
DEFAULT_PART_BOX_PADDING_FRACTION = 0.15
DEFAULT_PART_BOX_CONTEXT_FRACTION = 0.35
DEFAULT_MINIMUM_REFINEMENT_OVERLAP = 0.15
DEFAULT_MINIMUM_REFINEMENT_AREA_RATIO = 0.20
DEFAULT_MAXIMUM_REFINEMENT_AREA_RATIO = 5.0
DEFAULT_MINIMUM_REGISTERED_IOU = 0.60
DEFAULT_MINIMUM_REGISTERED_PRECISION = 0.75
DEFAULT_MINIMUM_REGISTERED_RECALL = 0.45
COATING_CONSISTENCY_SCHEMA_VERSION = "qwen-part-id-coating-consistency/v1"
PARAMETER_CANDIDATE_SCHEMA_VERSION = "qwen-part-id-parameter-candidates/v1"
COLOR_EVIDENCE_GATE_SCHEMA_VERSION = "qwen-part-id-color-evidence-gate/v1"
DEFAULT_MAXIMUM_COATING_ALBEDO_DISTANCE = 0.12
DEFAULT_MINIMUM_COATING_COMPONENT_PARTS = 2
DEFAULT_MINIMUM_COATING_ANCHOR_PIXELS = 128
DEFAULT_MINIMUM_COATING_COMPONENT_PIXELS = 256
DEFAULT_MINIMUM_COLOR_FOREGROUND_OVERLAP = 0.75
DEFAULT_MINIMUM_COLOR_ALIGNMENT_SCORE = 0.75
DEFAULT_MINIMUM_COLOR_INLIER_FRACTION = 0.60
DEFAULT_MAXIMUM_COLOR_MEDIAN_DELTA_E = 18.0
DEFAULT_MAXIMUM_ALBEDO_REFERENCE_DELTA_E = 30.0
# ``usd apply --include-review`` accepts the review band, but it deliberately
# rejects values below this floor.  A lower-confidence Qwen answer is not a
# provisional appearance candidate: it is insufficient evidence and must keep
# the exact-cover baseline rather than authoring an arbitrary retrieved MDL.
MINIMUM_APPLYABLE_REVIEW_CONFIDENCE = 0.60
_CHROMATIC_FAMILIES = {
    "red": frozenset({"red"}),
    "warm": frozenset({"orange", "brown", "yellow"}),
    "green": frozenset({"green"}),
    "cyan_blue": frozenset({"cyan", "blue"}),
    "pink": frozenset({"pink"}),
}
_OBSERVED_ASSIGNMENT_RESET_FIELDS = (
    # These fields belong to fallback/source-preserve or subset authoring
    # actions.  Once a photo-observed Part-ID receives a fresh independent
    # material decision, retaining any of them would make the new ``auto`` or
    # ``review`` assignment internally contradictory.
    "apply_action",
    "source_visual_material_prim_path",
    "source_visual_material_binding_sha256",
    "preserve_parent_material_binding",
    "face_subsets",
    # Material groups are never an assignment authority in the Part-ID flow.
    "canonical_group_id",
    "material_region_group_id",
    "group_id",
    # A stale fallback/group parameterization never survives a fresh Part-ID
    # decision.  A new evidence-bound colour-only delta may be authored later.
    "parameters",
)


class PartIdProjectionError(ValueError):
    """Raised when Part-ID/photo evidence cannot satisfy its strict contract."""


def _part_color(part_id: str) -> tuple[int, int, int]:
    """Return the deterministic RGB color used by CAD Part-ID renders."""
    suffix = part_id[1:] if part_id.startswith("P") else ""
    number = int(suffix) if suffix.isdigit() else sum(map(ord, part_id))
    hue = (number * 0.618033988749895) % 1.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.72, 0.95)
    return int(red * 255), int(green * 255), int(blue * 255)


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_object(
    value: str | Path | Mapping[str, Any], label: str
) -> tuple[dict[str, Any], Path | None]:
    if isinstance(value, Mapping):
        return copy.deepcopy(dict(value)), None
    path = Path(value).expanduser().resolve(strict=True)
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PartIdProjectionError(f"unable to read {label}: {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise PartIdProjectionError(f"{label} must be a JSON object")
    return document, path


def _resolve_file(value: Any, *, owner: Path | None, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise PartIdProjectionError(f"{label} must be a non-empty file path")
    path = Path(value).expanduser()
    if not path.is_absolute() and owner is not None:
        path = owner.parent / path
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise PartIdProjectionError(f"{label} does not exist: {path}") from exc
    if not resolved.is_file():
        raise PartIdProjectionError(f"{label} is not a file: {resolved}")
    return resolved


def _open_rgb(path: Path, label: str) -> np.ndarray:
    try:
        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            array = np.asarray(image, dtype=np.uint8)
    except OSError as exc:
        raise PartIdProjectionError(f"unable to read {label}: {path}: {exc}") from exc
    if array.ndim != 3 or array.shape[2] != 3:
        raise PartIdProjectionError(f"{label} is not an RGB image: {path}")
    return array


def _open_mask(path: Path, label: str) -> np.ndarray:
    try:
        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened).convert("L")
            array = np.asarray(image, dtype=np.uint8)
    except OSError as exc:
        raise PartIdProjectionError(f"unable to read {label}: {path}: {exc}") from exc
    if array.ndim != 2:
        raise PartIdProjectionError(f"{label} is not a grayscale mask: {path}")
    return (array >= 128).astype(np.uint8) * 255


def _expanded_mask_box(
    mask: np.ndarray,
    *,
    padding_fraction: float,
) -> tuple[int, int, int, int]:
    """Return a clipped ``left, top, right, bottom`` box around ``mask``.

    The right/bottom coordinates are exclusive.  Padding is proportional to
    the longer projected Part-ID side so the same policy works for both tiny
    fasteners and large sheet-metal parts.
    """

    ys, xs = np.where(mask > 0)
    if not len(xs):
        raise PartIdProjectionError("cannot build a box around an empty mask")
    height, width = mask.shape
    raw_left = int(xs.min())
    raw_right = int(xs.max()) + 1
    raw_top = int(ys.min())
    raw_bottom = int(ys.max()) + 1
    padding = max(
        2,
        int(
            round(
                max(raw_right - raw_left, raw_bottom - raw_top)
                * float(padding_fraction)
            )
        ),
    )
    return (
        max(0, raw_left - padding),
        max(0, raw_top - padding),
        min(width, raw_right + padding),
        min(height, raw_bottom + padding),
    )


def _raw_mask_box(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if not len(xs):
        raise PartIdProjectionError("cannot build a box around an empty mask")
    return (
        int(xs.min()),
        int(ys.min()),
        int(xs.max()) + 1,
        int(ys.max()) + 1,
    )


def _mask_centroid(mask: np.ndarray) -> tuple[float, float]:
    moments = cv2.moments((mask > 0).astype(np.uint8), binaryImage=True)
    if moments["m00"] <= 0.0:
        raise PartIdProjectionError("cannot register an empty Part-ID mask")
    return (
        float(moments["m10"] / moments["m00"]),
        float(moments["m01"] / moments["m00"]),
    )


def _similarity_matrix(
    *,
    source_centroid: tuple[float, float],
    target_centroid: tuple[float, float],
    scale: float,
    rotation_degrees: float,
    translation_offset: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    matrix = cv2.getRotationMatrix2D(
        source_centroid,
        float(rotation_degrees),
        float(scale),
    ).astype(np.float32)
    matrix[0, 2] += (
        float(target_centroid[0])
        - float(source_centroid[0])
        + float(translation_offset[0])
    )
    matrix[1, 2] += (
        float(target_centroid[1])
        - float(source_centroid[1])
        + float(translation_offset[1])
    )
    return matrix


def _binary_registration_metrics(
    registered: np.ndarray,
    target: np.ndarray,
) -> dict[str, float | int]:
    registered_selected = registered > 0
    target_selected = target > 0
    registered_pixels = int(np.count_nonzero(registered_selected))
    target_pixels = int(np.count_nonzero(target_selected))
    intersection = int(np.count_nonzero(registered_selected & target_selected))
    union = registered_pixels + target_pixels - intersection
    return {
        "registered_pixels": registered_pixels,
        "target_pixels": target_pixels,
        "intersection_pixels": intersection,
        "iou": intersection / max(1, union),
        "precision": intersection / max(1, registered_pixels),
        "recall": intersection / max(1, target_pixels),
    }


def _register_similarity_mask(
    source: np.ndarray,
    target: np.ndarray,
    *,
    maximum_side: int = 256,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit a rendered CAD mask to a reference-image mask.

    Only a bounded 2-D similarity transform is optimized: uniform scale,
    in-plane rotation, and translation.  The source USD and its geometry are
    never changed.  A coarse deterministic search is followed by a local
    refinement, which is more stable for sparse binary masks than unconstrained
    affine ECC and cannot introduce shear.  Part-ID refinement uses a shared
    canvas; whole-asset camera registration may use two different canvas sizes.
    """

    if source.ndim != 2 or target.ndim != 2:
        raise PartIdProjectionError("similarity registration masks must both be 2-D")
    if not np.any(source > 0) or not np.any(target > 0):
        raise PartIdProjectionError(
            "similarity registration requires two non-empty masks"
        )

    target_height, target_width = target.shape
    same_canvas = source.shape == target.shape
    if same_canvas:
        # Part-ID refinement operates after the render has been projected into
        # reference-image coordinates.  Work in the common local
        # neighbourhood of the two masks so a tiny fastener is not erased by
        # resizing the entire high-resolution image.
        union = (source > 0) | (target > 0)
        union_y, union_x = np.where(union)
        raw_left = int(union_x.min())
        raw_right = int(union_x.max()) + 1
        raw_top = int(union_y.min())
        raw_bottom = int(union_y.max()) + 1
        crop_padding = max(
            8,
            int(round(0.10 * max(raw_right - raw_left, raw_bottom - raw_top))),
        )
        crop_left = max(0, raw_left - crop_padding)
        crop_right = min(target_width, raw_right + crop_padding)
        crop_top = max(0, raw_top - crop_padding)
        crop_bottom = min(target_height, raw_bottom + crop_padding)
        work_source = source[crop_top:crop_bottom, crop_left:crop_right]
        work_target = target[crop_top:crop_bottom, crop_left:crop_right]
        optimization_domain = "shared_canvas_union_crop"
    else:
        # Whole-asset camera registration deliberately compares a square CAD
        # render with the reference photograph at its native aspect ratio.
        # These masks occupy different pixel coordinate systems and therefore
        # cannot be cropped from a pixel-wise union.  Preserve both complete
        # canvases and let centroid/scale registration map the source into the
        # target coordinate system, as the original global path did.
        work_source = source
        work_target = target
        crop_left = 0
        crop_top = 0
        crop_right = target_width
        crop_bottom = target_height
        optimization_domain = "independent_full_canvases"
    resize_factor = min(
        1.0,
        float(maximum_side)
        / max(
            work_source.shape[0],
            work_source.shape[1],
            work_target.shape[0],
            work_target.shape[1],
        ),
    )
    if resize_factor < 1.0:
        small_source_size = (
            max(1, int(round(work_source.shape[1] * resize_factor))),
            max(1, int(round(work_source.shape[0] * resize_factor))),
        )
        small_target_size = (
            max(1, int(round(work_target.shape[1] * resize_factor))),
            max(1, int(round(work_target.shape[0] * resize_factor))),
        )
        if same_canvas:
            # INTER_AREA followed by a non-zero threshold behaves like a
            # bounded occupancy reduction: tiny local masks remain present
            # instead of randomly disappearing under nearest-neighbour
            # sampling.
            small_source = (
                cv2.resize(
                    work_source,
                    small_source_size,
                    interpolation=cv2.INTER_AREA,
                )
                > 0
            ).astype(np.uint8) * 255
            small_target = (
                cv2.resize(
                    work_target,
                    small_target_size,
                    interpolation=cv2.INTER_AREA,
                )
                > 0
            ).astype(np.uint8) * 255
        else:
            # Global silhouettes contain ample pixels.  Keep the historical
            # nearest-neighbour reduction so this compatibility branch changes
            # only the crash, not established camera-search scoring.
            small_source = cv2.resize(
                work_source,
                small_source_size,
                interpolation=cv2.INTER_NEAREST,
            )
            small_target = cv2.resize(
                work_target,
                small_target_size,
                interpolation=cv2.INTER_NEAREST,
            )
    else:
        small_source = work_source
        small_target = work_target

    source_centroid = _mask_centroid(small_source)
    target_centroid = _mask_centroid(small_target)
    source_pixels = int(np.count_nonzero(small_source))
    target_pixels = int(np.count_nonzero(small_target))
    area_scale = math.sqrt(target_pixels / max(1, source_pixels))
    scales = sorted(
        {
            round(float(np.clip(value, 0.55, 2.25)), 6)
            for value in (
                *np.linspace(0.60, 1.80, 13),
                area_scale * 0.80,
                area_scale * 0.90,
                area_scale,
                area_scale * 1.10,
            )
        }
    )
    rotations = np.linspace(-35.0, 35.0, 15)
    output_size = (small_target.shape[1], small_target.shape[0])

    def evaluate(
        scale: float,
        rotation: float,
        offset_x: float,
        offset_y: float,
    ) -> tuple[float, dict[str, float | int]]:
        matrix = _similarity_matrix(
            source_centroid=source_centroid,
            target_centroid=target_centroid,
            scale=scale,
            rotation_degrees=rotation,
            translation_offset=(offset_x, offset_y),
        )
        warped = cv2.warpAffine(
            small_source,
            matrix,
            output_size,
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        metrics = _binary_registration_metrics(warped, small_target)
        # IoU is authoritative.  Precision is a small tie-breaker so a CAD
        # projection that stays inside SAM3 wins over a similarly scoring mask
        # that spills into a neighbouring component.
        score = float(metrics["iou"]) + 0.04 * float(metrics["precision"])
        return score, metrics

    coarse: list[tuple[float, float, float, float, float]] = []
    for scale in scales:
        for rotation in rotations:
            score, _metrics = evaluate(scale, float(rotation), 0.0, 0.0)
            coarse.append((score, scale, float(rotation), 0.0, 0.0))
    coarse.sort(reverse=True)

    translation_radius = max(
        3.0,
        min(10.0, 0.08 * max(small_source.shape)),
    )
    translated: list[tuple[float, float, float, float, float]] = []
    offsets = np.linspace(-translation_radius, translation_radius, 5)
    for _score, scale, rotation, _x, _y in coarse[:6]:
        for offset_x in offsets:
            for offset_y in offsets:
                score, _metrics = evaluate(
                    scale,
                    rotation,
                    float(offset_x),
                    float(offset_y),
                )
                translated.append(
                    (
                        score,
                        scale,
                        rotation,
                        float(offset_x),
                        float(offset_y),
                    )
                )
    translated.sort(reverse=True)
    seed = translated[0] if translated else coarse[0]

    fine: list[tuple[float, float, float, float, float]] = []
    for scale in np.linspace(max(0.50, seed[1] - 0.10), seed[1] + 0.10, 5):
        for rotation in np.linspace(seed[2] - 4.0, seed[2] + 4.0, 5):
            for offset_x in np.linspace(seed[3] - 2.0, seed[3] + 2.0, 3):
                for offset_y in np.linspace(seed[4] - 2.0, seed[4] + 2.0, 3):
                    score, _metrics = evaluate(
                        float(scale),
                        float(rotation),
                        float(offset_x),
                        float(offset_y),
                    )
                    fine.append(
                        (
                            score,
                            float(scale),
                            float(rotation),
                            float(offset_x),
                            float(offset_y),
                        )
                    )
    fine.sort(reverse=True)
    best = fine[0] if fine else seed

    full_source_centroid = _mask_centroid(source)
    full_target_centroid = _mask_centroid(target)
    full_offset = (
        best[3] / resize_factor,
        best[4] / resize_factor,
    )
    full_matrix = _similarity_matrix(
        source_centroid=full_source_centroid,
        target_centroid=full_target_centroid,
        scale=best[1],
        rotation_degrees=best[2],
        translation_offset=full_offset,
    )
    # The bounded search above is intentionally performed on a small mask,
    # but its quantization can choose a transform that regresses after being
    # lifted to the source-photo resolution.  Finish with a small deterministic
    # full-resolution neighborhood and make full-resolution IoU authoritative.
    # This preserves one similarity transform for the whole source and cannot
    # introduce shear or per-part deformation.
    full_candidates: list[
        tuple[float, float, float, float, float, np.ndarray, dict[str, float | int]]
    ] = []
    full_scale_step = max(0.005, 1.5 / max(target_width, target_height))
    scale_deltas = (
        (
            -2.0 * full_scale_step,
            -full_scale_step,
            0.0,
            full_scale_step,
            2.0 * full_scale_step,
        )
        if resize_factor < 1.0
        else (0.0,)
    )
    rotation_deltas = (-1.0, -0.5, 0.0, 0.5, 1.0) if resize_factor < 1.0 else (0.0,)
    translation_deltas = (-2.0, 0.0, 2.0) if resize_factor < 1.0 else (0.0,)
    for scale_delta in scale_deltas:
        candidate_scale = max(0.50, float(best[1]) + scale_delta)
        for rotation_delta in rotation_deltas:
            candidate_rotation = float(best[2]) + rotation_delta
            for offset_x in translation_deltas:
                for offset_y in translation_deltas:
                    candidate_offset = (
                        float(full_offset[0]) + offset_x,
                        float(full_offset[1]) + offset_y,
                    )
                    candidate_matrix = _similarity_matrix(
                        source_centroid=full_source_centroid,
                        target_centroid=full_target_centroid,
                        scale=candidate_scale,
                        rotation_degrees=candidate_rotation,
                        translation_offset=candidate_offset,
                    )
                    candidate_registered = cv2.warpAffine(
                        source,
                        candidate_matrix,
                        (target_width, target_height),
                        flags=cv2.INTER_NEAREST,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=0,
                    )
                    candidate_metrics = _binary_registration_metrics(
                        candidate_registered, target
                    )
                    candidate_score = float(candidate_metrics["iou"]) + 0.01 * float(
                        candidate_metrics["precision"]
                    )
                    full_candidates.append(
                        (
                            candidate_score,
                            float(candidate_metrics["iou"]),
                            candidate_scale,
                            candidate_rotation,
                            math.hypot(offset_x, offset_y),
                            candidate_matrix,
                            candidate_metrics,
                        )
                    )
    (
        _full_score,
        _full_iou,
        full_scale,
        full_rotation,
        _full_offset_norm,
        full_matrix,
        metrics,
    ) = max(
        full_candidates,
        key=lambda item: (
            item[0],
            item[1],
            -abs(item[3] - float(best[2])),
            -item[4],
        ),
    )
    registered = cv2.warpAffine(
        source,
        full_matrix,
        (target_width, target_height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    final_offset = (
        float(full_matrix[0, 2])
        - float(
            _similarity_matrix(
                source_centroid=full_source_centroid,
                target_centroid=full_target_centroid,
                scale=full_scale,
                rotation_degrees=full_rotation,
            )[0, 2]
        ),
        float(full_matrix[1, 2])
        - float(
            _similarity_matrix(
                source_centroid=full_source_centroid,
                target_centroid=full_target_centroid,
                scale=full_scale,
                rotation_degrees=full_rotation,
            )[1, 2]
        ),
    )
    audit = {
        **metrics,
        "uniform_scale": round(full_scale, 8),
        "rotation_degrees": round(full_rotation, 8),
        "translation_xy": [
            round(float(full_matrix[0, 2]), 8),
            round(float(full_matrix[1, 2]), 8),
        ],
        "translation_offset_xy": [
            round(float(final_offset[0]), 8),
            round(float(final_offset[1]), 8),
        ],
        "affine_2x3": [
            [round(float(value), 8) for value in row] for row in full_matrix.tolist()
        ],
        "optimization": (
            (
                "local_union_crop_bounded_uniform_scale_rotation_translation_"
                "coarse_to_fine_full_resolution"
            )
            if same_canvas
            else (
                "independent_canvas_bounded_uniform_scale_rotation_translation_"
                "coarse_to_fine_full_resolution"
            )
        ),
        "optimization_crop_xyxy": [
            crop_left,
            crop_top,
            crop_right,
            crop_bottom,
        ],
        "optimization_domain": optimization_domain,
        "source_canvas_hw": [int(source.shape[0]), int(source.shape[1])],
        "target_canvas_hw": [int(target.shape[0]), int(target.shape[1])],
    }
    return registered, audit


def _open_grayscale(path: Path, label: str) -> np.ndarray:
    try:
        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened).convert("L")
            array = np.asarray(image, dtype=np.uint8)
    except OSError as exc:
        raise PartIdProjectionError(f"unable to read {label}: {path}: {exc}") from exc
    if array.ndim != 2:
        raise PartIdProjectionError(f"{label} is not grayscale: {path}")
    return array


def _selected_cad_shape_candidate(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return a candidate bound to the sealed view-shared CAD alignment."""

    shared = record.get("view_shared_alignment")
    if not isinstance(shared, Mapping):
        raise PartIdProjectionError(
            "Part-ID SAM3 refinement has no view-shared alignment"
        )
    translation = shared.get("translation_xy_pixels")
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
        or shared.get("part_specific_translation_allowed") is not False
    ):
        raise PartIdProjectionError(
            "Part-ID SAM3 view-shared alignment contract is malformed"
        )

    box_audits = record.get("box_audits")
    if not isinstance(box_audits, list) or len(box_audits) != 1:
        raise PartIdProjectionError(
            "Part-ID SAM3 refinement must contain one shape-guided box audit"
        )
    box_audit = box_audits[0]
    if not isinstance(box_audit, Mapping) or box_audit.get("accepted") is not True:
        raise PartIdProjectionError(
            "Part-ID SAM3 refinement lacks an accepted shape-guided box"
        )
    selected_index = box_audit.get("selected_candidate_index")
    candidates = box_audit.get("candidates")
    if (
        isinstance(selected_index, bool)
        or not isinstance(selected_index, int)
        or not isinstance(candidates, list)
    ):
        raise PartIdProjectionError(
            "Part-ID SAM3 shape-guided candidate selection is malformed"
        )
    selected = next(
        (
            item
            for item in candidates
            if isinstance(item, Mapping)
            and item.get("candidate_index") == selected_index
            and item.get("accepted") is True
        ),
        None,
    )
    if selected is None:
        raise PartIdProjectionError("Part-ID SAM3 selected shape candidate is missing")
    refinement = box_audit.get("shape_point_refinement")
    prompt_audit = (
        refinement.get("prompt_audit") if isinstance(refinement, Mapping) else None
    )
    if (
        not isinstance(refinement, Mapping)
        or refinement.get("accepted") is not True
        or not isinstance(prompt_audit, Mapping)
        or prompt_audit.get("per_mesh_pose_change_allowed") is not False
        or prompt_audit.get("part_specific_translation_allowed") is not False
        or prompt_audit.get("part_local_translation_xy_pixels") != [0.0, 0.0]
        or prompt_audit.get("translation_xy_pixels")
        != [float(translation[0]), float(translation[1])]
    ):
        raise PartIdProjectionError(
            "Part-ID SAM3 candidate is not bound to the view-shared alignment"
        )
    seed_pixels = selected.get("cad_shape_seed_pixels")
    shape_iou = selected.get("cad_shape_iou")
    area_agreement = selected.get("cad_shape_area_agreement")
    base_contract_valid = (
        not isinstance(seed_pixels, bool)
        and isinstance(seed_pixels, int)
        and seed_pixels >= DEFAULT_MINIMUM_CAD_SHAPE_SEED_PIXELS
        and selected.get("cad_shape_location_invariant") is True
    )
    if selected.get("cad_amodal_occlusion_aware") is True:
        amodal_precision = selected.get("cad_amodal_candidate_precision")
        amodal_shape_iou = selected.get("cad_amodal_shape_iou")
        visible_recall = selected.get("cad_seed_recall")
        amodal_area_ratio = selected.get("candidate_to_cad_amodal_area_ratio")
        shape_contract_valid = (
            all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                for value in (
                    amodal_precision,
                    amodal_shape_iou,
                    visible_recall,
                    amodal_area_ratio,
                )
            )
            and float(amodal_precision)
            >= DEFAULT_MINIMUM_AMODAL_CANDIDATE_PRECISION
            and float(amodal_shape_iou)
            >= DEFAULT_MINIMUM_AMODAL_COMPLETION_SHAPE_IOU
            and float(visible_recall) >= DEFAULT_MINIMUM_VISIBLE_CAD_SEED_RECALL
            and float(amodal_area_ratio)
            <= DEFAULT_MAXIMUM_CANDIDATE_TO_AMODAL_AREA_RATIO
        )
    else:
        shape_contract_valid = (
            isinstance(shape_iou, (int, float))
            and not isinstance(shape_iou, bool)
            and math.isfinite(float(shape_iou))
            and float(shape_iou) >= DEFAULT_MINIMUM_CAD_SHAPE_IOU
            and isinstance(area_agreement, (int, float))
            and not isinstance(area_agreement, bool)
            and math.isfinite(float(area_agreement))
            and float(area_agreement)
            >= DEFAULT_MINIMUM_CAD_SHAPE_AREA_AGREEMENT
        )
    if not base_contract_valid or not shape_contract_valid:
        raise PartIdProjectionError(
            "Part-ID SAM3 selected candidate failed the CAD shape contract"
        )
    return copy.deepcopy(dict(selected))


def _open_bgr(path: Path, label: str) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise PartIdProjectionError(f"unable to read {label}: {path}")
    return image


def _validate_unit_interval(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise PartIdProjectionError(f"{label} must be between zero and one")
    return float(value)


def _standard_base_color(rgb: Sequence[float]) -> str:
    red, green, blue = [float(value) for value in rgb]
    maximum = max(red, green, blue)
    minimum = min(red, green, blue)
    spread = maximum - minimum
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    if luminance < 0.10:
        return "black"
    if spread < 0.08:
        if luminance > 0.82:
            return "white"
        if luminance > 0.55:
            return "silver"
        return "gray"
    if red > 0.58 and green > 0.40 and blue < 0.28:
        return "orange"
    if red > 0.48 and green < 0.38 and blue < 0.38:
        return "red"
    if green > red * 1.15 and green > blue * 1.10:
        return "green"
    if blue > red * 1.12 and blue > green * 1.08:
        return "blue"
    if red > 0.42 and green > 0.28 and blue < 0.24:
        return "brown"
    return "other"


def _appearance_descriptor(
    rgb_samples: np.ndarray,
    *,
    mvinverse_samples: Mapping[str, np.ndarray] | None,
    sam3_local_refinement: bool = False,
) -> dict[str, Any]:
    if rgb_samples.ndim != 2 or rgb_samples.shape[1] != 3 or not len(rgb_samples):
        raise PartIdProjectionError("part appearance requires RGB samples")
    normalized = rgb_samples.astype(np.float32) / 255.0
    median_rgb = np.median(normalized, axis=0)
    labels = Counter(
        _standard_base_color(np.asarray(pixel, dtype=np.float32) / 255.0)
        for pixel in rgb_samples
    )
    descriptor: dict[str, Any] = {
        "visual_description": (
            "appearance sampled from one SAM3-localized CAD Part-ID inside "
            "the human-confirmed whole-workpiece foreground"
            if sam3_local_refinement
            else "appearance sampled from exactly one CAD Part-ID projection "
            "inside the human-confirmed SAM3 workpiece foreground"
        ),
        "family_hint": "unknown",
        "base_color": _standard_base_color(median_rgb),
        "finish_hint": "unknown",
        "surface_class": "unknown",
        "median_rgb": [round(float(value), 8) for value in median_rgb],
        "pixel_color_counts": dict(sorted(labels.items())),
        "robust_color_evidence": _robust_color_summary(rgb_samples),
    }
    if mvinverse_samples:
        roughness = mvinverse_samples.get("roughness")
        metallic = mvinverse_samples.get("metallic")
        albedo = mvinverse_samples.get("albedo")
        if isinstance(roughness, np.ndarray) and roughness.size:
            descriptor["roughness_hint"] = round(
                float(np.median(roughness.astype(np.float32) / 255.0)), 8
            )
        if isinstance(metallic, np.ndarray) and metallic.size:
            descriptor["metallicity_hint"] = round(
                float(np.median(metallic.astype(np.float32) / 255.0)), 8
            )
            descriptor["surface_class"] = (
                "conductor" if descriptor["metallicity_hint"] >= 0.5 else "dielectric"
            )
        if (
            isinstance(albedo, np.ndarray)
            and albedo.ndim == 2
            and albedo.shape[1] == 3
            and len(albedo)
        ):
            albedo_rgb = np.median(albedo.astype(np.float32) / 255.0, axis=0)
            descriptor["mvinverse_albedo_median_rgb"] = [
                round(float(value), 8) for value in albedo_rgb
            ]
    return descriptor


def _rgb_to_lab(rgb: Sequence[float]) -> np.ndarray:
    array = np.asarray(rgb, dtype=np.float32)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise PartIdProjectionError("RGB-to-Lab input must contain three values")
    if np.any(array < 0.0) or np.any(array > 1.0):
        raise PartIdProjectionError("RGB-to-Lab input must be normalized")
    return cv2.cvtColor(
        array.reshape(1, 1, 3),
        cv2.COLOR_RGB2LAB,
    ).reshape(3)


def _robust_color_summary(rgb_samples: np.ndarray) -> dict[str, Any]:
    """Describe color coherence without naming or classifying any color.

    The medoid and fixed-radius inlier fraction are deliberately material- and
    hue-agnostic.  They detect a crop containing multiple incompatible colors
    without encoding rules for copper, paint, green, orange, or any Part ID.
    """

    if rgb_samples.ndim != 2 or rgb_samples.shape[1] != 3 or not len(rgb_samples):
        raise PartIdProjectionError("robust color evidence requires RGB samples")
    # Bound CPU and JSON cost while sampling deterministically across the mask.
    if len(rgb_samples) > 4096:
        indices = np.linspace(0, len(rgb_samples) - 1, 4096, dtype=np.int64)
        sampled = rgb_samples[indices]
    else:
        sampled = rgb_samples
    normalized = sampled.astype(np.float32) / 255.0
    lab = cv2.cvtColor(
        normalized.reshape(1, -1, 3),
        cv2.COLOR_RGB2LAB,
    ).reshape(-1, 3)
    coordinate_median = np.median(lab, axis=0)
    medoid_index = int(np.argmin(np.linalg.norm(lab - coordinate_median, axis=1)))
    medoid_lab = lab[medoid_index]
    distances = np.linalg.norm(lab - medoid_lab, axis=1)
    inliers = distances <= 20.0
    if not np.any(inliers):
        inliers[medoid_index] = True
    robust_rgb = np.median(normalized[inliers], axis=0)
    return {
        "method": "cielab_medoid_fixed_radius",
        "sample_count": int(len(rgb_samples)),
        "evaluated_sample_count": int(len(sampled)),
        "inlier_delta_e_radius": 20.0,
        "inlier_fraction": round(float(np.mean(inliers)), 8),
        "median_delta_e": round(float(np.median(distances)), 8),
        "p90_delta_e": round(float(np.quantile(distances, 0.90)), 8),
        "robust_reference_srgb": [round(float(value), 8) for value in robust_rgb],
    }


def _selected_part_observation(
    part_evidence: Mapping[str, Any],
) -> Mapping[str, Any]:
    observations = part_evidence.get("observations")
    if not isinstance(observations, list):
        raise PartIdProjectionError("Part-ID evidence observations are invalid")
    selected = [
        row
        for row in observations
        if isinstance(row, Mapping)
        and row.get("selected_for_material_inference") is True
    ]
    if len(selected) != 1:
        raise PartIdProjectionError(
            "observed Part-ID evidence must select exactly one observation"
        )
    return selected[0]


def _robust_color_from_evidence(
    part_evidence: Mapping[str, Any],
    observation: Mapping[str, Any],
) -> dict[str, Any] | None:
    descriptor = part_evidence.get("descriptor")
    if isinstance(descriptor, Mapping):
        summary = descriptor.get("robust_color_evidence")
        if isinstance(summary, Mapping):
            return copy.deepcopy(dict(summary))
    # Compatibility with already-generated, hash-sealed evidence: derive the
    # same statistics from its selected image/mask pair without mutating it.
    raw_image = observation.get("image")
    raw_mask = observation.get("mask")
    if not isinstance(raw_image, str) or not isinstance(raw_mask, str):
        return None
    image_path = Path(raw_image).expanduser()
    mask_path = Path(raw_mask).expanduser()
    if not image_path.is_file() or not mask_path.is_file():
        return None
    image = _open_rgb(image_path, "Part-ID color evidence image")
    mask = _open_mask(mask_path, "Part-ID color evidence mask")
    if image.shape[:2] != mask.shape:
        return None
    samples = image[mask > 0]
    chromatic_coverage = observation.get("chromatic_coverage")
    tiny_chromatic_rescue = bool(
        isinstance(chromatic_coverage, Mapping)
        and chromatic_coverage.get("applied") is True
        and chromatic_coverage.get("tiny_part_rescue") is True
    )
    minimum_samples = (
        DEFAULT_MINIMUM_CHROMATIC_RESCUE_PIXELS if tiny_chromatic_rescue else 8
    )
    if len(samples) < minimum_samples:
        return None
    return _robust_color_summary(samples)


def evaluate_part_id_color_evidence(
    *,
    part_evidence: Mapping[str, Any],
    sam3_role: Any,
) -> dict[str, Any]:
    """Fail closed before a selected MDL may receive a color candidate."""

    observation = _selected_part_observation(part_evidence)
    reasons: list[str] = []
    raw_pixels = observation.get("trusted_foreground_pixels")
    raw_projected = observation.get("sampling_projection_pixels")
    trusted_pixels = (
        int(raw_pixels)
        if isinstance(raw_pixels, int) and not isinstance(raw_pixels, bool)
        else 0
    )
    projected_pixels = (
        int(raw_projected)
        if isinstance(raw_projected, int)
        and not isinstance(raw_projected, bool)
        and raw_projected > 0
        else trusted_pixels
    )
    chromatic_coverage = observation.get("chromatic_coverage")
    chromatic_isolation_applied = (
        isinstance(chromatic_coverage, Mapping)
        and chromatic_coverage.get("applied") is True
    )
    tiny_chromatic_rescue = (
        chromatic_isolation_applied
        and chromatic_coverage.get("tiny_part_rescue") is True
    )
    minimum_trusted_pixels = (
        DEFAULT_MINIMUM_CHROMATIC_RESCUE_PIXELS
        if tiny_chromatic_rescue
        else max(
            DEFAULT_MINIMUM_PROJECTED_PIXELS,
            min(128, int(math.ceil(0.12 * max(1, projected_pixels)))),
        )
    )
    if trusted_pixels < minimum_trusted_pixels:
        reasons.append("insufficient_trusted_pixels")

    overlap = observation.get("foreground_overlap")
    overlap_value = (
        float(overlap)
        if isinstance(overlap, (int, float))
        and not isinstance(overlap, bool)
        and math.isfinite(float(overlap))
        else 0.0
    )
    minimum_foreground_overlap = (
        0.60 if tiny_chromatic_rescue else DEFAULT_MINIMUM_COLOR_FOREGROUND_OVERLAP
    )
    if overlap_value < minimum_foreground_overlap:
        reasons.append("insufficient_foreground_overlap")

    alignment = observation.get("alignment_score")
    alignment_value = (
        float(alignment)
        if isinstance(alignment, (int, float))
        and not isinstance(alignment, bool)
        and math.isfinite(float(alignment))
        else 0.0
    )
    if alignment_value < DEFAULT_MINIMUM_COLOR_ALIGNMENT_SCORE:
        reasons.append("insufficient_alignment_score")

    refinement = observation.get("part_id_sam3_refinement")
    local_refinement_expected = (
        isinstance(sam3_role, str) and "automatic_local_part_refinement" in sam3_role
    )
    refinement_applied = (
        isinstance(refinement, Mapping) and refinement.get("applied") is True
    )
    if (
        local_refinement_expected
        and not refinement_applied
        and not chromatic_isolation_applied
    ):
        reasons.append("local_part_refinement_not_applied")
    boundary_policy = observation.get("boundary_policy")
    if (
        isinstance(boundary_policy, str)
        and "fallback" in boundary_policy
        and not chromatic_isolation_applied
    ):
        reasons.append("projection_fallback_not_color_authoritative")

    robust = _robust_color_from_evidence(part_evidence, observation)
    inlier_fraction = 0.0
    median_delta_e = math.inf
    robust_reference: list[float] | None = None
    minimum_inlier_fraction = (
        0.40 if tiny_chromatic_rescue else DEFAULT_MINIMUM_COLOR_INLIER_FRACTION
    )
    maximum_median_delta_e = (
        25.0 if tiny_chromatic_rescue else DEFAULT_MAXIMUM_COLOR_MEDIAN_DELTA_E
    )
    if robust is None:
        reasons.append("missing_robust_color_evidence")
    else:
        raw_inlier = robust.get("inlier_fraction")
        raw_delta = robust.get("median_delta_e")
        raw_reference = robust.get("robust_reference_srgb")
        if (
            isinstance(raw_inlier, (int, float))
            and not isinstance(raw_inlier, bool)
            and math.isfinite(float(raw_inlier))
        ):
            inlier_fraction = float(raw_inlier)
        if (
            isinstance(raw_delta, (int, float))
            and not isinstance(raw_delta, bool)
            and math.isfinite(float(raw_delta))
        ):
            median_delta_e = float(raw_delta)
        if (
            isinstance(raw_reference, Sequence)
            and not isinstance(raw_reference, (str, bytes))
            and len(raw_reference) == 3
            and all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                and 0.0 <= float(value) <= 1.0
                for value in raw_reference
            )
        ):
            robust_reference = [float(value) for value in raw_reference]
        if inlier_fraction < minimum_inlier_fraction:
            reasons.append("multimodal_or_contaminated_reference_color")
        if median_delta_e > maximum_median_delta_e:
            reasons.append("excessive_reference_color_dispersion")
        if robust_reference is None:
            reasons.append("invalid_robust_reference_color")

    descriptor = part_evidence.get("descriptor")
    albedo: list[float] | None = None
    if isinstance(descriptor, Mapping):
        raw_albedo = descriptor.get("mvinverse_albedo_median_rgb")
        if (
            isinstance(raw_albedo, Sequence)
            and not isinstance(raw_albedo, (str, bytes))
            and len(raw_albedo) == 3
            and all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                and 0.0 <= float(value) <= 1.0
                for value in raw_albedo
            )
        ):
            albedo = [float(value) for value in raw_albedo]
    target_color_source = "mvinverse_albedo"
    # At 6--23 pixels MVInverse often cannot produce a stable per-Part
    # estimate. The isolated chromatic component is already bounded by CAD
    # Part-ID support, foreground overlap and local contrast, so it may propose
    # the H1 colour. The actual CAD H0/H1 render comparison remains final.
    if albedo is None and tiny_chromatic_rescue and robust_reference is not None:
        albedo = list(robust_reference)
        target_color_source = "single_view_chromatic_component"

    albedo_delta_e = math.inf
    if albedo is None:
        reasons.append("missing_mvinverse_albedo")
    elif robust_reference is not None:
        albedo_delta_e = float(
            np.linalg.norm(_rgb_to_lab(albedo) - _rgb_to_lab(robust_reference))
        )
        if albedo_delta_e > DEFAULT_MAXIMUM_ALBEDO_REFERENCE_DELTA_E:
            reasons.append("mvinverse_reference_color_disagreement")

    reason_codes = sorted(set(reasons))
    eligible = not reason_codes
    components = {
        "trusted_pixels": trusted_pixels,
        "minimum_trusted_pixels": minimum_trusted_pixels,
        "foreground_overlap": round(overlap_value, 8),
        "minimum_foreground_overlap": minimum_foreground_overlap,
        "alignment_score": round(alignment_value, 8),
        "minimum_alignment_score": DEFAULT_MINIMUM_COLOR_ALIGNMENT_SCORE,
        "local_refinement_expected": local_refinement_expected,
        "local_refinement_applied": refinement_applied,
        "single_view_chromatic_isolation_applied": (chromatic_isolation_applied),
        "tiny_chromatic_rescue": tiny_chromatic_rescue,
        "robust_color_inlier_fraction": round(inlier_fraction, 8),
        "minimum_robust_color_inlier_fraction": minimum_inlier_fraction,
        "robust_color_median_delta_e": (
            round(median_delta_e, 8) if math.isfinite(median_delta_e) else None
        ),
        "maximum_robust_color_median_delta_e": maximum_median_delta_e,
        "mvinverse_reference_delta_e": (
            round(albedo_delta_e, 8) if math.isfinite(albedo_delta_e) else None
        ),
        "maximum_mvinverse_reference_delta_e": (
            DEFAULT_MAXIMUM_ALBEDO_REFERENCE_DELTA_E
        ),
    }
    return {
        "schema_version": COLOR_EVIDENCE_GATE_SCHEMA_VERSION,
        "status": "PASS" if eligible else "REJECT",
        "eligible_for_h1_color_candidate": eligible,
        "selected_view_id": observation.get("view_id"),
        "target_color_srgb": albedo if eligible else None,
        "target_color_source": target_color_source if eligible else None,
        "reason_codes": reason_codes,
        "components": components,
        "robust_color_evidence": robust,
    }


def _build_part_id_parameter_candidates(
    *,
    part_id: str,
    material_id: str,
    part_evidence: Mapping[str, Any],
    sam3_role: Any,
    enabled: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Always produce native H0 and conditionally produce color-only H1.

    Candidate generation never mutates the material assignment.  A later
    render-and-compare stage is the sole authority that may select H1.
    """

    h0 = {
        "candidate_id": "H0",
        "kind": "native_mdl",
        "material_id": material_id,
        "parameters": {},
    }
    candidates = [h0]
    gate: dict[str, Any] | None = None
    profile = tuning_profile_for_material(material_id)
    h1_status = "disabled_by_caller"
    color_audit: dict[str, Any] = {
        "status": "native_h0_selected",
        "material_id": material_id,
        "selected_candidate_id": "H0",
        "parameters_applied": False,
    }
    if enabled:
        gate = evaluate_part_id_color_evidence(
            part_evidence=part_evidence,
            sam3_role=sam3_role,
        )
        if profile is None:
            h1_status = "no_reviewed_color_parameter_interface"
            color_audit["h1_rejection_reason"] = h1_status
        elif gate["eligible_for_h1_color_candidate"] is not True:
            h1_status = "evidence_gate_rejected"
            color_audit["h1_rejection_reason"] = h1_status
            color_audit["color_evidence_gate"] = copy.deepcopy(gate)
        else:
            raw_color = gate.get("target_color_srgb")
            if not isinstance(raw_color, list):
                raise PartIdProjectionError(
                    f"eligible color evidence has no target color for {part_id}"
                )
            parameters, authored = color_parameters_for_target_srgb(
                profile,
                raw_color,
            )
            h1_status = "generated_pending_render_comparison"
            h1 = {
                "candidate_id": "H1",
                "kind": "evidence_gated_color_only",
                "material_id": material_id,
                "parameters": parameters,
                "tuning_profile_id": profile.profile_id,
                "target_color_srgb": list(raw_color),
                "color_parameter_semantics": authored["color_parameter_semantics"],
                "authored_parameter_names": sorted(parameters),
                "evidence_gate": copy.deepcopy(gate),
            }
            candidates.append(h1)
            color_audit.update(
                {
                    "h1_status": h1_status,
                    "tuning_profile_id": profile.profile_id,
                    "parameterization_mode": (
                        "part_id_evidence_gated_h0_h1_color_only"
                    ),
                    "color_source": (f"{gate['target_color_source']}_evidence_gated"),
                    "base_color_srgb": list(raw_color),
                    "base_color_linear": authored["base_color_linear"],
                    "authored_color_linear": authored["authored_color_linear"],
                    "color_parameter_semantics": authored["color_parameter_semantics"],
                    "authored_parameter_names": sorted(parameters),
                    "color_evidence_gate": copy.deepcopy(gate),
                }
            )
    candidate_set = {
        "schema_version": PARAMETER_CANDIDATE_SCHEMA_VERSION,
        "part_id": part_id,
        "material_id": material_id,
        "selection_status": "PENDING_RENDER_COMPARISON",
        "selected_candidate_id": "H0",
        "native_h0_is_default": True,
        "parameters_applied_to_plan": False,
        "h1_status": h1_status,
        "candidates": candidates,
    }
    candidate_set["integrity"] = {"document_sha256": _canonical_sha256(candidate_set)}
    return candidate_set, color_audit


def _observation_colorfulness(rgb_samples: np.ndarray) -> float:
    """Return robust median HSV saturation for one projected observation."""

    if rgb_samples.ndim != 2 or rgb_samples.shape[1] != 3 or not len(rgb_samples):
        return 0.0
    normalized = rgb_samples.astype(np.float32) / 255.0
    median_rgb = np.median(normalized, axis=0)
    _hue, saturation, _value = colorsys.rgb_to_hsv(
        *(float(value) for value in median_rgb)
    )
    return float(saturation)


def _dominant_chromatic_component(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    foreground_mask: np.ndarray,
    part_support_mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Isolate a coherent chromatic component inside one small CAD Part-ID.

    The classifier is deliberately material-agnostic.  It does not consume
    Qwen palette groups, share evidence across Part IDs, or require the colour
    to appear in another view.  A component is authoritative only when it
    occupies a substantial fraction of this Part-ID's trusted single-view
    projection.  H0/H1 rendering remains the final authoring authority.
    """

    selected = mask > 0
    sample_count = int(np.count_nonzero(selected))
    empty = np.zeros(mask.shape, dtype=np.uint8)
    if (
        sample_count < DEFAULT_MINIMUM_CHROMATIC_RESCUE_PIXELS
        or sample_count > DEFAULT_MAXIMUM_CHROMATIC_RESCUE_PART_PIXELS
    ):
        return empty, {
            "eligible": False,
            "reason": (
                "outside_small_part_pixel_range"
                if sample_count
                else "empty_part_projection"
            ),
            "source_part_pixels": sample_count,
        }
    coordinates = np.argwhere(selected)
    family_counts = {family: 0 for family in _CHROMATIC_FAMILIES}
    labels: list[str] = []
    for y, x in coordinates:
        red, green, blue = (int(value) for value in image[int(y), int(x)])
        label = pixel_color_label(red, green, blue)
        labels.append(label)
        for family, members in _CHROMATIC_FAMILIES.items():
            if label in members:
                family_counts[family] += 1
                break
    family, matching_pixels = max(
        family_counts.items(),
        key=lambda item: (item[1], item[0]),
    )
    share = matching_pixels / max(1, sample_count)
    # Measure neighbouring colour outside the complete CAD Part-ID support,
    # not outside the eroded trusted sampling mask.  The latter still contains
    # the part's own boundary and would incorrectly penalise a uniformly
    # coloured part.
    part_support = part_support_mask > 0
    ring = cv2.dilate(
        (part_support.astype(np.uint8) * 255),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        iterations=1,
    )
    ring = (ring > 0) & ~part_support & (foreground_mask > 0)
    ring_coordinates = np.argwhere(ring)
    ring_family_pixels = 0
    members = _CHROMATIC_FAMILIES[family]
    for y, x in ring_coordinates:
        red, green, blue = (int(value) for value in image[int(y), int(x)])
        if pixel_color_label(red, green, blue) in members:
            ring_family_pixels += 1
    ring_pixel_count = int(len(ring_coordinates))
    ring_family_share = ring_family_pixels / max(1, ring_pixel_count)
    local_contrast = 1.0 - ring_family_share
    eligible = (
        matching_pixels >= DEFAULT_MINIMUM_CHROMATIC_RESCUE_PIXELS
        and share >= DEFAULT_MINIMUM_CHROMATIC_RESCUE_SHARE
    )
    tiny_part_rescue_eligible = (
        eligible and local_contrast >= DEFAULT_MINIMUM_CHROMATIC_LOCAL_CONTRAST
    )
    component = empty
    if eligible:
        component = np.zeros(mask.shape, dtype=np.uint8)
        for (y, x), label in zip(coordinates, labels):
            if label in members:
                component[int(y), int(x)] = 255
    return component, {
        "eligible": eligible,
        "family": family if matching_pixels else None,
        "matching_pixels": int(matching_pixels),
        "source_part_pixels": sample_count,
        "matching_share": round(float(share), 8),
        "ring_pixels": ring_pixel_count,
        "ring_matching_family_pixels": int(ring_family_pixels),
        "ring_matching_family_share": round(
            float(ring_family_share),
            8,
        ),
        "local_chromatic_contrast": round(float(local_contrast), 8),
        "minimum_matching_pixels": (DEFAULT_MINIMUM_CHROMATIC_RESCUE_PIXELS),
        "minimum_matching_share": DEFAULT_MINIMUM_CHROMATIC_RESCUE_SHARE,
        "minimum_tiny_rescue_local_contrast": (
            DEFAULT_MINIMUM_CHROMATIC_LOCAL_CONTRAST
        ),
        "tiny_part_rescue_eligible": tiny_part_rescue_eligible,
        "maximum_source_part_pixels": (DEFAULT_MAXIMUM_CHROMATIC_RESCUE_PART_PIXELS),
        "family_counts": {
            key: int(value) for key, value in sorted(family_counts.items()) if value
        },
        "assignment_authority": False,
        "cross_part_sharing": False,
        "final_authority": "actual_cad_h0_h1_render_tournament",
    }


def _select_material_observation_index(
    *,
    observations: Sequence[Mapping[str, Any]],
    rgb_samples: Sequence[np.ndarray],
) -> tuple[int, dict[str, Any]]:
    """Select one view without hiding a small colored part in a coverage tie.

    Large parts retain maximum-coverage selection.  For a small part, a
    coherent single-view chromatic component is preferred by purity before
    pixel count.  Otherwise views within 75% of maximum trusted pixels are
    treated as a coverage tie and robust colorfulness breaks that tie.
    """

    if len(observations) != len(rgb_samples) or not observations:
        raise PartIdProjectionError(
            "observation selection requires matching non-empty inputs"
        )
    trusted_counts = [
        int(observation["trusted_foreground_pixels"]) for observation in observations
    ]
    maximum_trusted = max(trusted_counts)
    small_part = maximum_trusted < 512
    eligible = [
        index
        for index, trusted in enumerate(trusted_counts)
        if not small_part or trusted >= 0.75 * maximum_trusted
    ]
    colorfulness = [_observation_colorfulness(samples) for samples in rgb_samples]
    chromatic_candidates = [
        index
        for index, observation in enumerate(observations)
        if isinstance(observation.get("chromatic_coverage"), Mapping)
        and observation["chromatic_coverage"].get("eligible") is True
    ]
    if small_part and chromatic_candidates:
        selected_index = max(
            chromatic_candidates,
            key=lambda index: (
                float(observations[index]["chromatic_coverage"]["matching_share"])
                * float(
                    observations[index]["chromatic_coverage"][
                        "local_chromatic_contrast"
                    ]
                ),
                float(observations[index]["chromatic_coverage"]["matching_share"]),
                int(observations[index]["chromatic_coverage"]["matching_pixels"]),
                trusted_counts[index],
                float(observations[index]["foreground_overlap"]),
                float(observations[index].get("alignment_score") or 0.0),
                str(observations[index]["view_id"]),
            ),
        )
        policy = "small_part_single_view_chromatic_purity_first"
    else:
        selected_index = max(
            eligible,
            key=lambda index: (
                colorfulness[index] if small_part else 0.0,
                trusted_counts[index],
                float(observations[index]["foreground_overlap"]),
                float(observations[index].get("alignment_score") or 0.0),
                str(observations[index]["view_id"]),
            ),
        )
        policy = (
            "small_part_near_coverage_tie_colorfulness_first"
            if small_part and len(eligible) > 1
            else "maximum_trusted_pixels"
        )
    return selected_index, {
        "policy": policy,
        "small_part": small_part,
        "maximum_trusted_pixels": maximum_trusted,
        "near_coverage_ratio": 0.75,
        "eligible_view_ids": [
            str(observations[index]["view_id"]) for index in eligible
        ],
        "view_colorfulness": {
            str(observation["view_id"]): round(colorfulness[index], 8)
            for index, observation in enumerate(observations)
        },
        "chromatic_candidate_view_ids": [
            str(observations[index]["view_id"]) for index in chromatic_candidates
        ],
        "view_chromatic_purity": {
            str(observation["view_id"]): (
                observation.get("chromatic_coverage") or {}
            ).get("matching_share")
            for observation in observations
        },
        "view_chromatic_local_contrast": {
            str(observation["view_id"]): (
                observation.get("chromatic_coverage") or {}
            ).get("local_chromatic_contrast")
            for observation in observations
        },
    }


def _mvinverse_maps_by_view(
    ledger: Mapping[str, Any] | None,
    ledger_path: Path | None,
) -> dict[str, dict[str, Path]]:
    if ledger is None:
        return {}
    outputs = ledger.get("outputs")
    maps = outputs.get("maps") if isinstance(outputs, Mapping) else None
    if not isinstance(maps, list):
        raise PartIdProjectionError("MVInverse ledger outputs.maps is invalid")
    output_dir = outputs.get("directory")
    if not isinstance(output_dir, str) or not output_dir:
        raise PartIdProjectionError("MVInverse ledger output directory is invalid")
    if ledger_path is None:
        base = Path(output_dir).expanduser()
    else:
        base = ledger_path.parent / output_dir
    result: dict[str, dict[str, Path]] = {}
    for index, record in enumerate(maps):
        if not isinstance(record, Mapping):
            raise PartIdProjectionError(f"MVInverse map record {index} is invalid")
        view_id = record.get("view_id")
        map_name = record.get("map")
        raw_path = record.get("path")
        if (
            not isinstance(view_id, str)
            or not view_id
            or map_name not in {"albedo", "roughness", "metallic"}
            or not isinstance(raw_path, str)
            or not raw_path
        ):
            continue
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            # Ledger map paths are relative to the MVInverse output root, not
            # to the JSON file itself.
            path = ledger_path.parent / path if ledger_path is not None else base / path
        path = path.resolve(strict=True)
        expected = record.get("sha256")
        if not isinstance(expected, str) or _sha256_file(path) != expected:
            raise PartIdProjectionError(
                f"MVInverse map failed SHA-256 validation: {view_id}/{map_name}"
            )
        result.setdefault(view_id, {})[str(map_name)] = path
    return result


def _sample_mvinverse(
    maps: Mapping[str, Path],
    mask: np.ndarray,
) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    for name in ("albedo", "roughness", "metallic"):
        path = maps.get(name)
        if path is None:
            continue
        if name == "albedo":
            array = _open_rgb(path, f"MVInverse {name}")
        else:
            array = _open_grayscale(path, f"MVInverse {name}")
        resized = cv2.resize(
            mask,
            (int(array.shape[1]), int(array.shape[0])),
            interpolation=cv2.INTER_NEAREST,
        )
        selected = resized > 0
        if int(np.count_nonzero(selected)) < 8:
            continue
        values = array[selected]
        if name == "albedo":
            output[name] = values.reshape(-1, 3)
        else:
            output[name] = values.reshape(-1)
    return output


def build_part_id_reference_evidence(
    *,
    reference_manifest: str | Path | Mapping[str, Any],
    rendered_registry: str | Path | Mapping[str, Any],
    spatial_mapping_report: str | Path | Mapping[str, Any],
    output_dir: str | Path,
    camera_alignment_acceptance: str | Path | Mapping[str, Any] | None = None,
    mvinverse_ledger: str | Path | Mapping[str, Any] | None = None,
    part_id_sam3_manifest: str | Path | Mapping[str, Any] | None = None,
    minimum_projected_pixels: int = DEFAULT_MINIMUM_PROJECTED_PIXELS,
    minimum_foreground_overlap: float = DEFAULT_MINIMUM_FOREGROUND_OVERLAP,
    minimum_refinement_overlap: float = DEFAULT_MINIMUM_REFINEMENT_OVERLAP,
    minimum_refinement_area_ratio: float = (DEFAULT_MINIMUM_REFINEMENT_AREA_RATIO),
    maximum_refinement_area_ratio: float = (DEFAULT_MAXIMUM_REFINEMENT_AREA_RATIO),
    minimum_registered_iou: float = DEFAULT_MINIMUM_REGISTERED_IOU,
    minimum_registered_precision: float = (DEFAULT_MINIMUM_REGISTERED_PRECISION),
    minimum_registered_recall: float = DEFAULT_MINIMUM_REGISTERED_RECALL,
    part_box_padding_fraction: float = DEFAULT_PART_BOX_PADDING_FRACTION,
    part_box_context_fraction: float = DEFAULT_PART_BOX_CONTEXT_FRACTION,
) -> dict[str, Any]:
    """Build box-first evidence for every visible CAD Part ID.

    A projected Part-ID bounding box is only the coarse correspondence and
    local-search authority.  When a location-independent CAD-shape check
    accepts a SAM3 candidate, the SAM3 instance boundary becomes the photo
    sampling authority.  Otherwise the renderer-authored projection remains
    an explicitly audited fallback, never a claimed photo segmentation.
    """

    if (
        isinstance(minimum_projected_pixels, bool)
        or not isinstance(minimum_projected_pixels, int)
        or minimum_projected_pixels < 1
    ):
        raise PartIdProjectionError(
            "minimum_projected_pixels must be a positive integer"
        )
    overlap_floor = _validate_unit_interval(
        minimum_foreground_overlap,
        "minimum_foreground_overlap",
    )
    refinement_overlap_floor = _validate_unit_interval(
        minimum_refinement_overlap,
        "minimum_refinement_overlap",
    )
    refinement_area_floor = _validate_unit_interval(
        minimum_refinement_area_ratio,
        "minimum_refinement_area_ratio",
    )
    if (
        isinstance(maximum_refinement_area_ratio, bool)
        or not isinstance(maximum_refinement_area_ratio, (int, float))
        or not math.isfinite(float(maximum_refinement_area_ratio))
        or float(maximum_refinement_area_ratio) < 1.0
        or float(maximum_refinement_area_ratio) < refinement_area_floor
    ):
        raise PartIdProjectionError(
            "maximum_refinement_area_ratio must be finite, at least one, "
            "and no smaller than minimum_refinement_area_ratio"
        )
    refinement_area_ceiling = float(maximum_refinement_area_ratio)
    registered_iou_floor = _validate_unit_interval(
        minimum_registered_iou,
        "minimum_registered_iou",
    )
    registered_precision_floor = _validate_unit_interval(
        minimum_registered_precision,
        "minimum_registered_precision",
    )
    registered_recall_floor = _validate_unit_interval(
        minimum_registered_recall,
        "minimum_registered_recall",
    )
    box_padding = _validate_unit_interval(
        part_box_padding_fraction,
        "part_box_padding_fraction",
    )
    box_context = _validate_unit_interval(
        part_box_context_fraction,
        "part_box_context_fraction",
    )
    if box_context < box_padding:
        raise PartIdProjectionError(
            "part_box_context_fraction must be at least part_box_padding_fraction"
        )
    manifest, manifest_path = _read_object(reference_manifest, "reference manifest")
    registry, registry_path = _read_object(rendered_registry, "rendered registry")
    spatial, spatial_path = _read_object(
        spatial_mapping_report,
        "spatial mapping report",
    )
    camera_acceptance: dict[str, Any] | None = None
    camera_acceptance_path: Path | None = None
    camera_acceptance_views: Mapping[str, Any] = {}
    if camera_alignment_acceptance is not None:
        camera_acceptance, camera_acceptance_path = _read_object(
            camera_alignment_acceptance,
            "camera alignment acceptance",
        )
        if camera_acceptance.get("policy") not in {
            "tiered_box_first_part_id_alignment/v1",
            "two_layer_box_first_part_id_alignment/v2",
        }:
            raise PartIdProjectionError(
                "camera alignment acceptance uses an unsupported policy"
            )
        raw_acceptance_views = camera_acceptance.get("views")
        if not isinstance(raw_acceptance_views, Mapping):
            raise PartIdProjectionError(
                "camera alignment acceptance has no view weights"
            )
        camera_acceptance_views = raw_acceptance_views
    ledger: dict[str, Any] | None = None
    ledger_path: Path | None = None
    if mvinverse_ledger is not None:
        ledger, ledger_path = _read_object(mvinverse_ledger, "MVInverse ledger")
        if ledger.get("status") not in {"SUCCESS", "REUSED"}:
            raise PartIdProjectionError(
                "MVInverse ledger is not a verified successful result"
            )
    mvinverse_by_view = _mvinverse_maps_by_view(ledger, ledger_path)
    refinement: dict[str, Any] | None = None
    refinement_path: Path | None = None
    refinement_by_identity: dict[tuple[str, str], dict[str, Any]] = {}
    refinement_records_by_identity: dict[
        tuple[str, str], Mapping[str, Any]
    ] = {}
    if part_id_sam3_manifest is not None:
        refinement, refinement_path = _read_object(
            part_id_sam3_manifest,
            "Part-ID SAM3 refinement manifest",
        )
        if refinement.get("schema_version") != "qwen-sam3-region-result/v1":
            raise PartIdProjectionError(
                "unsupported Part-ID SAM3 refinement manifest schema"
            )
        refinement_unsigned = copy.deepcopy(refinement)
        refinement_integrity = refinement_unsigned.pop("integrity", None)
        if not isinstance(refinement_integrity, Mapping) or refinement_integrity.get(
            "result_sha256"
        ) != _canonical_sha256(refinement_unsigned):
            raise PartIdProjectionError(
                "Part-ID SAM3 refinement manifest failed its integrity seal"
            )
        records = refinement.get("records")
        if not isinstance(records, list):
            raise PartIdProjectionError(
                "Part-ID SAM3 refinement manifest has no records"
            )
        refinement_policy = refinement.get("policy")
        if (
            not isinstance(refinement_policy, Mapping)
            or refinement_policy.get("per_mesh_pose_change_allowed") is not False
            or refinement_policy.get("automatic_shape_point_refinement")
            != "always_run_same_view_cad_shape_positive_negative_points"
        ):
            raise PartIdProjectionError(
                "Part-ID SAM3 manifest does not enforce view-shared CAD guidance"
            )
        shared_alignment_by_view: dict[str, dict[str, Any]] = {}
        for index, raw in enumerate(records):
            if not isinstance(raw, Mapping):
                raise PartIdProjectionError(f"Part-ID SAM3 record {index} is invalid")
            identity = (str(raw.get("view_id")), str(raw.get("group_id")))
            if (
                not identity[0]
                or not identity[1]
                or identity in refinement_records_by_identity
            ):
                raise PartIdProjectionError(
                    "Part-ID SAM3 records contain invalid duplicate identities"
                )
            refinement_records_by_identity[identity] = raw
            shared = raw.get("view_shared_alignment")
            if not isinstance(shared, Mapping):
                raise PartIdProjectionError(
                    f"Part-ID SAM3 record {identity} has no view-shared alignment"
                )
            shared_document = copy.deepcopy(dict(shared))
            prior_shared = shared_alignment_by_view.setdefault(
                identity[0], shared_document
            )
            if prior_shared != shared_document:
                raise PartIdProjectionError(
                    f"Part-ID SAM3 records disagree on shared alignment for {identity[0]}"
                )
            if raw.get("accepted") is not True:
                continue
            mask_record = raw.get("mask")
            mask_value = (
                mask_record.get("path") if isinstance(mask_record, Mapping) else None
            )
            if refinement_path is None:
                raise PartIdProjectionError(
                    "in-memory Part-ID SAM3 manifest cannot resolve mask paths"
                )
            mask_path = _resolve_file(
                mask_value,
                owner=refinement_path,
                label=f"Part-ID SAM3 mask {identity[0]}/{identity[1]}",
            )
            expected_sha256 = (
                mask_record.get("sha256") if isinstance(mask_record, Mapping) else None
            )
            if not isinstance(expected_sha256, str) or expected_sha256 != _sha256_file(
                mask_path
            ):
                raise PartIdProjectionError(
                    f"Part-ID SAM3 mask hash mismatch for {identity}"
                )
            seed_record = raw.get("cad_projection_seed")
            if not isinstance(seed_record, Mapping):
                raise PartIdProjectionError(
                    f"Part-ID SAM3 record {identity} has no sealed CAD seed"
                )
            seed_path = _resolve_file(
                seed_record.get("path"),
                owner=refinement_path,
                label=f"Part-ID SAM3 CAD seed {identity[0]}/{identity[1]}",
            )
            if seed_record.get("sha256") != _sha256_file(seed_path):
                raise PartIdProjectionError(
                    f"Part-ID SAM3 CAD seed hash mismatch for {identity}"
                )
            shape_candidate = _selected_cad_shape_candidate(raw)
            refinement_by_identity[identity] = {
                "mask_path": mask_path,
                "cad_seed_path": seed_path,
                "record": raw,
                "shape_candidate": shape_candidate,
                "view_shared_alignment": shared_document,
            }

    source_views = manifest.get("source_views")
    if not isinstance(source_views, list) or not source_views:
        raise PartIdProjectionError("reference manifest has no source_views")
    references: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(source_views):
        if not isinstance(raw, Mapping):
            raise PartIdProjectionError(
                f"reference manifest source_views[{index}] is invalid"
            )
        view_id = raw.get("id")
        if not isinstance(view_id, str) or not view_id or view_id in references:
            raise PartIdProjectionError("reference view IDs must be unique")
        if raw.get("palette_mask_authority") != (
            "sam3_foreground_before_material_inference"
        ):
            raise PartIdProjectionError(
                f"reference {view_id} is not bound to the SAM3 foreground stage"
            )
        image_path = _resolve_file(
            raw.get("image"),
            owner=manifest_path,
            label=f"reference image {view_id}",
        )
        foreground_path = _resolve_file(
            raw.get("palette_mask"),
            owner=manifest_path,
            label=f"human SAM3 foreground {view_id}",
        )
        image = _open_rgb(image_path, f"reference image {view_id}")
        foreground = _open_mask(
            foreground_path,
            f"human SAM3 foreground {view_id}",
        )
        if image.shape[:2] != foreground.shape:
            raise PartIdProjectionError(
                f"reference/SAM3 dimensions differ for {view_id}"
            )
        references[view_id] = {
            "image_path": image_path,
            "foreground_path": foreground_path,
            "image": image,
            "foreground": foreground,
        }
    for (view_id, part_id), raw in refinement_records_by_identity.items():
        reference = references.get(view_id)
        if reference is None:
            raise PartIdProjectionError(
                f"Part-ID SAM3 record uses unknown reference view {view_id}/{part_id}"
            )
        if raw.get("source_image_sha256") != _sha256_file(
            reference["image_path"]
        ):
            raise PartIdProjectionError(
                f"Part-ID SAM3 source image mismatch for {view_id}/{part_id}"
            )

    parts = registry.get("parts")
    render_set = registry.get("render_set")
    raw_render_views = (
        render_set.get("views") if isinstance(render_set, Mapping) else None
    )
    if (
        not isinstance(parts, list)
        or not parts
        or not isinstance(raw_render_views, list)
        or not raw_render_views
    ):
        raise PartIdProjectionError("rendered registry lacks parts/render views")
    part_ids: list[str] = []
    for raw in parts:
        part_id = raw.get("part_id") if isinstance(raw, Mapping) else None
        if not isinstance(part_id, str) or not part_id or part_id in part_ids:
            raise PartIdProjectionError("registry Part IDs must be unique")
        part_ids.append(part_id)
    unknown_refinement_parts = sorted(
        {
            part_id
            for _view_id, part_id in refinement_records_by_identity
            if part_id not in part_ids
        }
    )
    if unknown_refinement_parts:
        raise PartIdProjectionError(
            "Part-ID SAM3 manifest contains unknown Part IDs: "
            + ", ".join(unknown_refinement_parts)
        )
    render_views: dict[str, dict[str, Any]] = {}
    for raw in raw_render_views:
        if not isinstance(raw, Mapping):
            raise PartIdProjectionError("rendered registry contains an invalid view")
        view_id = raw.get("view_id")
        if not isinstance(view_id, str) or not view_id or view_id in render_views:
            raise PartIdProjectionError("render view IDs must be unique")
        part_ids_path = _resolve_file(
            raw.get("part_ids_raw") or raw.get("part_ids"),
            owner=registry_path,
            label=f"Part-ID render {view_id}",
        )
        render_views[view_id] = {
            "part_ids_path": part_ids_path,
            "part_ids": _open_bgr(part_ids_path, f"Part-ID render {view_id}"),
        }

    alignments = spatial.get("view_alignments")
    if not isinstance(alignments, list):
        raise PartIdProjectionError("spatial mapping report has no view_alignments")
    trusted_alignments: list[dict[str, Any]] = []
    for raw in alignments:
        if (
            not isinstance(raw, Mapping)
            or raw.get("trusted") is not True
            or raw.get("observation_eligible") is not True
        ):
            continue
        reference_id = raw.get("reference_view_id")
        render_id = raw.get("selected_render_view_id")
        if reference_id not in references or render_id not in render_views:
            raise PartIdProjectionError(
                "trusted spatial alignment references an unknown view"
            )
        acceptance = camera_acceptance_views.get(reference_id)
        if camera_acceptance is not None and not isinstance(acceptance, Mapping):
            raise PartIdProjectionError(
                f"camera alignment acceptance does not cover {reference_id}"
            )
        if (
            isinstance(acceptance, Mapping)
            and acceptance.get("observation_eligible", True) is not True
        ):
            continue
        raw_weight = (
            acceptance.get("evidence_weight")
            if isinstance(acceptance, Mapping)
            else 1.0
        )
        if (
            isinstance(raw_weight, bool)
            or not isinstance(raw_weight, (int, float))
            or not math.isfinite(float(raw_weight))
            or not 0.0 < float(raw_weight) <= 1.0
        ):
            raise PartIdProjectionError(
                f"camera alignment evidence weight is invalid for {reference_id}"
            )
        raw_alignment_score = raw.get("score")
        alignment_score = (
            float(raw_alignment_score)
            if isinstance(raw_alignment_score, (int, float))
            and not isinstance(raw_alignment_score, bool)
            and math.isfinite(float(raw_alignment_score))
            else 1.0
        )
        bbox_affine = np.asarray(raw.get("bbox_affine"), dtype=np.float32)
        ecc_warp = np.asarray(raw.get("ecc_warp"), dtype=np.float32)
        if bbox_affine.shape != (2, 3) or ecc_warp.shape != (2, 3):
            raise PartIdProjectionError(
                f"trusted spatial alignment is malformed for {reference_id}"
            )
        trusted_alignments.append(
            {
                "reference_view_id": reference_id,
                "render_view_id": render_id,
                "quarter_turns_ccw": int(raw.get("quarter_turns_ccw", 0)),
                "bbox_affine": bbox_affine,
                "ecc_warp": ecc_warp,
                "alignment_score": round(
                    max(0.0, min(1.0, alignment_score)) * float(raw_weight),
                    8,
                ),
                "camera_alignment_tier": (
                    acceptance.get("tier")
                    if isinstance(acceptance, Mapping)
                    else "legacy_unweighted"
                ),
                "camera_alignment_evidence_weight": float(raw_weight),
            }
        )
    if not trusted_alignments:
        raise PartIdProjectionError(
            "no trusted photo/CAD alignment is available for Part-ID evidence"
        )

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    mask_dir = destination / "masks"
    box_mask_dir = destination / "box_masks"
    crop_dir = destination / "box_crops"
    isolated_crop_dir = destination / "isolated_crops"
    mask_dir.mkdir(parents=True, exist_ok=True)
    box_mask_dir.mkdir(parents=True, exist_ok=True)
    crop_dir.mkdir(parents=True, exist_ok=True)
    isolated_crop_dir.mkdir(parents=True, exist_ok=True)

    records: dict[str, dict[str, Any]] = {
        part_id: {
            "part_id": part_id,
            "status": "unobserved",
            "observations": [],
            "_rgb_samples": [],
            "_mvinverse_samples": [],
        }
        for part_id in part_ids
    }
    for alignment in trusted_alignments:
        reference_id = alignment["reference_view_id"]
        render_id = alignment["render_view_id"]
        reference = references[reference_id]
        chromatic_rescue_pixel_floor = min(
            minimum_projected_pixels,
            DEFAULT_MINIMUM_CHROMATIC_RESCUE_PIXELS,
        )
        ids = np.rot90(
            render_views[render_id]["part_ids"],
            int(alignment["quarter_turns_ccw"]),
        ).copy()
        output_size = (
            int(reference["image"].shape[1]),
            int(reference["image"].shape[0]),
        )
        for part_id in part_ids:
            red, green, blue = _part_color(part_id)
            part_mask = (
                np.all(
                    ids == np.asarray((blue, green, red), dtype=np.uint8),
                    axis=2,
                ).astype(np.uint8)
                * 255
            )
            decoded_pixels = int(np.count_nonzero(part_mask))
            if decoded_pixels < chromatic_rescue_pixel_floor:
                continue
            normalized = cv2.warpAffine(
                part_mask,
                alignment["bbox_affine"],
                output_size,
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            projected = cv2.warpAffine(
                normalized,
                alignment["ecc_warp"],
                output_size,
                flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            projected_pixels = int(np.count_nonzero(projected))
            if projected_pixels < chromatic_rescue_pixel_floor:
                continue
            # Material pixels at a projected silhouette boundary are the most
            # sensitive to sub-pixel registration and reference/CAD occlusion
            # differences.  Prefer the one-pixel interior whenever it still
            # contains enough evidence; this is an anti-contamination step,
            # not a semantic segmentation.
            projected_interior = cv2.erode(
                projected,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                iterations=1,
            )
            if int(np.count_nonzero(projected_interior)) >= minimum_projected_pixels:
                sampling_projection = projected_interior
                boundary_policy = "one_pixel_projected_mask_interior"
            else:
                sampling_projection = projected
                boundary_policy = "full_projected_mask_small_part_fallback"
            refinement_audit: dict[str, Any] | None = None
            if refinement is not None:
                refined_record = refinement_by_identity.get((reference_id, part_id))
                # Local SAM3 is a contamination guard, not a second geometry
                # registration authority.  It is allowed to tighten a
                # globally aligned CAD Part-ID mask, but a missing/rejected
                # local semantic mask must not erase a geometrically valid
                # CAD observation.  Tiny fasteners and nested sub-parts are
                # commonly merged into their parent by semantic segmentation.
                if refined_record is None:
                    trusted_mask = cv2.bitwise_and(
                        sampling_projection,
                        reference["foreground"],
                    )
                    boundary_policy = "global_cad_projection_human_foreground_fallback"
                    refinement_audit = {
                        "applied": False,
                        "status": "local_sam3_missing_or_rejected",
                        "authority": ("whole_asset_aligned_cad_part_id_projection"),
                        "per_part_geometric_warp_applied": False,
                    }
                else:
                    sealed_record = refined_record["record"]
                    if sealed_record.get("source_image_sha256") != _sha256_file(
                        reference["image_path"]
                    ):
                        raise PartIdProjectionError(
                            f"Part-ID SAM3 source image mismatch for "
                            f"{reference_id}/{part_id}"
                        )
                    refined = _open_mask(
                        refined_record["mask_path"],
                        f"Part-ID SAM3 mask {reference_id}/{part_id}",
                    )
                    if refined.shape != reference["foreground"].shape:
                        raise PartIdProjectionError(
                            f"Part-ID SAM3 mask shape differs for "
                            f"{reference_id}/{part_id}"
                        )
                    refined = cv2.bitwise_and(
                        refined,
                        reference["foreground"],
                    )
                    sealed_seed = _open_mask(
                        refined_record["cad_seed_path"],
                        f"Part-ID SAM3 CAD seed {reference_id}/{part_id}",
                    )
                    if sealed_seed.shape != refined.shape:
                        raise PartIdProjectionError(
                            f"Part-ID SAM3 CAD seed shape differs for "
                            f"{reference_id}/{part_id}"
                        )
                    shared_translation = refined_record[
                        "view_shared_alignment"
                    ]["translation_xy_pixels"]
                    aligned_sealed_seed = cv2.warpAffine(
                        sealed_seed,
                        np.asarray(
                            [
                                [1.0, 0.0, float(shared_translation[0])],
                                [0.0, 1.0, float(shared_translation[1])],
                            ],
                            dtype=np.float32,
                        ),
                        (refined.shape[1], refined.shape[0]),
                        flags=cv2.INTER_NEAREST,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=0,
                    )
                    refined_pixels = int(np.count_nonzero(refined))
                    coarse_pixels = int(np.count_nonzero(aligned_sealed_seed))
                    intersection = int(
                        np.count_nonzero(
                            (refined > 0) & (aligned_sealed_seed > 0)
                        )
                    )
                    overlap_smaller = intersection / max(
                        1, min(refined_pixels, coarse_pixels)
                    )
                    area_ratio = refined_pixels / max(1, coarse_pixels)
                    # The SAM3 manifest already sealed one residual alignment
                    # for the complete workpiece.  Re-fitting a similarity
                    # transform here per Part-ID would silently reintroduce the
                    # old failure mode where a mesh template follows the wrong
                    # photo candidate.  Verify the two masks directly in the
                    # shared reference-image coordinate system.
                    registration = {
                        **_binary_registration_metrics(
                            aligned_sealed_seed,
                            refined,
                        ),
                        "uniform_scale": 1.0,
                        "rotation_degrees": 0.0,
                        "translation_xy": [0.0, 0.0],
                        "translation_offset_xy": [0.0, 0.0],
                        "affine_2x3": [
                            [1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                        ],
                        "optimization": (
                            "none_view_shared_manifest_direct_overlap"
                        ),
                        "optimization_domain": "shared_reference_image_canvas",
                    }
                    # Candidate geometry, amodal occlusion, neighboring-Part
                    # ownership, and view-shared alignment were already
                    # evaluated and sealed by SAM3.  Requiring only a usable
                    # foreground remainder here avoids creating a second,
                    # contradictory per-Part geometry authority.
                    refinement_geometry_passed = (
                        refined_pixels >= minimum_projected_pixels
                    )
                    if refinement_geometry_passed:
                        # The CAD silhouette has now served both of its valid
                        # roles: locating the SAM3 search box and verifying the
                        # selected instance shape under the sealed view-shared
                        # alignment.  The photographed SAM3 boundary is the final
                        # pixel authority.  Intersecting it with the original
                        # projection here would re-introduce the very camera
                        # residual that local segmentation is meant to remove.
                        refined_interior = cv2.erode(
                            refined,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                            iterations=1,
                        )
                        trusted_mask = (
                            refined_interior
                            if int(np.count_nonzero(refined_interior))
                            >= minimum_projected_pixels
                            else refined
                        )
                        boundary_policy = (
                            "one_pixel_shape_guided_sam3_photo_mask_interior"
                        )
                        refinement_status = "applied_as_photo_instance_mask"
                    else:
                        # SAM3 often merges a small CAD child with its parent.
                        # Preserve coverage using only the global CAD
                        # projection and the manually confirmed whole-object
                        # foreground; never warp this Part-ID independently.
                        trusted_mask = cv2.bitwise_and(
                            sampling_projection,
                            reference["foreground"],
                        )
                        boundary_policy = (
                            "global_cad_projection_human_foreground_"
                            "local_sam3_incompatible_fallback"
                        )
                        refinement_status = "not_applied_geometry_incompatible"
                    refinement_audit = {
                        "applied": refinement_geometry_passed,
                        "status": refinement_status,
                        "authority": (
                            "sam3_component_bound_to_view_shared_cad_alignment"
                            if refinement_geometry_passed
                            else "whole_asset_aligned_cad_part_id_projection"
                        ),
                        "per_part_geometric_warp_applied": False,
                        "view_shared_alignment": copy.deepcopy(
                            refined_record["view_shared_alignment"]
                        ),
                        "mask": str(refined_record["mask_path"]),
                        "mask_sha256": _sha256_file(refined_record["mask_path"]),
                        "coarse_refined_intersection_pixels": intersection,
                        "coarse_refined_overlap_smaller": round(overlap_smaller, 8),
                        "refined_to_coarse_area_ratio": round(area_ratio, 8),
                        "registration": registration,
                        "sealed_cad_seed": {
                            "path": str(refined_record["cad_seed_path"]),
                            "sha256": _sha256_file(
                                refined_record["cad_seed_path"]
                            ),
                        },
                        "shape_candidate": copy.deepcopy(
                            refined_record["shape_candidate"]
                        ),
                        "cad_projection_role": "location_and_shape_prior_only",
                        "photo_mask_authority": (
                            "sam3_shape_selected_local_instance_boundary"
                            if refinement_geometry_passed
                            else None
                        ),
                        "trusted_photo_mask_pixels": (
                            int(np.count_nonzero(trusted_mask))
                            if refinement_geometry_passed
                            else None
                        ),
                    }
            else:
                trusted_mask = cv2.bitwise_and(
                    sampling_projection,
                    reference["foreground"],
                )
            foreground_trusted_pixels = int(np.count_nonzero(trusted_mask))
            sampling_projection_pixels = int(np.count_nonzero(sampling_projection))
            overlap = foreground_trusted_pixels / sampling_projection_pixels
            if (
                foreground_trusted_pixels < chromatic_rescue_pixel_floor
                or overlap < overlap_floor
            ):
                continue
            chromatic_mask, chromatic_coverage = _dominant_chromatic_component(
                reference["image"],
                trusted_mask,
                foreground_mask=reference["foreground"],
                part_support_mask=projected,
            )
            tiny_projection = foreground_trusted_pixels < minimum_projected_pixels
            if (
                tiny_projection
                and chromatic_coverage["tiny_part_rescue_eligible"] is not True
            ):
                continue
            if chromatic_coverage["eligible"] is True:
                trusted_mask = chromatic_mask
                boundary_policy = "single_view_part_id_dominant_chromatic_component"
                chromatic_coverage["applied"] = True
                chromatic_coverage["tiny_part_rescue"] = tiny_projection
                chromatic_coverage[
                    "pre_isolation_foreground_pixels"
                ] = foreground_trusted_pixels
            else:
                chromatic_coverage["applied"] = False
                chromatic_coverage["tiny_part_rescue"] = False
            trusted_pixels = int(np.count_nonzero(trusted_mask))
            if trusted_pixels < chromatic_rescue_pixel_floor:
                continue

            safe_view = "".join(
                character if character.isalnum() or character in "._-" else "_"
                for character in reference_id
            )
            mask_path = mask_dir / f"{safe_view}__{part_id}.png"
            if not cv2.imwrite(str(mask_path), trusted_mask):
                raise PartIdProjectionError(f"unable to write Part-ID mask {mask_path}")
            raw_box = _raw_mask_box(projected)
            target_box = _expanded_mask_box(
                projected,
                padding_fraction=box_padding,
            )
            context_box = _expanded_mask_box(
                projected,
                padding_fraction=box_context,
            )
            left, top, right, bottom = target_box
            box_mask = np.zeros_like(trusted_mask)
            # The box is the Part-ID correspondence authority.  Intersecting
            # it with the already human-confirmed whole-workpiece foreground
            # removes photographic background without creating a per-part
            # semantic segmentation.
            box_mask[top:bottom, left:right] = reference["foreground"][
                top:bottom, left:right
            ]
            box_mask_path = box_mask_dir / f"{safe_view}__{part_id}.png"
            if not cv2.imwrite(str(box_mask_path), box_mask):
                raise PartIdProjectionError(
                    f"unable to write Part-ID box mask {box_mask_path}"
                )

            (
                isolated_left,
                isolated_top,
                isolated_right,
                isolated_bottom,
            ) = _expanded_mask_box(
                trusted_mask,
                padding_fraction=0.10,
            )
            crop_rgb = reference["image"][
                isolated_top:isolated_bottom,
                isolated_left:isolated_right,
            ]
            crop_mask = trusted_mask[
                isolated_top:isolated_bottom,
                isolated_left:isolated_right,
            ]
            neutral = np.full_like(crop_rgb, 127)
            neutral[crop_mask > 0] = crop_rgb[crop_mask > 0]
            part_isolated_dir = isolated_crop_dir / part_id
            part_isolated_dir.mkdir(parents=True, exist_ok=True)
            isolated_crop_path = part_isolated_dir / f"{safe_view}.png"
            Image.fromarray(neutral, mode="RGB").save(isolated_crop_path)

            context_left, context_top, context_right, context_bottom = context_box
            boxed_crop = reference["image"][
                context_top:context_bottom,
                context_left:context_right,
            ].copy()
            relative_left = left - context_left
            relative_top = top - context_top
            relative_right = right - context_left - 1
            relative_bottom = bottom - context_top - 1
            box_thickness = max(
                1,
                int(round(max(boxed_crop.shape[:2]) / 160.0)),
            )
            cv2.rectangle(
                boxed_crop,
                (relative_left, relative_top),
                (relative_right, relative_bottom),
                (255, 220, 0),
                thickness=box_thickness,
            )
            part_crop_dir = crop_dir / part_id
            part_crop_dir.mkdir(parents=True, exist_ok=True)
            crop_path = part_crop_dir / f"{safe_view}.png"
            Image.fromarray(boxed_crop, mode="RGB").save(crop_path)

            sampled_rgb = reference["image"][trusted_mask > 0]
            photo_part_segmentation_applied = bool(
                isinstance(refinement_audit, Mapping)
                and refinement_audit.get("applied") is True
            )
            record = records[part_id]
            record["_rgb_samples"].append(sampled_rgb)
            view_mvinverse = _sample_mvinverse(
                mvinverse_by_view.get(reference_id, {}),
                trusted_mask,
            )
            record["_mvinverse_samples"].append(view_mvinverse)
            record["observations"].append(
                {
                    "view_id": reference_id,
                    "render_view_id": render_id,
                    "image": str(reference["image_path"]),
                    "image_sha256": _sha256_file(reference["image_path"]),
                    "human_sam3_foreground": str(reference["foreground_path"]),
                    "human_sam3_foreground_sha256": _sha256_file(
                        reference["foreground_path"]
                    ),
                    "mask": str(mask_path.resolve(strict=True)),
                    "mask_sha256": _sha256_file(mask_path),
                    "box_mask": str(box_mask_path.resolve(strict=True)),
                    "box_mask_sha256": _sha256_file(box_mask_path),
                    "crop": str(crop_path.resolve(strict=True)),
                    "crop_sha256": _sha256_file(crop_path),
                    "isolated_crop": str(isolated_crop_path.resolve(strict=True)),
                    "isolated_crop_sha256": _sha256_file(isolated_crop_path),
                    "correspondence_mode": (
                        "cad_box_shape_guided_sam3_photo_instance"
                        if photo_part_segmentation_applied
                        else "cad_projected_part_id_bounding_box"
                    ),
                    "projected_box_xyxy": list(raw_box),
                    "target_box_xyxy": list(target_box),
                    "context_box_xyxy": list(context_box),
                    "box_padding_fraction": box_padding,
                    "box_context_fraction": box_context,
                    "box_foreground_pixels": int(np.count_nonzero(box_mask)),
                    "box_core_share": round(
                        trusted_pixels / max(1, int(np.count_nonzero(box_mask))),
                        8,
                    ),
                    "decoded_part_pixels": decoded_pixels,
                    "projected_part_pixels": projected_pixels,
                    "sampling_projection_pixels": sampling_projection_pixels,
                    "trusted_foreground_pixels": trusted_pixels,
                    "pre_isolation_foreground_pixels": (foreground_trusted_pixels),
                    "foreground_overlap": round(overlap, 8),
                    "boundary_policy": boundary_policy,
                    "chromatic_coverage": chromatic_coverage,
                    "alignment_score": alignment["alignment_score"],
                    "camera_alignment_tier": alignment["camera_alignment_tier"],
                    "camera_alignment_evidence_weight": alignment[
                        "camera_alignment_evidence_weight"
                    ],
                    "evidence_authority": (
                        "sam3_shape_selected_local_photo_instance_mask_"
                        "validated_by_cad_shape_prior"
                        if photo_part_segmentation_applied
                        else "globally_fitted_whole_asset_cad_part_id_"
                        "projection_intersect_human_sam3_foreground"
                    ),
                    "photo_part_segmentation_applied": (
                        photo_part_segmentation_applied
                    ),
                    "sampling_core_authority": (
                        "sam3_photo_instance_mask_for_color_and_pbr"
                        if photo_part_segmentation_applied
                        else "renderer_authored_cad_projection_for_color_and_pbr_only"
                    ),
                    "part_id_sam3_refinement": refinement_audit,
                }
            )

    materialized_records: list[dict[str, Any]] = []
    for part_id in part_ids:
        record = records[part_id]
        rgb_arrays = record.pop("_rgb_samples")
        mvinverse_by_observation = record.pop("_mvinverse_samples")
        if rgb_arrays:
            if len(rgb_arrays) != len(record["observations"]) or len(
                mvinverse_by_observation
            ) != len(record["observations"]):
                raise PartIdProjectionError(
                    f"internal Part-ID sample/observation mismatch for {part_id}"
                )
            # One best visible reference is authoritative for a part.  Other
            # views are retained for audit but are not averaged and do not
            # vote.  This handles parts that exist in only one photograph and
            # prevents a weaker occluded projection from polluting a clear one.
            selected_index, selection_audit = _select_material_observation_index(
                observations=record["observations"],
                rgb_samples=rgb_arrays,
            )
            selected_view_id = str(record["observations"][selected_index]["view_id"])
            selected_mvinverse = mvinverse_by_observation[selected_index]
            for index, observation in enumerate(record["observations"]):
                observation["selected_for_material_inference"] = index == selected_index
            record["status"] = "observed"
            record["selected_observation_view_id"] = selected_view_id
            record["observation_selection_audit"] = selection_audit
            descriptor = _appearance_descriptor(
                rgb_arrays[selected_index],
                mvinverse_samples=selected_mvinverse,
                sam3_local_refinement=refinement is not None,
            )
            selected_chromatic = record["observations"][selected_index].get(
                "chromatic_coverage"
            )
            if (
                isinstance(selected_chromatic, Mapping)
                and selected_chromatic.get("applied") is True
            ):
                chromatic_family = selected_chromatic.get("family")
                canonical_color = {
                    "red": "red",
                    "warm": "orange",
                    "green": "green",
                    "cyan_blue": "blue",
                    "pink": "pink",
                }.get(chromatic_family)
                if canonical_color is not None:
                    descriptor["base_color"] = canonical_color
                    descriptor["chromatic_component_family"] = chromatic_family
                    descriptor["visual_description"] = (
                        f"{descriptor['visual_description']}; dominant "
                        "single-view chromatic component isolated inside "
                        "this CAD Part-ID"
                    )
            record["descriptor"] = descriptor
            record["observations"].sort(key=lambda item: item["view_id"])
        else:
            record["descriptor"] = None
        materialized_records.append(record)

    input_files = [
        {
            "label": "reference_manifest",
            "path": str(manifest_path) if manifest_path is not None else None,
            "document_sha256": _canonical_sha256(manifest),
        },
        {
            "label": "rendered_registry",
            "path": str(registry_path) if registry_path is not None else None,
            "document_sha256": _canonical_sha256(registry),
        },
        {
            "label": "spatial_mapping_report",
            "path": str(spatial_path) if spatial_path is not None else None,
            "document_sha256": _canonical_sha256(spatial),
        },
    ]
    if ledger is not None:
        input_files.append(
            {
                "label": "mvinverse_ledger",
                "path": str(ledger_path) if ledger_path is not None else None,
                "document_sha256": _canonical_sha256(ledger),
            }
        )
    if camera_acceptance is not None:
        input_files.append(
            {
                "label": "camera_alignment_acceptance",
                "path": (
                    str(camera_acceptance_path)
                    if camera_acceptance_path is not None
                    else None
                ),
                "document_sha256": _canonical_sha256(camera_acceptance),
            }
        )
    if refinement is not None:
        input_files.append(
            {
                "label": "part_id_sam3_refinement_manifest",
                "path": (str(refinement_path) if refinement_path is not None else None),
                "document_sha256": _canonical_sha256(refinement),
            }
        )
    unsigned = {
        "schema_version": SCHEMA_VERSION,
        "assignment_unit": "part_id",
        "sam3_role": (
            "human_confirmed_whole_workpiece_plus_automatic_local_part_refinement"
            if refinement is not None
            else "whole_workpiece_foreground_only"
        ),
        "part_segmentation_authority": (
            "shape_guided_sam3_photo_instance_when_valid_otherwise_"
            "audited_cad_projection_fallback"
            if refinement is not None
            else "renderer_authored_cad_part_id_masks"
        ),
        "part_correspondence_authority": (
            "whole_asset_camera_then_view_shared_residual_then_cad_shape_"
            "selected_sam3_instance_without_part_local_warp"
            if refinement is not None
            else "cad_projected_part_id_bounding_boxes"
        ),
        "photo_part_segmentation_applied": refinement is not None,
        "cross_view_consensus_required": False,
        "unseen_views_cast_vote": False,
        "multi_view_appearance_averaging": False,
        "policy": {
            "minimum_projected_pixels": minimum_projected_pixels,
            "minimum_chromatic_rescue_pixels": (
                DEFAULT_MINIMUM_CHROMATIC_RESCUE_PIXELS
            ),
            "minimum_chromatic_rescue_share": (DEFAULT_MINIMUM_CHROMATIC_RESCUE_SHARE),
            "minimum_chromatic_local_contrast": (
                DEFAULT_MINIMUM_CHROMATIC_LOCAL_CONTRAST
            ),
            "maximum_chromatic_rescue_part_pixels": (
                DEFAULT_MAXIMUM_CHROMATIC_RESCUE_PART_PIXELS
            ),
            "minimum_foreground_overlap": overlap_floor,
            "minimum_refinement_overlap": refinement_overlap_floor,
            "minimum_refinement_area_ratio": refinement_area_floor,
            "maximum_refinement_area_ratio": refinement_area_ceiling,
            "minimum_registered_iou": registered_iou_floor,
            "minimum_registered_precision": registered_precision_floor,
            "minimum_registered_recall": registered_recall_floor,
            "refinement_geometry_authority": (
                "sealed_sam3_visible_amodal_neighbor_and_view_shared_contract"
                if refinement is not None
                else "not_applicable"
            ),
            "part_box_padding_fraction": box_padding,
            "part_box_context_fraction": box_context,
            "retrieval_region": (
                "projected_box_intersect_human_whole_workpiece_foreground"
            ),
            "color_pbr_sampling_region": ("renderer_authored_projected_part_id_core"),
            "camera_alignment_weighting": (
                "two_layer_box_first_part_id_alignment/v2"
                if camera_acceptance is not None
                else "legacy_unweighted"
            ),
            "missing_or_rejected_refinement_policy": (
                "whole_asset_aligned_projection_intersect_human_"
                "foreground_fallback_when_local_box_refinement_is_unusable"
                if refinement is not None
                else "not_applicable"
            ),
            "part_local_geometric_warp_allowed": False,
            "single_trusted_view_may_authorize_observation": True,
            "per_part_observation_selection": (
                "single_view_chromatic_purity_then_coverage_without_"
                "cross_part_or_cross_view_votes"
            ),
        },
        "inputs": input_files,
        "parts": materialized_records,
        "summary": {
            "registry_part_count": len(part_ids),
            "observed_part_count": sum(
                record["status"] == "observed" for record in materialized_records
            ),
            "unobserved_part_count": sum(
                record["status"] == "unobserved" for record in materialized_records
            ),
            "observation_count": sum(
                len(record["observations"]) for record in materialized_records
            ),
            "trusted_reference_view_count": len(trusted_alignments),
            "sam3_refined_observation_count": sum(
                sum(
                    (observation.get("part_id_sam3_refinement") or {}).get("applied")
                    is True
                    for observation in record["observations"]
                )
                for record in materialized_records
            ),
            "global_projection_fallback_observation_count": sum(
                sum(
                    observation.get("part_id_sam3_refinement") is not None
                    and (observation.get("part_id_sam3_refinement") or {}).get(
                        "applied"
                    )
                    is not True
                    for observation in record["observations"]
                )
                for record in materialized_records
            ),
            "chromatic_isolated_observation_count": sum(
                sum(
                    (observation.get("chromatic_coverage") or {}).get("applied") is True
                    for observation in record["observations"]
                )
                for record in materialized_records
            ),
            "tiny_chromatic_rescue_observation_count": sum(
                sum(
                    (observation.get("chromatic_coverage") or {}).get(
                        "tiny_part_rescue"
                    )
                    is True
                    for observation in record["observations"]
                )
                for record in materialized_records
            ),
            "selected_reference_view_coverage": {
                alignment["reference_view_id"]: {
                    "visible_part_count": sum(
                        any(
                            observation.get("view_id") == alignment["reference_view_id"]
                            for observation in record["observations"]
                        )
                        for record in materialized_records
                    ),
                    "selected_part_count": sum(
                        any(
                            observation.get("view_id") == alignment["reference_view_id"]
                            and observation.get("selected_for_material_inference")
                            is True
                            for observation in record["observations"]
                        )
                        for record in materialized_records
                    ),
                    "selected_chromatic_part_count": sum(
                        any(
                            observation.get("view_id") == alignment["reference_view_id"]
                            and observation.get("selected_for_material_inference")
                            is True
                            and (observation.get("chromatic_coverage") or {}).get(
                                "applied"
                            )
                            is True
                            for observation in record["observations"]
                        )
                        for record in materialized_records
                    ),
                }
                for alignment in trusted_alignments
            },
        },
    }
    return {
        **unsigned,
        "integrity": {"document_sha256": _canonical_sha256(unsigned)},
    }


def build_part_id_retrieval_request(
    *,
    evidence: Mapping[str, Any],
    catalog: str | Path,
    material_root: str | Path,
) -> dict[str, Any]:
    """Create a SigLIP/DINO request whose entity IDs are CAD Part IDs."""

    if evidence.get("schema_version") != SCHEMA_VERSION:
        raise PartIdProjectionError("unsupported Part-ID evidence schema")
    parts = evidence.get("parts")
    if not isinstance(parts, list):
        raise PartIdProjectionError("Part-ID evidence has no parts")
    groups: list[dict[str, Any]] = []
    for raw in parts:
        if not isinstance(raw, Mapping) or raw.get("status") != "observed":
            continue
        part_id = raw.get("part_id")
        descriptor = raw.get("descriptor")
        observations = raw.get("observations")
        if (
            not isinstance(part_id, str)
            or not isinstance(descriptor, Mapping)
            or not isinstance(observations, list)
            or not observations
        ):
            raise PartIdProjectionError("observed Part-ID evidence is malformed")
        groups.append(
            {
                # The generic retrieval runtime calls this field group_id, but
                # its value and assignment unit are explicitly the CAD Part ID.
                "group_id": part_id,
                "assignment_unit": "part_id",
                "part_id": part_id,
                "descriptor": copy.deepcopy(dict(descriptor)),
                "observations": [
                    {
                        "view_id": observation["view_id"],
                        "image": observation["image"],
                        # The bounding box establishes tolerant Part-ID
                        # correspondence only.  Material retrieval must use
                        # the registered CAD core, otherwise a neighbouring
                        # panel or occluder inside a valid box becomes false
                        # colour/texture evidence.
                        "mask": observation["mask"],
                        "core_mask": observation["mask"],
                        "correspondence_box_mask": observation.get("box_mask"),
                        "material_sampling_mode": "registered_cad_part_id_core",
                        "correspondence_mode": observation.get(
                            "correspondence_mode",
                            "legacy_projected_mask",
                        ),
                    }
                    for observation in observations
                    if observation.get("selected_for_material_inference") is True
                ],
            }
        )
    if not groups:
        raise PartIdProjectionError(
            "no CAD Part ID has a trusted observation for visual retrieval"
        )
    return {
        "schema_version": RETRIEVAL_REQUEST_SCHEMA_VERSION,
        "catalog": str(Path(catalog).expanduser().resolve(strict=True)),
        "material_root": str(Path(material_root).expanduser().resolve(strict=True)),
        "groups": groups,
        "assignment_unit": "part_id",
        "group_field_compatibility_note": (
            "retrieval.group_id contains the exact CAD part_id; no palette "
            "fusion or material-region group exists"
        ),
        "part_id_evidence_sha256": evidence["integrity"]["document_sha256"],
    }


def _valid_unit_rgb(value: Any) -> list[float] | None:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 3
        or any(
            isinstance(channel, bool)
            or not isinstance(channel, (int, float))
            or not math.isfinite(float(channel))
            or not 0.0 <= float(channel) <= 1.0
            for channel in value
        )
    ):
        return None
    return [float(channel) for channel in value]


def _part_evidence_albedo(
    part_evidence: Mapping[str, Any],
) -> tuple[list[float] | None, str | None]:
    descriptor = part_evidence.get("descriptor")
    if not isinstance(descriptor, Mapping):
        return None, None
    for field in ("mvinverse_albedo_median_rgb", "median_rgb"):
        color = _valid_unit_rgb(descriptor.get(field))
        if color is not None:
            return color, field
    return None, None


def _selected_observation_weight(part_evidence: Mapping[str, Any]) -> tuple[float, int]:
    observations = part_evidence.get("observations")
    if not isinstance(observations, list):
        return 0.0, 0
    selected = [
        row
        for row in observations
        if isinstance(row, Mapping)
        and row.get("selected_for_material_inference") is True
    ]
    if len(selected) != 1:
        return 0.0, 0
    observation = selected[0]
    raw_pixels = observation.get("trusted_foreground_pixels")
    pixels = (
        int(raw_pixels)
        if isinstance(raw_pixels, int)
        and not isinstance(raw_pixels, bool)
        and raw_pixels > 0
        else 0
    )
    raw_alignment = observation.get("alignment_score", 1.0)
    alignment = (
        float(raw_alignment)
        if isinstance(raw_alignment, (int, float))
        and not isinstance(raw_alignment, bool)
        and math.isfinite(float(raw_alignment))
        else 1.0
    )
    raw_overlap = observation.get("foreground_overlap", 1.0)
    overlap = (
        float(raw_overlap)
        if isinstance(raw_overlap, (int, float))
        and not isinstance(raw_overlap, bool)
        and math.isfinite(float(raw_overlap))
        else 1.0
    )
    return (
        float(max(1, pixels))
        * max(0.05, min(1.0, alignment))
        * max(0.05, min(1.0, overlap)),
        pixels,
    )


def _registry_assembly_domain(
    part: Mapping[str, Any],
    *,
    default_prim: Any,
) -> str | None:
    parent_path = part.get("parent_path")
    if not isinstance(parent_path, str) or not parent_path:
        return None
    segments = [segment for segment in parent_path.split("/") if segment]
    if not segments:
        return None
    root_name = (
        str(default_prim).rstrip("/").split("/")[-1]
        if isinstance(default_prim, str) and default_prim
        else None
    )
    if root_name:
        while segments and segments[0] == root_name:
            segments.pop(0)
    elif len(segments) >= 2 and segments[0] == segments[1]:
        segments = segments[2:]
    return segments[0] if segments else root_name


def _rgb_distance(first: Sequence[float], second: Sequence[float]) -> float:
    return math.sqrt(
        sum((float(first[index]) - float(second[index])) ** 2 for index in range(3))
    )


def _weighted_channel_median(
    rows: Sequence[Mapping[str, Any]],
    channel: int,
) -> float:
    ordered = sorted(
        ((float(row["color"][channel]), float(row["weight"])) for row in rows),
        key=lambda item: item[0],
    )
    total = sum(weight for _value, weight in ordered)
    threshold = total * 0.5
    cumulative = 0.0
    for value, weight in ordered:
        cumulative += weight
        if cumulative >= threshold:
            return value
    return ordered[-1][0]


def _apply_part_id_coating_consistency(
    *,
    assignments: list[dict[str, Any]],
    audit_rows: list[dict[str, Any]],
    evidence_by_part: Mapping[str, Mapping[str, Any]],
    part_registry: Mapping[str, Any],
    allow_color_parameters: bool,
    maximum_albedo_distance: float,
) -> dict[str, Any]:
    """Unify only high-confidence same-coating Part-ID components.

    This is deliberately not palette grouping.  Every output assignment keeps
    its CAD Part-ID primary key.  Source appearance and assembly ancestry only
    propose membership, while independent MVInverse/reference albedo must also
    agree before a component is allowed to share one immutable MDL and color.
    """

    raw_registry_parts = part_registry.get("parts")
    if not isinstance(raw_registry_parts, list):
        raise PartIdProjectionError("Part-ID coating consistency needs registry parts")
    registry_by_part: dict[str, Mapping[str, Any]] = {}
    for raw in raw_registry_parts:
        part_id = raw.get("part_id") if isinstance(raw, Mapping) else None
        if not isinstance(part_id, str) or not part_id or part_id in registry_by_part:
            raise PartIdProjectionError(
                "Part-ID coating consistency registry has invalid Part IDs"
            )
        registry_by_part[part_id] = raw
    assignment_by_part = {
        str(row["part_id"]): row
        for row in assignments
        if isinstance(row.get("part_id"), str)
    }
    audit_by_part = {
        str(row["part_id"]): row
        for row in audit_rows
        if isinstance(row.get("part_id"), str)
    }
    if set(registry_by_part) != set(assignment_by_part):
        raise PartIdProjectionError(
            "Part-ID coating consistency registry does not exactly cover the plan"
        )

    default_prim = part_registry.get("default_prim")
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = {}
    skipped: Counter[str] = Counter()
    for part_id, evidence_row in evidence_by_part.items():
        if evidence_row.get("status") != "observed":
            skipped["unobserved"] += 1
            continue
        registry_row = registry_by_part[part_id]
        source_signature = registry_row.get("source_appearance_sha256")
        assembly_domain = _registry_assembly_domain(
            registry_row,
            default_prim=default_prim,
        )
        color, color_source = _part_evidence_albedo(evidence_row)
        weight, pixels = _selected_observation_weight(evidence_row)
        if not isinstance(source_signature, str) or not source_signature:
            skipped["missing_source_appearance"] += 1
            continue
        if assembly_domain is None:
            skipped["missing_assembly_domain"] += 1
            continue
        if color is None or color_source is None:
            skipped["missing_color_evidence"] += 1
            continue
        buckets.setdefault((assembly_domain, source_signature), []).append(
            {
                "part_id": part_id,
                "color": color,
                "color_source": color_source,
                "weight": weight,
                "trusted_pixels": pixels,
            }
        )

    proposed_clusters: list[dict[str, Any]] = []
    for (assembly_domain, source_signature), rows in sorted(buckets.items()):
        clusters: list[list[dict[str, Any]]] = []
        for row in sorted(rows, key=lambda item: (-item["weight"], item["part_id"])):
            compatible = [
                cluster
                for cluster in clusters
                if all(
                    _rgb_distance(row["color"], member["color"])
                    <= maximum_albedo_distance
                    for member in cluster
                )
            ]
            if compatible:
                compatible[0].append(row)
            else:
                clusters.append([row])
        for cluster in clusters:
            if len(cluster) < DEFAULT_MINIMUM_COATING_COMPONENT_PARTS:
                skipped["singleton_color_cluster"] += len(cluster)
                continue
            total_pixels = sum(int(row["trusted_pixels"]) for row in cluster)
            maximum_pixels = max(int(row["trusted_pixels"]) for row in cluster)
            if (
                total_pixels < DEFAULT_MINIMUM_COATING_COMPONENT_PIXELS
                or maximum_pixels < DEFAULT_MINIMUM_COATING_ANCHOR_PIXELS
            ):
                skipped["insufficient_component_evidence"] += len(cluster)
                continue
            proposed_clusters.append(
                {
                    "assembly_domain": assembly_domain,
                    "source_appearance_sha256": source_signature,
                    "rows": cluster,
                }
            )

    component_audits: list[dict[str, Any]] = []
    changed_part_ids: set[str] = set()
    for proposal in proposed_clusters:
        rows = proposal["rows"]
        material_eligible_rows = [
            row
            for row in rows
            if (
                not allow_color_parameters
                or tuning_profile_for_material(
                    str(assignment_by_part[row["part_id"]].get("material_id", ""))
                )
                is not None
            )
        ]
        anchor_pool = material_eligible_rows or rows
        anchor = max(
            anchor_pool,
            key=lambda row: (float(row["weight"]), row["part_id"]),
        )
        anchor_part_id = str(anchor["part_id"])
        canonical_material_id = str(assignment_by_part[anchor_part_id]["material_id"])
        canonical_color = [
            _weighted_channel_median(rows, channel) for channel in range(3)
        ]
        canonical_parameters: dict[str, Any] | None = None
        canonical_color_audit: dict[str, Any] = {
            "status": "material_not_color_tunable",
            "material_id": canonical_material_id,
        }
        profile = tuning_profile_for_material(canonical_material_id)
        if allow_color_parameters and profile is not None:
            (
                canonical_parameters,
                authored_color_audit,
            ) = color_parameters_for_target_srgb(profile, canonical_color)
            canonical_color_audit = {
                "status": "authored",
                "material_id": canonical_material_id,
                "tuning_profile_id": profile.profile_id,
                "parameterization_mode": (
                    "coating_component_weighted_mvinverse_albedo_color_only"
                ),
                "color_source": "component_weighted_median_part_id_albedo",
                "base_color_srgb": canonical_color,
                "base_color_linear": authored_color_audit["base_color_linear"],
                "authored_color_linear": authored_color_audit["authored_color_linear"],
                "color_parameter_semantics": authored_color_audit[
                    "color_parameter_semantics"
                ],
                "authored_parameter_names": sorted(canonical_parameters),
            }
        elif allow_color_parameters:
            anchor_parameters = assignment_by_part[anchor_part_id].get("parameters")
            if isinstance(anchor_parameters, Mapping):
                canonical_parameters = copy.deepcopy(dict(anchor_parameters))
                anchor_audit = assignment_by_part[anchor_part_id].get("provenance", {})
                if isinstance(anchor_audit, Mapping):
                    raw_color_audit = anchor_audit.get("mdl_color_parameterization")
                    if isinstance(raw_color_audit, Mapping):
                        canonical_color_audit = copy.deepcopy(dict(raw_color_audit))

        member_part_ids = sorted(str(row["part_id"]) for row in rows)
        component_id = "CC_" + _canonical_sha256(
            {
                "assembly_domain": proposal["assembly_domain"],
                "source_appearance_sha256": proposal["source_appearance_sha256"],
                "member_part_ids": member_part_ids,
            }
        )[:12]
        pre_materials: dict[str, str] = {}
        for row in rows:
            part_id = str(row["part_id"])
            assignment = assignment_by_part[part_id]
            pre_material = str(assignment.get("material_id"))
            pre_materials[part_id] = pre_material
            if pre_material != canonical_material_id:
                changed_part_ids.add(part_id)
            assignment["material_id"] = canonical_material_id
            if canonical_parameters is not None:
                assignment["parameters"] = copy.deepcopy(canonical_parameters)
            else:
                assignment.pop("parameters", None)
            assignment["semantic"] = (
                "independent CAD Part-ID evidence with high-confidence "
                "same-coating consistency"
            )
            provenance = assignment.get("provenance")
            if not isinstance(provenance, dict):
                provenance = {}
                assignment["provenance"] = provenance
            provenance["pre_coating_consistency_material_id"] = pre_material
            provenance[
                "selection_basis"
            ] = "part_id_independent_evidence_then_safe_same_coating_consistency"
            provenance["mdl_color_parameterization"] = copy.deepcopy(
                canonical_color_audit
            )
            provenance["coating_consistency"] = {
                "component_id": component_id,
                "anchor_part_id": anchor_part_id,
                "canonical_material_id": canonical_material_id,
                "member_part_ids": member_part_ids,
                "membership_basis": (
                    "same_assembly_domain_plus_source_appearance_plus_"
                    "independent_albedo_complete_link"
                ),
                "maximum_albedo_rgb_distance": maximum_albedo_distance,
                "palette_or_material_group_used": False,
            }
            audit_row = audit_by_part[part_id]
            audit_row["pre_coating_consistency_material_id"] = pre_material
            audit_row["material_id"] = canonical_material_id
            audit_row["mdl_color_parameterization"] = copy.deepcopy(
                canonical_color_audit
            )
            audit_row["coating_consistency"] = copy.deepcopy(
                provenance["coating_consistency"]
            )

        component_audits.append(
            {
                "component_id": component_id,
                "assembly_domain": proposal["assembly_domain"],
                "source_appearance_sha256": proposal["source_appearance_sha256"],
                "member_part_ids": member_part_ids,
                "anchor_part_id": anchor_part_id,
                "anchor_trusted_pixels": int(anchor["trusted_pixels"]),
                "canonical_material_id": canonical_material_id,
                "canonical_color_srgb": canonical_color,
                "pre_consistency_material_ids": pre_materials,
                "changed_part_ids": sorted(
                    part_id
                    for part_id, material_id in pre_materials.items()
                    if material_id != canonical_material_id
                ),
                "total_trusted_pixels": sum(int(row["trusted_pixels"]) for row in rows),
            }
        )

    violations: list[dict[str, Any]] = []
    for component in component_audits:
        member_assignments = [
            assignment_by_part[part_id] for part_id in component["member_part_ids"]
        ]
        material_ids = sorted(
            {str(row.get("material_id")) for row in member_assignments}
        )
        parameter_hashes = sorted(
            {_canonical_sha256(row.get("parameters")) for row in member_assignments}
        )
        if len(material_ids) != 1 or len(parameter_hashes) != 1:
            violations.append(
                {
                    "component_id": component["component_id"],
                    "material_ids": material_ids,
                    "parameter_hashes": parameter_hashes,
                }
            )
    gate_status = "PASS" if not violations else "FAIL_CLOSED"
    if violations:
        raise PartIdProjectionError(
            "Part-ID same-coating consistency gate failed after plan authoring"
        )
    return {
        "schema_version": COATING_CONSISTENCY_SCHEMA_VERSION,
        "status": gate_status,
        "assignment_unit": "part_id",
        "palette_fusion_used": False,
        "part_material_groups_used": False,
        "policy": {
            "maximum_albedo_rgb_distance": maximum_albedo_distance,
            "minimum_component_parts": DEFAULT_MINIMUM_COATING_COMPONENT_PARTS,
            "minimum_anchor_trusted_pixels": (DEFAULT_MINIMUM_COATING_ANCHOR_PIXELS),
            "minimum_component_trusted_pixels": (
                DEFAULT_MINIMUM_COATING_COMPONENT_PIXELS
            ),
            "clustering": "deterministic_complete_link",
            "canonical_selection": (
                "maximum_trusted_visual_evidence_color_tunable_anchor"
            ),
            "canonical_color": "weighted_median_independent_part_id_albedo",
        },
        "components": component_audits,
        "summary": {
            "component_count": len(component_audits),
            "constrained_part_count": sum(
                len(component["member_part_ids"]) for component in component_audits
            ),
            "material_changed_part_count": len(changed_part_ids),
            "material_changed_part_ids": sorted(changed_part_ids),
            "skipped_counts": dict(sorted(skipped.items())),
            "violation_count": len(violations),
        },
    }


def build_part_id_material_plan(
    *,
    base_plan: Mapping[str, Any],
    evidence: Mapping[str, Any],
    retrieval_result: Mapping[str, Any],
    qwen_choices: Mapping[str, str] | None = None,
    qwen_confidences: Mapping[str, float] | None = None,
    qwen_material_predictions: Mapping[str, Mapping[str, Any]] | None = None,
    allow_color_parameters: bool = False,
    part_registry: Mapping[str, Any] | None = None,
    enforce_coating_consistency: bool = True,
    maximum_coating_albedo_distance: float = (DEFAULT_MAXIMUM_COATING_ALBEDO_DISTANCE),
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Replace observed parts independently while preserving hidden fallbacks."""

    assignments = base_plan.get("assignments")
    if not isinstance(assignments, list) or not assignments:
        raise PartIdProjectionError("base material plan has no assignments")
    base_by_part: dict[str, dict[str, Any]] = {}
    for raw in assignments:
        part_id = raw.get("part_id") if isinstance(raw, Mapping) else None
        if not isinstance(part_id, str) or not part_id or part_id in base_by_part:
            raise PartIdProjectionError("base plan has invalid or duplicate Part IDs")
        base_by_part[part_id] = copy.deepcopy(dict(raw))

    evidence_parts = evidence.get("parts")
    if not isinstance(evidence_parts, list):
        raise PartIdProjectionError("Part-ID evidence has no parts")
    evidence_by_part = {
        str(raw["part_id"]): raw
        for raw in evidence_parts
        if isinstance(raw, Mapping) and isinstance(raw.get("part_id"), str)
    }
    if set(evidence_by_part) != set(base_by_part):
        raise PartIdProjectionError(
            "Part-ID evidence does not exactly cover the base-plan Part IDs"
        )
    if retrieval_result.get("schema_version") != (
        "qwen-visual-material-retrieval-result/v1"
    ):
        raise PartIdProjectionError("unsupported visual retrieval result schema")
    retrieval_integrity = retrieval_result.get("integrity")
    retrieval_unsigned = copy.deepcopy(dict(retrieval_result))
    retrieval_unsigned.pop("integrity", None)
    if not isinstance(retrieval_integrity, Mapping) or retrieval_integrity.get(
        "result_sha256"
    ) != _canonical_sha256(retrieval_unsigned):
        raise PartIdProjectionError("visual retrieval result failed its integrity seal")
    raw_groups = retrieval_result.get("groups")
    if not isinstance(raw_groups, list):
        raise PartIdProjectionError("visual retrieval result has no groups")
    retrieval_by_part: dict[str, Mapping[str, Any]] = {}
    for raw in raw_groups:
        part_id = raw.get("group_id") if isinstance(raw, Mapping) else None
        if not isinstance(part_id, str) or part_id in retrieval_by_part:
            raise PartIdProjectionError(
                "visual retrieval result has invalid duplicate entity IDs"
            )
        retrieval_by_part[part_id] = raw
    expected_retrieval_parts = {
        part_id
        for part_id, raw in evidence_by_part.items()
        if raw.get("status") == "observed"
    }
    if set(retrieval_by_part) != expected_retrieval_parts:
        raise PartIdProjectionError(
            "visual retrieval entities do not exactly match observed Part IDs"
        )

    selected_qwen = dict(qwen_choices or {})
    unexpected_qwen = set(selected_qwen) - set(base_by_part)
    if unexpected_qwen:
        raise PartIdProjectionError(
            f"Qwen choices contain unknown Part IDs: {sorted(unexpected_qwen)}"
        )
    if selected_qwen and set(selected_qwen) != expected_retrieval_parts:
        raise PartIdProjectionError(
            "Qwen choices must exactly cover every photo-observed Part ID"
        )
    selected_qwen_confidences = dict(qwen_confidences or {})
    if set(selected_qwen_confidences) - set(selected_qwen):
        raise PartIdProjectionError(
            "Qwen confidences contain Part IDs without Qwen choices"
        )
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
        for value in selected_qwen_confidences.values()
    ):
        raise PartIdProjectionError("Qwen confidences must be finite unit floats")
    selected_material_predictions = {
        str(part_id): copy.deepcopy(dict(prediction))
        for part_id, prediction in (qwen_material_predictions or {}).items()
        if isinstance(part_id, str) and isinstance(prediction, Mapping)
    }
    if qwen_material_predictions is not None:
        if set(selected_material_predictions) != expected_retrieval_parts:
            raise PartIdProjectionError(
                "material predictions must exactly cover every observed Part ID"
            )
        for part_id, prediction in selected_material_predictions.items():
            if (
                prediction.get("part_id") != part_id
                or prediction.get("status")
                not in {"APPLYABLE", "INSUFFICIENT_EVIDENCE"}
                or not isinstance(prediction.get("catalog_family"), str)
                or not prediction["catalog_family"]
                or not isinstance(prediction.get("physical_substrate"), str)
                or not prediction["physical_substrate"]
                or not isinstance(prediction.get("material_species"), str)
                or not prediction["material_species"]
                or not isinstance(prediction.get("surface_treatment"), str)
                or not prediction["surface_treatment"]
                or not isinstance(prediction.get("optical_behavior"), str)
                or not prediction["optical_behavior"]
                or not isinstance(prediction.get("surface_finish"), str)
                or not prediction["surface_finish"]
                or prediction.get("identity_resolution")
                not in {
                    "exact_material",
                    "corresponding_material",
                    "insufficient_evidence",
                }
                or isinstance(prediction.get("substrate_confidence"), bool)
                or not isinstance(prediction.get("substrate_confidence"), (int, float))
                or isinstance(prediction.get("species_confidence"), bool)
                or not isinstance(prediction.get("species_confidence"), (int, float))
                or isinstance(prediction.get("treatment_confidence"), bool)
                or not isinstance(prediction.get("treatment_confidence"), (int, float))
                or isinstance(prediction.get("confidence"), bool)
                or not isinstance(prediction.get("confidence"), (int, float))
                or not math.isfinite(float(prediction["confidence"]))
                or not 0.0 <= float(prediction["confidence"]) <= 1.0
            ):
                raise PartIdProjectionError(
                    f"material prediction for {part_id} is invalid"
                )
    if not isinstance(allow_color_parameters, bool):
        raise PartIdProjectionError("allow_color_parameters must be boolean")
    if not isinstance(enforce_coating_consistency, bool):
        raise PartIdProjectionError("enforce_coating_consistency must be boolean")
    if (
        isinstance(maximum_coating_albedo_distance, bool)
        or not isinstance(maximum_coating_albedo_distance, (int, float))
        or not math.isfinite(float(maximum_coating_albedo_distance))
        or not 0.0 < float(maximum_coating_albedo_distance) <= math.sqrt(3.0)
    ):
        raise PartIdProjectionError(
            "maximum_coating_albedo_distance must be a positive RGB distance"
        )
    output_assignments: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for part_id in sorted(base_by_part):
        assignment = base_by_part[part_id]
        part_evidence = evidence_by_part[part_id]
        if part_evidence.get("status") != "observed":
            provenance = assignment.get("provenance")
            group_keys = {
                "canonical_group_id",
                "material_region_group_id",
                "group_id",
            }
            if (
                assignment.get("status") != "policy_fallback"
                or any(assignment.get(key) is not None for key in group_keys)
                or (
                    isinstance(provenance, Mapping)
                    and any(provenance.get(key) is not None for key in group_keys)
                )
            ):
                raise PartIdProjectionError(
                    f"unobserved Part ID {part_id} must use an independent "
                    "policy fallback, never a palette/material group"
                )
            output_assignments.append(assignment)
            audit_rows.append(
                {
                    "part_id": part_id,
                    "status": "unobserved_preserved",
                    "material_id": assignment.get("material_id"),
                    "evidence_view_ids": [],
                }
            )
            continue
        retrieved = retrieval_by_part.get(part_id)
        fused = (
            retrieved.get("fused_ranking") if isinstance(retrieved, Mapping) else None
        )
        if not isinstance(fused, list) or not fused:
            raise PartIdProjectionError(
                f"observed Part ID {part_id} has no fused retrieval ranking"
            )
        candidates = [
            str(row["material_id"])
            for row in fused
            if isinstance(row, Mapping) and isinstance(row.get("material_id"), str)
        ]
        if not candidates:
            raise PartIdProjectionError(
                f"observed Part ID {part_id} has no exact material candidates"
            )
        qwen_material = selected_qwen.get(part_id)
        material_prediction = selected_material_predictions.get(part_id)
        if qwen_material is not None and qwen_material not in candidates:
            raise PartIdProjectionError(
                f"Qwen selected a non-retrieved material for {part_id}: {qwen_material}"
            )
        observations = part_evidence["observations"]
        selected_observations = [
            observation
            for observation in observations
            if observation.get("selected_for_material_inference") is True
        ]
        if len(selected_observations) != 1:
            raise PartIdProjectionError(
                f"observed Part ID {part_id} must have exactly one selected view"
            )
        selected_observation = selected_observations[0]
        qwen_confidence = (
            float(selected_qwen_confidences.get(part_id, 0.85))
            if qwen_material is not None
            else None
        )
        if (
            qwen_material is not None
            and qwen_confidence < MINIMUM_APPLYABLE_REVIEW_CONFIDENCE
        ):
            # A direct CAD Part-ID decision may enter the render-QA candidate
            # set only at or above the explicit review floor.  Keeping the
            # baseline exact-cover assignment here avoids authoring a visually
            # implausible material from a tiny/ambiguous photo crop while still
            # assigning every CAD Part-ID a material.
            baseline_provenance = assignment.get("provenance")
            if not isinstance(baseline_provenance, dict):
                baseline_provenance = {}
                assignment["provenance"] = baseline_provenance
            baseline_provenance.update(
                {
                    "observed_part_id_qwen_selection_rejected": True,
                    "observed_part_id_qwen_rejection_reason": (
                        "qwen_confidence_below_applyable_review_floor"
                    ),
                    "rejected_qwen_material_id": qwen_material,
                    "rejected_qwen_confidence": qwen_confidence,
                    "applyable_review_confidence_floor": (
                        MINIMUM_APPLYABLE_REVIEW_CONFIDENCE
                    ),
                    "selected_reference_view_id_for_rejected_qwen": str(
                        selected_observation["view_id"]
                    ),
                    "candidate_material_ids": candidates,
                }
            )
            if material_prediction is not None:
                baseline_provenance["material_prediction"] = copy.deepcopy(
                    material_prediction
                )
            output_assignments.append(assignment)
            rejected_audit_row = {
                "part_id": part_id,
                "status": "observed_low_confidence_baseline_retained",
                "material_id": assignment.get("material_id"),
                "rejected_qwen_material_id": qwen_material,
                "rejected_qwen_confidence": qwen_confidence,
                "evidence_view_ids": [str(selected_observation["view_id"])],
            }
            if material_prediction is not None:
                rejected_audit_row["material_prediction"] = copy.deepcopy(
                    material_prediction
                )
            audit_rows.append(rejected_audit_row)
            continue
        selected = qwen_material or candidates[0]
        selected_row = next(row for row in fused if row.get("material_id") == selected)
        assignment["material_id"] = selected
        assignment["semantic"] = (
            "independent CAD Part-ID appearance matched to manually "
            "foreground-segmented reference pixels"
        )
        assignment["confidence"] = (
            qwen_confidence
            if qwen_material is not None and qwen_confidence is not None
            else min(
                0.849999,
                max(0.60, float(selected_row.get("score", 0.0)) * 12.0),
            )
        )
        assignment["evidence_views"] = [str(selected_observation["view_id"])]
        # The USD authoring contract requires auto assignments to carry at
        # least 0.85 confidence.  Preserve Qwen's reported confidence instead
        # of inflating it.  The lower-than-auto but applyable review band is
        # rendered provisionally; lower values retained the baseline above.
        assignment["status"] = (
            "auto"
            if qwen_material is not None and float(assignment["confidence"]) >= 0.85
            else "review"
        )
        for stale_field in _OBSERVED_ASSIGNMENT_RESET_FIELDS:
            assignment.pop(stale_field, None)
        assignment["provenance"] = {
            "assignment_unit": "part_id",
            "part_id": part_id,
            "material_region_group_id": None,
            "palette_fusion_used": False,
            "sam3_role": evidence.get("sam3_role"),
            "part_mask_authority": evidence.get("part_segmentation_authority"),
            "selected_reference_view_id": str(selected_observation["view_id"]),
            "selected_single_view_chromatic_coverage": copy.deepcopy(
                selected_observation.get("chromatic_coverage")
            ),
            "non_selected_views_retained_for_audit_only": sorted(
                str(observation["view_id"])
                for observation in observations
                if observation is not selected_observation
            ),
            "selection_basis": (
                "qwen_choice_within_part_id_siglip2_dinov2_mvinverse_candidates"
                if qwen_material is not None
                else "part_id_siglip2_dinov2_mvinverse_fused_rank_1"
            ),
            "selected_retrieval_rank": int(selected_row.get("rank", 1)),
            "qwen_confidence": (qwen_confidence if qwen_material is not None else None),
            "candidate_material_ids": candidates,
            "evidence_mask_sha256s": sorted([str(selected_observation["mask_sha256"])]),
        }
        if material_prediction is not None:
            assignment["provenance"]["material_prediction"] = copy.deepcopy(
                material_prediction
            )
        output_assignments.append(assignment)
        selected_audit_row = {
            "part_id": part_id,
            "status": "independently_selected",
            "material_id": selected,
            "selected_retrieval_rank": int(selected_row.get("rank", 1)),
            "qwen_reranked": qwen_material is not None,
            "evidence_view_ids": assignment["evidence_views"],
        }
        if material_prediction is not None:
            selected_audit_row["material_prediction"] = copy.deepcopy(
                material_prediction
            )
        audit_rows.append(selected_audit_row)

    coating_consistency = {
        "schema_version": COATING_CONSISTENCY_SCHEMA_VERSION,
        "status": "NOT_RUN",
        "reason": (
            "part_registry_not_supplied"
            if part_registry is None
            else "disabled_by_caller"
        ),
        "components": [],
        "summary": {
            "component_count": 0,
            "constrained_part_count": 0,
            "material_changed_part_count": 0,
            "material_changed_part_ids": [],
            "violation_count": 0,
        },
    }
    if enforce_coating_consistency and part_registry is not None:
        coating_consistency = _apply_part_id_coating_consistency(
            assignments=output_assignments,
            audit_rows=audit_rows,
            evidence_by_part=evidence_by_part,
            part_registry=part_registry,
            # Coating consistency may constrain the selected MDL, but it must
            # not author parameters. H0/H1 candidates are built afterwards
            # against the final material ID and remain unapplied until the
            # render tournament.
            allow_color_parameters=False,
            maximum_albedo_distance=float(maximum_coating_albedo_distance),
        )

    assignment_by_part = {str(row["part_id"]): row for row in output_assignments}
    audit_by_part = {str(row["part_id"]): row for row in audit_rows}
    h1_candidate_count = 0
    evidence_rejected_count = 0
    for part_id, part_evidence in sorted(evidence_by_part.items()):
        if part_evidence.get("status") != "observed":
            continue
        assignment = assignment_by_part[part_id]
        material_id = assignment.get("material_id")
        if not isinstance(material_id, str) or not material_id:
            raise PartIdProjectionError(
                f"observed Part ID {part_id} has no final material ID"
            )
        candidate_set, color_parameter_audit = _build_part_id_parameter_candidates(
            part_id=part_id,
            material_id=material_id,
            part_evidence=part_evidence,
            sam3_role=evidence.get("sam3_role"),
            enabled=allow_color_parameters,
        )
        if len(candidate_set["candidates"]) == 2:
            h1_candidate_count += 1
        if candidate_set["h1_status"] == "evidence_gate_rejected":
            evidence_rejected_count += 1
        provenance = assignment.get("provenance")
        if not isinstance(provenance, dict):
            raise PartIdProjectionError(
                f"observed Part ID {part_id} has invalid provenance"
            )
        provenance["mdl_parameter_candidates"] = copy.deepcopy(candidate_set)
        provenance["mdl_color_parameterization"] = copy.deepcopy(color_parameter_audit)
        audit_by_part[part_id]["mdl_parameter_candidates"] = copy.deepcopy(
            candidate_set
        )
        audit_by_part[part_id]["mdl_color_parameterization"] = copy.deepcopy(
            color_parameter_audit
        )

    provenance = copy.deepcopy(base_plan.get("provenance"))
    if not isinstance(provenance, dict):
        provenance = {}
    provenance.update(
        {
            "assignment_unit": "part_id",
            "palette_fusion_used_for_material_assignment": False,
            "human_sam3_role": "whole_workpiece_foreground_only",
            "part_id_evidence_sha256": evidence["integrity"]["document_sha256"],
            "retrieval_result_sha256": _canonical_sha256(retrieval_result),
            "part_id_color_parameters_enabled": False,
            "part_id_parameter_candidate_generation_enabled": (allow_color_parameters),
            "part_id_parameter_candidate_schema_version": (
                PARAMETER_CANDIDATE_SCHEMA_VERSION
            ),
            "part_id_parameter_candidates_require_render_comparison": True,
            "coating_consistency_enabled": (
                enforce_coating_consistency and part_registry is not None
            ),
            "coating_consistency_schema_version": coating_consistency["schema_version"],
        }
    )
    if selected_material_predictions:
        provenance["material_prediction_mode"] = "catalog_family_first"
    plan = {
        "schema_version": ASSIGNMENT_SCHEMA_VERSION,
        "assignment_unit": "part_id",
        "palette_fusion_used": False,
        "part_material_groups_used": False,
        "coating_consistency_used": coating_consistency["status"] == "PASS",
        "assignments": output_assignments,
        "provenance": provenance,
    }
    audit_unsigned = {
        "schema_version": "qwen-part-id-material-plan-audit/v1",
        "assignment_unit": "part_id",
        "palette_fusion_used": False,
        "base_plan_sha256": _canonical_sha256(base_plan),
        "part_id_evidence_sha256": evidence["integrity"]["document_sha256"],
        "retrieval_result_sha256": _canonical_sha256(retrieval_result),
        "output_plan_sha256": _canonical_sha256(plan),
        "coating_consistency_gate": coating_consistency,
        "parts": audit_rows,
        "summary": {
            "part_count": len(output_assignments),
            "independently_selected_count": sum(
                row["status"] == "independently_selected" for row in audit_rows
            ),
            "unobserved_preserved_count": sum(
                row["status"] == "unobserved_preserved" for row in audit_rows
            ),
            "observed_low_confidence_baseline_retained_count": sum(
                row["status"] == "observed_low_confidence_baseline_retained"
                for row in audit_rows
            ),
            "qwen_reranked_count": sum(
                bool(row.get("qwen_reranked")) for row in audit_rows
            ),
            "color_parameterized_count": 0,
            "native_h0_selected_count": sum(
                row.get("mdl_color_parameterization", {}).get("status")
                == "native_h0_selected"
                for row in audit_rows
            ),
            "h1_color_candidate_count": h1_candidate_count,
            "h1_evidence_rejected_count": evidence_rejected_count,
            "exact_cover": len(output_assignments) == len(base_by_part),
            "coating_component_count": coating_consistency["summary"][
                "component_count"
            ],
            "coating_constrained_part_count": coating_consistency["summary"][
                "constrained_part_count"
            ],
            "coating_material_changed_part_count": coating_consistency["summary"][
                "material_changed_part_count"
            ],
        },
    }
    if selected_material_predictions:
        audit_unsigned["summary"]["material_prediction_count"] = len(
            selected_material_predictions
        )
    audit = {
        **audit_unsigned,
        "integrity": {"document_sha256": _canonical_sha256(audit_unsigned)},
    }
    return plan, audit
