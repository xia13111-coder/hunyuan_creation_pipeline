"""Deterministic, fail-closed spatial gate for unattended material mapping.

Qwen's semantic agreement is useful, but it is not proof that a CAD part and a
photo material region occupy the same place.  This module adds that missing
proof without training another model:

* associate reference and canonical CAD views from foreground silhouettes;
* refine the selected image-space registration with deterministic ECC affine
  alignment and verify the resulting silhouette overlap;
* project each rendered part mask into every trusted reference view;
* classify the observed reference pixels against that view's local palette;
* require two independent spatial observations of the same canonical group.

The report is hash-bound to every image and JSON input.  The application step
rechecks those hashes.  The default application step remains downgrade-only;
high-assurance recovery is handled separately so its stricter policy and
provenance remain explicit.
"""

from __future__ import annotations

import colorsys
import copy
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
from PIL import Image, ImageOps

from qwen_material_pipeline.core.progress import ProgressCallback, report_progress
from qwen_material_pipeline.evidence.color_semantics import pixel_color_label
from qwen_material_pipeline.evidence.part_id_projection import (
    _register_similarity_mask,
)


SCHEMA_VERSION = "qwen-spatial-mapping-audit/v1"
GATE_AUDIT_SCHEMA_VERSION = "qwen-spatial-mapping-gate/v1"
ISOLATED_EVIDENCE_SCHEMA_VERSION = "qwen-isolated-part-evidence/v1"
MIN_CANONICAL_PALETTE_SOURCE_VIEWS = 2
CANONICAL_RENDER_VIEW_COUNT = 6
PROGRESS_SCOPE = "qwen_material_pipeline"

# Dark painted parts photographed on a black background need a separate
# foreground proof.  Ordinary colour labels cannot provide it: a projected
# mask that misses the object entirely also samples "black".  These fixed,
# scale-normalized thresholds were calibrated against real CAD/photo
# registrations and deliberately form a repair-only, fail-closed lane.
DARK_FOREGROUND_POLICY: dict[str, float | int] = {
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

DEFAULT_POLICY: dict[str, float | int] = {
    "normalized_mask_size": 128,
    # The refined silhouette/ECC pair is the authoritative registration
    # evidence.  Real CAD/photo pairs differ in small hoses and fixtures, so a
    # slightly lower coarse score is acceptable when both refined measures and
    # affine constraints pass.
    "minimum_alignment_score": 0.60,
    "minimum_d4_margin": 0.04,
    "minimum_render_margin": 0.04,
    # A dense view bank can contain distant poses with nearly identical
    # whole-asset silhouettes (especially for symmetric industrial
    # assemblies).  Keep a geometrically strong whole-asset registration as
    # an eligible, audited observation; the downstream CAD Part-ID/SAM3
    # intersection remains fail-closed for individual components.
    "strong_geometry_ambiguous_pose_iou": 0.75,
    "strong_geometry_ambiguous_pose_ecc": 0.75,
    # An exact or effectively exact tie remains unresolved: strong silhouette
    # overlap alone cannot distinguish symmetric opposite sides.
    "minimum_strong_geometry_ambiguity_margin": 0.005,
    # A dense view bank deliberately contains nearby camera samples.  A
    # low-margin alternative is harmless when it lies in the same local pose
    # neighbourhood; distant alternatives remain a fail-closed ambiguity.
    "maximum_equivalent_pose_degrees": 24.0,
    "pose_candidate_count": 5,
    "configuration_mismatch_iou_ceiling": 0.80,
    "configuration_mismatch_minimum_ecc": 0.80,
    "minimum_paired_direction_margin": 0.08,
    "minimum_refined_iou": 0.68,
    "minimum_refined_ecc": 0.74,
    "minimum_ecc_scale": 0.65,
    "maximum_ecc_scale": 1.55,
    "maximum_ecc_condition": 1.75,
    "maximum_ecc_rotation_degrees": 15.0,
    "maximum_ecc_shear": 0.25,
    "maximum_ecc_translation_ratio": 0.25,
    "minimum_visible_pixels": 256,
    # Preserve bounded diagnostic evidence for small CAD parts without
    # relaxing the automatic spatial gate.  A single diagnostic can never
    # authorize a mapping; the QA repair stage requires the same part/group in
    # two content- and pose-distinct trusted references.
    "minimum_diagnostic_visible_pixels": 32,
    # Renderer-authored isolated masks can prove that a tiny CAD part is
    # present without pretending that its enlarged crop contains more source
    # material pixels.  This lower floor is diagnostics-only and is enabled
    # only after the per-view source counts are cross-checked against the raw
    # part-ID renders.
    "minimum_isolated_source_visible_pixels": 12,
    "minimum_isolated_source_view_count": 2,
    "minimum_diagnostic_resolved_samples": 3,
    "minimum_diagnostic_consensus_ratio": 0.75,
    "minimum_diagnostic_color_share": 0.35,
    "minimum_diagnostic_color_margin": 0.10,
    # Cross-view canonical colour supplements are repair diagnostics only.
    # Requiring most projected pixels to overlap the inferred object
    # foreground prevents a black image background from becoming a black
    # material observation.
    "minimum_canonical_supplement_foreground_overlap": 0.50,
    "minimum_color_share": 0.35,
    "minimum_color_margin": 0.15,
    "minimum_spatial_support_views": 2,
    # A pair of high-confidence Qwen decisions from byte-distinct reference
    # images is an independent validation lane.  Spatial projection remains a
    # contradiction gate, but lack of pixel-perfect CAD/photo registration
    # must not erase otherwise consistent multiview evidence.
    "minimum_semantic_support_references": 2,
    "minimum_semantic_confidence": 0.85,
    "minimum_semantic_conflict_confidence": 0.60,
    "maximum_reference_phash_distance": 6,
}


class SpatialMappingError(ValueError):
    """Raised when spatial evidence is malformed, stale, or untrustworthy."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise SpatialMappingError(f"input is not canonical JSON: {exc}") from exc


def _sha256_document(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _read_object(
    value: str | Path | Mapping[str, Any], label: str
) -> tuple[dict[str, Any], Path | None]:
    if isinstance(value, Mapping):
        return copy.deepcopy(dict(value)), None
    try:
        path = Path(value).expanduser().resolve(strict=True)
    except (OSError, TypeError) as exc:
        raise SpatialMappingError(f"unable to resolve {label}: {value!r}") from exc
    if not path.is_file():
        raise SpatialMappingError(f"{label} is not a file: {path}")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SpatialMappingError(f"unable to read {label}: {path}") from exc
    if not isinstance(document, dict):
        raise SpatialMappingError(f"{label} must be a JSON object")
    return document, path


def _resolve_file(value: Any, owner: Path | None, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise SpatialMappingError(f"{label} must be a non-empty path")
    candidate = Path(value).expanduser()
    if not candidate.is_absolute() and owner is not None:
        candidate = owner.parent / candidate
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise SpatialMappingError(f"{label} does not exist: {candidate}") from exc
    if not resolved.is_file():
        raise SpatialMappingError(f"{label} is not a file: {resolved}")
    return resolved


def _open_bgr(path: Path, label: str) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise SpatialMappingError(f"unable to decode {label}: {path}")
    return image


def _normalized_pixel_sha256(image: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(tuple(int(value) for value in image.shape)).encode("ascii"))
    digest.update(image.tobytes(order="C"))
    return digest.hexdigest()


def _perceptual_hash(image: np.ndarray) -> str:
    """Return a deterministic 64-bit DCT hash for near-duplicate clustering."""

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    normalized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    frequencies = cv2.dct(normalized.astype(np.float32))[:8, :8].reshape(-1)
    median = float(np.median(frequencies[1:]))
    bits = 0
    for value in frequencies:
        bits = (bits << 1) | int(float(value) >= median)
    return f"{bits:016x}"


def _perceptual_clusters(
    hashes_by_view: Mapping[str, str],
    maximum_distance: int,
) -> dict[str, str]:
    view_ids = sorted(hashes_by_view)
    parents = list(range(len(view_ids)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[max(left_root, right_root)] = min(left_root, right_root)

    for left in range(len(view_ids)):
        left_hash = int(hashes_by_view[view_ids[left]], 16)
        for right in range(left + 1, len(view_ids)):
            right_hash = int(hashes_by_view[view_ids[right]], 16)
            if (left_hash ^ right_hash).bit_count() <= maximum_distance:
                union(left, right)
    members: dict[int, list[str]] = {}
    for index, view_id in enumerate(view_ids):
        members.setdefault(find(index), []).append(view_id)
    cluster_ids = {
        root: f"PH{cluster_index:02d}"
        for cluster_index, root in enumerate(sorted(members), start=1)
    }
    return {view_id: cluster_ids[find(index)] for index, view_id in enumerate(view_ids)}


def _part_color(part_id: str) -> tuple[int, int, int]:
    number = int(part_id[1:]) if part_id[1:].isdigit() else sum(map(ord, part_id))
    hue = (number * 0.618033988749895) % 1.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.72, 0.95)
    return int(red * 255), int(green * 255), int(blue * 255)


def _clean_components(mask: np.ndarray, *, minimum_area: int = 40) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8) * 255
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, 8)
    output = np.zeros_like(binary)
    height, width = binary.shape
    for index in range(1, count):
        x, y, component_width, component_height, area = [
            int(value) for value in stats[index]
        ]
        if area < minimum_area:
            continue
        # Camera overlays and floor/horizon artefacts are often long, one-pixel
        # lines.  They must not dominate silhouette association.
        if component_width >= int(width * 0.80) and component_height <= 3:
            continue
        if component_height >= int(height * 0.80) and component_width <= 3:
            continue
        output[labels == index] = 255
    return output


def _reference_foreground(image: np.ndarray) -> np.ndarray:
    border = np.concatenate(
        (image[0, :, :], image[-1, :, :], image[:, 0, :], image[:, -1, :]),
        axis=0,
    ).astype(np.float32)
    background = np.median(border, axis=0)
    border_distances = np.linalg.norm(border - background[None, :], axis=1)
    threshold = min(
        45.0,
        max(12.0, float(np.percentile(border_distances, 95)) + 6.0),
    )
    distances = np.linalg.norm(
        image.astype(np.float32) - background[None, None, :],
        axis=2,
    )
    mask = (distances >= threshold).astype(np.uint8) * 255
    # Remove thin grid/axis overlays before closing tiny capture gaps.  A 3 px
    # opening is insufficient for JPEG-compressed CAD-viewer axes: their colour
    # fringes can remain connected to the workpiece and stretch its bounding
    # box across the whole frame.  Five pixels is still small relative to the
    # minimum supported reference image, while deliberately discarding hoses
    # and annotations that are too thin to be reliable silhouette evidence.
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    cleaned = _clean_components(mask)
    if not np.any(cleaned):
        raise SpatialMappingError(
            "automatic background segmentation produced no reference foreground"
        )
    return cleaned


def _reference_foreground_from_manifest(
    *,
    source: Mapping[str, Any],
    manifest_path: Path | None,
    image: np.ndarray,
    view_id: str,
) -> tuple[np.ndarray, Path | None, str]:
    """Return the sealed foreground used by camera calibration when present.

    Camera calibration and Part-ID projection are two consumers of the same
    reference geometry.  Re-segmenting the RGB independently here used to
    create a second, incompatible coordinate contract: the camera could pass
    against the SAM foreground while spatial ECC rejected the identical
    camera against a border-colour heuristic.  A manifest-declared mask is
    therefore authoritative and fail-closed.  Legacy manifests without one
    retain the deterministic RGB fallback.
    """

    raw_mask = source.get("palette_mask")
    if raw_mask is None:
        return _reference_foreground(image), None, "deterministic_rgb_fallback"
    mask_path = _resolve_file(
        raw_mask,
        manifest_path,
        f"reference foreground mask {view_id}",
    )
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise SpatialMappingError(
            f"unable to decode reference foreground mask {view_id}: {mask_path}"
        )
    if mask.shape != image.shape[:2]:
        raise SpatialMappingError(
            f"reference foreground mask {view_id} shape {mask.shape} does not "
            f"match image shape {image.shape[:2]}"
        )
    foreground = (mask > 0).astype(np.uint8) * 255
    _bbox(foreground, f"reference foreground mask {view_id}")
    return foreground, mask_path, "manifest_palette_mask"


def _cad_foreground(
    rgb: np.ndarray,
    part_ids: np.ndarray,
    part_colors_bgr: Sequence[np.ndarray],
) -> np.ndarray:
    # Compare packed 24-bit IDs in one vectorized pass.  A STEP assembly can
    # contain hundreds of Part IDs; scanning the full image once per colour
    # made continuous camera search unnecessarily quadratic.
    packed_ids = (
        part_ids[:, :, 0].astype(np.uint32)
        | (part_ids[:, :, 1].astype(np.uint32) << 8)
        | (part_ids[:, :, 2].astype(np.uint32) << 16)
    )
    packed_colors = np.asarray(
        [
            int(color[0])
            | (int(color[1]) << 8)
            | (int(color[2]) << 16)
            for color in part_colors_bgr
        ],
        dtype=np.uint32,
    )
    exact = np.isin(packed_ids, packed_colors).astype(np.uint8) * 255

    border = np.concatenate(
        (rgb[0, :, :], rgb[-1, :, :], rgb[:, 0, :], rgb[:, -1, :]), axis=0
    ).astype(np.float32)
    background = np.median(border, axis=0)
    distance = np.linalg.norm(
        rgb.astype(np.float32) - background[None, None, :], axis=2
    )
    appearance = (distance >= 20.0).astype(np.uint8) * 255
    appearance = cv2.morphologyEx(
        appearance,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    return _clean_components(cv2.bitwise_or(exact, appearance))


def _bbox(mask: np.ndarray, label: str) -> tuple[int, int, int, int]:
    if not np.any(mask):
        raise SpatialMappingError(f"{label} foreground mask is empty")
    x, y, width, height = cv2.boundingRect(mask)
    if width < 2 or height < 2:
        raise SpatialMappingError(f"{label} foreground mask is degenerate")
    return int(x), int(y), int(width), int(height)


def _normalized_mask(mask: np.ndarray, size: int) -> tuple[np.ndarray, float]:
    x, y, width, height = _bbox(mask, "alignment")
    crop = mask[y : y + height, x : x + width]
    # Non-uniform bbox normalization removes camera scale/translation while the
    # original aspect ratio remains an explicit, separately weighted feature.
    normalized = cv2.resize(crop, (size, size), interpolation=cv2.INTER_NEAREST)
    return normalized > 0, width / height


def _pose_search_mask(mask: np.ndarray, size: int) -> np.ndarray:
    """Return a bounded binary mask for dense camera-pose candidate scoring."""

    normalized, _aspect = _normalized_mask(mask, size)
    return normalized.astype(np.uint8) * 255


def _alignment_metrics(
    left_mask: np.ndarray, right_mask: np.ndarray, size: int
) -> dict[str, float]:
    left, left_aspect = _normalized_mask(left_mask, size)
    right, right_aspect = _normalized_mask(right_mask, size)
    intersection = int(np.count_nonzero(left & right))
    union = int(np.count_nonzero(left | right))
    iou = intersection / union if union else 0.0

    kernel = np.ones((3, 3), dtype=np.uint8)
    left_edge = (
        cv2.morphologyEx(left.astype(np.uint8) * 255, cv2.MORPH_GRADIENT, kernel) > 0
    )
    right_edge = (
        cv2.morphologyEx(right.astype(np.uint8) * 255, cv2.MORPH_GRADIENT, kernel) > 0
    )
    if np.any(left_edge) and np.any(right_edge):
        distance_to_right = cv2.distanceTransform(
            (~right_edge).astype(np.uint8), cv2.DIST_L2, 3
        )
        distance_to_left = cv2.distanceTransform(
            (~left_edge).astype(np.uint8), cv2.DIST_L2, 3
        )
        # Dividing the symmetric sum by four expresses Chamfer distance in the
        # half-resolution convention used by the fixed exp(-d/8) policy.
        chamfer = (
            float(np.mean(distance_to_right[left_edge]))
            + float(np.mean(distance_to_left[right_edge]))
        ) / 4.0
        chamfer_score = math.exp(-chamfer / 8.0)
    else:
        chamfer = float(size)
        chamfer_score = 0.0

    tolerance_kernel = np.ones((7, 7), dtype=np.uint8)
    dilated_left = cv2.dilate(left.astype(np.uint8), tolerance_kernel) > 0
    dilated_right = cv2.dilate(right.astype(np.uint8), tolerance_kernel) > 0
    tolerance_overlap = 0.5 * (
        np.count_nonzero(left & dilated_right) / max(1, np.count_nonzero(left))
        + np.count_nonzero(right & dilated_left) / max(1, np.count_nonzero(right))
    )
    aspect_score = math.exp(-abs(math.log(max(1e-9, left_aspect / right_aspect))))
    score = (
        0.40 * iou
        + 0.25 * chamfer_score
        + 0.20 * tolerance_overlap
        + 0.15 * aspect_score
    )
    return {
        "score": round(float(score), 8),
        "silhouette_iou": round(float(iou), 8),
        "chamfer_pixels": round(float(chamfer), 8),
        "chamfer_score": round(float(chamfer_score), 8),
        "tolerance_overlap": round(float(tolerance_overlap), 8),
        "aspect_score": round(float(aspect_score), 8),
    }


def _validated_direction(value: Any, label: str) -> list[float] | None:
    if value is None:
        return None
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 3
        or any(
            isinstance(component, bool)
            or not isinstance(component, (int, float))
            or not math.isfinite(float(component))
            for component in value
        )
    ):
        raise SpatialMappingError(f"{label} must contain three finite numbers")
    normalized = np.asarray([float(component) for component in value], dtype=np.float64)
    length = float(np.linalg.norm(normalized))
    if length <= 1e-12:
        raise SpatialMappingError(f"{label} cannot be zero")
    return [round(float(component / length), 10) for component in normalized]


def _pose_angle_degrees(
    left: Sequence[float] | None,
    right: Sequence[float] | None,
) -> float | None:
    if left is None or right is None:
        return None
    dot = sum(float(left[index]) * float(right[index]) for index in range(3))
    return round(math.degrees(math.acos(max(-1.0, min(1.0, dot)))), 8)


def _mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    left_binary = left > 0
    right_binary = right > 0
    union = int(np.count_nonzero(left_binary | right_binary))
    if union == 0:
        return 0.0
    return float(np.count_nonzero(left_binary & right_binary) / union)


def _bbox_affine(
    reference_mask: np.ndarray,
    render_mask: np.ndarray,
) -> np.ndarray:
    """Return the deterministic render-to-reference bbox normalization."""

    ref_x, ref_y, ref_width, ref_height = _bbox(reference_mask, "reference")
    cad_x, cad_y, cad_width, cad_height = _bbox(render_mask, "render")
    scale_x = ref_width / cad_width
    scale_y = ref_height / cad_height
    return np.asarray(
        [
            [scale_x, 0.0, ref_x - cad_x * scale_x],
            [0.0, scale_y, ref_y - cad_y * scale_y],
        ],
        dtype=np.float32,
    )


def _refine_projection_affine_ecc(
    reference_mask: np.ndarray,
    render_mask: np.ndarray,
    policy: Mapping[str, float | int],
) -> dict[str, Any]:
    """Refine bbox registration with ECC and return auditable matrices.

    OpenCV's ECC warp is stored in its native inverse-map convention.  Part
    masks are projected by first applying ``bbox_affine`` and then applying
    ``ecc_warp`` with ``WARP_INVERSE_MAP``.  A refinement that is non-finite or
    materially reduces IoU is rejected and cannot become trusted evidence.
    """

    affine = _bbox_affine(reference_mask, render_mask)
    output_size = (int(reference_mask.shape[1]), int(reference_mask.shape[0]))
    bbox_warp = cv2.warpAffine(
        render_mask,
        affine,
        output_size,
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    before_iou = _mask_iou(reference_mask, bbox_warp)
    ecc_warp = np.eye(2, 3, dtype=np.float32)
    status = "success"
    correlation = 0.0
    refined = bbox_warp
    diagnostics: dict[str, float | bool | list[str]] = {
        "determinant": 1.0,
        "minimum_scale": 1.0,
        "maximum_scale": 1.0,
        "condition_number": 1.0,
        "rotation_degrees": 0.0,
        "shear": 0.0,
        "translation_ratio": 0.0,
        "constraints_passed": True,
        "constraint_failures": [],
    }
    try:
        correlation, candidate_warp = cv2.findTransformECC(
            (reference_mask > 0).astype(np.float32),
            (bbox_warp > 0).astype(np.float32),
            ecc_warp.copy(),
            cv2.MOTION_AFFINE,
            (
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                500,
                1e-7,
            ),
            None,
            5,
        )
        if not np.isfinite(candidate_warp).all() or not math.isfinite(
            float(correlation)
        ):
            raise SpatialMappingError("ECC returned non-finite registration values")
        linear = candidate_warp[:, :2].astype(np.float64)
        determinant = float(np.linalg.det(linear))
        left, singular_values, right = np.linalg.svd(linear)
        rotation = left @ right
        rotation_degrees = math.degrees(
            math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
        )
        stretch = right.T @ np.diag(singular_values) @ right
        mean_stretch = max(1e-9, 0.5 * float(stretch[0, 0] + stretch[1, 1]))
        shear = abs(float(stretch[0, 1])) / mean_stretch
        minimum_scale = float(np.min(singular_values))
        maximum_scale = float(np.max(singular_values))
        condition = maximum_scale / max(1e-9, minimum_scale)
        translation_ratio = max(
            abs(float(candidate_warp[0, 2])) / max(1, reference_mask.shape[1]),
            abs(float(candidate_warp[1, 2])) / max(1, reference_mask.shape[0]),
        )
        failures: list[str] = []
        if determinant <= 0.0:
            failures.append("reflection_or_singular")
        if minimum_scale < float(policy["minimum_ecc_scale"]):
            failures.append("minimum_scale")
        if maximum_scale > float(policy["maximum_ecc_scale"]):
            failures.append("maximum_scale")
        if condition > float(policy["maximum_ecc_condition"]):
            failures.append("condition_number")
        if abs(rotation_degrees) > float(policy["maximum_ecc_rotation_degrees"]):
            failures.append("rotation")
        if shear > float(policy["maximum_ecc_shear"]):
            failures.append("shear")
        if translation_ratio > float(policy["maximum_ecc_translation_ratio"]):
            failures.append("translation")
        diagnostics = {
            "determinant": round(determinant, 8),
            "minimum_scale": round(minimum_scale, 8),
            "maximum_scale": round(maximum_scale, 8),
            "condition_number": round(condition, 8),
            "rotation_degrees": round(rotation_degrees, 8),
            "shear": round(shear, 8),
            "translation_ratio": round(translation_ratio, 8),
            "constraints_passed": not failures,
            "constraint_failures": failures,
        }
        if failures:
            status = "rejected_transform_constraints"
            correlation = 0.0
            raise SpatialMappingError("ECC transform exceeds fail-closed constraints")
        candidate = cv2.warpAffine(
            bbox_warp,
            candidate_warp,
            output_size,
            flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        candidate_iou = _mask_iou(reference_mask, candidate)
        if candidate_iou + 0.02 < before_iou:
            status = "rejected_iou_regression"
            correlation = 0.0
        else:
            ecc_warp = candidate_warp.astype(np.float32)
            refined = candidate
    except (cv2.error, SpatialMappingError):
        if status == "success":
            status = "failed"
        correlation = 0.0

    refined_iou = _mask_iou(reference_mask, refined)
    return {
        "bbox_affine": [[round(float(value), 10) for value in row] for row in affine],
        "ecc_warp": [[round(float(value), 10) for value in row] for row in ecc_warp],
        "ecc_status": status,
        "ecc_correlation": round(float(correlation), 8),
        "projection_iou_before": round(float(before_iou), 8),
        "projection_iou": round(float(refined_iou), 8),
        "ecc_transform_audit": diagnostics,
    }


def _refine_projection(
    reference_mask: np.ndarray,
    render_mask: np.ndarray,
    policy: Mapping[str, float | int],
) -> dict[str, Any]:
    """Register the complete rendered asset to one reference silhouette.

    A single bounded similarity transform is fitted for the whole CAD render:
    uniform scale, in-plane rotation, and translation.  Every Part-ID later
    inherits this exact matrix, preserving the assembly's relative geometry.
    The legacy ``bbox_affine``/``ecc_warp`` fields remain populated for
    downstream compatibility, but no per-axis scale, shear, or per-part warp
    is introduced.
    """

    legacy_bbox = _bbox_affine(reference_mask, render_mask)
    output_size = (int(reference_mask.shape[1]), int(reference_mask.shape[0]))
    bbox_warp = cv2.warpAffine(
        render_mask,
        legacy_bbox,
        output_size,
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    before_iou = _mask_iou(reference_mask, bbox_warp)
    registered, registration = _register_similarity_mask(
        render_mask,
        reference_mask,
    )
    raw_scale = float(registration["uniform_scale"])
    # The registration matrix maps render pixels into reference pixels.  Its
    # raw scale therefore also contains the arbitrary raster-resolution ratio
    # (for example a 256 px search render scored against a 582 px reference).
    # Compare physical residual scale only after removing that ratio; without
    # this normalization the exact same camera was invalid at search
    # resolution and valid at delivery resolution.
    resolution_scale_normalization = math.sqrt(
        float(render_mask.size) / max(1.0, float(reference_mask.size))
    )
    scale = raw_scale * resolution_scale_normalization
    rotation_degrees = float(registration["rotation_degrees"])
    matrix = np.asarray(registration["affine_2x3"], dtype=np.float32)
    # ``affine_2x3`` maps between different raster canvases, so its absolute
    # translation includes the expected recentering caused by resolution and
    # scale.  Only the optimizer's residual offset is a physical alignment
    # constraint.
    residual_translation = registration.get("translation_offset_xy")
    if (
        not isinstance(residual_translation, Sequence)
        or isinstance(residual_translation, (str, bytes))
        or len(residual_translation) != 2
    ):
        raise SpatialMappingError(
            "whole-asset registration lacks residual translation evidence"
        )
    translation_ratio = max(
        abs(float(residual_translation[0])) / max(1, reference_mask.shape[1]),
        abs(float(residual_translation[1])) / max(1, reference_mask.shape[0]),
    )
    failures: list[str] = []
    if scale < float(policy["minimum_ecc_scale"]):
        failures.append("minimum_scale")
    if scale > float(policy["maximum_ecc_scale"]):
        failures.append("maximum_scale")
    if abs(rotation_degrees) > float(policy["maximum_ecc_rotation_degrees"]):
        failures.append("rotation")
    if translation_ratio > float(policy["maximum_ecc_translation_ratio"]):
        failures.append("translation")
    try:
        correlation = float(
            cv2.computeECC(
                (reference_mask > 0).astype(np.float32),
                (registered > 0).astype(np.float32),
            )
        )
    except cv2.error:
        correlation = 0.0
        failures.append("correlation")
    if not math.isfinite(correlation):
        correlation = 0.0
        failures.append("correlation")
    diagnostics: dict[str, Any] = {
        "registration_mode": (
            "whole_asset_uniform_scale_rotation_translation"
        ),
        "determinant": round(raw_scale * raw_scale, 8),
        "minimum_scale": round(scale, 8),
        "maximum_scale": round(scale, 8),
        "raw_uniform_scale": round(raw_scale, 8),
        "resolution_scale_normalization": round(
            resolution_scale_normalization,
            8,
        ),
        "condition_number": 1.0,
        "rotation_degrees": round(rotation_degrees, 8),
        "shear": 0.0,
        "translation_ratio": round(translation_ratio, 8),
        "constraints_passed": not failures,
        "constraint_failures": sorted(set(failures)),
        "registration": registration,
    }
    return {
        # This matrix is a direct render-to-reference transform.  The second
        # compatibility transform is identity, so existing projection
        # consumers apply exactly one whole-asset similarity operation.
        "bbox_affine": [
            [round(float(value), 10) for value in row]
            for row in matrix.tolist()
        ],
        "ecc_warp": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        "global_similarity_affine": [
            [round(float(value), 10) for value in row]
            for row in matrix.tolist()
        ],
        "ecc_status": "success" if not failures else "rejected_transform_constraints",
        "ecc_correlation": round(correlation, 8) if not failures else 0.0,
        "projection_iou_before": round(float(before_iou), 8),
        "projection_iou": round(float(registration["iou"]), 8),
        "ecc_transform_audit": diagnostics,
    }


def _candidate_render_ids(
    reference_id: str, render_ids: Sequence[str]
) -> tuple[list[str], bool]:
    tokens = set(filter(None, re.split(r"[^a-z0-9]+", reference_id.lower())))
    render_set = set(render_ids)
    requested: list[str] | None = None
    auxiliary_family = False
    if tokens & {"top", "above", "overhead"}:
        requested = ["top"]
    elif tokens & {"iso", "isometric", "oblique", "perspective"}:
        requested = ["iso"]
    elif "front" in tokens:
        requested = ["front"]
    elif tokens & {"rear", "back"}:
        requested = ["rear"]
    elif "left" in tokens:
        requested = ["left"]
    elif "right" in tokens:
        requested = ["right"]
    elif "side" in tokens:
        requested = ["left", "right"]
        auxiliary_family = True
    candidates = [
        item for item in (requested or list(render_ids)) if item in render_set
    ]
    return (candidates or list(render_ids)), auxiliary_family


def _associate_views(
    references: list[dict[str, Any]],
    renders: list[dict[str, Any]],
    policy: Mapping[str, float | int],
    *,
    progress_callback: ProgressCallback | None = None,
) -> list[dict[str, Any]]:
    render_by_id = {record["view_id"]: record for record in renders}
    render_ids = sorted(render_by_id)
    alignment_pair_total = len(references) * len(render_ids)
    if alignment_pair_total:
        report_progress(
            progress_callback,
            scope=PROGRESS_SCOPE,
            stage="spatial_view_alignment",
            state="start",
            current=0,
            total=alignment_pair_total,
            unit="pairs",
            detail="spatial reference/render alignment started",
        )
    else:
        report_progress(
            progress_callback,
            scope=PROGRESS_SCOPE,
            stage="spatial_view_alignment",
            state="start",
            detail="spatial reference/render alignment has no candidate pairs",
        )
    dense_pose_search = len(render_ids) > CANONICAL_RENDER_VIEW_COUNT
    # The ordinary coarse metric remains at the policy's historic resolution,
    # while dense ECC search gets enough contour detail for small CAD features.
    pose_search_size = max(256, int(policy["normalized_mask_size"]))
    evaluated: list[dict[str, Any]] = []
    alignment_pair_index = 0
    for reference in references:
        declared_candidates, _paired_direction_family = _candidate_render_ids(
            reference["view_id"], render_ids
        )
        render_records: list[dict[str, Any]] = []
        bounded_reference = (
            _pose_search_mask(reference["foreground"], pose_search_size)
            if dense_pose_search
            else reference["foreground"]
        )
        # View IDs are useful audit metadata, not geometry truth.  CAD may have
        # any global rigid orientation, so evaluate every render.  Dense pose
        # banks are compared on bounded masks; only the selected poses are
        # refined at source-photo resolution below.
        for render_id in render_ids:
            rotations: list[dict[str, Any]] = []
            for quarter_turns in range(4):
                rotated = np.rot90(
                    render_by_id[render_id]["foreground"], quarter_turns
                ).copy()
                metrics = _alignment_metrics(
                    reference["foreground"],
                    rotated,
                    int(policy["normalized_mask_size"]),
                )
                projection = _refine_projection(
                    bounded_reference,
                    (
                        _pose_search_mask(rotated, pose_search_size)
                        if dense_pose_search
                        else rotated
                    ),
                    policy,
                )
                projection_score = 0.5 * float(metrics["score"]) + 0.5 * float(
                    projection["projection_iou"]
                )
                rotations.append(
                    {
                        "quarter_turns_ccw": quarter_turns,
                        **metrics,
                        **projection,
                        "projection_score": round(projection_score, 8),
                    }
                )
            rotations.sort(
                key=lambda item: (
                    -float(item["projection_score"]),
                    int(item["quarter_turns_ccw"]),
                )
            )
            render_records.append(
                {
                    "render_view_id": render_id,
                    "analysis_direction": render_by_id[render_id].get(
                        "analysis_direction"
                    ),
                    "world_direction": render_by_id[render_id].get("world_direction"),
                    "camera_up_axis": render_by_id[render_id].get("camera_up_axis"),
                    "camera_position": render_by_id[render_id].get("camera_position"),
                    "focal_length_mm": render_by_id[render_id].get("focal_length_mm"),
                    "best": rotations[0],
                    "runner_up_rotation_score": rotations[1]["projection_score"],
                    "d4_margin": round(
                        float(rotations[0]["projection_score"])
                        - float(rotations[1]["projection_score"]),
                        8,
                    ),
                    "rotations": rotations,
                }
            )
            alignment_pair_index += 1
            report_progress(
                progress_callback,
                scope=PROGRESS_SCOPE,
                stage="spatial_view_alignment",
                state="update",
                current=alignment_pair_index,
                total=alignment_pair_total,
                unit="pairs",
                detail=(
                    f"spatial alignment {reference['view_id']}/{render_id} "
                    "evaluated"
                ),
            )
        render_records.sort(
            key=lambda item: (
                -float(item["best"]["projection_score"]),
                item["render_view_id"],
            )
        )
        evaluated.append(
            {
                "reference": reference,
                "render_records": render_records,
                "declared_candidates": declared_candidates,
            }
        )

    score_tables = [
        {
            record["render_view_id"]: float(record["best"]["projection_score"])
            for record in item["render_records"]
        }
        for item in evaluated
    ]

    def best_global_assignment(
        forbidden: tuple[int, str] | None = None,
    ) -> tuple[float, tuple[str | None, ...]]:
        # Dynamic programming produces a maximum-weight one-to-one assignment.
        # It also handles more photos than canonical render directions by
        # leaving the lowest-value extras unmatched.
        states: dict[int, tuple[float, tuple[str | None, ...]]] = {0: (0.0, ())}
        for reference_index, scores in enumerate(score_tables):
            next_states: dict[int, tuple[float, tuple[str | None, ...]]] = {}

            def retain(
                used_mask: int,
                score: float,
                assignment: tuple[str | None, ...],
            ) -> None:
                current = next_states.get(used_mask)
                if current is None:
                    next_states[used_mask] = (score, assignment)
                    return
                current_score, current_assignment = current
                candidate_key = (
                    round(score, 12),
                    sum(value is not None for value in assignment),
                    tuple(value or "~" for value in assignment),
                )
                current_key = (
                    round(current_score, 12),
                    sum(value is not None for value in current_assignment),
                    tuple(value or "~" for value in current_assignment),
                )
                if candidate_key[:2] > current_key[:2] or (
                    candidate_key[:2] == current_key[:2]
                    and candidate_key[2] < current_key[2]
                ):
                    next_states[used_mask] = (score, assignment)

            for used_mask, (score, assignment) in states.items():
                retain(used_mask, score, (*assignment, None))
                for render_index, render_id in enumerate(render_ids):
                    if used_mask & (1 << render_index):
                        continue
                    if forbidden == (reference_index, render_id):
                        continue
                    retain(
                        used_mask | (1 << render_index),
                        score + scores[render_id],
                        (*assignment, render_id),
                    )
            states = next_states
        best_score = max(round(item[0], 12) for item in states.values())
        score_tied = [
            item for item in states.values() if round(item[0], 12) == best_score
        ]
        best_count = max(
            sum(value is not None for value in item[1]) for item in score_tied
        )
        finalists = [
            item
            for item in score_tied
            if sum(value is not None for value in item[1]) == best_count
        ]
        return min(
            finalists,
            key=lambda item: tuple(value or "~" for value in item[1]),
        )

    global_score, assignment = best_global_assignment()
    output: list[dict[str, Any]] = []
    for reference_index, item in enumerate(evaluated):
        reference = item["reference"]
        render_records = item["render_records"]
        assigned_render_id = assignment[reference_index]
        unmatched = assigned_render_id is None
        best = render_records[0]
        if assigned_render_id is not None:
            best = next(
                record
                for record in render_records
                if record["render_view_id"] == assigned_render_id
            )
        pose_search_best = best["best"]
        if dense_pose_search:
            full_resolution_rotated = np.rot90(
                render_by_id[best["render_view_id"]]["foreground"],
                int(pose_search_best["quarter_turns_ccw"]),
            ).copy()
            full_resolution_projection = _refine_projection(
                reference["foreground"],
                full_resolution_rotated,
                policy,
            )
            best = {
                **best,
                "best": {
                    **pose_search_best,
                    **full_resolution_projection,
                    "projection_score": round(
                        0.5 * float(pose_search_best["score"])
                        + 0.5
                        * float(full_resolution_projection["projection_iou"]),
                        8,
                    ),
                },
            }
        raw_candidates = sorted(
            (
                {
                    "render_view_id": record["render_view_id"],
                    "best": max(
                        record["rotations"], key=lambda item: float(item["score"])
                    ),
                }
                for record in render_records
            ),
            key=lambda item: (-float(item["best"]["score"]), item["render_view_id"]),
        )
        raw_best = raw_candidates[0]
        raw_runner = raw_candidates[1] if len(raw_candidates) > 1 else None
        raw_render_margin = (
            round(
                float(raw_best["best"]["score"]) - float(raw_runner["best"]["score"]),
                8,
            )
            if raw_runner is not None
            else None
        )
        selected_raw_rotations = sorted(
            best["rotations"], key=lambda item: -float(item["score"])
        )
        raw_d4_margin = round(
            float(selected_raw_rotations[0]["score"])
            - float(selected_raw_rotations[1]["score"]),
            8,
        )
        alternative_score, alternative_assignment = best_global_assignment(
            (reference_index, best["render_view_id"])
        )
        render_margin = round(global_score - alternative_score, 8)
        alternative_render_id = alternative_assignment[reference_index]
        alternative_record = (
            next(
                (
                    record
                    for record in render_records
                    if record["render_view_id"] == alternative_render_id
                ),
                None,
            )
            if alternative_render_id is not None
            else None
        )
        alternative_pose_degrees = (
            _pose_angle_degrees(
                best.get("analysis_direction"),
                alternative_record.get("analysis_direction"),
            )
            if alternative_record is not None
            else None
        )
        reasons: list[str] = []
        warnings: list[str] = []
        if unmatched:
            reasons.append("global_one_to_one_assignment_unmatched")
        if float(best["best"]["score"]) < float(policy["minimum_alignment_score"]):
            reasons.append("alignment_score_below_threshold")
        if float(best["best"]["projection_iou"]) < float(policy["minimum_refined_iou"]):
            reasons.append("refined_projection_iou_below_threshold")
        if float(best["best"]["ecc_correlation"]) < float(
            policy["minimum_refined_ecc"]
        ):
            reasons.append("ecc_correlation_below_threshold")
        if best["best"]["ecc_status"] != "success":
            reasons.append("ecc_refinement_not_successful")
        # The global refined assignment is authoritative only after every
        # score/IoU/ECC/D4/margin gate below passes.  A disagreement with the
        # local coarse-shape winner is therefore useful audit information, but
        # is not independently grounds to discard a strong one-to-one pose.
        if raw_best["render_view_id"] != best["render_view_id"]:
            warnings.append("refined_render_direction_disagrees_with_raw_shape")
        if int(selected_raw_rotations[0]["quarter_turns_ccw"]) != int(
            best["best"]["quarter_turns_ccw"]
        ):
            warnings.append("refined_rotation_disagrees_with_raw_shape")
        if float(best["d4_margin"]) < float(policy["minimum_d4_margin"]):
            reasons.append("d4_rotation_ambiguous")
        if raw_d4_margin < float(policy["minimum_d4_margin"]):
            reasons.append("raw_d4_rotation_ambiguous")
        if render_margin is not None and render_margin < float(
            policy["minimum_render_margin"]
        ):
            if (
                alternative_pose_degrees is not None
                and alternative_pose_degrees
                <= float(policy["maximum_equivalent_pose_degrees"])
            ):
                warnings.append("equivalent_neighbor_pose_ambiguous")
            elif (
                render_margin
                >= float(policy["minimum_strong_geometry_ambiguity_margin"])
                and
                float(best["best"]["projection_iou"])
                >= float(policy["strong_geometry_ambiguous_pose_iou"])
                and float(best["best"]["ecc_correlation"])
                >= float(policy["strong_geometry_ambiguous_pose_ecc"])
            ):
                warnings.append(
                    "global_render_assignment_ambiguous_but_"
                    "whole_asset_geometry_strong"
                )
            else:
                reasons.append("global_render_assignment_ambiguous")
        if not reasons:
            configuration_status = "CAD_REFERENCE_SILHOUETTE_CONSISTENT"
        elif (
            float(best["best"]["projection_iou"])
            < float(policy["configuration_mismatch_iou_ceiling"])
            and float(best["best"]["ecc_correlation"])
            >= float(policy["configuration_mismatch_minimum_ecc"])
            and float(best["best"]["score"])
            >= float(policy["minimum_alignment_score"])
        ):
            configuration_status = (
                "POSSIBLE_REFERENCE_CAD_CONFIGURATION_MISMATCH"
            )
        else:
            configuration_status = "POSE_UNRESOLVED"
        hypothesis_count = int(policy["pose_candidate_count"])
        pose_hypotheses = [
            {
                "rank": rank,
                "render_view_id": record["render_view_id"],
                "analysis_direction": record.get("analysis_direction"),
                "world_direction": record.get("world_direction"),
                "projection_score": record["best"]["projection_score"],
                "projection_iou": record["best"]["projection_iou"],
                "ecc_correlation": record["best"]["ecc_correlation"],
                "quarter_turns_ccw": record["best"]["quarter_turns_ccw"],
            }
            for rank, record in enumerate(
                render_records[:hypothesis_count], start=1
            )
        ]
        output.append(
            {
                "reference_view_id": reference["view_id"],
                "selected_render_view_id": best["render_view_id"],
                "quarter_turns_ccw": best["best"]["quarter_turns_ccw"],
                "score": best["best"]["score"],
                "projection_score": best["best"]["projection_score"],
                "projection_iou_before": best["best"]["projection_iou_before"],
                "projection_iou": best["best"]["projection_iou"],
                **(
                    {
                        "pose_search_projection_score": pose_search_best[
                            "projection_score"
                        ],
                        "pose_search_projection_iou": pose_search_best[
                            "projection_iou"
                        ],
                        "pose_search_ecc_correlation": pose_search_best[
                            "ecc_correlation"
                        ],
                        "pose_search_mask_size": pose_search_size,
                    }
                    if dense_pose_search
                    else {}
                ),
                "ecc_status": best["best"]["ecc_status"],
                "ecc_correlation": best["best"]["ecc_correlation"],
                "ecc_transform_audit": best["best"]["ecc_transform_audit"],
                "bbox_affine": best["best"]["bbox_affine"],
                "ecc_warp": best["best"]["ecc_warp"],
                "d4_margin": best["d4_margin"],
                "raw_d4_margin": raw_d4_margin,
                "render_margin": render_margin,
                "raw_render_margin": raw_render_margin,
                "alternative_render_view_id": alternative_render_id,
                "alternative_pose_degrees": alternative_pose_degrees,
                "camera_pose": {
                    "analysis_direction": best.get("analysis_direction"),
                    "world_direction": best.get("world_direction"),
                    "camera_position": best.get("camera_position"),
                    "camera_up_axis": best.get("camera_up_axis"),
                    "focal_length_mm": best.get("focal_length_mm"),
                    "image_space_quarter_turns_ccw": best["best"][
                        "quarter_turns_ccw"
                    ],
                },
                "pose_hypotheses": pose_hypotheses,
                "configuration_diagnostic": {
                    "status": configuration_status,
                    "foreground_residual_share": round(
                        1.0 - float(best["best"]["projection_iou"]), 8
                    ),
                    "material_mutation_authorized": False,
                },
                "paired_direction_family": False,
                "declared_direction_candidates": item["declared_candidates"],
                "declared_direction_matches_selected": (
                    best["render_view_id"] in item["declared_candidates"]
                ),
                "global_one_to_one_assignment": True,
                "registration_authority": (
                    "whole_asset_uniform_scale_rotation_translation"
                ),
                **(
                    {
                        "pose_search_method": (
                            "bounded_dense_search_then_"
                            "full_resolution_selected_refinement"
                        )
                    }
                    if dense_pose_search
                    else {}
                ),
                "trusted": not reasons,
                "observation_eligible": not reasons,
                "reason_codes": reasons,
                "warning_codes": warnings,
                "candidates": render_records,
            }
        )
    if alignment_pair_total:
        report_progress(
            progress_callback,
            scope=PROGRESS_SCOPE,
            stage="spatial_view_alignment",
            state="complete",
            current=alignment_pair_total,
            total=alignment_pair_total,
            unit="pairs",
            detail="spatial reference/render alignment completed",
        )
    else:
        report_progress(
            progress_callback,
            scope=PROGRESS_SCOPE,
            stage="spatial_view_alignment",
            state="complete",
            detail="spatial reference/render alignment completed with no pairs",
        )
    return output


def _pixel_label(bgr: Sequence[int]) -> str:
    blue, green, red = (float(value) / 255.0 for value in bgr)
    hue, saturation, value = colorsys.rgb_to_hsv(red, green, blue)
    degrees = hue * 360.0
    if value < 0.16:
        return "black"
    if saturation < 0.14:
        if value < 0.38:
            return "darkgray"
        if value < 0.78:
            return "gray"
        return "white"
    if degrees < 20.0 or degrees >= 345.0:
        return "red"
    if degrees < 55.0:
        return "brown"
    if degrees < 75.0:
        return "yellow"
    if degrees < 170.0:
        return "green"
    if degrees < 260.0:
        return "blue"
    return "purple"


def _base_color_labels(base_color: str) -> set[str]:
    value = base_color.strip().lower()
    aliases = {
        "grey": "gray",
        "silver": "gray",
        "orange": "brown",
        "copper": "brown",
        "bronze": "brown",
        "cyan": "blue",
        "navy": "blue",
        "purple": "purple",
        "magenta": "purple",
        "pink": "red",
    }
    value = aliases.get(value, value)
    if value == "black":
        return {"black", "darkgray"}
    if value == "gray":
        return {"darkgray", "gray"}
    if value == "white":
        return {"white"}
    return {value}


def _palette_groups(document: Mapping[str, Any], view_id: str) -> list[dict[str, Any]]:
    if document.get("schema_version") != "qwen-material-palette/v1":
        raise SpatialMappingError(f"palette {view_id}.schema_version is unsupported")
    if document.get("source_view_id") != view_id:
        raise SpatialMappingError(f"palette {view_id} source_view_id mismatch")
    groups = document.get("groups")
    if not isinstance(groups, list) or not groups:
        raise SpatialMappingError(f"palette {view_id} has no groups")
    seen: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(groups):
        if not isinstance(raw, Mapping):
            raise SpatialMappingError(f"palette {view_id} group {index} is invalid")
        group_id = raw.get("group_id")
        base_color = raw.get("base_color")
        boxes = raw.get("boxes")
        confidence = raw.get("confidence")
        if not isinstance(group_id, str) or not group_id or group_id in seen:
            raise SpatialMappingError(
                f"palette {view_id} has invalid/duplicate group_id"
            )
        if not isinstance(base_color, str) or not base_color:
            raise SpatialMappingError(f"palette {view_id}.{group_id} has no base_color")
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not math.isfinite(float(confidence))
            or not 0.0 <= float(confidence) <= 1.0
        ):
            raise SpatialMappingError(
                f"palette {view_id}.{group_id} has invalid confidence"
            )
        if not isinstance(boxes, list) or not boxes:
            raise SpatialMappingError(f"palette {view_id}.{group_id} has no boxes")
        for box in boxes:
            if (
                not isinstance(box, list)
                or len(box) != 4
                or any(
                    isinstance(value, bool) or not isinstance(value, (int, float))
                    for value in box
                )
            ):
                raise SpatialMappingError(
                    f"palette {view_id}.{group_id} has invalid box"
                )
        seen.add(group_id)
        normalized.append(
            {
                "group_id": group_id,
                "base_color": base_color,
                "boxes": copy.deepcopy(boxes),
                "confidence": float(confidence),
                "visual_description": (
                    str(raw["visual_description"])
                    if isinstance(raw.get("visual_description"), str)
                    else None
                ),
            }
        )
    return normalized


def _accepted_palette_evidence_regions(
    *,
    image_path: Path,
    image_shape: Sequence[int],
    palette_groups: Sequence[Mapping[str, Any]],
    group_id_map: Mapping[str, str],
    palette_audit: Mapping[str, Any] | None,
    view_id: str,
) -> list[dict[str, Any]]:
    """Reconstruct accepted chromatic evidence pixels without guessing.

    The palette verifier records exact boxes, background policy, accepted
    coarse labels, and matching-pixel counts.  Recompute those pixels at the
    original reference resolution and retain a region only when its count
    exactly matches the audit.  Large boxes that the verifier resized, masked
    evidence that is unavailable here, or any stale/tampered audit therefore
    fail closed and cannot support a pose/occlusion limitation.
    """

    if not isinstance(palette_audit, Mapping):
        return []
    if palette_audit.get("mask") is not None:
        return []
    background = palette_audit.get("estimated_background_rgb")
    background_distance = palette_audit.get("background_distance")
    raw_groups = palette_audit.get("groups")
    if (
        not isinstance(background, Sequence)
        or isinstance(background, (str, bytes))
        or len(background) != 3
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value <= 255
            for value in background
        )
        or isinstance(background_distance, bool)
        or not isinstance(background_distance, (int, float))
        or not math.isfinite(float(background_distance))
        or float(background_distance) < 0.0
        or not isinstance(raw_groups, Sequence)
        or isinstance(raw_groups, (str, bytes))
    ):
        return []
    palette_by_id = {str(group["group_id"]): group for group in palette_groups}
    audit_by_id: dict[str, Mapping[str, Any]] = {}
    for raw_group in raw_groups:
        if not isinstance(raw_group, Mapping):
            return []
        group_id = raw_group.get("group_id")
        if not isinstance(group_id, str) or not group_id or group_id in audit_by_id:
            return []
        audit_by_id[group_id] = raw_group

    try:
        with Image.open(image_path) as opened:
            evidence_rgb = np.asarray(
                ImageOps.exif_transpose(opened).convert("RGB"),
                dtype=np.uint8,
            )
    except (OSError, ValueError) as exc:
        raise SpatialMappingError(
            f"unable to decode palette evidence image {view_id}: {image_path}"
        ) from exc
    height, width = (int(image_shape[0]), int(image_shape[1]))
    if evidence_rgb.shape[:2] != (height, width):
        return []
    background_rgb = np.asarray(
        (int(background[0]), int(background[1]), int(background[2])),
        dtype=np.float32,
    )
    results: list[dict[str, Any]] = []
    for group_id, palette_group in sorted(palette_by_id.items()):
        audit_group = audit_by_id.get(group_id)
        canonical_group_id = group_id_map.get(group_id)
        if (
            audit_group is None
            or audit_group.get("accepted") is not True
            or not isinstance(canonical_group_id, str)
            or not canonical_group_id
            or str(audit_group.get("base_color", "")).strip().casefold()
            != str(palette_group["base_color"]).strip().casefold()
        ):
            continue
        base_color = str(palette_group["base_color"]).strip().casefold()
        accepted_labels = _base_color_labels(base_color)
        # Pose-limit resolution is deliberately restricted to chromatic
        # accents; black/white/gray evidence is too easy to confuse with
        # shadows, highlights, and CAD background.
        if accepted_labels <= {"black", "darkgray", "gray", "white"}:
            continue
        raw_boxes = audit_group.get("boxes")
        palette_boxes = palette_group.get("boxes")
        if (
            not isinstance(raw_boxes, Sequence)
            or isinstance(raw_boxes, (str, bytes))
            or not isinstance(palette_boxes, Sequence)
            or isinstance(palette_boxes, (str, bytes))
        ):
            continue
        evidence_mask = np.zeros((height, width), dtype=np.uint8)
        accepted_box_records: list[dict[str, Any]] = []
        audited_matching_total = 0
        valid = True
        for raw_box in raw_boxes:
            if not isinstance(raw_box, Mapping) or raw_box.get("accepted") is not True:
                continue
            box_index = raw_box.get("box_index")
            box = raw_box.get("box")
            matching_count = raw_box.get(
                "matching_pixel_count", raw_box.get("matching_pixels")
            )
            foreground_method = raw_box.get("foreground_method")
            accepted_color_labels = raw_box.get("accepted_color_labels")
            effective_distance = raw_box.get("effective_background_distance")
            normalized_audit_labels = (
                {
                    normalized
                    for value in accepted_color_labels
                    for normalized in _base_color_labels(str(value))
                }
                if isinstance(accepted_color_labels, Sequence)
                and not isinstance(accepted_color_labels, (str, bytes))
                else set()
            )
            audit_label_set = (
                {str(value) for value in accepted_color_labels}
                if isinstance(accepted_color_labels, Sequence)
                and not isinstance(accepted_color_labels, (str, bytes))
                else set()
            )
            if (
                isinstance(box_index, bool)
                or not isinstance(box_index, int)
                or not 0 <= box_index < len(palette_boxes)
                or box != palette_boxes[box_index]
                or not isinstance(box, Sequence)
                or isinstance(box, (str, bytes))
                or len(box) != 4
                or isinstance(matching_count, bool)
                or not isinstance(matching_count, int)
                or matching_count < 1
                or foreground_method != "color_distance"
                or not isinstance(accepted_color_labels, Sequence)
                or isinstance(accepted_color_labels, (str, bytes))
                or normalized_audit_labels != accepted_labels
                or isinstance(effective_distance, bool)
                or not isinstance(effective_distance, (int, float))
                or float(effective_distance) != float(background_distance)
            ):
                valid = False
                break
            left = max(0, int(math.floor(float(box[0]) * width / 1000.0)))
            top = max(0, int(math.floor(float(box[1]) * height / 1000.0)))
            right = min(width, int(math.ceil(float(box[2]) * width / 1000.0)))
            bottom = min(height, int(math.ceil(float(box[3]) * height / 1000.0)))
            if right <= left or bottom <= top or max(right - left, bottom - top) > 160:
                valid = False
                break
            crop = evidence_rgb[top:bottom, left:right]
            distances = np.linalg.norm(
                crop.astype(np.float32) - background_rgb,
                axis=2,
            )
            foreground = distances >= float(background_distance)
            label_mask = np.asarray(
                [
                    pixel_color_label(
                        int(pixel[0]),
                        int(pixel[1]),
                        int(pixel[2]),
                    )
                    in audit_label_set
                    for pixel in crop.reshape(-1, 3)
                ],
                dtype=bool,
            ).reshape(crop.shape[:2])
            matching = foreground & label_mask
            reproduced_count = int(np.count_nonzero(matching))
            if reproduced_count != matching_count:
                valid = False
                break
            existing = evidence_mask[top:bottom, left:right] > 0
            if np.any(existing & matching):
                valid = False
                break
            evidence_mask[top:bottom, left:right][matching] = 255
            audited_matching_total += matching_count
            accepted_box_records.append(
                {
                    "box_index": box_index,
                    "box": [int(value) for value in box],
                    "matching_pixel_count": matching_count,
                }
            )
        evidence_pixel_count = int(np.count_nonzero(evidence_mask))
        if (
            not valid
            or not accepted_box_records
            or evidence_pixel_count != audited_matching_total
        ):
            continue
        group_audit_payload = {
            "view_id": view_id,
            "local_group_id": group_id,
            "canonical_group_id": canonical_group_id,
            "base_color": base_color,
            "accepted_boxes": accepted_box_records,
            "evidence_pixel_count": evidence_pixel_count,
        }
        results.append(
            {
                **group_audit_payload,
                "evidence_audit_sha256": _sha256_document(group_audit_payload),
                "_mask": evidence_mask,
            }
        )
    return results


def _unique_multiview_canonical_palette_groups(
    references: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Return canonical groups backed by a globally unique multiview family.

    The source-view and family checks live in one helper so canonical palette
    propagation and the dark-foreground diagnostic use exactly the same
    uniqueness proof.
    """

    observations: dict[
        str, list[tuple[str, tuple[str, ...], str, bool]]
    ] = {}
    for reference in references:
        view_id = reference.get("view_id")
        groups = reference.get("palette_groups")
        group_map = reference.get("group_id_map")
        if (
            not isinstance(view_id, str)
            or not isinstance(groups, Sequence)
            or isinstance(groups, (str, bytes))
            or not isinstance(group_map, Mapping)
        ):
            continue
        for group in groups:
            if not isinstance(group, Mapping):
                continue
            local_group_id = group.get("group_id")
            base_color = group.get("base_color")
            canonical_group_id = group_map.get(local_group_id)
            if (
                not isinstance(local_group_id, str)
                or not isinstance(base_color, str)
                or not isinstance(canonical_group_id, str)
                or not canonical_group_id
            ):
                continue
            family = tuple(sorted(_base_color_labels(base_color)))
            unresolved_light_neutral = (
                family == ("white",)
                and str(group.get("visual_description") or "").strip().casefold()
                == (
                    "connected light neutral surface region detected from "
                    "pixels; physical material unresolved"
                )
            )
            observations.setdefault(canonical_group_id, []).append(
                (
                    view_id,
                    family,
                    base_color.strip().lower(),
                    unresolved_light_neutral,
                )
            )

    eligible: dict[str, dict[str, Any]] = {}
    for canonical_group_id, records in observations.items():
        source_view_ids = sorted({record[0] for record in records})
        families = {record[1] for record in records}
        if (
            len(source_view_ids) < MIN_CANONICAL_PALETTE_SOURCE_VIEWS
            or len(families) != 1
        ):
            continue
        family = next(iter(families))
        eligible[canonical_group_id] = {
            "canonical_group_id": canonical_group_id,
            "base_color": sorted({record[2] for record in records})[0],
            "accepted_labels": list(family),
            "source_view_ids": source_view_ids,
        }
    family_claims: dict[tuple[str, ...], set[str]] = {}
    for canonical_group_id, records in observations.items():
        for family in {record[1] for record in records}:
            family_claims.setdefault(family, set()).add(canonical_group_id)
    proof_bound_light_neutral_group_ids = {
        canonical_group_id
        for canonical_group_id, records in observations.items()
        if len({record[0] for record in records})
        >= MIN_CANONICAL_PALETTE_SOURCE_VIEWS
        and records
        and all(record[3] for record in records)
    }

    def uniquely_supported(
        canonical_group_id: str,
        record: Mapping[str, Any],
    ) -> bool:
        family = tuple(record["accepted_labels"])
        claims = family_claims[family]
        if (
            family == ("black", "darkgray")
            and len(record["source_view_ids"]) >= 3
        ):
            # Near-black evidence is uniquely protected by the separate
            # foreground/object diagnostic below.  A single resolved black
            # observation must not veto a distinct three-view, mask-bound
            # dark group; another independently supported black group does.
            claims = {group_id for group_id in claims if group_id in eligible}
        elif (
            family == ("white",)
            and canonical_group_id in proof_bound_light_neutral_group_ids
        ):
            # A resolved white singleton may describe a separate control
            # module while the augmentation lane independently recovers the
            # machine's light-neutral surfaces in several other photographs.
            # Only the exact unresolved-pixel description is allowed to use
            # this exception; ordinary white claims remain mutually
            # ambiguous and fail closed.
            claims = {group_id for group_id in claims if group_id in eligible}
        return claims == {canonical_group_id}

    return {
        canonical_group_id: record
        for canonical_group_id, record in eligible.items()
        if uniquely_supported(canonical_group_id, record)
    }


def _canonical_palette_supplements(
    references: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Propagate only globally unique colour families across reference views.

    A local Qwen palette can omit a colour that is plainly present in its raw
    pixels. We may still classify those pixels when the same canonical group
    has the same colour family in at least two source views and no different
    canonical group claims that family. Ambiguous achromatic families
    therefore remain local and fail closed.
    """

    unique_eligible = _unique_multiview_canonical_palette_groups(references)
    supplements: dict[str, list[dict[str, Any]]] = {}
    for reference in references:
        view_id = reference.get("view_id")
        group_map = reference.get("group_id_map")
        if not isinstance(view_id, str) or not isinstance(group_map, Mapping):
            continue
        local_canonical_ids = {
            value for value in group_map.values() if isinstance(value, str) and value
        }
        supplements[view_id] = [
            copy.deepcopy(record)
            for canonical_group_id, record in sorted(unique_eligible.items())
            if canonical_group_id not in local_canonical_ids
        ]
    return supplements


def _dark_foreground_diagnostic(
    *,
    image: np.ndarray,
    projected_mask: np.ndarray,
    alignment: Mapping[str, Any],
    projected_part_pixels: int,
    canonical_palette_groups: Sequence[Mapping[str, Any]],
    alternative_canonical_group_ids: Sequence[str] = (),
) -> dict[str, Any]:
    """Prove that a near-black projection contains an object, not background.

    This diagnostic is intentionally independent from ordinary palette
    classification.  It normalizes every reference to a 512-pixel long edge,
    derives background and edge thresholds from that image's border, requires
    an interior dark core, and compares the projected location with eight
    bbox-sized negative-control shifts.  It can authorize only a later,
    bounded repair lane; the ordinary canonical supplement is unchanged.
    """

    selected = DARK_FOREGROUND_POLICY
    height, width = image.shape[:2]
    long_edge = int(selected["normalized_long_edge_pixels"])
    scale = long_edge / max(height, width)
    normalized_width = max(1, int(round(width * scale)))
    normalized_height = max(1, int(round(height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    normalized_image = cv2.resize(
        image,
        (normalized_width, normalized_height),
        interpolation=interpolation,
    )
    normalized_projected = (
        cv2.resize(
            projected_mask,
            (normalized_width, normalized_height),
            interpolation=cv2.INTER_NEAREST,
        )
        > 0
    )
    normalized_projected_pixels = int(np.count_nonzero(normalized_projected))

    border = np.concatenate(
        (
            normalized_image[0, :, :],
            normalized_image[-1, :, :],
            normalized_image[:, 0, :],
            normalized_image[:, -1, :],
        ),
        axis=0,
    ).astype(np.float32)
    background = np.median(border, axis=0)
    border_distances = np.linalg.norm(border - background[None, :], axis=1)
    border_distance_p95 = float(np.percentile(border_distances, 95))
    background_threshold = min(45.0, max(12.0, border_distance_p95 + 6.0))
    pixel_distances = np.linalg.norm(
        normalized_image.astype(np.float32) - background[None, None, :],
        axis=2,
    )
    non_background = pixel_distances >= background_threshold

    channels = normalized_image.astype(np.int16)
    channel_max = np.max(channels, axis=2)
    channel_min = np.min(channels, axis=2)
    near_black = (channel_max < int(selected["near_black_max_channel_exclusive"])) & (
        channel_max - channel_min <= int(selected["near_black_max_channel_spread"])
    )
    dark_signal = normalized_projected & near_black & non_background
    near_black_pixels = int(np.count_nonzero(normalized_projected & near_black))
    non_background_pixels = int(np.count_nonzero(normalized_projected & non_background))
    dark_signal_pixels = int(np.count_nonzero(dark_signal))
    near_black_share = (
        near_black_pixels / normalized_projected_pixels
        if normalized_projected_pixels
        else 0.0
    )
    non_background_share = (
        non_background_pixels / normalized_projected_pixels
        if normalized_projected_pixels
        else 0.0
    )
    dark_signal_share = (
        dark_signal_pixels / normalized_projected_pixels
        if normalized_projected_pixels
        else 0.0
    )
    dark_signal_purity = (
        dark_signal_pixels / non_background_pixels if non_background_pixels else 0.0
    )

    distance_to_boundary = cv2.distanceTransform(
        normalized_projected.astype(np.uint8),
        cv2.DIST_L2,
        5,
    )
    core = distance_to_boundary >= float(selected["core_distance_pixels"])
    core_pixels = int(np.count_nonzero(core))
    core_dark_signal = core & near_black & non_background
    core_dark_signal_pixels = int(np.count_nonzero(core_dark_signal))
    core_dark_signal_share = (
        core_dark_signal_pixels / core_pixels if core_pixels else 0.0
    )

    gray = cv2.cvtColor(normalized_image, cv2.COLOR_BGR2GRAY)
    gradient_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gradient_x, gradient_y)
    border_gradient = np.concatenate(
        (
            gradient[0, :],
            gradient[-1, :],
            gradient[:, 0],
            gradient[:, -1],
        )
    )
    border_gradient_p99 = float(np.percentile(border_gradient, 99))
    adaptive_edge_threshold = max(12.0, border_gradient_p99 + 6.0)
    adaptive_edges = gradient >= adaptive_edge_threshold
    adaptive_edge_pixels = int(np.count_nonzero(normalized_projected & adaptive_edges))
    adaptive_edge_density = (
        adaptive_edge_pixels / normalized_projected_pixels
        if normalized_projected_pixels
        else 0.0
    )
    # Canny is retained as a second, fully audited edge measurement.  The
    # calibrated decision threshold uses the continuous Sobel response above;
    # raw one-pixel Canny density is resolution- and shape-perimeter-dependent.
    canny_high = int(min(255, max(12, round(adaptive_edge_threshold))))
    canny_low = max(1, int(round(canny_high * 0.5)))
    canny_edges = cv2.Canny(gray, canny_low, canny_high) > 0
    canny_edge_pixels = int(np.count_nonzero(normalized_projected & canny_edges))
    canny_edge_density = (
        canny_edge_pixels / normalized_projected_pixels
        if normalized_projected_pixels
        else 0.0
    )

    null_shifts: list[dict[str, Any]] = []
    null_scores: list[float] = []
    if normalized_projected_pixels:
        x, y, bbox_width, bbox_height = cv2.boundingRect(
            normalized_projected.astype(np.uint8)
        )
        del x, y
        offset_x = max(
            int(selected["minimum_null_offset_pixels"]),
            int(bbox_width),
        )
        offset_y = max(
            int(selected["minimum_null_offset_pixels"]),
            int(bbox_height),
        )
        offsets = (
            (-offset_x, 0),
            (offset_x, 0),
            (0, -offset_y),
            (0, offset_y),
            (-offset_x, -offset_y),
            (-offset_x, offset_y),
            (offset_x, -offset_y),
            (offset_x, offset_y),
        )
        source_mask = normalized_projected.astype(np.uint8) * 255
        for shift_x, shift_y in offsets:
            shifted = (
                cv2.warpAffine(
                    source_mask,
                    np.asarray(
                        [[1.0, 0.0, shift_x], [0.0, 1.0, shift_y]],
                        dtype=np.float32,
                    ),
                    (normalized_width, normalized_height),
                    flags=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )
                > 0
            )
            retained_pixels = int(np.count_nonzero(shifted))
            valid_area_ratio = retained_pixels / normalized_projected_pixels
            valid = valid_area_ratio >= float(selected["minimum_null_valid_area_ratio"])
            shifted_dark_pixels = int(
                np.count_nonzero(shifted & near_black & non_background)
            )
            shifted_dark_share = (
                shifted_dark_pixels / retained_pixels if retained_pixels else 0.0
            )
            if valid:
                null_scores.append(shifted_dark_share)
            null_shifts.append(
                {
                    "offset_pixels": [int(shift_x), int(shift_y)],
                    "retained_pixels": retained_pixels,
                    "valid_area_ratio": round(valid_area_ratio, 8),
                    "valid": valid,
                    "dark_signal_pixels": shifted_dark_pixels,
                    "dark_signal_share": round(shifted_dark_share, 8),
                    "mask_sha256": _normalized_pixel_sha256(shifted.astype(np.uint8)),
                }
            )
    valid_null_shift_count = len(null_scores)
    null_q75 = (
        float(np.percentile(np.asarray(null_scores, dtype=np.float64), 75))
        if null_scores
        else None
    )
    null_margin = dark_signal_share - null_q75 if null_q75 is not None else None

    black_groups: dict[str, dict[str, Any]] = {}
    for raw_group in canonical_palette_groups:
        group = raw_group if isinstance(raw_group, Mapping) else {}
        canonical_group_id = group.get("canonical_group_id")
        base_color = group.get("base_color")
        accepted_labels = group.get("accepted_labels")
        source_view_ids = group.get("source_view_ids")
        if (
            isinstance(canonical_group_id, str)
            and canonical_group_id
            and isinstance(base_color, str)
            and base_color.strip().casefold() == "black"
            and isinstance(accepted_labels, Sequence)
            and not isinstance(accepted_labels, (str, bytes))
            and all(isinstance(label, str) for label in accepted_labels)
            and set(accepted_labels) == {"black", "darkgray"}
            and isinstance(source_view_ids, Sequence)
            and not isinstance(source_view_ids, (str, bytes))
            and all(isinstance(view_id, str) and view_id for view_id in source_view_ids)
        ):
            black_groups[canonical_group_id] = {
                "canonical_group_id": canonical_group_id,
                "source_view_ids": sorted(set(source_view_ids)),
            }
    # A singleton black observation cannot authorize the dark-on-background
    # recovery lane.  It must also not suppress a separate black group that
    # has independent multiview support.  This occurs naturally when one
    # camera describes a resolved black material while three other cameras
    # recover a mask-bound dark region whose physical class is unresolved.
    # Only independently supported black groups participate in authority and
    # ambiguity; singleton groups remain audited, non-authoritative context.
    eligible_black_groups = {
        group_id: record
        for group_id, record in black_groups.items()
        if len(record["source_view_ids"]) >= MIN_CANONICAL_PALETTE_SOURCE_VIEWS
    }
    canonical_group_id = (
        next(iter(sorted(eligible_black_groups)))
        if len(eligible_black_groups) == 1
        else None
    )
    canonical_source_view_ids = (
        black_groups[canonical_group_id]["source_view_ids"]
        if canonical_group_id is not None
        else []
    )
    alternatives = sorted(
        {
            value
            for value in alternative_canonical_group_ids
            if isinstance(value, str) and value and value != canonical_group_id
        }
        | {
            group_id
            for group_id in eligible_black_groups
            if group_id != canonical_group_id
        }
    )

    def alignment_metric(name: str) -> float | None:
        value = alignment.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            return None
        return float(value)

    alignment_score = alignment_metric("score")
    projection_score = alignment_metric("projection_score")
    projection_iou = alignment_metric("projection_iou")
    ecc_correlation = alignment_metric("ecc_correlation")
    alignment_reason_codes = alignment.get("reason_codes")
    transform_audit = alignment.get("ecc_transform_audit")
    trusted_alignment = (
        alignment.get("trusted") is True
        and alignment_reason_codes == []
        and alignment.get("ecc_status") == "success"
        and isinstance(transform_audit, Mapping)
        and transform_audit.get("constraints_passed") is True
        and alignment_score is not None
        and alignment_score >= float(selected["minimum_alignment_score"])
        and projection_score is not None
        and projection_score >= float(selected["minimum_projection_score"])
        and projection_iou is not None
        and projection_iou >= float(selected["minimum_projection_iou"])
        and ecc_correlation is not None
        and ecc_correlation >= float(selected["minimum_ecc_correlation"])
    )

    reasons: list[str] = []
    if len(eligible_black_groups) != 1:
        reasons.append("DARK_CANONICAL_BLACK_GROUP_NOT_UNIQUE")
    if (
        canonical_group_id is None
        and len(black_groups) == 1
        and len(next(iter(black_groups.values()))["source_view_ids"])
        < MIN_CANONICAL_PALETTE_SOURCE_VIEWS
    ):
        reasons.append("DARK_CANONICAL_SOURCE_VIEWS_BELOW_FLOOR")
    if alternatives:
        reasons.append("DARK_CANONICAL_GROUP_CONFLICT")
    if not trusted_alignment:
        reasons.append("DARK_ALIGNMENT_NOT_STRONG")
    if normalized_projected_pixels < int(
        selected["minimum_normalized_projected_pixels"]
    ):
        reasons.append("DARK_NORMALIZED_PROJECTED_PIXELS_BELOW_FLOOR")
    if near_black_share < float(selected["minimum_near_black_share"]):
        reasons.append("DARK_NEAR_BLACK_SHARE_BELOW_FLOOR")
    if non_background_pixels < int(selected["minimum_non_background_pixels"]):
        reasons.append("DARK_NON_BACKGROUND_PIXELS_BELOW_FLOOR")
    if dark_signal_share < float(selected["minimum_dark_signal_share"]):
        reasons.append("DARK_SIGNAL_SHARE_BELOW_FLOOR")
    if dark_signal_purity < float(selected["minimum_dark_signal_purity"]):
        reasons.append("DARK_SIGNAL_PURITY_BELOW_FLOOR")
    if core_pixels < int(selected["minimum_core_pixels"]):
        reasons.append("DARK_CORE_PIXELS_BELOW_FLOOR")
    if core_dark_signal_share < float(selected["minimum_core_dark_signal_share"]):
        reasons.append("DARK_CORE_SIGNAL_SHARE_BELOW_FLOOR")
    if adaptive_edge_density < float(selected["minimum_adaptive_edge_density"]):
        reasons.append("DARK_EDGE_DENSITY_BELOW_FLOOR")
    if valid_null_shift_count < int(selected["minimum_valid_null_shifts"]):
        reasons.append("DARK_VALID_NULL_SHIFTS_BELOW_FLOOR")
    if null_margin is None or null_margin < float(selected["minimum_null_q75_margin"]):
        reasons.append("DARK_NULL_Q75_MARGIN_BELOW_FLOOR")

    diagnostic: dict[str, Any] = {
        "status": "resolved" if not reasons else "rejected",
        "reason_codes": reasons,
        "evidence_scope": "dark_on_black_foreground_repair_only",
        "canonical_group_id": canonical_group_id,
        "canonical_source_view_ids": canonical_source_view_ids,
        "black_group_count": len(black_groups),
        "eligible_multiview_black_group_count": len(eligible_black_groups),
        "non_authoritative_singleton_black_group_ids": sorted(
            set(black_groups) - set(eligible_black_groups)
        ),
        "alternative_canonical_group_ids": alternatives,
        "projected_part_pixels": int(projected_part_pixels),
        "normalized_projected_pixels": normalized_projected_pixels,
        "normalization": {
            "long_edge_pixels": long_edge,
            "original_size": [int(width), int(height)],
            "normalized_size": [normalized_width, normalized_height],
            "scale": round(scale, 10),
        },
        "alignment": {
            "trusted": alignment.get("trusted") is True,
            "reason_codes_empty": alignment_reason_codes == [],
            "score": (
                round(alignment_score, 8) if alignment_score is not None else None
            ),
            "projection_score": (
                round(projection_score, 8) if projection_score is not None else None
            ),
            "projection_iou": (
                round(projection_iou, 8) if projection_iou is not None else None
            ),
            "ecc_status": alignment.get("ecc_status"),
            "ecc_correlation": (
                round(ecc_correlation, 8) if ecc_correlation is not None else None
            ),
            "transform_constraints_passed": (
                isinstance(transform_audit, Mapping)
                and transform_audit.get("constraints_passed") is True
            ),
            "strong": trusted_alignment,
        },
        "background": {
            "median_bgr": [round(float(value), 8) for value in background],
            "border_distance_p95": round(border_distance_p95, 8),
            "distance_threshold": round(background_threshold, 8),
        },
        "thresholds": copy.deepcopy(selected),
        "near_black_pixels": near_black_pixels,
        "near_black_share": round(near_black_share, 8),
        "non_background_pixels": non_background_pixels,
        "non_background_share": round(non_background_share, 8),
        "dark_signal_pixels": dark_signal_pixels,
        "dark_signal_share": round(dark_signal_share, 8),
        "dark_signal_purity": round(dark_signal_purity, 8),
        "core_pixels": core_pixels,
        "core_dark_signal_pixels": core_dark_signal_pixels,
        "core_dark_signal_share": round(core_dark_signal_share, 8),
        "core_distance_pixels": float(selected["core_distance_pixels"]),
        "adaptive_edge_pixels": adaptive_edge_pixels,
        "adaptive_edge_density": round(adaptive_edge_density, 8),
        "adaptive_edge_threshold": round(adaptive_edge_threshold, 8),
        "border_gradient_p99": round(border_gradient_p99, 8),
        "canny_low_threshold": canny_low,
        "canny_high_threshold": canny_high,
        "canny_edge_pixels": canny_edge_pixels,
        "canny_edge_density": round(canny_edge_density, 8),
        "null_shifts": null_shifts,
        "valid_null_shift_count": valid_null_shift_count,
        "null_dark_signal_share_q75": (
            round(null_q75, 8) if null_q75 is not None else None
        ),
        "dark_signal_null_margin": (
            round(null_margin, 8) if null_margin is not None else None
        ),
        "normalized_reference_pixel_sha256": _normalized_pixel_sha256(normalized_image),
        "normalized_projected_mask_sha256": _normalized_pixel_sha256(
            normalized_projected.astype(np.uint8)
        ),
        "normalized_near_black_mask_sha256": _normalized_pixel_sha256(
            near_black.astype(np.uint8)
        ),
        "normalized_non_background_mask_sha256": _normalized_pixel_sha256(
            non_background.astype(np.uint8)
        ),
        "normalized_dark_signal_mask_sha256": _normalized_pixel_sha256(
            dark_signal.astype(np.uint8)
        ),
        "normalized_adaptive_edge_mask_sha256": _normalized_pixel_sha256(
            adaptive_edges.astype(np.uint8)
        ),
    }
    diagnostic["diagnostic_sha256"] = _sha256_document(diagnostic)
    return diagnostic


def _project_part_observation(
    *,
    reference: Mapping[str, Any],
    render: Mapping[str, Any],
    alignment: Mapping[str, Any],
    part_id: str,
    palette_groups: Sequence[Mapping[str, Any]],
    group_id_map: Mapping[str, str],
    policy: Mapping[str, float | int],
    canonical_palette_groups: Sequence[Mapping[str, Any]] = (),
    unique_canonical_palette_groups: Sequence[Mapping[str, Any]] = (),
    accepted_palette_evidence: Sequence[Mapping[str, Any]] = (),
    isolated_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    declared = render["visible_pixels"].get(part_id, 0)
    isolated_view_pixels = (
        isolated_evidence.get("source_visible_pixels_by_view", {})
        if isinstance(isolated_evidence, Mapping)
        else {}
    )
    isolated_eligible = (
        isinstance(isolated_evidence, Mapping)
        and isolated_evidence.get("schema_version") == ISOLATED_EVIDENCE_SCHEMA_VERSION
        and isolated_evidence.get("material_neutralized") is True
        and isolated_evidence.get("background_removed") is True
        and int(isolated_evidence.get("source_evidence_view_count", 0))
        >= int(policy["minimum_isolated_source_view_count"])
        and isinstance(isolated_view_pixels, Mapping)
        and isolated_view_pixels.get(render["view_id"]) == declared
        and declared >= int(policy["minimum_isolated_source_visible_pixels"])
    )
    base = {
        "reference_view_id": reference["view_id"],
        "render_view_id": render["view_id"],
        "declared_visible_pixels": declared,
        "evidence_mode": (
            "isolated_mask_multiview_diagnostic"
            if isolated_eligible
            else "source_projection"
        ),
        "isolated_evidence_sha256": (
            isolated_evidence.get("sha256") if isolated_eligible else None
        ),
        "isolated_source_view_count": (
            isolated_evidence.get("source_evidence_view_count")
            if isolated_eligible
            else None
        ),
    }
    diagnostic_floor = min(
        int(policy["minimum_visible_pixels"]),
        (
            int(policy["minimum_isolated_source_visible_pixels"])
            if isolated_eligible
            else int(policy["minimum_diagnostic_visible_pixels"])
        ),
    )
    if declared < diagnostic_floor:
        return {
            **base,
            "classification": "insufficient_visibility",
            "reason_code": "part_visible_pixels_below_diagnostic_floor",
        }

    quarter_turns = int(alignment["quarter_turns_ccw"])
    ids = np.rot90(render["part_ids_image"], quarter_turns).copy()
    red, green, blue = _part_color(part_id)
    color_bgr = np.asarray((blue, green, red), dtype=np.uint8)
    part_mask = np.all(ids == color_bgr, axis=2).astype(np.uint8) * 255
    decoded_pixels = int(np.count_nonzero(part_mask))
    if decoded_pixels < 8:
        return {
            **base,
            "classification": "insufficient_visibility",
            "reason_code": "decoded_part_mask_too_small",
        }

    image = reference["image"]
    try:
        bbox_affine = np.asarray(alignment["bbox_affine"], dtype=np.float32)
        ecc_warp = np.asarray(alignment["ecc_warp"], dtype=np.float32)
    except (KeyError, TypeError, ValueError) as exc:
        raise SpatialMappingError(
            "trusted alignment lacks projection matrices"
        ) from exc
    if bbox_affine.shape != (2, 3) or ecc_warp.shape != (2, 3):
        raise SpatialMappingError("trusted alignment projection matrices are malformed")
    output_size = (int(image.shape[1]), int(image.shape[0]))
    normalized_mask = cv2.warpAffine(
        part_mask,
        bbox_affine,
        output_size,
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    projected_mask = cv2.warpAffine(
        normalized_mask,
        ecc_warp,
        output_size,
        flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    projected_y, projected_x = np.where(projected_mask > 0)
    if len(projected_x) < diagnostic_floor:
        return {
            **base,
            "decoded_part_pixels": decoded_pixels,
            "projected_part_pixels": int(len(projected_x)),
            "classification": "insufficient_visibility",
            "reason_code": "projected_part_pixels_below_diagnostic_floor",
        }

    accepted_evidence_box_overlaps: list[dict[str, Any]] = []
    for raw_region in accepted_palette_evidence:
        region = raw_region if isinstance(raw_region, Mapping) else {}
        evidence_mask = region.get("_mask")
        if (
            not isinstance(evidence_mask, np.ndarray)
            or evidence_mask.shape != projected_mask.shape
        ):
            raise SpatialMappingError("accepted palette evidence mask is malformed")
        overlap_pixels = int(
            np.count_nonzero((projected_mask > 0) & (evidence_mask > 0))
        )
        evidence_pixels = int(region["evidence_pixel_count"])
        accepted_evidence_box_overlaps.append(
            {
                "local_group_id": region["local_group_id"],
                "canonical_group_id": region["canonical_group_id"],
                "base_color": region["base_color"],
                "evidence_pixel_count": evidence_pixels,
                "projected_overlap_pixels": overlap_pixels,
                "projected_overlap_share": round(
                    overlap_pixels / evidence_pixels if evidence_pixels else 0.0,
                    8,
                ),
                "evidence_audit_sha256": region["evidence_audit_sha256"],
            }
        )

    def score_labels(
        label_counts: Counter[str],
        *,
        include_canonical_supplements: bool = False,
    ) -> tuple[list[dict[str, Any]], int, float]:
        sample_count = sum(label_counts.values())
        rows: list[dict[str, Any]] = []
        for group in palette_groups:
            accepted_labels = _base_color_labels(str(group["base_color"]))
            matching = sum(label_counts[label] for label in accepted_labels)
            rows.append(
                {
                    "local_group_id": group["group_id"],
                    "canonical_group_id": group_id_map.get(str(group["group_id"])),
                    "base_color": group["base_color"],
                    "matching_pixels": matching,
                    "color_share": round(
                        matching / sample_count if sample_count else 0.0, 8
                    ),
                    "evidence_scope": "view_local_palette",
                }
            )
        for group in canonical_palette_groups if include_canonical_supplements else ():
            accepted_labels = group.get("accepted_labels")
            if (
                not isinstance(accepted_labels, Sequence)
                or isinstance(accepted_labels, (str, bytes))
                or not accepted_labels
                or any(not isinstance(label, str) for label in accepted_labels)
            ):
                raise SpatialMappingError(
                    "canonical palette supplement has invalid accepted_labels"
                )
            canonical_group_id = group.get("canonical_group_id")
            if not isinstance(canonical_group_id, str) or not canonical_group_id:
                raise SpatialMappingError(
                    "canonical palette supplement has invalid canonical_group_id"
                )
            matching = sum(label_counts[label] for label in accepted_labels)
            rows.append(
                {
                    "local_group_id": f"__canonical__:{canonical_group_id}",
                    "canonical_group_id": canonical_group_id,
                    "base_color": group.get("base_color"),
                    "matching_pixels": matching,
                    "color_share": round(
                        matching / sample_count if sample_count else 0.0, 8
                    ),
                    "evidence_scope": "canonical_multiview_propagation",
                    "canonical_source_view_ids": list(group.get("source_view_ids", [])),
                }
            )
        rows.sort(
            key=lambda item: (-float(item["color_share"]), item["local_group_id"])
        )
        runner = float(rows[1]["color_share"]) if len(rows) > 1 else 0.0
        return rows, sample_count, float(rows[0]["color_share"]) - runner

    labels = Counter(_pixel_label(pixel) for pixel in image[projected_y, projected_x])
    scored, sampled, margin = score_labels(labels)
    bbox_y, bbox_x = np.where(normalized_mask > 0)
    bbox_labels = Counter(_pixel_label(pixel) for pixel in image[bbox_y, bbox_x])
    bbox_scored, bbox_sampled, bbox_margin = score_labels(bbox_labels)
    best = scored[0]
    bbox_best = bbox_scored[0]

    def resolved_group(
        row: Mapping[str, Any],
        row_margin: float,
        *,
        minimum_share: float,
        minimum_margin: float,
    ) -> str | None:
        if float(row["color_share"]) < minimum_share:
            return None
        if row_margin < minimum_margin:
            return None
        canonical = row.get("canonical_group_id")
        return canonical if isinstance(canonical, str) and canonical else None

    refined_canonical = resolved_group(
        best,
        margin,
        minimum_share=float(policy["minimum_color_share"]),
        minimum_margin=float(policy["minimum_color_margin"]),
    )
    bbox_canonical = resolved_group(
        bbox_best,
        bbox_margin,
        minimum_share=float(policy["minimum_color_share"]),
        minimum_margin=float(policy["minimum_color_margin"]),
    )
    diagnostic_refined_canonical = resolved_group(
        best,
        margin,
        minimum_share=float(policy["minimum_diagnostic_color_share"]),
        minimum_margin=float(policy["minimum_diagnostic_color_margin"]),
    )
    diagnostic_bbox_canonical = resolved_group(
        bbox_best,
        bbox_margin,
        minimum_share=float(policy["minimum_diagnostic_color_share"]),
        minimum_margin=float(policy["minimum_diagnostic_color_margin"]),
    )

    def canonical_foreground_sample(
        sample_x: np.ndarray,
        sample_y: np.ndarray,
    ) -> dict[str, Any]:
        foreground = reference.get("foreground")
        if (
            not isinstance(foreground, np.ndarray)
            or foreground.shape != image.shape[:2]
        ):
            raise SpatialMappingError(
                "reference foreground is unavailable for canonical supplement"
            )
        on_foreground = foreground[sample_y, sample_x] > 0
        foreground_x = sample_x[on_foreground]
        foreground_y = sample_y[on_foreground]
        foreground_labels = Counter(
            _pixel_label(pixel) for pixel in image[foreground_y, foreground_x]
        )
        foreground_scores, foreground_sampled, foreground_margin = score_labels(
            foreground_labels,
            include_canonical_supplements=True,
        )
        winner = foreground_scores[0]
        canonical_group_id = resolved_group(
            winner,
            foreground_margin,
            minimum_share=float(policy["minimum_diagnostic_color_share"]),
            minimum_margin=float(policy["minimum_diagnostic_color_margin"]),
        )
        if winner.get("evidence_scope") != "canonical_multiview_propagation":
            canonical_group_id = None
        return {
            "sampled_projection_pixels": int(len(sample_x)),
            "sampled_foreground_pixels": foreground_sampled,
            "foreground_overlap_ratio": round(
                foreground_sampled / len(sample_x) if len(sample_x) else 0.0,
                8,
            ),
            "canonical_group_id": canonical_group_id,
            "local_group_id": winner["local_group_id"],
            "best_color_share": winner["color_share"],
            "color_margin": round(foreground_margin, 8),
            "group_scores": foreground_scores,
            "canonical_source_view_ids": list(
                winner.get("canonical_source_view_ids", [])
            ),
        }

    canonical_palette_diagnostic: dict[str, Any] | None = None
    if canonical_palette_groups:
        canonical_direct = canonical_foreground_sample(projected_x, projected_y)
        canonical_bbox = canonical_foreground_sample(bbox_x, bbox_y)
        canonical_perturbations: list[dict[str, Any]] = []
        for offset_x, offset_y in ((-2, 0), (2, 0), (0, -2), (0, 2)):
            shifted_x = projected_x + offset_x
            shifted_y = projected_y + offset_y
            inside = (
                (shifted_x >= 0)
                & (shifted_x < image.shape[1])
                & (shifted_y >= 0)
                & (shifted_y < image.shape[0])
            )
            sample = canonical_foreground_sample(
                shifted_x[inside],
                shifted_y[inside],
            )
            canonical_perturbations.append(
                {
                    "offset_pixels": [offset_x, offset_y],
                    **sample,
                }
            )

        canonical_target = canonical_direct["canonical_group_id"]
        canonical_reasons: list[str] = []
        canonical_floor = int(policy["minimum_diagnostic_visible_pixels"])
        minimum_overlap = float(
            policy["minimum_canonical_supplement_foreground_overlap"]
        )
        samples = [
            canonical_direct,
            canonical_bbox,
            *canonical_perturbations,
        ]
        if canonical_target is None:
            canonical_reasons.append("CANONICAL_SUPPLEMENT_DIRECT_SAMPLE_UNRESOLVED")
        if any(
            sample["sampled_foreground_pixels"] < canonical_floor for sample in samples
        ):
            canonical_reasons.append(
                "CANONICAL_SUPPLEMENT_FOREGROUND_PIXELS_BELOW_FLOOR"
            )
        if any(
            sample["foreground_overlap_ratio"] < minimum_overlap for sample in samples
        ):
            canonical_reasons.append(
                "CANONICAL_SUPPLEMENT_FOREGROUND_OVERLAP_BELOW_FLOOR"
            )
        if canonical_bbox["canonical_group_id"] != canonical_target:
            canonical_reasons.append("CANONICAL_SUPPLEMENT_BBOX_SAMPLE_DISAGREES")
        if any(
            sample["canonical_group_id"] != canonical_target
            for sample in canonical_perturbations
        ):
            canonical_reasons.append("CANONICAL_SUPPLEMENT_PERTURBATION_DISAGREES")
        source_view_ids = canonical_direct["canonical_source_view_ids"]
        if (
            not isinstance(source_view_ids, list)
            or source_view_ids != sorted(set(source_view_ids))
            or len(source_view_ids) < MIN_CANONICAL_PALETTE_SOURCE_VIEWS
        ):
            canonical_reasons.append("CANONICAL_SUPPLEMENT_SOURCE_VIEWS_INVALID")
        alternatives = sorted(
            {
                str(row["canonical_group_id"])
                for row in canonical_direct["group_scores"]
                if isinstance(row.get("canonical_group_id"), str)
                and row["canonical_group_id"] != canonical_target
                and float(row["color_share"])
                >= float(policy["minimum_diagnostic_color_share"])
            }
        )
        canonical_palette_diagnostic = {
            "status": "resolved" if not canonical_reasons else "rejected",
            "reason_codes": canonical_reasons,
            "evidence_scope": "canonical_multiview_propagation_repair_only",
            "local_group_id": canonical_direct["local_group_id"],
            "canonical_group_id": canonical_target,
            "bbox_canonical_group_id": canonical_bbox["canonical_group_id"],
            "registration_label_stable": (
                canonical_bbox["canonical_group_id"] == canonical_target
                if canonical_target is not None
                else None
            ),
            "perturbation_label_stable": (
                all(
                    sample["canonical_group_id"] == canonical_target
                    for sample in canonical_perturbations
                )
                if canonical_target is not None
                else None
            ),
            "resolved_sample_count": sum(
                isinstance(sample["canonical_group_id"], str) for sample in samples
            ),
            "target_sample_count": sum(
                sample["canonical_group_id"] == canonical_target
                for sample in samples
                if canonical_target is not None
            ),
            "consensus_ratio": round(
                (
                    sum(
                        sample["canonical_group_id"] == canonical_target
                        for sample in samples
                    )
                    / len(samples)
                )
                if canonical_target is not None
                else 0.0,
                8,
            ),
            "canonical_source_view_ids": source_view_ids,
            "minimum_foreground_overlap": minimum_overlap,
            "direct_sample": canonical_direct,
            "bbox_sample": canonical_bbox,
            "projection_perturbations": canonical_perturbations,
            "alternative_canonical_group_ids": alternatives,
        }
    # Bbox normalization is only the deterministic initializer for ECC.  An
    # unresolved bbox sample is inconclusive, not a contradictory material.
    # A registration conflict exists only when both projections confidently
    # resolve and their canonical labels differ.
    registration_label_stable = (
        bbox_canonical == refined_canonical
        if bbox_canonical is not None and refined_canonical is not None
        else None
    )
    perturbation_groups: list[dict[str, Any]] = []
    if refined_canonical is not None or diagnostic_refined_canonical is not None:
        for offset_x, offset_y in ((-2, 0), (2, 0), (0, -2), (0, 2)):
            shifted_x = projected_x + offset_x
            shifted_y = projected_y + offset_y
            inside = (
                (shifted_x >= 0)
                & (shifted_x < image.shape[1])
                & (shifted_y >= 0)
                & (shifted_y < image.shape[0])
            )
            shifted_labels = Counter(
                _pixel_label(pixel)
                for pixel in image[shifted_y[inside], shifted_x[inside]]
            )
            shifted_scores, shifted_sampled, shifted_margin = score_labels(
                shifted_labels
            )
            shifted_canonical = resolved_group(
                shifted_scores[0],
                shifted_margin,
                minimum_share=float(policy["minimum_color_share"]),
                minimum_margin=float(policy["minimum_color_margin"]),
            )
            diagnostic_shifted_canonical = resolved_group(
                shifted_scores[0],
                shifted_margin,
                minimum_share=float(policy["minimum_diagnostic_color_share"]),
                minimum_margin=float(policy["minimum_diagnostic_color_margin"]),
            )
            perturbation_groups.append(
                {
                    "offset_pixels": [offset_x, offset_y],
                    "sampled_reference_pixels": shifted_sampled,
                    "canonical_group_id": shifted_canonical,
                    "diagnostic_canonical_group_id": (diagnostic_shifted_canonical),
                    "best_color_share": shifted_scores[0]["color_share"],
                    "color_margin": round(shifted_margin, 8),
                }
            )
    perturbation_label_stable = (
        all(
            item["canonical_group_id"] == refined_canonical
            for item in perturbation_groups
        )
        if perturbation_groups
        else None
    )
    diagnostic_sample_groups = [
        group_id
        for group_id in (
            diagnostic_refined_canonical,
            diagnostic_bbox_canonical,
            *(
                item.get("diagnostic_canonical_group_id")
                for item in perturbation_groups
            ),
        )
        if isinstance(group_id, str) and group_id
    ]
    diagnostic_counts = Counter(diagnostic_sample_groups)
    diagnostic_target_count = (
        diagnostic_counts.get(diagnostic_refined_canonical, 0)
        if diagnostic_refined_canonical is not None
        else 0
    )
    diagnostic_consensus_ratio = (
        diagnostic_target_count / len(diagnostic_sample_groups)
        if diagnostic_sample_groups
        else 0.0
    )
    diagnostic_reasons: list[str] = []
    diagnostic_registration_stable = (
        diagnostic_bbox_canonical == diagnostic_refined_canonical
        if diagnostic_bbox_canonical is not None
        and diagnostic_refined_canonical is not None
        else None
    )
    if diagnostic_refined_canonical is None:
        diagnostic_reasons.append("DIAGNOSTIC_REFINED_SAMPLE_UNRESOLVED")
    if diagnostic_registration_stable is not True:
        diagnostic_reasons.append("DIAGNOSTIC_BBOX_SAMPLE_DISAGREES")
    if len(diagnostic_sample_groups) < int(
        policy["minimum_diagnostic_resolved_samples"]
    ):
        diagnostic_reasons.append("DIAGNOSTIC_RESOLVED_SAMPLE_COUNT_BELOW_FLOOR")
    if diagnostic_consensus_ratio < float(policy["minimum_diagnostic_consensus_ratio"]):
        diagnostic_reasons.append("DIAGNOSTIC_PERTURBATION_CONSENSUS_BELOW_FLOOR")
    is_bounded_diagnostic_case = (
        declared < int(policy["minimum_visible_pixels"])
        or len(projected_x) < int(policy["minimum_visible_pixels"])
        or perturbation_label_stable is False
    )
    if not is_bounded_diagnostic_case:
        diagnostic_reasons.append("DIAGNOSTIC_NOT_REQUIRED_FOR_STABLE_AUTO_PROJECTION")
    small_part_diagnostic = {
        "status": "resolved" if not diagnostic_reasons else "rejected",
        "reason_codes": diagnostic_reasons,
        "local_group_id": (
            best["local_group_id"] if diagnostic_refined_canonical is not None else None
        ),
        "canonical_group_id": diagnostic_refined_canonical,
        "bbox_canonical_group_id": diagnostic_bbox_canonical,
        "registration_label_stable": diagnostic_registration_stable,
        "resolved_sample_count": len(diagnostic_sample_groups),
        "target_sample_count": diagnostic_target_count,
        "consensus_ratio": round(diagnostic_consensus_ratio, 8),
        "alternative_canonical_group_ids": sorted(
            group_id
            for group_id in diagnostic_counts
            if group_id != diagnostic_refined_canonical
        ),
    }
    unique_black_group_ids = {
        str(group["canonical_group_id"])
        for group in unique_canonical_palette_groups
        if isinstance(group, Mapping)
        and isinstance(group.get("canonical_group_id"), str)
        and str(group.get("base_color", "")).strip().casefold() == "black"
        and isinstance(group.get("accepted_labels"), Sequence)
        and not isinstance(group.get("accepted_labels"), (str, bytes))
        and all(isinstance(label, str) for label in group.get("accepted_labels", []))
        and set(group.get("accepted_labels", [])) == {"black", "darkgray"}
    }
    dark_alternative_group_ids: set[str] = set()
    if isinstance(canonical_palette_diagnostic, Mapping):
        alternatives = canonical_palette_diagnostic.get(
            "alternative_canonical_group_ids"
        )
        if isinstance(alternatives, Sequence) and not isinstance(
            alternatives, (str, bytes)
        ):
            dark_alternative_group_ids.update(
                str(group_id)
                for group_id in alternatives
                if isinstance(group_id, str) and group_id
            )
        diagnostic_group_id = canonical_palette_diagnostic.get("canonical_group_id")
        if (
            isinstance(diagnostic_group_id, str)
            and diagnostic_group_id
            and diagnostic_group_id not in unique_black_group_ids
        ):
            dark_alternative_group_ids.add(diagnostic_group_id)
    if (
        isinstance(diagnostic_refined_canonical, str)
        and diagnostic_refined_canonical
        and diagnostic_refined_canonical not in unique_black_group_ids
    ):
        dark_alternative_group_ids.add(diagnostic_refined_canonical)
    dark_foreground_diagnostic = _dark_foreground_diagnostic(
        image=image,
        projected_mask=projected_mask,
        alignment=alignment,
        projected_part_pixels=int(len(projected_x)),
        canonical_palette_groups=unique_canonical_palette_groups,
        alternative_canonical_group_ids=sorted(dark_alternative_group_ids),
    )
    centroid = [
        round(float(np.mean(projected_x) / image.shape[1] * 1000.0), 4),
        round(float(np.mean(projected_y) / image.shape[0] * 1000.0), 4),
    ]
    evidence = {
        **base,
        "decoded_part_pixels": decoded_pixels,
        "projected_part_pixels": int(len(projected_x)),
        "projected_centroid_0_1000": centroid,
        "sampled_reference_pixels": sampled,
        "reference_color_counts": dict(sorted(labels.items())),
        "group_scores": scored,
        "color_margin": round(margin, 8),
        "bbox_sampled_reference_pixels": bbox_sampled,
        "bbox_reference_color_counts": dict(sorted(bbox_labels.items())),
        "bbox_group_scores": bbox_scored,
        "bbox_color_margin": round(bbox_margin, 8),
        "bbox_canonical_group_id": bbox_canonical,
        "registration_label_stable": registration_label_stable,
        "projection_perturbations": perturbation_groups,
        "perturbation_label_stable": perturbation_label_stable,
        "small_part_diagnostic": small_part_diagnostic,
        "canonical_palette_diagnostic": canonical_palette_diagnostic,
        "dark_foreground_diagnostic": dark_foreground_diagnostic,
        "accepted_evidence_box_overlaps": accepted_evidence_box_overlaps,
    }
    if declared < int(policy["minimum_visible_pixels"]):
        return {
            **evidence,
            "classification": "insufficient_visibility",
            "reason_code": "part_visible_pixels_below_floor",
        }
    if len(projected_x) < int(policy["minimum_visible_pixels"]):
        return {
            **evidence,
            "classification": "insufficient_visibility",
            "reason_code": "projected_part_pixels_below_floor",
        }
    if registration_label_stable is False:
        return {
            **evidence,
            "classification": "conflict",
            "reason_code": "registration_material_label_flip",
            "local_group_id": best["local_group_id"],
            "canonical_group_id": refined_canonical,
        }
    if perturbation_label_stable is False:
        return {
            **evidence,
            "classification": "conflict",
            "reason_code": "projection_perturbation_material_instability",
            "local_group_id": best["local_group_id"],
            "canonical_group_id": refined_canonical,
        }
    if float(best["color_share"]) < float(policy["minimum_color_share"]):
        return {
            **evidence,
            "classification": "conflict",
            "reason_code": "unresolved_reference_material",
        }
    if margin < float(policy["minimum_color_margin"]):
        return {
            **evidence,
            "classification": "conflict",
            "reason_code": "ambiguous_reference_material",
        }
    if best["canonical_group_id"] is None:
        return {
            **evidence,
            "classification": "conflict",
            "reason_code": "local_material_group_has_no_canonical_mapping",
            "local_group_id": best["local_group_id"],
            "canonical_group_id": None,
        }
    if str(best.get("base_color", "")).strip().casefold() == "black" and (
        dark_foreground_diagnostic["status"] != "resolved"
        or dark_foreground_diagnostic["canonical_group_id"]
        != best["canonical_group_id"]
    ):
        return {
            **evidence,
            "classification": "conflict",
            "reason_code": "black_projection_lacks_dark_foreground_proof",
            "local_group_id": best["local_group_id"],
            "canonical_group_id": best["canonical_group_id"],
        }
    return {
        **evidence,
        "classification": "resolved",
        "reason_code": "spatial_color_projection_resolved",
        "local_group_id": best["local_group_id"],
        "canonical_group_id": best["canonical_group_id"],
    }


def _validated_policy(policy: Mapping[str, Any] | None) -> dict[str, float | int]:
    output = dict(DEFAULT_POLICY)
    if policy is not None:
        unknown = set(policy) - set(output)
        if unknown:
            raise SpatialMappingError(
                f"unknown spatial policy fields: {sorted(unknown)}"
            )
        output.update(policy)
    for name in (
        "normalized_mask_size",
        "minimum_visible_pixels",
        "minimum_diagnostic_visible_pixels",
        "minimum_isolated_source_visible_pixels",
        "minimum_isolated_source_view_count",
        "minimum_diagnostic_resolved_samples",
        "minimum_spatial_support_views",
        "minimum_semantic_support_references",
        "maximum_reference_phash_distance",
        "pose_candidate_count",
    ):
        value = output[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise SpatialMappingError(f"policy.{name} must be a positive integer")
    for name in (
        "minimum_alignment_score",
        "minimum_d4_margin",
        "minimum_render_margin",
        "minimum_paired_direction_margin",
        "minimum_refined_iou",
        "minimum_refined_ecc",
        "minimum_ecc_scale",
        "maximum_ecc_shear",
        "maximum_ecc_translation_ratio",
        "minimum_color_share",
        "minimum_color_margin",
        "minimum_diagnostic_consensus_ratio",
        "minimum_diagnostic_color_share",
        "minimum_diagnostic_color_margin",
        "minimum_canonical_supplement_foreground_overlap",
        "minimum_semantic_confidence",
        "minimum_semantic_conflict_confidence",
        "configuration_mismatch_iou_ceiling",
        "configuration_mismatch_minimum_ecc",
    ):
        value = output[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not 0.0 <= float(value) <= 1.0
        ):
            raise SpatialMappingError(f"policy.{name} must be in [0,1]")
        output[name] = float(value)
    pose_degrees = output["maximum_equivalent_pose_degrees"]
    if (
        isinstance(pose_degrees, bool)
        or not isinstance(pose_degrees, (int, float))
        or not 0.0 <= float(pose_degrees) <= 180.0
    ):
        raise SpatialMappingError(
            "policy.maximum_equivalent_pose_degrees must be in [0,180]"
        )
    output["maximum_equivalent_pose_degrees"] = float(pose_degrees)
    for name in ("maximum_ecc_scale", "maximum_ecc_condition"):
        value = output[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or float(value) < 1.0
        ):
            raise SpatialMappingError(f"policy.{name} must be at least 1")
        output[name] = float(value)
    rotation = output["maximum_ecc_rotation_degrees"]
    if (
        isinstance(rotation, bool)
        or not isinstance(rotation, (int, float))
        or not 0.0 <= float(rotation) <= 45.0
    ):
        raise SpatialMappingError(
            "policy.maximum_ecc_rotation_degrees must be in [0,45]"
        )
    output["maximum_ecc_rotation_degrees"] = float(rotation)
    if float(output["minimum_ecc_scale"]) > float(output["maximum_ecc_scale"]):
        raise SpatialMappingError(
            "policy.minimum_ecc_scale cannot exceed maximum_ecc_scale"
        )
    if int(output["minimum_diagnostic_visible_pixels"]) > int(
        output["minimum_visible_pixels"]
    ):
        raise SpatialMappingError(
            "policy.minimum_diagnostic_visible_pixels cannot exceed "
            "minimum_visible_pixels"
        )
    if int(output["minimum_isolated_source_visible_pixels"]) > int(
        output["minimum_diagnostic_visible_pixels"]
    ):
        raise SpatialMappingError(
            "policy.minimum_isolated_source_visible_pixels cannot exceed "
            "minimum_diagnostic_visible_pixels"
        )
    if int(output["minimum_diagnostic_resolved_samples"]) > 6:
        raise SpatialMappingError(
            "policy.minimum_diagnostic_resolved_samples cannot exceed 6"
        )
    if float(output["minimum_semantic_conflict_confidence"]) > float(
        output["minimum_semantic_confidence"]
    ):
        raise SpatialMappingError(
            "policy.minimum_semantic_conflict_confidence cannot exceed "
            "minimum_semantic_confidence"
        )
    if int(output["maximum_reference_phash_distance"]) > 64:
        raise SpatialMappingError(
            "policy.maximum_reference_phash_distance cannot exceed 64"
        )
    return output


def _recover_multiview_dark_consensus(
    observations: list[dict[str, Any]],
    *,
    policy: Mapping[str, float | int],
) -> dict[str, Any] | None:
    """Recover a dark part only when two independent projections agree.

    A black part on a black studio background can fail the single-view edge or
    null-shift proof even when its projected interior contains a strong,
    foreground-separated dark signal.  Requiring the same part, canonical
    group, stable direct/bbox/perturbation label, and dark interior evidence
    in two separately registered reference views removes that degeneracy
    without accepting a lone background sample.
    """

    minimum_support = int(policy["minimum_spatial_support_views"])
    minimum_color_share = float(policy["minimum_color_share"])
    by_group: dict[str, list[dict[str, Any]]] = {}
    resolved_alternatives: set[str] = set()
    for observation in observations:
        canonical_group_id = observation.get("canonical_group_id")
        if (
            observation.get("classification") == "resolved"
            and isinstance(canonical_group_id, str)
            and canonical_group_id
        ):
            resolved_alternatives.add(canonical_group_id)
        if (
            observation.get("classification") != "conflict"
            or observation.get("reason_code")
            != "black_projection_lacks_dark_foreground_proof"
            or not isinstance(canonical_group_id, str)
            or not canonical_group_id
            or observation.get("registration_label_stable") is not True
            or observation.get("perturbation_label_stable") is not True
            or observation.get("bbox_canonical_group_id") != canonical_group_id
        ):
            continue
        scores = observation.get("group_scores")
        winner = (
            scores[0]
            if isinstance(scores, Sequence)
            and not isinstance(scores, (str, bytes))
            and scores
            and isinstance(scores[0], Mapping)
            else None
        )
        diagnostic = observation.get("dark_foreground_diagnostic")
        if (
            winner is None
            or winner.get("canonical_group_id") != canonical_group_id
            or str(winner.get("base_color", "")).strip().casefold() != "black"
            or not isinstance(winner.get("color_share"), (int, float))
            or isinstance(winner.get("color_share"), bool)
            or float(winner["color_share"]) < minimum_color_share
            or not isinstance(diagnostic, Mapping)
            or float(diagnostic.get("non_background_share") or 0.0) < 0.20
            or float(diagnostic.get("dark_signal_share") or 0.0)
            < float(DARK_FOREGROUND_POLICY["minimum_dark_signal_share"])
            or float(diagnostic.get("dark_signal_purity") or 0.0)
            < float(DARK_FOREGROUND_POLICY["minimum_dark_signal_purity"])
            or float(diagnostic.get("core_dark_signal_share") or 0.0)
            < float(DARK_FOREGROUND_POLICY["minimum_core_dark_signal_share"])
            or float(diagnostic.get("dark_signal_null_margin") or 0.0)
            < float(DARK_FOREGROUND_POLICY["minimum_null_q75_margin"])
        ):
            continue
        by_group.setdefault(canonical_group_id, []).append(observation)

    eligible = {
        group_id: rows
        for group_id, rows in by_group.items()
        if len({str(row["reference_view_id"]) for row in rows}) >= minimum_support
        and not (resolved_alternatives - {group_id})
    }
    if len(eligible) != 1:
        return None
    group_id, supports = next(iter(eligible.items()))
    support_view_ids = sorted(
        {str(observation["reference_view_id"]) for observation in supports}
    )
    audit = {
        "status": "resolved",
        "canonical_group_id": group_id,
        "supporting_view_ids": support_view_ids,
        "minimum_independent_support_views": minimum_support,
        "evidence_contract": (
            "stable_projection_and_dark_interior_multiview_consensus"
        ),
    }
    for observation in supports:
        observation["classification"] = "resolved"
        observation["reason_code"] = "multiview_dark_consensus_resolved"
        observation["multiview_dark_consensus"] = copy.deepcopy(audit)
    return audit


def _validated_isolated_evidence_by_part(
    parts: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Cross-check isolated evidence against renderer-authored raw visibility."""

    result: dict[str, dict[str, Any]] = {}
    for part in parts:
        part_id = part["part_id"]
        evidence = part.get("isolated_evidence")
        if evidence is None:
            continue
        if not isinstance(evidence, Mapping):
            raise SpatialMappingError(
                f"registry part {part_id} isolated_evidence must be an object"
            )
        if evidence.get("schema_version") != ISOLATED_EVIDENCE_SCHEMA_VERSION:
            raise SpatialMappingError(
                f"registry part {part_id} isolated evidence schema is unsupported"
            )
        selected = evidence.get("selected_view_ids")
        source_by_view = evidence.get("source_visible_pixels_by_view")
        normalized_by_view = evidence.get("normalized_visible_pixels_by_view")
        source_floor = evidence.get("source_pixel_floor")
        if (
            not isinstance(selected, list)
            or not selected
            or selected != list(dict.fromkeys(selected))
            or any(not isinstance(view_id, str) or not view_id for view_id in selected)
            or not isinstance(source_by_view, Mapping)
            or not isinstance(normalized_by_view, Mapping)
            or set(source_by_view) != set(selected)
            or set(normalized_by_view) != set(selected)
            or isinstance(source_floor, bool)
            or not isinstance(source_floor, int)
            or source_floor < 1
            or evidence.get("material_neutralized") is not True
            or evidence.get("background_removed") is not True
        ):
            raise SpatialMappingError(
                f"registry part {part_id} isolated evidence is malformed"
            )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for values in (source_by_view, normalized_by_view)
            for value in values.values()
        ):
            raise SpatialMappingError(
                f"registry part {part_id} isolated pixel counts are invalid"
            )
        raw_renders = part.get("renders")
        if not isinstance(raw_renders, list):
            raise SpatialMappingError(
                f"registry part {part_id} renders must be an array"
            )
        raw_by_view: dict[str, int] = {}
        for raw_render in raw_renders:
            if (
                not isinstance(raw_render, Mapping)
                or not isinstance(raw_render.get("view_id"), str)
                or isinstance(raw_render.get("visible_pixels"), bool)
                or not isinstance(raw_render.get("visible_pixels"), int)
                or raw_render["visible_pixels"] < 0
            ):
                raise SpatialMappingError(
                    f"registry part {part_id} has malformed render visibility"
                )
            view_id = raw_render["view_id"]
            if view_id in raw_by_view:
                raise SpatialMappingError(
                    f"registry part {part_id} has duplicate render view {view_id}"
                )
            raw_by_view[view_id] = raw_render["visible_pixels"]
        if any(
            view_id not in raw_by_view
            or source_by_view[view_id] != raw_by_view[view_id]
            for view_id in selected
        ):
            raise SpatialMappingError(
                f"registry part {part_id} isolated/source visibility differs"
            )
        source_max = max(source_by_view.values())
        normalized_max = max(normalized_by_view.values())
        evidence_views = sorted(
            view_id
            for view_id, pixels in source_by_view.items()
            if pixels >= source_floor
        )
        digest = evidence.get("sha256")
        if (
            evidence.get("source_max_visible_pixels") != source_max
            or evidence.get("normalized_max_visible_pixels") != normalized_max
            or evidence.get("source_evidence_view_count") != len(evidence_views)
            or evidence.get("source_evidence_view_ids") != evidence_views
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        ):
            raise SpatialMappingError(
                f"registry part {part_id} isolated evidence summary is inconsistent"
            )
        result[part_id] = {
            "schema_version": ISOLATED_EVIDENCE_SCHEMA_VERSION,
            "sha256": digest,
            "source_visible_pixels_by_view": dict(source_by_view),
            "normalized_visible_pixels_by_view": dict(normalized_by_view),
            "source_max_visible_pixels": source_max,
            "normalized_max_visible_pixels": normalized_max,
            "source_evidence_view_count": len(evidence_views),
            "source_evidence_view_ids": evidence_views,
            "source_pixel_floor": source_floor,
            "material_neutralized": True,
            "background_removed": True,
        }
    return result


def build_spatial_mapping_report(
    reference_manifest: str | Path | Mapping[str, Any],
    rendered_registry: str | Path | Mapping[str, Any],
    view_group_id_maps: Mapping[str, Mapping[str, str]],
    votes: Sequence[Mapping[str, Any]] | None = None,
    *,
    normalized_palettes_by_view: Mapping[str, Mapping[str, Any]] | None = None,
    palette_audits_by_view: Mapping[str, Mapping[str, Any]] | None = None,
    policy: Mapping[str, Any] | None = None,
    include_all_parts: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Build hash-bound per-part spatial observations from existing artefacts."""

    selected_policy = _validated_policy(policy)
    manifest, manifest_path = _read_object(reference_manifest, "reference manifest")
    registry, registry_path = _read_object(rendered_registry, "rendered registry")
    if registry.get("schema_version") != "qwen-material-parts/v1":
        raise SpatialMappingError("rendered registry.schema_version is unsupported")
    parts = registry.get("parts")
    render_set = registry.get("render_set")
    if not isinstance(parts, list) or not parts or not isinstance(render_set, Mapping):
        raise SpatialMappingError("rendered registry requires parts and render_set")
    part_ids: list[str] = []
    for part in parts:
        if not isinstance(part, Mapping) or not isinstance(part.get("part_id"), str):
            raise SpatialMappingError("registry part entries require part_id")
        part_ids.append(part["part_id"])
    if len(part_ids) != len(set(part_ids)):
        raise SpatialMappingError("rendered registry contains duplicate part IDs")
    isolated_evidence_by_part = _validated_isolated_evidence_by_part(parts)
    all_part_ids = set(part_ids)
    if not isinstance(include_all_parts, bool):
        raise SpatialMappingError("include_all_parts must be boolean")
    target_ids = set(part_ids)
    if votes is not None:
        vote_ids = {
            vote.get("part_id")
            for vote in votes
            if isinstance(vote, Mapping) and isinstance(vote.get("part_id"), str)
        }
        unknown_vote_ids = vote_ids - target_ids
        if unknown_vote_ids:
            raise SpatialMappingError(
                f"votes contain unknown part IDs: {sorted(unknown_vote_ids)}"
            )
        if not include_all_parts:
            target_ids = vote_ids

    raw_sources = manifest.get("source_views")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise SpatialMappingError("reference manifest requires source_views")
    usable_sources = [
        source
        for source in raw_sources
        if isinstance(source, Mapping)
        and source.get("palette_status", "usable") == "usable"
    ]
    if not usable_sources:
        raise SpatialMappingError("reference manifest has no usable palette views")
    if normalized_palettes_by_view is None:
        loaded_palettes: dict[str, dict[str, Any]] = {}
        for source in usable_sources:
            if not isinstance(source, Mapping) or not isinstance(source.get("id"), str):
                raise SpatialMappingError("reference source view is invalid")
            artifacts = source.get("palette_artifacts")
            if not isinstance(artifacts, Mapping):
                raise SpatialMappingError(
                    "reference source view has no palette_artifacts"
                )
            palette_path = _resolve_file(
                artifacts.get("normalized"), manifest_path, "normalized palette"
            )
            palette_doc, _ = _read_object(palette_path, "normalized palette")
            loaded_palettes[source["id"]] = palette_doc
        normalized_palettes_by_view = loaded_palettes
    if palette_audits_by_view is None:
        palette_audits_by_view = {}

    files: list[dict[str, str]] = []
    if manifest_path is not None:
        files.append(
            {
                "label": "reference_manifest",
                "path": str(manifest_path),
                "sha256": _sha256_file(manifest_path),
            }
        )
    if registry_path is not None:
        files.append(
            {
                "label": "rendered_registry",
                "path": str(registry_path),
                "sha256": _sha256_file(registry_path),
            }
        )

    references: list[dict[str, Any]] = []
    reference_sha256_by_view: dict[str, str] = {}
    reference_pixel_sha256_by_view: dict[str, str] = {}
    reference_phash_by_view: dict[str, str] = {}
    seen_reference_ids: set[str] = set()
    for source in usable_sources:
        if not isinstance(source, Mapping):
            raise SpatialMappingError("reference source view is invalid")
        view_id = source.get("id")
        if not isinstance(view_id, str) or not view_id or view_id in seen_reference_ids:
            raise SpatialMappingError("reference IDs must be unique non-empty strings")
        seen_reference_ids.add(view_id)
        image_path = _resolve_file(
            source.get("image"), manifest_path, f"reference image {view_id}"
        )
        image = _open_bgr(image_path, f"reference image {view_id}")
        foreground, foreground_mask_path, foreground_authority = (
            _reference_foreground_from_manifest(
                source=source,
                manifest_path=manifest_path,
                image=image,
                view_id=view_id,
            )
        )
        _bbox(foreground, f"reference {view_id}")
        palette = normalized_palettes_by_view.get(view_id)
        if not isinstance(palette, Mapping):
            raise SpatialMappingError(f"missing normalized palette for {view_id}")
        groups = _palette_groups(palette, view_id)
        group_map = view_group_id_maps.get(view_id)
        if not isinstance(group_map, Mapping):
            raise SpatialMappingError(f"missing view_group_id_map for {view_id}")
        palette_audit = palette_audits_by_view.get(view_id)
        accepted_palette_evidence = _accepted_palette_evidence_regions(
            image_path=image_path,
            image_shape=image.shape,
            palette_groups=groups,
            group_id_map=group_map,
            palette_audit=(
                palette_audit if isinstance(palette_audit, Mapping) else None
            ),
            view_id=view_id,
        )
        reference_image_sha256 = _sha256_file(image_path)
        files.append(
            {
                "label": f"reference_image:{view_id}",
                "path": str(image_path),
                "sha256": reference_image_sha256,
            }
        )
        foreground_mask_sha256 = None
        if foreground_mask_path is not None:
            foreground_mask_sha256 = _sha256_file(foreground_mask_path)
            files.append(
                {
                    "label": f"reference_foreground_mask:{view_id}",
                    "path": str(foreground_mask_path),
                    "sha256": foreground_mask_sha256,
                }
            )
        reference_sha256_by_view[view_id] = reference_image_sha256
        reference_pixel_sha256_by_view[view_id] = _normalized_pixel_sha256(image)
        reference_phash_by_view[view_id] = _perceptual_hash(image)
        references.append(
            {
                "view_id": view_id,
                "image_path": image_path,
                "image": image,
                "foreground": foreground,
                "foreground_authority": foreground_authority,
                "foreground_mask_path": foreground_mask_path,
                "foreground_mask_sha256": foreground_mask_sha256,
                "palette_groups": groups,
                "group_id_map": dict(group_map),
                "accepted_palette_evidence": accepted_palette_evidence,
            }
        )

    unique_canonical_palette_groups = _unique_multiview_canonical_palette_groups(
        references
    )
    canonical_palette_supplements = _canonical_palette_supplements(references)
    reference_content_cluster_by_view = _perceptual_clusters(
        reference_phash_by_view,
        int(selected_policy["maximum_reference_phash_distance"]),
    )
    raw_render_views = render_set.get("views")
    if not isinstance(raw_render_views, list) or not raw_render_views:
        raise SpatialMappingError("rendered registry render_set requires views")
    render_total = len(raw_render_views)
    report_progress(
        progress_callback,
        scope=PROGRESS_SCOPE,
        stage="spatial_render_decode",
        state="start",
        current=0,
        total=render_total,
        unit="views",
        detail="spatial render decoding started",
    )
    part_colors_bgr = [
        np.asarray(
            (_part_color(part_id)[2], _part_color(part_id)[1], _part_color(part_id)[0]),
            dtype=np.uint8,
        )
        for part_id in part_ids
    ]
    renders: list[dict[str, Any]] = []
    seen_render_ids: set[str] = set()
    for render_index, raw in enumerate(raw_render_views, start=1):
        if not isinstance(raw, Mapping):
            raise SpatialMappingError("render view is invalid")
        view_id = raw.get("view_id")
        if not isinstance(view_id, str) or not view_id or view_id in seen_render_ids:
            raise SpatialMappingError(
                "render view IDs must be unique non-empty strings"
            )
        seen_render_ids.add(view_id)
        rgb_path = _resolve_file(raw.get("rgb"), registry_path, f"render RGB {view_id}")
        ids_path = _resolve_file(
            raw.get("part_ids_raw") or raw.get("part_ids"),
            registry_path,
            f"raw part IDs {view_id}",
        )
        rgb = _open_bgr(rgb_path, f"render RGB {view_id}")
        ids_image = _open_bgr(ids_path, f"part IDs {view_id}")
        if rgb.shape != ids_image.shape:
            raise SpatialMappingError(f"render {view_id} RGB/part-ID dimensions differ")
        visible_raw = raw.get("visible_parts")
        if not isinstance(visible_raw, list):
            raise SpatialMappingError(f"render {view_id} has no visible_parts")
        analysis_direction = _validated_direction(
            raw.get("analysis_direction"),
            f"render {view_id}.analysis_direction",
        )
        world_direction = _validated_direction(
            raw.get("world_direction"),
            f"render {view_id}.world_direction",
        )
        camera_up_axis = _validated_direction(
            raw.get("camera_up_axis"),
            f"render {view_id}.camera_up_axis",
        )
        camera_position_raw = raw.get("camera_position")
        camera_position = None
        if camera_position_raw is not None:
            if (
                not isinstance(camera_position_raw, Sequence)
                or isinstance(camera_position_raw, (str, bytes))
                or len(camera_position_raw) != 3
                or any(
                    isinstance(component, bool)
                    or not isinstance(component, (int, float))
                    or not math.isfinite(float(component))
                    for component in camera_position_raw
                )
            ):
                raise SpatialMappingError(
                    f"render {view_id}.camera_position must contain three "
                    "finite numbers"
                )
            camera_position = [
                round(float(component), 10) for component in camera_position_raw
            ]
        focal_length = raw.get("focal_length_mm")
        if focal_length is not None and (
            isinstance(focal_length, bool)
            or not isinstance(focal_length, (int, float))
            or not math.isfinite(float(focal_length))
            or float(focal_length) <= 0.0
        ):
            raise SpatialMappingError(
                f"render {view_id}.focal_length_mm must be positive"
            )
        visible_pixels: dict[str, int] = {}
        for item in visible_raw:
            if (
                not isinstance(item, Mapping)
                or not isinstance(item.get("part_id"), str)
                or isinstance(item.get("pixels"), bool)
                or not isinstance(item.get("pixels"), int)
                or item["pixels"] < 0
            ):
                raise SpatialMappingError(f"render {view_id} has invalid visible_parts")
            visible_pixels[item["part_id"]] = item["pixels"]
        decoded_pixels: dict[str, int] = {}
        for part_id in part_ids:
            red, green, blue = _part_color(part_id)
            color_bgr = np.asarray((blue, green, red), dtype=np.uint8)
            decoded_pixels[part_id] = int(
                np.count_nonzero(np.all(ids_image == color_bgr, axis=2))
            )
        mismatches = {
            part_id: {
                "declared": visible_pixels.get(part_id, 0),
                "decoded": decoded_pixels[part_id],
            }
            for part_id in part_ids
            if decoded_pixels[part_id] != visible_pixels.get(part_id, 0)
        }
        if mismatches:
            sample = dict(list(sorted(mismatches.items()))[:8])
            raise SpatialMappingError(
                "raw part-ID image does not exactly match visible_parts for "
                f"{view_id}; the image may contain labels or lossy pixels: {sample}"
            )
        foreground = _cad_foreground(rgb, ids_image, part_colors_bgr)
        _bbox(foreground, f"render {view_id}")
        for label, path in (
            (f"render_rgb:{view_id}", rgb_path),
            (f"part_ids:{view_id}", ids_path),
        ):
            files.append(
                {"label": label, "path": str(path), "sha256": _sha256_file(path)}
            )
        renders.append(
            {
                "view_id": view_id,
                "rgb_path": rgb_path,
                "part_ids_path": ids_path,
                "rgb": rgb,
                "part_ids_image": ids_image,
                "foreground": foreground,
                "visible_pixels": visible_pixels,
                "analysis_direction": analysis_direction,
                "world_direction": world_direction,
                "camera_up_axis": camera_up_axis,
                "camera_position": camera_position,
                "focal_length_mm": (
                    round(float(focal_length), 8)
                    if focal_length is not None
                    else None
                ),
            }
        )
        report_progress(
            progress_callback,
            scope=PROGRESS_SCOPE,
            stage="spatial_render_decode",
            state="update",
            current=render_index,
            total=render_total,
            unit="views",
            detail=f"spatial render {view_id} decoded",
        )
    report_progress(
        progress_callback,
        scope=PROGRESS_SCOPE,
        stage="spatial_render_decode",
        state="complete",
        current=render_total,
        total=render_total,
        unit="views",
        detail="spatial render decoding completed",
    )

    alignments = _associate_views(
        references,
        renders,
        selected_policy,
        progress_callback=progress_callback,
    )
    reference_by_id = {item["view_id"]: item for item in references}
    render_by_id = {item["view_id"]: item for item in renders}
    alignment_by_reference_id = {item["reference_view_id"]: item for item in alignments}
    semantic_votes_by_part: dict[str, list[dict[str, Any]]] = {
        part_id: [] for part_id in target_ids
    }
    raw_votes = list(votes) if votes is not None else []
    semantic_vote_total = len(raw_votes)
    if semantic_vote_total:
        report_progress(
            progress_callback,
            scope=PROGRESS_SCOPE,
            stage="spatial_semantic_votes",
            state="start",
            current=0,
            total=semantic_vote_total,
            unit="votes",
            detail="spatial semantic vote processing started",
        )
    else:
        report_progress(
            progress_callback,
            scope=PROGRESS_SCOPE,
            stage="spatial_semantic_votes",
            state="start",
            detail="spatial semantic vote processing has no votes",
        )
    seen_semantic_votes: set[tuple[str, str]] = set()
    for index, raw_vote in enumerate(raw_votes):
        if not isinstance(raw_vote, Mapping):
            raise SpatialMappingError(f"votes[{index}] must be an object")
        part_id = raw_vote.get("part_id")
        view_id = raw_vote.get("view_id")
        status = raw_vote.get("status")
        confidence = raw_vote.get("confidence")
        local_group_id = raw_vote.get("local_group_id")
        canonical_group_id = raw_vote.get("canonical_group_id")
        reason_code = raw_vote.get("reason_code")
        if not isinstance(part_id, str) or part_id not in all_part_ids:
            raise SpatialMappingError(f"votes[{index}] has an invalid part_id")
        if not isinstance(view_id, str) or view_id not in reference_by_id:
            raise SpatialMappingError(f"votes[{index}] has an invalid view_id")
        identity = (part_id, view_id)
        if identity in seen_semantic_votes:
            raise SpatialMappingError(
                f"duplicate semantic vote for part/view: {part_id}/{view_id}"
            )
        seen_semantic_votes.add(identity)
        if status not in {"matched", "review", "unknown"}:
            raise SpatialMappingError(f"votes[{index}] has an invalid status")
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not math.isfinite(float(confidence))
            or not 0.0 <= float(confidence) <= 1.0
        ):
            raise SpatialMappingError(f"votes[{index}] has invalid confidence")
        if canonical_group_id is not None and (
            not isinstance(canonical_group_id, str) or not canonical_group_id
        ):
            raise SpatialMappingError(f"votes[{index}] has invalid canonical_group_id")
        if status == "unknown" and canonical_group_id is not None:
            raise SpatialMappingError(
                f"votes[{index}] unknown vote cannot cite a canonical group"
            )
        if status != "unknown" and not isinstance(reason_code, str):
            raise SpatialMappingError(f"votes[{index}] has invalid reason_code")
        if local_group_id is not None and (
            not isinstance(local_group_id, str) or not local_group_id
        ):
            raise SpatialMappingError(f"votes[{index}] has invalid local_group_id")
        palette_groups = {
            str(group["group_id"]): group
            for group in reference_by_id[view_id]["palette_groups"]
        }
        palette_group = (
            palette_groups.get(local_group_id)
            if isinstance(local_group_id, str)
            else None
        )
        expected_canonical = (
            reference_by_id[view_id]["group_id_map"].get(local_group_id)
            if isinstance(local_group_id, str)
            else None
        )
        unique_canonical_join = (
            canonical_group_id is not None
            and expected_canonical == canonical_group_id
            and palette_group is not None
        )
        palette_confidence = (
            float(palette_group["confidence"]) if palette_group is not None else 0.0
        )
        effective_confidence = min(float(confidence), palette_confidence)
        alignment = alignment_by_reference_id[view_id]
        if part_id not in semantic_votes_by_part:
            report_progress(
                progress_callback,
                scope=PROGRESS_SCOPE,
                stage="spatial_semantic_votes",
                state="update",
                current=index + 1,
                total=semantic_vote_total,
                unit="votes",
                detail=f"spatial semantic vote {part_id}/{view_id} processed",
            )
            continue
        aligned_render_view_id = alignment["selected_render_view_id"]
        aligned_render = render_by_id[aligned_render_view_id]
        declared_part_pixels = int(aligned_render["visible_pixels"].get(part_id, 0))
        isolated = isolated_evidence_by_part.get(part_id)
        isolated_source_by_view = (
            isolated.get("source_visible_pixels_by_view", {})
            if isinstance(isolated, Mapping)
            else {}
        )
        isolated_visibility = (
            isinstance(isolated, Mapping)
            and isolated.get("material_neutralized") is True
            and isolated.get("background_removed") is True
            and int(isolated.get("source_evidence_view_count", 0))
            >= int(selected_policy["minimum_isolated_source_view_count"])
            and isinstance(isolated_source_by_view, Mapping)
            and isolated_source_by_view.get(aligned_render_view_id)
            == declared_part_pixels
            and declared_part_pixels
            >= int(selected_policy["minimum_isolated_source_visible_pixels"])
        )
        semantic_visibility_floor = (
            int(selected_policy["minimum_isolated_source_visible_pixels"])
            if isolated_visibility
            else int(selected_policy["minimum_diagnostic_visible_pixels"])
        )
        semantic_visibility_eligible = (
            bool(alignment["trusted"])
            and declared_part_pixels >= semantic_visibility_floor
        )
        semantic_votes_by_part[part_id].append(
            {
                "view_id": view_id,
                "reference_sha256": reference_sha256_by_view[view_id],
                "normalized_pixel_sha256": (reference_pixel_sha256_by_view[view_id]),
                "perceptual_hash": reference_phash_by_view[view_id],
                "content_cluster_id": reference_content_cluster_by_view[view_id],
                "pose_cluster_id": (
                    alignment["selected_render_view_id"]
                    if alignment["trusted"]
                    else None
                ),
                "alignment_trusted": bool(alignment["trusted"]),
                "cad_part_visible_pixels": declared_part_pixels,
                "cad_part_visibility_floor": semantic_visibility_floor,
                "cad_part_visibility_eligible": semantic_visibility_eligible,
                "cad_part_evidence_mode": (
                    "isolated_mask_multiview"
                    if isolated_visibility
                    else "source_projection"
                ),
                "isolated_evidence_sha256": (
                    isolated.get("sha256") if isolated_visibility else None
                ),
                "local_group_id": local_group_id,
                "canonical_group_id": canonical_group_id,
                "status": status,
                "mapping_confidence": float(confidence),
                "palette_confidence": palette_confidence,
                "pixel_gate_accepted": palette_group is not None,
                "unique_canonical_join": unique_canonical_join,
                "effective_confidence": effective_confidence,
                "reason_code": reason_code,
            }
        )
        report_progress(
            progress_callback,
            scope=PROGRESS_SCOPE,
            stage="spatial_semantic_votes",
            state="update",
            current=index + 1,
            total=semantic_vote_total,
            unit="votes",
            detail=f"spatial semantic vote {part_id}/{view_id} processed",
        )
    if semantic_vote_total:
        report_progress(
            progress_callback,
            scope=PROGRESS_SCOPE,
            stage="spatial_semantic_votes",
            state="complete",
            current=semantic_vote_total,
            total=semantic_vote_total,
            unit="votes",
            detail="spatial semantic vote processing completed",
        )
    else:
        report_progress(
            progress_callback,
            scope=PROGRESS_SCOPE,
            stage="spatial_semantic_votes",
            state="complete",
            detail="spatial semantic vote processing completed with no votes",
        )
    part_reports: list[dict[str, Any]] = []
    ordered_target_ids = sorted(target_ids)
    observation_part_total = len(ordered_target_ids)
    if observation_part_total:
        report_progress(
            progress_callback,
            scope=PROGRESS_SCOPE,
            stage="spatial_part_observations",
            state="start",
            current=0,
            total=observation_part_total,
            unit="parts",
            detail="spatial part observations started",
        )
    else:
        report_progress(
            progress_callback,
            scope=PROGRESS_SCOPE,
            stage="spatial_part_observations",
            state="start",
            detail="spatial part observations have no candidate parts",
        )
    for part_index, part_id in enumerate(ordered_target_ids, start=1):
        observations: list[dict[str, Any]] = []
        for alignment in alignments:
            if not alignment["observation_eligible"]:
                continue
            ref_id = alignment["reference_view_id"]
            observations.append(
                _project_part_observation(
                    reference=reference_by_id[ref_id],
                    render=render_by_id[alignment["selected_render_view_id"]],
                    alignment=alignment,
                    part_id=part_id,
                    palette_groups=reference_by_id[ref_id]["palette_groups"],
                    group_id_map=reference_by_id[ref_id]["group_id_map"],
                    policy=selected_policy,
                    canonical_palette_groups=canonical_palette_supplements.get(
                        ref_id, []
                    ),
                    unique_canonical_palette_groups=[
                        copy.deepcopy(record)
                        for _canonical_group_id, record in sorted(
                            unique_canonical_palette_groups.items()
                        )
                    ],
                    accepted_palette_evidence=reference_by_id[ref_id].get(
                        "accepted_palette_evidence", []
                    ),
                    isolated_evidence=isolated_evidence_by_part.get(part_id),
                )
            )
        multiview_dark_consensus = _recover_multiview_dark_consensus(
            observations,
            policy=selected_policy,
        )
        resolved_counts = Counter(
            observation["canonical_group_id"]
            for observation in observations
            if observation["classification"] == "resolved"
        )
        conflict_views = sorted(
            observation["reference_view_id"]
            for observation in observations
            if observation["classification"] == "conflict"
        )
        part_reports.append(
            {
                "part_id": part_id,
                "observations": observations,
                "resolved_support_counts": dict(sorted(resolved_counts.items())),
                "conflict_view_ids": conflict_views,
                "multiview_dark_consensus": multiview_dark_consensus,
                "semantic_votes": sorted(
                    semantic_votes_by_part[part_id],
                    key=lambda item: item["view_id"],
                ),
            }
        )
        report_progress(
            progress_callback,
            scope=PROGRESS_SCOPE,
            stage="spatial_part_observations",
            state="update",
            current=part_index,
            total=observation_part_total,
            unit="parts",
            detail=(
                f"spatial observations for {part_id} completed "
                f"({len(observations)} views)"
            ),
        )
    if observation_part_total:
        report_progress(
            progress_callback,
            scope=PROGRESS_SCOPE,
            stage="spatial_part_observations",
            state="complete",
            current=observation_part_total,
            total=observation_part_total,
            unit="parts",
            detail="spatial part observations completed",
        )
    else:
        report_progress(
            progress_callback,
            scope=PROGRESS_SCOPE,
            stage="spatial_part_observations",
            state="complete",
            detail="spatial part observations completed with no candidate parts",
        )

    documents = {
        "reference_manifest": _sha256_document(manifest),
        "rendered_registry": _sha256_document(registry),
        "view_group_id_maps": _sha256_document(view_group_id_maps),
        "votes": _sha256_document(raw_votes if votes is not None else None),
        "normalized_palettes_by_view": _sha256_document(normalized_palettes_by_view),
        "palette_audits_by_view": _sha256_document(palette_audits_by_view),
        "include_all_parts": _sha256_document(include_all_parts),
    }
    configured_view_presets = render_set.get("view_presets", [])
    if not isinstance(configured_view_presets, list) or any(
        not isinstance(value, str) for value in configured_view_presets
    ):
        raise SpatialMappingError("rendered registry view_presets must be an array")
    camera_pose_search = {
        "schema_version": "qwen-camera-pose-search/v1",
        "mode": (
            "dense_spherical_render_and_compare"
            if configured_view_presets or len(renders) > CANONICAL_RENDER_VIEW_COUNT
            else "canonical_render_and_compare"
        ),
        "view_bank_size": len(renders),
        "view_presets": list(configured_view_presets),
        "top_k_per_reference": int(selected_policy["pose_candidate_count"]),
        "selection_features": [
            "foreground_silhouette_iou",
            "symmetric_edge_chamfer",
            "bbox_aspect",
            "ecc_refined_projection_iou",
        ],
        "rgb_material_used_for_pose_selection": False,
        "references": [
            {
                "reference_view_id": alignment["reference_view_id"],
                "selected_render_view_id": alignment["selected_render_view_id"],
                "camera_pose": copy.deepcopy(alignment["camera_pose"]),
                "pose_hypotheses": copy.deepcopy(alignment["pose_hypotheses"]),
                "configuration_diagnostic": copy.deepcopy(
                    alignment["configuration_diagnostic"]
                ),
            }
            for alignment in alignments
        ],
    }
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "policy": selected_policy,
        "inputs": {"files": files, "document_sha256": documents},
        "canonical_palette_propagation": {
            "minimum_source_view_count": MIN_CANONICAL_PALETTE_SOURCE_VIEWS,
            "unique_multiview_groups": [
                copy.deepcopy(record)
                for _canonical_group_id, record in sorted(
                    unique_canonical_palette_groups.items()
                )
            ],
            "supplements_by_view": canonical_palette_supplements,
        },
        "reference_evidence": [
            {
                "view_id": reference["view_id"],
                "raw_sha256": reference_sha256_by_view[reference["view_id"]],
                "normalized_pixel_sha256": (
                    reference_pixel_sha256_by_view[reference["view_id"]]
                ),
                "perceptual_hash": reference_phash_by_view[reference["view_id"]],
                "content_cluster_id": (
                    reference_content_cluster_by_view[reference["view_id"]]
                ),
                "foreground_authority": reference["foreground_authority"],
                "foreground_mask_sha256": reference["foreground_mask_sha256"],
                "selected_render_view_id": alignment_by_reference_id[
                    reference["view_id"]
                ]["selected_render_view_id"],
                "pose_cluster_id": (
                    alignment_by_reference_id[reference["view_id"]][
                        "selected_render_view_id"
                    ]
                    if alignment_by_reference_id[reference["view_id"]]["trusted"]
                    else None
                ),
                "alignment_trusted": bool(
                    alignment_by_reference_id[reference["view_id"]]["trusted"]
                ),
                "alignment_observation_eligible": bool(
                    alignment_by_reference_id[reference["view_id"]][
                        "observation_eligible"
                    ]
                ),
                "alignment_score": alignment_by_reference_id[reference["view_id"]][
                    "score"
                ],
                "alignment_reason_codes": copy.deepcopy(
                    alignment_by_reference_id[reference["view_id"]]["reason_codes"]
                ),
                "alignment_warning_codes": copy.deepcopy(
                    alignment_by_reference_id[reference["view_id"]]["warning_codes"]
                ),
                "accepted_palette_evidence": [
                    {
                        key: copy.deepcopy(value)
                        for key, value in record.items()
                        if key != "_mask"
                    }
                    for record in reference.get("accepted_palette_evidence", [])
                ],
            }
            for reference in references
        ],
        "observation_view_mapping": {
            alignment["reference_view_id"]: alignment["selected_render_view_id"]
            for alignment in alignments
            if alignment["observation_eligible"]
        },
        "camera_pose_search": camera_pose_search,
        "view_alignments": alignments,
        "parts": part_reports,
        "summary": {
            "reference_view_count": len(references),
            "render_view_count": len(renders),
            "trusted_alignment_count": sum(1 for item in alignments if item["trusted"]),
            "observation_eligible_alignment_count": sum(
                1 for item in alignments if item["observation_eligible"]
            ),
            "candidate_part_count": len(part_reports),
            "camera_pose_search_mode": camera_pose_search["mode"],
            "camera_pose_bank_size": len(renders),
            "possible_configuration_mismatch_count": sum(
                alignment["configuration_diagnostic"]["status"]
                == "POSSIBLE_REFERENCE_CAD_CONFIGURATION_MISMATCH"
                for alignment in alignments
            ),
            "resolved_observation_count": sum(
                observation["classification"] == "resolved"
                for part in part_reports
                for observation in part["observations"]
            ),
            "conflict_observation_count": sum(
                observation["classification"] == "conflict"
                for part in part_reports
                for observation in part["observations"]
            ),
            "resolved_small_part_diagnostic_count": sum(
                isinstance(observation.get("small_part_diagnostic"), Mapping)
                and observation["small_part_diagnostic"].get("status") == "resolved"
                for part in part_reports
                for observation in part["observations"]
            ),
            "resolved_dark_foreground_diagnostic_count": sum(
                isinstance(observation.get("dark_foreground_diagnostic"), Mapping)
                and observation["dark_foreground_diagnostic"].get("status")
                == "resolved"
                for part in part_reports
                for observation in part["observations"]
            ),
            "resolved_multiview_dark_consensus_part_count": sum(
                part.get("multiview_dark_consensus") is not None
                for part in part_reports
            ),
            "canonical_palette_supplement_count": sum(
                len(groups) for groups in canonical_palette_supplements.values()
            ),
            "fail_closed": True,
        },
    }
    report["integrity"] = {"report_sha256": _sha256_document(report)}
    return report


def _verify_report(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != SCHEMA_VERSION:
        raise SpatialMappingError("spatial report.schema_version is unsupported")
    integrity = report.get("integrity")
    if not isinstance(integrity, Mapping) or not isinstance(
        integrity.get("report_sha256"), str
    ):
        raise SpatialMappingError("spatial report has no integrity hash")
    unsigned = copy.deepcopy(dict(report))
    unsigned.pop("integrity", None)
    if _sha256_document(unsigned) != integrity["report_sha256"]:
        raise SpatialMappingError("spatial report SHA256 integrity check failed")
    inputs = report.get("inputs")
    if not isinstance(inputs, Mapping) or not isinstance(inputs.get("files"), list):
        raise SpatialMappingError("spatial report inputs are invalid")
    for record in inputs["files"]:
        if not isinstance(record, Mapping):
            raise SpatialMappingError("spatial report file record is invalid")
        path = _resolve_file(record.get("path"), None, "spatial input")
        expected = record.get("sha256")
        if not isinstance(expected, str) or _sha256_file(path) != expected:
            raise SpatialMappingError(f"spatial input hash changed or is stale: {path}")
    if not isinstance(report.get("parts"), list):
        raise SpatialMappingError("spatial report parts are invalid")


def apply_spatial_gate_to_batches(
    qwen_gate_batches: Sequence[Mapping[str, Any]],
    spatial_report: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate existing automatic mappings; never create a new match.

    A mapping stays automatic when either two trusted spatial projections or
    two high-confidence votes from byte-distinct reference images support the
    same canonical group.  A resolved spatial or semantic contradiction still
    fails closed.  Missing projection evidence is inconclusive rather than a
    contradiction, which is important for real photos with occlusion and
    perspective differences.
    """

    _verify_report(spatial_report)
    policy = spatial_report.get("policy")
    if not isinstance(policy, Mapping):
        raise SpatialMappingError("spatial report policy is invalid")
    minimum_support = int(policy["minimum_spatial_support_views"])
    minimum_semantic_support = int(policy["minimum_semantic_support_references"])
    minimum_semantic_confidence = float(policy["minimum_semantic_confidence"])
    minimum_semantic_conflict_confidence = float(
        policy["minimum_semantic_conflict_confidence"]
    )
    parts_by_id: dict[str, Mapping[str, Any]] = {}
    for part in spatial_report["parts"]:
        if not isinstance(part, Mapping) or not isinstance(part.get("part_id"), str):
            raise SpatialMappingError("spatial report contains an invalid part")
        if part["part_id"] in parts_by_id:
            raise SpatialMappingError("spatial report contains duplicate parts")
        parts_by_id[part["part_id"]] = part

    gate_batches = copy.deepcopy(list(qwen_gate_batches))
    decisions: list[dict[str, Any]] = []
    seen_parts: set[str] = set()
    for batch in gate_batches:
        if not isinstance(batch, dict) or not isinstance(batch.get("mappings"), list):
            raise SpatialMappingError("Qwen gate batch is invalid")
        batch_id = batch.get("batch_id")
        if not isinstance(batch_id, str):
            raise SpatialMappingError("Qwen gate batch has no batch_id")
        for row in batch["mappings"]:
            if not isinstance(row, dict) or not isinstance(row.get("part_id"), str):
                raise SpatialMappingError("Qwen gate mapping is invalid")
            part_id = row["part_id"]
            if part_id in seen_parts:
                raise SpatialMappingError(f"duplicate gate mapping for {part_id}")
            seen_parts.add(part_id)
            input_status = row.get("status")
            input_group = row.get("group_id")
            input_confidence = row.get("mapping_confidence")
            if input_status not in {"matched", "review", "unknown"}:
                raise SpatialMappingError(f"invalid mapping status for {part_id}")
            if input_status != "matched":
                # Preserve the object byte-for-byte at the decision level.  The
                # spatial gate is deliberately incapable of promotion.
                continue
            if not isinstance(input_group, str) or not input_group:
                raise SpatialMappingError(f"matched mapping lacks group for {part_id}")
            if isinstance(input_confidence, bool) or not isinstance(
                input_confidence, (int, float)
            ):
                raise SpatialMappingError(
                    f"mapping confidence is invalid for {part_id}"
                )
            part = parts_by_id.get(part_id)
            observations = (
                part.get("observations", []) if isinstance(part, Mapping) else []
            )
            supports = sorted(
                observation["reference_view_id"]
                for observation in observations
                if isinstance(observation, Mapping)
                and observation.get("classification") == "resolved"
                and observation.get("canonical_group_id") == input_group
                and observation.get("registration_label_stable") is True
                and observation.get("perturbation_label_stable") is True
            )
            spatial_conflicts = sorted(
                observation["reference_view_id"]
                for observation in observations
                if isinstance(observation, Mapping)
                and observation.get("canonical_group_id") not in {None, input_group}
                and observation.get("classification") == "resolved"
                and observation.get("registration_label_stable") is True
                and observation.get("perturbation_label_stable") is True
            )
            semantic_votes = (
                part.get("semantic_votes", []) if isinstance(part, Mapping) else []
            )
            semantic_support_rows = [
                vote
                for vote in semantic_votes
                if isinstance(vote, Mapping)
                and vote.get("status") == "matched"
                and vote.get("canonical_group_id") == input_group
                and vote.get("pixel_gate_accepted") is True
                and vote.get("unique_canonical_join") is True
                and vote.get("alignment_trusted") is True
                and vote.get("cad_part_visibility_eligible") is True
                and isinstance(vote.get("effective_confidence"), (int, float))
                and not isinstance(vote.get("effective_confidence"), bool)
                and float(vote["effective_confidence"]) >= minimum_semantic_confidence
            ]
            semantic_support_raw_hashes = sorted(
                {
                    str(vote["reference_sha256"])
                    for vote in semantic_support_rows
                    if isinstance(vote.get("reference_sha256"), str)
                }
            )
            semantic_support_pixel_hashes = sorted(
                {
                    str(vote["normalized_pixel_sha256"])
                    for vote in semantic_support_rows
                    if isinstance(vote.get("normalized_pixel_sha256"), str)
                }
            )
            semantic_support_content_clusters = sorted(
                {
                    str(vote["content_cluster_id"])
                    for vote in semantic_support_rows
                    if isinstance(vote.get("content_cluster_id"), str)
                }
            )
            semantic_support_pose_clusters = sorted(
                {
                    str(vote["pose_cluster_id"])
                    for vote in semantic_support_rows
                    if isinstance(vote.get("pose_cluster_id"), str)
                }
            )
            semantic_support_views = sorted(
                str(vote["view_id"])
                for vote in semantic_support_rows
                if isinstance(vote.get("view_id"), str)
            )
            semantic_conflicts = sorted(
                str(vote["view_id"])
                for vote in semantic_votes
                if isinstance(vote, Mapping)
                and vote.get("status") in {"matched", "review"}
                and vote.get("canonical_group_id") not in {None, input_group}
                and vote.get("unique_canonical_join") is True
                and vote.get("cad_part_visibility_eligible") is True
                and isinstance(vote.get("effective_confidence"), (int, float))
                and not isinstance(vote.get("effective_confidence"), bool)
                and float(vote["effective_confidence"])
                >= minimum_semantic_conflict_confidence
            )
            semantic_unresolved = sorted(
                str(vote["view_id"])
                for vote in semantic_votes
                if isinstance(vote, Mapping)
                and vote.get("status") in {"matched", "review"}
                and vote.get("unique_canonical_join") is not True
                and vote.get("cad_part_visibility_eligible") is True
                and isinstance(vote.get("mapping_confidence"), (int, float))
                and not isinstance(vote.get("mapping_confidence"), bool)
                and float(vote["mapping_confidence"])
                >= minimum_semantic_conflict_confidence
            )
            semantic_multi_material = sorted(
                str(vote["view_id"])
                for vote in semantic_votes
                if isinstance(vote, Mapping)
                and vote.get("cad_part_visibility_eligible") is True
                and vote.get("reason_code") == "multi_material_mesh"
            )
            conclusions_by_content_cluster: dict[str, set[str]] = {}
            for vote in semantic_votes:
                if (
                    not isinstance(vote, Mapping)
                    or vote.get("status") not in {"matched", "review"}
                    or vote.get("unique_canonical_join") is not True
                    or vote.get("cad_part_visibility_eligible") is not True
                    or not isinstance(vote.get("content_cluster_id"), str)
                    or not isinstance(vote.get("effective_confidence"), (int, float))
                    or isinstance(vote.get("effective_confidence"), bool)
                    or float(vote["effective_confidence"])
                    < minimum_semantic_conflict_confidence
                    or not isinstance(vote.get("canonical_group_id"), str)
                ):
                    continue
                conclusions_by_content_cluster.setdefault(
                    str(vote["content_cluster_id"]), set()
                ).add(str(vote["canonical_group_id"]))
            semantic_nondeterministic_clusters = sorted(
                cluster_id
                for cluster_id, conclusions in conclusions_by_content_cluster.items()
                if len(conclusions) > 1
            )
            spatial_validated = (
                len(set(supports)) >= minimum_support and not spatial_conflicts
            )
            semantic_validated = (
                len(semantic_support_pixel_hashes) >= minimum_semantic_support
                and len(semantic_support_content_clusters) >= minimum_semantic_support
                and len(semantic_support_pose_clusters) >= minimum_semantic_support
                and not semantic_conflicts
                and not semantic_unresolved
                and not semantic_multi_material
                and not semantic_nondeterministic_clusters
            )
            validation_lanes = [
                lane
                for lane, accepted in (
                    ("spatial_projection", spatial_validated),
                    ("semantic_multiview", semantic_validated),
                )
                if accepted
            ]
            reason_codes: list[str] = []
            if spatial_conflicts or semantic_conflicts:
                decision = "downgraded_preserve"
                if spatial_conflicts:
                    reason_codes.append("spatial_material_conflict")
                if semantic_conflicts:
                    reason_codes.append("semantic_material_conflict")
                row["group_id"] = None
                row["mapping_confidence"] = min(float(input_confidence), 0.599999)
                row["evidence_view_id"] = None
                row["evidence_box_index"] = None
                row["status"] = "unknown"
                row["reason_code"] = "ambiguous"
            elif not validation_lanes:
                decision = "downgraded_review"
                reason_codes.append("insufficient_independent_validation")
                if len(set(supports)) < minimum_support:
                    reason_codes.append("insufficient_independent_spatial_support")
                if (
                    len(semantic_support_content_clusters) < minimum_semantic_support
                    or len(semantic_support_pose_clusters) < minimum_semantic_support
                ):
                    reason_codes.append("insufficient_distinct_semantic_support")
                if semantic_unresolved:
                    reason_codes.append("unresolved_semantic_group_vote")
                if semantic_multi_material:
                    reason_codes.append("multi_material_mesh_vote")
                if semantic_nondeterministic_clusters:
                    reason_codes.append("intra_cluster_model_disagreement")
                row["mapping_confidence"] = min(float(input_confidence), 0.849999)
                row["status"] = "review"
                row["reason_code"] = "ambiguous"
            else:
                decision = "kept_auto"
                reason_codes.append("independent_validation_met")
                reason_codes.extend(
                    f"{lane}_validation_met" for lane in validation_lanes
                )
                reason_codes.append("no_material_conflicts")
            decisions.append(
                {
                    "part_id": part_id,
                    "batch_id": batch_id,
                    "input_group_id": input_group,
                    "input_status": input_status,
                    "input_confidence": float(input_confidence),
                    "output_group_id": row.get("group_id"),
                    "output_status": row.get("status"),
                    "output_confidence": float(row.get("mapping_confidence")),
                    "decision": decision,
                    "reason_codes": reason_codes,
                    "supporting_view_ids": supports,
                    "conflicting_view_ids": spatial_conflicts,
                    "semantic_supporting_view_ids": semantic_support_views,
                    "semantic_supporting_reference_sha256s": (
                        semantic_support_raw_hashes
                    ),
                    "semantic_supporting_pixel_sha256s": (
                        semantic_support_pixel_hashes
                    ),
                    "semantic_supporting_content_cluster_ids": (
                        semantic_support_content_clusters
                    ),
                    "semantic_supporting_pose_cluster_ids": (
                        semantic_support_pose_clusters
                    ),
                    "semantic_conflicting_view_ids": semantic_conflicts,
                    "semantic_unresolved_view_ids": semantic_unresolved,
                    "semantic_multi_material_view_ids": semantic_multi_material,
                    "semantic_nondeterministic_content_cluster_ids": (
                        semantic_nondeterministic_clusters
                    ),
                    "validation_lanes": validation_lanes,
                }
            )

    counts = Counter(record["decision"] for record in decisions)
    audit = {
        "schema_version": GATE_AUDIT_SCHEMA_VERSION,
        "policy": {
            "minimum_spatial_support_views": minimum_support,
            "minimum_semantic_support_references": minimum_semantic_support,
            "minimum_semantic_confidence": minimum_semantic_confidence,
            "minimum_semantic_conflict_confidence": (
                minimum_semantic_conflict_confidence
            ),
        },
        "decisions": decisions,
        "summary": {
            "decision_count": len(decisions),
            "kept_auto_count": counts["kept_auto"],
            "downgraded_review_count": counts["downgraded_review"],
            "downgraded_preserve_count": counts["downgraded_preserve"],
            "fail_closed": True,
            "never_promotes": True,
        },
    }
    return {"gate_batches": gate_batches, "audit": audit}
