#!/usr/bin/env python3
"""Jointly refine rigid pose and camera intrinsics from multiple pose seeds.

The source USD is immutable.  Six camera-extrinsic variables are the exact
inverse of applying one SE(3) transform to the complete registered workpiece;
they never become per-Mesh or per-subtree edits.  The sealed camera and every
GigaPose Top-K proposal are explored, then a shared-frame Isaac render decides
whether any jointly perturbed pose/intrinsic candidate is allowed to replace
the sealed camera.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from qwen_material_pipeline.evidence.camera_calibration import (
    MAX_FOCAL_LENGTH_MM,
    _alignment_candidate_sort_key,
    _angles,
    _direction,
    _normalize,
    _reference_image,
    _reference_masks,
    _roll_up_axis,
    _run_render,
    _score_candidates,
    _spec_from_score,
    _transport_up_axis,
    _write_object,
    _classify_multiview_residuals,
)
from qwen_material_pipeline.evidence.pose_model_camera_seed import (
    MODEL_NAME,
    REPORT_SCHEMA_VERSION as SEED_REPORT_SCHEMA_VERSION,
    VIEW_SPEC_SCHEMA_VERSION,
    _residual,
    _seed_by_view,
    _sha256,
    _view_record,
    pose_to_camera_spec,
    validate_proposals,
)


REPORT_SCHEMA_VERSION = "qwen-rigid-pose-camera-joint-refinement/v2"
OPTIMIZER_NAME = "sealed-multiview-anchor-multistart-trust-region"
PARAMETER_NAMES = (
    "azimuth_degrees",
    "elevation_degrees",
    "roll_degrees",
    "log_distance_multiplier",
    "target_offset_u",
    "target_offset_v",
    "log_focal_length_mm",
    "principal_point_u",
    "principal_point_v",
    "radial_distortion_k1",
    "radial_distortion_k2",
)
BASELINE_SCALES = np.asarray(
    (2.5, 2.5, 2.0, 0.06, 0.018, 0.018, 0.06, 0.018, 0.018, 0.035, 0.015),
    dtype=np.float64,
)
MODEL_SCALES = np.asarray(
    (8.0, 8.0, 5.0, 0.16, 0.04, 0.04, 0.14, 0.035, 0.035, 0.06, 0.025),
    dtype=np.float64,
)
_HALTON_BASES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31)


def _read_object(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {resolved}")
    return value


def _resolved_bound_path(
    owner: Path, document: Mapping[str, Any], key: str, hash_key: str
) -> Path:
    value = document.get(key)
    expected = document.get(hash_key)
    if not isinstance(value, str) or not isinstance(expected, str):
        raise ValueError(f"Pose seed report is missing {key}/{hash_key}")
    path = Path(value).expanduser().resolve(strict=True)
    if _sha256(path) != expected:
        raise ValueError(f"Pose seed report {key} hash does not match: {owner}")
    return path


def _radical_inverse(index: int, base: int) -> float:
    output = 0.0
    factor = 1.0 / base
    value = index
    while value:
        value, digit = divmod(value, base)
        output += digit * factor
        factor /= base
    return output


def joint_directions(*, count: int, sequence_offset: int = 0) -> list[np.ndarray]:
    """Return deterministic dense 11-D directions with antithetic coverage."""

    if count <= 0:
        return []
    output: list[np.ndarray] = []
    pair_count = (count + 1) // 2
    for pair in range(pair_count):
        index = sequence_offset + pair + 3
        vector = np.asarray(
            [2.0 * _radical_inverse(index, base) - 1.0 for base in _HALTON_BASES],
            dtype=np.float64,
        )
        # Every non-exact proposal must couple pose and intrinsics.  Avoid a
        # near-zero coordinate accidentally turning a sample into a one-axis
        # coordinate-search step.
        vector = np.where(
            np.abs(vector) < 0.06,
            np.where(vector < 0.0, -0.06, 0.06),
            vector,
        )
        output.append(vector)
        if len(output) < count:
            output.append(-vector)
    return output


def _float(seed: Mapping[str, Any], key: str, default: float) -> float:
    value = seed.get(key, default)
    if value is None:
        return default
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Camera seed {key} is not finite")
    return result


def joint_candidate_spec(
    *,
    seed: Mapping[str, Any],
    reference_id: str,
    candidate_id: str,
    vector: Sequence[float] | None,
    scales: Sequence[float],
    round_index: int,
    lineage: Mapping[str, Any],
    frame_anchor: bool = False,
) -> dict[str, Any]:
    """Apply one coupled pose/intrinsic perturbation to a physical camera."""

    raw = np.zeros(len(PARAMETER_NAMES), dtype=np.float64)
    if vector is not None:
        raw = np.asarray(vector, dtype=np.float64)
    scale = np.asarray(scales, dtype=np.float64)
    if raw.shape != (len(PARAMETER_NAMES),) or scale.shape != raw.shape:
        raise ValueError("Joint camera vector/scales must have eleven values")
    if not np.isfinite(raw).all() or not np.isfinite(scale).all():
        raise ValueError("Joint camera vector/scales must be finite")
    delta = raw * scale
    source_direction = _normalize(seed["analysis_direction"])
    source_up = _normalize(seed["analysis_up_axis"])
    azimuth, elevation = _angles(source_direction)
    exact = bool(np.allclose(delta, 0.0, atol=1e-12))
    if exact:
        # At a pole, azimuth is undefined.  Round-tripping an exact top-view
        # seed through spherical angles can introduce a small false rotation;
        # preserve the sealed camera bit-for-bit instead.
        direction = source_direction.tolist()
        up_axis = source_up.tolist()
    else:
        direction = _direction(
            azimuth + float(delta[0]),
            float(np.clip(elevation + float(delta[1]), -89.95, 89.95)),
        )
        transported_up = _transport_up_axis(source_direction, source_up, direction)
        up_axis = _roll_up_axis(direction, transported_up, float(delta[2]))
    distance = float(
        np.clip(
            _float(seed, "distance_multiplier", 2.15) * math.exp(float(delta[3])),
            1.05,
            100.0,
        )
    )
    focal = float(
        np.clip(
            _float(seed, "focal_length_mm", 45.0) * math.exp(float(delta[6])),
            12.0,
            MAX_FOCAL_LENGTH_MM,
        )
    )
    target_u = float(np.clip(_float(seed, "target_offset_u", 0.0) + delta[4], -0.5, 0.5))
    target_v = float(np.clip(_float(seed, "target_offset_v", 0.0) + delta[5], -0.5, 0.5))
    principal_u = float(np.clip(_float(seed, "principal_point_u", 0.0) + delta[7], -0.20, 0.20))
    principal_v = float(np.clip(_float(seed, "principal_point_v", 0.0) + delta[8], -0.20, 0.20))
    radial_k1 = float(np.clip(_float(seed, "radial_distortion_k1", 0.0) + delta[9], -0.35, 0.35))
    radial_k2 = float(np.clip(_float(seed, "radial_distortion_k2", 0.0) + delta[10], -0.20, 0.20))
    return {
        "view_id": candidate_id,
        "analysis_direction": direction,
        "analysis_up_axis": up_axis,
        "focal_length_mm": focal,
        "distance_multiplier": distance,
        "target_offset_u": target_u,
        "target_offset_v": target_v,
        # analysis_up_axis already contains the complete camera roll.
        "roll_degrees": 0.0,
        "principal_point_u": principal_u,
        "principal_point_v": principal_v,
        "radial_distortion_k1": radial_k1,
        "radial_distortion_k2": radial_k2,
        "projection_mode": str(seed.get("projection_mode", "perspective")),
        "orthographic_span_multiplier": _float(
            seed, "orthographic_span_multiplier", 2.0
        ),
        "calibration": {
            "reference_view_id": reference_id,
            "phase": f"pose_camera_joint_round_{round_index}",
            "frame_anchor": frame_anchor,
            "optimizer": OPTIMIZER_NAME,
            "round_index": round_index,
            "lineage": dict(lineage),
            "exact_start": exact,
            "joint_parameter_names": list(PARAMETER_NAMES),
            "normalized_joint_step": [round(float(value), 8) for value in raw],
            "physical_joint_delta": [round(float(value), 8) for value in delta],
            "whole_asset_se3_equivalent_only": True,
            "camera_intrinsics_jointly_optimized": True,
            "per_mesh_or_subtree_transform_applied": False,
        },
    }


def _baseline_anchor(
    baseline: Mapping[str, Any], *, reference_id: str, candidate_id: str, round_index: int
) -> dict[str, Any]:
    return joint_candidate_spec(
        seed=baseline,
        reference_id=reference_id,
        candidate_id=candidate_id,
        vector=None,
        scales=BASELINE_SCALES,
        round_index=round_index,
        lineage={"kind": "sealed_baseline", "rank": 0},
        frame_anchor=True,
    )


def _seed_from_score(score: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "analysis_direction": score["analysis_direction"],
        "analysis_up_axis": score["analysis_up_axis"],
        "focal_length_mm": score["focal_length_mm"],
        "distance_multiplier": score["distance_multiplier"],
        "target_offset_u": score.get("target_offset_u", 0.0),
        "target_offset_v": score.get("target_offset_v", 0.0),
        "principal_point_u": score.get("principal_point_u", 0.0),
        "principal_point_v": score.get("principal_point_v", 0.0),
        "radial_distortion_k1": score.get("radial_distortion_k1", 0.0),
        "radial_distortion_k2": score.get("radial_distortion_k2", 0.0),
        "projection_mode": score.get("projection_mode", "perspective"),
        "orthographic_span_multiplier": score.get(
            "orthographic_span_multiplier", 2.0
        ),
    }


def _lineage(score: Mapping[str, Any]) -> dict[str, Any]:
    calibration = score.get("calibration", {})
    value = calibration.get("lineage", {}) if isinstance(calibration, Mapping) else {}
    return dict(value) if isinstance(value, Mapping) else {"kind": "unknown"}


def _distinct_scores(
    scores: Sequence[Mapping[str, Any]], *, count: int
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for score in sorted(scores, key=_alignment_candidate_sort_key):
        signature = (
            tuple(round(float(value), 7) for value in score["analysis_direction"]),
            tuple(round(float(value), 7) for value in score["analysis_up_axis"]),
            round(float(score["focal_length_mm"]), 5),
            round(float(score["distance_multiplier"]), 5),
            round(float(score.get("target_offset_u", 0.0)), 5),
            round(float(score.get("target_offset_v", 0.0)), 5),
            round(float(score.get("principal_point_u", 0.0)), 5),
            round(float(score.get("principal_point_v", 0.0)), 5),
            round(float(score.get("radial_distortion_k1", 0.0)), 5),
            round(float(score.get("radial_distortion_k2", 0.0)), 5),
        )
        if signature in seen:
            continue
        seen.add(signature)
        output.append(dict(score))
        if len(output) >= count:
            break
    return output


def select_final_candidate(
    records: Sequence[Mapping[str, Any]],
    *,
    maximum_iou_regression: float = 0.002,
    maximum_boundary_regression_px: float = 0.5,
    minimum_score_improvement: float = 0.001,
    require_fixed_anchor: bool = False,
) -> tuple[str, dict[str, Any]]:
    baselines = [
        raw
        for raw in records
        if _lineage(raw).get("kind") == "sealed_baseline"
        and bool(raw.get("calibration", {}).get("frame_anchor"))
    ]
    if len(baselines) != 1:
        raise ValueError("Joint refinement requires exactly one sealed baseline anchor")
    baseline = dict(baselines[0])
    baseline_consensus = bool(baseline.get("rigid_consensus_valid"))
    baseline_anchor_score = float(baseline.get("fixed_anchor_score", 0.0))
    eligible = [
        dict(raw)
        for raw in records
        if raw is not baselines[0]
        and float(raw["projection_iou"])
        >= float(baseline["projection_iou"]) - maximum_iou_regression
        and float(raw["boundary_p95_px"])
        <= float(baseline["boundary_p95_px"]) + maximum_boundary_regression_px
        and float(raw["score"])
        >= float(baseline["score"]) + minimum_score_improvement
        and (not baseline_consensus or bool(raw.get("rigid_consensus_valid")))
        and (
            not require_fixed_anchor
            or (
                bool(raw.get("fixed_anchor_valid"))
                and float(raw.get("fixed_anchor_score", 0.0))
                >= baseline_anchor_score + minimum_score_improvement
            )
        )
    ]
    if not eligible:
        return "BASELINE_RETAINED", baseline
    return "JOINT_REFINEMENT_ACCEPTED", dict(
        min(eligible, key=_alignment_candidate_sort_key)
    )


def _score_round(
    *,
    args: argparse.Namespace,
    registry: Path,
    manifest: Path,
    reference_id: str,
    specs: Sequence[Mapping[str, Any]],
    output: Path,
    round_name: str,
    fixed_anchor_part_ids: Sequence[str] = (),
) -> tuple[list[dict[str, Any]], Path]:
    specs_path = _write_object(
        output / reference_id / f"{round_name}_view_specs.json",
        {"schema_version": VIEW_SPEC_SCHEMA_VERSION, "views": list(specs)},
    )
    rendered = _run_render(
        isaac_python=args.isaac_python.expanduser().resolve(strict=True),
        registry=registry,
        output_dir=output / reference_id / f"{round_name}_renders",
        view_specs=specs_path,
        resolution=args.search_resolution,
        rt_subframes=args.rt_subframes,
        analysis_up_axis=args.analysis_up_axis,
        analysis_front_axis=args.analysis_front_axis,
    )
    references = _reference_masks(manifest)
    mask, reference_row = references[reference_id]
    image = _reference_image(reference_row, manifest, mask.shape)
    _, records = _score_candidates(
        reference_id=reference_id,
        reference_mask=mask,
        reference_image=image,
        registry_path=rendered,
        fixed_anchor_part_ids=fixed_anchor_part_ids,
    )
    _write_object(
        output / reference_id / f"{round_name}_scores.json",
        {
            "schema_version": REPORT_SCHEMA_VERSION,
            "reference_view_id": reference_id,
            "round": round_name,
            "rendered_registry": str(rendered),
            "candidates": records,
        },
    )
    return records, rendered


def _screen_specs(
    *,
    reference_id: str,
    baseline: Mapping[str, Any],
    model_seeds: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    starts = [
        (
            baseline,
            BASELINE_SCALES,
            {"kind": "sealed_baseline", "rank": 0},
        ),
        *[
            (
                raw,
                MODEL_SCALES,
                {
                    "kind": "gigapose",
                    "rank": int(raw["calibration"]["proposal_rank"]),
                    "model_score": float(raw["calibration"]["pose_model_score"]),
                },
            )
            for raw in model_seeds
        ],
    ]
    output: list[dict[str, Any]] = []
    for start_index, (seed, scales, lineage) in enumerate(starts):
        output.append(
            joint_candidate_spec(
                seed=seed,
                reference_id=reference_id,
                candidate_id=f"joint_{reference_id}_r1_s{start_index:02d}_exact",
                vector=None,
                scales=scales,
                round_index=1,
                lineage=lineage,
                frame_anchor=start_index == 0,
            )
        )
        for sample_index, vector in enumerate(
            joint_directions(count=3, sequence_offset=17 * start_index), start=1
        ):
            output.append(
                joint_candidate_spec(
                    seed=seed,
                    reference_id=reference_id,
                    candidate_id=(
                        f"joint_{reference_id}_r1_s{start_index:02d}_{sample_index:02d}"
                    ),
                    vector=vector,
                    scales=scales,
                    round_index=1,
                    lineage=lineage,
                )
            )
    return output


def _branch_specs(
    *,
    reference_id: str,
    baseline: Mapping[str, Any],
    centers: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output = [
        _baseline_anchor(
            baseline,
            reference_id=reference_id,
            candidate_id=f"joint_{reference_id}_r2_baseline",
            round_index=2,
        )
    ]
    for center_index, center in enumerate(centers):
        lineage = _lineage(center)
        scales = (MODEL_SCALES if lineage.get("kind") == "gigapose" else BASELINE_SCALES) * 0.5
        seed = _seed_from_score(center)
        output.append(
            joint_candidate_spec(
                seed=seed,
                reference_id=reference_id,
                candidate_id=f"joint_{reference_id}_r2_b{center_index:02d}_exact",
                vector=None,
                scales=scales,
                round_index=2,
                lineage=lineage,
            )
        )
        for sample_index, vector in enumerate(
            joint_directions(count=10, sequence_offset=101 + 19 * center_index),
            start=1,
        ):
            output.append(
                joint_candidate_spec(
                    seed=seed,
                    reference_id=reference_id,
                    candidate_id=f"joint_{reference_id}_r2_b{center_index:02d}_{sample_index:02d}",
                    vector=vector,
                    scales=scales,
                    round_index=2,
                    lineage=lineage,
                )
            )
    return output


def _polish_specs(
    *,
    reference_id: str,
    baseline: Mapping[str, Any],
    center: Mapping[str, Any],
) -> list[dict[str, Any]]:
    lineage = _lineage(center)
    scales = (MODEL_SCALES if lineage.get("kind") == "gigapose" else BASELINE_SCALES) * 0.2
    seed = _seed_from_score(center)
    output = [
        _baseline_anchor(
            baseline,
            reference_id=reference_id,
            candidate_id=f"joint_{reference_id}_r3_baseline",
            round_index=3,
        ),
        joint_candidate_spec(
            seed=seed,
            reference_id=reference_id,
            candidate_id=f"joint_{reference_id}_r3_center",
            vector=None,
            scales=scales,
            round_index=3,
            lineage=lineage,
        ),
    ]
    for sample_index, vector in enumerate(
        joint_directions(count=16, sequence_offset=211), start=1
    ):
        output.append(
            joint_candidate_spec(
                seed=seed,
                reference_id=reference_id,
                candidate_id=f"joint_{reference_id}_r3_{sample_index:02d}",
                vector=vector,
                scales=scales,
                round_index=3,
                lineage=lineage,
            )
        )
    return output


def _interpolate_camera_seed(
    baseline: Mapping[str, Any],
    center: Mapping[str, Any],
    *,
    fraction: float,
) -> dict[str, Any]:
    """Interpolate one complete camera without changing any CAD transform."""

    if not 0.0 <= fraction <= 1.0:
        raise ValueError("Camera interpolation fraction must be in [0, 1]")
    baseline_direction = _normalize(baseline["analysis_direction"])
    center_direction = _normalize(center["analysis_direction"])
    direction = _normalize(
        (1.0 - fraction) * baseline_direction + fraction * center_direction
    )
    baseline_up = np.asarray(
        _transport_up_axis(
            baseline_direction,
            _normalize(baseline["analysis_up_axis"]),
            direction,
        ),
        dtype=np.float64,
    )
    center_up = np.asarray(
        _transport_up_axis(
            center_direction,
            _normalize(center["analysis_up_axis"]),
            direction,
        ),
        dtype=np.float64,
    )
    if float(np.dot(baseline_up, center_up)) < 0.0:
        center_up = -np.asarray(center_up, dtype=np.float64)
    up = _normalize((1.0 - fraction) * baseline_up + fraction * center_up)

    def linear(key: str, default: float = 0.0) -> float:
        return (1.0 - fraction) * _float(baseline, key, default) + fraction * _float(
            center, key, default
        )

    def geometric(key: str, default: float) -> float:
        first = _float(baseline, key, default)
        second = _float(center, key, default)
        return math.exp((1.0 - fraction) * math.log(first) + fraction * math.log(second))

    return {
        "analysis_direction": direction.tolist(),
        "analysis_up_axis": up.tolist(),
        "focal_length_mm": geometric("focal_length_mm", 45.0),
        "distance_multiplier": geometric("distance_multiplier", 2.15),
        "target_offset_u": linear("target_offset_u"),
        "target_offset_v": linear("target_offset_v"),
        "principal_point_u": linear("principal_point_u"),
        "principal_point_v": linear("principal_point_v"),
        "radial_distortion_k1": linear("radial_distortion_k1"),
        "radial_distortion_k2": linear("radial_distortion_k2"),
        "projection_mode": str(baseline.get("projection_mode", "perspective")),
        "orthographic_span_multiplier": linear(
            "orthographic_span_multiplier", 2.0
        ),
    }


def _anchor_line_center(
    scores: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    baselines = [
        dict(raw)
        for raw in scores
        if _lineage(raw).get("kind") == "sealed_baseline"
        and bool(raw.get("calibration", {}).get("frame_anchor"))
    ]
    if not baselines:
        raise ValueError("Anchor line search lost its sealed baseline")
    baseline = min(baselines, key=_alignment_candidate_sort_key)
    eligible = [
        dict(raw)
        for raw in scores
        if bool(raw.get("fixed_anchor_valid"))
        and float(raw["projection_iou"]) >= float(baseline["projection_iou"]) - 0.05
        and float(raw["boundary_p95_px"])
        <= float(baseline["boundary_p95_px"]) + 6.0
    ]
    center = min(
        eligible or [baseline],
        key=lambda raw: (
            -float(raw.get("fixed_anchor_score", 0.0)),
            -float(raw["projection_iou"]),
            float(raw["boundary_p95_px"]),
            str(raw["view_id"]),
        ),
    )
    return baseline, center


def _line_search_specs(
    *,
    reference_id: str,
    baseline: Mapping[str, Any],
    center: Mapping[str, Any],
) -> list[dict[str, Any]]:
    output = [
        _baseline_anchor(
            baseline,
            reference_id=reference_id,
            candidate_id=f"joint_{reference_id}_r4_baseline",
            round_index=4,
        )
    ]
    baseline_seed = dict(baseline)
    center_seed = _seed_from_score(center)
    for index, fraction in enumerate(
        (0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0), start=1
    ):
        seed = _interpolate_camera_seed(
            baseline_seed,
            center_seed,
            fraction=fraction,
        )
        output.append(
            joint_candidate_spec(
                seed=seed,
                reference_id=reference_id,
                candidate_id=f"joint_{reference_id}_r4_line_{index:02d}",
                vector=None,
                scales=BASELINE_SCALES * 0.1,
                round_index=4,
                lineage={
                    "kind": "sealed_anchor_line_search",
                    "source_view_id": str(center["view_id"]),
                    "fraction": fraction,
                },
            )
        )
    return output


def _candidate_public(score: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "view_id": score["view_id"],
        "lineage": _lineage(score),
        "projection_iou": score["projection_iou"],
        "boundary_p95_px": score["boundary_p95_px"],
        "score": score["score"],
        "rigid_consensus_valid": bool(score.get("rigid_consensus_valid")),
        "fixed_anchor_valid": bool(score.get("fixed_anchor_valid")),
        "fixed_anchor_score": score.get("fixed_anchor_score"),
        "fixed_anchor_residual_px": score.get("fixed_anchor_residual_px"),
        "fixed_anchor_coverage": score.get("fixed_anchor_coverage"),
        "focal_length_mm": score["focal_length_mm"],
        "distance_multiplier": score["distance_multiplier"],
        "principal_point_u": score.get("principal_point_u", 0.0),
        "principal_point_v": score.get("principal_point_v", 0.0),
        "radial_distortion_k1": score.get("radial_distortion_k1", 0.0),
        "radial_distortion_k2": score.get("radial_distortion_k2", 0.0),
    }


def seal_multiview_rigid_anchors(
    baseline_scores: Mapping[str, Mapping[str, Any]],
    *,
    minimum_observed_views: int = 2,
) -> dict[str, Any]:
    """Seal Parts accepted by every baseline view in which they are observable."""

    if minimum_observed_views < 2:
        raise ValueError("A multi-view rigid anchor needs at least two views")
    diagnosis = _classify_multiview_residuals(baseline_scores)
    stable: list[str] = []
    for row in diagnosis["part_diagnoses"]:
        if (
            row["classification"] == "rigid_consensus_inlier"
            and int(row["visible_view_count"]) >= minimum_observed_views
        ):
            stable.append(str(row["part_id"]))
    stable.sort()
    if len(stable) < 3:
        raise RuntimeError("Baseline views do not establish three stable rigid anchors")
    visible_by_view = {
        view_id: sorted(
            part_id
            for part_id in stable
            if any(
                raw.get("part_id") == part_id
                and bool(raw.get("consensus_observable", True))
                for raw in score.get("rigid_consensus_part_residuals", [])
            )
        )
        for view_id, score in baseline_scores.items()
    }
    if any(len(part_ids) < 3 for part_ids in visible_by_view.values()):
        raise RuntimeError("A baseline view has fewer than three visible rigid anchors")
    return {
        "schema_version": "qwen-sealed-multiview-rigid-anchors/v1",
        "selection_authority": "baseline_multiview_robust_consensus_intersection",
        "minimum_observed_views": minimum_observed_views,
        "part_ids": stable,
        "part_count": len(stable),
        "visible_part_ids_by_view": visible_by_view,
        "diagnosis": diagnosis,
        "per_mesh_or_subtree_transform_applied": False,
    }


def _baseline_scores_from_seed_render(
    *,
    manifest: Path,
    registry: Path,
    requested: Sequence[str],
) -> dict[str, dict[str, Any]]:
    references = _reference_masks(manifest)
    output: dict[str, dict[str, Any]] = {}
    for view_id in requested:
        mask, row = references[view_id]
        image = _reference_image(row, manifest, mask.shape)
        _, records = _score_candidates(
            reference_id=view_id,
            reference_mask=mask,
            reference_image=image,
            registry_path=registry,
        )
        baseline = [
            raw
            for raw in records
            if raw.get("calibration", {}).get("frame_anchor") is True
            and int(raw.get("calibration", {}).get("proposal_rank", 0)) == 0
        ]
        if len(baseline) != 1:
            raise ValueError(f"Seed render does not have one baseline for {view_id}")
        output[view_id] = dict(baseline[0])
    return output


def full_resolution_spec(
    score: Mapping[str, Any], *, view_id: str, mark_anchor: bool
) -> dict[str, Any]:
    """Promote a search score without leaking its pixel-resolution anchor."""

    spec = _spec_from_score(
        score,
        view_id=view_id,
        phase="pose_camera_joint_full_resolution",
        mark_phase_incumbent=mark_anchor,
    )
    # The affine is expressed in search-render pixels.  The full-resolution
    # baseline must establish a new shared affine in its own pixel grid.
    spec["calibration"].pop("frame_anchor_affine", None)
    spec["calibration"]["lineage"] = _lineage(score)
    spec["calibration"]["optimizer"] = OPTIMIZER_NAME
    spec["calibration"]["whole_asset_se3_equivalent_only"] = True
    spec["calibration"]["per_mesh_or_subtree_transform_applied"] = False
    return spec


def _viewer(*, output: Path, report: Mapping[str, Any]) -> None:
    assets = output / "viewer" / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    cards: list[str] = []
    figures: list[str] = []
    for raw in report["views"]:
        view_id = str(raw["view_id"])
        for source, name in (
            (Path(raw["baseline"]["rgb"]), f"{view_id}_before.png"),
            (Path(raw["selected"]["rgb"]), f"{view_id}_after.png"),
            (Path(raw["baseline"]["residual"]["residual_overlay"]), f"{view_id}_before_residual.png"),
            (Path(raw["selected"]["residual"]["residual_overlay"]), f"{view_id}_after_residual.png"),
        ):
            shutil.copy2(source, assets / name)
        before, after = raw["baseline"], raw["selected"]
        rows = "".join(
            "<tr>"
            f"<td>{html.escape(str(item['lineage'].get('kind')))}</td>"
            f"<td>{float(item['projection_iou']):.4f}</td>"
            f"<td>{float(item['boundary_p95_px']):.2f}</td>"
            f"<td>{float(item['score']):.4f}</td></tr>"
            for item in raw["full_resolution_finalists"]
        )
        cards.append(
            f"<article><h2>{html.escape(view_id)}</h2><b>{html.escape(raw['decision'])}</b>"
            f"<p>IoU {before['projection_iou']:.4f} → {after['projection_iou']:.4f}<br>"
            f"P95 {before['boundary_p95_px']:.2f} → {after['boundary_p95_px']:.2f}px<br>"
            f"目标 {before['score']:.4f} → {after['score']:.4f}</p>"
            "<table><thead><tr><th>起点</th><th>IoU</th><th>P95</th><th>目标</th></tr></thead>"
            f"<tbody>{rows}</tbody></table></article>"
        )
        figures.append(
            f"<section><h2>{html.escape(view_id)}</h2><div class='grid'>"
            f"<figure><img src='assets/{view_id}_before.png'><figcaption>原相机</figcaption></figure>"
            f"<figure><img src='assets/{view_id}_after.png'><figcaption>联合优化后</figcaption></figure>"
            f"<figure><img src='assets/{view_id}_before_residual.png'><figcaption>原残差</figcaption></figure>"
            f"<figure><img src='assets/{view_id}_after_residual.png'><figcaption>优化后残差</figcaption></figure>"
            "</div></section>"
        )
    page = f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>整件 SE(3) + 相机内参联合优化</title><style>
body{{margin:0;background:#0d1117;color:#f0f6fc;font:15px/1.55 system-ui}}main{{width:min(1500px,96vw);margin:auto;padding:28px 0}}a{{color:#58a6ff}}p,figcaption{{color:#9da7b3}}.cards,.grid{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}}article,figure{{margin:0;background:#161b22;border:1px solid #30363d;border-radius:10px;overflow:hidden}}article{{padding:14px}}figure img{{display:block;width:100%}}figcaption{{padding:8px}}b{{color:#56d364}}table{{width:100%;border-collapse:collapse;font-size:12px}}th,td{{padding:4px;border-bottom:1px solid #30363d;text-align:right}}th:first-child,td:first-child{{text-align:left}}@media(max-width:900px){{.cards,.grid{{grid-template-columns:1fr 1fr}}}}@media(max-width:560px){{.cards,.grid{{grid-template-columns:1fr}}}}</style></head><body><main>
<h1>原相机 + GigaPose Top-K · 整件 SE(3) 与内参联合优化</h1><p>每个候选同时扰动整件刚体等价的 6DoF 外参、焦距、主点和径向畸变。所有候选共享原相机帧锚点；没有单独移动任何 Mesh。红=照片独有，蓝=CAD独有，绿=重合。</p><div class='cards'>{''.join(cards)}</div>{''.join(figures)}<p><a href='pose_camera_joint_refinement_report.json'>完整 JSON</a> · <a href='optimized_view_specs.json'>优化相机参数</a></p></main></body></html>"""
    viewer = output / "viewer"
    _write_object(viewer / "pose_camera_joint_refinement_report.json", report)
    shutil.copy2(output / "optimized_view_specs.json", viewer / "optimized_view_specs.json")
    (viewer / "index.html").write_text(page, encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Joint refinement output already exists: {output}")
    output.mkdir(parents=True)
    seed_report_path = args.pose_seed_report.expanduser().resolve(strict=True)
    seed_report = _read_object(seed_report_path)
    if seed_report.get("schema_version") != SEED_REPORT_SCHEMA_VERSION:
        raise ValueError("Joint refinement requires a sealed pose-model seed report")
    if seed_report.get("whole_asset_only") is not True or seed_report.get(
        "per_mesh_or_subtree_transform_applied"
    ) is not False:
        raise ValueError("Pose-model seed report violates the rigid whole-asset contract")
    registry = _resolved_bound_path(
        seed_report_path, seed_report, "registry", "registry_sha256"
    )
    manifest = _resolved_bound_path(
        seed_report_path,
        seed_report,
        "reference_manifest",
        "reference_manifest_sha256",
    )
    baseline_path = _resolved_bound_path(
        seed_report_path,
        seed_report,
        "baseline_view_specs",
        "baseline_view_specs_sha256",
    )
    proposals_path = _resolved_bound_path(
        seed_report_path, seed_report, "proposals", "proposals_sha256"
    )
    seed_rendered = Path(str(seed_report.get("rendered_registry", ""))).expanduser().resolve(
        strict=True
    )
    baseline_by_view = _seed_by_view(_read_object(baseline_path))
    requested = [value.strip() for value in args.views.split(",") if value.strip()]
    if len(requested) != len(set(requested)) or set(requested) != set(baseline_by_view):
        raise ValueError("Requested views must exactly match the sealed baseline")
    proposals = validate_proposals(
        _read_object(proposals_path), expected_view_ids=requested
    )
    sealed_anchor = seal_multiview_rigid_anchors(
        _baseline_scores_from_seed_render(
            manifest=manifest,
            registry=seed_rendered,
            requested=requested,
        )
    )
    fixed_anchor_part_ids_by_view = dict(sealed_anchor["visible_part_ids_by_view"])
    _write_object(output / "sealed_multiview_rigid_anchors.json", sealed_anchor)
    model_starts = {
        view_id: [
            pose_to_camera_spec(
                baseline=baseline_by_view[view_id],
                view_id=view_id,
                candidate=candidate,
            )
            for candidate in proposals[view_id]
        ]
        for view_id in requested
    }
    search_records: dict[str, list[dict[str, Any]]] = {}
    round_audits: dict[str, list[dict[str, Any]]] = {}
    for view_index, view_id in enumerate(requested, start=1):
        print(f"[JOINT] view {view_index}/{len(requested)} {view_id}", flush=True)
        baseline = baseline_by_view[view_id]
        first, _ = _score_round(
            args=args,
            registry=registry,
            manifest=manifest,
            reference_id=view_id,
            specs=_screen_specs(
                reference_id=view_id,
                baseline=baseline,
                model_seeds=model_starts[view_id],
            ),
            output=output,
            round_name="round_1_multistart",
            fixed_anchor_part_ids=fixed_anchor_part_ids_by_view[view_id],
        )
        best_global = min(first, key=_alignment_candidate_sort_key)
        model_rows = [raw for raw in first if _lineage(raw).get("kind") == "gigapose"]
        best_model = min(model_rows, key=_alignment_candidate_sort_key)
        centers = _distinct_scores((best_global, best_model), count=2)
        second, _ = _score_round(
            args=args,
            registry=registry,
            manifest=manifest,
            reference_id=view_id,
            specs=_branch_specs(
                reference_id=view_id,
                baseline=baseline,
                centers=centers,
            ),
            output=output,
            round_name="round_2_branch_refine",
            fixed_anchor_part_ids=fixed_anchor_part_ids_by_view[view_id],
        )
        center = min((*first, *second), key=_alignment_candidate_sort_key)
        third, _ = _score_round(
            args=args,
            registry=registry,
            manifest=manifest,
            reference_id=view_id,
            specs=_polish_specs(
                reference_id=view_id,
                baseline=baseline,
                center=center,
            ),
            output=output,
            round_name="round_3_joint_polish",
            fixed_anchor_part_ids=fixed_anchor_part_ids_by_view[view_id],
        )
        line_baseline, line_center = _anchor_line_center((*first, *second, *third))
        fourth, _ = _score_round(
            args=args,
            registry=registry,
            manifest=manifest,
            reference_id=view_id,
            specs=_line_search_specs(
                reference_id=view_id,
                baseline=baseline,
                center=line_center,
            ),
            output=output,
            round_name="round_4_nonregressive_line_search",
            fixed_anchor_part_ids=fixed_anchor_part_ids_by_view[view_id],
        )
        search_records[view_id] = [*first, *second, *third, *fourth]
        round_audits[view_id] = [
            {
                "round": 1,
                "candidate_count": len(first),
                "winner": _candidate_public(best_global),
                "best_gigapose_lineage": _candidate_public(best_model),
            },
            {
                "round": 2,
                "candidate_count": len(second),
                "winner": _candidate_public(
                    min(second, key=_alignment_candidate_sort_key)
                ),
            },
            {
                "round": 3,
                "candidate_count": len(third),
                "winner": _candidate_public(
                    min(third, key=_alignment_candidate_sort_key)
                ),
            },
            {
                "round": 4,
                "candidate_count": len(fourth),
                "source_baseline": _candidate_public(line_baseline),
                "source_anchor_center": _candidate_public(line_center),
                "winner": _candidate_public(
                    min(fourth, key=_alignment_candidate_sort_key)
                ),
            },
        ]

    final_specs: list[dict[str, Any]] = []
    for view_id in requested:
        baseline = baseline_by_view[view_id]
        baseline_rows = [
            raw
            for raw in search_records[view_id]
            if _lineage(raw).get("kind") == "sealed_baseline"
            and bool(raw.get("calibration", {}).get("frame_anchor"))
        ]
        if not baseline_rows:
            raise RuntimeError(f"Search lost the baseline anchor for {view_id}")
        model_rows = [
            raw for raw in search_records[view_id] if _lineage(raw).get("kind") == "gigapose"
        ]
        required = [
            min(baseline_rows, key=_alignment_candidate_sort_key),
            min(model_rows, key=_alignment_candidate_sort_key),
        ]
        ranked_others = _distinct_scores(search_records[view_id], count=64)
        # Restore the baseline as the one explicit shared-frame anchor in the
        # full-resolution authority batch and force both the learned branch
        # and the baseline into it.  The remaining slots come from the global
        # objective; neither required start may be crowded out.
        ordered: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for raw in (*required, *ranked_others):
            source_id = str(raw["view_id"])
            if source_id in seen_ids:
                continue
            seen_ids.add(source_id)
            ordered.append(dict(raw))
            if len(ordered) >= 5:
                break
        for rank, score in enumerate(ordered):
            spec = full_resolution_spec(
                score,
                view_id=f"joint_final_{view_id}_{rank:02d}",
                mark_anchor=rank == 0,
            )
            final_specs.append(spec)
    final_specs_path = _write_object(
        output / "full_resolution_finalists.json",
        {"schema_version": VIEW_SPEC_SCHEMA_VERSION, "views": final_specs},
    )
    final_rendered = _run_render(
        isaac_python=args.isaac_python.expanduser().resolve(strict=True),
        registry=registry,
        output_dir=output / "full_resolution_finalists",
        view_specs=final_specs_path,
        resolution=args.final_resolution,
        rt_subframes=args.rt_subframes,
        analysis_up_axis=args.analysis_up_axis,
        analysis_front_axis=args.analysis_front_axis,
    )
    references = _reference_masks(manifest)
    selected_specs: list[dict[str, Any]] = []
    public_views: list[dict[str, Any]] = []
    rendered_doc = _read_object(final_rendered)
    for view_id in requested:
        mask, reference_row = references[view_id]
        image = _reference_image(reference_row, manifest, mask.shape)
        _, records = _score_candidates(
            reference_id=view_id,
            reference_mask=mask,
            reference_image=image,
            registry_path=final_rendered,
            fixed_anchor_part_ids=fixed_anchor_part_ids_by_view[view_id],
        )
        decision, winner = select_final_candidate(
            records,
            require_fixed_anchor=True,
        )
        baseline_score = next(
            raw
            for raw in records
            if _lineage(raw).get("kind") == "sealed_baseline"
            and bool(raw.get("calibration", {}).get("frame_anchor"))
        )
        selected_specs.append(
            _spec_from_score(
                winner,
                view_id=view_id,
                phase="pose_camera_joint_selected",
                bind_frame_anchor=True,
            )
        )

        def public_score(score: Mapping[str, Any], label: str) -> dict[str, Any]:
            view = _view_record(rendered_doc, str(score["view_id"]))
            rgb = str(Path(str(view["rgb"])).expanduser().resolve(strict=True))
            return {
                **_candidate_public(score),
                "rgb": rgb,
                "residual": _residual(
                    registry_path=final_rendered,
                    view_id=str(score["view_id"]),
                    reference_mask=mask,
                    score=score,
                    output_dir=output / "residuals" / view_id / label,
                ),
            }

        public_views.append(
            {
                "view_id": view_id,
                "decision": decision,
                "baseline": public_score(baseline_score, "baseline"),
                "selected": public_score(winner, "selected"),
                "search_rounds": round_audits[view_id],
                "search_candidate_count": len(search_records[view_id]),
                "full_resolution_finalists": [
                    _candidate_public(raw)
                    for raw in sorted(records, key=_alignment_candidate_sort_key)
                ],
                "whole_asset_se3_equivalent_only": True,
                "camera_intrinsics_jointly_optimized": True,
                "per_mesh_or_subtree_transform_applied": False,
            }
        )
    optimized_specs_path = _write_object(
        output / "optimized_view_specs.json",
        {"schema_version": VIEW_SPEC_SCHEMA_VERSION, "views": selected_specs},
    )
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "PASS",
        "optimizer": OPTIMIZER_NAME,
        "pose_seed_report": str(seed_report_path),
        "pose_seed_report_sha256": _sha256(seed_report_path),
        "registry": str(registry),
        "registry_sha256": _sha256(registry),
        "reference_manifest": str(manifest),
        "reference_manifest_sha256": _sha256(manifest),
        "baseline_view_specs": str(baseline_path),
        "baseline_view_specs_sha256": _sha256(baseline_path),
        "gigapose_proposals": str(proposals_path),
        "gigapose_proposals_sha256": _sha256(proposals_path),
        "optimized_view_specs": str(optimized_specs_path),
        "rendered_registry": str(final_rendered),
        "search_resolution": args.search_resolution,
        "final_resolution": args.final_resolution,
        "joint_parameters": list(PARAMETER_NAMES),
        "whole_asset_only": True,
        "whole_asset_se3_implemented_as_inverse_camera_extrinsics": True,
        "per_mesh_or_subtree_transform_applied": False,
        "camera_intrinsics_jointly_optimized": True,
        "shared_baseline_frame_anchor_per_view": True,
        "sealed_multiview_rigid_anchors": sealed_anchor,
        "candidate_wise_anchor_reselection": False,
        "learned_pose_is_initialization_only": True,
        "isaac_full_resolution_render_is_selection_authority": True,
        "views": public_views,
    }
    _write_object(output / "pose_camera_joint_refinement_report.json", report)
    _viewer(output=output, report=report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose-seed-report", type=Path, required=True)
    parser.add_argument("--isaac-python", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--views", default="front,side,top,iso")
    parser.add_argument("--search-resolution", type=int, default=256)
    parser.add_argument("--final-resolution", type=int, default=512)
    parser.add_argument("--rt-subframes", type=int, default=4)
    parser.add_argument("--analysis-up-axis", default="z")
    parser.add_argument("--analysis-front-axis", default="-y")
    return parser.parse_args()


def main() -> int:
    report = run(parse_args())
    print(
        json.dumps(
            {
                "status": report["status"],
                "views": [
                    {
                        "view_id": raw["view_id"],
                        "decision": raw["decision"],
                        "before": raw["baseline"]["projection_iou"],
                        "after": raw["selected"]["projection_iou"],
                    }
                    for raw in report["views"]
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
