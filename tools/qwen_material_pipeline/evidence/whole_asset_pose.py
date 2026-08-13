#!/usr/bin/env python3
"""Optimize one complete CAD workpiece SE(3) pose across sealed cameras."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from qwen_material_pipeline.evidence.camera_calibration import (
    _read_object,
    _reference_image,
    _reference_masks,
)
from qwen_material_pipeline.scripts.evaluate_multiview_assembly_pose import (
    _camera_contract,
    _render,
    _rgb,
    _score_view,
    _viewer,
    _write_object,
)


REPORT_SCHEMA_VERSION = "qwen-whole-asset-multiview-pose-optimization/v1"
POSE_SCHEMA_VERSION = "qwen-whole-asset-pose-override/v1"


def registered_asset_root_path(registry: Mapping[str, Any]) -> str:
    """Derive the lowest common authored path for all registered Mesh Parts."""

    paths = [
        str(raw["prim_path"])
        for raw in registry.get("parts", [])
        if isinstance(raw, Mapping) and isinstance(raw.get("prim_path"), str)
    ]
    if not paths or len(paths) != len(registry.get("parts", [])):
        raise ValueError("Whole-asset pose requires every registered Part path")
    if len(set(paths)) != len(paths):
        raise ValueError("Whole-asset pose requires unique registered Part paths")
    components = [raw.strip("/").split("/") for raw in paths]
    length = 0
    while all(
        len(raw) > length and raw[length] == components[0][length]
        for raw in components
    ):
        length += 1
    if length <= 0:
        raise ValueError("Registered Parts do not share one asset root")
    root = "/" + "/".join(components[0][:length])
    prefix = root.rstrip("/") + "/"
    if any(not raw.startswith(prefix) for raw in paths):
        raise ValueError("Registered Part coverage is not exact at the asset root")
    return root


def registry_bounds(registry: Mapping[str, Any]) -> dict[str, Any]:
    minima: list[np.ndarray] = []
    maxima: list[np.ndarray] = []
    for raw in registry.get("parts", []):
        if not isinstance(raw, Mapping):
            raise ValueError("Registry Part must be an object")
        bounds = np.asarray(raw.get("world_bbox"), dtype=np.float64)
        if bounds.shape != (2, 3) or not np.isfinite(bounds).all():
            raise ValueError("Whole-asset pose requires finite Part world bounds")
        minima.append(bounds[0])
        maxima.append(bounds[1])
    minimum = np.min(np.stack(minima), axis=0)
    maximum = np.max(np.stack(maxima), axis=0)
    center = 0.5 * (minimum + maximum)
    diagonal = float(np.linalg.norm(maximum - minimum))
    if not math.isfinite(diagonal) or diagonal <= 0.0:
        raise ValueError("Whole-asset pose requires non-degenerate asset bounds")
    return {
        "minimum": minimum.tolist(),
        "maximum": maximum.tolist(),
        "center": center.tolist(),
        "diagonal": diagonal,
    }


def phase_one_pose_candidates(
    *, asset_root: str, asset_diagonal: float
) -> list[dict[str, Any]]:
    translation_step = 0.01 * asset_diagonal
    rotation_step = 1.5
    output: list[dict[str, Any]] = []
    for axis in range(3):
        for sign in (-1.0, 1.0):
            translation = [0.0, 0.0, 0.0]
            translation[axis] = sign * translation_step
            output.append(
                {
                    "candidate_id": f"translate_{'xyz'[axis]}_{'minus' if sign < 0 else 'plus'}",
                    "asset_root_prim_path": asset_root,
                    "world_translation": translation,
                    "world_rotation_rotvec_degrees": [0.0, 0.0, 0.0],
                    "pivot": "asset_bounds_center",
                }
            )
    for axis in range(3):
        for sign in (-1.0, 1.0):
            rotation = [0.0, 0.0, 0.0]
            rotation[axis] = sign * rotation_step
            output.append(
                {
                    "candidate_id": f"rotate_{'xyz'[axis]}_{'minus' if sign < 0 else 'plus'}",
                    "asset_root_prim_path": asset_root,
                    "world_translation": [0.0, 0.0, 0.0],
                    "world_rotation_rotvec_degrees": rotation,
                    "pivot": "asset_bounds_center",
                }
            )
    return output


def _pose_document(raw: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": POSE_SCHEMA_VERSION,
        "asset_root_prim_path": raw["asset_root_prim_path"],
        "world_translation": raw["world_translation"],
        "world_rotation_rotvec_degrees": raw[
            "world_rotation_rotvec_degrees"
        ],
        "pivot": "asset_bounds_center",
    }


def _score_registry(
    *,
    view_ids: Sequence[str],
    registry_path: Path,
    manifest_path: Path,
    output: Path,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    references = _reference_masks(manifest_path)
    rows: list[dict[str, Any]] = []
    for view_id in view_ids:
        mask, reference_row = references[view_id]
        reference_image = _reference_image(reference_row, manifest_path, mask.shape)
        score, residual = _score_view(
            view_id=view_id,
            registry_path=registry_path,
            reference_mask=mask,
            reference_image=reference_image,
            output=output / view_id,
        )
        rows.append(
            {
                "view_id": view_id,
                "projection_iou": float(score["projection_iou"]),
                "boundary_p95_px": float(score["boundary_p95_px"]),
                "mismatch_over_union": float(residual["mismatch_over_union"]),
                "rgb": str(_rgb(registry_path, view_id)),
                "residual": residual,
            }
        )
    return rows, {
        "mean_projection_iou": float(np.mean([raw["projection_iou"] for raw in rows])),
        "mean_boundary_p95_px": float(
            np.mean([raw["boundary_p95_px"] for raw in rows])
        ),
        "mean_mismatch_over_union": float(
            np.mean([raw["mismatch_over_union"] for raw in rows])
        ),
        "minimum_projection_iou": float(
            min(raw["projection_iou"] for raw in rows)
        ),
    }


def evaluate_candidate(
    *,
    candidate: Mapping[str, Any],
    baseline_views: Sequence[Mapping[str, Any]],
    view_ids: Sequence[str],
    registry_path: Path,
    manifest_path: Path,
    output: Path,
    python_sh: Path,
    repository_root: Path,
    view_specs: Path,
    resolution: int,
    rt_subframes: int,
) -> dict[str, Any]:
    pose_path = output / "whole_asset_pose.json"
    _write_object(pose_path, _pose_document(candidate))
    rendered_registry = _render(
        python_sh=python_sh,
        repository_root=repository_root,
        source_registry=registry_path,
        view_specs=view_specs,
        output=output / "renders",
        resolution=resolution,
        rt_subframes=rt_subframes,
        override=None,
        whole_asset_pose=pose_path,
    )
    views, aggregate = _score_registry(
        view_ids=view_ids,
        registry_path=rendered_registry,
        manifest_path=manifest_path,
        output=output / "residuals",
    )
    baseline_by_id = {str(raw["view_id"]): raw for raw in baseline_views}
    regressions = [
        float(baseline_by_id[raw["view_id"]]["projection_iou"])
        - float(raw["projection_iou"])
        for raw in views
    ]
    return {
        "candidate_id": candidate["candidate_id"],
        "pose": _pose_document(candidate),
        "pose_path": str(pose_path),
        "rendered_registry": str(rendered_registry),
        "views": views,
        "aggregate": aggregate,
        "worst_view_iou_regression": max(0.0, max(regressions)),
        "all_registered_parts_share_one_transform": True,
        "local_part_or_subtree_override_applied": False,
    }


def select_whole_asset_pose(
    *, baseline: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]]
) -> tuple[str, Mapping[str, Any] | None]:
    eligible = [
        raw
        for raw in candidates
        if float(raw["worst_view_iou_regression"]) <= 0.003
        and float(raw["aggregate"]["mean_projection_iou"])
        >= float(baseline["aggregate"]["mean_projection_iou"]) + 0.001
        and float(raw["aggregate"]["mean_mismatch_over_union"])
        <= float(baseline["aggregate"]["mean_mismatch_over_union"]) - 0.001
    ]
    if not eligible:
        return "NO_OP", None
    eligible.sort(
        key=lambda raw: (
            -float(raw["aggregate"]["mean_projection_iou"]),
            float(raw["aggregate"]["mean_mismatch_over_union"]),
            float(raw["aggregate"]["mean_boundary_p95_px"]),
            str(raw["candidate_id"]),
        )
    )
    return "OPTIMIZED", eligible[0]


def optimize(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Whole-asset pose output already exists: {output}")
    output.mkdir(parents=True)
    report_path = args.camera_report.resolve(strict=True)
    report = _read_object(report_path)
    view_ids = [raw.strip() for raw in args.views.split(",") if raw.strip()]
    if len(view_ids) < 2 or len(set(view_ids)) != len(view_ids):
        raise ValueError("Whole-asset pose requires unique multiple reference views")
    source_registry_path = Path(str(report["source_registry"])).resolve(strict=True)
    source_registry = _read_object(source_registry_path)
    camera_reports = {view_id: report_path for view_id in view_ids}
    camera_provenance, manifest_path, view_specs = _camera_contract(
        camera_reports=camera_reports,
        source_registry=source_registry,
    )
    view_specs_path = output / "multiview_view_specs.json"
    _write_object(
        view_specs_path,
        {"schema_version": "qwen-camera-view-specs/v1", "views": view_specs},
    )
    asset_root = registered_asset_root_path(source_registry)
    bounds = registry_bounds(source_registry)
    baseline_registry = _render(
        python_sh=args.python_sh.resolve(strict=True),
        repository_root=args.repository_root.resolve(strict=True),
        source_registry=source_registry_path,
        view_specs=view_specs_path,
        output=output / "baseline" / "renders",
        resolution=args.resolution,
        rt_subframes=args.rt_subframes,
        override=None,
        whole_asset_pose=None,
    )
    baseline_views, baseline_aggregate = _score_registry(
        view_ids=view_ids,
        registry_path=baseline_registry,
        manifest_path=manifest_path,
        output=output / "baseline" / "residuals",
    )
    baseline = {
        "rendered_registry": str(baseline_registry),
        "views": baseline_views,
        "aggregate": baseline_aggregate,
    }
    candidates = []
    for index, raw in enumerate(
        phase_one_pose_candidates(
            asset_root=asset_root, asset_diagonal=float(bounds["diagonal"])
        ),
        start=1,
    ):
        print(
            f"[WHOLE_ASSET_POSE] candidate {index}/12 {raw['candidate_id']}",
            flush=True,
        )
        candidates.append(
            evaluate_candidate(
                candidate=raw,
                baseline_views=baseline_views,
                view_ids=view_ids,
                registry_path=source_registry_path,
                manifest_path=manifest_path,
                output=output / "candidates" / str(raw["candidate_id"]),
                python_sh=args.python_sh.resolve(strict=True),
                repository_root=args.repository_root.resolve(strict=True),
                view_specs=view_specs_path,
                resolution=args.resolution,
                rt_subframes=args.rt_subframes,
            )
        )
    decision, winner = select_whole_asset_pose(
        baseline=baseline, candidates=candidates
    )
    if winner is None:
        winner_registry = baseline_registry
        winner_views = baseline_views
        winner_aggregate = baseline_aggregate
        winning_pose = {
            "schema_version": POSE_SCHEMA_VERSION,
            "asset_root_prim_path": asset_root,
            "world_translation": [0.0, 0.0, 0.0],
            "world_rotation_rotvec_degrees": [0.0, 0.0, 0.0],
            "pivot": "asset_bounds_center",
        }
    else:
        winner_registry = Path(str(winner["rendered_registry"])).resolve(strict=True)
        winner_views = list(winner["views"])
        winner_aggregate = dict(winner["aggregate"])
        winning_pose = dict(winner["pose"])
    winning_pose_path = output / "whole_asset_pose.json"
    _write_object(winning_pose_path, winning_pose)
    before_by_id = {str(raw["view_id"]): raw for raw in baseline_views}
    after_by_id = {str(raw["view_id"]): raw for raw in winner_views}
    public_views = []
    for view_id in view_ids:
        before = before_by_id[view_id]
        after = after_by_id[view_id]
        public_views.append(
            {
                "view_id": view_id,
                "source_decision": decision,
                "camera": camera_provenance[view_id],
                "before": before,
                "after": after,
                "delta": {
                    "projection_iou": round(
                        after["projection_iou"] - before["projection_iou"], 8
                    ),
                    "boundary_p95_px": round(
                        after["boundary_p95_px"] - before["boundary_p95_px"], 8
                    ),
                    "mismatch_over_union": round(
                        after["mismatch_over_union"]
                        - before["mismatch_over_union"],
                        8,
                    ),
                },
            }
        )
    result = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "PASS",
        "decision": decision,
        "camera_report": str(report_path),
        "source_registry": str(source_registry_path),
        "source_asset": str(source_registry["asset_usd"]),
        "reference_manifest": str(manifest_path),
        "asset_root_prim_path": asset_root,
        "registered_part_count": len(source_registry["parts"]),
        "registered_part_coverage": 1.0,
        "asset_bounds": bounds,
        "local_part_or_subtree_override_applied": False,
        "whole_asset_translation_and_rotation_only": True,
        "candidate_count": len(candidates),
        "baseline": baseline,
        "candidates": candidates,
        "winner": {
            "candidate_id": winner["candidate_id"] if winner is not None else "identity",
            "pose": winning_pose,
            "rendered_registry": str(winner_registry),
            "aggregate": winner_aggregate,
        },
        "views": public_views,
        "gates": {
            "minimum_mean_iou_gain": 0.001,
            "minimum_mean_mismatch_gain": 0.001,
            "maximum_per_view_iou_regression": 0.003,
            "all_views_share_one_asset_pose": True,
            "all_registered_parts_share_one_transform": True,
        },
    }
    _write_object(output / "whole_asset_pose_report.json", result)
    # Reuse the four-view presentation contract while exposing the new pose file.
    presentation = {
        **result,
        "status": "PASS" if decision == "OPTIMIZED" else "PASS_NO_OP",
        "primary_optimization_view": "all_views_jointly",
        "assembly_pose_overrides": str(winning_pose_path),
        "gates": {
            "primary_iou_gain": winner_aggregate["mean_projection_iou"]
            - baseline_aggregate["mean_projection_iou"],
            "primary_mismatch_gain": baseline_aggregate["mean_mismatch_over_union"]
            - winner_aggregate["mean_mismatch_over_union"],
            "worst_secondary_iou_regression": max(
                0.0,
                max(-float(raw["delta"]["projection_iou"]) for raw in public_views),
            ),
            "maximum_allowed_secondary_iou_regression": 0.003,
        },
    }
    _write_object(output / "multiview_assembly_pose_report.json", presentation)
    shutil.copy2(winning_pose_path, output / "assembly_pose_overrides.json")
    _viewer(output=output, report=presentation)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--views", default="front,side,top,iso")
    parser.add_argument("--python-sh", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--rt-subframes", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    result = optimize(parse_args())
    print(
        json.dumps(
            {
                "status": result["status"],
                "decision": result["decision"],
                "winner": result["winner"],
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
