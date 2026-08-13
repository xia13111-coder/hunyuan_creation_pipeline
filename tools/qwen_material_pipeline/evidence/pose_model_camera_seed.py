#!/usr/bin/env python3
"""Validate 6D pose-model proposals as rigid whole-asset camera seeds.

The learned model is deliberately an initializer, never a camera authority.
Every proposed object-to-camera pose is converted to one physical camera, the
unchanged USD asset is rendered once, and the existing complete-object
alignment objective decides whether that proposal is allowed to replace the
sealed baseline.  No Mesh or assembly subtree transform is authored.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from qwen_material_pipeline.evidence.camera_calibration import (
    _alignment_candidate_sort_key,
    _part_residual_attribution,
    _read_object,
    _reference_image,
    _reference_masks,
    _run_render,
    _score_candidates,
    _write_residual_audit,
    _write_object,
)


PROPOSAL_SCHEMA_VERSION = "qwen-rigid-pose-model-proposals/v1"
REPORT_SCHEMA_VERSION = "qwen-rigid-pose-model-camera-seed/v1"
VIEW_SPEC_SCHEMA_VERSION = "qwen-camera-view-specs/v1"
MODEL_NAME = "GigaPose"
MODEL_REVISION = "17fcf97f493f79e56a215ab10ebff16d95cfe34b"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unit_vector(value: object, *, label: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError(f"{label} must contain three finite numbers")
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-8:
        raise ValueError(f"{label} must be non-zero")
    return vector / norm


def _rotation(value: object, *, label: str) -> np.ndarray:
    rotation = np.asarray(value, dtype=np.float64)
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise ValueError(f"{label} must be a finite 3x3 matrix")
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=2e-4):
        raise ValueError(f"{label} is not orthonormal")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=2e-4):
        raise ValueError(f"{label} must be a proper rotation")
    return rotation


def _seed_by_view(document: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    if document.get("schema_version") != VIEW_SPEC_SCHEMA_VERSION:
        raise ValueError("Baseline camera specs have an unsupported schema")
    output: dict[str, dict[str, Any]] = {}
    for raw in document.get("views", []):
        if not isinstance(raw, Mapping) or not isinstance(raw.get("view_id"), str):
            raise ValueError("Baseline camera view must have a string view_id")
        view_id = str(raw["view_id"])
        if view_id in output:
            raise ValueError(f"Duplicate baseline camera view: {view_id}")
        output[view_id] = dict(raw)
    if not output:
        raise ValueError("Baseline camera specs contain no views")
    return output


def validate_proposals(
    document: Mapping[str, Any], *, expected_view_ids: Sequence[str]
) -> dict[str, list[dict[str, Any]]]:
    """Return exact Top-K rigid poses per view after fail-closed validation."""

    if document.get("schema_version") != PROPOSAL_SCHEMA_VERSION:
        raise ValueError("Pose-model proposal report has an unsupported schema")
    model = document.get("model")
    if not isinstance(model, Mapping) or model.get("name") != MODEL_NAME:
        raise ValueError("Pose-model proposal report is not from GigaPose")
    if model.get("repository_revision") != MODEL_REVISION:
        raise ValueError("GigaPose proposal report has an unsealed code revision")
    raw_views = document.get("views")
    if not isinstance(raw_views, list):
        raise ValueError("Pose-model proposal report requires a views array")
    expected = list(expected_view_ids)
    output: dict[str, list[dict[str, Any]]] = {}
    for raw_view in raw_views:
        if not isinstance(raw_view, Mapping) or not isinstance(
            raw_view.get("view_id"), str
        ):
            raise ValueError("Pose-model view must have a string view_id")
        view_id = str(raw_view["view_id"])
        if view_id in output:
            raise ValueError(f"Duplicate pose-model view: {view_id}")
        raw_candidates = raw_view.get("candidates")
        if not isinstance(raw_candidates, list) or not raw_candidates:
            raise ValueError(f"Pose-model view {view_id} has no candidates")
        candidates: list[dict[str, Any]] = []
        seen_ranks: set[int] = set()
        for raw in raw_candidates:
            if not isinstance(raw, Mapping):
                raise ValueError(f"Pose-model view {view_id} candidate is not an object")
            rank = raw.get("rank")
            if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
                raise ValueError(f"Pose-model view {view_id} has an invalid rank")
            if rank in seen_ranks:
                raise ValueError(f"Pose-model view {view_id} has duplicate rank {rank}")
            seen_ranks.add(rank)
            score = raw.get("model_score")
            if (
                isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not math.isfinite(float(score))
            ):
                raise ValueError(f"Pose-model view {view_id} has an invalid score")
            rotation = _rotation(
                raw.get("object_to_camera_rotation"),
                label=f"{view_id} rank {rank} rotation",
            )
            translation = np.asarray(
                raw.get("object_to_camera_translation"), dtype=np.float64
            )
            if (
                translation.shape != (3,)
                or not np.isfinite(translation).all()
                or float(translation[2]) <= 0.0
            ):
                raise ValueError(
                    f"Pose-model view {view_id} rank {rank} translation is invalid"
                )
            candidates.append(
                {
                    "rank": rank,
                    "model_score": float(score),
                    "object_to_camera_rotation": rotation.tolist(),
                    "object_to_camera_translation": translation.tolist(),
                    "template_view_id": raw.get("template_view_id"),
                    "inlier_fraction": raw.get("inlier_fraction"),
                }
            )
        candidates.sort(key=lambda item: int(item["rank"]))
        output[view_id] = candidates
    if set(output) != set(expected):
        raise ValueError(
            "Pose-model views do not exactly match baseline cameras: "
            f"expected={sorted(expected)} actual={sorted(output)}"
        )
    return output


def pose_to_camera_spec(
    *,
    baseline: Mapping[str, Any],
    view_id: str,
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    """Convert one OpenCV object-to-camera pose to an analysis-space camera.

    Inverting one rigid object pose into one camera pose is exactly equivalent
    for rendering.  Keeping the source USD unchanged prevents the common bug
    where individual Mesh prims are moved independently.
    """

    rotation = _rotation(
        candidate["object_to_camera_rotation"], label=f"{view_id} rotation"
    )
    translation = np.asarray(
        candidate["object_to_camera_translation"], dtype=np.float64
    )
    camera_center = -(rotation.T @ translation)
    direction = _unit_vector(camera_center, label=f"{view_id} camera center")
    # OpenCV +Y points down; the physical camera up vector is therefore -Y.
    up = _unit_vector(rotation.T @ np.asarray((0.0, -1.0, 0.0)), label=f"{view_id} up")
    up = _unit_vector(up - direction * float(np.dot(up, direction)), label=f"{view_id} up")
    calibration = {
        "reference_view_id": view_id,
        "phase": "pose_model_proposal",
        "proposal_rank": int(candidate["rank"]),
        "pose_model": MODEL_NAME,
        "pose_model_score": float(candidate["model_score"]),
        "frame_anchor": False,
    }
    return {
        **{
            key: value
            for key, value in baseline.items()
            if key not in {"view_id", "analysis_direction", "analysis_up_axis", "calibration"}
        },
        "view_id": f"pose_{view_id}_{int(candidate['rank']):02d}",
        "analysis_direction": direction.tolist(),
        "analysis_up_axis": up.tolist(),
        # The learned intrinsics/translation are intentionally advisory because
        # these photographs have no trusted EXIF calibration.  Existing focal,
        # distance and lens terms remain the physical scale seed and Isaac
        # refines them after the proposal gate.
        "calibration": calibration,
    }


def baseline_camera_spec(*, baseline: Mapping[str, Any], view_id: str) -> dict[str, Any]:
    return {
        **dict(baseline),
        "view_id": f"pose_{view_id}_00_baseline",
        "calibration": {
            "reference_view_id": view_id,
            "phase": "pose_model_proposal",
            "proposal_rank": 0,
            "pose_model": "sealed_baseline",
            "pose_model_score": 0.0,
            "frame_anchor": True,
        },
    }


def select_verified_seed(
    records: Sequence[Mapping[str, Any]], *, maximum_iou_regression: float = 0.002
) -> tuple[str, dict[str, Any]]:
    baselines = [
        raw
        for raw in records
        if raw.get("calibration", {}).get("proposal_rank") == 0
    ]
    if len(baselines) != 1:
        raise ValueError("Pose-model verification requires exactly one baseline")
    baseline = baselines[0]
    trusted = [
        raw
        for raw in records
        if int(raw.get("calibration", {}).get("proposal_rank", -1)) > 0
        and float(raw["projection_iou"])
        >= float(baseline["projection_iou"]) - maximum_iou_regression
        and float(raw["boundary_p95_px"])
        <= float(baseline["boundary_p95_px"]) + 1.0
        and float(raw["score"]) > float(baseline["score"])
    ]
    if not trusted:
        return "BASELINE_RETAINED", dict(baseline)
    return "POSE_MODEL_ACCEPTED", dict(min(trusted, key=_alignment_candidate_sort_key))


def _view_record(registry: Mapping[str, Any], view_id: str) -> Mapping[str, Any]:
    rows = [
        raw
        for raw in registry.get("render_set", {}).get("views", [])
        if isinstance(raw, Mapping) and raw.get("view_id") == view_id
    ]
    if len(rows) != 1:
        raise ValueError(f"Rendered registry has no unique view {view_id}")
    return rows[0]


def _residual(
    *,
    registry_path: Path,
    view_id: str,
    reference_mask: np.ndarray,
    score: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    from qwen_material_pipeline.evidence.camera_calibration import _foreground_for_view

    foreground = _foreground_for_view(registry_path, view_id)
    return _write_residual_audit(
        reference_id=view_id,
        reference_mask=reference_mask,
        foreground=foreground,
        score=score,
        part_residuals=_part_residual_attribution(
            registry_path=registry_path,
            view_id=view_id,
            reference_mask=reference_mask,
            score=score,
        ),
        output_dir=output_dir,
    )


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
        candidate_rows = "".join(
            "<tr>"
            f"<td>{item['rank']}</td>"
            f"<td>{float(item['projection_iou']):.4f}</td>"
            f"<td>{float(item['boundary_p95_px']):.2f}</td>"
            f"<td>{float(item['score']):.4f}</td>"
            f"<td>{'采用' if item['accepted'] else '拒绝'}</td>"
            "</tr>"
            for item in raw.get("candidate_scores", [])
        )
        cards.append(
            f"<article><h2>{html.escape(view_id)}</h2><b>{html.escape(raw['decision'])}</b>"
            f"<p>IoU {before['projection_iou']:.4f} → {after['projection_iou']:.4f}<br>"
            f"Boundary P95 {before['boundary_p95_px']:.2f} → {after['boundary_p95_px']:.2f}px<br>"
            f"模型 rank: {raw['selected_rank']}</p><table><thead><tr><th>rank</th><th>IoU</th><th>P95</th><th>目标</th><th>门禁</th></tr></thead>"
            f"<tbody>{candidate_rows}</tbody></table></article>"
        )
        figures.append(
            f"<section><h2>{html.escape(view_id)}</h2><div class='grid'>"
            f"<figure><img src='assets/{view_id}_before.png'><figcaption>原相机</figcaption></figure>"
            f"<figure><img src='assets/{view_id}_after.png'><figcaption>GigaPose 提议 + 真渲染门禁</figcaption></figure>"
            f"<figure><img src='assets/{view_id}_before_residual.png'><figcaption>原残差</figcaption></figure>"
            f"<figure><img src='assets/{view_id}_after_residual.png'><figcaption>新残差</figcaption></figure>"
            "</div></section>"
        )
    page = f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>GigaPose 四视角整件位姿</title><style>
body{{margin:0;background:#0d1117;color:#f0f6fc;font:15px/1.55 system-ui}}main{{width:min(1500px,96vw);margin:auto;padding:28px 0}}a{{color:#58a6ff}}p,figcaption{{color:#9da7b3}}.cards,.grid{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}}article,figure{{margin:0;background:#161b22;border:1px solid #30363d;border-radius:10px;overflow:hidden}}article{{padding:14px}}figure img{{display:block;width:100%}}figcaption{{padding:8px}}b{{color:#56d364}}table{{width:100%;border-collapse:collapse;font-size:12px}}th,td{{padding:4px;border-bottom:1px solid #30363d;text-align:right}}th:first-child,td:first-child{{text-align:left}}@media(max-width:900px){{.cards,.grid{{grid-template-columns:1fr 1fr}}}}@media(max-width:560px){{.cards,.grid{{grid-template-columns:1fr}}}}</style></head><body><main>
<h1>GigaPose Top-K · 四视角整件刚体相机种子</h1><p>每张照片允许独立相机，但 CAD 内部始终是同一个完整工件；没有移动单个 Mesh。模型只提初值，最终选择来自 Isaac 真渲染与现有刚体对齐目标；不改善即回退原相机。红=照片独有，蓝=CAD独有，绿=重合。</p>
<div class='cards'>{''.join(cards)}</div>{''.join(figures)}<p><a href='pose_model_camera_seed_report.json'>完整 JSON</a> · <a href='initial_view_specs.json'>下一阶段相机种子</a></p></main></body></html>"""
    _write_object(output / "viewer" / "pose_model_camera_seed_report.json", report)
    shutil.copy2(output / "initial_view_specs.json", output / "viewer" / "initial_view_specs.json")
    (output / "viewer" / "index.html").write_text(page, encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Pose-model seed output already exists: {output}")
    output.mkdir(parents=True)
    registry_path = args.registry.expanduser().resolve(strict=True)
    manifest_path = args.reference_manifest.expanduser().resolve(strict=True)
    baseline_path = args.baseline_view_specs.expanduser().resolve(strict=True)
    proposal_path = args.proposals.expanduser().resolve(strict=True)
    baseline_by_view = _seed_by_view(_read_object(baseline_path))
    requested = [value.strip() for value in args.views.split(",") if value.strip()]
    if set(requested) != set(baseline_by_view):
        raise ValueError("Requested views must exactly match baseline camera specs")
    proposals = validate_proposals(
        _read_object(proposal_path), expected_view_ids=requested
    )
    specs: list[dict[str, Any]] = []
    for view_id in requested:
        specs.append(baseline_camera_spec(baseline=baseline_by_view[view_id], view_id=view_id))
        specs.extend(
            pose_to_camera_spec(
                baseline=baseline_by_view[view_id], view_id=view_id, candidate=raw
            )
            for raw in proposals[view_id]
        )
    candidate_specs_path = _write_object(
        output / "pose_model_candidate_view_specs.json",
        {"schema_version": VIEW_SPEC_SCHEMA_VERSION, "views": specs},
    )
    rendered = _run_render(
        isaac_python=args.isaac_python.expanduser().resolve(strict=True),
        registry=registry_path,
        output_dir=output / "renders",
        view_specs=candidate_specs_path,
        resolution=args.resolution,
        rt_subframes=args.rt_subframes,
        analysis_up_axis=args.analysis_up_axis,
        analysis_front_axis=args.analysis_front_axis,
    )
    references = _reference_masks(manifest_path)
    registry = _read_object(rendered)
    output_specs: list[dict[str, Any]] = []
    public_views: list[dict[str, Any]] = []
    for view_id in requested:
        reference_mask, reference_row = references[view_id]
        reference_image = _reference_image(reference_row, manifest_path, reference_mask.shape)
        _, records = _score_candidates(
            reference_id=view_id,
            reference_mask=reference_mask,
            reference_image=reference_image,
            registry_path=rendered,
        )
        decision, winner = select_verified_seed(records)
        baseline_score = next(
            raw
            for raw in records
            if raw.get("calibration", {}).get("proposal_rank") == 0
        )
        selected_rank = int(winner["calibration"]["proposal_rank"])
        selected_spec = (
            baseline_by_view[view_id]
            if selected_rank == 0
            else next(
                raw
                for raw in specs
                if raw["view_id"] == winner["view_id"]
            )
        )
        output_specs.append(
            {
                **{
                    key: value
                    for key, value in selected_spec.items()
                    if key not in {"view_id", "calibration"}
                },
                "view_id": view_id,
                "calibration": {
                    "reference_view_id": view_id,
                    "phase": "pose_model_verified_seed",
                    "pose_model_decision": decision,
                    "proposal_rank": selected_rank,
                    "frame_anchor_affine": winner["calibration"]["frame_anchor_affine"],
                },
            }
        )

        def public_score(score: Mapping[str, Any], label: str) -> dict[str, Any]:
            view = _view_record(registry, str(score["view_id"]))
            rgb = str(Path(str(view["rgb"])).expanduser().resolve(strict=True))
            residual = _residual(
                registry_path=rendered,
                view_id=str(score["view_id"]),
                reference_mask=reference_mask,
                score=score,
                output_dir=output / "residuals" / view_id / label,
            )
            return {
                "view_id": score["view_id"],
                "projection_iou": score["projection_iou"],
                "boundary_p95_px": score["boundary_p95_px"],
                "score": score["score"],
                "rgb": rgb,
                "residual": residual,
            }

        public_views.append(
            {
                "view_id": view_id,
                "decision": decision,
                "selected_rank": selected_rank,
                "baseline": public_score(baseline_score, "baseline"),
                "selected": public_score(winner, "selected"),
                "candidate_count": len(records) - 1,
                "candidate_scores": [
                    {
                        "rank": int(raw["calibration"]["proposal_rank"]),
                        "view_id": raw["view_id"],
                        "projection_iou": raw["projection_iou"],
                        "boundary_p95_px": raw["boundary_p95_px"],
                        "score": raw["score"],
                        "rigid_consensus_score": raw.get("rigid_consensus_score"),
                        "accepted": raw["view_id"] == winner["view_id"],
                    }
                    for raw in sorted(
                        records,
                        key=lambda item: int(
                            item.get("calibration", {}).get("proposal_rank", 999)
                        ),
                    )
                ],
                "all_registered_parts_share_one_rigid_asset": True,
                "per_mesh_transform_applied": False,
            }
        )
    initial_specs_path = _write_object(
        output / "initial_view_specs.json",
        {"schema_version": VIEW_SPEC_SCHEMA_VERSION, "views": output_specs},
    )
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "PASS",
        "model": {"name": MODEL_NAME, "repository_revision": MODEL_REVISION},
        "registry": str(registry_path),
        "registry_sha256": _sha256(registry_path),
        "reference_manifest": str(manifest_path),
        "reference_manifest_sha256": _sha256(manifest_path),
        "baseline_view_specs": str(baseline_path),
        "baseline_view_specs_sha256": _sha256(baseline_path),
        "proposals": str(proposal_path),
        "proposals_sha256": _sha256(proposal_path),
        "rendered_registry": str(rendered),
        "initial_view_specs": str(initial_specs_path),
        "whole_asset_only": True,
        "per_mesh_or_subtree_transform_applied": False,
        "learned_pose_is_proposal_only": True,
        "isaac_render_is_selection_authority": True,
        "views": public_views,
    }
    _write_object(output / "pose_model_camera_seed_report.json", report)
    _viewer(output=output, report=report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--baseline-view-specs", type=Path, required=True)
    parser.add_argument("--proposals", type=Path, required=True)
    parser.add_argument("--isaac-python", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--views", default="front,side,top,iso")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--rt-subframes", type=int, default=4)
    parser.add_argument("--analysis-up-axis", default="z")
    parser.add_argument("--analysis-front-axis", default="-y")
    return parser.parse_args()


def main() -> int:
    report = run(parse_args())
    print(json.dumps({"status": report["status"], "views": report["views"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
