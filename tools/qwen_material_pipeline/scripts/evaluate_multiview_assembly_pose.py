#!/usr/bin/env python3
"""Render and validate one assembly-pose correction across several cameras."""

from __future__ import annotations

import argparse
import html
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Mapping

import cv2

from qwen_material_pipeline.evidence.assembly_pose import _foreground_mask, _ids_image
from qwen_material_pipeline.evidence.camera_calibration import (
    _part_residual_attribution,
    _read_object,
    _reference_image,
    _reference_masks,
    _score_candidates,
    _write_residual_audit,
)


SCHEMA_VERSION = "qwen-multiview-assembly-pose-evaluation/v1"


def _write_object(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _camera_sources(values: list[str]) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for raw in values:
        if "=" not in raw:
            raise ValueError("--camera must be VIEW=REPORT")
        view_id, raw_path = raw.split("=", 1)
        if not view_id or view_id in output:
            raise ValueError(f"Duplicate or empty camera view: {view_id!r}")
        output[view_id] = Path(raw_path).expanduser().resolve(strict=True)
    if len(output) < 2:
        raise ValueError("Multi-view assembly evaluation requires at least two views")
    return output


def _camera_contract(
    *, camera_reports: Mapping[str, Path], source_registry: Mapping[str, Any]
) -> tuple[dict[str, Any], Path, list[dict[str, Any]]]:
    expected_parts = sorted(
        (str(raw["part_id"]), str(raw["prim_path"]))
        for raw in source_registry.get("parts", [])
        if isinstance(raw, Mapping)
        and isinstance(raw.get("part_id"), str)
        and isinstance(raw.get("prim_path"), str)
    )
    if not expected_parts:
        raise ValueError("Source registry has no Part-ID hierarchy")
    manifest: Path | None = None
    views: list[dict[str, Any]] = []
    provenance: dict[str, Any] = {}
    for view_id, report_path in camera_reports.items():
        report = _read_object(report_path)
        candidate_manifest = Path(str(report["reference_manifest"])).resolve(strict=True)
        if manifest is None:
            manifest = candidate_manifest
        elif candidate_manifest != manifest:
            raise ValueError("Camera reports do not share one reference manifest")
        report_registry_path = Path(str(report["source_registry"])).resolve(strict=True)
        report_registry = _read_object(report_registry_path)
        report_parts = sorted(
            (str(raw["part_id"]), str(raw["prim_path"]))
            for raw in report_registry.get("parts", [])
            if isinstance(raw, Mapping)
            and isinstance(raw.get("part_id"), str)
            and isinstance(raw.get("prim_path"), str)
        )
        if report_parts != expected_parts:
            raise ValueError(f"Camera {view_id!r} belongs to another Part hierarchy")
        rows = [
            raw
            for raw in report.get("views", [])
            if isinstance(raw, Mapping)
            and raw.get("reference_view_id") == view_id
            and isinstance(raw.get("final"), Mapping)
        ]
        if len(rows) != 1:
            raise ValueError(f"Camera report does not exactly bind {view_id!r}")
        specs_path = Path(str(report["final_view_specs"])).resolve(strict=True)
        specs = _read_object(specs_path)
        spec_rows = [
            dict(raw)
            for raw in specs.get("views", [])
            if isinstance(raw, Mapping) and raw.get("view_id") == view_id
        ]
        if len(spec_rows) != 1:
            raise ValueError(f"Camera specs do not exactly bind {view_id!r}")
        spec = spec_rows[0]
        calibration = dict(spec.get("calibration", {}))
        calibration["frame_anchor_affine"] = rows[0]["final"][
            "whole_asset_similarity"
        ]["bbox_affine"]
        spec["calibration"] = calibration
        views.append(spec)
        provenance[view_id] = {
            "camera_report": str(report_path),
            "camera_view_specs": str(specs_path),
            "baseline_projection_iou": rows[0]["final"]["projection_iou"],
            "baseline_boundary_p95_px": rows[0]["final"]["boundary_p95_px"],
        }
    assert manifest is not None
    return provenance, manifest, views


def _render(
    *,
    python_sh: Path,
    repository_root: Path,
    source_registry: Path,
    view_specs: Path,
    output: Path,
    resolution: int,
    rt_subframes: int,
    override: Path | None,
    whole_asset_pose: Path | None = None,
) -> Path:
    command = [
        str(python_sh),
        "-m",
        "qwen_material_pipeline",
        "usd",
        "render",
        "--registry",
        str(source_registry),
        "--output-dir",
        str(output),
        "--resolution",
        str(resolution),
        "--view-specs",
        str(view_specs),
        "--rt-subframes",
        str(rt_subframes),
        "--lighting-profile",
        "material-neutral",
        "--analysis-up-axis",
        "z",
        "--analysis-front-axis=-y",
        "--rgb-only",
    ]
    if override is not None:
        command.extend(("--assembly-pose-overrides", str(override)))
    if whole_asset_pose is not None:
        command.extend(("--whole-asset-pose", str(whole_asset_pose)))
    environment = dict(os.environ)
    tools_path = str(repository_root / "tools")
    environment["PYTHONPATH"] = (
        tools_path
        if not environment.get("PYTHONPATH")
        else tools_path + os.pathsep + environment["PYTHONPATH"]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with (output.parent / f"{output.name}.log").open("w", encoding="utf-8") as stream:
        result = subprocess.run(
            command,
            cwd=repository_root,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode != 0:
        raise RuntimeError(f"Multi-view render failed for {output}: {result.returncode}")
    registry = output / "part_registry.rendered.json"
    if not registry.is_file():
        raise RuntimeError(f"Renderer produced no sealed registry: {output}")
    return registry.resolve(strict=True)


def _rgb(registry_path: Path, view_id: str) -> Path:
    registry = _read_object(registry_path)
    for raw in registry.get("render_set", {}).get("views", []):
        if isinstance(raw, Mapping) and raw.get("view_id") == view_id:
            path = Path(str(raw["rgb"])).expanduser()
            if not path.is_absolute():
                path = registry_path.parent / path
            return path.resolve(strict=True)
    raise ValueError(f"Registry has no RGB view {view_id!r}")


def _score_view(
    *,
    view_id: str,
    registry_path: Path,
    reference_mask: Any,
    reference_image: Any,
    output: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    score, _ = _score_candidates(
        reference_id=view_id,
        reference_mask=reference_mask,
        reference_image=reference_image,
        registry_path=registry_path,
    )
    ids = _ids_image(_read_object(registry_path), registry_path, view_id)
    foreground = _foreground_mask(ids, _read_object(registry_path))
    residual = _write_residual_audit(
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
        output_dir=output,
    )
    return score, residual


def _viewer(*, output: Path, report: Mapping[str, Any]) -> None:
    assets = output / "viewer" / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    figures: list[str] = []
    cards: list[str] = []
    for row in report["views"]:
        view_id = str(row["view_id"])
        for source, name in (
            (Path(row["before"]["rgb"]), f"{view_id}_before_rgb.png"),
            (Path(row["after"]["rgb"]), f"{view_id}_after_rgb.png"),
            (
                Path(row["before"]["residual"]["residual_overlay"]),
                f"{view_id}_before_residual.png",
            ),
            (
                Path(row["after"]["residual"]["residual_overlay"]),
                f"{view_id}_after_residual.png",
            ),
        ):
            shutil.copy2(source, assets / name)
        before = row["before"]
        after = row["after"]
        status = "检测到并应用 3D 修正" if row["source_decision"] == "OPTIMIZED" else "无合格装配残差，保持不动"
        cards.append(
            f"<article><h2>{html.escape(view_id)}</h2><p>{status}</p>"
            f"<b>IoU {before['projection_iou']:.4f} → {after['projection_iou']:.4f}</b><br>"
            f"Boundary P95 {before['boundary_p95_px']:.2f} → {after['boundary_p95_px']:.2f} px<br>"
            f"错配 {100*before['residual']['mismatch_over_union']:.2f}% → {100*after['residual']['mismatch_over_union']:.2f}%</article>"
        )
        figures.append(
            f"<section><h2>{html.escape(view_id)}</h2><div class='grid'>"
            f"<figure><img src='assets/{view_id}_before_rgb.png'><figcaption>优化前 RGB</figcaption></figure>"
            f"<figure><img src='assets/{view_id}_after_rgb.png'><figcaption>优化后 RGB</figcaption></figure>"
            f"<figure><img src='assets/{view_id}_before_residual.png'><figcaption>优化前残差</figcaption></figure>"
            f"<figure><img src='assets/{view_id}_after_residual.png'><figcaption>优化后残差</figcaption></figure>"
            "</div></section>"
        )
    gates = report["gates"]
    if report["status"] == "PASS":
        conclusion = (
            "同一组 3D 装配变换在主视角产生改善，并通过其余视角非退化门禁。"
        )
    elif report["status"] == "PASS_NO_OP":
        conclusion = (
            "联合搜索未找到同时改善全部视角的共享 3D 变换，因此安全保留"
            "工件原始根位姿。所有候选都只移动整件工件，没有局部 Mesh 或"
            "子装配变换。"
        )
    else:
        conclusion = (
            "联合拒绝：主视角 IoU 改善 "
            f"{float(gates['primary_iou_gain']):+.4f}，但其他视角最坏退化 "
            f"{float(gates['worst_secondary_iou_regression']):.4f}，超过 "
            f"{float(gates['maximum_allowed_secondary_iou_regression']):.4f}。"
            "这说明当前参考照片不能由这一组固定 3D 装配姿态共同解释。"
        )
    shutil.copy2(
        output / "multiview_assembly_pose_report.json",
        output / "viewer" / "multiview_assembly_pose_report.json",
    )
    shutil.copy2(
        output / "assembly_pose_overrides.json",
        output / "viewer" / "assembly_pose_overrides.json",
    )
    page = f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>通用多视角装配位姿验收</title><style>
body{{margin:0;background:#0d1117;color:#f0f6fc;font:15px/1.55 system-ui}}main{{width:min(1500px,96vw);margin:auto;padding:28px 0}}p,figcaption{{color:#9da7b3}}.cards,.grid{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}}article,figure{{margin:0;background:#161b22;border:1px solid #30363d;border-radius:10px;overflow:hidden}}article{{padding:14px}}figure img{{display:block;width:100%}}figcaption{{padding:8px}}b{{color:#56d364}}code{{color:#ffa657}}@media(max-width:900px){{.cards,.grid{{grid-template-columns:1fr 1fr}}}}@media(max-width:560px){{.cards,.grid{{grid-template-columns:1fr}}}}</style></head><body><main>
<h1>通用分层装配刚体配准 · 四视角联合验收</h1><p>四个视角独立检测残差，但共享唯一 3D 装配状态。只有产生可靠装配证据的视角提出变换；同一变换随后在全部相机中真实重渲染。红=仅照片，蓝=仅 CAD，绿=重合。没有做 2D warp，也没有按视角分别扭曲模型。</p>
<article><h2>结论：{html.escape(str(report['status']))}</h2><p>{html.escape(conclusion)}</p></article>
<div class='cards'>{''.join(cards)}</div>{''.join(figures)}
<p>完整数据：<a href='multiview_assembly_pose_report.json'>JSON 报告</a>；3D 变换：<a href='assembly_pose_overrides.json'>override</a>。</p></main></body></html>"""
    (output / "viewer" / "index.html").write_text(page, encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    source_registry_path = args.source_registry.resolve(strict=True)
    source_registry = _read_object(source_registry_path)
    camera_reports = _camera_sources(args.camera)
    provenance, manifest_path, view_specs = _camera_contract(
        camera_reports=camera_reports, source_registry=source_registry
    )
    optimizer_report_path = args.optimization_report.resolve(strict=True)
    optimizer_report = _read_object(optimizer_report_path)
    if optimizer_report.get("status") != "PASS":
        raise ValueError("Assembly optimizer report is not sealed PASS")
    primary_view = str(optimizer_report["reference_view_id"])
    if primary_view not in camera_reports:
        raise ValueError("Optimizer view is absent from multi-view cameras")
    override = (
        args.assembly_pose_overrides.resolve(strict=True)
        if args.assembly_pose_overrides is not None
        else Path(
            str(optimizer_report["winner"]["assembly_pose_overrides"])
        ).resolve(strict=True)
    )
    output.mkdir(parents=True)
    _write_object(
        output / "multiview_view_specs.json",
        {"schema_version": "qwen-camera-view-specs/v1", "views": view_specs},
    )
    shutil.copy2(override, output / "assembly_pose_overrides.json")
    before_registry = _render(
        python_sh=args.python_sh.resolve(strict=True),
        repository_root=args.repository_root.resolve(strict=True),
        source_registry=source_registry_path,
        view_specs=output / "multiview_view_specs.json",
        output=output / "before" / "renders",
        resolution=args.resolution,
        rt_subframes=args.rt_subframes,
        override=None,
        whole_asset_pose=None,
    )
    after_registry = _render(
        python_sh=args.python_sh.resolve(strict=True),
        repository_root=args.repository_root.resolve(strict=True),
        source_registry=source_registry_path,
        view_specs=output / "multiview_view_specs.json",
        output=output / "after" / "renders",
        resolution=args.resolution,
        rt_subframes=args.rt_subframes,
        override=output / "assembly_pose_overrides.json",
        whole_asset_pose=None,
    )
    references = _reference_masks(manifest_path)
    rows: list[dict[str, Any]] = []
    for view_id in camera_reports:
        mask, reference_row = references[view_id]
        image = _reference_image(reference_row, manifest_path, mask.shape)
        before_score, before_residual = _score_view(
            view_id=view_id,
            registry_path=before_registry,
            reference_mask=mask,
            reference_image=image,
            output=output / "before" / "residuals" / view_id,
        )
        after_score, after_residual = _score_view(
            view_id=view_id,
            registry_path=after_registry,
            reference_mask=mask,
            reference_image=image,
            output=output / "after" / "residuals" / view_id,
        )
        rows.append(
            {
                "view_id": view_id,
                "source_decision": "OPTIMIZED" if view_id == primary_view else "NO_OP",
                "camera": provenance[view_id],
                "before": {
                    "projection_iou": before_score["projection_iou"],
                    "boundary_p95_px": before_score["boundary_p95_px"],
                    "rgb": str(_rgb(before_registry, view_id)),
                    "residual": before_residual,
                },
                "after": {
                    "projection_iou": after_score["projection_iou"],
                    "boundary_p95_px": after_score["boundary_p95_px"],
                    "rgb": str(_rgb(after_registry, view_id)),
                    "residual": after_residual,
                },
                "delta": {
                    "projection_iou": round(
                        after_score["projection_iou"] - before_score["projection_iou"],
                        8,
                    ),
                    "boundary_p95_px": round(
                        after_score["boundary_p95_px"] - before_score["boundary_p95_px"],
                        8,
                    ),
                    "mismatch_over_union": round(
                        after_residual["mismatch_over_union"]
                        - before_residual["mismatch_over_union"],
                        8,
                    ),
                },
            }
        )
    primary = next(raw for raw in rows if raw["view_id"] == primary_view)
    secondary = [raw for raw in rows if raw["view_id"] != primary_view]
    gates = {
        "primary_iou_gain": round(primary["delta"]["projection_iou"], 8),
        "primary_mismatch_gain": round(
            -primary["delta"]["mismatch_over_union"], 8
        ),
        "worst_secondary_iou_regression": round(
            max(0.0, max(-raw["delta"]["projection_iou"] for raw in secondary)), 8
        ),
        "maximum_allowed_secondary_iou_regression": 0.003,
    }
    accepted = (
        gates["primary_iou_gain"] > 0.0
        and gates["primary_mismatch_gain"] > 0.0
        and gates["worst_secondary_iou_regression"] <= 0.003
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS" if accepted else "REJECTED",
        "source_registry": str(source_registry_path),
        "source_asset": str(source_registry["asset_usd"]),
        "reference_manifest": str(manifest_path),
        "optimization_report": str(optimizer_report_path),
        "evaluated_override_source": str(override),
        "primary_optimization_view": primary_view,
        "assembly_pose_overrides": str(output / "assembly_pose_overrides.json"),
        "before_rendered_registry": str(before_registry),
        "after_rendered_registry": str(after_registry),
        "views": rows,
        "gates": gates,
    }
    _write_object(output / "multiview_assembly_pose_report.json", result)
    _viewer(output=output, report=result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-registry", type=Path, required=True)
    parser.add_argument("--optimization-report", type=Path, required=True)
    parser.add_argument(
        "--assembly-pose-overrides",
        type=Path,
        help="Evaluate this optimizer-generated candidate instead of its winner",
    )
    parser.add_argument("--camera", action="append", default=[], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python-sh", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--rt-subframes", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    result = run(parse_args())
    print(json.dumps({"status": result["status"], "gates": result["gates"]}, indent=2))
    return 0 if result["status"] == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
