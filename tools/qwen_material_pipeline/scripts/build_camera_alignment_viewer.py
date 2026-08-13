#!/usr/bin/env python3
"""Build a static Phase-1/Phase-2 camera alignment comparison page."""

from __future__ import annotations

import argparse
import html
import json
import shutil
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _side(report: Mapping[str, Any]) -> Mapping[str, Any]:
    for raw in report.get("views", []):
        if isinstance(raw, Mapping) and raw.get("reference_view_id") == "side":
            final = raw.get("final")
            if isinstance(final, Mapping):
                return final
    raise ValueError("Camera report has no side final score")


def _side_row(report: Mapping[str, Any]) -> Mapping[str, Any]:
    for raw in report.get("views", []):
        if isinstance(raw, Mapping) and raw.get("reference_view_id") == "side":
            return raw
    raise ValueError("Camera report has no side row")


def _final_rgb(report: Mapping[str, Any], report_path: Path) -> Path:
    registry_path = Path(str(report["final_rendered_registry"]))
    registry = _read(registry_path)
    for raw in registry.get("render_set", {}).get("views", []):
        if isinstance(raw, Mapping) and raw.get("view_id") == "side":
            return Path(str(raw["rgb"])).expanduser().resolve(strict=True)
    raise ValueError(f"Final registry has no side render: {registry_path}")


def _fit_reference_to_render(
    *, reference: Path, reference_mask: Path, render: Path, output: Path
) -> None:
    image = cv2.imread(str(reference), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(reference_mask), cv2.IMREAD_GRAYSCALE)
    rendered = cv2.imread(str(render), cv2.IMREAD_COLOR)
    if image is None or mask is None or rendered is None:
        raise ValueError("Unable to read comparison image")
    y, x = np.nonzero(mask > 0)
    if not len(x):
        raise ValueError("Reference mask is empty")
    left, right = int(x.min()), int(x.max()) + 1
    top, bottom = int(y.min()), int(y.max()) + 1
    crop = image[top:bottom, left:right]
    canvas = np.zeros_like(rendered)
    scale = min(
        rendered.shape[1] / max(1, crop.shape[1]),
        rendered.shape[0] / max(1, crop.shape[0]),
    )
    size = (
        max(1, round(crop.shape[1] * scale)),
        max(1, round(crop.shape[0] * scale)),
    )
    resized = cv2.resize(crop, size, interpolation=cv2.INTER_AREA)
    ox = (rendered.shape[1] - size[0]) // 2
    oy = (rendered.shape[0] - size[1]) // 2
    canvas[oy : oy + size[1], ox : ox + size[0]] = resized
    if not cv2.imwrite(str(output), canvas):
        raise ValueError(f"Unable to write {output}")


def _f(value: Any, digits: int = 4) -> str:
    return "—" if value is None else f"{float(value):.{digits}f}"


def _delta(old: Any, new: Any, digits: int = 4) -> str:
    if old is None or new is None:
        return "—"
    return f"{float(new) - float(old):+.{digits}f}"


def build(*, phase1: Path, phase2: Path, manifest: Path, output: Path) -> None:
    phase1_report_path = phase1 / "camera_calibration_report.json"
    phase2_report_path = phase2 / "camera_calibration_report.json"
    p1_report = _read(phase1_report_path)
    p2_report = _read(phase2_report_path)
    p1 = _side(p1_report)
    p2 = _side(p2_report)
    p1_row = _side_row(p1_report)
    p2_row = _side_row(p2_report)
    manifest_doc = _read(manifest)
    source = next(
        raw
        for raw in manifest_doc.get("source_views", [])
        if isinstance(raw, Mapping) and raw.get("id") == "side"
    )
    reference = Path(str(source["image"])).expanduser().resolve(strict=True)
    confirmed = source.get("confirmed_mask")
    if not isinstance(confirmed, Mapping):
        raise ValueError("Side reference has no confirmed mask")
    reference_mask = (manifest.parent / str(confirmed["path"])).resolve(strict=True)
    output.mkdir(parents=True, exist_ok=True)
    assets = output / "assets"
    assets.mkdir(exist_ok=True)
    p1_rgb = _final_rgb(p1_report, phase1_report_path)
    p2_rgb = _final_rgb(p2_report, phase2_report_path)
    for source_path, name in (
        (p1_rgb, "phase1_side.png"),
        (p2_rgb, "phase2_side.png"),
        (Path(str(p1_row["residual_audit"]["residual_overlay"])), "phase1_residual.png"),
        (Path(str(p2_row["residual_audit"]["residual_overlay"])), "phase2_residual.png"),
    ):
        shutil.copy2(source_path, assets / name)
    _fit_reference_to_render(
        reference=reference,
        reference_mask=reference_mask,
        render=p2_rgb,
        output=assets / "reference_side.png",
    )

    clusters = p2.get("assembly_residual_clusters", [])
    cluster_rows = "".join(
        "<tr>"
        f"<td>{html.escape(', '.join(raw.get('part_ids', [])))}</td>"
        f"<td>{html.escape(str(raw.get('classification', '')))}</td>"
        f"<td>{_f(raw.get('median_residual_px'), 2)} px</td>"
        f"<td>{_f(raw.get('residual_direction_coherence'), 2)}</td>"
        "</tr>"
        for raw in clusters
        if isinstance(raw, Mapping)
    )
    page = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>通用刚性共识相机对齐 · Phase 2</title><style>
:root{{--bg:#0d1117;--panel:#161b22;--line:#30363d;--text:#f0f6fc;--muted:#9da7b3;--good:#56d364;--blue:#79c0ff;--warn:#e3b341}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}}
main{{width:min(1500px,96vw);margin:auto;padding:32px 0 56px}}h1{{margin:0;font-size:30px}}h2{{margin:30px 0 14px}}p{{color:var(--muted)}}
.cards,.grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}}.card,figure{{margin:0;background:var(--panel);border:1px solid var(--line);border-radius:12px;overflow:hidden}}
.card{{padding:17px}}figure img{{display:block;width:100%;height:auto;background:#05070a}}figcaption{{padding:10px 13px;color:var(--muted)}}
.metric{{font-size:25px;font-weight:700}}.good{{color:var(--good)}}.muted{{color:var(--muted)}}table{{width:100%;border-collapse:collapse;background:var(--panel);border:1px solid var(--line)}}th,td{{padding:10px 12px;text-align:left;border-bottom:1px solid var(--line)}}th{{color:var(--blue)}}a{{color:var(--blue)}}code{{color:#ffa657}}@media(max-width:900px){{.cards,.grid{{grid-template-columns:1fr}}}}
</style></head><body><main>
<h1>通用刚性共识相机对齐 · Phase 2</h1>
<p>真实 512 px side A/B。无 DTN100 Part-ID 白名单、无局部几何 warp；相机由可观测 Part 的稳健共识参与排序，离群装配只诊断、不反向拖动全局相机。</p>
<section class="cards">
<article class="card"><div class="muted">Boundary P95</div><div class="metric good">{_f(p1.get('boundary_p95_px'),2)} → {_f(p2.get('boundary_p95_px'),2)} px</div><p>{_delta(p1.get('boundary_p95_px'),p2.get('boundary_p95_px'),2)} px（越低越好）</p></article>
<article class="card"><div class="muted">Part structure P75</div><div class="metric good">{_f(p1.get('structure_p75_px'),2)} → {_f(p2.get('structure_p75_px'),2)} px</div><p>{_delta(p1.get('structure_p75_px'),p2.get('structure_p75_px'),2)} px（越低越好）</p></article>
<article class="card"><div class="muted">Silhouette IoU</div><div class="metric">{_f(p1.get('projection_iou'))} → {_f(p2.get('projection_iou'))}</div><p>{_delta(p1.get('projection_iou'),p2.get('projection_iou'))}；以减少长尾局部残差为主要目标</p></article>
</section>
<h2>照片 / Phase 1 / Phase 2</h2><section class="grid">
<figure><img src="assets/reference_side.png"><figcaption>参考照片 · 前景裁切并适配同画布</figcaption></figure>
<figure><img src="assets/phase1_side.png"><figcaption>Phase 1 · P95 {_f(p1.get('boundary_p95_px'),2)} px</figcaption></figure>
<figure><img src="assets/phase2_side.png"><figcaption>Phase 2 v15 · P95 {_f(p2.get('boundary_p95_px'),2)} px</figcaption></figure>
</section>
<h2>轮廓残差</h2><p>绿色=重合，红色=仅照片，蓝色=仅 CAD。</p><section class="grid" style="grid-template-columns:repeat(2,minmax(0,1fr))">
<figure><img src="assets/phase1_residual.png"><figcaption>Phase 1 residual</figcaption></figure>
<figure><img src="assets/phase2_residual.png"><figcaption>Phase 2 residual</figcaption></figure></section>
<h2>通用刚性共识诊断</h2>
<section class="cards">
<article class="card"><div class="muted">稳健共识残差</div><div class="metric">{_f(p2.get('rigid_consensus_residual_px'),2)} px</div></article>
<article class="card"><div class="muted">刚性共识像素覆盖</div><div class="metric">{100*float(p2.get('rigid_consensus_pixel_coverage',0)):.1f}%</div></article>
<article class="card"><div class="muted">可观测 / 不确定 Part</div><div class="metric">{int(p2.get('rigid_consensus_candidate_part_count',0))} / {int(p2.get('rigid_consensus_indeterminate_part_count',0))}</div></article>
</section>
<p>只有同一装配子树中多个 Part 的二维残差方向一致，才标记为 assembly mismatch。照片里缺少可观测接缝的内部大面板标为 indeterminate，不参与相机投票。</p>
<table><thead><tr><th>Part IDs</th><th>分类</th><th>中位残差</th><th>方向一致性</th></tr></thead><tbody>{cluster_rows}</tbody></table>
<h2>验收结论</h2><p>相对 Phase 1，长尾边界与内部结构均改善；但 <code>P0471/P0473</code> 仍形成同一装配分支、同向偏移的高置信局部残差。因此这是更稳健的全局相机，不是通过移动局部 CAD 来制造的“完美贴合”。</p>
<footer><a href="camera_calibration_report.json">Phase 2 完整报告</a> · <a href="final_view_specs.json">Phase 2 相机参数</a> · <a href="phase1_camera_calibration_report.json">Phase 1 报告</a></footer>
</main></body></html>"""
    (output / "index.html").write_text(page, encoding="utf-8")
    shutil.copy2(phase2_report_path, output / "camera_calibration_report.json")
    shutil.copy2(phase2 / "final_view_specs.json", output / "final_view_specs.json")
    shutil.copy2(phase1_report_path, output / "phase1_camera_calibration_report.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase1", type=Path, required=True)
    parser.add_argument("--phase2", type=Path, required=True)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(
        phase1=args.phase1.resolve(strict=True),
        phase2=args.phase2.resolve(strict=True),
        manifest=args.reference_manifest.resolve(strict=True),
        output=args.output.resolve(),
    )


if __name__ == "__main__":
    main()
