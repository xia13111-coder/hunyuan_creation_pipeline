#!/usr/bin/env python3
"""Build a static before/after viewer for rigid assembly pose optimization."""

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


def _view_image(registry_path: Path, view_id: str, key: str) -> Path:
    registry = _read(registry_path)
    for raw in registry.get("render_set", {}).get("views", []):
        if isinstance(raw, Mapping) and raw.get("view_id") == view_id:
            return Path(str(raw[key])).expanduser().resolve(strict=True)
    raise ValueError(f"Registry has no {view_id!r} view: {registry_path}")


def _regional_crop(*, source: Path, bounds: list[int], output: Path) -> None:
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to load residual overlay: {source}")
    left, top, right, bottom = bounds
    crop = image[top:bottom, left:right]
    if crop.size == 0:
        raise ValueError("Regional assembly comparison is empty")
    crop = cv2.resize(crop, None, fx=5, fy=5, interpolation=cv2.INTER_NEAREST)
    if not cv2.imwrite(str(output), crop):
        raise ValueError(f"Unable to write {output}")


def _number(value: Any, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def build(*, run: Path, output: Path) -> None:
    report_path = run / "assembly_pose_report.json"
    report = _read(report_path)
    baseline = report["baseline"]
    winner = report["winner"]
    nested = report.get("nested_refinement", {})
    output.mkdir(parents=True, exist_ok=True)
    assets = output / "assets"
    assets.mkdir(exist_ok=True)
    baseline_registry = Path(str(baseline["rendered_registry"])).resolve(strict=True)
    winner_registry = Path(str(winner["rendered_registry"])).resolve(strict=True)
    for source, name in (
        (_view_image(baseline_registry, "side", "rgb"), "before_rgb.png"),
        (_view_image(winner_registry, "side", "rgb"), "after_rgb.png"),
        (Path(str(winner["residual_audit"]["residual_overlay"])), "after_residual.png"),
    ):
        shutil.copy2(source, assets / name)
    baseline_camera_report = _read(Path(str(report["camera_report"])))
    side_row = next(
        raw
        for raw in baseline_camera_report["views"]
        if raw.get("reference_view_id") == "side"
    )
    shutil.copy2(
        Path(str(side_row["residual_audit"]["residual_overlay"])),
        assets / "before_residual.png",
    )
    bounds = report["acceptance"]["regional_support_bounds_xyxy"]
    _regional_crop(
        source=Path(str(side_row["residual_audit"]["residual_overlay"])),
        bounds=bounds,
        output=assets / "before_local.png",
    )
    _regional_crop(
        source=Path(str(winner["residual_audit"]["residual_overlay"])),
        bounds=bounds,
        output=assets / "after_local.png",
    )
    subtree = html.escape(str(winner["assembly_subtree"]))
    members = html.escape(", ".join(winner["subtree_member_part_ids"]))
    before_local = baseline["local"]
    after_local = winner["local"]
    before_regional = baseline["regional"]
    after_regional = winner["regional"]
    before_global = baseline["global"]
    after_global = winner["global"]
    before_mismatch = float(side_row["residual_audit"]["mismatch_over_union"])
    after_mismatch = float(winner["residual_audit"]["mismatch_over_union"])
    nested_summary = ""
    if nested.get("accepted") is True:
        nested_subtree = html.escape(str(nested["assembly_subtree"]))
        nested_part = html.escape(str(nested["part_id"]))
        before_inside = float(nested["baseline_residual"]["inside_reference_ratio"])
        after_inside = float(nested["final_part_residual"]["inside_reference_ratio"])
        nested_summary = (
            "<h2>父装配内部二次残差</h2>"
            "<p>父层修正后，程序又在其内部自动选择残差最大的独立 Xform，"
            "用两次真实渲染标定屏幕/世界位移响应，再由第三次真实渲染验收。"
            f"选中 Part：<code>{nested_part}</code>，Xform：<code>{nested_subtree}</code>；"
            f"照片轮廓内覆盖率 {100*before_inside:.1f}% → {100*after_inside:.1f}%。"
            "任何一步不改善都会回退到父装配结果。</p>"
        )
    page = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>通用装配子树刚体校正 · Side</title><style>
:root{{--bg:#0d1117;--panel:#161b22;--line:#30363d;--text:#f0f6fc;--muted:#9da7b3;--green:#56d364;--blue:#79c0ff}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}}main{{width:min(1500px,96vw);margin:auto;padding:30px 0 60px}}h1{{margin:0;font-size:30px}}h2{{margin:30px 0 14px}}p{{color:var(--muted)}}.cards,.grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}}.card,figure{{margin:0;background:var(--panel);border:1px solid var(--line);border-radius:12px;overflow:hidden}}.card{{padding:16px}}.metric{{font-size:25px;font-weight:750;color:var(--green)}}figure img{{display:block;width:100%;height:auto;background:#05070a}}figcaption{{padding:10px 13px;color:var(--muted)}}code{{color:#ffa657;overflow-wrap:anywhere}}a{{color:var(--blue)}}.two{{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:900px){{.cards,.grid,.two{{grid-template-columns:1fr}}}}</style></head><body><main>
<h1>通用分层装配刚体校正 · Side</h1><p>先固定全局相机，再自动发现同一 USD Xform 子树内、方向一致的多 Part 残差；父层通过后，再对其内部最大的独立 Xform 残差做一次真实渲染响应校正。不做 2D warp，不写死资产名称或 Part ID。</p>
<section class="cards"><article class="card"><div>局部整片 IoU（层级权威）</div><div class="metric">{_number(before_regional['iou'])} → {_number(after_regional['iou'])}</div></article><article class="card"><div>局部蓝色多余像素</div><div class="metric">{before_regional['cad_only_pixels']} → {after_regional['cad_only_pixels']}</div></article><article class="card"><div>锚点组件 IoU</div><div class="metric">{_number(before_local['iou'])} → {_number(after_local['iou'])}</div></article><article class="card"><div>锚点质心误差</div><div class="metric">{_number(before_local['centroid_error_px'],2)} → {_number(after_local['centroid_error_px'],2)} px</div></article><article class="card"><div>整机 IoU</div><div class="metric">{_number(before_global['projection_iou'])} → {_number(after_global['projection_iou'])}</div></article><article class="card"><div>整机 Boundary P95</div><div class="metric">{_number(before_global['boundary_p95_px'],2)} → {_number(after_global['boundary_p95_px'],2)} px</div></article><article class="card"><div>总错配 / union</div><div class="metric">{100*before_mismatch:.2f}% → {100*after_mismatch:.2f}%</div></article><article class="card"><div>固定几何共识残差</div><div class="metric">{_number(before_global['fixed_geometry_consensus_residual_px'],2)} → {_number(after_global['fixed_geometry_consensus_residual_px'],2)} px</div></article></section>
<h2>你指出的局部问题</h2><p>绿色=重合，红色=仅照片，蓝色=仅 CAD。这里展示整片邻域，不再只截取一个目标块。</p><section class="grid two"><figure><img src="assets/before_local.png"><figcaption>修正前：区域 IoU {_number(before_regional['iou'])}，蓝色多余 {before_regional['cad_only_pixels']} px</figcaption></figure><figure><img src="assets/after_local.png"><figcaption>修正后：区域 IoU {_number(after_regional['iou'])}，蓝色多余 {after_regional['cad_only_pixels']} px</figcaption></figure></section>
<h2>真实 512px RGB</h2><section class="grid two"><figure><img src="assets/before_rgb.png"><figcaption>修正前 · Phase 2 相机</figcaption></figure><figure><img src="assets/after_rgb.png"><figcaption>修正后 · 相机不变，只移动装配子树</figcaption></figure></section>
<h2>整机残差</h2><section class="grid two"><figure><img src="assets/before_residual.png"><figcaption>修正前 · mismatch {100*before_mismatch:.2f}%</figcaption></figure><figure><img src="assets/after_residual.png"><figcaption>修正后 · mismatch {100*after_mismatch:.2f}%</figcaption></figure></section>
{nested_summary}
<h2>通用约束</h2><p>检测出的 Xform：<code>{subtree}</code><br>该子树的完整注册成员：<code>{members}</code><br>世界平移：<code>{html.escape(json.dumps(winner['world_translation']))}</code>，范数 {_number(winner['translation_norm'],4)}。优化只接受：局部显著增益、整机 IoU 不退化、且排除移动子树后的固定几何共识不退化。</p>
<footer><a href="assembly_pose_report.json">完整优化报告</a> · <a href="assembly_pose_overrides.json">可复现刚体变换</a></footer></main></body></html>"""
    (output / "index.html").write_text(page, encoding="utf-8")
    shutil.copy2(report_path, output / "assembly_pose_report.json")
    shutil.copy2(
        run / "assembly_pose_overrides.json",
        output / "assembly_pose_overrides.json",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(run=args.run.resolve(strict=True), output=args.output.resolve())


if __name__ == "__main__":
    main()
