#!/usr/bin/env python3
"""Build a browser report comparing per-Part and view-shared CAD guidance."""

from __future__ import annotations

import argparse
import html
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

from .entityseg_comparison_report import _annotate, _mask, _overlay, _read_manifest
from .entityseg_regions import EntitySegRegionError
from .hybrid_part_mask_report import _all_records, _record_mask


def _template(row: Mapping[str, Any], key: str, shape: tuple[int, int]) -> np.ndarray:
    document = row.get(key)
    if not isinstance(document, Mapping) or not isinstance(document.get("path"), str):
        raise EntitySegRegionError(f"missing {key}")
    return _mask(Path(document["path"]), shape)


def _largest_old_translation(row: Mapping[str, Any]) -> tuple[float, float]:
    translations: list[tuple[float, float]] = []
    for box in row.get("box_audits", []):
        if not isinstance(box, Mapping):
            continue
        refinement = box.get("shape_point_refinement")
        prompt = refinement.get("prompt_audit") if isinstance(refinement, Mapping) else None
        value = prompt.get("translation_xy_pixels") if isinstance(prompt, Mapping) else None
        if isinstance(value, list) and len(value) == 2:
            translations.append((float(value[0]), float(value[1])))
    return max(translations, key=lambda item: abs(item[0]) + abs(item[1]), default=(0.0, 0.0))


def _tile(
    source: np.ndarray,
    seed: np.ndarray,
    amodal: np.ndarray,
    old_mask: np.ndarray,
    new_mask: np.ndarray,
    *,
    old_accepted: bool,
    new_accepted: bool,
) -> np.ndarray:
    union = seed | amodal | old_mask | new_mask
    ys, xs = np.where(union)
    pad = 20
    left = max(0, int(xs.min()) - pad)
    right = min(source.shape[1], int(xs.max()) + pad + 1)
    top = max(0, int(ys.min()) - pad)
    bottom = min(source.shape[0], int(ys.max()) + pad + 1)
    definitions = (
        ("CAD visible", seed, (0, 0, 255)),
        ("isolated mesh", amodal, (0, 165, 255)),
        ("OLD per-Part SAM3" if old_accepted else "OLD rejected", old_mask, (0, 255, 0)),
        ("NEW shared-view SAM3" if new_accepted else "NEW rejected", new_mask, (255, 0, 255)),
    )
    panels: list[np.ndarray] = []
    for label, mask, color in definitions:
        crop = _overlay(source, mask, color)[top:bottom, left:right]
        scale = min(360.0 / max(1, crop.shape[1]), 260.0 / max(1, crop.shape[0]))
        resized = cv2.resize(
            crop,
            (max(1, int(round(crop.shape[1] * scale))), max(1, int(round(crop.shape[0] * scale)))),
            interpolation=cv2.INTER_NEAREST,
        )
        panel = np.zeros((260, 360, 3), dtype=np.uint8)
        y0 = (260 - resized.shape[0]) // 2
        x0 = (360 - resized.shape[1]) // 2
        panel[y0 : y0 + resized.shape[0], x0 : x0 + resized.shape[1]] = resized
        _annotate(panel, label)
        panels.append(panel)
    return np.hstack(panels)


def build_report(*, old_manifest: Path, new_manifest: Path, output_dir: Path) -> dict[str, Any]:
    old_manifest = old_manifest.expanduser().resolve(strict=True)
    new_manifest = new_manifest.expanduser().resolve(strict=True)
    old_doc = _read_manifest(old_manifest, "old SAM3 manifest")
    new_doc = _read_manifest(new_manifest, "new SAM3 manifest")
    old = _all_records(old_doc, "old SAM3")
    new = _all_records(new_doc, "new SAM3")
    if set(old) != set(new):
        raise EntitySegRegionError("SAM3 manifests have different region sets")
    output_dir = output_dir.expanduser().resolve()
    assets = output_dir / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for key in sorted(new):
        old_row, new_row = old[key], new[key]
        if old_row.get("source_image_sha256") != new_row.get("source_image_sha256"):
            raise EntitySegRegionError(f"source image mismatch: {key}")
        source_path = Path(str(new_row["source_image"])).resolve(strict=True)
        source = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        if source is None:
            raise EntitySegRegionError(f"unable to read source: {source_path}")
        seed = _template(new_row, "cad_projection_seed", source.shape[:2])
        amodal = _template(new_row, "cad_amodal_template", source.shape[:2])
        old_mask = _record_mask(old_manifest.parent, old_row, source.shape[:2])
        new_mask = _record_mask(new_manifest.parent, new_row, source.shape[:2])
        old_ok = old_row.get("accepted") is True
        new_ok = new_row.get("accepted") is True
        old_dx, old_dy = _largest_old_translation(old_row)
        if old_ok and max(abs(old_dx), abs(old_dy)) > 12.0:
            decision = "large_part_shift_removed"
        elif not old_ok and new_ok:
            decision = "new_safe_candidate"
        elif old_ok and new_ok:
            decision = "retained_safe_candidate"
        elif old_ok and not new_ok:
            decision = "old_candidate_rejected_by_shared_gate"
        else:
            decision = "still_rejected"
        asset = f"{decision}__{key[0]}__{key[1]}.png"
        if not cv2.imwrite(
            str(assets / asset),
            _tile(
                source,
                seed,
                amodal,
                old_mask,
                new_mask,
                old_accepted=old_ok,
                new_accepted=new_ok,
            ),
        ):
            raise EntitySegRegionError(f"unable to write report asset: {asset}")
        shared = new_row.get("view_shared_alignment", {})
        rows.append(
            {
                "view": key[0],
                "part": key[1],
                "decision": decision,
                "asset": f"assets/{asset}",
                "old_translation": [old_dx, old_dy],
                "new_translation": shared.get("translation_xy_pixels", [0.0, 0.0]),
                "old_accepted": old_ok,
                "new_accepted": new_ok,
            }
        )
    labels = {
        "large_part_shift_removed": "已消除旧版大幅单零件平移",
        "new_safe_candidate": "新版新增安全候选",
        "retained_safe_candidate": "两版都安全通过",
        "old_candidate_rejected_by_shared_gate": "旧候选被新版安全门拒绝",
        "still_rejected": "两版均未安全通过",
    }
    sections: list[str] = []
    for decision, label in labels.items():
        cards = []
        for row in rows:
            if row["decision"] != decision:
                continue
            old_t = row["old_translation"]
            new_t = row["new_translation"]
            cards.append(
                "<article class='card'>"
                f"<h3>{html.escape(row['view'])} / {html.escape(row['part'])}</h3>"
                f"<a target='_blank' href='{html.escape(row['asset'])}'><img loading='lazy' src='{html.escape(row['asset'])}'></a>"
                f"<p>旧单件平移 ({old_t[0]:+.1f}, {old_t[1]:+.1f}) px · "
                f"新整件共享平移 ({float(new_t[0]):+.1f}, {float(new_t[1]):+.1f}) px</p>"
                "</article>"
            )
        sections.append(
            f"<section id='{decision}'><h2>{label} ({len(cards)})</h2><div class='grid'>"
            + "".join(cards)
            + "</div></section>"
        )
    counts = {name: sum(row["decision"] == name for row in rows) for name in labels}
    page = """<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>
<title>整件共享相机残差 A/B</title><style>
body{margin:0;background:#0b1118;color:#dce8f5;font:15px system-ui}main{max-width:1800px;margin:auto;padding:24px}.lead{color:#a9bdd2}.stats{display:flex;gap:12px;flex-wrap:wrap}.stat{background:#15202b;padding:10px 14px;border-radius:8px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(620px,1fr));gap:14px}.card{background:#121c26;border:1px solid #26394b;border-radius:12px;padding:12px}.card img{width:100%;height:auto}.card p{color:#abc0d5}a{color:#7dc7ff}h2{margin-top:32px}
</style></head><body><main><h1>单零件平移 → 整件共享相机残差</h1>
<p class='lead'>红色为原始可见 CAD，黄色为完整 mesh；绿色是旧版允许每个 Part-ID 独立平移后的 SAM3，紫色是新版所有零件共享同一视角残差后的 SAM3。新版不允许任何零件单独移动。</p>
<div class='stats'>"""
    page += (
        f"<div class='stat'>旧版通过 "
        f"{old_doc['summary']['accepted_region_count']}/{len(rows)}</div>"
    )
    page += (
        f"<div class='stat'>新版通过 "
        f"{new_doc['summary']['accepted_region_count']}/{len(rows)}</div>"
    )
    page += f"<div class='stat'>清除大幅单件平移 {counts['large_part_shift_removed']}</div>"
    page += "</div>" + "".join(sections) + "</main></body></html>"
    (output_dir / "index.html").write_text(page, encoding="utf-8")
    return {"region_count": len(rows), "decision_counts": counts, "index": str(output_dir / "index.html")}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-manifest", required=True, type=Path)
    parser.add_argument("--new-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    print(build_report(old_manifest=args.old_manifest, new_manifest=args.new_manifest, output_dir=args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
