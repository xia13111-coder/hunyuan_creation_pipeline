#!/usr/bin/env python3
"""Build a browser report comparing CAD-guided SAM3 and EntitySeg masks."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

from .entityseg_regions import EntitySegRegionError, _boundary_metrics


def _read_manifest(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.expanduser().resolve(strict=True).read_text("utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EntitySegRegionError(f"unable to read {label}: {path}") from exc
    if not isinstance(value, dict) or not isinstance(value.get("records"), list):
        raise EntitySegRegionError(f"{label} is not a region manifest")
    return value


def _accepted_records(
    document: Mapping[str, Any],
) -> dict[tuple[str, str], Mapping[str, Any]]:
    output: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in document["records"]:
        if isinstance(row, Mapping) and row.get("accepted") is True:
            key = (str(row.get("view_id")), str(row.get("group_id")))
            if key in output:
                raise EntitySegRegionError(f"duplicate accepted region: {key}")
            output[key] = row
    return output


def _mask(path: Path, expected_shape: tuple[int, int]) -> np.ndarray:
    image = cv2.imread(str(path.expanduser().resolve(strict=True)), cv2.IMREAD_GRAYSCALE)
    if image is None or image.shape != expected_shape:
        raise EntitySegRegionError(f"invalid comparison mask: {path}")
    return image >= 128


def _record_mask(
    root: Path,
    record: Mapping[str, Any] | None,
    expected_shape: tuple[int, int],
) -> np.ndarray:
    if record is None:
        return np.zeros(expected_shape, dtype=bool)
    mask = record.get("mask")
    if not isinstance(mask, Mapping) or not isinstance(mask.get("path"), str):
        raise EntitySegRegionError("accepted record has no mask path")
    path = Path(mask["path"])
    if not path.is_absolute():
        path = root / path
    return _mask(path, expected_shape)


def _overlay(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    output = image.copy()
    output[mask] = (0.55 * output[mask] + 0.45 * np.asarray(color)).astype(np.uint8)
    boundary = cv2.morphologyEx(
        mask.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)
    ) > 0
    output[boundary] = color
    return output


def _annotate(image: np.ndarray, label: str) -> None:
    cv2.putText(
        image, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 3, cv2.LINE_AA
    )
    cv2.putText(
        image, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (8, 8, 8), 1, cv2.LINE_AA
    )


def _comparison_tile(
    *,
    source: np.ndarray,
    seed: np.ndarray,
    sam_mask: np.ndarray,
    entity_mask: np.ndarray,
    sam_accepted: bool,
    entity_accepted: bool,
) -> np.ndarray:
    union = seed | sam_mask | entity_mask
    if not np.any(union):
        raise EntitySegRegionError("comparison tile has no mask pixels")
    ys, xs = np.where(union)
    pad = 20
    left = max(0, int(xs.min()) - pad)
    right = min(source.shape[1], int(xs.max()) + pad + 1)
    top = max(0, int(ys.min()) - pad)
    bottom = min(source.shape[0], int(ys.max()) + pad + 1)
    panels: list[np.ndarray] = []
    for label, mask, color in (
        ("CAD projection", seed, (0, 0, 255)),
        ("SAM3" if sam_accepted else "SAM3 rejected", sam_mask, (0, 255, 0)),
        (
            "EntitySeg" if entity_accepted else "EntitySeg rejected",
            entity_mask,
            (255, 255, 0),
        ),
    ):
        crop = _overlay(source, mask, color)[top:bottom, left:right]
        panel = cv2.resize(crop, (340, 250), interpolation=cv2.INTER_NEAREST)
        _annotate(panel, label)
        panels.append(panel)
    return np.hstack(panels)


def build_report(
    *,
    sam_manifest_path: Path,
    entity_manifest_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    sam_manifest_path = sam_manifest_path.expanduser().resolve(strict=True)
    entity_manifest_path = entity_manifest_path.expanduser().resolve(strict=True)
    sam_root = sam_manifest_path.parent
    entity_root = entity_manifest_path.parent
    sam = _accepted_records(_read_manifest(sam_manifest_path, "SAM3 manifest"))
    entity = _accepted_records(_read_manifest(entity_manifest_path, "EntitySeg manifest"))
    output_dir = output_dir.expanduser().resolve()
    assets_dir = output_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    common = set(sam) & set(entity)
    entity_only = set(entity) - set(sam)
    sam_only = set(sam) - set(entity)
    ordered = [
        *(sorted(common)),
        *(sorted(entity_only)),
        *(sorted(sam_only)),
    ]
    rows: list[dict[str, Any]] = []
    for view_id, part_id in ordered:
        sam_row = sam.get((view_id, part_id))
        entity_row = entity.get((view_id, part_id))
        authority = entity_row or sam_row
        assert authority is not None
        source_path = Path(str(authority["source_image"])).expanduser().resolve(strict=True)
        source = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        if source is None:
            raise EntitySegRegionError(f"unable to decode source image: {source_path}")
        seed_doc = authority.get("cad_projection_seed")
        if not isinstance(seed_doc, Mapping):
            raise EntitySegRegionError(f"missing CAD seed for {view_id}/{part_id}")
        seed = _mask(Path(str(seed_doc["path"])), source.shape[:2])
        sam_mask = _record_mask(sam_root, sam_row, source.shape[:2])
        entity_mask = _record_mask(entity_root, entity_row, source.shape[:2])
        category = (
            "common"
            if sam_row is not None and entity_row is not None
            else "entity_only"
            if entity_row is not None
            else "sam_only"
        )
        tile = _comparison_tile(
            source=source,
            seed=seed,
            sam_mask=sam_mask,
            entity_mask=entity_mask,
            sam_accepted=sam_row is not None,
            entity_accepted=entity_row is not None,
        )
        asset_name = f"{category}__{view_id}__{part_id}.png"
        asset_path = assets_dir / asset_name
        if not cv2.imwrite(str(asset_path), tile):
            raise EntitySegRegionError(f"unable to write comparison tile: {asset_path}")
        metrics: dict[str, Any] = {}
        if sam_row is not None:
            metrics["sam_boundary"] = _boundary_metrics(source, sam_mask)
        if entity_row is not None:
            metrics["entity_boundary"] = _boundary_metrics(source, entity_mask)
            selected = entity_row.get("selected_candidate")
            if isinstance(selected, Mapping):
                metrics["entity_cad_shape_iou"] = selected.get("cad_shape_iou")
                metrics["entity_cad_direct_iou"] = selected.get("cad_direct_iou")
                metrics["entity_centroid_distance"] = selected.get(
                    "cad_centroid_distance_normalized"
                )
        if sam_row is not None and entity_row is not None:
            intersection = int(np.count_nonzero(sam_mask & entity_mask))
            union = int(np.count_nonzero(sam_mask | entity_mask))
            metrics["sam_entity_mask_iou"] = intersection / max(union, 1)
        rows.append(
            {
                "view_id": view_id,
                "part_id": part_id,
                "category": category,
                "asset": f"assets/{asset_name}",
                "metrics": metrics,
            }
        )

    common_rows = [row for row in rows if row["category"] == "common"]
    sam_edge = [
        float(row["metrics"]["sam_boundary"]["image_edge_support_fraction_025"])
        for row in common_rows
    ]
    entity_edge = [
        float(row["metrics"]["entity_boundary"]["image_edge_support_fraction_025"])
        for row in common_rows
    ]
    summary = {
        "sam_accepted": len(sam),
        "entity_accepted": len(entity),
        "common": len(common),
        "entity_only": len(entity_only),
        "sam_only": len(sam_only),
        "common_sam_edge_support_mean": float(np.mean(sam_edge)) if sam_edge else None,
        "common_entity_edge_support_mean": (
            float(np.mean(entity_edge)) if entity_edge else None
        ),
    }
    sections: list[str] = []
    labels = {
        "common": "两者都通过",
        "entity_only": "仅 EntitySeg 通过",
        "sam_only": "仅 SAM3 通过",
    }
    for category in ("common", "entity_only", "sam_only"):
        cards: list[str] = []
        for row in rows:
            if row["category"] != category:
                continue
            metrics = row["metrics"]
            metric_lines: list[str] = []
            if "sam_entity_mask_iou" in metrics:
                metric_lines.append(f"两种 mask IoU {metrics['sam_entity_mask_iou']:.3f}")
            if "sam_boundary" in metrics:
                metric_lines.append(
                    "SAM 边缘支持 "
                    f"{metrics['sam_boundary']['image_edge_support_fraction_025']:.3f}"
                )
            if "entity_boundary" in metrics:
                metric_lines.append(
                    "Entity 边缘支持 "
                    f"{metrics['entity_boundary']['image_edge_support_fraction_025']:.3f}"
                )
            if metrics.get("entity_cad_shape_iou") is not None:
                metric_lines.append(f"CAD 形状 IoU {metrics['entity_cad_shape_iou']:.3f}")
            cards.append(
                '<article class="card">'
                f"<h3>{html.escape(row['view_id'])} / {html.escape(row['part_id'])}</h3>"
                f'<a href="{html.escape(row["asset"])}" target="_blank">'
                f'<img src="{html.escape(row["asset"])}" loading="lazy"></a>'
                f"<p>{' · '.join(metric_lines)}</p></article>"
            )
        sections.append(
            f'<section id="{category}"><h2>{labels[category]} ({len(cards)})</h2>'
            f'<div class="grid">{"".join(cards)}</div></section>'
        )
    page = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CAD Part-ID · SAM3 / EntitySeg A/B</title>
<style>
:root{{--bg:#0b0f14;--panel:#131a22;--text:#e8eef5;--muted:#9fb0c1;--accent:#38d4d4}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:15px/1.5 system-ui,sans-serif}}
main{{max-width:1500px;margin:auto;padding:28px}}h1{{margin:.1em 0}}h2{{margin-top:42px}}.lead{{color:var(--muted);max-width:1000px}}
.stats{{display:flex;gap:10px;flex-wrap:wrap;margin:20px 0}}.stat{{background:var(--panel);border:1px solid #253342;border-radius:10px;padding:12px 16px}}
nav a{{color:var(--accent);margin-right:18px}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(470px,1fr));gap:14px}}
.card{{background:var(--panel);border:1px solid #253342;border-radius:12px;padding:12px;overflow:hidden}}.card h3{{margin:0 0 8px}}
.card img{{display:block;width:100%;height:auto;border-radius:7px;image-rendering:auto}}.card p{{color:var(--muted);margin:8px 2px 0}}
code{{color:#c9f6f6}}@media(max-width:600px){{main{{padding:14px}}.grid{{grid-template-columns:1fr}}}}
</style></head><body><main>
<h1>CAD Part‑ID · SAM3 / EntitySeg 边界 A/B</h1>
<p class="lead">红色是注册后的 CAD Part‑ID 投影，绿色是 SAM3，青色是 CropFormer/EntitySeg。EntitySeg 只作为边界候选；CAD 仍是零件身份权威。点击任意对照图查看原始像素。</p>
<div class="stats"><div class="stat">SAM3 通过 <b>{summary['sam_accepted']}</b></div><div class="stat">EntitySeg 通过 <b>{summary['entity_accepted']}</b></div><div class="stat">共同 <b>{summary['common']}</b></div><div class="stat">Entity 补充 <b>{summary['entity_only']}</b></div><div class="stat">共同样本边缘支持：SAM <b>{summary['common_sam_edge_support_mean']:.3f}</b> → Entity <b>{summary['common_entity_edge_support_mean']:.3f}</b></div></div>
<p class="lead">判定合同：模型置信度 ≥0.30、位置无关 CAD 形状 IoU ≥0.50、面积一致性 ≥0.50、候选中心相对 CAD bbox 对角线偏移 ≤0.15。任何一项不满足即拒绝。</p>
<nav><a href="#common">两者都通过</a><a href="#entity_only">仅 EntitySeg</a><a href="#sam_only">仅 SAM3</a></nav>
{''.join(sections)}
</main></body></html>"""
    (output_dir / "index.html").write_text(page, encoding="utf-8")
    result = {"summary": summary, "records": rows}
    (output_dir / "comparison.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sam-manifest", required=True, type=Path)
    parser.add_argument("--entity-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    result = build_report(
        sam_manifest_path=args.sam_manifest,
        entity_manifest_path=args.entity_manifest,
        output_dir=args.output_dir,
    )
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
