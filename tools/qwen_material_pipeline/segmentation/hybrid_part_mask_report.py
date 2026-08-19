#!/usr/bin/env python3
"""Build a browser report for the CAD/SAM3/EntitySeg hybrid decision."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

from .entityseg_comparison_report import _annotate, _mask, _overlay, _read_manifest
from .entityseg_regions import EntitySegRegionError, _internal_repair_support


def _all_records(
    document: Mapping[str, Any], label: str
) -> dict[tuple[str, str], Mapping[str, Any]]:
    output: dict[tuple[str, str], Mapping[str, Any]] = {}
    for index, row in enumerate(document["records"]):
        if not isinstance(row, Mapping):
            raise EntitySegRegionError(f"{label} record {index} is malformed")
        key = (str(row.get("view_id")), str(row.get("group_id")))
        if key in output:
            raise EntitySegRegionError(f"duplicate {label} record: {key}")
        output[key] = row
    return output


def _record_mask(
    root: Path,
    row: Mapping[str, Any],
    expected_shape: tuple[int, int],
    *,
    field: str = "mask",
) -> np.ndarray:
    mask = row.get(field)
    if not isinstance(mask, Mapping) or not isinstance(mask.get("path"), str):
        return np.zeros(expected_shape, dtype=bool)
    path = Path(mask["path"])
    if not path.is_absolute():
        path = root / path
    return _mask(path, expected_shape)


def _aligned_template(
    seed: np.ndarray,
    row: Mapping[str, Any],
) -> np.ndarray:
    audit = row.get("aligned_cad_template")
    translation = (
        audit.get("translation_xy_pixels") if isinstance(audit, Mapping) else None
    )
    if not isinstance(translation, list) or len(translation) != 2:
        return seed
    matrix = np.asarray(
        [[1.0, 0.0, float(translation[0])], [0.0, 1.0, float(translation[1])]],
        dtype=np.float32,
    )
    return (
        cv2.warpAffine(
            seed.astype(np.uint8),
            matrix,
            (seed.shape[1], seed.shape[0]),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        > 0
    )


def _tile(
    *,
    source: np.ndarray,
    seed: np.ndarray,
    amodal: np.ndarray,
    sam: np.ndarray,
    entity: np.ndarray,
    hybrid: np.ndarray,
    sam_accepted: bool,
    entity_accepted: bool,
    entity_internal_repair: bool,
    selected_source: str,
) -> np.ndarray:
    union = seed | amodal | sam | entity | hybrid
    ys, xs = np.where(union)
    pad = 20
    left = max(0, int(xs.min()) - pad)
    right = min(source.shape[1], int(xs.max()) + pad + 1)
    top = max(0, int(ys.min()) - pad)
    bottom = min(source.shape[0], int(ys.max()) + pad + 1)
    panels: list[np.ndarray] = []
    definitions = (
        ("CAD visible (occluded)", seed, (0, 0, 255)),
        ("isolated mesh shape", amodal, (0, 165, 255)),
        ("SAM3" if sam_accepted else "SAM3 rejected", sam, (0, 255, 0)),
        (
            "EntitySeg"
            if entity_accepted
            else "EntitySeg repair support"
            if entity_internal_repair
            else "EntitySeg rejected",
            entity,
            (255, 255, 0),
        ),
        (
            (
                "FINAL: iterative mesh-guided"
                if selected_source == "shape_guided_iterative"
                else "FINAL: Entity+CAD"
                if selected_source == "entityseg"
                else "FINAL: SAM3"
                if selected_source == "sam3"
                else "FINAL: rejected"
            ),
            hybrid,
            (255, 0, 255),
        ),
    )
    for label, mask, color in definitions:
        crop = _overlay(source, mask, color)[top:bottom, left:right]
        # Keep the original crop aspect ratio.  The old fixed resize made a
        # correct CAD silhouette look wider or taller than its mesh.
        scale = min(300.0 / max(1, crop.shape[1]), 230.0 / max(1, crop.shape[0]))
        resized_width = max(1, int(round(crop.shape[1] * scale)))
        resized_height = max(1, int(round(crop.shape[0] * scale)))
        resized = cv2.resize(
            crop,
            (resized_width, resized_height),
            interpolation=cv2.INTER_NEAREST,
        )
        panel = np.zeros((230, 300, 3), dtype=np.uint8)
        left_pad = (300 - resized_width) // 2
        top_pad = (230 - resized_height) // 2
        panel[
            top_pad : top_pad + resized_height,
            left_pad : left_pad + resized_width,
        ] = resized
        _annotate(panel, label)
        panels.append(panel)
    return np.hstack(panels)


def build_report(
    *,
    sam_manifest_path: Path,
    entity_manifest_path: Path,
    hybrid_manifest_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    sam_manifest_path = sam_manifest_path.expanduser().resolve(strict=True)
    entity_manifest_path = entity_manifest_path.expanduser().resolve(strict=True)
    hybrid_manifest_path = hybrid_manifest_path.expanduser().resolve(strict=True)
    sam_root = sam_manifest_path.parent
    entity_root = entity_manifest_path.parent
    hybrid_root = hybrid_manifest_path.parent
    sam = _all_records(_read_manifest(sam_manifest_path, "SAM3 manifest"), "SAM3")
    entity = _all_records(
        _read_manifest(entity_manifest_path, "EntitySeg manifest"), "EntitySeg"
    )
    hybrid_document = _read_manifest(hybrid_manifest_path, "hybrid manifest")
    hybrid = _all_records(hybrid_document, "hybrid")
    if set(sam) != set(entity) or set(sam) != set(hybrid):
        raise EntitySegRegionError("comparison manifests have different region sets")

    output_dir = output_dir.expanduser().resolve()
    assets_dir = output_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for key in sorted(hybrid):
        final_row = hybrid[key]
        sam_row = sam[key]
        entity_row = entity[key]
        source_path = (
            Path(str(final_row["source_image"])).expanduser().resolve(strict=True)
        )
        source = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        if source is None:
            raise EntitySegRegionError(f"unable to decode source: {source_path}")
        seed_doc = entity_row.get("cad_projection_seed") or sam_row.get(
            "cad_projection_seed"
        )
        if not isinstance(seed_doc, Mapping):
            raise EntitySegRegionError(f"missing CAD seed: {key}")
        seed = _aligned_template(
            _mask(Path(str(seed_doc["path"])), source.shape[:2]),
            final_row,
        )
        amodal_doc = (
            final_row.get("cad_amodal_template")
            or entity_row.get("cad_amodal_template")
            or sam_row.get("cad_amodal_template")
        )
        if isinstance(amodal_doc, Mapping) and isinstance(amodal_doc.get("path"), str):
            amodal = _aligned_template(
                _mask(Path(str(amodal_doc["path"])), source.shape[:2]),
                final_row,
            )
        else:
            amodal = seed
        sam_mask = _record_mask(sam_root, sam_row, source.shape[:2])
        entity_internal_repair = (
            entity_row.get("internal_repair_mask") is not None
            and entity_row.get("accepted") is not True
        )
        entity_mask = _record_mask(
            entity_root,
            entity_row,
            source.shape[:2],
            field=("internal_repair_mask" if entity_internal_repair else "mask"),
        )
        if entity_internal_repair:
            # The persisted repair mask is the raw (possibly oversized)
            # EntitySeg proposal. Display only the bounded internal support
            # that the fusion algorithm is actually authorized to consume.
            entity_mask, _repair_metrics = _internal_repair_support(
                entity_mask, seed, amodal
            )
        final_mask = _record_mask(hybrid_root, final_row, source.shape[:2])
        decision = str(final_row["decision"])
        asset_name = f"{decision}__{key[0]}__{key[1]}.png"
        asset_path = assets_dir / asset_name
        tile = _tile(
            source=source,
            seed=seed,
            amodal=amodal,
            sam=sam_mask,
            entity=entity_mask,
            hybrid=final_mask,
            sam_accepted=sam_row.get("accepted") is True,
            entity_accepted=entity_row.get("accepted") is True,
            entity_internal_repair=entity_internal_repair,
            selected_source=str(final_row["selected_source"]),
        )
        if not cv2.imwrite(str(asset_path), tile):
            raise EntitySegRegionError(f"unable to write hybrid tile: {asset_path}")
        rows.append(
            {
                "view_id": key[0],
                "part_id": key[1],
                "decision": decision,
                "selected_source": final_row["selected_source"],
                "entityseg_rejection_reasons": final_row.get(
                    "entityseg_fusion_rejection_reasons", []
                ),
                "metrics": final_row.get("fusion_metrics"),
                "cad_support_trim": final_row.get("cad_support_trim"),
                "iterative_refinement": final_row.get("iterative_refinement"),
                "entityseg_internal_repair_candidate_available": final_row.get(
                    "entityseg_internal_repair_candidate_available"
                )
                is True,
                "aligned_cad_template": final_row.get("aligned_cad_template"),
                "accepted": final_row.get("accepted") is True,
                "asset": f"assets/{asset_name}",
            }
        )

    labels = {
        "iterative_refinement_from_sam3_entityseg": "SAM3 + EntitySeg 联合迭代优化",
        "iterative_refinement_from_sam3": "SAM3 + mesh/可见性迭代优化",
        "iterative_refinement_from_entityseg": "EntitySeg + mesh/可见性迭代优化",
        "iterative_refinement_rejected": "迭代优化未通过安全约束",
        "no_safe_candidate": "没有通过双模板约束的候选",
    }
    sections: list[str] = []
    for decision in labels:
        cards: list[str] = []
        for row in rows:
            if row["decision"] != decision:
                continue
            metrics = row.get("metrics") or {}
            details: list[str] = []
            if "entity_edge_improvement" in metrics:
                details.append(f"边缘增益 {metrics['entity_edge_improvement']:+.3f}")
            if "entity_cad_direct_iou" in metrics:
                details.append(f"CAD 直接 IoU {metrics['entity_cad_direct_iou']:.3f}")
            if "entity_to_cad_area_ratio" in metrics:
                details.append(f"面积比 {metrics['entity_to_cad_area_ratio']:.2f}")
            trim = row.get("cad_support_trim")
            if isinstance(trim, Mapping):
                details.append(
                    "CAD 支持域裁剪：保留 "
                    f"{100.0 * float(trim['retained_entity_fraction']):.1f}%"
                )
            refinement = row.get("iterative_refinement")
            if isinstance(refinement, Mapping):
                initial = refinement.get("initial_metrics")
                final = refinement.get("final_metrics")
                if isinstance(initial, Mapping) and isinstance(final, Mapping):
                    details.append(
                        "迭代 "
                        f"{int(refinement.get('selected_iteration', 0))}/"
                        f"{int(refinement.get('iteration_budget', 0))}："
                        "边缘支持 "
                        f"{float(initial['image_edge_support']):.3f}→"
                        f"{float(final['image_edge_support']):.3f}，"
                        "可见 CAD IoU "
                        f"{float(initial['visible_seed_iou']):.3f}→"
                        f"{float(final['visible_seed_iou']):.3f}"
                    )
                removed = refinement.get(
                    "known_occluded_primary_candidate_pixels_removed"
                )
                if isinstance(removed, int) and removed:
                    details.append(f"移除遮挡误归属 {removed} px")
                repair_pixels = refinement.get("entityseg_internal_repair_pixels")
                if (
                    refinement.get("entityseg_internal_repair_applied") is True
                    and isinstance(repair_pixels, int)
                    and repair_pixels
                ):
                    final_repair_pixels = int(
                        refinement.get("entityseg_internal_repair_final_pixels", 0)
                    )
                    details.append(
                        "EntitySeg 修复封闭或尺度受限的 CAD 内部缺口 "
                        f"{final_repair_pixels}/{repair_pixels} px"
                    )
                elif (
                    refinement.get("entityseg_internal_repair_authorized") is True
                    and isinstance(repair_pixels, int)
                    and repair_pixels
                ):
                    details.append(f"EntitySeg 内部修复候选 {repair_pixels} px 未提升目标，未应用")
            alignment = row.get("aligned_cad_template")
            if isinstance(alignment, Mapping):
                translation = alignment.get("translation_xy_pixels", [0.0, 0.0])
                details.append(
                    "整件相机模板残差平移 "
                    f"({float(translation[0]):+.1f}, {float(translation[1]):+.1f}) px"
                )
            reasons = row["entityseg_rejection_reasons"]
            if reasons:
                details.append("拒绝原因：" + ", ".join(reasons))
            cards.append(
                '<article class="card">'
                f"<h3>{html.escape(row['view_id'])} / {html.escape(row['part_id'])}</h3>"
                f'<a href="{html.escape(row["asset"])}" target="_blank">'
                f'<img src="{html.escape(row["asset"])}" loading="lazy"></a>'
                f"<p>{html.escape(' · '.join(details))}</p></article>"
            )
        sections.append(
            f'<section id="{decision}"><h2>{labels[decision]} ({len(cards)})</h2>'
            f'<div class="grid">{"".join(cards)}</div></section>'
        )

    summary = hybrid_document["summary"]
    page = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SAM3 + EntitySeg 融合结果</title><style>
:root{{--bg:#0b0f14;--panel:#131a22;--text:#e8eef5;--muted:#9fb0c1;--accent:#40ded7}}*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--text);font:15px/1.5 system-ui,sans-serif}}main{{max-width:1560px;margin:auto;padding:28px}}
h1{{margin:.1em 0}}h2{{margin-top:42px}}.lead,.card p{{color:var(--muted)}}.stats{{display:flex;gap:10px;flex-wrap:wrap;margin:20px 0}}
.stat,.card{{background:var(--panel);border:1px solid #253342;border-radius:11px}}.stat{{padding:12px 16px}}nav a{{color:var(--accent);margin-right:18px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(590px,1fr));gap:14px}}.card{{padding:12px;overflow:hidden}}.card h3{{margin:0 0 8px}}
.card img{{display:block;width:100%;height:auto;border-radius:7px}}.card p{{margin:8px 2px 0}}@media(max-width:650px){{main{{padding:14px}}.grid{{grid-template-columns:1fr}}}}
</style></head><body><main><h1>SAM3 + EntitySeg 辅助分割</h1>
<p class="lead">红色是整机渲染得到的当前视图可见 Part-ID（遮挡归属权威），黄色是保持同一整机相机和刚体位姿、仅隐藏其他 mesh 后投影出的完整零件形状。绿色和青色是 SAM3 与 EntitySeg 的前序估计；紫色不是二选一，而是在完整 mesh、当前可见性和照片边缘之间逐轮优化并选择最优安全迭代。每个视角只允许整件工件共享一个有界 2D 相机残差，不允许单独移动、旋转或缩放 mesh。</p>
<div class="stats"><div class="stat">最终通过 <b>{summary['accepted_region_count']}</b> / {summary['region_count']}</div><div class="stat">联合候选迭代 <b>{summary['decision_counts'].get('iterative_refinement_from_sam3_entityseg', 0)}</b></div><div class="stat">单候选 + mesh 迭代 <b>{summary['decision_counts'].get('iterative_refinement_from_sam3', 0) + summary['decision_counts'].get('iterative_refinement_from_entityseg', 0)}</b></div><div class="stat">EntitySeg 内部修复 <b>{summary.get('entityseg_internal_repair_applied_region_count', 0)}</b> / 候选 {summary.get('entityseg_internal_repair_authorized_region_count', 0)}</div><div class="stat">最终边界：迭代优化 <b>{summary['selected_source_counts'].get('shape_guided_iterative', 0)}</b></div></div>
<p class="lead">这是正式 Part-ID 材质证据使用的边界融合结果；无安全候选时会回退到已审计的 CAD 投影，不会猜测零件边界。点击图片可查看原始像素。</p>
<nav>{''.join(f'<a href="#{key}">{value}</a>' for key,value in labels.items())}</nav>{''.join(sections)}</main></body></html>"""
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
    parser.add_argument("--hybrid-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    result = build_report(
        sam_manifest_path=args.sam_manifest,
        entity_manifest_path=args.entity_manifest,
        hybrid_manifest_path=args.hybrid_manifest,
        output_dir=args.output_dir,
    )
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
