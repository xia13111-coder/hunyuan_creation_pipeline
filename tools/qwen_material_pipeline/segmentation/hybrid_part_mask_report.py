#!/usr/bin/env python3
"""Build a browser report for the CAD/SAM3/EntitySeg hybrid decision."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

from .entityseg_comparison_report import _annotate, _mask, _overlay, _read_manifest
from .entityseg_regions import EntitySegRegionError


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


def _amodal_records(
    document: Mapping[str, Any],
) -> dict[tuple[str, str], Mapping[str, Any]]:
    output: dict[tuple[str, str], Mapping[str, Any]] = {}
    for index, row in enumerate(document["records"]):
        if not isinstance(row, Mapping):
            raise EntitySegRegionError(f"amodal record {index} is malformed")
        key = (str(row.get("view_id")), str(row.get("part_id")))
        if not all(key) or key in output:
            raise EntitySegRegionError(f"duplicate or invalid amodal record: {key}")
        output[key] = row
    return output


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _document_mask(
    document: Mapping[str, Any],
    *,
    label: str,
    expected_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    raw_path = document.get("path")
    expected_sha = document.get("sha256")
    if not isinstance(raw_path, str) or not isinstance(expected_sha, str):
        raise EntitySegRegionError(f"{label} mask binding is malformed")
    path = Path(raw_path).expanduser().resolve(strict=True)
    if _sha256_file(path) != expected_sha:
        raise EntitySegRegionError(f"{label} mask hash changed: {path}")
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None or (expected_shape is not None and image.shape != expected_shape):
        raise EntitySegRegionError(f"{label} mask is invalid: {path}")
    return image >= 128


def _record_mask(
    root: Path, row: Mapping[str, Any], expected_shape: tuple[int, int]
) -> np.ndarray:
    mask = row.get("mask")
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


def _locally_registered_template(
    seed: np.ndarray,
    row: Mapping[str, Any],
) -> np.ndarray:
    iterative = row.get("iterative_refinement")
    audit = (
        iterative.get("reference_space_local_registration")
        if isinstance(iterative, Mapping)
        else None
    )
    translation = (
        audit.get("translation_xy_pixels") if isinstance(audit, Mapping) else None
    )
    rotation = audit.get("rotation_degrees") if isinstance(audit, Mapping) else None
    if (
        not isinstance(translation, list)
        or len(translation) != 2
        or not isinstance(rotation, (int, float))
    ):
        return seed
    ys, xs = np.where(seed)
    if not len(xs):
        return seed
    matrix = cv2.getRotationMatrix2D(
        (float(xs.mean()), float(ys.mean())), float(rotation), 1.0
    )
    matrix[0, 2] += float(translation[0])
    matrix[1, 2] += float(translation[1])
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


def _model_shape_photo_proposal(
    model_shape: np.ndarray,
    row: Mapping[str, Any],
    expected_shape: tuple[int, int],
) -> tuple[np.ndarray, bool]:
    iterative = row.get("iterative_refinement")
    audit = (
        iterative.get("model_domain_photo_registration")
        if isinstance(iterative, Mapping)
        else None
    )
    affine = (
        audit.get("model_to_photo_affine_2x3") if isinstance(audit, Mapping) else None
    )
    if (
        not isinstance(affine, list)
        or len(affine) != 2
        or any(not isinstance(value, list) or len(value) != 3 for value in affine)
    ):
        return np.zeros(expected_shape, dtype=bool), False
    matrix = np.asarray(affine, dtype=np.float32)
    proposal = (
        cv2.warpAffine(
            model_shape.astype(np.uint8),
            matrix,
            (expected_shape[1], expected_shape[0]),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        > 0
    )
    return proposal, bool(audit.get("accepted") is True)


def _tile(
    *,
    source: np.ndarray,
    model_source: np.ndarray,
    model_amodal: np.ndarray,
    model_photo_proposal: np.ndarray,
    model_photo_proposal_accepted: bool,
    seed: np.ndarray,
    amodal: np.ndarray,
    sam: np.ndarray,
    entity: np.ndarray,
    hybrid: np.ndarray,
    sam_accepted: bool,
    entity_accepted: bool,
    selected_source: str,
) -> np.ndarray:
    union = seed | model_photo_proposal | sam | entity | hybrid
    ys, xs = np.where(union)
    pad = 20
    left = max(0, int(xs.min()) - pad)
    right = min(source.shape[1], int(xs.max()) + pad + 1)
    top = max(0, int(ys.min()) - pad)
    bottom = min(source.shape[0], int(ys.max()) + pad + 1)
    panels: list[np.ndarray] = []

    def panel_for(
        image: np.ndarray,
        mask: np.ndarray,
        color: tuple[int, int, int],
        label: str,
        bounds: tuple[int, int, int, int],
    ) -> np.ndarray:
        bound_left, bound_top, bound_right, bound_bottom = bounds
        crop = _overlay(image, mask, color)[
            bound_top:bound_bottom, bound_left:bound_right
        ]
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
        return panel

    model_y, model_x = np.where(model_amodal)
    if not len(model_x):
        raise EntitySegRegionError("model-space isolated mesh projection is empty")
    model_pad = 20
    model_bounds = (
        max(0, int(model_x.min()) - model_pad),
        max(0, int(model_y.min()) - model_pad),
        min(model_amodal.shape[1], int(model_x.max()) + model_pad + 1),
        min(model_amodal.shape[0], int(model_y.max()) + model_pad + 1),
    )
    panels.append(
        panel_for(
            model_source,
            model_amodal,
            (0, 165, 255),
            "MODEL IMAGE: target Part-ID",
            model_bounds,
        )
    )
    definitions = (
        (
            (
                "MODEL SHAPE: registered"
                if model_photo_proposal_accepted
                else "MODEL SHAPE: proposal rejected"
            ),
            model_photo_proposal,
            (0, 255, 255),
        ),
        ("REFERENCE: CAD mask registered", seed, (0, 0, 255)),
        ("SAM3" if sam_accepted else "SAM3 rejected", sam, (0, 255, 0)),
        (
            "EntitySeg" if entity_accepted else "EntitySeg rejected",
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
        panels.append(
            panel_for(
                source,
                mask,
                color,
                label,
                (left, top, right, bottom),
            )
        )
    return np.hstack(panels)


def build_report(
    *,
    sam_manifest_path: Path,
    entity_manifest_path: Path,
    hybrid_manifest_path: Path,
    amodal_manifest_path: Path | None = None,
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
    amodal_records: dict[tuple[str, str], Mapping[str, Any]] = {}
    model_views: dict[str, Mapping[str, Any]] = {}
    if amodal_manifest_path is not None:
        amodal_manifest_path = amodal_manifest_path.expanduser().resolve(strict=True)
        amodal_document = _read_manifest(
            amodal_manifest_path, "CAD amodal template manifest"
        )
        amodal_records = _amodal_records(amodal_document)
        if set(amodal_records) != set(hybrid):
            raise EntitySegRegionError(
                "CAD amodal templates do not exactly cover comparison regions"
            )
        inputs = amodal_document.get("inputs")
        registry_binding = (
            inputs.get("registry") if isinstance(inputs, Mapping) else None
        )
        if not isinstance(registry_binding, Mapping) or not isinstance(
            registry_binding.get("path"), str
        ):
            raise EntitySegRegionError("CAD amodal manifest has no model registry")
        registry_path = (
            Path(str(registry_binding["path"])).expanduser().resolve(strict=True)
        )
        if registry_binding.get("sha256") != _sha256_file(registry_path):
            raise EntitySegRegionError("CAD model registry hash changed")
        try:
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise EntitySegRegionError("unable to read CAD model registry") from exc
        render_set = (
            registry.get("render_set") if isinstance(registry, Mapping) else None
        )
        raw_views = render_set.get("views") if isinstance(render_set, Mapping) else None
        if not isinstance(raw_views, list):
            raise EntitySegRegionError("CAD model registry has no rendered views")
        model_views = {
            str(row.get("view_id")): row
            for row in raw_views
            if isinstance(row, Mapping) and isinstance(row.get("view_id"), str)
        }

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
        seed = _locally_registered_template(
            _aligned_template(
                _mask(Path(str(seed_doc["path"])), source.shape[:2]),
                final_row,
            ),
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
        amodal_record = amodal_records.get(key)
        if amodal_record is not None:
            raw_amodal_doc = amodal_record.get("raw_amodal_mask")
            aligned_amodal_doc = amodal_record.get("aligned_amodal_mask")
            if not isinstance(raw_amodal_doc, Mapping) or not isinstance(
                aligned_amodal_doc, Mapping
            ):
                raise EntitySegRegionError(f"CAD amodal record is incomplete: {key}")
            model_reference = final_row.get("model_domain_shape_reference")
            local_model_shape = (
                model_reference.get("model_local_shape_mask")
                if isinstance(model_reference, Mapping)
                else None
            )
            model_shape_document = (
                local_model_shape
                if isinstance(local_model_shape, Mapping)
                else raw_amodal_doc
            )
            model_amodal = _document_mask(
                model_shape_document,
                label=f"model-space local target shape {key}",
            )
            render_view_id = amodal_record.get("render_view_id")
            model_view = model_views.get(str(render_view_id))
            model_rgb_path = (
                Path(str(model_view.get("rgb", ""))).expanduser().resolve(strict=True)
                if isinstance(model_view, Mapping)
                else None
            )
            model_source = (
                cv2.imread(str(model_rgb_path), cv2.IMREAD_COLOR)
                if model_rgb_path is not None
                else None
            )
            if model_source is None or model_source.shape[:2] != model_amodal.shape:
                raise EntitySegRegionError(
                    f"model image does not match isolated Part-ID projection: {key}"
                )
            if not isinstance(amodal_doc, Mapping) or amodal_doc.get(
                "sha256"
            ) != aligned_amodal_doc.get("sha256"):
                raise EntitySegRegionError(
                    f"reference-space amodal template differs from model projection: {key}"
                )
        else:
            model_amodal = amodal
            model_source = np.zeros((*model_amodal.shape, 3), dtype=np.uint8)
            model_rgb_path = None
        sam_mask = _record_mask(sam_root, sam_row, source.shape[:2])
        entity_mask = _record_mask(entity_root, entity_row, source.shape[:2])
        final_mask = _record_mask(hybrid_root, final_row, source.shape[:2])
        model_proposal_shape = model_amodal
        iterative = final_row.get("iterative_refinement")
        model_registration = (
            iterative.get("model_domain_photo_registration")
            if isinstance(iterative, Mapping)
            else None
        )
        variant_index = (
            model_registration.get("variant_index")
            if isinstance(model_registration, Mapping)
            else None
        )
        variant_documents = (
            model_reference.get("model_shape_variant_masks")
            if isinstance(model_reference, Mapping)
            else None
        )
        if isinstance(variant_index, int) and isinstance(variant_documents, list):
            variant_document = next(
                (
                    document
                    for document in variant_documents
                    if isinstance(document, Mapping)
                    and document.get("variant_index") == variant_index
                ),
                None,
            )
            if isinstance(variant_document, Mapping):
                model_proposal_shape = _document_mask(
                    variant_document,
                    label=f"selected model-space shape variant {key}",
                )
        (
            model_photo_proposal,
            model_photo_proposal_accepted,
        ) = _model_shape_photo_proposal(
            model_proposal_shape,
            final_row,
            source.shape[:2],
        )
        decision = str(final_row["decision"])
        asset_name = f"{decision}__{key[0]}__{key[1]}.png"
        asset_path = assets_dir / asset_name
        tile = _tile(
            source=source,
            model_source=model_source,
            model_amodal=model_amodal,
            model_photo_proposal=model_photo_proposal,
            model_photo_proposal_accepted=model_photo_proposal_accepted,
            seed=seed,
            amodal=amodal,
            sam=sam_mask,
            entity=entity_mask,
            hybrid=final_mask,
            sam_accepted=sam_row.get("accepted") is True,
            entity_accepted=entity_row.get("accepted") is True,
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
                "aligned_cad_template": final_row.get("aligned_cad_template"),
                "accepted": final_row.get("accepted") is True,
                "model_space_projection": (
                    final_row.get("model_domain_shape_reference", {}).get(
                        "model_local_shape_mask"
                    )
                    if isinstance(
                        final_row.get("model_domain_shape_reference"), Mapping
                    )
                    else amodal_record.get("raw_amodal_mask")
                    if amodal_record is not None
                    else None
                ),
                "model_image": (
                    str(model_rgb_path) if model_rgb_path is not None else None
                ),
                "reference_space_shape_prior": (
                    amodal_record.get("aligned_amodal_mask")
                    if amodal_record is not None
                    else amodal_doc
                ),
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
                model_registration = refinement.get("model_domain_photo_registration")
                if isinstance(model_registration, Mapping):
                    model_translation = model_registration.get(
                        "local_translation_xy_pixels", [0, 0]
                    )
                    model_status = (
                        "采用" if model_registration.get("accepted") is True else "未采用"
                    )
                    details.append(
                        "模型图零件形状→参考图自动配准 "
                        f"{float(model_registration.get('local_uniform_scale', 1.0)):.3f}× / "
                        f"{float(model_registration.get('local_rotation_degrees', 0.0)):+.2f}° / "
                        f"({int(model_translation[0]):+d}, {int(model_translation[1]):+d}) px；"
                        f"整机邻件位置约束{model_status}"
                    )
                local_registration = refinement.get(
                    "reference_space_local_registration"
                )
                if isinstance(local_registration, Mapping):
                    translation = local_registration.get(
                        "translation_xy_pixels", [0, 0]
                    )
                    details.append(
                        "照片平面 CAD 分割模板校正 "
                        f"({int(translation[0]):+d}, {int(translation[1]):+d}) px / "
                        f"{float(local_registration.get('rotation_degrees', 0.0)):+.2f}°；"
                        "mesh/相机未改"
                    )
                initial = refinement.get("initial_metrics")
                final = refinement.get("final_metrics")
                if isinstance(initial, Mapping) and isinstance(final, Mapping):
                    details.append(
                        "优化 "
                        f"{html.escape(str(refinement.get('selected_optimization_lane', 'legacy')))} "
                        f"{int(refinement.get('selected_lane_iteration', refinement.get('selected_iteration', 0)))}/"
                        f"{int(refinement.get('iteration_budget', 0))}："
                        "边缘支持 "
                        f"{float(initial['image_edge_support']):.3f}→"
                        f"{float(final['image_edge_support']):.3f}"
                    )
                    if "model_domain_shape_score" in final:
                        details.append(
                            "模型图归一化形状 "
                            f"{float(initial['model_domain_shape_score']):.3f}→"
                            f"{float(final['model_domain_shape_score']):.3f}，"
                            "轮廓支持 "
                            f"{float(initial['model_domain_shape_boundary_score']):.3f}→"
                            f"{float(final['model_domain_shape_boundary_score']):.3f}"
                        )
                removed = refinement.get(
                    "known_occluded_primary_candidate_pixels_removed"
                )
                if isinstance(removed, int) and removed:
                    details.append(f"移除遮挡误归属 {removed} px")
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
<p class="lead">第一栏是在 CAD 整机渲染图上高亮目标 Part-ID，显示它相对其他零件的真实位置。第二栏把该模型形状通过整机位置锚点自动注册到参考图；第三栏是整机 CAD 的可见投影。模型形状只允许在由当前零件尺度自动确定的小范围内校正，并以周围 CAD 零件作为禁入约束，不修改 USD、相机或任何 mesh 变换。绿色和青色是 SAM3 与 EntitySeg 候选；紫色是综合模型形状、整机相对位置、候选与照片边缘优化出的结果。</p>
<div class="stats"><div class="stat">最终通过 <b>{summary['accepted_region_count']}</b> / {summary['region_count']}</div><div class="stat">联合候选迭代 <b>{summary['decision_counts'].get('iterative_refinement_from_sam3_entityseg', 0)}</b></div><div class="stat">单候选 + mesh 迭代 <b>{summary['decision_counts'].get('iterative_refinement_from_sam3', 0) + summary['decision_counts'].get('iterative_refinement_from_entityseg', 0)}</b></div><div class="stat">最终边界：迭代优化 <b>{summary['selected_source_counts'].get('shape_guided_iterative', 0)}</b></div></div>
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
    parser.add_argument("--amodal-manifest", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    result = build_report(
        sam_manifest_path=args.sam_manifest,
        entity_manifest_path=args.entity_manifest,
        hybrid_manifest_path=args.hybrid_manifest,
        amodal_manifest_path=args.amodal_manifest,
        output_dir=args.output_dir,
    )
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
