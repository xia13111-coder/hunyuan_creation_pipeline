#!/usr/bin/env python3
"""Fuse CAD-bound SAM3 instances with safely matched EntitySeg boundaries."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

from .entityseg_regions import EntitySegRegionError, _boundary_metrics


SCHEMA_VERSION = "qwen-cad-sam3-entityseg-hybrid/v1"
MAXIMUM_ENTITY_TO_CAD_AREA_RATIO = 1.85
MINIMUM_ENTITY_CAD_DIRECT_IOU = 0.50
MINIMUM_ENTITY_CAD_SHAPE_IOU = 0.55
MAXIMUM_ENTITY_CAD_CENTROID_DISTANCE = 0.15
MINIMUM_ENTITY_EDGE_SUPPORT = 0.70
MINIMUM_ENTITY_EDGE_IMPROVEMENT = 0.03
MINIMUM_SAM_ENTITY_OVERLAP_SMALLER = 0.50
MINIMUM_DIRECT_IOU_WHEN_SAM_DISAGREES = 0.60
MINIMUM_CONNECTED_COMPONENT_PIXELS = 16


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _read_manifest(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.expanduser().resolve(strict=True).read_text("utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EntitySegRegionError(f"unable to read {label}: {path}") from exc
    if not isinstance(value, dict) or not isinstance(value.get("records"), list):
        raise EntitySegRegionError(f"{label} is not a region manifest")
    return value


def _records(document: Mapping[str, Any], label: str) -> dict[tuple[str, str], Mapping[str, Any]]:
    output: dict[tuple[str, str], Mapping[str, Any]] = {}
    for index, row in enumerate(document["records"]):
        if not isinstance(row, Mapping):
            raise EntitySegRegionError(f"{label} record {index} is malformed")
        key = (str(row.get("view_id")), str(row.get("group_id")))
        if key in output:
            raise EntitySegRegionError(f"duplicate {label} region: {key}")
        output[key] = row
    return output


def _resolved_mask_path(root: Path, row: Mapping[str, Any]) -> Path:
    mask = row.get("mask")
    if not isinstance(mask, Mapping) or not isinstance(mask.get("path"), str):
        raise EntitySegRegionError("accepted region has no mask path")
    path = Path(mask["path"])
    if not path.is_absolute():
        path = root / path
    return path.expanduser().resolve(strict=True)


def _load_mask(path: Path, expected_shape: tuple[int, int]) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.shape != expected_shape:
        raise EntitySegRegionError(f"invalid mask: {path}")
    return mask >= 128


def _connected_component_count(mask: np.ndarray) -> int:
    count, _labels, statistics, _centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    return int(
        sum(
            int(statistics[index, cv2.CC_STAT_AREA])
            >= MINIMUM_CONNECTED_COMPONENT_PIXELS
            for index in range(1, count)
        )
    )


def _entity_rejection_reasons(
    metrics: Mapping[str, float | int], *, sam_accepted: bool
) -> list[str]:
    reasons: list[str] = []
    if int(metrics["connected_component_count"]) != 1:
        reasons.append("entity_mask_is_not_one_connected_component")
    if float(metrics["entity_to_cad_area_ratio"]) > MAXIMUM_ENTITY_TO_CAD_AREA_RATIO:
        reasons.append("entity_mask_area_exceeds_cad_bound")
    if float(metrics["entity_cad_direct_iou"]) < MINIMUM_ENTITY_CAD_DIRECT_IOU:
        reasons.append("entity_direct_cad_iou_below_threshold")
    if float(metrics["entity_cad_shape_iou"]) < MINIMUM_ENTITY_CAD_SHAPE_IOU:
        reasons.append("entity_cad_shape_iou_below_threshold")
    if (
        float(metrics["entity_cad_centroid_distance"])
        > MAXIMUM_ENTITY_CAD_CENTROID_DISTANCE
    ):
        reasons.append("entity_centroid_too_far_from_cad_part")
    if float(metrics["entity_edge_support"]) < MINIMUM_ENTITY_EDGE_SUPPORT:
        reasons.append("entity_boundary_has_insufficient_image_edge_support")
    if sam_accepted:
        if (
            float(metrics["entity_edge_improvement"])
            < MINIMUM_ENTITY_EDGE_IMPROVEMENT
        ):
            reasons.append("entity_boundary_does_not_improve_over_sam3")
        if (
            float(metrics["sam_entity_overlap_smaller"])
            < MINIMUM_SAM_ENTITY_OVERLAP_SMALLER
            and float(metrics["entity_cad_direct_iou"])
            < MINIMUM_DIRECT_IOU_WHEN_SAM_DISAGREES
        ):
            reasons.append("entity_disagrees_with_both_sam3_and_cad_location")
    return reasons


def _entity_metrics(
    *,
    image: np.ndarray,
    seed: np.ndarray,
    entity_mask: np.ndarray,
    entity_row: Mapping[str, Any],
    sam_mask: np.ndarray | None,
) -> dict[str, float | int]:
    selected = entity_row.get("selected_candidate")
    if not isinstance(selected, Mapping):
        raise EntitySegRegionError("accepted EntitySeg record has no selected candidate")
    seed_pixels = int(np.count_nonzero(seed))
    entity_boundary = _boundary_metrics(image, entity_mask)
    output: dict[str, float | int] = {
        "connected_component_count": _connected_component_count(entity_mask),
        "entity_mask_pixels": int(np.count_nonzero(entity_mask)),
        "cad_seed_pixels": seed_pixels,
        "entity_to_cad_area_ratio": int(np.count_nonzero(entity_mask))
        / max(seed_pixels, 1),
        "entity_cad_direct_iou": float(selected.get("cad_direct_iou", -1.0)),
        "entity_cad_shape_iou": float(selected.get("cad_shape_iou", -1.0)),
        "entity_cad_centroid_distance": float(
            selected.get("cad_centroid_distance_normalized", float("inf"))
        ),
        "entity_edge_support": float(
            entity_boundary["image_edge_support_fraction_025"]
        ),
    }
    if sam_mask is not None:
        intersection = int(np.count_nonzero(sam_mask & entity_mask))
        sam_pixels = int(np.count_nonzero(sam_mask))
        entity_pixels = int(np.count_nonzero(entity_mask))
        sam_boundary = _boundary_metrics(image, sam_mask)
        output.update(
            {
                "sam_mask_pixels": sam_pixels,
                "sam_entity_iou": intersection
                / max(int(np.count_nonzero(sam_mask | entity_mask)), 1),
                "sam_entity_overlap_smaller": intersection
                / max(min(sam_pixels, entity_pixels), 1),
                "sam_edge_support": float(
                    sam_boundary["image_edge_support_fraction_025"]
                ),
                "entity_edge_improvement": float(
                    entity_boundary["image_edge_support_fraction_025"]
                )
                - float(sam_boundary["image_edge_support_fraction_025"]),
            }
        )
    return output


def build_hybrid_masks(
    *,
    sam_manifest_path: Path,
    entity_manifest_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    sam_manifest_path = sam_manifest_path.expanduser().resolve(strict=True)
    entity_manifest_path = entity_manifest_path.expanduser().resolve(strict=True)
    sam_document = _read_manifest(sam_manifest_path, "SAM3 manifest")
    entity_document = _read_manifest(entity_manifest_path, "EntitySeg manifest")
    if sam_document.get("request") != entity_document.get("request"):
        raise EntitySegRegionError("SAM3 and EntitySeg manifests bind different requests")
    sam_records = _records(sam_document, "SAM3")
    entity_records = _records(entity_document, "EntitySeg")
    if set(sam_records) != set(entity_records):
        raise EntitySegRegionError("SAM3 and EntitySeg region sets differ")
    sam_root = sam_manifest_path.parent
    entity_root = entity_manifest_path.parent
    output_dir = output_dir.expanduser().resolve()
    masks_dir = output_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    decision_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    for key in sorted(sam_records):
        sam_row = sam_records[key]
        entity_row = entity_records[key]
        authority = entity_row if entity_row.get("source_image") else sam_row
        source_path = Path(str(authority["source_image"])).expanduser().resolve(strict=True)
        image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        if image is None:
            raise EntitySegRegionError(f"unable to decode source image: {source_path}")
        seed_doc = entity_row.get("cad_projection_seed") or sam_row.get(
            "cad_projection_seed"
        )
        if not isinstance(seed_doc, Mapping):
            raise EntitySegRegionError(f"missing CAD seed: {key}")
        seed = _load_mask(
            Path(str(seed_doc["path"])).expanduser().resolve(strict=True),
            image.shape[:2],
        )
        sam_accepted = sam_row.get("accepted") is True
        entity_accepted = entity_row.get("accepted") is True
        sam_mask = (
            _load_mask(_resolved_mask_path(sam_root, sam_row), image.shape[:2])
            if sam_accepted
            else None
        )
        entity_mask = (
            _load_mask(
                _resolved_mask_path(entity_root, entity_row), image.shape[:2]
            )
            if entity_accepted
            else None
        )
        metrics: dict[str, float | int] | None = None
        entity_reasons: list[str] = []
        if entity_mask is not None:
            metrics = _entity_metrics(
                image=image,
                seed=seed,
                entity_mask=entity_mask,
                entity_row=entity_row,
                sam_mask=sam_mask,
            )
            entity_reasons = _entity_rejection_reasons(
                metrics, sam_accepted=sam_accepted
            )

        final_mask: np.ndarray | None
        if entity_mask is not None and not entity_reasons:
            final_mask = entity_mask
            selected_source = "entityseg"
            decision = (
                "entityseg_replaces_sam3_boundary"
                if sam_accepted
                else "entityseg_fills_sam3_gap"
            )
        elif sam_mask is not None:
            final_mask = sam_mask
            selected_source = "sam3"
            decision = (
                "sam3_retained_after_entityseg_rejection"
                if entity_mask is not None
                else "sam3_only_candidate"
            )
        else:
            final_mask = None
            selected_source = "none"
            decision = "no_safe_candidate"

        mask_document: dict[str, Any] | None = None
        if final_mask is not None:
            mask_path = masks_dir / f"{key[0]}__{key[1]}.png"
            if not cv2.imwrite(str(mask_path), final_mask.astype(np.uint8) * 255):
                raise EntitySegRegionError(f"unable to write hybrid mask: {mask_path}")
            mask_document = {
                "path": str(mask_path.relative_to(output_dir)),
                "sha256": _sha256_file(mask_path),
                "mask_pixels": int(np.count_nonzero(final_mask)),
            }
        decision_counts[decision] += 1
        source_counts[selected_source] += 1
        records.append(
            {
                "view_id": key[0],
                "group_id": key[1],
                "source_image": str(source_path),
                "accepted": final_mask is not None,
                "selected_source": selected_source,
                "decision": decision,
                "entityseg_candidate_accepted": entity_accepted,
                "entityseg_fusion_rejection_reasons": entity_reasons,
                "fusion_metrics": metrics,
                "mask": mask_document,
            }
        )

    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "inputs": {
            "sam3_manifest": {
                "path": str(sam_manifest_path),
                "sha256": _sha256_file(sam_manifest_path),
                "document_sha256": _canonical_sha256(sam_document),
            },
            "entityseg_manifest": {
                "path": str(entity_manifest_path),
                "sha256": _sha256_file(entity_manifest_path),
                "document_sha256": _canonical_sha256(entity_document),
            },
        },
        "policy": {
            "identity_authority": "registered_cad_part_id_plus_sam3_instance",
            "entityseg_role": "boundary_candidate_only",
            "maximum_entity_to_cad_area_ratio": MAXIMUM_ENTITY_TO_CAD_AREA_RATIO,
            "minimum_entity_cad_direct_iou": MINIMUM_ENTITY_CAD_DIRECT_IOU,
            "minimum_entity_cad_shape_iou": MINIMUM_ENTITY_CAD_SHAPE_IOU,
            "maximum_entity_cad_centroid_distance": MAXIMUM_ENTITY_CAD_CENTROID_DISTANCE,
            "minimum_entity_edge_support": MINIMUM_ENTITY_EDGE_SUPPORT,
            "minimum_entity_edge_improvement": MINIMUM_ENTITY_EDGE_IMPROVEMENT,
            "minimum_sam_entity_overlap_smaller": MINIMUM_SAM_ENTITY_OVERLAP_SMALLER,
            "minimum_direct_iou_when_sam_disagrees": MINIMUM_DIRECT_IOU_WHEN_SAM_DISAGREES,
        },
        "records": records,
        "summary": {
            "region_count": len(records),
            "accepted_region_count": sum(row["accepted"] for row in records),
            "selected_source_counts": dict(sorted(source_counts.items())),
            "decision_counts": dict(sorted(decision_counts.items())),
            "selected_unique_part_count": len(
                {row["group_id"] for row in records if row["accepted"]}
            ),
        },
    }
    result["integrity"] = {"result_sha256": _canonical_sha256(result)}
    (output_dir / "manifest.json").write_text(
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
    result = build_hybrid_masks(
        sam_manifest_path=args.sam_manifest,
        entity_manifest_path=args.entity_manifest,
        output_dir=args.output_dir,
    )
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
